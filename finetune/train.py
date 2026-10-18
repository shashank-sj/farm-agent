"""
Supervised fine-tuning of a small instruct model on agricultural Q&A.

One script, three methods, so they can be compared like-for-like on the same
data, split, prompt and loss mask:

  qlora  base frozen in 4-bit NF4 (+ double quantisation), LoRA adapters trained
         in 16/32-bit. Lowest memory.
  lora   base frozen in fp16/bf16, LoRA adapters trained. Same trainable params
         as QLoRA without the quantisation error.
  full   every weight trained (fp32 master weights, mixed-precision compute).
         Upper bound on quality and on memory.

Default base model: Qwen/Qwen2.5-0.5B-Instruct — Apache-2.0 and ungated, and
small enough to serve on the CPU-only HF Space as the agent's fallback model
(see src/llm/local_llm.py). Swap in a 1.5B/3B model with --base-model.

Run on a single GPU (Kaggle/Colab T4):
    python -m finetune.data --out data/finetune
    python -m finetune.train --method qlora --out outputs/finetune/qlora

Outputs in --out:
    final/              adapter (qlora/lora) or full model, + tokenizer
    train_summary.json  params, memory, time, best eval loss / perplexity
    log_history.json    loss curves for plotting
"""

import argparse
import json
import math
import time
from pathlib import Path

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_LR = {"qlora": 2e-4, "lora": 2e-4, "full": 2e-5}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["qlora", "lora", "full"], default="qlora")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data-dir", default="data/finetune")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--max-train-samples", type=int, default=None, help="subsample train (for quick method comparisons)")
    ap.add_argument("--max-eval-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    args.out = args.out or f"outputs/finetune/{args.method}"
    args.lr = args.lr or DEFAULT_LR[args.method]
    return args


def load_model(args, compute_dtype):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if args.method == "qlora":
        from peft import prepare_model_for_kbit_training

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",            # quantiles of a normal distribution, matching weight stats
            bnb_4bit_use_double_quant=True,       # also quantise the per-block scales (~0.4 bit/param saved)
            bnb_4bit_compute_dtype=compute_dtype, # weights are de-quantised to this for each matmul
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb, torch_dtype=compute_dtype, device_map={"": 0}
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    elif args.method == "lora":
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=compute_dtype, device_map={"": 0})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # With a frozen base, the embedding output has requires_grad=False, so
        # checkpointed segments would have nothing to back-propagate into.
        model.enable_input_require_grads()
    else:
        # fp32 master weights; the Trainer's fp16/bf16 autocast does the compute.
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32, device_map={"": 0})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.method in ("qlora", "lora"):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,           # update is scaled by alpha / r
            lora_dropout=args.lora_dropout,
            target_modules=TARGET_MODULES,        # all attention + MLP projections
            bias="none",
            task_type="CAUSAL_LM",
        ))
        trainable, total = model.get_nb_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.config.use_cache = False   # KV cache is useless in training and conflicts with checkpointing
    return model, int(trainable), int(total)


def tokenise_split(rows, tokenizer, max_len, limit=None):
    from finetune.data import build_example

    examples, skipped, truncated = [], 0, 0
    for r in rows[:limit] if limit else rows:
        ex = build_example(tokenizer, r["question"], r["answer"], max_len)
        if ex is None:
            skipped += 1
            continue
        truncated += ex.pop("truncated")
        examples.append(ex)
    return examples, {"kept": len(examples), "skipped_prompt_too_long": skipped, "answer_truncated": truncated}


def main(argv=None):
    args = parse_args(argv)

    import random

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    from finetune.data import CompletionCollator, read_jsonl

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16   # T4 → fp16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(Path(args.data_dir) / "train.jsonl")
    val_rows = read_jsonl(Path(args.data_dir) / "val.jsonl")
    random.Random(args.seed).shuffle(train_rows)
    train_ex, train_stats = tokenise_split(train_rows, tokenizer, args.max_len, args.max_train_samples)
    val_ex, val_stats = tokenise_split(val_rows, tokenizer, args.max_len, args.max_eval_samples)
    print(f"train {train_stats} | val {val_stats}")

    model, trainable, total = load_model(args, compute_dtype)
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=0.3 if args.method != "full" else 1.0,   # 0.3 as in the QLoRA paper
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim="paged_adamw_8bit",             # 8-bit optimiser states, paged to CPU on memory spikes
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,                 # batches of similar length → less padding
        length_column_name="length",
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_ex),
        eval_dataset=Dataset.from_list(val_ex),
        data_collator=CompletionCollator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    train_minutes = (time.time() - start) / 60
    final_eval = trainer.evaluate()

    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))

    summary = {
        "method": args.method,
        "base_model": args.base_model,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "lr": args.lr,
        "effective_batch": args.batch_size * args.grad_accum,
        "epochs_planned": args.epochs,
        "global_steps": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": round(final_eval["eval_loss"], 4),
        "best_eval_answer_perplexity": round(math.exp(final_eval["eval_loss"]), 3),
        "train_minutes": round(train_minutes, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0),
        "train_data": train_stats,
        "val_data": val_stats,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "targets": TARGET_MODULES} if args.method != "full" else None,
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
