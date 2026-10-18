"""
Evaluate a model on the held-out TEST split. Run it once per variant and
compare the JSON files:

    # base model (no fine-tuning) — the baseline every claim is measured against
    python -m finetune.evaluate --label base
    # QLoRA adapter
    python -m finetune.evaluate --label qlora --adapter outputs/finetune/qlora/final
    # same adapter, base loaded in 8-bit / 4-bit (post-training quantisation for serving)
    python -m finetune.evaluate --label qlora-int8 --adapter outputs/finetune/qlora/final --load 8bit
    # with retrieved context from an index built from the TRAIN split only
    python -m finetune.evaluate --label qlora+rag --adapter ... --rag-index data/rag_index_train_only

Metrics:
  answer_perplexity  exp(mean NLL of the reference answer tokens), prompt masked —
                     how likely the model finds the correct answer. Deterministic
                     and cheap, so it runs on the full test set.
  rouge_l / token_f1 overlap of a greedy-decoded answer with the reference.
  judge (optional)   a larger LLM scores correctness & helpfulness 1–5 — overlap
                     metrics undervalue correct paraphrases.
  memory / latency   GPU memory after load, seconds per generated answer.

The RAG variant must use an index built WITHOUT test answers, otherwise the
retriever could hand the model the gold answer and the score would be meaningless.
"""

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Optional

JUDGE_KEYS = ("correctness", "helpfulness")
JUDGE_PROMPT = """You are grading an agricultural assistant's answer for Indian farmers.
Question: {question}
Reference answer: {reference}
Assistant's answer: {prediction}

Score the assistant's answer from 1 to 5 on:
- correctness: is it factually consistent with the reference and with sound agronomy? (5 = fully correct)
- helpfulness: would a farmer be able to act on it? (5 = clear and actionable)
A correct answer phrased differently from the reference should still score high.
Reply with JSON only: {{"correctness": <1-5>, "helpfulness": <1-5>}}"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA/QLoRA adapter dir")
    ap.add_argument("--full-model", default=None, help="fully fine-tuned model dir (instead of base+adapter)")
    ap.add_argument("--load", choices=["fp16", "8bit", "4bit"], default="fp16")
    ap.add_argument("--test-file", default="data/finetune/test.jsonl")
    ap.add_argument("--rag-index", default=None)
    ap.add_argument("--n-ppl", type=int, default=None, help="rows for perplexity (default: all)")
    ap.add_argument("--n-gen", type=int, default=300, help="rows to generate answers for")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge via Groq (needs GROQ_API_KEY)")
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    ap.add_argument("--n-judge", type=int, default=100)
    ap.add_argument("--out", default="results/finetune")
    return ap.parse_args(argv)


def load(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    path = args.full_model or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": {"": 0}, "torch_dtype": torch.float16}
    if args.load == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer, torch.cuda.max_memory_allocated() / 1e9


def make_context_fn(index_dir):
    if not index_dir:
        return lambda q: ""
    from src.retrieval.embedder import SentenceTransformerEmbedder
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.index import KnowledgeIndex
    from src.retrieval.reranker import CrossEncoderReranker

    retriever = HybridRetriever(KnowledgeIndex.load(index_dir), SentenceTransformerEmbedder(), CrossEncoderReranker())
    return lambda q: "\n".join(f"- {h.chunk.text}" for h in retriever.retrieve(q, top_n=3))


def answer_perplexity(model, tokenizer, rows, contexts, args) -> dict:
    import torch

    from finetune.data import CompletionCollator, build_example

    examples = [build_example(tokenizer, r["question"], r["answer"], args.max_len, c) for r, c in zip(rows, contexts)]
    examples = [e for e in examples if e is not None]
    collate = CompletionCollator(tokenizer.pad_token_id)
    nll_sum, n_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(examples), args.batch_size):
            batch = {k: v.to(model.device) for k, v in collate(examples[i:i + args.batch_size]).items()}
            out = model(**batch)
            n = int((batch["labels"][:, 1:] != -100).sum())   # tokens actually scored (labels shift by one)
            nll_sum += out.loss.item() * n                       # HF returns the mean over those tokens
            n_tokens += n
    mean_nll = nll_sum / n_tokens
    return {"answer_nll": round(mean_nll, 4), "answer_perplexity": round(math.exp(mean_nll), 3),
            "n_rows": len(examples), "n_answer_tokens": n_tokens}


def generate(model, tokenizer, rows, contexts, args) -> tuple[list[str], float]:
    import torch

    from finetune.data import chat_messages

    tokenizer.padding_side = "left"          # decoder-only generation needs prompts aligned on the right
    preds, seconds = [], 0.0
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        prompts = [tokenizer.apply_chat_template(chat_messages(r["question"], c), add_generation_prompt=True, tokenize=False)
                   for r, c in zip(chunk, contexts[i:i + args.batch_size])]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        start = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
        seconds += time.time() - start
        preds += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [p.strip() for p in preds], seconds / max(len(rows), 1)


def judge_one(client, model: str, question: str, reference: str, prediction: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, prediction=prediction)}],
    )
    s = json.loads(resp.choices[0].message.content)
    return {k: float(s[k]) for k in JUDGE_KEYS}


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return [round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot) - 1], 3)]


def summarise_judge(per_row: list[Optional[dict]], model: str) -> dict:
    """Mean and 95% bootstrap CI per judge score, over the rows that were scored."""
    scored = [s for s in per_row if s]
    out = {"judge_n": len(scored), "judge_model": model}
    for k in JUDGE_KEYS:
        if scored:
            values = [s[k] for s in scored]
            out[f"judge_{k}"] = round(statistics.mean(values), 3)
            out[f"judge_{k}_ci95"] = bootstrap_ci(values)
    return out


def judge(rows, preds, args) -> list[Optional[dict]]:
    """Per-row judge scores (None where not judged), kept so a partial run can be resumed."""
    from groq import Groq

    # Per-minute 429s are retried by the client. A daily token limit is not; the
    # skipped rows can be scored later with `python -m finetune.judge_generations`.
    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=8)
    per_row = []
    for i, (r, p) in enumerate(zip(rows, preds)):
        if i >= args.n_judge:
            per_row.append(None)
            continue
        try:
            per_row.append(judge_one(client, args.judge_model, r["question"], r["answer"], p))
        except Exception as e:
            print(f"judge error (skipped): {e}")
            per_row.append(None)
    return per_row


def main(argv=None):
    args = parse_args(argv)
    from finetune.data import read_jsonl
    from finetune.metrics import rouge_l_f1, token_f1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.test_file)
    model, tokenizer, load_mem_gb = load(args)
    context_for = make_context_fn(args.rag_index)

    ppl_rows = rows[: args.n_ppl] if args.n_ppl else rows
    ppl = answer_perplexity(model, tokenizer, ppl_rows, [context_for(r["question"]) for r in ppl_rows], args)

    gen_rows = rows[: args.n_gen]
    gen_contexts = [context_for(r["question"]) for r in gen_rows]
    preds, sec_per_answer = generate(model, tokenizer, gen_rows, gen_contexts, args)

    result = {
        "label": args.label,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "full_model": args.full_model,
        "load": args.load,
        "rag_index": args.rag_index,
        **ppl,
        "rouge_l": round(statistics.mean(rouge_l_f1(p, r["answer"]) for p, r in zip(preds, gen_rows)), 4),
        "token_f1": round(statistics.mean(token_f1(p, r["answer"]) for p, r in zip(preds, gen_rows)), 4),
        "avg_answer_words": round(statistics.mean(len(p.split()) for p in preds), 1),
        "empty_answer_rate": round(sum(not p for p in preds) / len(preds), 4),
        "n_generated": len(preds),
        "sec_per_answer": round(sec_per_answer, 3),
        "gpu_mem_after_load_gb": round(load_mem_gb, 3),
    }
    judged = judge(gen_rows, preds, args) if args.judge else [None] * len(preds)
    if args.judge:
        result |= summarise_judge(judged, args.judge_model)

    (out_dir / f"eval_{args.label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out_dir / f"generations_{args.label}.jsonl", "w", encoding="utf-8") as f:
        for r, p, c, s in zip(gen_rows, preds, gen_contexts, judged):
            row = {"question": r["question"], "reference": r["answer"], "prediction": p, "context": c}
            if s:
                row["judge"] = s
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
