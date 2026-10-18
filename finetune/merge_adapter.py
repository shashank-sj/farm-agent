"""
Merge a LoRA/QLoRA adapter into its base model and save a standalone model
(what src/llm/local_llm.py loads via LOCAL_LLM_MODEL).

    python -m finetune.merge_adapter --adapter outputs/finetune/qlora/final \
        --out outputs/finetune/merged [--push-to-hub your-name/qwen2.5-0.5b-farm]

The base is reloaded in fp16 — NOT in 4-bit — before merging. Merging into a
4-bit base would round the merged weights back onto the 4-bit grid and throw
away part of what the adapter learned. Merged: W' = W + (alpha/r)·B·A, so
serving needs no PEFT and has zero adapter overhead.
"""

import argparse
import os


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="outputs/finetune/merged")
    ap.add_argument("--push-to-hub", default=None, help="Hub repo id, e.g. your-name/qwen2.5-0.5b-farm (needs HF_TOKEN)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"merged model saved to {args.out}")

    if args.push_to_hub:
        token = os.environ["HF_TOKEN"]
        model.push_to_hub(args.push_to_hub, token=token, private=args.private)
        tokenizer.push_to_hub(args.push_to_hub, token=token, private=args.private)
        print(f"pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
