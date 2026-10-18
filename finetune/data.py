"""
Fine-tuning data: split the agri Q&A dataset, then turn each pair into a
tokenised training example whose loss covers ONLY the answer.

    python -m finetune.data --out data/finetune

Writes train/val/test.jsonl and data_summary.json (cleaning counts, split
sizes, and a leakage report that should be all zeros).

Loss masking: the model input is  [system + question prompt][answer]<end>.
If every token were scored, a large share of the gradient would go into
re-learning the fixed system prompt and chat template (identical in every
example) instead of how to answer. So prompt positions get label −100, which
PyTorch's cross-entropy ignores, and only answer tokens (plus the end-of-turn
token, so the model learns to stop) contribute to the loss.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.agri_qa import assign_groups, clean_pairs, group_split, leakage_report, load_raw
from src.llm.prompts import FT_SYSTEM_PROMPT, build_user_message

IGNORE_INDEX = -100


# ── Split ──────────────────────────────────────────────────────────────────────

def prepare(out_dir: str, val_frac: float = 0.05, test_frac: float = 0.05) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs, cleaning = clean_pairs(load_raw())
    n_groups = assign_groups(pairs)
    splits = group_split(pairs, val_frac=val_frac, test_frac=test_frac)
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for p in rows:
                f.write(json.dumps({"question": p.question, "answer": p.answer, "group": p.group}, ensure_ascii=False) + "\n")
    summary = {
        "cleaning": cleaning,
        "n_groups": n_groups,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "leakage": leakage_report(splits),
    }
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Tokenisation with answer-only labels ──────────────────────────────────────

def chat_messages(question: str, context: str = "") -> list[dict]:
    return [
        {"role": "system", "content": FT_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, context)},
    ]


def build_example(tokenizer, question: str, answer: str, max_len: int, context: str = "") -> Optional[dict]:
    """input_ids + labels (prompt masked with −100). None if the prompt alone is too long.

    Prompt and answer are tokenised separately and concatenated, so the mask
    boundary is exact. That relies on the chat template being prefix-consistent
    (rendering [prompt] is a prefix of rendering [prompt + answer]); it's
    asserted rather than assumed, because a template that isn't would silently
    shift the mask by a few tokens.
    """
    messages = chat_messages(question, context)
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}], tokenize=False
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("Chat template is not prefix-consistent; the loss mask boundary would be wrong.")
    completion_text = full_text[len(prompt_text):]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) >= max_len - 8:
        return None
    truncated = len(prompt_ids) + len(completion_ids) > max_len
    completion_ids = completion_ids[: max_len - len(prompt_ids)]

    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    return {"input_ids": input_ids, "labels": labels, "length": len(input_ids), "truncated": truncated}


@dataclass
class CompletionCollator:
    """Right-pads a batch; padded positions get label −100 and attention 0."""
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = width - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/finetune")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    args = ap.parse_args(argv)
    print(json.dumps(prepare(args.out, args.val_frac, args.test_frac), indent=2))


if __name__ == "__main__":
    main()
