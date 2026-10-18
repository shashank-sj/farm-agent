"""
Build results/finetune/SUMMARY.md from the eval_*.json and *_train_summary.json
files in the results directory.

    python -m finetune.summarize
"""

import argparse
import glob
import json
import os

EVAL_COLS = ["answer_perplexity", "rouge_l", "token_f1", "judge_correctness", "judge_helpfulness", "judge_n",
             "avg_answer_words", "sec_per_answer", "gpu_mem_after_load_gb"]


def build(results_dir: str) -> str:
    lines = ["| variant | " + " | ".join(EVAL_COLS) + " |", "|---" * (len(EVAL_COLS) + 1) + "|"]
    for f in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        lines.append(f"| {r['label']} | " + " | ".join(str(r.get(c, "-")) for c in EVAL_COLS) + " |")
    lines += ["", "| run | trainable % | steps | best eval loss | answer ppl (val) | minutes | peak GPU GB |",
              "|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(os.path.join(results_dir, "*_train_summary.json"))):
        s = json.load(open(f, encoding="utf-8"))
        lines.append(f"| {s['method']} | {s['trainable_pct']} | {s['global_steps']} | {s['best_eval_loss']} | "
                     f"{s['best_eval_answer_perplexity']} | {s['train_minutes']} | {s['peak_gpu_mem_gb']} |")
    lines += ["", "Judge: mean of 1–5 scores on the same first `judge_n` test questions; "
                  "95% bootstrap CIs are in each eval_*.json (judge_*_ci95)."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/finetune")
    args = ap.parse_args(argv)
    summary = build(args.results_dir)
    with open(os.path.join(args.results_dir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
