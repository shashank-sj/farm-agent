"""
Score saved answers with the LLM judge, without re-running any model.

finetune.evaluate writes every generated answer to
results/finetune/generations_<label>.jsonl. This script judges those answers,
so the judge can be run (or finished) on a CPU machine — e.g. when the judge
API's daily token limit cut the GPU run short. Scores are written back per row
(a "judge" field), so an interrupted run resumes where it stopped, and the
judge_* fields of eval_<label>.json are recomputed from the per-row scores.

All variants are judged on the same first --n-judge test questions, so their
means are compared on identical questions.

    python -m finetune.judge_generations --labels lora full qlora+rag
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from finetune.evaluate import judge_one, summarise_judge

RETRY_HINT = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")


def wait_seconds(error) -> float:
    """Groq's 429 message says how long until enough tokens free up."""
    m = RETRY_HINT.search(str(error))
    return (int(m.group(1) or 0) * 60 + float(m.group(2)) if m else 60.0) + 5


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def update_eval(results: Path, label: str, rows: list[dict], n_judge: int, model: str) -> dict:
    eval_path = results / f"eval_{label}.json"
    result = json.loads(eval_path.read_text(encoding="utf-8"))
    result = {k: v for k, v in result.items() if not k.startswith("judge_")}
    result |= summarise_judge([r.get("judge") for r in rows[:n_judge]], model)
    eval_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--results-dir", default="results/finetune")
    ap.add_argument("--n-judge", type=int, default=100)
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    ap.add_argument("--max-wait-min", type=float, default=90, help="total time to spend waiting on rate limits")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from groq import Groq, RateLimitError

    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=2)
    return judge_labels(client, args, rate_limit_error=RateLimitError)


def judge_labels(client, args, rate_limit_error, sleep=time.sleep) -> int:
    """Judge every unscored row of each label; 0 when finished, 2 when stopped by rate limits."""
    results = Path(args.results_dir)
    waited = 0.0
    for label in args.labels:
        gen_path = results / f"generations_{label}.jsonl"
        rows = read_rows(gen_path)
        todo = [i for i, r in enumerate(rows[: args.n_judge]) if not r.get("judge")]
        print(f"{label}: {len(todo)} of {min(args.n_judge, len(rows))} rows to judge")
        for done, i in enumerate(todo, 1):
            r = rows[i]
            while True:
                try:
                    r["judge"] = judge_one(client, args.judge_model, r["question"], r["reference"], r["prediction"])
                    break
                except rate_limit_error as e:
                    pause = wait_seconds(e)
                    if waited + pause > args.max_wait_min * 60:
                        write_rows(gen_path, rows)
                        update_eval(results, label, rows, args.n_judge, args.judge_model)
                        print(f"stopping: rate-limit waits would exceed {args.max_wait_min} min. "
                              "Progress is saved; re-run later to resume.")
                        return 2
                    print(f"rate limited, waiting {pause:.0f}s")
                    sleep(pause)
                    waited += pause
                except Exception as e:
                    print(f"{label} row {i}: judge error (skipped): {e}")
                    break
            if done % 10 == 0:
                write_rows(gen_path, rows)
        write_rows(gen_path, rows)
        result = update_eval(results, label, rows, args.n_judge, args.judge_model)
        print(label, {k: v for k, v in result.items() if k.startswith("judge_")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
