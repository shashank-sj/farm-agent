"""
Retrieval benchmark: dense vs BM25 vs hybrid (RRF) vs dense+rerank vs hybrid+rerank.

Setup (standard question → answer-passage retrieval, like FiQA or BioASQ):
  corpus   every unique cleaned answer in the agri Q&A dataset
  queries  one question per leakage group (all splits by default — nothing in the
           retriever is trained on this data, so there is no train/test leakage
           to avoid, and using every group gives ~20× more queries than the test
           split alone)
  gold     the answer(s) belonging to that question's group

Metrics (top 10), each with a 95% bootstrap confidence interval:
  Recall@k  share of queries with a correct answer in the top k
  MRR@10    mean of 1/rank of the first correct answer (0 if none in top 10)
  nDCG@10   rank-discounted gain, normalised by the best possible ordering
  latency   per-query p50 / p95 in milliseconds
Plus a PAIRED bootstrap of each mode's difference from dense on Recall@5 and
MRR@10: the same queries are resampled for both modes, so the interval says
whether a difference is real or just noise.

Usage:
    python -m scripts.eval_retrieval --out results/retrieval
"""

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.agri_qa import answer_key, assign_groups, clean_pairs, group_split, load_raw
from src.retrieval.embedder import SentenceTransformerEmbedder
from src.retrieval.hybrid import MODES, HybridRetriever
from src.retrieval.index import Chunk, KnowledgeIndex
from src.retrieval.reranker import CrossEncoderReranker

K = 10
RECALL_AT = (1, 3, 5, 10)
N_BOOT = 2000


def ranking_metrics(ranked_ids: list[int], relevant: set[int], k: int = K) -> dict:
    ranked = ranked_ids[:k]
    first_hit = next((r for r, doc in enumerate(ranked, 1) if doc in relevant), None)
    dcg = sum(1.0 / math.log2(r + 1) for r, doc in enumerate(ranked, 1) if doc in relevant)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    out = {f"recall@{n}": float(first_hit is not None and first_hit <= n) for n in RECALL_AT}
    out["mrr@10"] = 1.0 / first_hit if first_hit else 0.0
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def build_corpus(pairs):
    """One chunk per unique answer; map each leakage group to its answer chunk ids."""
    chunk_of_answer: dict[str, int] = {}
    chunks: list[Chunk] = []
    group_answers: dict[str, set[int]] = defaultdict(set)
    for p in pairs:
        key = answer_key(p.answer)
        if key not in chunk_of_answer:
            chunk_of_answer[key] = len(chunks)
            chunks.append(Chunk(id=len(chunks), text=p.answer, source="agri-qa", metadata={"split": p.split}))
        group_answers[p.group].add(chunk_of_answer[key])
    return chunks, group_answers


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test", help="which splits' groups to use as queries")
    ap.add_argument("--n-queries", type=int, default=None, help="cap on queries (default: all groups)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/retrieval")
    ap.add_argument("--rerank-pool", type=int, default=20)
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, clean_stats = clean_pairs(load_raw())
    n_groups = assign_groups(pairs)
    splits = group_split(pairs)
    chunks, group_answers = build_corpus(pairs)

    wanted = set(args.splits.split(","))
    by_group = {}
    for name in ("train", "val", "test"):
        if name in wanted:
            for p in splits[name]:
                by_group.setdefault(p.group, p)         # one query per group
    queries = list(by_group.values())
    random.Random(args.seed).shuffle(queries)
    if args.n_queries:
        queries = queries[: args.n_queries]
    print(f"corpus: {len(chunks)} answers | groups: {n_groups} | queries: {len(queries)} ({args.splits})")

    embedder = SentenceTransformerEmbedder()
    t0 = time.time()
    index = KnowledgeIndex.build(chunks, embedder)
    print(f"embedded corpus in {time.time() - t0:.1f}s")
    retriever = HybridRetriever(index, embedder, CrossEncoderReranker(), rerank_pool=args.rerank_pool)

    rng = np.random.default_rng(args.seed)
    per_mode: dict[str, dict[str, np.ndarray]] = {}
    results = {}
    for mode in MODES:
        rows, latencies = [], []
        for q in queries:
            start = time.perf_counter()
            hits = retriever.retrieve(q.question, top_n=K, mode=mode)
            latencies.append((time.perf_counter() - start) * 1000)
            rows.append(ranking_metrics([h.chunk.id for h in hits], group_answers[q.group]))
        arrays = {m: np.array([r[m] for r in rows]) for m in rows[0]}
        per_mode[mode] = arrays
        summary = {}
        for m, vals in arrays.items():
            lo, hi = bootstrap_ci(vals, rng)
            summary[m] = {"mean": round(float(vals.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
        summary["latency_ms_p50"] = round(statistics.median(latencies), 1)
        summary["latency_ms_p95"] = round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1)
        results[mode] = summary
        print(mode, {m: v["mean"] if isinstance(v, dict) else v for m, v in summary.items()})

    paired = {}
    for mode in MODES:
        if mode == "dense":
            continue
        paired[mode] = {}
        for m in ("recall@5", "mrr@10"):
            diff = per_mode[mode][m] - per_mode["dense"][m]
            lo, hi = bootstrap_ci(diff, rng)
            paired[mode][m] = {"delta_vs_dense": round(float(diff.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "significant": bool(lo > 0 or hi < 0)}

    report = {
        "dataset_cleaning": clean_stats,
        "corpus_size": len(chunks),
        "n_groups": n_groups,
        "n_queries": len(queries),
        "query_splits": args.splits,
        "embedding_model": embedder.model_name,
        "reranker": retriever.reranker.model_name,
        "rerank_pool": args.rerank_pool,
        "results": results,
        "paired_vs_dense": paired,
    }
    (out_dir / "retrieval_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cols = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    lines = [f"Queries: {len(queries)} · corpus: {len(chunks)} answers · 95% bootstrap CI in brackets", "",
             "| mode | " + " | ".join(cols) + " | p50 ms | p95 ms |", "|---" * (len(cols) + 3) + "|"]
    for mode, r in results.items():
        cells = [f"{r[c]['mean']:.3f} [{r[c]['ci95'][0]:.3f}–{r[c]['ci95'][1]:.3f}]" for c in cols]
        lines.append(f"| {mode} | " + " | ".join(cells) + f" | {r['latency_ms_p50']} | {r['latency_ms_p95']} |")
    lines += ["", "| mode vs dense | Δ recall@5 [CI] | Δ MRR@10 [CI] |", "|---|---|---|"]
    for mode, p in paired.items():
        cells = [f"{p[m]['delta_vs_dense']:+.3f} [{p[m]['ci95'][0]:+.3f}, {p[m]['ci95'][1]:+.3f}]"
                 f"{' *' if p[m]['significant'] else ''}" for m in ("recall@5", "mrr@10")]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines.append("\n`*` = 95% CI excludes zero.")
    (out_dir / "retrieval_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
