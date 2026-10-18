"""
Hybrid retrieval: dense (FAISS) + sparse (BM25) → Reciprocal Rank Fusion → cross-encoder.

    query ──► dense top-k_dense  ─┐
          └─► BM25  top-k_sparse ─┴─► RRF ─► top rerank_pool ─► cross-encoder ─► top_n

Why both? Dense embeddings match meaning ("my wheat leaves are turning yellow"
→ nitrogen deficiency), but blur exact tokens. BM25 nails exact tokens — product
names, scheme acronyms (PMFBY, SMAM), variety codes (PR-126, HD-2967) — but
misses paraphrases.

What the benchmark showed (scripts/eval_retrieval.py, 1,922 paraphrase-style
queries, bootstrap CIs): the cross-encoder is what helps (+2.7 MRR@10 over
dense, significant); RRF *without* re-ranking is worse than dense alone, because
the weak BM25 list pulls wrong answers up; after re-ranking, adding BM25 is
neutral. hybrid_rerank stays the default because it costs no extra latency and
covers exact-token queries the Q&A benchmark doesn't contain — and without a
re-ranker we fall back to dense, not to plain hybrid.

Why RRF instead of adding scores? Cosine similarities (~0.3–0.9) and BM25 scores
(unbounded, query-dependent) live on different scales, so a weighted sum needs
per-query normalisation and tuning. RRF only uses ranks:
    rrf(d) = Σ_r 1 / (k + rank_r(d))      (k = 60, from Cormack et al., 2009)
so it is scale-free and has a single, insensitive parameter.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from src.retrieval.bm25 import BM25
from src.retrieval.index import Chunk, KnowledgeIndex

logger = logging.getLogger("farm-retrieval")

MODES = ("dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float                       # score from the last stage that ran
    dense_rank: Optional[int] = None   # 1-based rank in the dense list, if present
    sparse_rank: Optional[int] = None  # 1-based rank in the BM25 list, if present


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse several ranked id lists into one, best first."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        index: KnowledgeIndex,
        embedder,
        reranker=None,
        k_dense: int = 20,
        k_sparse: int = 20,
        rrf_k: int = 60,
        rerank_pool: int = 20,
    ):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.k_dense = k_dense
        self.k_sparse = k_sparse
        self.rrf_k = rrf_k
        self.rerank_pool = rerank_pool
        # BM25 is rebuilt from chunks.jsonl at load time — well under a second
        # for a few thousand chunks, so there's no second artifact to keep in sync.
        self.bm25 = BM25.from_texts([c.text for c in index.chunks])

    def retrieve(self, query: str, top_n: int = 4, mode: str = "hybrid_rerank") -> list[RetrievedChunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        chunks = self.index.chunks

        if mode == "hybrid_rerank" and self.reranker is None:
            # Unreranked RRF measured worse than dense, so dense is the safer fallback.
            logger.warning("hybrid_rerank requested but no reranker loaded — returning dense results")
            mode = "dense"

        if mode == "dense":
            hits = self.index.search(self.embedder.embed_query(query), top_n)
            return [RetrievedChunk(chunks[i], s, dense_rank=r) for r, (i, s) in enumerate(hits, 1)]

        if mode == "bm25":
            hits = self.bm25.top_k(query, top_n)
            return [RetrievedChunk(chunks[i], s, sparse_rank=r) for r, (i, s) in enumerate(hits, 1)]

        if mode == "dense_rerank":
            # Isolates the re-ranker's contribution: same pool size, no BM25.
            if self.reranker is None:
                raise ValueError("dense_rerank needs a reranker")
            pool = [i for i, _ in self.index.search(self.embedder.embed_query(query), self.rerank_pool)]
            scores = self.reranker.score(query, [chunks[i].text for i in pool])
            dense_rank = {doc_id: r for r, doc_id in enumerate(pool, 1)}
            reranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[:top_n]
            return [RetrievedChunk(chunks[i], s, dense_rank=dense_rank[i]) for i, s in reranked]

        dense_ids = [i for i, _ in self.index.search(self.embedder.embed_query(query), self.k_dense)]
        sparse_ids = [i for i, _ in self.bm25.top_k(query, self.k_sparse)]
        dense_rank = {doc_id: r for r, doc_id in enumerate(dense_ids, 1)}
        sparse_rank = {doc_id: r for r, doc_id in enumerate(sparse_ids, 1)}
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.rrf_k)

        if mode == "hybrid":
            return [
                RetrievedChunk(chunks[i], s, dense_rank.get(i), sparse_rank.get(i))
                for i, s in fused[:top_n]
            ]

        pool = [i for i, _ in fused[: self.rerank_pool]]
        scores = self.reranker.score(query, [chunks[i].text for i in pool])
        reranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            RetrievedChunk(chunks[i], s, dense_rank.get(i), sparse_rank.get(i))
            for i, s in reranked
        ]
