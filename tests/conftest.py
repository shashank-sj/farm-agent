"""Shared fakes so the suite runs offline: no API keys, no model downloads."""

import zlib

import numpy as np
import pytest

from src.retrieval.bm25 import tokenize


class HashEmbedder:
    """Deterministic bag-of-words embedder — stands in for BGE in tests."""
    model_name = "hash-test"
    dim = 512

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype("float32")

    def embed_query(self, text):
        return self._vec(text).reshape(1, -1)


class KeywordReranker:
    """Scores a passage by whether it contains a chosen keyword."""
    model_name = "keyword-test"

    def __init__(self, keyword: str):
        self.keyword = keyword

    def score(self, query, texts):
        return [1.0 if self.keyword in t.lower() else 0.0 for t in texts]


@pytest.fixture
def embedder():
    return HashEmbedder()
