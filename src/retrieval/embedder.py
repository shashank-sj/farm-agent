"""Sentence-embedding model used for the dense half of hybrid retrieval.

Runs locally (sentence-transformers), so building and querying the index needs
no API key and has no per-call cost or rate limit.
"""

from typing import Optional, Protocol

import numpy as np

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 33M params, 384-dim, fast on CPU

# BGE v1.5 models were trained with this instruction on the *query* side only
# (passages are embedded as-is). It helps short queries match longer passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """L2-normalised embeddings, so inner product == cosine similarity."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        query_prefix: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        if query_prefix is None:
            query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.query_prefix = query_prefix
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 512,
        )
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [self.query_prefix + text], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vec, dtype="float32")
