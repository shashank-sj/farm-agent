"""Cross-encoder re-ranker: the second, more expensive retrieval stage.

A bi-encoder (the embedder) encodes query and passage separately, so passage
vectors can be precomputed — fast, but the two texts never "see" each other.
A cross-encoder reads (query, passage) together through full attention, which
is far more accurate but has to run once per candidate at query time. So it is
only applied to the small candidate pool the first stage returns.
"""

from typing import Optional

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # 22M params


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, device: Optional[str] = None, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score per text. Only the ordering is meaningful — the scale
        is model-specific, so any cut-off threshold must be tuned on eval data."""
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False)
        return [float(s) for s in scores]
