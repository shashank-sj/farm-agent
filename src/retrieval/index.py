"""
KnowledgeIndex — a dense FAISS index plus the chunk texts it was built from.

On disk (one directory):
    index.faiss    raw FAISS index (exact inner-product search)
    chunks.jsonl   one chunk per line; line number == FAISS row id == BM25 doc id
    manifest.json  embedding model, dimension, chunking params, build time

Why not LangChain's FAISS.save_local / load_local? That pickles the docstore,
and loading it needs allow_dangerous_deserialization=True — a tampered index
file could then execute arbitrary code. Plain JSONL + a raw FAISS index can't,
and it lets BM25 and the dense index share the exact same chunk ids.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


class KnowledgeIndex:
    def __init__(self, chunks: list[Chunk], faiss_index, manifest: dict):
        if faiss_index.ntotal != len(chunks):
            raise ValueError(
                f"Index has {faiss_index.ntotal} vectors but {len(chunks)} chunks — rebuild the index."
            )
        for position, chunk in enumerate(chunks):
            if chunk.id != position:
                raise ValueError(f"Chunk ids must equal their position (got id {chunk.id} at {position}).")
        self.chunks = chunks
        self.faiss_index = faiss_index
        self.manifest = manifest

    # ── Build / persist ──────────────────────────────────────────────────────

    @classmethod
    def build(cls, chunks: list[Chunk], embedder, extra_manifest: Optional[dict] = None) -> "KnowledgeIndex":
        import faiss

        if not chunks:
            raise ValueError("No chunks to index.")
        vectors = embedder.embed_documents([c.text for c in chunks])
        # Exact search: at tens of thousands of chunks a flat index answers in
        # milliseconds. Approximate indexes (IVF, HNSW) only pay off around 10^6+.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        manifest = {
            "embedding_model": embedder.model_name,
            "dim": int(vectors.shape[1]),
            "n_chunks": len(chunks),
            "metric": "inner product on L2-normalised vectors (= cosine)",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(extra_manifest or {}),
        }
        return cls(chunks, index, manifest)

    def save(self, directory: str | Path) -> None:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(directory / INDEX_FILE))
        with open(directory / CHUNKS_FILE, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        (directory / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, expected_embedding_model: Optional[str] = None) -> "KnowledgeIndex":
        import faiss

        directory = Path(directory)
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        if expected_embedding_model and manifest.get("embedding_model") != expected_embedding_model:
            # Querying with a different model than the one that built the index
            # silently returns garbage, so fail loudly instead.
            raise ValueError(
                f"Index was built with {manifest.get('embedding_model')!r} but the retriever "
                f"is configured for {expected_embedding_model!r}. Rebuild the index."
            )
        with open(directory / CHUNKS_FILE, encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        index = faiss.read_index(str(directory / INDEX_FILE))
        return cls(chunks, index, manifest)

    @staticmethod
    def exists(directory: str | Path) -> bool:
        directory = Path(directory)
        return all((directory / name).exists() for name in (INDEX_FILE, CHUNKS_FILE, MANIFEST_FILE))

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """(chunk_id, cosine) pairs for the k nearest chunks, best first."""
        k = min(k, self.faiss_index.ntotal)
        if k <= 0:
            return []
        scores, ids = self.faiss_index.search(np.asarray(query_vector, dtype="float32").reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
