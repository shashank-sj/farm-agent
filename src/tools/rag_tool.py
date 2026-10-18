"""FarmRAGTool — hybrid search (BM25 + FAISS + cross-encoder) over the farm knowledge base."""

import logging
import os
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.retrieval.embedder import DEFAULT_EMBED_MODEL
from src.retrieval.reranker import DEFAULT_RERANK_MODEL

logger = logging.getLogger("farm-tools")

DEFAULT_INDEX_PATH = "data/rag_index"


class RAGInput(BaseModel):
    query: str = Field(description="Farming question to search the knowledge base for")


class FarmRAGTool:
    """Retrieval over the farm knowledge base built by scripts/build_rag_index.py."""

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        top_n: int = 4,
        use_reranker: bool = True,
        embed_model: str = DEFAULT_EMBED_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
    ):
        self.index_path = index_path
        self.top_n = top_n
        self.use_reranker = use_reranker and os.getenv("RAG_DISABLE_RERANKER", "") != "1"
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self._retriever = None
        self.unavailable_reason = ""

    def _get_retriever(self):
        """Lazy-load on first query so app start-up isn't blocked on model downloads."""
        if self._retriever is None and not self.unavailable_reason:
            try:
                from src.retrieval.embedder import SentenceTransformerEmbedder
                from src.retrieval.hybrid import HybridRetriever
                from src.retrieval.index import KnowledgeIndex

                if not KnowledgeIndex.exists(self.index_path):
                    self.unavailable_reason = f"no index at {self.index_path} (run scripts/build_rag_index.py)"
                    logger.warning(f"RAG unavailable: {self.unavailable_reason}")
                    return None
                index = KnowledgeIndex.load(self.index_path, expected_embedding_model=self.embed_model)
                embedder = SentenceTransformerEmbedder(self.embed_model)
                reranker = None
                if self.use_reranker:
                    from src.retrieval.reranker import CrossEncoderReranker
                    reranker = CrossEncoderReranker(self.rerank_model)
                self._retriever = HybridRetriever(index, embedder, reranker)
                logger.info(f"RAG index loaded ✓ ({len(index.chunks)} chunks, reranker={'on' if reranker else 'off'})")
            except Exception as e:
                self.unavailable_reason = f"init error: {e}"
                logger.error(f"RAG init error: {e}")
        return self._retriever

    def run(self, query: str) -> str:
        retriever = self._get_retriever()
        if retriever is None:
            return "Knowledge base not available right now. Answer only if another tool can help; otherwise say so."
        try:
            hits = retriever.retrieve(query, top_n=self.top_n)
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return "Knowledge base search failed for this query."
        if not hits:
            return "No relevant information found in the knowledge base for this query."
        return format_hits(hits)

    def as_tool(self) -> BaseTool:
        rag_instance = self

        class _RAGTool(BaseTool):
            name: str = "farm_rag"
            description: str = (
                "Search the farm knowledge base for detailed information about: "
                "crop cultivation, soil health, fertilizers, irrigation, pest and disease "
                "management, organic farming, and government farming schemes. "
                "Use this for any farming how-to questions."
            )
            args_schema: Type[BaseModel] = RAGInput

            def _run(self, query: str) -> str:
                return rag_instance.run(query)

        return _RAGTool()


def format_hits(hits) -> str:
    """Numbered passages so the LLM can cite them as [1], [2]…"""
    blocks = ["From the knowledge base (cite passages by their number, e.g. [1]):"]
    for n, hit in enumerate(hits, 1):
        blocks.append(f"[{n}] (source: {hit.chunk.source})\n{hit.chunk.text.strip()}")
    return "\n\n".join(blocks)
