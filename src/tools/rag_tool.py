"""FarmRAGTool — searches the farm knowledge base (FAISS + Gemini embeddings)."""

import os
import logging
from pathlib import Path
from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger("farm-tools")


class RAGInput(BaseModel):
    query: str = Field(description="Farming question to search the knowledge base for")


class FarmRAGTool:
    """Retrieval-augmented search over the farm knowledge base."""

    def __init__(self, gemini_api_key: str, index_path: str = "data/faiss_index"):
        self.gemini_api_key = gemini_api_key
        self.index_path = index_path
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                os.environ["GOOGLE_API_KEY"] = self.gemini_api_key
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                if Path(self.index_path).exists():
                    vectorstore = FAISS.load_local(
                        self.index_path, embeddings, allow_dangerous_deserialization=True
                    )
                    self._pipeline = vectorstore.as_retriever(search_kwargs={"k": 5})
                    logger.info("RAG index loaded ✓")
                else:
                    logger.warning(f"RAG index not found at {self.index_path}. Using fallback.")
                    self._pipeline = "fallback"
            except Exception as e:
                logger.error(f"RAG init error: {e}")
                self._pipeline = "fallback"
        return self._pipeline

    def run(self, query: str) -> str:
        pipeline = self._get_pipeline()
        if pipeline == "fallback":
            return "Knowledge base not available. Please build the RAG index first by uploading farm documents."
        try:
            docs = pipeline.invoke(query)
            if not docs:
                return "No relevant information found in knowledge base for this query."
            context = "\n\n".join([d.page_content for d in docs[:3]])
            return f"From knowledge base:\n{context}"
        except Exception as e:
            return f"RAG search error: {str(e)}"

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
