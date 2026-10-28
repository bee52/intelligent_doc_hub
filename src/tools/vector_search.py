"""Vector-store retrieval tool for Agentic RAG."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class VectorStoreTool:
    """Adapt a FAISS vector store to the Agentic RAG retrieval protocol."""

    name = "vector"

    def __init__(self, vector_store: FAISS, default_k: int = 4) -> None:
        if default_k <= 0:
            raise ValueError("default_k must be greater than zero")
        self._vector_store = vector_store
        self._default_k = default_k

    def search(self, query: str) -> list[Document]:
        """Return relevant documents for a non-empty query."""
        if not query.strip():
            raise ValueError("query must not be empty")
        return self._vector_store.similarity_search(query, k=self._default_k)
