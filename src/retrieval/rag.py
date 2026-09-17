"""Retrieval and generation orchestration for the document RAG pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class Generator(Protocol):
	"""Minimal interface required from an injected generation model."""

	def invoke(self, prompt: str) -> Any:
		...


@dataclass(frozen=True)
class RAGResponse:
	"""Generated answer together with the documents used as context."""

	answer: str
	sources: list[Document]


def retrieve_documents(
	vector_store: FAISS, query: str, k: int = 4
) -> list[Document]:
	"""Retrieve up to *k* relevant documents for *query*."""
	if not query.strip():
		raise ValueError("query must not be empty")
	if k <= 0:
		raise ValueError("k must be greater than zero")

	return vector_store.similarity_search(query, k=k)


def build_context(documents: Sequence[Document]) -> str:
	"""Format retrieved documents as grounded context for a model prompt."""
	sections: list[str] = []
	for index, document in enumerate(documents, start=1):
		source = document.metadata.get("source", "unknown")
		page = document.metadata.get("page")
		location = f"{source}, page {page}" if page is not None else str(source)
		sections.append(f"[Source {index}: {location}]\n{document.page_content}")
	return "\n\n".join(sections)


def generate_answer(
	vector_store: FAISS,
	generator: Generator,
	query: str,
	*,
	k: int = 4,
) -> RAGResponse:
	"""Retrieve context and ask *generator* for a grounded answer."""
	documents = retrieve_documents(vector_store, query, k=k)
	context = build_context(documents)
	prompt = (
		"Answer the question using only the supplied context. "
		"If the context does not contain the answer, say you do not know.\n\n"
		f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
	)
	result = generator.invoke(prompt)
	answer = getattr(result, "content", result)
	if not isinstance(answer, str):
		answer = str(answer)

	return RAGResponse(answer=answer, sources=documents)