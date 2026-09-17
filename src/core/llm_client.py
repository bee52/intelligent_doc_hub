"""Provider-neutral model adapters used by the RAG layers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from langchain_core.embeddings import Embeddings


class TextGenerator(Protocol):
	"""Minimal generation contract used by retrieval code."""

	def invoke(self, prompt: str) -> Any:
		...


class CallableGenerator:
	"""Adapt a plain prompt function to the generator contract."""

	def __init__(self, generate: Callable[[str], Any]) -> None:
		self._generate = generate

	def invoke(self, prompt: str) -> Any:
		return self._generate(prompt)


class CallableEmbeddings(Embeddings):
	"""Adapt document and query embedding functions to LangChain."""

	def __init__(
		self,
		embed_documents_fn: Callable[[list[str]], list[list[float]]],
		embed_query_fn: Callable[[str], list[float]],
	) -> None:
		self._embed_documents_fn = embed_documents_fn
		self._embed_query_fn = embed_query_fn

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		return self._embed_documents_fn(texts)

	def embed_query(self, text: str) -> list[float]:
		return self._embed_query_fn(text)


def validate_embedding_dimensions(
	embeddings: Sequence[Sequence[float]], expected: int
) -> None:
	"""Validate that vectors have the expected dimensionality."""
	if any(len(vector) != expected for vector in embeddings):
		raise ValueError("embedding vectors have inconsistent dimensions")