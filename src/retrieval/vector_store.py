"""FAISS vector-store helpers for chunked LangChain documents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
	from collections.abc import Sequence


logger = logging.getLogger(__name__)


def build_vector_store(
	chunks: Sequence[Document], embedding_model: Embeddings
) -> FAISS:
	"""Embed *chunks* and return a searchable FAISS vector store."""
	if not chunks:
		raise ValueError("at least one chunk is required to build a vector store")

	vector_store = FAISS.from_documents(list(chunks), embedding_model)
	logger.info("Stored %d embedded chunks", len(chunks))
	return vector_store


def save_vector_store(vector_store: FAISS, directory: str | Path) -> None:
	"""Persist a FAISS vector store to *directory*."""
	path = Path(directory)
	path.mkdir(parents=True, exist_ok=True)
	vector_store.save_local(str(path))


def load_vector_store(
	directory: str | Path,
	embedding_model: Embeddings,
	*,
	allow_dangerous_deserialization: bool = False,
) -> FAISS:
	"""Load a persisted FAISS vector store from *directory*."""
	return FAISS.load_local(
		str(directory),
		embedding_model,
		allow_dangerous_deserialization=allow_dangerous_deserialization,
	)