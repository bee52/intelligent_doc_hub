"""Named entry point for the basic retrieve-then-generate RAG flow."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from src.core.config import Settings
from src.ingestion.chunker import load_and_chunk_documents
from src.retrieval.rag import Generator, RAGResponse, generate_answer
from src.retrieval.vector_store import build_vector_store, load_vector_store


def answer_question(
	vector_store: FAISS,
	generator: Generator,
	query: str,
	*,
	k: int = 4,
) -> RAGResponse:
	"""Answer *query* with one retrieval pass and source-attributed context."""
	return generate_answer(vector_store, generator, query, k=k)


def run_basic_rag(
	query: str,
	embedding_model: Embeddings,
	generator: Generator,
	settings: Settings,
	*,
	force_rebuild: bool = False,
) -> RAGResponse:
	"""Build or load the configured index and answer one question."""
	vector_store = _get_vector_store(
		embedding_model, settings, force_rebuild=force_rebuild
	)
	return answer_question(
		vector_store, generator, query, k=settings.retrieval_k
	)


def _get_vector_store(
	embedding_model: Embeddings,
	settings: Settings,
	*,
	force_rebuild: bool,
) -> FAISS:
	if settings.index_dir.exists() and not force_rebuild:
		return load_vector_store(
			settings.index_dir,
			embedding_model,
			allow_dangerous_deserialization=True,
		)

	chunks = load_and_chunk_documents(settings.data_dir)
	vector_store = build_vector_store(chunks, embedding_model)
	vector_store.save_local(str(Path(settings.index_dir)))
	return vector_store