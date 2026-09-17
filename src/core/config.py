"""Environment-backed configuration for the RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
	"""Runtime settings shared by ingestion, retrieval, and generation."""

	data_dir: Path = Path("data/raw_documents")
	index_dir: Path = Path("data/processed_index")
	embedding_model: str = "configured-by-application"
	generation_model: str = "configured-by-application"
	retrieval_k: int = 4
	max_agent_rounds: int = 3

	@classmethod
	def from_env(cls) -> Settings:
		"""Build settings from environment variables, using local defaults."""
		settings = cls(
			data_dir=Path(os.getenv("RAG_DATA_DIR", cls.data_dir)),
			index_dir=Path(os.getenv("RAG_INDEX_DIR", cls.index_dir)),
			embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),
			generation_model=os.getenv("RAG_GENERATION_MODEL", cls.generation_model),
			retrieval_k=_read_positive_int("RAG_RETRIEVAL_K", cls.retrieval_k),
			max_agent_rounds=_read_positive_int(
				"RAG_MAX_AGENT_ROUNDS", cls.max_agent_rounds
			),
		)
		return settings


def _read_positive_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		parsed = int(value)
	except ValueError as error:
		raise ValueError(f"{name} must be an integer") from error
	if parsed <= 0:
		raise ValueError(f"{name} must be greater than zero")
	return parsed