from pathlib import Path

import pytest

from src.core.config import Settings
from src.core.llm_client import CallableEmbeddings, CallableGenerator
from src.retrieval.rag import RAGResponse
from src.retrieval.standard_rag import answer_question


def test_settings_use_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("RAG_RETRIEVAL_K", raising=False)

	settings = Settings.from_env()

	assert settings.data_dir == Path("data/raw_documents")
	assert settings.retrieval_k == 4


def test_settings_reject_invalid_positive_integer(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("RAG_RETRIEVAL_K", "0")

	with pytest.raises(ValueError, match="greater than zero"):
		Settings.from_env()


def test_callable_model_adapters() -> None:
	embeddings = CallableEmbeddings(
		lambda texts: [[float(len(text))] for text in texts],
		lambda text: [float(len(text))],
	)
	generator = CallableGenerator(lambda prompt: f"generated: {prompt[:8]}")

	assert embeddings.embed_documents(["abc"]) == [[3.0]]
	assert embeddings.embed_query("abcd") == [4.0]
	assert generator.invoke("question") == "generated: question"


def test_standard_rag_facade_delegates_to_generation() -> None:
	class FakeStore:
		def similarity_search(self, query: str, k: int) -> list:
			return []

	class FakeGenerator:
		def invoke(self, prompt: str) -> str:
			return "answer"

	response = answer_question(FakeStore(), FakeGenerator(), "question", k=1)

	assert response.answer == "answer"


def test_standard_rag_response_contract() -> None:
	assert RAGResponse(answer="answer", sources=[]).answer == "answer"