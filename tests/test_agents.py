import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retrieval.agentic_rag import (
	EvaluationResult,
	RetrievalPlan,
	RetrievalResult,
	run_agentic_rag,
)
from src.tools.vector_search import VectorStoreTool


class FakeTool:
	name = "vector"

	def __init__(self) -> None:
		self.queries: list[str] = []

	def search(self, query: str) -> list[Document]:
		self.queries.append(query)
		return [Document(page_content=f"evidence for {query}", metadata={"source": query})]


class FixedPlanner:
	def __init__(self, plans: list[RetrievalPlan]) -> None:
		self.plans = plans

	def plan(self, query: str, previous_results: list[RetrievalResult]) -> RetrievalPlan:
		return self.plans.pop(0)


class RetryEvaluator:
	def __init__(self) -> None:
		self.calls = 0

	def evaluate(
		self, query: str, results: list[RetrievalResult]
	) -> EvaluationResult:
		self.calls += 1
		if self.calls == 1:
			return EvaluationResult(
				accepted=False,
				follow_up=RetrievalPlan(queries=["follow-up"]),
			)
		return EvaluationResult(accepted=True)


class FakeGenerator:
	def invoke(self, prompt: str) -> str:
		return f"answer with context: {prompt.split('Context:', 1)[1][:20]}"


def test_agent_retries_with_follow_up_and_preserves_sources() -> None:
	tool = FakeTool()
	response = run_agentic_rag(
		"question",
		FixedPlanner([RetrievalPlan(["initial"])]),
		RetryEvaluator(),
		{"vector": tool},
		FakeGenerator(),
		max_rounds=2,
	)

	assert tool.queries == ["initial", "follow-up"]
	assert response.rounds == 2
	assert len(response.sources) == 2


def test_agent_respects_round_limit() -> None:
	tool = FakeTool()
	response = run_agentic_rag(
		"question",
		FixedPlanner([RetrievalPlan(["initial"])]),
		RetryEvaluator(),
		{"vector": tool},
		FakeGenerator(),
		max_rounds=1,
	)

	assert response.rounds == 1


def test_vector_store_tool_returns_documents_and_validates_queries() -> None:
	class FakeEmbeddings(Embeddings):
		def embed_documents(self, texts: list[str]) -> list[list[float]]:
			return [[float(len(text))] for text in texts]

		def embed_query(self, text: str) -> list[float]:
			return [float(len(text))]

	store = FAISS.from_documents(
		[Document(page_content="agentic evidence", metadata={"source": "fixture"})],
		FakeEmbeddings(),
	)
	tool = VectorStoreTool(store, default_k=1)

	results = tool.search("agentic")

	assert tool.name == "vector"
	assert results[0].metadata["source"] == "fixture"

	with pytest.raises(ValueError, match="query must not be empty"):
		tool.search(" ")

	with pytest.raises(ValueError, match="default_k must be greater than zero"):
		VectorStoreTool(store, default_k=0)


def test_retrieval_result_records_tool_and_preserves_document_metadata() -> None:
	document = Document(page_content="evidence", metadata={"source": "fixture", "page": 2})
	result = RetrievalResult(documents=[document], tool="vector")

	assert result.tool == "vector"
	assert result.documents[0].metadata == {"source": "fixture", "page": 2}