from langchain_core.documents import Document

from src.retrieval.agentic_rag import (
	EvaluationResult,
	RetrievalPlan,
	RetrievalResult,
	run_agentic_rag,
)


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