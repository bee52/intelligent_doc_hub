"""Bounded planning, retrieval, evaluation, and generation for Agentic RAG."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.documents import Document

from src.retrieval.rag import Generator, build_context


@dataclass(frozen=True)
class RetrievalPlan:
	"""A planner's retrieval instructions for one agent round."""

	queries: list[str]
	tools: list[str] = field(default_factory=lambda: ["vector"])


@dataclass(frozen=True)
class RetrievalResult:
	"""Retrieved evidence with the tool that produced it."""

	documents: list[Document]
	tool: str


@dataclass(frozen=True)
class EvaluationResult:
	"""Evaluator decision after reviewing retrieved evidence."""

	accepted: bool
	follow_up: RetrievalPlan | None = None


@dataclass(frozen=True)
class AgentResponse:
	"""Final answer plus evidence and execution diagnostics."""

	answer: str
	sources: list[Document]
	rounds: int
	plans: list[RetrievalPlan]


class Planner(Protocol):
	def plan(self, query: str, previous_results: Sequence[RetrievalResult]) -> RetrievalPlan:
		...


class Evaluator(Protocol):
	def evaluate(
		self, query: str, results: Sequence[RetrievalResult]
	) -> EvaluationResult:
		...


class RetrievalTool(Protocol):
	name: str

	def search(self, query: str) -> list[Document]:
		...


def run_agentic_rag(
	query: str,
	planner: Planner,
	evaluator: Evaluator,
	tools: dict[str, RetrievalTool],
	generator: Generator,
	*,
	max_rounds: int = 3,
) -> AgentResponse:
	"""Run the agent loop with a hard upper bound on retrieval rounds."""
	if not query.strip():
		raise ValueError("query must not be empty")
	if max_rounds <= 0:
		raise ValueError("max_rounds must be greater than zero")

	all_results: list[RetrievalResult] = []
	plans: list[RetrievalPlan] = []
	previous_results: list[RetrievalResult] = []

	for round_number in range(1, max_rounds + 1):
		plan = planner.plan(query, previous_results)
		plans.append(plan)
		round_results: list[RetrievalResult] = []
		for tool_name in plan.tools:
			if tool_name not in tools:
				raise ValueError(f"no retrieval tool configured for '{tool_name}'")
			for sub_query in plan.queries:
				round_results.append(
					RetrievalResult(documents=tools[tool_name].search(sub_query), tool=tool_name)
				)

		all_results.extend(round_results)
		previous_results = round_results
		decision = evaluator.evaluate(query, all_results)
		if decision.accepted:
			break
		if decision.follow_up is None and round_number < max_rounds:
			raise ValueError("evaluator rejected context without a follow-up plan")
		if decision.follow_up is not None:
			planner = _FollowUpPlanner(decision.follow_up)

	documents = _unique_documents(all_results)
	context = build_context(documents)
	prompt = (
		"Answer using only the retrieved context. State that you do not know "
		"when the context is insufficient.\n\n"
		f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
	)
	answer = generator.invoke(prompt)
	answer = getattr(answer, "content", answer)
	return AgentResponse(
		answer=str(answer), sources=documents, rounds=len(plans), plans=plans
	)


class _FollowUpPlanner:
	def __init__(self, plan: RetrievalPlan) -> None:
		self._plan = plan

	def plan(
		self, query: str, previous_results: Sequence[RetrievalResult]
	) -> RetrievalPlan:
		return self._plan


def _unique_documents(results: Sequence[RetrievalResult]) -> list[Document]:
	seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
	unique: list[Document] = []
	for result in results:
		for document in result.documents:
			key = (
				document.page_content,
				tuple(sorted((str(key), str(value)) for key, value in document.metadata.items())),
			)
			if key not in seen:
				seen.add(key)
				unique.append(document)
	return unique