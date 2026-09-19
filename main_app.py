"""Command-line and default app wiring for a Basic RAG application."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import Settings
from src.core.llm_client import CallableEmbeddings, CallableGenerator
from src.retrieval.rag import RAGResponse
from src.retrieval.standard_rag import run_basic_rag


def _stable_embedding_vector(text: str, index: int = 0) -> float:
	"""Create a deterministic low-dimensional embedding vector from text."""
	chars = [ord(character) for character in text[:64]]
	if not chars:
		return 1.0
	return float(sum(char + index + 1 for char in chars) % 17 + 1)


def _embed_text(text: str) -> list[float]:
	"""Map a text value to a stable 8-dimensional embedding vector."""
	return [
		_stable_embedding_vector(text, offset)
		for offset in range(8)
	]


def build_default_embedding_model() -> CallableEmbeddings:
	"""Return a deterministic embedding model for local testing and demos."""

	def embed_documents(texts: list[str]) -> list[list[float]]:
		return [_embed_text(text) for text in texts]

	def embed_query(text: str) -> list[float]:
		return _embed_text(text)

	return CallableEmbeddings(embed_documents, embed_query)


def _default_generate(prompt: str) -> str:
	"""Generate a simple grounded response from the prompt text."""
	question = prompt
	if "Question:" in prompt:
		question = prompt.split("Question:", 1)[1].split("\nAnswer:", 1)[0].strip()
	return (
		"Answer: This basic RAG app is running with local default adapters. "
		f"The question was: '{question}'. Add a hosted embedding or chat model "
		"in src/core/llm_client.py to replace this placeholder generator."
	)


def build_default_generator() -> CallableGenerator:
	"""Return a default generator used before a provider-backed model is configured."""
	return CallableGenerator(_default_generate)


def run_basic_rag_query(
	query: str,
	settings: Settings | None = None,
	embedding_model=None,
	generator=None,
	*,
	force_rebuild: bool = False,
) -> RAGResponse:
	"""Run the basic retrieval and generation flow with default local adapters."""
	resolved_settings = settings or Settings.from_env()
	resolved_embedding_model = embedding_model or build_default_embedding_model()
	resolved_generator = generator or build_default_generator()

	if not resolved_settings.data_dir.exists():
		resolved_settings.data_dir.mkdir(parents=True, exist_ok=True)
		return RAGResponse(
			answer=(
				"Answer: No PDF documents were found in the configured data directory. "
				f"Add PDFs under {resolved_settings.data_dir} and run again."
			),
			sources=[],
		)

	return run_basic_rag(
		query,
		resolved_embedding_model,
		resolved_generator,
		resolved_settings,
		force_rebuild=force_rebuild,
	)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Ask questions over indexed documents.")
	parser.add_argument("query", nargs="?", help="Question to answer")
	parser.add_argument(
		"--streamlit",
		action="store_true",
		help="Launch the local Streamlit interface instead of a CLI answer.",
	)
	return parser


def main() -> None:
	"""Run the app in CLI mode or bootstrap the Streamlit web UI."""
	args = build_parser().parse_args()
	if args.streamlit:
		from streamlit_app import main as streamlit_main

		streamlit_main()
		return
	if not args.query:
		raise SystemExit("Provide a query argument or run with --streamlit to open the web app.")
	response = run_basic_rag_query(args.query)
	print(response.answer)
	if response.sources:
		print("\nSources:")
		for source in response.sources:
			print(f"- {source.metadata.get('source', 'unknown')}")


if __name__ == "__main__":
	main()