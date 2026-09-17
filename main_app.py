"""Command-line entry point for a configured Basic RAG application."""

from __future__ import annotations

import argparse

from src.core.config import Settings


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Ask questions over indexed documents.")
	parser.add_argument("query", help="Question to answer")
	return parser


def main() -> None:
	"""Parse the query and display the required provider wiring."""
	args = build_parser().parse_args()
	settings = Settings.from_env()
	print(f"Query: {args.query}")
	print(
		"Inject an embedding model and generator into "
		f"run_basic_rag; configured index: {settings.index_dir}"
	)


if __name__ == "__main__":
	main()