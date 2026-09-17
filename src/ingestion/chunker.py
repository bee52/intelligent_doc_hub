"""Utilities for loading documents and splitting them into chunks."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pypdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_and_chunk_documents(
	data_dir: str | os.PathLike[str],
	chunk_size: int = 500,
	chunk_overlap: int = 50,
) -> list[Document]:
	"""Load PDF pages from *data_dir* and split them into LangChain documents."""
	if chunk_size <= 0:
		raise ValueError("chunk_size must be greater than zero")
	if chunk_overlap < 0 or chunk_overlap >= chunk_size:
		raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

	documents: list[Document] = []
	for filename in sorted(os.listdir(data_dir)):
		if not filename.lower().endswith(".pdf"):
			continue

		file_path = Path(data_dir) / filename
		with file_path.open("rb") as file:
			reader = pypdf.PdfReader(file)
			for page_number, page in enumerate(reader.pages):
				text = page.extract_text()
				if text:
					documents.append(
						Document(
							page_content=text,
							metadata={"source": str(file_path), "page": page_number},
						)
					)

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		separators=["\n\n", "\n", " ", ""],
	)
	chunked_documents = text_splitter.split_documents(documents)
	logger.info(
		"Processed %d pages into %d chunks", len(documents), len(chunked_documents)
	)
	return chunked_documents


def embed_chunks(
	chunks: Sequence[Document], embedding_model: Embeddings
) -> list[list[float]]:
	"""Embed chunk contents and return vectors in the same order as *chunks*."""
	texts = [document.page_content for document in chunks]
	if not texts:
		return []

	embeddings = embedding_model.embed_documents(texts)
	if len(embeddings) != len(chunks):
		raise ValueError("embedding model returned a different number of vectors")

	logger.info("Embedded %d chunks", len(embeddings))
	return embeddings


@dataclass(frozen=True)
class ChunkMetadata:
	"""Location metadata attached to a generated chunk."""

	index: int
	start: int
	end: int


class Chunk(str):
	"""A string chunk that also exposes its source offsets."""

	metadata: ChunkMetadata

	def __new__(
		cls, value: str, index: int, start: int, end: int
	) -> Self:
		chunk = super().__new__(cls, value)
		chunk.metadata = ChunkMetadata(index=index, start=start, end=end)
		return chunk


def _validate_parameters(chunk_size: int, overlap: int) -> None:
	if chunk_size <= 0:
		raise ValueError("chunk_size must be greater than zero")
	if overlap < 0 or overlap >= chunk_size:
		raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")


def _source_units(data: str | Sequence[str], strategy: str) -> tuple[list[str], list[int]]:
	if isinstance(data, str):
		if strategy == "by_chars":
			return list(data), list(range(len(data)))
		if strategy == "by_lines":
			units = data.splitlines(keepends=True)
			starts: list[int] = []
			offset = 0
			for unit in units:
				starts.append(offset)
				offset += len(unit)
			return units, starts
	elif strategy in {"by_chars", "by_lines"}:
		units = list(data)
		return units, list(range(len(units)))

	raise ValueError("strategy must be 'by_chars' or 'by_lines'")


def iter_chunks(
	data: str | Sequence[str],
	chunk_size: int,
	overlap: int = 0,
	strategy: str = "by_chars",
) -> Iterator[Chunk]:
	"""Yield fixed-size chunks lazily, optionally overlapping adjacent chunks."""
	_validate_parameters(chunk_size, overlap)
	units, starts = _source_units(data, strategy)
	step = chunk_size - overlap

	logger.info(
		"Starting chunking: strategy=%s, chunk_size=%d, overlap=%d",
		strategy,
		chunk_size,
		overlap,
	)
	index = 0
	for position in range(0, len(units), step):
		selected = units[position : position + chunk_size]
		if not selected:
			break
		start = starts[position]
		last_unit = position + len(selected) - 1
		end = starts[last_unit] + len(selected[-1])
		value = "".join(selected)
		yield Chunk(value, index=index, start=start, end=end)
		index += 1
	logger.info("Produced %d chunks", index)


def chunk(
	data: str | Sequence[str],
	chunk_size: int,
	overlap: int = 0,
	strategy: str = "by_chars",
) -> list[Chunk]:
	"""Return all chunks for *data* using the selected chunking strategy."""
	return list(iter_chunks(data, chunk_size, overlap, strategy))


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Split a text file into chunks.")
	parser.add_argument("--input", required=True, type=Path, help="Input text file")
	parser.add_argument("--chunk-size", required=True, type=int)
	parser.add_argument("--overlap", default=0, type=int)
	parser.add_argument("--strategy", choices=("by_chars", "by_lines"), default="by_chars")
	parser.add_argument("--output-dir", type=Path, help="Directory for individual chunk files")
	return parser


def main() -> None:
	"""Run the command-line chunker."""
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	args = _build_parser().parse_args()
	try:
		chunks = chunk(
			args.input.read_text(encoding="utf-8"),
			args.chunk_size,
			args.overlap,
			args.strategy,
		)
	except (OSError, ValueError) as error:
		logger.error("Chunking failed: %s", error)
		raise SystemExit(2) from error

	if args.output_dir:
		args.output_dir.mkdir(parents=True, exist_ok=True)
		for generated_chunk in chunks:
			path = args.output_dir / f"chunk_{generated_chunk.metadata.index:04d}.txt"
			path.write_text(str(generated_chunk), encoding="utf-8")
	else:
		for generated_chunk in chunks:
			print(f"--- Chunk {generated_chunk.metadata.index} ---")
			print(generated_chunk, end="" if str(generated_chunk).endswith("\n") else "\n")


if __name__ == "__main__":
	main()