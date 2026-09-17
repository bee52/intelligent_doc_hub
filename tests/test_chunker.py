from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.ingestion.chunker import chunk, embed_chunks
from src.retrieval.vector_store import build_vector_store


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def test_chunk_without_overlap() -> None:
    assert chunk("abcdefgh", 3) == ["abc", "def", "gh"]


def test_chunk_with_overlap() -> None:
    assert chunk("abcdef", 3, overlap=1) == ["abc", "cde", "ef"]


def test_chunk_smaller_than_chunk_size() -> None:
    result = chunk("short", 20)

    assert result == ["short"]
    assert result[0].metadata.index == 0
    assert result[0].metadata.start == 0
    assert result[0].metadata.end == 5


def test_chunk_by_lines() -> None:
    assert chunk("one\ntwo\nthree\n", 2, strategy="by_lines") == [
        "one\ntwo\n",
        "three\n",
    ]


def test_embed_chunks_passes_document_texts_to_model() -> None:
    chunks = [
        Document(page_content="first", metadata={"page": 0}),
        Document(page_content="second chunk", metadata={"page": 1}),
    ]

    assert embed_chunks(chunks, FakeEmbeddings()) == [[5.0], [12.0]]


def test_embed_chunks_returns_empty_list_for_no_chunks() -> None:
    assert embed_chunks([], FakeEmbeddings()) == []


def test_build_vector_store_keeps_chunk_metadata() -> None:
    chunks = [
        Document(page_content="short", metadata={"page": 2}),
        Document(page_content="a longer chunk", metadata={"page": 3}),
    ]

    vector_store = build_vector_store(chunks, FakeEmbeddings())
    results = vector_store.similarity_search("short", k=1)

    assert results[0].page_content == "short"
    assert results[0].metadata == {"page": 2}