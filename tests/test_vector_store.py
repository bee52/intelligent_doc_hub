from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retrieval.rag import build_context, generate_answer, retrieve_documents
from src.retrieval.vector_store import build_vector_store


class Test_Embeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class FakeGenerator:
    def invoke(self, prompt: str) -> str:
        assert "Answer the question using only the supplied context." in prompt
        return "The answer is in the retrieved context."


def test_retrieve_documents_returns_ranked_documents() -> None:
    documents = [
        Document(page_content="short", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="a much longer document", metadata={"source": "b.pdf"}),
    ]
    store = build_vector_store(documents, Test_Embeddings())

    results = retrieve_documents(store, "short", k=1)

    assert len(results) == 1
    assert results[0].page_content == "short"


def test_build_context_includes_source_metadata() -> None:
    documents = [Document(page_content="facts", metadata={"source": "a.pdf", "page": 2})]

    context = build_context(documents)

    assert "a.pdf, page 2" in context
    assert "facts" in context


def test_generate_answer_returns_answer_and_sources() -> None:
    documents = [Document(page_content="facts", metadata={"source": "a.pdf"})]
    store = build_vector_store(documents, Test_Embeddings())

    response = generate_answer(store, FakeGenerator(), "facts", k=1)

    assert response.answer == "The answer is in the retrieved context."
    assert response.sources[0].page_content == documents[0].page_content
    assert response.sources[0].metadata == documents[0].metadata

    