from main_app import build_default_embedding_model, build_default_generator


def test_default_app_wiring_uses_callable_adapters() -> None:
    embeddings = build_default_embedding_model()
    assert hasattr(embeddings, "embed_query")
    assert hasattr(embeddings, "embed_documents")
    assert len(embeddings.embed_query("hello world")) > 0
    assert len(embeddings.embed_documents(["hello", "world"])) == 2

    generator = build_default_generator()
    result = generator.invoke("Question: what is RAG?")
    assert isinstance(result, str)
    assert "Answer:" in result
