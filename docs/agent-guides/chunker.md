# Chunker

The reusable chunker lives in `src/ingestion/chunker.py`.

## Command line

```powershell
python -m src.ingestion.chunker --input data/input.txt --chunk-size 500 --overlap 50
```

Use `--strategy by_lines` to chunk by newline-delimited units. Use `--output-dir chunks` to write each chunk to a separate text file instead of printing to standard output.

## Python API

```python
from src.ingestion.chunker import chunk, iter_chunks

chunks = chunk(text, chunk_size=500, overlap=50)
for item in iter_chunks(text, chunk_size=500, strategy="by_lines"):
    print(item, item.metadata.index)
```

Each result is string-compatible and exposes `metadata.index`, `metadata.start`, and `metadata.end`.

## Embeddings

Pass the LangChain embedding model explicitly so the chunker stays independent of a provider:

```python
from src.ingestion.chunker import embed_chunks, load_and_chunk_documents

chunks = load_and_chunk_documents("data/raw_documents")
vectors = embed_chunks(chunks, embedding_model)
```

`vectors` matches `chunks` by position and can be inserted into a vector store.

## Vector store

Build a persistent FAISS index directly from the LangChain chunks:

```python
from src.retrieval.vector_store import (
    build_vector_store,
    load_vector_store,
    save_vector_store,
)

store = build_vector_store(chunks, embedding_model)
save_vector_store(store, "data/processed_index")

store = load_vector_store("data/processed_index", embedding_model)
matches = store.similarity_search("search text", k=4)
```

Loading persisted indexes requires `allow_dangerous_deserialization=True` only when the index files are trusted.

## Retrieval and generation

Use the RAG orchestration layer to retrieve relevant chunks and pass their source-attributed context to a generation model:

```python
from src.retrieval.rag import generate_answer

response = generate_answer(store, generator, "What does the document explain?", k=4)
print(response.answer)
for source in response.sources:
    print(source.metadata)
```

`generator` must provide an `invoke(prompt)` method, which keeps the retrieval layer independent of a specific model provider.
