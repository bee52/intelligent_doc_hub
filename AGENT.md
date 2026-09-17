# Intelligent Document Analysis Hub - Implementation Guide

## Purpose

Build and compare three document question-answering architectures:

1. Basic RAG: one embedding search followed by context augmentation and generation.
2. Agentic RAG: planning, iterative retrieval, evaluation, and generation.
3. Graph RAG: entity and relationship extraction, graph traversal, community summaries, and synthesis.

The architecture shown in `structure-diagram.txt` is the target design. Implement the paths incrementally and keep the three approaches independently testable.

## Repository Rules

- Keep PDF inputs out of version control. Store them under `data/raw_documents/` and keep that path ignored.
- Use the dependencies declared in `requirements.txt`; do not invent imports.
- Run `pytest tests/` before declaring a change complete.
- Run `ruff check .` after Python changes.
- Inject embedding and generation models instead of hard-coding provider credentials into retrieval code.
- Preserve source metadata through chunking, vector storage, graph extraction, and final responses.
- Prefer small interfaces and deterministic fakes in unit tests so tests do not require network access or API keys.

## Current Progress

### Implemented

- `src/ingestion/chunker.py`
  - Loads PDF pages with `pypdf`.
  - Splits pages into LangChain `Document` chunks.
  - Supports the existing string chunking API and CLI.
  - Embeds chunk text through an injected LangChain `Embeddings` object.
- `src/retrieval/vector_store.py`
  - Builds a FAISS store from LangChain documents.
  - Saves and loads local FAISS indexes.
- `src/retrieval/rag.py`
  - Retrieves documents from FAISS.
  - Formats source-attributed context.
  - Invokes an injected generator and returns `RAGResponse(answer, sources)`.
- Tests cover chunking, embedding calls, FAISS retrieval, metadata preservation, and basic generation orchestration.
- `src/core/config.py` provides environment-backed runtime settings.
- `src/core/llm_client.py` provides provider-neutral callable adapters.
- `src/retrieval/standard_rag.py` provides named Basic RAG and build/load index orchestration.
- `src/retrieval/agentic_rag.py` provides typed plans, retrieval/evaluation protocols, provenance, and bounded retries.
- `main_app.py` provides the CLI configuration entry point; concrete model providers remain injected by the application.
- Agent tests cover follow-up retrieval, source preservation, and round limits.

### Not Yet Implemented

The following files are described by the target structure but do not currently exist in the workspace:

- `src/ingestion/graph_extractor.py`
- `src/retrieval/graph_rag.py`
- `src/tools/web_search.py`
- `src/tools/db_queries.py`

The Basic RAG and core Agentic RAG slices are implemented, but concrete provider wiring and standalone web/database tools remain.

## Shared Contracts

### Documents

Use LangChain `Document` objects throughout the pipeline:

- `page_content`: text used for embedding and generation.
- `metadata.source`: input file path or stable document identifier.
- `metadata.page`: source page number when applicable.
- Additional metadata may include chunk index, section, entity, or graph identifiers.

### Embeddings

Embedding providers must be injected through LangChain's `Embeddings` interface. The ingestion layer should not know which provider is used.

### Generation

Generation providers should expose an `invoke(prompt)` operation, either directly or through a small adapter. The provider adapter belongs in `src/core/llm_client.py`, not inside retrieval algorithms.

### Responses

Every RAG path should return:

- `answer`: generated text.
- `sources`: source documents or source records used to produce the answer.
- Optional diagnostics such as retrieved documents, scores, plan steps, or graph entities.

## Implementation Order

### Phase 1: Complete Basic RAG

Owner modules:

- `src/core/config.py`
- `src/core/llm_client.py`
- `src/retrieval/standard_rag.py`
- `main_app.py`

Tasks:

1. Define environment-backed configuration for input directory, FAISS index directory, embedding model settings, generation model settings, and retrieval `k`.
2. Add model factories or adapters that create the configured embedding and generation models without leaking provider details into retrieval code.
3. Move or wrap the current orchestration in `standard_rag.py` so the basic path has a named public entry point.
4. Add an application flow that loads or builds the FAISS index, accepts a query, and prints the answer with sources.
5. Add tests for configuration defaults, missing required settings, model injection, prompt grounding, and source output.

Completion gate:

- A local or fake end-to-end flow can load documents, build/load an index, retrieve context, generate an answer, and return sources.
- No API key is needed for unit tests.

### Phase 2: Agentic RAG

Owner modules:

- `src/retrieval/agentic_rag.py`
- `src/tools/web_search.py`
- `src/tools/db_queries.py`
- `tests/test_agents.py`

Target flow from the diagram:

1. Accept the user query.
2. Planning agent decides whether retrieval is needed and creates sub-queries or tool selections.
3. Retrieval tools query the vector database, web search, or database tools.
4. Evaluator agent scores retrieved context and decides whether another retrieval pass is needed.
5. Context augmentation combines the accepted evidence.
6. Generator produces the final answer and cites the evidence used.

Tasks:

1. Define typed plan, retrieval result, evaluation result, and agent response objects.
2. Define tool protocols so vector search, web search, and database queries can be replaced by fakes.
3. Implement a bounded loop with a maximum number of retrieval rounds.
4. Require the evaluator to either accept context or return a concrete follow-up action.
5. Preserve provenance for every result, including tool name and source metadata.
6. Test planning decisions, tool routing, evaluator retry behavior, maximum-round termination, and final source reporting.

Completion gate:

- The agent can complete a direct retrieval query.
- The agent can request a follow-up retrieval when context is insufficient.
- The agent cannot loop indefinitely.
- Tests run entirely with deterministic fake tools and a fake generator.

### Phase 3: Graph RAG

Owner modules:

- `src/ingestion/graph_extractor.py`
- `src/retrieval/graph_rag.py`
- `data/processed_graphs/`

Target flow from the diagram:

1. Classify the user query as specific, broad, or global.
2. Retrieve relevant chunks and extract entities, relationships, and source references.
3. Store or update a graph with stable entity and relationship identifiers.
4. Traverse the graph from matched entities to collect linked context.
5. Build community reports or batch summaries for broader questions.
6. Map or rank report evidence, filter low-rated points, and synthesize the final answer.

Tasks:

1. Choose and document the graph representation and serialization format before implementation.
2. Define entity, relationship, evidence, and community-report schemas.
3. Extract graph facts from chunks while retaining source and page provenance.
4. Implement query classification with deterministic rules first; add an LLM classifier only behind an interface.
5. Implement local entity lookup and bounded graph traversal.
6. Add batch/community summarization only after entity lookup and traversal are tested.
7. Return graph evidence and source documents with every generated answer.

Completion gate:

- A small fixture corpus produces a reproducible graph.
- Entity lookup and traversal return the expected linked evidence.
- Specific, broad, and global query modes are separately tested.
- Graph artifacts are written under `data/processed_graphs/`, not committed as opaque generated data unless intentionally versioned.

## Testing Strategy

### Unit tests

Use deterministic fake implementations for:

- `Embeddings`
- Generator or chat model
- Vector search
- Planner and evaluator agents
- Web and database tools
- Graph extractor and graph store

Test behavior and contracts rather than provider internals.

### Integration tests

Add small fixture-based tests that exercise:

- PDF fixture -> chunks -> embeddings -> FAISS -> retrieval.
- Basic RAG query -> generated answer -> sources.
- Agentic retry path with a fake evaluator.
- Graph extraction -> graph persistence -> traversal -> synthesis.

Do not require external services in the default test command.

### Required checks

```powershell
.\venv\Scripts\python.exe -m pytest tests/
.\venv\Scripts\python.exe -m ruff check .
```

## Design Decisions To Resolve Before Implementation

- Which concrete embedding model should be used outside tests?
- Which generation provider should `src/core/llm_client.py` support first?
- Should the graph use NetworkX, a serialized JSON graph, or another graph store?
- What web search and database sources are required for Agentic RAG?
- What citation format should be shown to users?
- Should vector indexes and graph artifacts be rebuilt automatically or loaded only when present?
- What privacy and data-retention rules apply to raw PDFs, prompts, and generated answers?

Resolve these decisions in configuration or documentation before binding them into application code.

## Working Pattern For Future Changes

1. Identify the owning module and one behavior to implement.
2. Add or update a deterministic test for that behavior.
3. Implement the smallest interface-compatible change.
4. Run the focused test.
5. Run the full test suite and Ruff.
6. Update this guide when a phase or contract changes.
