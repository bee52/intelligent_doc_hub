# SKILL: Evaluate RAG Pipeline

## Description
Use this skill when asked to evaluate the performance, latency, or retrieval accuracy of any RAG pipeline (Standard, Agentic, or Graph).

## Process Steps
1. Identify which RAG pipeline is being evaluated.
2. Load the evaluation dataset from `data/raw_documents/test_queries.json`.
3. Execute the pipeline using the test queries.
4. Measure and log the end-to-end latency.
5. Compare the retrieved context against the ground truth.
6. Output a summary markdown table comparing execution time and relevance score.
