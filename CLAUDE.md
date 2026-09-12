# Intelligent Document Analysis Hub

## WHY & WHAT
This project implements and compares standard RAG, Agentic RAG, and Graph RAG architectures. The stack is Python, LangChain, FAISS, and PyTorch. 

## CRITICAL RULES
* ALWAYS run `pytest` before declaring a task complete.
* NEVER commit actual PDF data to the repository; use `.gitignore`.
* NEVER use the AI as a linter. We use `ruff` for all formatting.
* NEVER hallucinate library imports. Stick to the constraints in `requirements.txt`.

## COMMANDS
* Test: `pytest tests/`
* Lint: `ruff check .`
