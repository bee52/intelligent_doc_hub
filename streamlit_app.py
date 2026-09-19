"""Minimal Streamlit UI for the Basic RAG application."""

from __future__ import annotations

import streamlit as st

from main_app import build_default_embedding_model, build_default_generator, run_basic_rag_query
from src.core.config import Settings


def main() -> None:
    st.title("Intelligent Document Analysis Hub")
    st.caption("Basic RAG demo with local default adapters")

    settings = Settings.from_env()
    st.write(f"Data directory: {settings.data_dir}")
    st.write(f"Index directory: {settings.index_dir}")

    query = st.text_area("Ask a question about your documents", placeholder="e.g. What is this document about?")

    if st.button("Search") and query.strip():
        with st.spinner("Retrieving and generating an answer..."):
            response = run_basic_rag_query(
                query,
                settings=settings,
                embedding_model=build_default_embedding_model(),
                generator=build_default_generator(),
                force_rebuild=False,
            )
        st.subheader("Answer")
        st.write(response.answer)

        if response.sources:
            st.subheader("Sources")
            for source in response.sources:
                metadata = source.metadata
                label = metadata.get("source", "unknown")
                page = metadata.get("page")
                if page is not None:
                    label = f"{label} (page {page})"
                st.markdown(f"- {label}")


if __name__ == "__main__":
    main()
