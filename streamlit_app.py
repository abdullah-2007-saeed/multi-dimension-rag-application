from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.embeddings import get_embeddings
from src.ingestion import fallback
from src.llm import generate_answer, get_llm
from src.utils import RAGPipeline, build_rag_pipeline, documents_to_context

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_TYPES = ["generic", "pdf", "docx", "csv", "json", "txt", "web"]
UPLOAD_TYPES = ["pdf", "docx", "csv", "json", "txt"]


@st.cache_resource(show_spinner=False)
def create_pipeline(collection_name: str, chroma_path: str) -> RAGPipeline:
    """Create one in-process RAG pipeline for the Streamlit session."""
    embeddings = get_embeddings()
    return build_rag_pipeline(
        collection_name=collection_name,
        db_type="persistent",
        db_path=chroma_path,
        embeddings=embeddings,
    )


def load_uploaded_file(uploaded_file: Any) -> list[Any]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {f".{extension}" for extension in UPLOAD_TYPES}:
        raise ValueError(f"Unsupported upload type: {suffix or '<none>'}")

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
            temporary_path = file.name
            file.write(uploaded_file.getvalue())
        return fallback(file_path=temporary_path)
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


def index_uploaded_file(
    pipeline: RAGPipeline,
    uploaded_file: Any,
    document_type: str,
) -> int:
    documents = load_uploaded_file(uploaded_file)
    ids = pipeline.index_documents(documents, source_type=document_type)
    return len(ids)


def index_url(pipeline: RAGPipeline, url: str) -> int:
    documents = fallback(url=url)
    return len(pipeline.index_documents(documents, source_type="web"))


def answer_question(
    pipeline: RAGPipeline,
    question: str,
    document_type: str,
    k: int,
    model: str,
) -> str:
    documents = pipeline.retrieve(question, k=k)
    context = documents_to_context(documents)
    llm = get_llm(
        model=model,
        temperature=0.1,
        max_tokens=1500,
    )
    return generate_answer(
        question=question,
        context=context,
        document_type=document_type,
        llm=llm,
    )


def main() -> None:
    st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖")
    st.title("Local RAG Chatbot")
    st.caption("Standalone Streamlit RAG with Chroma and Groq.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("RAG configuration")
        collection_name = st.text_input("Collection name", value="documents")
        chroma_path = st.text_input("Chroma path", value="./chroma")
        model = st.text_input(
            "Groq chat model",
            value=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        )
        document_type = st.selectbox("Document type", SUPPORTED_TYPES)
        k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=4)

        if st.button("Initialize RAG"):
            try:
                create_pipeline.clear()
                create_pipeline(collection_name, chroma_path)
                st.success("RAG pipeline initialized.")
            except Exception as error:
                st.error(f"RAG initialization failed: {type(error).__name__}: {error}")

        st.divider()
        st.header("Add documents")
        uploaded_file = st.file_uploader(
            "Upload PDF, DOCX, CSV, JSON, or TXT",
            type=UPLOAD_TYPES,
        )
        if st.button("Index uploaded file", disabled=uploaded_file is None):
            try:
                pipeline = create_pipeline(collection_name, chroma_path)
                with st.spinner("Loading, chunking, and embedding..."):
                    count = index_uploaded_file(
                        pipeline,
                        uploaded_file,
                        document_type,
                    )
                st.success(f"Indexed {count} chunks.")
            except Exception as error:
                st.error(f"Document indexing failed: {type(error).__name__}: {error}")

        url = st.text_input("Web page URL")
        if st.button("Index web page", disabled=not url.strip()):
            try:
                pipeline = create_pipeline(collection_name, chroma_path)
                with st.spinner("Loading, chunking, and embedding web page..."):
                    count = index_url(pipeline, url.strip())
                st.success(f"Indexed {count} chunks from the web page.")
            except Exception as error:
                st.error(f"URL indexing failed: {type(error).__name__}: {error}")

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        with st.expander("Runtime diagnostics"):
            st.write(f"Chroma path: `{chroma_path}`")
            st.write(
                "Groq API key: "
                f"`{'configured' if os.getenv('GROQ_API_KEY') else 'missing'}`"
            )
            st.write(f"Chat model: `{model}`")
            st.caption(
                "Required services: Groq API access, a valid API key, "
                "and a model available to your account."
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your indexed documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                pipeline = create_pipeline(collection_name, chroma_path)
                with st.spinner("Retrieving context and asking Groq..."):
                    answer = answer_question(
                        pipeline,
                        question,
                        document_type,
                        k,
                        model,
                    )
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as error:
                st.error(f"Question failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
