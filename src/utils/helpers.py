from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from langchain_core.documents import Document

from src.chunking.chunker import chunk_documents
from src.llm.llm_client import generate_answer
from src.retrieval.retriever import QueryEmbedder, Retriever
from src.vectordb.vector_store import DatabaseType, VectorDBOperations


class DocumentEmbedder(QueryEmbedder, Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert document text into embeddings."""


def infer_document_type(source: str | None = None) -> str:
    """Infer a prompt type from a file path, URL, or source label."""
    if not source:
        return "generic"
    value = source.lower().split("?", maxsplit=1)[0]
    if value.startswith(("http://", "https://")):
        return "web"
    extension = value.rsplit(".", maxsplit=1)[-1] if "." in value else value
    aliases = {"html": "web", "htm": "web", "doc": "docx"}
    return aliases.get(extension, extension if extension in {
        "pdf", "docx", "csv", "json", "txt", "web", "database"
    } else "generic")


def _metadata_for_chroma(
    metadata: Mapping[str, Any],
    source_type: str,
) -> dict[str, str | int | float | bool]:
    """Keep only Chroma-compatible metadata values."""
    result: dict[str, str | int | float | bool] = {"source_type": source_type}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            result[str(key)] = value
    return result


def _document_id(document: Document, index: int) -> str:
    digest = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()[:16]
    source = str(document.metadata.get("source", "document"))
    return f"{source}:{index}:{digest}"


def documents_to_context(documents: Sequence[Document]) -> str:
    """Format retrieved documents for the grounded answer prompt."""
    if not documents:
        raise ValueError("At least one document is required for context")
    sections = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source") or document.metadata.get("id")
        label = f"Source {source}" if source else f"Context {index}"
        sections.append(f"[{label}]\n{document.page_content}")
    return "\n\n".join(sections)


class RAGPipeline:
    """Orchestrate indexing and question answering with injectable components."""

    def __init__(
        self,
        vector_store: VectorDBOperations,
        embeddings: DocumentEmbedder | None = None,
        collection_name: str = "documents",
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.vector_store.create_collection(collection_name)

    def _get_embeddings(self) -> DocumentEmbedder:
        if self.embeddings is None:
            from src.embeddings.embedder import get_embeddings

            self.embeddings = get_embeddings()
        return self.embeddings

    def index_documents(
        self,
        documents: Sequence[Document],
        *,
        source_type: str = "generic",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        """Chunk and upsert documents, returning IDs of indexed chunks."""
        if not documents:
            raise ValueError("At least one document is required")
        source_type = infer_document_type(source_type)
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise ValueError("Chunking produced no documents")

        embeddings = self._get_embeddings().embed_documents(
            [chunk.page_content for chunk in chunks]
        )
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")

        ids: list[str] = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            item_id = _document_id(chunk, index)
            metadata = _metadata_for_chroma(chunk.metadata, source_type)
            self.vector_store.insert_vector(
                item_id,
                embedding,
                metadata=metadata,
                document=chunk.page_content,
            )
            ids.append(item_id)
        return ids

    def retrieve(
        self,
        question: str,
        *,
        k: int = 4,
        where: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        """Retrieve relevant chunks for a question."""
        return Retriever(
            self.vector_store,
            embeddings=self._get_embeddings(),
        ).retrieve(question, k=k, where=where)

    def ask(
        self,
        question: str,
        *,
        document_type: str = "generic",
        k: int = 4,
        where: Mapping[str, Any] | None = None,
        **llm_options: Any,
    ) -> str:
        """Retrieve context and generate a grounded Groq answer."""
        documents = self.retrieve(question, k=k, where=where)
        context = documents_to_context(documents)
        return generate_answer(
            question=question,
            context=context,
            document_type=document_type,
            **llm_options,
        )


def build_rag_pipeline(
    collection_name: str = "documents",
    *,
    db_type: str = "persistent",
    db_path: str = "./chroma",
    embeddings: DocumentEmbedder | None = None,
) -> RAGPipeline:
    """Build the default local RAG pipeline."""
    vector_store = VectorDBOperations(db_type=cast(DatabaseType, db_type))
    vector_store.connect(path=db_path)
    return RAGPipeline(
        vector_store=vector_store,
        embeddings=embeddings,
        collection_name=collection_name,
    )


__all__ = [
    "RAGPipeline",
    "build_rag_pipeline",
    "documents_to_context",
    "infer_document_type",
]