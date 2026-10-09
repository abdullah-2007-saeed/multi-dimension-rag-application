from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.documents import Document

from src.vectordb.vector_store import VectorDBOperations


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]:
        """Convert one query into an embedding."""


@dataclass(frozen=True)
class RetrievedDocument:
    """A document returned by vector search together with its distance."""

    document: Document
    distance: float | None


class Retriever:
    """Retrieve LangChain documents from a Chroma-backed vector store."""

    def __init__(
        self,
        vector_store: VectorDBOperations,
        embeddings: QueryEmbedder | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embeddings = embeddings

    def _get_embeddings(self) -> QueryEmbedder:
        if self.embeddings is None:
            from src.embeddings.embedder import get_embeddings

            self.embeddings = get_embeddings()
        return self.embeddings

    @staticmethod
    def _to_documents(result: Mapping[str, Any]) -> list[RetrievedDocument]:
        ids = result.get("ids", [[]])
        documents = result.get("documents", [[]])
        metadatas = result.get("metadatas", [[]])
        distances = result.get("distances", [[]])

        row_ids = ids[0] if ids else []
        row_documents = documents[0] if documents else []
        row_metadatas = metadatas[0] if metadatas else []
        row_distances = distances[0] if distances else []

        retrieved: list[RetrievedDocument] = []
        for index, item_id in enumerate(row_ids):
            text = row_documents[index] if index < len(row_documents) else None
            metadata = (
                dict(row_metadatas[index])
                if index < len(row_metadatas) and row_metadatas[index] is not None
                else {}
            )
            metadata["id"] = item_id
            distance = (
                float(row_distances[index])
                if index < len(row_distances) and row_distances[index] is not None
                else None
            )
            if text is not None:
                retrieved.append(
                    RetrievedDocument(
                        document=Document(page_content=text, metadata=metadata),
                        distance=distance,
                    )
                )
        return retrieved

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        where: Mapping[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Return up to ``k`` documents ranked by vector distance."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        query_embedding = self._get_embeddings().embed_query(query)
        result = self.vector_store.query_vectors(
            query_embedding=query_embedding,
            n_results=k,
            where=dict(where) if where is not None else None,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_documents(result)

    def retrieve(
        self,
        query: str,
        k: int = 4,
        where: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        """Return retrieved documents without exposing vector distances."""
        return [
            item.document
            for item in self.retrieve_with_scores(query=query, k=k, where=where)
        ]


def retrieve(
    query: str,
    vector_store: VectorDBOperations,
    embeddings: QueryEmbedder | None = None,
    k: int = 4,
    where: Mapping[str, Any] | None = None,
) -> list[Document]:
    """Convenience function for one-off retrieval."""
    return Retriever(vector_store, embeddings).retrieve(query, k=k, where=where)
