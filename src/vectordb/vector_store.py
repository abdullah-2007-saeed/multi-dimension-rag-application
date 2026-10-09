from collections.abc import Mapping, Sequence
from typing import Any, Literal

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Include, Where

DatabaseType = Literal["persistent", "cloud", "ephemeral"]
Vector = Sequence[float]
Metadata = Mapping[str, str | int | float | bool]


def making_db_conn(
    db_type: DatabaseType = "ephemeral",
    path: str = "./chroma",
    api_key: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
) -> Any:
    """Create a Chroma client for the selected deployment type."""
    if db_type == "persistent":
        return chromadb.PersistentClient(path=path)
    if db_type == "cloud":
        missing = [
            name
            for name, value in (
                ("api_key", api_key),
                ("tenant", tenant),
                ("database", database),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Cloud Chroma requires: {', '.join(missing)}"
            )
        return chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )
    if db_type == "ephemeral":
        return chromadb.EphemeralClient()
    raise ValueError(f"Unsupported database type: {db_type}")


class VectorDBOperations:
    """Small, explicit wrapper around a Chroma collection."""

    def __init__(
        self,
        client: Any | None = None,
        db_type: DatabaseType = "ephemeral",
    ) -> None:
        self.db_type = db_type
        self.client = client
        self.collection: Collection | None = None

    def connect(
        self,
        path: str = "./chroma",
        api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
    ) -> Any:
        self.client = making_db_conn(
            db_type=self.db_type,
            path=path,
            api_key=api_key,
            tenant=tenant,
            database=database,
        )
        return self.client

    def create_collection(self, collection_name: str) -> Collection:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if self.client is None:
            self.connect()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        return self.collection

    def _require_collection(self) -> Collection:
        if self.collection is None:
            raise RuntimeError(
                "No collection is selected. Call create_collection first."
            )
        return self.collection

    @staticmethod
    def _normalise_id(vector_id: str | int) -> str:
        value = str(vector_id)
        if not value:
            raise ValueError("vector_id must not be empty")
        return value

    def insert_vector(
        self,
        vector_id: str | int,
        vector_data: Vector,
        metadata: Metadata | None = None,
        document: str | None = None,
    ) -> dict[str, Any]:
        """Insert or replace one vector and return the stored record."""
        item_id = self._normalise_id(vector_id)
        collection = self._require_collection()
        kwargs: dict[str, Any] = {
            "ids": [item_id],
            "embeddings": [list(vector_data)],
        }
        if metadata is not None:
            kwargs["metadatas"] = [dict(metadata)]
        if document is not None:
            kwargs["documents"] = [document]
        collection.upsert(**kwargs)
        return collection.get(ids=[item_id])

    def update_vector(
        self,
        vector_id: str | int,
        new_vector_data: Vector | None = None,
        metadata: Metadata | None = None,
        document: str | None = None,
    ) -> dict[str, Any]:
        """Update fields on an existing vector."""
        item_id = self._normalise_id(vector_id)
        collection = self._require_collection()
        kwargs: dict[str, Any] = {"ids": [item_id]}
        if new_vector_data is not None:
            kwargs["embeddings"] = [list(new_vector_data)]
        if metadata is not None:
            kwargs["metadatas"] = [dict(metadata)]
        if document is not None:
            kwargs["documents"] = [document]
        if len(kwargs) == 1:
            raise ValueError("At least one field must be supplied for update")
        collection.update(**kwargs)
        return collection.get(ids=[item_id])

    def delete_vector(self, vector_id: str | int) -> str:
        """Delete one vector and return its normalised id."""
        item_id = self._normalise_id(vector_id)
        self._require_collection().delete(ids=[item_id])
        return item_id

    def get_vectors(
        self,
        ids: Sequence[str | int] | None = None,
        where: Where | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Read vectors and their documents/metadata from the collection."""
        normalised_ids = [str(item_id) for item_id in ids] if ids is not None else None
        return self._require_collection().get(
            ids=normalised_ids,
            where=where,
            limit=limit,
        )

    def query_vectors(
        self,
        query_embedding: Vector,
        n_results: int = 4,
        where: Where | None = None,
        include: list[Include] | None = None,
    ) -> dict[str, Any]:
        """Find nearest vectors for one embedding."""
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [list(query_embedding)],
            "n_results": n_results,
            "where": where,
        }
        if include is not None:
            query_kwargs["include"] = include
        return self._require_collection().query(**query_kwargs)


# Backwards-compatible name used by the original implementation.
vectordb_operations = VectorDBOperations
