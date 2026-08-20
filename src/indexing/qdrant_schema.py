"""Qdrant collection schema for the PH recycling/waste-management corpus.

M1 uses a single named dense vector. The name is kept (rather than using an
unnamed default vector) so M2 can add a `sparse` named vector to the same
collection for hybrid search without a schema migration.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

COLLECTION_NAME = "ph_recycling_law"
DENSE_VECTOR_NAME = "dense"


def ensure_collection(client: QdrantClient, vector_size: int, collection_name: str = COLLECTION_NAME) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        },
    )


def recreate_collection(client: QdrantClient, vector_size: int, collection_name: str = COLLECTION_NAME) -> None:
    client.delete_collection(collection_name)
    ensure_collection(client, vector_size, collection_name)
