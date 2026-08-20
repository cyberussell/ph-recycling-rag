"""Qdrant collection schema for the PH recycling/waste-management corpus.

M2: named dense vector (BGE-M3, 1024-dim, cosine) + a named sparse vector
(BGE-M3 lexical weights) on the same collection, enabling hybrid search via
client-side Reciprocal Rank Fusion (see retrieval/hybrid_search.py).
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

COLLECTION_NAME = "ph_recycling_law"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def ensure_collection(client: QdrantClient, vector_size: int, collection_name: str = COLLECTION_NAME) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(),
        },
    )


def recreate_collection(client: QdrantClient, vector_size: int, collection_name: str = COLLECTION_NAME) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    ensure_collection(client, vector_size, collection_name)
