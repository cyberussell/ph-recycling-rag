"""Retrieval over the indexed corpus.

M1: plain dense top-k (no fusion, no rerank). M2 adds a `sparse` named vector
and Reciprocal Rank Fusion here; M3 adds a reranking pass on top of this
module's output. Keeping the function name/shape (`search(query, top_k) ->
list[chunk]`) stable now means M2/M3 change this file's internals, not its
callers (agentic_loop.py, generate.py).
"""

from ..embedding.embedder import embed_query
from ..indexing.client import get_client
from ..indexing.qdrant_schema import COLLECTION_NAME, DENSE_VECTOR_NAME


def search(query: str, top_k: int = 5, collection_name: str = COLLECTION_NAME) -> list[dict]:
    client = get_client()
    query_vector = embed_query(query)
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=top_k,
    )
    chunks = []
    for point in results.points:
        chunk = dict(point.payload)
        chunk["score"] = point.score
        chunks.append(chunk)
    return chunks
