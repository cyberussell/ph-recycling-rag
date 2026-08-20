"""Hybrid retrieval: dense + sparse search fused with Reciprocal Rank Fusion,
then reranked with a cross-encoder.

M1 was dense-only top-k. M2 added a sparse (lexical-weight) query against the
same collection, fused client-side with RRF rather than a black-box fusion
call — the formula is one line and is worth being able to point to directly.
M3 adds `rerank.py`'s cross-encoder pass on top of the fused candidates: RRF
can be fooled when one leg (sparse, for cross-lingual queries) degrades to
noise, since it scores retrieval rank rather than actual relevance — a
cross-encoder re-scores true query-chunk relevance and recovers from that.
"""

from qdrant_client.http import models as qm

from ..embedding.embedder import embed_query
from ..indexing.client import get_client
from ..indexing.qdrant_schema import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from .rerank import rerank

RRF_K = 60  # standard damping constant from the original RRF paper
FUSION_CANDIDATES = 20  # how many candidates each leg contributes before fusion


def _dense_search(client, collection_name: str, dense_vector: list[float], limit: int):
    return client.query_points(
        collection_name=collection_name,
        query=dense_vector,
        using=DENSE_VECTOR_NAME,
        limit=limit,
    ).points


def _sparse_search(client, collection_name: str, sparse: dict, limit: int):
    return client.query_points(
        collection_name=collection_name,
        query=qm.SparseVector(indices=sparse["indices"], values=sparse["values"]),
        using=SPARSE_VECTOR_NAME,
        limit=limit,
    ).points


def _reciprocal_rank_fusion(ranked_lists: list[list], k: int = RRF_K) -> list[tuple[str, float]]:
    """ranked_lists: list of [point, ...] each already sorted best-first.
    Returns [(point_id, fused_score), ...] sorted best-first.

    RRF score for a document = sum over each list it appears in of 1 / (k + rank),
    rank starting at 1. Documents absent from a list simply don't contribute
    that term — no need to normalize each list's raw similarity scores onto a
    common scale, which is RRF's main appeal over score-based fusion.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, point in enumerate(ranked, start=1):
            scores[point.id] = scores.get(point.id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _fuse(query: str, collection_name: str) -> list[dict]:
    """Dense + sparse search fused via RRF. Returns up to FUSION_CANDIDATES
    chunks, best-first, each carrying its RRF `score`."""
    client = get_client()
    embedding = embed_query(query)

    dense_results = _dense_search(client, collection_name, embedding["dense"], FUSION_CANDIDATES)
    sparse_results = _sparse_search(client, collection_name, embedding["sparse"], FUSION_CANDIDATES)

    points_by_id = {p.id: p for p in dense_results + sparse_results}
    fused = _reciprocal_rank_fusion([dense_results, sparse_results])

    chunks = []
    for point_id, rrf_score in fused:
        chunk = dict(points_by_id[point_id].payload)
        chunk["score"] = rrf_score
        chunks.append(chunk)
    return chunks


def search(query: str, top_k: int = 8, collection_name: str = COLLECTION_NAME) -> list[dict]:
    candidates = _fuse(query, collection_name)
    return rerank(query, candidates, top_k)


def search_with_trace(query: str, top_k: int = 8, collection_name: str = COLLECTION_NAME) -> dict:
    """Same as search(), but also returns the pre-rerank (RRF-only) ranking
    so a caller (CLI --debug, the eval harness) can show/measure what
    reranking changed."""
    candidates = _fuse(query, collection_name)
    return {
        "pre_rerank": candidates[:top_k],
        "post_rerank": rerank(query, candidates, top_k),
    }
