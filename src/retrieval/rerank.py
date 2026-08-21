"""Cross-encoder reranking pass over hybrid search's fused candidates.

Uses `bge-reranker-v2-m3` via sentence-transformers' `CrossEncoder` (not
FlagEmbedding's own `FlagReranker` — that path pins an older transformers
tokenizer API that conflicts with the transformers version the BGE-M3
embedder needs; CrossEncoder wraps the same model checkpoint without the
version conflict).

A cross-encoder scores actual query-chunk relevance jointly (unlike the bi-
encoder dense/sparse vectors, which score query and chunk independently),
which is what lets it recover precision RRF fusion can lose — see the M2
finding in the README about Taglish queries.

Forced onto CPU: sentence-transformers auto-selects Apple's MPS GPU backend
on this machine, and the agentic loop (M4) can call rerank() twice in one
process (initial attempt + reformulated retry) — MPS ran out of memory doing
that back-to-back alongside the embedder's own model. CPU is slower but
doesn't have that failure mode, and correctness matters more than latency here.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Cross-encoder cost scales with input length; a chunk can run up to ~700
# tokens (section_chunker.py's cap) but relevance is almost always evident
# from the opening portion. Capping the reranked text (not the text shown to
# the user or passed to generation — just what the reranker itself sees)
# was the other half of fixing real >100s response times on CPU-only hosting,
# alongside FUSION_CANDIDATES in hybrid_search.py.
RERANK_TEXT_CHARS = 400


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME, device="cpu")


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not chunks:
        return []
    pairs = [(query, c["text"][:RERANK_TEXT_CHARS]) for c in chunks]
    scores = _model().predict(pairs)
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
