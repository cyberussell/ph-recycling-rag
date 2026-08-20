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
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME)


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not chunks:
        return []
    pairs = [(query, c["text"]) for c in chunks]
    scores = _model().predict(pairs)
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
