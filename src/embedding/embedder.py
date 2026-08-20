"""Dense embedding wrapper.

M1 uses a small English-only model (`bge-small-en-v1.5`) to keep local
iteration fast while the rest of the pipeline (chunking, indexing, retrieval,
generation) gets proven end-to-end. The corpus itself is English legal text,
so this is sufficient for M1. M2 swaps this for `BAAI/bge-m3`, which is
multilingual (needed once Taglish-phrased queries and the sparse/hybrid path
are in scope) and produces dense + sparse vectors from a single forward pass
— see the retrieval/hybrid_search module added at that milestone.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of chunk texts (or queries) for cosine-similarity search."""
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def vector_size() -> int:
    return _model().get_embedding_dimension()
