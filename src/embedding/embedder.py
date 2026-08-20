"""Dense + sparse embedding wrapper (M2).

Upgraded from M1's English-only `bge-small-en-v1.5` to `BAAI/bge-m3`:
multilingual (needed for Taglish-phrased queries), 8192-token context, and —
the key reason for choosing it — a single forward pass yields both a dense
vector and a sparse (lexical-weight) vector, which is exactly what hybrid
search needs without stitching together two separate models.

Forced onto CPU (see rerank.py for the same note): sentence-transformers/
FlagEmbedding auto-select Apple's MPS GPU backend on this machine, and the
agentic loop (M4) embeds queries repeatedly in one process — MPS ran out of
memory under that load. CPU is slower but doesn't have that failure mode.
"""

from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel

MODEL_NAME = "BAAI/bge-m3"
DENSE_DIM = 1024


@lru_cache(maxsize=1)
def _model() -> BGEM3FlagModel:
    return BGEM3FlagModel(MODEL_NAME, use_fp16=False, devices="cpu")


def _sparse_from_lexical_weights(weights: dict) -> dict:
    """Converts FlagEmbedding's {token_id_str: weight} into Qdrant's
    {"indices": [...], "values": [...]} sparse vector format."""
    indices = [int(token_id) for token_id in weights]
    values = [float(w) for w in weights.values()]
    return {"indices": indices, "values": values}


def embed_texts(texts: list[str]) -> list[dict]:
    """Embeds a batch of chunk texts (or queries). Returns a list of
    {"dense": [...], "sparse": {"indices": [...], "values": [...]}}."""
    out = _model().encode(
        texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
    )
    return [
        {"dense": dense.tolist(), "sparse": _sparse_from_lexical_weights(sparse)}
        for dense, sparse in zip(out["dense_vecs"], out["lexical_weights"])
    ]


def embed_query(query: str) -> dict:
    return embed_texts([query])[0]


def vector_size() -> int:
    return DENSE_DIM
