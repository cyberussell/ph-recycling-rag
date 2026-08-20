"""End-to-end indexing: raw docs -> parse -> chunk -> embed -> upsert into Qdrant.

Expects `data/raw/` to already be populated by `src.ingestion.fetch` (and its
`source_manifest.json` written alongside). Also caches the chunked output to
`data/processed/<doc_id>.json` so the chunker's output is inspectable without
re-running embedding.
"""

import json
import uuid
from pathlib import Path

from qdrant_client.http import models as qm

from ..chunking.prose_chunker import chunk_prose_document
from ..chunking.section_chunker import CONFIGS, chunk_document
from ..embedding.embedder import embed_texts, vector_size
from ..ingestion.parse_html import parse_html
from ..ingestion.parse_pdf import parse_pdf
from ..ingestion.source_manifest import sources_for_milestone
from .client import get_client
from .qdrant_schema import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    recreate_collection,
)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Qdrant point ids must be an unsigned int or UUID; chunk_id stays in the
# payload as the human-readable citation key.
def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _parse_raw(doc_id: str, fmt: str, source_url: str) -> list[dict]:
    ext = "html" if fmt == "html" else "pdf"
    raw_bytes = (RAW_DIR / f"{doc_id}.{ext}").read_bytes()
    return parse_html(raw_bytes, source_url) if fmt == "html" else parse_pdf(raw_bytes, source_url)


def build_chunks(milestone: str = "m1") -> list[dict]:
    manifest = json.loads((RAW_DIR / "source_manifest.json").read_text())
    manifest_by_id = {m["doc_id"]: m for m in manifest}

    all_chunks: list[dict] = []
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for doc in sources_for_milestone(milestone):
        entry = manifest_by_id[doc.doc_id]
        pages = _parse_raw(doc.doc_id, entry["fmt"], entry["url"])
        if doc.doc_id in CONFIGS:
            chunks = chunk_document(pages, CONFIGS[doc.doc_id])
        else:
            chunks = chunk_prose_document(pages, doc.doc_id, doc.title, doc.doc_type)
        for c in chunks:
            c["source_url"] = entry["url"]
            c["fetch_date"] = entry["fetch_date"]
            c["jurisdiction"] = entry["jurisdiction"]
            c["language"] = entry["language"]
        (PROCESSED_DIR / f"{doc.doc_id}.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        print(f"  {doc.doc_id}: {len(pages)} pages -> {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks


def index_chunks(
    chunks: list[dict], collection_name: str = COLLECTION_NAME, batch_size: int = 16, fresh: bool = False
) -> None:
    client = get_client()
    if fresh:
        recreate_collection(client, vector_size(), collection_name)
    else:
        ensure_collection(client, vector_size(), collection_name)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embed_texts([c["embedded_text"] for c in batch])
        points = [
            qm.PointStruct(
                id=_point_id(c["chunk_id"]),
                vector={
                    DENSE_VECTOR_NAME: emb["dense"],
                    SPARSE_VECTOR_NAME: qm.SparseVector(
                        indices=emb["sparse"]["indices"], values=emb["sparse"]["values"]
                    ),
                },
                payload=c,
            )
            for c, emb in zip(batch, embeddings)
        ]
        client.upsert(collection_name=collection_name, points=points)
        print(f"  indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")


def build_and_index(milestone: str = "m1", fresh: bool = False) -> None:
    print(f"Building chunks for milestone={milestone} ...")
    chunks = build_chunks(milestone)
    print(f"Total chunks: {len(chunks)}. Embedding + indexing ...")
    index_chunks(chunks, fresh=fresh)
    print("Done.")


if __name__ == "__main__":
    import sys

    build_and_index(sys.argv[1] if len(sys.argv) > 1 else "m1")
