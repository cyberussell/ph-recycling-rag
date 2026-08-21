"""FastAPI wrapper around the agentic retrieval + generation pipeline.

Exposes one endpoint, POST /ask, that runs the full M4 pipeline (router ->
hybrid search + rerank -> sufficiency check -> bounded reformulate/retry ->
model-routed generation) and returns both the citizen-facing answer and a
full debug trace — the Streamlit UI's "How I found this" panel is just a
renderer for that trace, not a second implementation of it.
"""

import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from ..generation.generate import generate_answer
from ..indexing.build_index import index_chunks, load_processed_chunks
from ..indexing.client import get_client
from ..indexing.qdrant_schema import COLLECTION_NAME
from ..retrieval.agentic_loop import run_retrieval
from ..retrieval.rerank import rerank
from .rate_limit import check_and_increment
from .schemas import AskRequest, AskResponse, AttemptTrace, DebugInfo, RetrievedChunk, SourceRef

app = FastAPI(title="PH Recycling Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_index_ready = threading.Event()
_index_build_error: str | None = None


def _ensure_index_built() -> None:
    """Bootstraps the vector index from the already-committed data/processed/
    chunks if the collection is missing or empty — needed on a fresh
    container (e.g. a Hugging Face Space) where Qdrant's embedded local
    storage starts empty (this repo also commits a pre-built qdrant_local/,
    so this is normally a fast no-op fallback, not the primary path). Runs
    in a background thread so /health responds immediately; /ask returns 503
    until this completes.

    If the build fails, _index_ready is still set (so the app doesn't hang
    "warming up" forever) but the error is recorded for /health to surface —
    silently swallowing the failure previously left /ask serving a broken
    index while reporting index_ready=true, which is worse than a clear error."""
    global _index_build_error
    try:
        client = get_client()
        needs_build = True
        if client.collection_exists(COLLECTION_NAME):
            count = client.count(COLLECTION_NAME).count
            needs_build = count == 0
        if needs_build:
            index_chunks(load_processed_chunks("m2"), fresh=True)
        # Load the reranker model now rather than on the first real request —
        # otherwise the very first /ask after a cold start pays its
        # multi-second load time on top of already-slow CPU inference,
        # stacking onto the timeout problem this whole warmup is meant to help.
        rerank("warmup", [{"text": "warmup"}], top_k=1)
    except Exception as e:  # noqa: BLE001 - must not crash the background thread
        _index_build_error = f"{type(e).__name__}: {e}"
    finally:
        _index_ready.set()


@app.on_event("startup")
def _on_startup() -> None:
    threading.Thread(target=_ensure_index_built, daemon=True).start()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_ready": _index_ready.is_set(), "index_build_error": _index_build_error}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    if not _index_ready.is_set():
        raise HTTPException(
            status_code=503,
            detail="Still warming up (loading models and building the index) — this can take a few minutes on first boot. Please retry shortly.",
        )
    if _index_build_error:
        raise HTTPException(status_code=503, detail=f"Index build failed on startup: {_index_build_error}")

    client_ip = request.client.host if request.client else "unknown"
    allowed, reason = check_and_increment(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    result = run_retrieval(req.question, top_k=req.top_k)
    classification = result["classification"]

    if result["out_of_scope"]:
        return AskResponse(
            answer=result["redirect_message"],
            sources=[],
            debug=DebugInfo(
                intent=classification["intent"],
                intent_reason=classification["reason"],
                out_of_scope=True,
                attempts=[],
                reformulated_query=None,
                model="router_only",
                escalated=False,
                escalate_reason="",
                final_sufficient=True,
                chunks=[],
            ),
        )

    gen = generate_answer(
        req.question, result["chunks"], model=result["model"], low_confidence=not result["final_sufficient"]
    )

    sources = []
    seen_urls = set()
    for c in result["chunks"]:
        if c["source_url"] not in seen_urls:
            sources.append(SourceRef(title=c["source_title"], url=c["source_url"]))
            seen_urls.add(c["source_url"])

    return AskResponse(
        answer=gen["answer"],
        sources=sources,
        debug=DebugInfo(
            intent=classification["intent"],
            intent_reason=classification["reason"],
            out_of_scope=False,
            attempts=[
                AttemptTrace(
                    query=a["query"],
                    sufficient=a["assessment"]["sufficient"],
                    reason=a["assessment"]["reason"],
                    escalate=a["assessment"]["escalate"],
                    escalate_reason=a["assessment"]["escalate_reason"],
                )
                for a in result["attempts"]
            ],
            reformulated_query=result["reformulated_query"],
            model=result["model"],
            escalated=result["escalate"],
            escalate_reason=result["escalate_reason"],
            final_sufficient=result["final_sufficient"],
            chunks=[
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    source_title=c["source_title"],
                    section_id=c.get("section_id"),
                    section_title=c.get("section_title"),
                    page_number=c["page_number"],
                    doc_type=c["doc_type"],
                    rerank_score=c.get("rerank_score", 0.0),
                )
                for c in result["chunks"]
            ],
        ),
    )
