"""FastAPI wrapper around the agentic retrieval + generation pipeline.

Exposes one endpoint, POST /ask, that runs the full M4 pipeline (router ->
hybrid search + rerank -> sufficiency check -> bounded reformulate/retry ->
model-routed generation) and returns both the citizen-facing answer and a
full debug trace — the Streamlit UI's "How I found this" panel is just a
renderer for that trace, not a second implementation of it.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..generation.generate import generate_answer
from ..retrieval.agentic_loop import run_retrieval
from .schemas import AskRequest, AskResponse, AttemptTrace, DebugInfo, RetrievedChunk, SourceRef

app = FastAPI(title="PH Recycling Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
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
