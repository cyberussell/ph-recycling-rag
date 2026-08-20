"""Pydantic request/response models for the FastAPI layer."""

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 8


class SourceRef(BaseModel):
    title: str
    url: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_title: str
    section_id: str | None
    section_title: str | None
    page_number: int
    doc_type: str
    rerank_score: float


class AttemptTrace(BaseModel):
    query: str
    sufficient: bool
    reason: str
    escalate: bool
    escalate_reason: str


class DebugInfo(BaseModel):
    intent: str
    intent_reason: str
    out_of_scope: bool
    attempts: list[AttemptTrace]
    reformulated_query: str | None
    model: str
    escalated: bool
    escalate_reason: str
    final_sufficient: bool
    chunks: list[RetrievedChunk]


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    debug: DebugInfo
