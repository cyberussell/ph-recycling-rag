"""Corrective-retrieval agentic loop.

Ties together the router, hybrid search + rerank, a sufficiency check, and a
bounded query-reformulation retry. Concrete and scoped, per the plan: one
retrieve -> grade -> reformulate-or-fallback loop, hard-capped at 2 retrieval
attempts, not an open-ended agent.

Evidenced by the M3 finding (see README): reranking alone doesn't reliably
fix natural Taglish queries — this loop is where that actually gets fixed,
by reformulating the query into statute vocabulary before re-retrieving.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from ..generation.generate import DEFAULT_MODEL, ESCALATED_MODEL
from .hybrid_search import search
from .router import OUT_OF_SCOPE_REDIRECT, classify_query

load_dotenv()

LOOP_MODEL = "claude-haiku-4-5-20251001"
MAX_ATTEMPTS = 2
DOC_TYPE_BOOST_FACTOR = 1.15

# Common Taglish/slang terms in this domain don't literally appear in the
# statute, so a bare embedding/rerank match on them is unreliable (see the
# M3 finding). Nudging the reformulation call with a few examples keeps it
# from having to guess the domain's vocabulary from scratch.
_GLOSSARY_HINT = """Examples of Taglish/slang -> statute vocabulary in this domain:
- "tetra pack" / "juice box" / "milk carton" -> composite/laminated packaging materials
- "basura" -> solid waste
- "multa" / "parusa" -> fines and penalties
- "ikalat" / "pagkalat" / "nagkalat" -> littering, throwing, dumping of waste matters in public places
- "tapon" / "itapon" -> dispose, dumping
"""

_SUFFICIENCY_TOOL = {
    "name": "assess_sufficiency",
    "description": (
        "Assess whether the retrieved excerpts can fully answer the citizen's question, "
        "and whether the final answer needs a stronger model."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sufficient": {
                "type": "boolean",
                "description": "True only if the excerpts contain everything needed, including specific figures/section numbers if asked for.",
            },
            "reason": {"type": "string", "description": "One short sentence."},
            "escalate": {
                "type": "boolean",
                "description": (
                    "True if: multiple excerpts show conflicting or overlapping provisions that need "
                    "reconciling, the question asks for legal interpretation rather than a plain lookup, "
                    "or answering requires synthesizing several distinct provisions together."
                ),
            },
            "escalate_reason": {"type": "string", "description": "One short sentence, empty if escalate is false."},
        },
        "required": ["sufficient", "reason", "escalate", "escalate_reason"],
    },
}


def _client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    return Anthropic(api_key=api_key)


def _format_excerpts(chunks: list[dict]) -> str:
    # Truncated, but generously — a too-short preview (300 chars clipped a real
    # fine amount mid-sentence during testing) causes false "insufficient"
    # verdicts that are really just an artifact of the preview, not the chunk.
    lines = []
    for i, c in enumerate(chunks, start=1):
        loc = c.get("section_id") or f"p.{c['page_number']}"
        lines.append(f"[{i}] {c['source_title']} {loc} ({c.get('section_title') or ''}): {c['text'][:800]}")
    return "\n".join(lines)


def assess_sufficiency(query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"sufficient": False, "reason": "no chunks retrieved", "escalate": False, "escalate_reason": ""}

    user_message = f"Question: {query}\n\nRetrieved excerpts:\n{_format_excerpts(chunks)}"
    response = _client().messages.create(
        model=LOOP_MODEL,
        max_tokens=300,
        tools=[_SUFFICIENCY_TOOL],
        tool_choice={"type": "tool", "name": "assess_sufficiency"},
        messages=[{"role": "user", "content": user_message}],
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return dict(tool_use.input)


def reformulate_query(query: str) -> str:
    system = (
        "Rewrite the user's question in clear English using Philippine solid waste "
        "management statute vocabulary (RA 9003 and its IRR), preserving the original "
        "intent exactly. Output ONLY the rewritten question, nothing else.\n\n" + _GLOSSARY_HINT
    )
    response = _client().messages.create(
        model=LOOP_MODEL,
        max_tokens=150,
        system=system,
        messages=[{"role": "user", "content": query}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _apply_doc_type_boost(chunks: list[dict], boost_doc_types: list[str]) -> list[dict]:
    if not boost_doc_types:
        return chunks
    for c in chunks:
        if c.get("doc_type") in boost_doc_types:
            c["rerank_score"] = c.get("rerank_score", 0.0) * DOC_TYPE_BOOST_FACTOR
    return sorted(chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)


def run_retrieval(query: str, top_k: int = 5) -> dict:
    classification = classify_query(query)
    if classification["intent"] == "out_of_scope":
        return {
            "out_of_scope": True,
            "redirect_message": OUT_OF_SCOPE_REDIRECT,
            "classification": classification,
            "attempts": [],
        }

    attempts = []
    current_query = query
    chunks = search(current_query, top_k=top_k)
    chunks = _apply_doc_type_boost(chunks, classification["doc_type_boost"])
    assessment = assess_sufficiency(query, chunks)
    attempts.append({"query": current_query, "assessment": assessment})

    reformulated_query = None
    if not assessment["sufficient"] and len(attempts) < MAX_ATTEMPTS:
        reformulated_query = reformulate_query(query)
        chunks = search(reformulated_query, top_k=top_k)
        chunks = _apply_doc_type_boost(chunks, classification["doc_type_boost"])
        assessment = assess_sufficiency(query, chunks)
        attempts.append({"query": reformulated_query, "assessment": assessment})

    escalate = assessment["escalate"] or not assessment["sufficient"]
    escalate_reason = assessment["escalate_reason"] or (
        "sufficiency check still failed after reformulation" if not assessment["sufficient"] else ""
    )

    return {
        "out_of_scope": False,
        "classification": classification,
        "chunks": chunks,
        "attempts": attempts,
        "reformulated_query": reformulated_query,
        "model": ESCALATED_MODEL if escalate else DEFAULT_MODEL,
        "escalate": escalate,
        "escalate_reason": escalate_reason,
        "final_sufficient": assessment["sufficient"],
    }
