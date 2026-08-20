"""Query classification: cheap first step of the agentic loop.

A single Haiku tool-call classifies intent, which drives two things: an
early exit for off-topic questions (no wasted retrieval/generation cost, no
risk of hallucinating an answer to something outside the corpus), and a soft
doc_type boost applied later in agentic_loop.py (never a hard filter — the
plan is explicit that over-pruning recall is worse than a little noise).
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

ROUTER_MODEL = "claude-haiku-4-5-20251001"

INTENT_DOC_TYPE_BOOST = {
    "segregation_howto": ["advisory", "framework"],
    "penalty_legal": ["statute", "irr"],
    "definition": ["statute", "irr"],
    "out_of_scope": [],
}

OUT_OF_SCOPE_REDIRECT = (
    "I can only help with questions about Philippine solid waste management, "
    "recycling, and segregation rules (RA 9003 and related guidance). "
    "Could you rephrase your question around that topic?"
)

_CLASSIFY_TOOL = {
    "name": "classify_query",
    "description": "Classify a citizen's question about Philippine waste management.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(INTENT_DOC_TYPE_BOOST.keys()),
                "description": (
                    "segregation_howto: practical how-to-sort-my-trash questions. "
                    "penalty_legal: fines, penalties, what's prohibited. "
                    "definition: what counts as X, general RA 9003 concepts. "
                    "out_of_scope: unrelated to PH waste/recycling/segregation law."
                ),
            },
            "reason": {"type": "string", "description": "One short sentence."},
        },
        "required": ["intent", "reason"],
    },
}


def _client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    return Anthropic(api_key=api_key)


def classify_query(query: str) -> dict:
    response = _client().messages.create(
        model=ROUTER_MODEL,
        max_tokens=200,
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_query"},
        messages=[{"role": "user", "content": f"Classify this question: {query}"}],
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    result = dict(tool_use.input)
    result["doc_type_boost"] = INTENT_DOC_TYPE_BOOST.get(result["intent"], [])
    return result
