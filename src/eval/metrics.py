"""Eval metrics: retrieval recall, citation accuracy, LLM-judge faithfulness,
key-fact keyword coverage.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "claude-haiku-4-5-20251001"


def recall_hit(retrieved_chunks: list[dict], expected_chunk_ids: list[str]) -> bool:
    """True if any expected gold chunk appears anywhere in the retrieved list.
    Empty expected_chunk_ids (adversarial/out-of-scope questions) always counts
    as a hit — there's nothing to retrieve for those by design."""
    if not expected_chunk_ids:
        return True
    retrieved_ids = {c["chunk_id"] for c in retrieved_chunks}
    return bool(retrieved_ids & set(expected_chunk_ids))


def keyfact_coverage(answer: str, expected_keyfacts: list[str]) -> dict:
    if not expected_keyfacts:
        return {"coverage": 1.0, "found": [], "missing": []}
    answer_lower = answer.lower()
    found = [f for f in expected_keyfacts if f.lower() in answer_lower]
    missing = [f for f in expected_keyfacts if f not in found]
    return {"coverage": len(found) / len(expected_keyfacts), "found": found, "missing": missing}


_FAITHFULNESS_TOOL = {
    "name": "grade_faithfulness",
    "description": "Grade whether an answer's claims are all supported by the provided source excerpts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "description": "1 (mostly unsupported/hallucinated) to 5 (every claim clearly supported by the excerpts).",
            },
            "reason": {"type": "string", "description": "One short sentence."},
        },
        "required": ["score", "reason"],
    },
}


def _client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    return Anthropic(api_key=api_key)


def judge_faithfulness(question: str, chunks: list[dict], answer: str) -> dict:
    excerpts = "\n\n".join(f"[{i+1}] {c['text'][:500]}" for i, c in enumerate(chunks))
    user_message = (
        f"Question: {question}\n\nSource excerpts the answer should be grounded in:\n{excerpts}"
        f"\n\nGenerated answer:\n{answer}"
    )
    response = _client().messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        tools=[_FAITHFULNESS_TOOL],
        tool_choice={"type": "tool", "name": "grade_faithfulness"},
        messages=[{"role": "user", "content": user_message}],
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return dict(tool_use.input)
