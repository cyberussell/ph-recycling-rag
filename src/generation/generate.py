"""Claude API generation call.

Model routing (per the project plan): default to Haiku for the final answer,
escalate to Sonnet only when the agentic loop (M4) signals it's warranted —
conflicting provisions, multi-section synthesis, a legal-interpretation
question, or a failed sufficiency check. M1 always uses the default model;
the escalation policy plugs in once router.py/agentic_loop.py exist.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from .citation_check import verify_citations
from .prompt_templates import SYSTEM_PROMPT, build_user_message, sources_footer

load_dotenv()

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
ESCALATED_MODEL = "claude-sonnet-5"


def _client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Anthropic(api_key=api_key)


def generate_answer(query: str, chunks: list[dict], model: str = DEFAULT_MODEL) -> dict:
    if not chunks:
        return {
            "answer": "I don't have any indexed source material to answer that yet.",
            "model": model,
            "cited": [],
            "invalid_citations": [],
        }

    user_message = build_user_message(query, chunks)
    response = _client().messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw_answer = "".join(block.text for block in response.content if block.type == "text")

    checked = verify_citations(raw_answer, num_chunks_provided=len(chunks))
    full_answer = f"{checked['answer']}\n\n{sources_footer(chunks)}"

    return {
        "answer": full_answer,
        "model": model,
        "cited": checked["cited"],
        "invalid_citations": checked["invalid_citations"],
    }
