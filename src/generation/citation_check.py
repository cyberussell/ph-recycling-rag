"""Programmatic guardrail against citation hallucination.

Prompting an LLM to only cite provided sources reduces but does not eliminate
invented citations. This regex-extracts every [S#] token from a generated
answer and verifies it refers to a chunk that was actually in that call's
context, stripping/flagging anything that wasn't.
"""

import re

CITATION_RE = re.compile(r"\[S(\d+)\]")


def verify_citations(answer: str, num_chunks_provided: int) -> dict:
    cited = {int(m) for m in CITATION_RE.findall(answer)}
    valid = {c for c in cited if 1 <= c <= num_chunks_provided}
    invalid = cited - valid

    cleaned_answer = answer
    for bad in invalid:
        cleaned_answer = re.sub(rf"\[S{bad}\]", "[unverified citation removed]", cleaned_answer)

    return {
        "answer": cleaned_answer,
        "cited": sorted(cited),
        "valid_citations": sorted(valid),
        "invalid_citations": sorted(invalid),
        "has_hallucinated_citation": bool(invalid),
    }
