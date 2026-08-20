"""CLI: ask a question against the indexed corpus end-to-end.

Usage:
    python -m src.cli ask "Can I recycle a Tetra Pak?"
    python -m src.cli ask "Magkano ang multa sa pagkalat ng basura?" --debug
    python -m src.cli ask "..." --no-agent   # bypass the agentic loop (M3 behavior only)
"""

import argparse

from .generation.generate import generate_answer
from .retrieval.agentic_loop import run_retrieval
from .retrieval.hybrid_search import search


def _label(c: dict) -> str:
    loc = c.get("section_id") or f"p.{c['page_number']}"
    return f"{c['source_title']} {loc}" + (f" - {c['section_title']}" if c.get("section_title") else "")


def _print_debug_trace(result: dict) -> None:
    cls = result["classification"]
    print(f"--- classification: {cls['intent']} ({cls['reason']}) ---")
    if result["out_of_scope"]:
        return

    for i, attempt in enumerate(result["attempts"], start=1):
        a = attempt["assessment"]
        print(f"--- attempt {i}: query={attempt['query']!r} ---")
        print(f"  sufficient={a['sufficient']}  reason={a['reason']}")
        print(f"  escalate={a['escalate']}  escalate_reason={a['escalate_reason']!r}")

    print(f"--- final chunks (model={result['model']}, escalate={result['escalate']}) ---")
    for c in result["chunks"]:
        print(f"  rerank={c.get('rerank_score', 0):.4f}  {_label(c)}")
    print()


def ask(query: str, top_k: int = 5, debug: bool = False, use_agent: bool = True) -> None:
    if not use_agent:
        chunks = search(query, top_k=top_k)
        result_answer = generate_answer(query, chunks)
        print(result_answer["answer"])
        return

    result = run_retrieval(query, top_k=top_k)

    if debug:
        _print_debug_trace(result)

    if result["out_of_scope"]:
        print(result["redirect_message"])
        return

    answer = generate_answer(
        query,
        result["chunks"],
        model=result["model"],
        low_confidence=not result["final_sufficient"],
    )
    print(answer["answer"])
    if answer["invalid_citations"]:
        print(f"\n[warning] stripped hallucinated citations: {answer['invalid_citations']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PH Recycling Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", type=str)
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--debug", action="store_true")
    ask_parser.add_argument("--no-agent", action="store_true", help="bypass the agentic loop")

    args = parser.parse_args()
    if args.command == "ask":
        ask(args.query, top_k=args.top_k, debug=args.debug, use_agent=not args.no_agent)


if __name__ == "__main__":
    main()
