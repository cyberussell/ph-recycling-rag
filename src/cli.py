"""M1 CLI: ask a question against the indexed corpus end-to-end.

Usage:
    python -m src.cli ask "Can I recycle a Tetra Pak?"
    python -m src.cli ask "What's the penalty for littering?" --top-k 5 --debug
"""

import argparse

from .generation.generate import generate_answer
from .retrieval.hybrid_search import search, search_with_trace


def _label(c: dict) -> str:
    loc = c["section_id"] or f"p.{c['page_number']}"
    return f"{c['source_title']} {loc}" + (f" - {c['section_title']}" if c.get("section_title") else "")


def ask(query: str, top_k: int = 5, debug: bool = False) -> None:
    if debug:
        trace = search_with_trace(query, top_k=top_k)
        print(f"--- pre-rerank (RRF fusion, top {top_k}) ---")
        for c in trace["pre_rerank"]:
            print(f"  rrf={c['score']:.4f}  {_label(c)}")
        print(f"--- post-rerank (cross-encoder, top {top_k}) ---")
        for c in trace["post_rerank"]:
            print(f"  rerank={c['rerank_score']:.4f}  {_label(c)}")
        print()
        chunks = trace["post_rerank"]
    else:
        chunks = search(query, top_k=top_k)

    result = generate_answer(query, chunks)

    print(result["answer"])
    if result["invalid_citations"]:
        print(f"\n[warning] stripped hallucinated citations: {result['invalid_citations']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PH Recycling Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", type=str)
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.command == "ask":
        ask(args.query, top_k=args.top_k, debug=args.debug)


if __name__ == "__main__":
    main()
