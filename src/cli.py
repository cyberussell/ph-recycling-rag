"""M1 CLI: ask a question against the indexed corpus end-to-end.

Usage:
    python -m src.cli ask "Can I recycle a Tetra Pak?"
    python -m src.cli ask "What's the penalty for littering?" --top-k 5 --debug
"""

import argparse

from .generation.generate import generate_answer
from .retrieval.hybrid_search import search


def ask(query: str, top_k: int = 5, debug: bool = False) -> None:
    chunks = search(query, top_k=top_k)

    if debug:
        print(f"--- retrieved {len(chunks)} chunks ---")
        for c in chunks:
            print(f"  {c['chunk_id']}  score={c['score']:.3f}  {c['source_title']} {c['section_id']}")
        print()

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
