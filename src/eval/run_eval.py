"""Evaluation harness.

Two tiers, deliberately scoped to keep runtime/cost reasonable for a
portfolio project rather than a production eval suite:

- Retrieval recall (Recall@5, Recall@10, pre- vs post-rerank) runs on the
  FULL qa_set.jsonl — no Claude calls, just embedding + reranking, so it's
  cheap enough to run exhaustively. This is the "what did reranking buy you"
  comparison from the M2/M3 README findings, now measured instead of
  eyeballed.
- Citation accuracy, faithfulness (LLM-judge), and key-fact coverage require
  running the full agentic pipeline (router -> search -> sufficiency check
  -> maybe reformulate+retry -> generate), which is several Claude calls per
  question. Run only on the ~15 questions flagged `use_for_generation_eval`
  in the eval set — a stratified sample across all 5 categories, sized to
  match the scale the project plan itself suggested for manual citation
  spot-checks.

Usage: python -m src.eval.run_eval
"""

import json
from pathlib import Path

from ..generation.generate import generate_answer
from ..retrieval.agentic_loop import run_retrieval
from ..retrieval.hybrid_search import search_with_trace
from .metrics import judge_faithfulness, keyfact_coverage, recall_hit

QA_SET_PATH = Path(__file__).parent / "qa_set.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_qa_set() -> list[dict]:
    with open(QA_SET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_retrieval(item: dict) -> dict:
    trace = search_with_trace(item["question"], top_k=10)
    expected = item["expected_chunk_ids"]
    return {
        "id": item["id"],
        "category": item["category"],
        "pre_rerank_hit_at_5": recall_hit(trace["pre_rerank"][:5], expected),
        "pre_rerank_hit_at_10": recall_hit(trace["pre_rerank"][:10], expected),
        "post_rerank_hit_at_5": recall_hit(trace["post_rerank"][:5], expected),
        "post_rerank_hit_at_10": recall_hit(trace["post_rerank"][:10], expected),
    }


def eval_generation(item: dict) -> dict:
    result = run_retrieval(item["question"], top_k=5)

    if result["out_of_scope"]:
        answer = result["redirect_message"]
        kf = keyfact_coverage(answer, item["expected_answer_keyfacts"])
        return {
            "id": item["id"],
            "category": item["category"],
            "model": "router_only",
            "answer": answer,
            "keyfact_coverage": kf["coverage"],
            "keyfact_missing": kf["missing"],
            "citation_clean": True,
            "faithfulness_score": None,
            "faithfulness_reason": "n/a (out of scope, no generation call)",
        }

    gen = generate_answer(
        item["question"], result["chunks"], model=result["model"], low_confidence=not result["final_sufficient"]
    )
    kf = keyfact_coverage(gen["answer"], item["expected_answer_keyfacts"])
    faith = judge_faithfulness(item["question"], result["chunks"], gen["answer"])

    return {
        "id": item["id"],
        "category": item["category"],
        "model": result["model"],
        "escalated": result["escalate"],
        "reformulated": result["reformulated_query"] is not None,
        "answer": gen["answer"],
        "keyfact_coverage": kf["coverage"],
        "keyfact_missing": kf["missing"],
        "citation_clean": not gen["invalid_citations"],
        "faithfulness_score": faith["score"],
        "faithfulness_reason": faith["reason"],
    }


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(retrieval_results: list[dict], generation_results: list[dict]) -> dict:
    def rate(key: str) -> float:
        return _avg([1.0 if r[key] else 0.0 for r in retrieval_results])

    return {
        "n_retrieval": len(retrieval_results),
        "n_generation": len(generation_results),
        "recall_at_5_pre_rerank": rate("pre_rerank_hit_at_5"),
        "recall_at_5_post_rerank": rate("post_rerank_hit_at_5"),
        "recall_at_10_pre_rerank": rate("pre_rerank_hit_at_10"),
        "recall_at_10_post_rerank": rate("post_rerank_hit_at_10"),
        "avg_keyfact_coverage": _avg([r["keyfact_coverage"] for r in generation_results]),
        "citation_clean_rate": _avg([1.0 if r["citation_clean"] else 0.0 for r in generation_results]),
        "avg_faithfulness": _avg([r["faithfulness_score"] for r in generation_results if r["faithfulness_score"]]),
        "model_usage": {
            model: sum(1 for r in generation_results if r["model"] == model)
            for model in {r["model"] for r in generation_results}
        },
    }


def render_markdown(summary: dict, retrieval_results: list[dict], generation_results: list[dict]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Retrieval recall measured on {summary['n_retrieval']} questions; "
        f"generation metrics measured on {summary['n_generation']} questions "
        "(stratified subset, per the scope note in run_eval.py).",
        "",
        "## Retrieval: what reranking bought you",
        "",
        "| Metric | Pre-rerank (RRF only) | Post-rerank |",
        "|---|---|---|",
        f"| Recall@5 | {summary['recall_at_5_pre_rerank']:.0%} | {summary['recall_at_5_post_rerank']:.0%} |",
        f"| Recall@10 | {summary['recall_at_10_pre_rerank']:.0%} | {summary['recall_at_10_post_rerank']:.0%} |",
        "",
        "## Generation quality",
        "",
        f"- Average key-fact coverage: {summary['avg_keyfact_coverage']:.0%}",
        f"- Citation-clean rate (no hallucinated `[S#]`): {summary['citation_clean_rate']:.0%}",
        f"- Average faithfulness (LLM-judge, 1-5): {summary['avg_faithfulness']:.2f}",
        f"- Model usage: {summary['model_usage']}",
        "",
        "## Per-category retrieval recall (post-rerank, Recall@5)",
        "",
        "| Category | Recall@5 |",
        "|---|---|",
    ]
    categories = sorted({r["category"] for r in retrieval_results})
    for cat in categories:
        cat_results = [r for r in retrieval_results if r["category"] == cat]
        cat_recall = _avg([1.0 if r["post_rerank_hit_at_5"] else 0.0 for r in cat_results])
        lines.append(f"| {cat} | {cat_recall:.0%} |")

    lines += ["", "## Generation results detail", ""]
    for r in generation_results:
        lines.append(f"### {r['id']} ({r['category']}) — model={r['model']}")
        lines.append(f"- key-fact coverage: {r['keyfact_coverage']:.0%} (missing: {r['keyfact_missing']})")
        lines.append(f"- citation clean: {r['citation_clean']}")
        lines.append(f"- faithfulness: {r['faithfulness_score']} ({r['faithfulness_reason']})")
        lines.append("")

    return "\n".join(lines)


def run() -> None:
    qa_set = load_qa_set()
    print(f"Loaded {len(qa_set)} questions.")

    print("Evaluating retrieval (recall@5/@10, pre/post rerank) on all questions...")
    retrieval_results = []
    for item in qa_set:
        r = eval_retrieval(item)
        retrieval_results.append(r)
        print(f"  {item['id']}: pre@5={r['pre_rerank_hit_at_5']} post@5={r['post_rerank_hit_at_5']}")

    gen_items = [item for item in qa_set if item["use_for_generation_eval"]]
    print(f"\nEvaluating generation (citation/faithfulness/keyfact) on {len(gen_items)} questions...")
    generation_results = []
    for item in gen_items:
        r = eval_generation(item)
        generation_results.append(r)
        print(f"  {item['id']}: model={r['model']} keyfact={r['keyfact_coverage']:.0%} faithfulness={r['faithfulness_score']}")

    summary = summarize(retrieval_results, generation_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "eval_results.json").write_text(
        json.dumps(
            {"summary": summary, "retrieval": retrieval_results, "generation": generation_results}, indent=2
        ),
        encoding="utf-8",
    )
    report = render_markdown(summary, retrieval_results, generation_results)
    (RESULTS_DIR / "eval_report.md").write_text(report, encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull report: {RESULTS_DIR / 'eval_report.md'}")


if __name__ == "__main__":
    run()
