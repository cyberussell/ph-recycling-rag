# PH Recycling Assistant (RAG)

A citizen-facing recycling/waste-segregation assistant for the Philippines,
grounded in real public regulatory text: RA 9003 (Ecological Solid Waste
Management Act of 2000) and its Implementing Rules and Regulations (DENR DAO
2001-34). Every answer cites the specific section it came from.

Full architecture/design plan: see `PLAN.md` (or ask for it — it covers the
hybrid search, reranking, and agentic-loop design planned for M2-M4).

## Status: M3 (reranking) complete

- Ingestion: fetches RA 9003 (lawphil.net), its IRR, the NSWMC National SWM
  Framework, and two household/consumer segregation guides — all from
  nswmc.emb.gov.ph/lawphil.net with a browser User-Agent (the bare
  `officialgazette.gov.ph`/`emb.gov.ph` hosts 403 default clients — see
  `src/ingestion/fetch.py`). Landscape-oriented PDF pages (the household
  guide's 2-column layout) are extracted as left/right column crops so text
  doesn't interleave across columns (`src/ingestion/parse_pdf.py`).
  Table-of-contents pages (including multi-page ToCs and unicode `…`
  leaders) are detected and dropped before chunking.
- Chunking: two chunkers behind one output schema. `section_chunker.py`
  handles the statute/IRR's numbered Section/Article/Rule structure
  (subsection splitting, self-describing citation headers). `prose_chunker.py`
  handles the narrative guidebook PDFs, which have no numbered sections —
  citations there are page-based, with best-effort ALL-CAPS/lettered heading
  detection for a human-readable label.
- Embedding: `BAAI/bge-m3` — multilingual (handles Taglish), dense (1024-dim)
  + sparse (lexical-weight) from a single forward pass.
- Indexing: Qdrant, embedded local mode by default (no Docker needed — set
  `QDRANT_URL` once you have a real server running via `docker-compose up`).
  Collection now carries both a `dense` and a `sparse` named vector.
- Retrieval: hybrid search — dense + sparse queried separately (top 20 each),
  fused client-side with Reciprocal Rank Fusion, then reranked down to top-k
  by a `bge-reranker-v2-m3` cross-encoder (`src/retrieval/hybrid_search.py`,
  `src/retrieval/rerank.py`). `search_with_trace()` exposes both the
  pre-rerank and post-rerank ranking for debugging/eval — `python -m src.cli
  ask ... --debug` prints both.
- Generation: unchanged from M1 — Claude Haiku with inline `[S#]` citations
  and the anti-hallucination guardrail.

481 chunks indexed across 5 documents (236 statute, 92 IRR, 84 framework, 5
household guide, 64 SWM guidebook).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Build the index

```bash
python -m src.ingestion.fetch m2       # downloads + caches raw docs to data/raw/
python -c "from src.indexing.build_index import build_and_index; build_and_index('m2', fresh=True)"
```

## Ask a question

```bash
python -m src.cli ask "What is the penalty for littering?"
python -m src.cli ask "Can I recycle a Tetra Pak?" --debug   # --debug shows retrieved chunks + scores
```

## Known limitation (M2→M3): reranking fixes English precision, not Taglish

M2 found that naive equal-weight RRF can demote the correct chunk on a
Taglish query (BGE-M3's sparse/lexical vector barely token-overlaps a
Tagalog query against an English corpus, degrading to near-random results
that dilute a correct dense-driven #1 down to #3). The hypothesis going into
M3 was that cross-encoder reranking would fix this, since it scores actual
query-chunk relevance rather than retrieval rank. Tested, and the real
result is more specific than that hypothesis:

- **English queries: reranking works very well.** For "What is the penalty
  for littering?", rerank scores show confident, sharp separation — top 2
  chunks at 0.78/0.73, everything else at ~0.03 — correctly promoting the
  IRR's fines section and demoting the merely-topically-adjacent "Prohibited
  Acts" section.
- **Natural Taglish queries: reranking does *not* reliably fix it.** For
  "Magkano ang multa sa pagkalat ng basura?" ("How much is the fine for
  littering?"), rerank scores stay uniformly low and flat (~0.01, 0.004,
  0.003...) — no confident separation at all — and reranking actually drops
  the correct IRR fines chunk out of the top 5 entirely, versus leaving it
  ranked #4 pre-rerank.

The flat, low, unconfident rerank scores on Tagalog input (vs. the sharp
English separation) are the tell: `bge-reranker-v2-m3`, despite being
labeled multilingual, doesn't appear well-calibrated for Filipino specifically.
Neither dense-only, hybrid RRF, nor cross-encoder reranking reliably solves
natural Taglish queries in this pipeline — end-to-end generation quality is
saved on these queries by Claude's generation step still picking the right
citation out of an imperfect candidate set (see `swm-made-easy`/IRR example
in commit history), not by retrieval ranking it correctly. That's a fragile
safety net, not a fix. This is why M4's agentic loop reformulates the query
itself (Taglish → statute-vocabulary English) rather than trying to fix this
at the ranking layer — the evidence now says the fix has to happen upstream
of retrieval, not within it.

## Roadmap

- **M4** — agentic corrective-retrieval loop: sufficiency check, bounded
  query reformulation (now evidenced as necessary, not just nice-to-have —
  see the Taglish reranking finding above), explicit fallback, +
  Haiku/Sonnet model routing.
- **M5** — evaluation harness (labeled Q&A set, recall/citation/faithfulness metrics).
- **M6** — FastAPI + Streamlit UI with a "how I found this" transparency panel.
