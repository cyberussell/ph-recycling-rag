# PH Recycling Assistant (RAG)

A citizen-facing recycling/waste-segregation assistant for the Philippines,
grounded in real public regulatory text: RA 9003 (Ecological Solid Waste
Management Act of 2000) and its Implementing Rules and Regulations (DENR DAO
2001-34). Every answer cites the specific section it came from.

Full architecture/design plan: see `PLAN.md` (or ask for it — it covers the
hybrid search, reranking, and agentic-loop design planned for M2-M4).

## Status: M2 (multilingual + hybrid search) complete

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
- Retrieval: hybrid search — dense + sparse queried separately, fused
  client-side with Reciprocal Rank Fusion (`src/retrieval/hybrid_search.py`).
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

## Known limitation (M2): naive RRF can hurt cross-lingual queries

Verified finding, not a bug to silently patch: for a fully Taglish query
(e.g. "Ano ang parusa sa pag-litter?"), dense-only search alone correctly
ranks the right chunk (IRR Sec. 3, Fines and Penalties) #1 — BGE-M3's
multilingual embedding handles it fine. But BGE-M3's *sparse* vector is
lexical, so a Tagalog query barely token-overlaps with the English corpus;
sparse search degrades to near-random low-confidence results. Equal-weight
RRF then lets that sparse noise dilute the correct dense-driven answer from
#1 down to #3. The chunk isn't lost (still in the top 5), but precision at
rank 1 regresses versus dense-only for this query class. This is exactly the
precision problem M3's reranking pass (a cross-encoder scores actual
query-doc relevance, independent of how a candidate got shortlisted) is
designed to fix — left as-is rather than hand-tuning RRF weights ahead of
that milestone.

## Roadmap

- **M3** — cross-encoder reranking (`bge-reranker-v2-m3`) — next up: expected
  to directly fix the Taglish RRF precision regression noted above.
- **M4** — agentic corrective-retrieval loop (sufficiency check, bounded
  query reformulation, explicit fallback) + Haiku/Sonnet model routing.
- **M5** — evaluation harness (labeled Q&A set, recall/citation/faithfulness metrics).
- **M6** — FastAPI + Streamlit UI with a "how I found this" transparency panel.
