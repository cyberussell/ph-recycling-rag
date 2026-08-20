# PH Recycling Assistant (RAG)

A citizen-facing recycling/waste-segregation assistant for the Philippines,
grounded in real public regulatory text: RA 9003 (Ecological Solid Waste
Management Act of 2000) and its Implementing Rules and Regulations (DENR DAO
2001-34). Every answer cites the specific section it came from.

Full architecture/design plan: see `PLAN.md` (or ask for it — it covers the
hybrid search, reranking, and agentic-loop design planned for M2-M4).

## Status: M1 (baseline) complete

- Ingestion: fetches RA 9003 (lawphil.net) + IRR (nswmc.emb.gov.ph), with a
  browser User-Agent (the bare `officialgazette.gov.ph`/`emb.gov.ph` hosts
  403 default clients — see `src/ingestion/fetch.py`).
- Chunking: section-aware, not fixed-size — regex-detects Section/Article/
  Chapter (statute) and Part/Rule/Section (IRR) boundaries, splits oversized
  sections at lettered subsections, and prepends a citation header to the
  embedded text so every chunk is self-describing (`src/chunking/section_chunker.py`).
- Embedding: `BAAI/bge-small-en-v1.5` (dense-only for M1; upgrades to
  multilingual `BAAI/bge-m3` dense+sparse at M2 for hybrid search).
- Indexing: Qdrant, embedded local mode by default (no Docker needed — set
  `QDRANT_URL` once you have a real server running via `docker-compose up`).
- Retrieval + generation: plain dense top-k, Claude Haiku generation with
  inline `[S#]` citations, and a programmatic guardrail that strips any
  citation the model invents that wasn't actually in its context.

328 chunks indexed from RA 9003 (236) + the IRR (92).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Build the index

```bash
python -m src.ingestion.fetch m1       # downloads + caches raw docs to data/raw/
python -m src.indexing.build_index m1  # parses, chunks, embeds, upserts to Qdrant
```

## Ask a question

```bash
python -m src.cli ask "What is the penalty for littering?"
python -m src.cli ask "Can I recycle a Tetra Pak?" --debug   # --debug shows retrieved chunks + scores
```

## Known limitation (M1)

`bge-small-en-v1.5` is English-only. Taglish-phrased queries (e.g. "pwede ba
i-recycle ang tetra pack?") retrieve noticeably worse than English queries
phrased in statute vocabulary — this is expected and is exactly the gap M2
(multilingual BGE-M3) and M4 (agentic query-reformulation loop) are designed
to close, not a bug to fix in M1.

## Roadmap

- **M2** — expand corpus (NSWMC framework, household segregation guides),
  switch to multilingual BGE-M3 dense+sparse, add hybrid search (RRF fusion).
- **M3** — cross-encoder reranking (`bge-reranker-v2-m3`).
- **M4** — agentic corrective-retrieval loop (sufficiency check, bounded
  query reformulation, explicit fallback) + Haiku/Sonnet model routing.
- **M5** — evaluation harness (labeled Q&A set, recall/citation/faithfulness metrics).
- **M6** — FastAPI + Streamlit UI with a "how I found this" transparency panel.
