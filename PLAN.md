# Philippine Recycling Assistant — RAG Portfolio Project

## Context

The user wants a portfolio-quality RAG project that reads as a real, practical system rather than a generic "chat with your PDF" demo. After clarifying requirements, we landed on a **citizen-facing recycling/waste-segregation assistant for the Philippines**, grounded in real public regulatory text (RA 9003 and its IRR, NSWMC/DENR-EMB guidance). The user explicitly wants an **open-source stack for broader platform exposure** (not the Supabase/Vercel tooling already connected to this session) and an **advanced scope**: hybrid search, reranking, and a genuine (not hand-wavy) agentic retrieval mechanism. The working directory is currently empty — this is a greenfield build.

The intended outcome: a working end-to-end pipeline (ingest → chunk → hybrid index → retrieve → rerank → agentic sufficiency check → cited generation) with a lightweight eval harness proving retrieval/answer quality, demoable via CLI first and a simple FastAPI + Streamlit UI later.

## Source verification (do first, don't skip)

Government doc URLs (emb.gov.ph / nswmc.emb.gov.ph paths especially) shift over time. Candidate sources found via search, **to be re-verified via WebSearch/WebFetch at ingestion time** (confirm resolves, capture fetch date + checksum, cache locally — never depend on link staying live):

- RA 9003 full text: `officialgazette.gov.ph/2001/01/26/republic-act-no-9003/`
- RA 9003 PDF (EMB mirror): `nswmc.emb.gov.ph/wp-content/uploads/2025/04/RA-9003.pdf`
- IRR of RA 9003 (DENR DAO 2001-34): `nswmc.emb.gov.ph/wp-content/uploads/2025/04/RA-9003-IRR.pdf`
- NSWMC National Solid Waste Management Framework/Strategy: `nswmc.emb.gov.ph/wp-content/uploads/2017/11/NSWMC-FRAMEWORK-PDF.pdf`
- EMB Waste Segregation Advisory: `emb.gov.ph/waste-segregation-advisory/`

Stretch/optional, not blocking M1–M5: LGU-specific segregation sheets (tag `jurisdiction: lgu:<name>`), EPR Act RA 11898 + IRR.

## Stack (explicit choices — do not substitute)

- **Vector DB:** Qdrant, self-hosted via Docker (hybrid dense+sparse named vectors)
- **Embeddings:** `BAAI/bge-m3` (multilingual — handles English/Filipino/Taglish, 8192 token context, produces dense + sparse from one forward pass). Fallback if dev iteration is too slow on CPU: `intfloat/multilingual-e5-large` (dense) + `rank_bm25`/Qdrant sparse, upgrade to bge-m3 at M2.
- **Reranker:** `BAAI/bge-reranker-v2-m3` (multilingual cross-encoder)
- **Generation:** Claude API — Haiku for cheap routing/sufficiency/rewrite calls, and Haiku by default for the final citizen-facing answer too, escalating to Sonnet only when needed (see Generation section below for escalation rules)
- **Orchestration:** plain Python, no LangChain/LlamaIndex — at this scope (custom RRF fusion, custom agentic loop) a thin hand-written pipeline is more debuggable and a stronger interview artifact than a framework config
- **App layer:** CLI first (fastest end-to-end proof) → FastAPI backend → Streamlit frontend. Design: chat-style main panel (input + history + clickable example questions), structured answer rendering (direct answer / basis citations / hedge / sources), and an expandable "How I found this" debug panel showing pre/post-rerank chunks, query classification, agentic-loop trace (did it reformulate?), and which model answered (Haiku/Sonnet + why) — this panel is what actually demonstrates the advanced pipeline to a reviewer, not just the final answer. Footer disclaimer: not legal advice, verify with official sources.

## Project structure

```
waste-rag/
├── data/{raw,processed}/
├── src/
│   ├── ingestion/          source_manifest.py, fetch.py, parse_pdf.py, parse_html.py
│   ├── chunking/            section_chunker.py
│   ├── embedding/           embedder.py            (BGE-M3 dense+sparse wrapper)
│   ├── indexing/            qdrant_schema.py, build_index.py
│   ├── retrieval/           hybrid_search.py, rerank.py, router.py, agentic_loop.py
│   ├── generation/          prompt_templates.py, generate.py, citation_check.py
│   ├── api/                 main.py, schemas.py     (FastAPI)
│   ├── ui/                  streamlit_app.py
│   └── eval/                qa_set.jsonl, run_eval.py, metrics.py
├── docker-compose.yml        (qdrant service)
├── requirements.txt
└── .env.example              (ANTHROPIC_API_KEY, QDRANT_URL, QDRANT_COLLECTION)
```

## Ingestion & chunking

Parse PDFs with `pdfplumber` (layout-aware, needed to detect section headers), HTML with `requests` + `BeautifulSoup`. Normalize to `{doc_id, raw_text, page_number, source_url}`.

**Section-aware chunking** (the key design decision — Philippine statutes/IRRs have regular structure: `SECTION 48.`, `Rule XVII`, `(a)`/`(b)` subsections):
- Regex-detect section/article/rule headers as hard chunk boundaries; never split a section unless it exceeds the size cap.
- Sub-split long sections at subsection-letter boundaries, keeping parent section number attached.
- Target ~300–500 tokens, hard cap ~700. No generic fixed-size sliding window — it breaks legal citation mapping.
- Prepend a citation header to the *embedded* text (e.g. `"[RA 9003, Sec. 48 — Prohibited Acts] <text>"`) while storing clean original text separately for display — improves embedding quality and makes citation attachment trivial later.

## Qdrant indexing

Collection `ph_recycling_law`, named vectors `dense` (1024-dim cosine) and `sparse` (BGE-M3 lexical-weight, or BM25 fallback). Payload includes `doc_id, doc_type (statute|irr|framework|advisory|lgu_guideline), section_id, section_title, subsection, page_number, source_url, fetch_date, jurisdiction (national|lgu:<name>), language, text, embedded_text`. `doc_type`/`jurisdiction` drive soft-boost filtering in the router, not hard filters (avoid over-pruning recall).

## Retrieval pipeline

1. **Router:** cheap Haiku call classifies query → `segregation_howto | penalty_legal | definition | out_of_scope`. `out_of_scope` short-circuits to a canned redirect (no wasted retrieval/generation, no hallucinated off-topic answers). Classification softly biases `doc_type` relevance.
2. **Hybrid search:** embed query with BGE-M3 (dense+sparse in one call), query both named vectors in Qdrant, fuse with **Reciprocal Rank Fusion** — implement RRF client-side manually (not just relying on a black-box fusion call) so the mechanism is explainable in an interview. Retrieve top ~20.
3. **Rerank:** `bge-reranker-v2-m3` cross-encoder over the top 20, keep top 5–6. Log before/after-rerank recall for the eval report (good "what each stage bought you" story).
4. **Agentic loop — Corrective Retrieval (the one concrete agentic mechanism to build):**
   - Sufficiency check: Haiku call given query + reranked chunks → structured `{sufficient: bool, reason}`.
   - If insufficient (e.g. Taglish slang not matching statute terminology): one query-reformulation call (small in-prompt glossary of common Taglish recycling terms → legal terms helps), re-run steps 2–3 once.
   - **Hard cap: 2 retrieval attempts total.** If still insufficient, generation is told to answer only what's supported and explicitly say "I don't have the exact provision — check [official source]" rather than guess.

## Generation

**Model routing:** default the final answer call to **Haiku**. Escalate to **Sonnet** only when at least one of these holds:
- multiple retrieved provisions conflict or need reconciling
- the question asks for legal interpretation (not just a lookup — e.g. "does this apply to my situation")
- retrieved evidence spans multiple provisions/sections that must be synthesized together
- the sufficiency check (agentic loop) flagged low confidence, even after reformulation

This escalation decision piggybacks on signals already computed earlier in the pipeline (router classification, sufficiency-check result, number of distinct `section_id`s among the top reranked chunks) rather than an extra dedicated call — keeps cost low while reserving the stronger model for genuinely harder cases. Log which model handled each query in the eval report (Section: Evaluation) so the eval can show the cost/quality tradeoff of the routing decision.

System prompt: answer only from numbered context chunks (`[S1]`, `[S2]`...), cite every factual claim inline, **never state a specific penalty/fine/imprisonment term unless verbatim/clearly derivable from a retrieved chunk**, explicitly hedge on anything that varies by LGU (segregation schedules, MRF locations — corpus is national-scope only), plain citizen-friendly tone, match response language to query (English/Filipino).

Answer format: direct answer → "Basis" bullets with `[S#]` citations → optional hedge note → numbered source list with URLs.

**Citation-hallucination guardrail (programmatic, not just prompted):** after generation, regex-extract `[S#]` tokens, verify each exists in the actual chunk set passed to that call; strip/flag any invented citation before returning to the user.

## Evaluation

40–60 hand-labeled Q&A pairs (`eval/qa_set.jsonl`): ~15 segregation how-to, ~15 penalties/legal, ~10 definitions, ~10 Taglish-phrased variants, ~5 adversarial/out-of-scope (tests refusal/hedge behavior). Each item: `question, expected_chunk_ids, expected_answer_keyfacts`.

Metrics: Recall@5/@10 (before vs. after rerank, to show reranker's lift), citation accuracy (programmatic existence check + manual spot-check on ~15 items), answer faithfulness (LLM-as-judge Claude call, RAGAS-style hand-rolled), key-fact coverage (keyword match). Output a markdown/JSON report — a strong README artifact.

## Build phasing (each milestone independently demoable)

- **M1 — Baseline:** ingest RA 9003 + IRR only, section chunking, dense-only retrieval (bge-small or bge-m3 dense-only), plain top-k, no rerank/agentic loop, Claude generation with citations, CLI only. *Goal: prove the core loop works end-to-end.*
- **M2 — Multilingual + hybrid:** expand corpus (NSWMC framework, EMB advisory), switch to BGE-M3 dense+sparse, add sparse named vector, implement RRF fusion.
- **M3 — Reranking:** plug in bge-reranker-v2-m3, add before/after logging.
- **M4 — Agentic loop:** router, sufficiency check, bounded reformulate-and-retry, explicit fallback.
- **M5 — Evaluation harness:** build labeled QA set, implement metrics, run against M1–M4 configs for a comparison report.
- **M6 — UI polish:** FastAPI wrapper, Streamlit frontend (chat panel + structured answer + expandable "How I found this" debug panel per the App/UI design above), Docker Compose bundling Qdrant+API, README with architecture diagram.

## Dependencies

Docker required for local Qdrant (`qdrant/qdrant` image, persistent volume, port 6333).

```
qdrant-client>=1.10  FlagEmbedding  sentence-transformers  torch
anthropic  pdfplumber  pypdf  beautifulsoup4  requests
pydantic  fastapi  uvicorn  streamlit  python-dotenv  tenacity
rank-bm25  pytest
```

## Verification plan

- After M1: run CLI queries against known RA 9003 sections, manually confirm returned citation matches the real section number and text.
- After M3: run the eval set's Recall@5 before/after rerank, confirm measurable improvement.
- After M4: test with intentionally Taglish/slang-phrased queries (e.g. "pwede ba i-recycle yung tetra pack?") to confirm the reformulate-and-retry loop actually fires and improves the result; test an adversarial/out-of-scope query to confirm clean refusal.
- After M5: `run_eval.py` produces the full metrics report; sanity-check a handful of faithfulness scores by hand to confirm the LLM-judge isn't rubber-stamping.
- After M6: `docker-compose up`, run the Streamlit app end-to-end for a full citizen-query demo with visible retrieved chunks/rerank/agentic-loop panel.
