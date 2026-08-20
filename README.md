# PH Recycling Assistant (RAG)

A citizen-facing recycling/waste-segregation assistant for the Philippines,
grounded in real public regulatory text: RA 9003 (Ecological Solid Waste
Management Act of 2000) and its Implementing Rules and Regulations (DENR DAO
2001-34). Every answer cites the specific section it came from.

Full architecture/design plan: see `PLAN.md` (or ask for it — it covers the
hybrid search, reranking, and agentic-loop design planned for M2-M4).

## Status: M5 (evaluation harness) complete

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
- Agentic loop (`src/retrieval/agentic_loop.py`): a Haiku tool-call router
  classifies each query (`segregation_howto` / `penalty_legal` / `definition`
  / `out_of_scope`) — out-of-scope questions short-circuit to a canned
  redirect before any retrieval or generation cost, and in-scope
  classifications softly boost matching `doc_type`s post-rerank (never a
  hard filter). A sufficiency-check tool-call then judges whether the
  reranked chunks actually answer the question; if not, one reformulation
  call rewrites the query into statute vocabulary and retrieval runs once
  more (hard-capped at 2 attempts total, then falls back to an explicit
  "I don't have this" hedge rather than guessing).
- Model routing: the same sufficiency-check call also flags whether the
  final answer needs escalation (conflicting/multi-provision evidence, a
  legal-interpretation question, or sufficiency still failing after retry) —
  Haiku by default, Sonnet only when flagged. No extra dedicated call for
  this; it reuses the sufficiency check's own judgment.
- Generation: Claude Haiku/Sonnet (per the routing above) with inline
  `[S#]` citations and the anti-hallucination guardrail; a low-confidence
  note is appended to the prompt when the final sufficiency check still
  failed, reinforcing (not replacing) the system prompt's hedge rule.

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
python -m src.cli ask "Can I recycle a Tetra Pak?" --debug   # --debug shows the full agentic trace
python -m src.cli ask "..." --no-agent                       # bypass the loop (M3 behavior only)
```

## Run the evaluation

```bash
python -m src.eval.run_eval
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

## M4 validated: reformulation fixes what reranking alone couldn't

Re-testing the exact Taglish query that regressed under reranking alone
("Magkano ang multa sa pagkalat ng basura?"):

1. Router classifies it `penalty_legal`.
2. Attempt 1 (original query): sufficiency check correctly says no — the
   littering-specific penalty isn't in the top chunks.
3. Reformulation rewrites it to *"What are the fines and penalties imposed
   for littering and throwing of waste matters in public places?"* — close
   to RA 9003 Sec. 48's actual statute language.
4. Attempt 2 (reformulated query): rerank scores jump to sharp, confident
   values (1.14, 1.03, 1.02, 0.99) — the same quality signature M3 only saw
   on clean English queries — and the correct IRR fines section lands #1.
5. The sufficiency check still flags mild doubt (an excerpt-preview
   truncation artifact, since fixed by widening the preview from 300→800
   chars), which correctly triggers escalation to Sonnet. The final answer
   is accurate, in Filipino (matching the query's language), cites the right
   sections, and appropriately hedges that current fine amounts may have
   risen with RA 9003's built-in inflation adjustment.

Two other cases confirm the loop isn't just reformulating everything: the
Tetra Pak query is judged sufficient on the *first* attempt (the M2 corpus
expansion already put an explicit "Tetra Pak" mention in reach of a sharp
0.58 rerank score) and stays on cheap Haiku with no wasted second call; an
out-of-scope query ("best pizza topping") is caught by the router and
redirected before any retrieval or generation cost at all.

**Bug found and fixed along the way:** `sentence-transformers`/`FlagEmbedding`
auto-selected Apple's MPS GPU backend on this machine, and the loop's second
retrieval attempt (embed + rerank running twice in one process) exhausted
MPS memory and crashed. Both `embedder.py` and `rerank.py` now force CPU
explicitly — slower, but it doesn't have that failure mode, which matters
more here.

## M5: evaluation harness — measurement overturned an M3 conclusion

`src/eval/qa_set.jsonl` — 30 hand-labeled questions (8 segregation, 8
penalty/legal, 6 definition, 5 Taglish, 3 adversarial), gold `expected_chunk_ids`
taken from the real indexed corpus (not guessed). Run: `python -m
src.eval.run_eval` → `src/eval/results/eval_report.md` + `.json`.

Scope note: retrieval recall (no Claude calls) runs on all 30 questions.
Citation/faithfulness/key-fact metrics require the full agentic pipeline
(several chained Claude calls per question), so they run on a 16-question
stratified subset — sized to match the plan's own "~15 items" scale for
manual citation spot-checks, to keep runtime/cost reasonable for a portfolio
project rather than a production eval suite.

**Headline result — and it contradicts the M3 finding above:**

| Metric | Pre-rerank (RRF only) | Post-rerank |
|---|---|---|
| Recall@5 | 87% | **80%** |
| Recall@10 | 97% | 90% |

M3's manual spot-check (2-3 hand-picked queries) found reranking sharply
improved English precision. That's still true and reproducible on those
specific queries. But measured systematically across 30 questions, reranking
*costs* 6.7 points of Recall@5 overall. This is exactly why a real eval
harness matters more than anecdotal spot-checks — a couple of good examples
generalized into a conclusion ("M3 fixes precision") that the fuller picture
doesn't support.

Investigated rather than taken at face value — for `def-4` ("What is the
declared policy behind RA 9003?"), pre-rerank correctly puts the exact right
chunk (`ra9003-ChapterIArticle1-sec2`, Declaration of Policies) at **#1**.
Reranking demotes it **out of the top 5 entirely**, replaced by a references/
bibliography page and forewords. Same pattern on `def-2`. The reranker
appears to favor generically-topical guidebook prose over precise legal
citations on definitional questions — plausibly because `bge-reranker-v2-m3`
isn't fine-tuned for legal-precision distinctions between passages at
different specificity levels talking about the same general topic. Combined
with the already-documented Taglish miscalibration (`tgl-1` reproduces that
finding again here), reranking has two identified failure modes, not one.

**Action taken on this evidence:** since Recall@10 post-rerank (90%) is much
closer to full recall than Recall@5 (80%), the right chunk is usually just
outside the old top-5 cutoff rather than truly lost. Default `top_k` across
`cli.py`/`hybrid_search.py`/`agentic_loop.py` is bumped from 5 to 8 to
capture more of that margin before generation. Not re-validated with a full
eval re-run this session (would cost another full pass of Claude calls) —
flagged honestly as a plausible improvement backed by this data, not a
confirmed one.

**A second finding, about the eval metrics themselves:** `pen-2` ("What acts
are prohibited under RA 9003?") scored **100% key-fact coverage** but
**faithfulness = 1/5** — the model claimed "16 prohibited acts" when the
retrieved excerpts only actually supported about 4. Keyword-based key-fact
coverage is gameable by fluent hallucination that happens to use the right
vocabulary; it isn't a substitute for the LLM-judge faithfulness check, only
a complement to it. Worth stating plainly rather than only reporting the
metric that looked good.

**Other real numbers from this run:** citation-clean rate 100% (the
anti-hallucination guardrail from M1 never had to strip anything across all
16 generation questions), average faithfulness 3.21/5, average key-fact
coverage 67%, model usage 13 Haiku / 1 Sonnet / 2 router-only-redirect out of
16. Full per-question detail (including the two adversarial questions and
their answers) is in `src/eval/results/eval_report.md`.

**Bug found and fixed along the way:** the local Python environment's
`transformers`/`huggingface-hub` install became corrupted mid-session
(missing submodules, mismatched pinned versions from earlier back-and-forth
dependency changes) — unrelated to any code in this repo, but it blocked
investigation until reinstalled cleanly.

## Roadmap

- **M6** — FastAPI + Streamlit UI with a "how I found this" transparency panel.
