# Evaluation Report

Retrieval recall measured on 30 questions; generation metrics measured on 16 questions (stratified subset, per the scope note in run_eval.py).

## Retrieval: what reranking bought you

| Metric | Pre-rerank (RRF only) | Post-rerank |
|---|---|---|
| Recall@5 | 87% | 80% |
| Recall@10 | 97% | 90% |

## Generation quality

- Average key-fact coverage: 67%
- Citation-clean rate (no hallucinated `[S#]`): 100%
- Average faithfulness (LLM-judge, 1-5): 3.21
- Model usage: {'router_only': 2, 'claude-haiku-4-5-20251001': 13, 'claude-sonnet-5': 1}

## Per-category retrieval recall (post-rerank, Recall@5)

| Category | Recall@5 |
|---|---|
| adversarial | 100% |
| definition | 50% |
| penalty_legal | 88% |
| segregation_howto | 100% |
| taglish | 60% |

## Generation results detail

### seg-1 (segregation_howto) — model=claude-haiku-4-5-20251001
- key-fact coverage: 33% (missing: ['separate container for each type of waste', 'compostable, non-recyclable, recyclable'])
- citation clean: True
- faithfulness: 3 (The answer's main claims about segregating waste into four types and using separate containers are supported by excerpts [1] and [2], but specific details about recyclables, biodegradables, residual waste, Materials Recovery Facilities, composting instructions, and local office procedures are not found in the provided source excerpts.)

### seg-2 (segregation_howto) — model=claude-haiku-4-5-20251001
- key-fact coverage: 67% (missing: ['barangay-owned or -leased land'])
- citation clean: True
- faithfulness: 5 (Every claim in the answer is directly supported by the provided source excerpts, including the definition of MRF, its functions, and location requirements.)

### seg-4 (segregation_howto) — model=claude-haiku-4-5-20251001
- key-fact coverage: 100% (missing: [])
- citation clean: True
- faithfulness: 3 (The answer correctly identifies that Tetra Paks are mentioned as recyclable in excerpt [1], but the claim about "handbag made out of recycled tetrapaks" is not supported by any of the provided excerpts.)

### seg-6 (segregation_howto) — model=claude-haiku-4-5-20251001
- key-fact coverage: 100% (missing: [])
- citation clean: True
- faithfulness: 4 (The answer correctly identifies the three Rs as Reduce, Reuse, and Recycle, which is clearly supported by excerpts [1] and [2]; however, the basis section mentions that Reuse involves "finding creative ways to use items again" which is not explicitly stated in the provided excerpts.)

### pen-1 (penalty_legal) — model=claude-haiku-4-5-20251001
- key-fact coverage: 40% (missing: ['P300', 'P1,000', '1 day'])
- citation clean: True
- faithfulness: 5 (The answer's claims about fines (₱300 to ₱1,000) and community service (1-15 days) for littering are directly supported by excerpt [1], which explicitly lists these penalties for littering in public places.)

### pen-2 (penalty_legal) — model=claude-haiku-4-5-20251001
- key-fact coverage: 100% (missing: [])
- citation clean: True
- faithfulness: 1 (The answer claims 16 prohibited acts but the provided excerpts only clearly show 4 acts (littering, operating without permits, open burning, and non-segregated collection), with the rest being unsupported hallucinations not found in the source material.)

### pen-4 (penalty_legal) — model=claude-haiku-4-5-20251001
- key-fact coverage: 100% (missing: [])
- citation clean: True
- faithfulness: 3 (The main penalty claim (fine P10,000-P200,000 or 30 days to 3 years imprisonment) is clearly supported by excerpt [2], but the claim about alien deportation is not found in any of the provided excerpts.)

### pen-6 (penalty_legal) — model=claude-haiku-4-5-20251001
- key-fact coverage: 50% (missing: ['prohibited'])
- citation clean: True
- faithfulness: 2 (The answer claims open burning is illegal and cites specific penalties, but the provided source excerpts do not explicitly mention "open burning" as a prohibited act or list penalties for it.)

### def-1 (definition) — model=claude-haiku-4-5-20251001
- key-fact coverage: 100% (missing: [])
- citation clean: True
- faithfulness: 5 (Every claim in the answer is directly supported by the source excerpts, particularly excerpt [2] which provides the legal definition and examples, and excerpt [1] which gives practical examples of recyclable materials.)

### def-2 (definition) — model=claude-haiku-4-5-20251001
- key-fact coverage: 0% (missing: ['treating used or waste materials', 'beneficial use', 'new products'])
- citation clean: True
- faithfulness: 3 (The direct answer is mostly supported by the excerpts, but the second sentence about "transporting them to processing centers" and "sorted, cleaned, and prepared for reuse" goes beyond what is explicitly stated in the provided source excerpts.)

### def-5 (definition) — model=claude-haiku-4-5-20251001
- key-fact coverage: 67% (missing: ['implementation and enforcement'])
- citation clean: True
- faithfulness: 4 (Most claims are well-supported by the excerpts, though the answer references details about municipal/city responsibilities for non-recyclable and special wastes that are only partially stated in the incomplete source excerpt [1].)

### tgl-1 (taglish) — model=claude-haiku-4-5-20251001
- key-fact coverage: 100% (missing: [])
- citation clean: True
- faithfulness: 1 (The generated answer claims specific penalties (P300-P1,000 and jail terms) that are not supported by any of the provided source excerpts, which only mention penalties for unsegregated waste collection, dumping, and toxic waste importation.)

### tgl-2 (taglish) — model=claude-sonnet-5
- key-fact coverage: 67% (missing: ['compostable'])
- citation clean: True
- faithfulness: 2 (The answer makes specific claims about four waste categories and provides detailed segregation instructions that are not supported by the provided source excerpts, which only mention segregation at source as a general principle without detailing the four specific categories.)

### tgl-3 (taglish) — model=claude-haiku-4-5-20251001
- key-fact coverage: 100% (missing: [])
- citation clean: True
- faithfulness: 4 (The main claim that tetrapak can be recycled is clearly supported by excerpt [1] which explicitly mentions "Tetrapak" under recyclables with the instruction to "Flatten Tetrapak and paper boxes to reduce space," though the answer includes additional claims about handbag examples that are not found in the provided excerpts.)

### adv-1 (adversarial) — model=router_only
- key-fact coverage: 50% (missing: ['out of scope'])
- citation clean: True
- faithfulness: None (n/a (out of scope, no generation call))

### adv-2 (adversarial) — model=router_only
- key-fact coverage: 0% (missing: ['not legal advice', 'official source', 'local'])
- citation clean: True
- faithfulness: None (n/a (out of scope, no generation call))
