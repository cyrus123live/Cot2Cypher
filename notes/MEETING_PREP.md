# Meeting Prep — Evidence Ledger & Narrative (for Alex)

Compiled 2026-06-29. Honest state of the evidence and the case for one more semester.
This is the "what do we actually have" document — written to be defensible, not flattering.

## TL;DR

The original thesis — *"CoT distillation beats Neo4j's fine-tuned Gemma baseline"* — is **false**
under a controlled matched baseline (the original comparison was confounded: it used Neo4j's
published adapter, not a matched direct-answer control). But the controlled investigation that
overturned it produced a coherent study with **real positive and negative findings, unified by one
hypothesis (constrained output space)**. The paper pivots from *"CoT helps Text2Cypher"* to
**"What transfers from Text-to-SQL to Text2Cypher, and why."** Not a finished paper; a real
contribution + a clear plan for one more semester.

**Framing for the meeting:** lead with the rigor (a controlled study that found the truth, including
when the truth was negative) and the positive portfolio — not with "the paper isn't done."

---

## A. Defensible POSITIVE results

1. **A stronger SFT recipe.** Our matched direct-answer Gemma beats Neo4j's *published* model by a
   wide margin (GLEU **0.7854 vs 0.5560**). Completion-only loss masking is a confirmed **+0.044
   GLEU** contributor (1c ablation: 0.7854 completion-only vs 0.7415 full-sequence). We have a model
   that beats the published baseline — a positive result in its own right. *(Caveat: masking explains
   only ~31% of the gap to Neo4j's adapter; the rest is open — packing is the next suspect.)*

2. **A confirmed positive transfer — execution-based selection.** MBR / result-clustering selection
   beats string voting on Cypher: **0.2665 exec EM vs 0.2509 (string SC) and 0.2554 (greedy)**. Clean
   confirmation of the constrained-output-space hypothesis *and* the CSC-SQL transfer. String
   diversity fails (nothing to vote across); execution-grounded selection works.

3. **The framework made a correct novel prediction.** Constrained-output-space predicts string
   metrics work for Cypher (one canonical form) but **fail for SQL** (many equivalent forms). We then
   measured SQL with string EM and got a "CoT hurts" result that is **likely a metric artifact** — the
   framework predicted its own measurement pitfall. (Re-scoring with SQL-aware metrics / execution
   accuracy is the live test — see §D.)

4. **Causal mechanism (Test D).** A holistic-path reasoning variant recovers **~40% of the CoT
   penalty** by cutting path fragmentation 5–6×. This *causally localizes* why CoT hurts graph queries
   (decomposition fragments connected patterns), distinguishing it from a vague "CoT is bad" claim.

5. **Oracle anatomy of self-consistency.** Decomposed the SC ceiling: **57% generation-bound / 27%
   selector-solved / 16% verifier-addressable** (77% of the addressable pool is lone-correct). A clean
   diagnostic that bounds what selection methods can buy.

## B. The robust NEGATIVE result (a contribution, not a failure)

CoT distillation does not help text-to-query in a matched strong-SFT pipeline, across:
- **Formalisms:** Cypher (Neo4j + clean ZOGRASCOPE splits), SPARQL (GLEU −0.147), SQL (string metric, caveated).
- **Model families:** Gemma-2-9B and Llama-3.1-8B (Cypher).
- **Trace types:** naive distillation AND execution-verified STaR-style traces.
- **Regimes:** in-distribution, compositional, and length generalization (clean ZOGRASCOPE).

This **overturns the assumed-positive Text-to-SQL CoT literature** (STaR-SQL +18%, etc.) in a
controlled setting. Corrective/negative results that overturn a common assumption are publishable.

## C. The methodological lesson (a genuine contribution)

A confounded baseline (a *published artifact* used as a control instead of a matched in-pipeline one)
produced ~8 months of false "CoT helps" conclusions. **"Always train your own matched baseline before
attributing a delta to your intervention."** This is a real cautionary contribution for the
distillation/eval-methodology literature.

## D. Honest OPEN questions (what next semester buys)

1. **SQL measured correctly.** Re-score existing SQL predictions with SQL-aware metrics
   (`score_sql_semantic.py`), and run **Spider + execution accuracy** (the metric the literature uses).
   Decides whether the SQL "negative" was a metric artifact and whether CoT helps SQL in our pipeline.
   *This is the single most important pre-meeting experiment.*
2. **The +0.14 recipe gap.** Mostly unexplained (masking = 31%); `packing=True` is the next suspect.
3. **Baseline-strength hypothesis.** Does CoT's literature benefit vanish against a strong SFT
   baseline? The unifying explanation for the negative result — a hypothesis, not yet a result.

## E. Candidate paper framings (decide with Alex)

- **(A) Transfer study** — *"What transfers from Text-to-SQL to Text2Cypher."* Positive (execution
  selection, recipe) + negative (CoT, string diversity) under one hypothesis. Findings-strong.
- **(B) Corrective negative + mechanism** — *"CoT distillation does not improve Text2Cypher,"* with the
  causal mechanism (Test D) and the matched-baseline methodology lesson.
- **(C) Compositional prior** *(only if Spider+execution revives it)* — *"CoT helps compositional SQL
  but hurts holistic graph queries."* The top-tier shot, contingent on the SQL execution result.

## F. The ask

- One more semester.
- A framing decision (A / B / C) — gated on the Spider+execution result if we have it by the meeting.
- Compute for: Spider+execution SQL control, the packing ablation, and (if framing A/C) a trained
  verifier / best-of-N to chase the oracle ceiling.

## G. Pre-meeting checklist (in progress)

- [ ] Re-score SQL predictions with SQL-aware metrics (`score_sql_semantic.py`) — needs predictions scp'd back.
- [ ] Spider + execution SQL control — needs the Spider database zip sourced.
- [x] Evidence ledger (this doc).
- [ ] One-slide narrative: original hypothesis → confound found → controlled study → positives + robust negative + framework → plan.
