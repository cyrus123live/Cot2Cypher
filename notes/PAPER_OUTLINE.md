# A+B Paper Outline — the earned floor (per Alex, 2026-07-03)

**One paper: the transfer study (A) with the matched-baseline negative result + Test D
mechanism (B) as its core.** Written now; depends on no pending experiment. C (compositional
prior) is a **stretch gated on the Spider+execution result** — do NOT write toward it until it
lands. Everything below is supported by `notes/EXPERIMENT_LOG.md`.

LaTeX scaffold: `paper/transfer_study.tex` (reuses `paper/paper.tex` preamble + `references.bib`).
Old paper `paper/paper.tex` kept as archive (its Related Work / Method prose is largely reusable).

## Title (working)

Primary: **"What Transfers from Text-to-SQL to Text-to-Cypher? A Controlled Study of Reasoning,
Selection, and Training Recipe."**
Alt: "When Reasoning Hurts: Chain-of-Thought Does Not Transfer to Text-to-Cypher."

## Thesis / spine

Text-to-SQL has a large toolbox (CoT distillation, self-consistency, execution selection, RL,
schema filtering). **Which transfer to Text-to-Cypher, and why?** We answer with a controlled,
matched-pipeline study unified by the **constrained-output-space hypothesis**: Cypher has few
semantically-equivalent surface forms (~one canonical form) and its graph pattern is a single
*connected* object, whereas SQL is *compositional* with many equivalent forms. This predicts:
- **P1** — string/diversity methods fail (nothing to vote across).
- **P2** — execution-grounded methods (fix/select on results) still transfer.
- **P3** — CoT/decomposition hurts (decomposition fragments a connected pattern).

## Contributions (reframed)

1. A **controlled matched-baseline** result: CoT distillation — the highest-profile SQL transfer —
   does **not** improve Text-to-Cypher and modestly **hurts** it, across two model families, naive
   and execution-verified traces, and in-distribution / compositional / length-generalization
   regimes. Prior apparent gains were a stronger SFT recipe + a leaked benchmark split.
2. A **causal mechanism** (Test D): CoT's decompositional bias fragments connected graph patterns;
   a holistic-reasoning ablation recovers ~40% of the penalty by cutting fragmentation 5–6×.
3. A **positive transfer**: execution-grounded (MBR) selection beats string voting (+1.1pp), while
   string self-consistency fails — a clean confirmation of the constrained-output-space hypothesis.
4. A **methodological lesson**: a published artifact used as a control (instead of a matched
   in-pipeline baseline) produced months of false "CoT helps" conclusions. Train your own baseline.

## Section-by-section (claim → supporting evidence in EXPERIMENT_LOG)

1. **Introduction** — the transfer question; the constrained-output-space hypothesis; the four
   contributions. Frame the negative result as overturning an assumed-positive SQL transfer.
2. **Related Work** — reuse `paper.tex` §Related Work (Text2Cypher; CoT-for-SQL incl. STaR-SQL,
   Tai et al.; RL; selection; benchmarks). Reframe: these are the SQL toolbox we test for transfer.
3. **The Constrained-Output-Space Hypothesis** — formalize Cypher (connected/canonical) vs SQL
   (compositional/many-forms); state P1–P3. This is the paper's organizing idea.
4. **Matched Experimental Protocol** — same base (Gemma-2-9B-it, also Llama-3.1-8B), same QLoRA
   config, **completion-only masking**, CoT distillation via gpt-oss-120b; **only the training
   target differs** (direct query vs reasoning+query). Applied across Cypher / SPARQL / SQL.
   *This is the methodological backbone — foreground it.*
5. **Results**
   - 5.1 **Matched negative (Cypher)** — Table 1 (§A: A5 vs A3; A6 vs A8). CoT hurts both families.
     Decomposition of the old "+0.1227": +0.1399 pipeline − 0.017 CoT (Table 2).
   - 5.2 **Robustness** — leakage audit (§C, §H): unseen-only split still negative. STaR
     execution-verified CoT (§J) still negative. Clean ZOGRASCOPE IID/comp/length (§I) — Table 3,
     direct wins every split incl. length (kills the old "length SOTA").
   - 5.3 **Cross-formalism** — Table 6: Cypher −0.017 GLEU, SPARQL −0.147 GLEU, SQL(gretel)
     −0.048 canon EM. **SQL execution-accuracy control (Spider) is PENDING** — report as an open
     item; do NOT claim the compositional-prior direction here (that is C).
   - 5.4 **Positive transfers** — Table 5 (§B): greedy 0.2554 / string-SC 0.2509 / MBR 0.2665 /
     oracle 0.4302. Recipe: completion-only masking (1c) → stronger direct baseline than published.
6. **Mechanism: why CoT hurts Cypher** — Table 4 (ladder: direct / QDecomp / Holistic / Enum,
   §MECHANISM Test D + E4). Fragmentation (causally fixed by holistic), truncation (planning-stage,
   E1; resists enumeration, E4), residual reasoning cost. Bounded and causal.
7. **Discussion** — the constrained-output-space account of P1–P3; the methodological lesson; scope
   (why SQL's CoT gains may not transfer). One paragraph flagging the Spider+execution control as
   the test that would extend this to a compositional-prior claim (C) — as future work, not a claim.
8. **Limitations** — string-metric caveat for SQL (why Spider+execution is needed); single teacher;
   2024-dataset leakage (which we turn into evidence, §C); approximate execution comparison.
9. **Conclusion.**

## Key tables (numbers from EXPERIMENT_LOG — all in the scaffold)

- **T1 Matched CoT effect (Neo4j):** direct vs CoT, Gemma + Llama (GLEU / String EM / Exec EM).
- **T2 Decomposition** of the reported +0.1227 (pipeline vs CoT).
- **T3 Clean ZOGRASCOPE** exec accuracy (IID / comp / length), direct vs CoT.
- **T4 Reasoning-format ladder** (direct / QDecomp / holistic / enum) × 3 splits.
- **T5 Selection** (greedy / string-SC / MBR / oracle).
- **T6 Cross-formalism CoT delta** (Cypher / SPARQL / SQL; Spider row = pending).

## Do-not-do (per Alex)

- Do not write toward C (compositional prior) until Spider+execution lands.
- Do not resurrect the leaderboard/#4 or the ZOGRASCOPE-SOTA claims (overturned).
- Keep the two-arm comparison matched everywhere; never compare against a published artifact as a control.
