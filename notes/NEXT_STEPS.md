# Next Steps — Pre-Writeup Loose Ends + Positive-Angle Plan

Compiled 2026-06-27. Three work items: two cheap calibration loose ends that make the
absolute numbers bulletproof (neither can change the negative-result conclusion), plus
scoping for the optional positive angle (execution-grounded selection → trained verifier).

Priority order: **1 and 2 before writeup; 3 is a go/no-go scoping task for the Alex conversation.**

---

## Item 1 — Explain the +0.14 harness gap (completion-only masking hypothesis)

**Why:** Our matched direct-answer Gemma (A5, GLEU 0.7854) beats Neo4j's *published* adapter
(0.5560) by +0.23, and beats their adapter under *our identical inference* (A2, 0.6455) by
+0.14. "Same QLoRA config" should not move GLEU by 0.14. Until we reproduce 0.5560 or explain
the gap, the harness is uncalibrated against the published leaderboard and a reviewer can
attack every absolute number. Leading suspect: **completion-only loss masking** (we mask the
prompt via `DataCollatorForCompletionOnlyLM`, compute loss only on the answer; Neo4j may train
on the full sequence).

- [x] **1a. Confirm Neo4j's loss masking. — DONE (2026-06-27).** Verified from
      `neo4j-labs/text2cypher` finetuning notebooks (StarCoder2_3B, CodeLlama_13B): their
      `SFTTrainer` is instantiated with `dataset_text_field="text"` and **`packing=True`**, and
      **no `DataCollatorForCompletionOnlyLM` / `response_template`** — i.e. **full-sequence loss**
      (loss on every token, the schema-heavy system message included). The Gemma-2024v1 model
      card lists an `SFTConfig` with **no data collator named**, consistent with the same recipe
      (exact Gemma notebook not public → strong inference, not a direct read). Our pipeline masks
      the prompt (`DataCollatorForCompletionOnlyLM`), so the configs differ in the one place that
      matters most. With a ~30:1 schema:answer token ratio, full-sequence loss spends ~97% of the
      gradient reconstructing schemas — a plausible mechanism for the +0.14 GLEU training gap.
- [~] **1b. Reproduce the published 0.5560. — SUBMITTED ON FIR (2026-08-20), awaiting result.**
      Implemented as `scripts/drac_neo4j_repro_eval.sh`: the published adapter through
      `drac_inference.py` with `--no-cot-prompt --max-length 1600` (right-truncation, greedy,
      4-bit — identical to A2 except the truncation). Adapter provided via git-lfs clone at
      `~/scratch/neo4j_published_adapter/` (the hf_hub route hit a CVMFS Errno-5 on the login
      node; the job now accepts either). If it lands near 0.5560 → harness calibrated to the
      leaderboard scale; if not → residual is framework/version, stays documented.
- [x] **1c. Ablate completion-only masking in OUR pipeline. — DONE (ran on Fir).** Result:
      full-sequence GLEU **0.7415** vs A5 completion-only **0.7854** → masking accounts for
      **+0.044** of the +0.14 training gap (~31%). Masking is a real, causal contributor but not
      the whole story → packing became the next suspect (1e). In the paper as the recipe
      paragraph's controlled ablation.
- [~] **1e. Packing ablation (Alex-approved). — SUBMITTED ON FIR (2026-08-20), train→eval chained
      via --dependency, awaiting result.**
      `--packing` flag on the A5 trainer (TRL `packing=True`, full-sequence loss, no collator —
      Neo4j's exact published path); wrappers `drac_train_gemma_packing.sh` /
      `drac_gemma_packing_eval.sh`. Ladder: A5 0.7854 (completion-only) → 1c 0.7415 (full-seq)
      → packing this run → A2 0.6455 (published adapter). If packing lands near 0.6455, the
      +0.14 training gap is FULLY explained (masking +0.044 + packing) and the paper's
      "remainder unexplained" caveat retires; if not, the residual stays flagged.
- [ ] **1d. Write the verdict.** Two honest outcomes, both fine for the paper:
      (i) we reproduce 0.5560 → harness calibrated, the +0.14 is a real **stronger-SFT-recipe**
      contribution (completion-only masking) worth a paragraph; or (ii) we don't → document the
      residual as a framework/version artifact and stop trusting cross-harness absolute compares.

---

## Item 2 — Complete the Gemma direct-answer execution EM (A5)

**Why:** The matched-pipeline exec-EM row is the only blank cell in EXPERIMENT_LOG §A. We have
string EM showing CoT hurts (A5 0.4331 vs A3 0.3799); exec EM should confirm it on the execution
metric too. Predictions already exist — this is one eval run, no inference needed.

- [x] **2a. Run execution eval — DONE (2026-07-27).** `eval_execution.py` over
      `results/predictions_gemma_baseline_greedy.jsonl` (demo.neo4jlabs.com, 2,471 DB-eligible).
- [x] **2b. Result: A5 exec EM 0.2975 (735/2471), 100 pred errors** — vs A3 CoT 0.2554 / 114
      errors. CoT effect −0.0421; direct also makes *fewer* invalid queries. Caveat: this run saw
      51 reference errors vs 166 in the earlier A2/A3 runs (live-DB drift since June).
- [x] **2c. `EXPERIMENT_LOG.md` updated** — A5 cell filled, decomposition completed
      (GLEU/StrEM/ExecEM triple, all negative), bottom-line table now 7/7 clean negatives.

---

## Item 3 — Scope the positive angle: execution-grounded selection → trained verifier

**Why:** The one positive transfer finding (MBR execution-result voting, B2 = 0.2665, beats
greedy and string voting) has a large oracle ceiling: **MBR 0.27 → oracle 0.43**. A trained
verifier / best-of-N (the CSC-SQL / STaR-ORM direction) could chase that gap and pair a positive
contribution with the negative result — aiming higher than a negative-results venue. This item is
**planning only**: produce a scoped plan + cost estimate for a go/no-go decision with Alex.

- [ ] **3a. Anatomize the oracle gap** (analysis, no training). For the SC@5 candidates, partition
      the 0.43 oracle cases: (i) MBR already picks the correct result cluster, (ii) correct cluster
      exists but is a non-plurality minority (→ selection problem, a verifier helps), (iii) no
      candidate is correct (→ candidate-generation problem, need bigger N / better base model).
      The (ii):(iii) ratio decides whether to invest in a *selector* or in *generation*.
- [ ] **3b. Decide the base model for candidates.** Current SC@5 candidates come from the **CoT**
      model. The paper's conclusion is direct > CoT, so candidates should likely come from the
      **direct-answer Gemma (A5)** instead. Plan a candidate-regeneration run from A5 (reuse the
      SC harness) so the verifier sits on the stronger model.
- [ ] **3c. Choose the selection method.** Trained outcome-reward verifier / best-of-N
      (STaR-SQL ORM@N, CSC-SQL) vs. cheaper unsupervised options (Universal Self-Consistency,
      larger N, execution-feature reranking). Define the verifier's input signal (execution
      result features, row counts, schema-linking overlap) and training data (label candidates by
      reference-result match — `eval_execution_selection.py` already produces these labels).
- [ ] **3d. Estimate compute/cost + reusable assets.** `eval_execution_selection.py` (MBR) and the
      saved SC@5 candidates are reusable; scope the delta for candidate regen from A5 + verifier
      training. Produce a one-paragraph cost/time estimate.
- [ ] **3e. Go/no-go with Alex.** Decide: ship the mechanism paper with this as a "future work"
      paragraph, OR add it as a real experiment arm for a main-track positive contribution.

---

### Sequencing
1 and 2 are independent and can run in parallel (1 needs GPU/training; 2 is a local eval).
3 is desk analysis (3a–3d) feeding the Alex conversation (3e); 3a only needs the existing
SC@5 candidate files — do it before deciding to spend any compute.
