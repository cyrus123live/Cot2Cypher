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
- [ ] **1b. Reproduce the published 0.5560.** Run Neo4j's adapter through `run_neo4j.py` with
      inference **truncated at `max_seq_len=1600`** + exact model-card prompt (no system prompt).
      A2 (0.6455) used full schema / `max_length=7680`; the addendum already shows >1600-token
      prompts score lower, so truncation should pull our number toward 0.5560. If it lands near
      0.5560 → harness is on the leaderboard scale (good). If not → investigate framework/version.
- [~] **1c. Ablate completion-only masking in OUR pipeline. — SCRIPTS BUILT (2026-06-27), pending
      cluster run.** Implemented as a single `--full-sequence` flag on the A5 baseline trainer
      (`scripts/drac_train_gemma_baseline.py`), so the ONLY variable vs A5 is the loss mask.
      Submission wrappers: `scripts/drac_train_gemma_fullseq.sh` (train → `~/scratch/gemma_fullseq_adapter/`)
      and `scripts/drac_gemma_fullseq_eval.sh` (eval with matched `--no-cot-prompt` →
      `~/scratch/results_gemma_fullseq/`). **Run on DRAC (Fir):**
      `sbatch scripts/drac_train_gemma_fullseq.sh` then `sbatch scripts/drac_gemma_fullseq_eval.sh`
      (sync `scripts/` + `data/cot_training_data.jsonl` to `~/scratch/thesis/` first).
      **Interpretation:** A5 completion-only = GLEU 0.7854 / EM 0.4331. If 1c full-sequence drops
      toward ~0.64, completion-only masking *is* the +0.14 driver → gap fully explained, headline
      was a stronger SFT recipe (not CoT). If it stays high, masking is not the cause and we keep
      digging (packing, framework/version).
- [ ] **1d. Write the verdict.** Two honest outcomes, both fine for the paper:
      (i) we reproduce 0.5560 → harness calibrated, the +0.14 is a real **stronger-SFT-recipe**
      contribution (completion-only masking) worth a paragraph; or (ii) we don't → document the
      residual as a framework/version artifact and stop trusting cross-harness absolute compares.

---

## Item 2 — Complete the Gemma direct-answer execution EM (A5)

**Why:** The matched-pipeline exec-EM row is the only blank cell in EXPERIMENT_LOG §A. We have
string EM showing CoT hurts (A5 0.4331 vs A3 0.3799); exec EM should confirm it on the execution
metric too. Predictions already exist — this is one eval run, no inference needed.

- [ ] **2a. Run execution eval** on the existing predictions:
      `eval_execution.py` over `results/predictions_gemma_baseline_greedy.jsonl`
      (demo.neo4jlabs.com, same 2,471 DB-eligible / 2,305-valid-ref methodology as A3).
- [ ] **2b. Record** exec EM + pred-error count; compare to A3 CoT (0.2554, 114 errors).
- [ ] **2c. Update `EXPERIMENT_LOG.md`** — fill the A5 exec-EM cell and the matched-pipeline
      decomposition (the A5→A3 CoT effect on exec EM, completing the GLEU/StrEM/ExecEM triple).

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
