# Complete Experiment Log

Compiled 2026-06-20 (updated 2026-06-27). Every experiment run, with results and current validity status.

**Update 2026-06-27 — reasoning-format ablations + mechanism (see notes/MECHANISM_ANALYSIS.md):**
Clean ZOGRASCOPE ablation ladder (execution accuracy, length / iid / comp):
direct 0.475/0.826/0.661 · QDecomp-CoT 0.258/0.609/0.418 · Holistic-CoT (Test D)
0.303/0.710/0.511 · Enum-CoT (E4) 0.231/0.669/0.429. Holistic (minimal connected-
path reasoning) is the best CoT variant but still trails direct by 12–17pp; no
reasoning-format intervention closes the gap. Mechanism: NOT truncation (alone
explains 2%); the robust CoT-specific culprit is filter-value corruption during
reasoning (4× direct, 92% corrupted in the trace), within a broader error-
accumulation effect on what is largely a copy-and-place task (more reasoning →
strictly worse). Differs from SQL because SQL's clean CoT-SFT gain is only ~+6pp
(rest is verifiers/RL), decomposition fits SQL's compositional structure but
fragments Cypher's connected patterns, and Cypher's canonical output space leaves
nothing for diversity/selection methods to exploit.

**Reading guide for validity tags:**
- ✅ **VALID** — clean, controlled, trustworthy.
- ⚠️ **CONFOUNDED** — result is real but the comparison mixes variables (e.g. our pipeline vs Neo4j's published pipeline); cannot attribute a delta to the intended cause.
- ❌ **INVALID** — leakage or a bug makes the absolute numbers untrustworthy.

Benchmarks: **Neo4j Text2Cypher 2024** test set (n=4,833; execution subset n=2,471 DB-eligible). **ZOGRASCOPE** Pole graph (IID/comp/length splits).

---

## A. Neo4j benchmark — translation + execution metrics

| # | Model / config | Pipeline | GLEU | String EM | Exec EM | Validity |
|---|----------------|----------|:----:|:---------:|:-------:|----------|
| A1 | Neo4j published Gemma adapter (their numbers) | Neo4j | 0.5560 | — | 0.2104 | reference |
| A2 | Neo4j Gemma adapter via OUR inference (greedy, full schema) | mixed | 0.6455 | 0.1924 | 0.1862 (460/2471) | ⚠️ used as "baseline" for whole project |
| A3 | **Gemma + CoT** | ours | 0.7682 | 0.3799 | 0.2554 (631/2471) | ⚠️ vs A2 confounded |
| A4 | Gemma + CoT, **latent** (CoT adapter + baseline prompt) | ours | 0.7082 | 0.2197 | 0.2153 (532/2471) | ✅ vs A3 (ablation) |
| A5 | **Gemma direct-answer baseline (the missing control)** | ours | **0.7854** | **0.4331** | *pending* | ✅ matched to A3 |
| A6 | **Llama-3.1-8B direct-answer baseline** (matched prompt) | ours | 0.7680 | 0.4223 | 0.2865 (708/2471) | ✅ matched to A8 |
| A7 | Llama-3.1-8B direct-answer baseline (mismatched CoT prompt) | ours | 0.7024 | 0.3888 | — | ⚠️ prompt mismatch |
| A8 | **Llama-3.1-8B + CoT** | ours | 0.7416 | 0.3000 | 0.2161 (534/2471) | ✅ matched to A6 |
| A9 | Llama published baseline (their numbers) | Neo4j | 0.6470 | — | 0.2299 | reference |

**The headline decompositions (matched-pipeline = only training target differs):**
- **Gemma CoT effect (A5→A3):** GLEU **−0.0172**, String EM **−0.0532** — CoT *hurts*.
- **Llama CoT effect (A6→A8):** GLEU **−0.0264**, String EM **−0.1223**, Exec EM **−0.0704** — CoT *hurts*.
- The reported "+0.1227 GLEU from CoT" (A2→A3) was **+0.1399 pipeline** (A2→A5) **−0.0172 CoT** (A5→A3). The gain was the pipeline; CoT subtracted.

---

## B. Inference-time techniques (Neo4j, on the CoT model)

| # | Technique | Metric | Result | Validity |
|---|-----------|--------|--------|----------|
| B1 | Self-consistency SC@5 (T=0.7), **string** majority vote | GLEU / String EM / Exec EM | 0.7634 / 0.3886 / 0.2509 (620/2471) | ✅ |
| B2 | SC@5, **execution-result** voting (MBR) | Exec EM | **0.2665 (645/2420)** | ✅ |
| B3 | Oracle (any of 5 candidates matches reference) | Exec EM ceiling | 0.4302 | ✅ |

**Finding:** string voting (B1, 0.2509) falls *below* greedy CoT (A3, 0.2554); execution-result voting (B2, 0.2665) rises *above* both. Confirms the constrained-output-space hypothesis. Oracle 0.43 = headroom for a verifier.

---

## C. Seen/unseen leakage split (Neo4j; string EM)

Neo4j 2024 has 31.7% of test questions appearing verbatim in train (0 instance-ID overlap, but adversarial — often a *different* gold Cypher). Split of A5 (direct) vs A3 (CoT):

| Subset | Direct-answer (A5) | CoT (A3) |
|--------|:---:|:---:|
| SEEN (31.7%, leaked) | 0.1667 | 0.1242 |
| **UNSEEN (68.3%, genuine)** | **0.5861** | 0.5180 |
| Overall | 0.4533 | 0.3933 |

**On genuinely-unseen questions, direct-answer beats CoT by +0.068 EM** (per-instance 307 direct-only vs 82 CoT-only wins, 3.7:1). Kills the "leakage-rescue" hypothesis — CoT loses *most* where leakage is absent.

---

## D. Ablation: latent vs active reasoning (Neo4j)

| Config | GLEU | String EM | Exec EM | Recovery of A2→A3 gain |
|--------|:----:|:---------:|:-------:|:----:|
| Baseline (A2) | 0.6455 | 0.1924 | 0.1862 | — |
| Latent (A4) | 0.7082 | 0.2197 | 0.2153 | GLEU 51% / Exec 42% / StrEM 15% |
| Active (A3) | 0.7682 | 0.3799 | 0.2554 | remainder |

*(Note: this ablation is internally valid, but its "baseline" is A2, the confounded one. Re-anchoring to A5 is pending.)*

---

## E. Mechanistic analyses (Neo4j) — now reinterpreted

All compared CoT (A3) against the **confounded** baseline (A2), so they describe differences between *those two models*, not causal effects of CoT:
- Schema-ambiguity penalty: baseline 4.6pp drop on "correct relationship type" under ambiguous schema → CoT 0pp. (1,119 1-hop queries.)
- 1-hop paradox: baseline worst on simplest traversals (2.3% EM 1-hop vs 19.7% 2-hop).
- Graph-reasoning taxonomy: per-feature CoT deltas (UNION +0.52, var-length +0.32, ORDER BY +0.09).
- Complexity scaling: CoT delta +0.15 (short) → +0.63 (500+ char).

**Status: cannot be attributed to CoT** until re-run against A5. May still describe A2-vs-A3 differences.

---

## F. ZOGRASCOPE — ❌ ALL INVALID (train/test leakage)

We trained on `train_v1 + length_train_v1` **merged** and tested on the combined test set. This cross-contaminated: `train_v1` ∩ `length_test` = 555, `length_train` ∩ `test_v1` = 1419 → **58.6% instance-ID leakage** (44% of length-test, 67% of main-test were in training).

| Config | IID | Comp | Length | Validity |
|--------|:---:|:---:|:------:|----------|
| F1 CoT zero-shot (no Pole training) | 14.45% | 6.23% | 3.35% | ⚠️ (vs paper baselines, no matched control) |
| F2 CoT fine-tuned ("SOTA on length") | 64.71% | 53.08% | 32.24% | ❌ leaked |
| F3 Direct-answer fine-tuned | 87.89% | 79.91% | 71.43% | ❌ leaked |

**The original "SOTA 32.24% on length generalization" headline is INVALID.** On the leaked set, direct-answer beat CoT on every split (+39pp on length) — consistent with everything else but absolute numbers untrustworthy.

**Clean redo is possible:** the proper splits, verified at question + (question,cypher) level:
- `length_train → length_test`: **0% overlap at every level** (pristine; by design test queries are longer).
- `train_v1 → test_v1`: 0% question, 0% (q,cypher) overlap (clean; 35% cypher-only is benign).

The clean length-split redo (the real distribution-shift test) is the key pending experiment.

---

## G. STaR-style execution-filtered CoT (in progress)

The one untested form of CoT — verified-correct forward traces (the actual SQL-paper ingredient our post-hoc rationalization lacked).

| Step | Result |
|------|--------|
| G1 Forward generation (Groq, gpt-oss-120b, 7,000 examples, k=4) | ✅ 99.7% candidate success, 72 min |
| G2 Execution filter v1 (column-name-sensitive — buggy) | 15.5% keep rate (1,061) — deflated by alias false-negatives |
| G3 Execution filter v2 (value-only match) | *running* — corrected keep rate |
| G4 Retrain on filtered traces | built, pending G3 |
| G5 Matched direct-answer control on same filtered instances | to build |

---

## H. Data-leakage audits (method lesson)

| Dataset / split | Instance-ID overlap | Question overlap | (Q,Cypher) overlap | Verdict |
|-----------------|:---:|:---:|:---:|---------|
| Neo4j train vs test | 0 | 31.7% | 0.6% | adversarial leakage (hurts EM) |
| ZOGRASCOPE merged (our setup) | 58.6% | 58.6% | 58.6% | ❌ broke the experiment |
| ZOGRASCOPE `length_train→length_test` | 0 | 0% | 0% | ✅ clean |
| ZOGRASCOPE `train_v1→test_v1` | 0 | 0% | 0% | ✅ clean |

**Lesson:** verify overlap at question + pair level, not just instance ID (ID-disjoint missed the 31.7% Neo4j question leakage).

---

## I. CLEAN ZOGRASCOPE redo (per-experiment splits, leakage-verified) — 2026-06-23

Replaces the invalidated §F. Each experiment trained and tested within one split (no merging); both pairings verified 0% question/pair overlap.

**Execution accuracy:**

| Split | Direct-answer | CoT | Δ (direct − CoT) |
|-------|:---:|:---:|:---:|
| IID | 0.8255 (634/768) | 0.6094 (468/768) | +0.216 |
| Compositional | 0.6612 (892/1349) | 0.4181 (564/1349) | +0.243 |
| **Length (distribution shift)** | **0.4397 (551/1253)** | 0.2306 (289/1253) | **+0.209** |

**Direct-answer wins every clean split, including length — the last place CoT could have won.** Degradation IID→length is ~38pp for *both* (direct 82.6→44.0, CoT 60.9→23.1) — same rate, so "CoT degrades gracefully" is also false. The leaked §F numbers (length 71%/32%) were inflated artifacts.

Files: `results/pred_zog_{length,regular}_{cot,baseline}.jsonl`, `results/zog_clean_exec.log`.

## J. STaR execution-filtered CoT — RESULT (2026-06-23) ✅ negative

Trained both arms on the same 3,938 execution-verified instances (only reasoning prefix differs — the STaR-SQL design). Eval on Neo4j test (exec, n=2,471):

| Arm | Exec EM | Pred errors |
|-----|:---:|:---:|
| Direct (verified cypher only) | **0.1052 (260/2471)** | 153 |
| CoT (verified reasoning + cypher) | 0.0939 (232/2471) | 246 |

**Even verified-correct reasoning traces — the actual SQL-paper ingredient — do not help.** CoT −1.1pp and 60% more broken queries. Refutes the "you never used execution filtering" objection. (Low absolutes: 3,938-example training in teacher's Cypher style; comparison is internally clean.)

Value-only filter keep rate: **57.6%** (3,938/6,833) — comparable to STaR-SQL's ~50%, so forward Cypher generation is NOT meaningfully harder than SQL.

## K. The +0.14 training gap — loss masking (2026-06-27)

Why our direct-answer Gemma (A5, GLEU 0.7854) beats Neo4j's published adapter (0.5560) by +0.23,
and beats their adapter under identical inference (A2, 0.6455) by +0.14, despite "same QLoRA config."

**Decomposition of the +0.23-over-published:**
- **+0.09 = eval harness** (same weights: A2 0.6455 vs published 0.5560) — no inference truncation
  (we keep full schemas; they truncate at 1600) + GLEU tokenizer sensitivity (~0.16 swing on
  tokenization alone for the same predictions). Measurement, not a better model.
- **+0.14 = training** (A5 0.7854 vs A2 0.6455). The configs are NOT the same in the decisive place:
  **loss masking.**

**1a (verified):** Neo4j's public recipe (`neo4j-labs/text2cypher` StarCoder2/CodeLlama notebooks)
uses `SFTTrainer(dataset_text_field="text", packing=True, ...)` with **no completion-only collator**
→ **full-sequence loss**, schema in the system message. The Gemma-2024v1 card lists an `SFTConfig`
with no collator named (consistent; exact notebook not public). Our pipeline uses
`DataCollatorForCompletionOnlyLM` (loss on answer tokens only). With ~30:1 schema:answer token ratio,
their loss is ~97% schema-reconstruction; ours is 100% Cypher — a mechanism fully consistent with +0.14.

**Reality checks:** not leakage (A5 scores 0.59 EM on *unseen* vs 0.17 on seen, §C); GLEU-inflated
(exec-EM gap vs published Gemma 0.2104 is only ~+0.05–0.08 — the believable semantic margin).

**1c (pending):** `--full-sequence` ablation on the A5 trainer (only the loss mask differs).
Scripts: `drac_train_gemma_fullseq.sh` / `drac_gemma_fullseq_eval.sh`. If GLEU drops 0.7854 → ~0.64,
masking is confirmed as THE driver. **Reframes the headline:** the gain mis-attributed to CoT was a
stronger direct-answer SFT recipe (completion-only masking on long-schema inputs).

## Bottom line

**CoT distillation does not help Text2Cypher in ANY clean comparison — six now, all negative:**

| # | Comparison | Metric | Direct | CoT | Winner |
|---|-----------|:------:|:------:|:---:|:------:|
| 1 | Neo4j Gemma (naive CoT) | GLEU | 0.7854 | 0.7682 | direct |
| 2 | Neo4j Llama (naive CoT) | GLEU | 0.7680 | 0.7416 | direct |
| 3 | Neo4j STaR (verified CoT) | Exec | 0.1052 | 0.0939 | direct |
| 4 | ZOGRASCOPE IID | Exec | 0.8255 | 0.6094 | direct |
| 5 | ZOGRASCOPE compositional | Exec | 0.6612 | 0.4181 | direct |
| 6 | ZOGRASCOPE length (dist. shift) | Exec | 0.4397 | 0.2306 | direct |

- In-distribution, compositional, and length-generalization; naive AND execution-verified CoT; two model families; two benchmarks. **No regime where CoT helps.**
- The original "+0.1227 GLEU from CoT" was the pipeline, not CoT (matched control A5).
- **Execution-based selection (B2) beats string voting (B1)** — the one small positive transfer-study finding.
- The "ZOGRASCOPE SOTA on length" was a leakage artifact; the clean redo reverses it.

**Remaining (optional / calibration):** Gemma direct-answer exec EM (A5); Neo4j harness calibration (reproduce published 0.5560). Neither can change the conclusion.

**The honest paper:** *"Chain-of-thought distillation does not improve Text2Cypher — in-distribution or under distribution shift, with naive or execution-verified traces, across two model families. Apparent gains in prior framing were a stronger SFT recipe and a leaked benchmark split. Among Text-to-SQL techniques, only execution-based selection transfers; diversity- and reasoning-based methods do not, consistent with Cypher's constrained output space."*
