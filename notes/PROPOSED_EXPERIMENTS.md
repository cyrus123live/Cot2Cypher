# Proposed Final Experiments — pre-submission menu for Alex

Compiled 2026-09-22. The paper is complete and self-contained without any of these (12/12
negative CoT deltas, all CIs exclude zero; mechanism causally shown; recipe/harness fully
decomposed). These are targeted strengtheners, ordered by value-per-day given the timeline:
**co-edit to done by ~Oct 3 → few days to thesis → defend end of October.** Predictions are
stated in advance so whatever lands is maximally convincing.

---

## E-A. The weak-recipe 2×2 (recipe strength × formalism) — PRIORITY 1

**Question.** Is the baseline-strength account causal? The paper currently argues it from
comparative statics (our direct-SFT Gemma at 76.7% Spider dev outscores STaR-SQL's
CoT-fine-tuned 75.0%). The direct test: train **CoT arms under the weak, literature-typical
recipe** (full-sequence loss / packing) and compare against the weak-recipe direct arms.

**Design.** Four cells, two already filled on the direct side for Cypher (fullseq direct
0.7415, packing direct 0.7330):

| | strong recipe (masking) | weak recipe (full-sequence) |
|---|---|---|
| Cypher (Neo4j) | CoT hurts (−0.017, have it) | **train fullseq-CoT** → predict CoT still hurts, by less |
| SQL (Spider) | CoT hurts (−0.099, have it) | **train weak direct + weak CoT** → predict **CoT flips to helping** |

**Pre-registered predictions.** The mechanism (fragmentation, value corruption) is a
structural *cost* independent of recipe, so Cypher stays negative even with headroom. SQL has
no structural cost, so the weak recipe's headroom should let CoT win — i.e., **we reproduce
the literature's CoT-helps-SQL result inside our own pipeline, using the literature's
recipe**, and show the gain vanishes under completion-only masking.

**What it buys.** Upgrades baseline-strength from interpretive account to crossed causal
demonstration; fully reconciles our negative with the published positives. This is the
single highest-value addition available.

**Cost.** Neo4j fullseq-CoT: 1 train (~12h H100) + 1 eval (~6h) — one-flag change to the
existing trainer (`--full-sequence` already exists; just point it at the CoT target).
Spider weak pair: add a `--full-sequence` flag to `drac_train_sql.py` (small build, mirrors
the Gemma trainer), then one combined job (~16h, same shape as `drac_spider.sh`).
**Fits this week comfortably.**

---

## E-B. Direct-arm candidate diversity (the fair fight for selection) — PRIORITY 2

**Question.** STaR-SQL's celebrated +18pp was CoT **plus verifier best-of-16** — CoT's real
role there may be *candidate diversity for selection*, not better single answers. We never
sampled candidates from the direct model, so the selection comparison (SC@5, MBR, oracle)
has only been run on the CoT arm.

**Design.** Sample SC@5 (T=0.7) from the direct Gemma (A5) on the Neo4j exec subset; compute
string-vote / MBR / oracle; compare to the CoT arm's existing numbers (0.2509 / 0.2665 /
0.4302).

**Pre-registered prediction.** Under the constrained-output-space hypothesis, Cypher offers
little diversity to either arm — oracles should be similar, with direct's likely higher
(its greedy is +4.2pp). If instead CoT's oracle is clearly higher, CoT retains a defensible
role as a diversity engine for best-of-N, and the paper's conclusion gets an honest nuance.

**What it buys.** Closes the last "you didn't give CoT its strongest form" objection;
completes the selection story either way.

**Cost.** Inference-only: one GPU job (5 samples × test set, existing `--self-consistency`
harness) + local execution clustering (existing scripts). No build. **Fits this week.**

---

## E-C. Length-matched filler control — PRIORITY 3 (cheap, deflationary, contingent)

**Question.** *Why* would CoT help under the weak recipe (if E-A's Spider cell flips)? Under
full-sequence loss ~97% of the gradient reconstructs schema; appending a reasoning trace
mechanically shifts loss mass onto the response. So weak-recipe CoT may partially mimic
completion-only masking **by accident** — a loss-rebalancing effect wearing a reasoning
costume.

**Design.** Weak-recipe arm trained on *length-matched neutral filler text + query* (no
semantic reasoning). Compare to weak-recipe CoT.

**Pre-registered prediction.** If filler ≈ CoT under the weak recipe, a chunk of the
literature's CoT-SFT gains reduces to an undiagnosed loss-masking fix — the most
deflationary (and most memorable) finding available.

**Cost.** Filler generation is local and free; one train+eval pair (~18h GPU). Run only if
E-A's Spider cell lands positive. **Piggybacks on E-A's build.**

---

## E-D. Exact STaR-SQL replication — PRIORITY 4 (recommend: thesis future work)

**Question.** The definitive "did we do something wrong" check: replicate their clean
comparison *exactly* (Llama-3.1-8B-Instruct, their Spider prep, their prompt format,
direct-SFT vs CoT-SFT) and see if their +6.4pp appears.

**Why deferred.** Highest build risk — their data prep and prompt details are only partly
documented, so this can eat a week on reconstruction alone. E-A answers most of the same
question more cheaply (if weak-recipe CoT helps SQL in our pipeline, our pipeline can
produce the literature's result). Keep for the thesis's future-work section or a
camera-ready/journal version.

---

## Parked (one line each, future work)

- **One-pass self-healing** (feed execution error back, regenerate): P2 predicts it
  *transfers* — a chance at another confirmed positive prediction. Cheap; bonus if editing
  goes fast.
- **Trained verifier / best-of-N**: expected gain already bounded by the 16%
  verifier-addressable slice; only worth it paired with E-B results.
- **BIRD** (value-inference-heavy SQL): the one regime where the copy-and-place mechanism
  predicts CoT could win; tests the mechanism's boundary. Too big for the timeline.
- **DPO on both arms**: tests the literature claim that DPO *requires* CoT. Out of scope.

---

## Recommended package given the timeline

**Run E-A + E-B in parallel this week** (3–4 DRAC jobs total, one small build), fold results
into the paper during the editing pass, run E-C only if E-A's Spider cell flips. Defer E-D
to future work. If Alex prefers zero new experiments, the paper stands as-is and everything
above moves to the thesis's future-work chapter.
