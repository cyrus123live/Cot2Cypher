# Advisor Feedback for Cot2Cypher

## Status (2026-05-20)

A clean, well-executed experiment: CoT distillation on Gemma-2-9B for Text2Cypher with a tightly controlled setup (same QLoRA config as Neo4j baseline, only training data differs).

**Headline numbers on the Neo4j Text2Cypher 2024 benchmark:**

| Metric | Baseline (reproduced) | CoT Model | Delta |
|--------|----------------------|-----------|-------|
| GLEU (n=4,833) | 0.6455 | 0.7682 | +0.1227 |
| String EM (n=4,833) | 0.1924 | 0.3799 | +0.1875 |
| Exec EM (n=2,471) | 0.1862 | 0.2554 | +0.0692 |
| Prediction Errors | 242 | 114 | -53% |

- #4 on the GLEU leaderboard, best open-weight on the Neo4j benchmark.
- Nearly matches the only published RL approach (Tran et al., 0.7682 vs 0.7701) with SFT alone and no proprietary data.
- ZOGRASCOPE length-generalization fine-tune: **32.24% — SOTA over Mistral 7B (23.46%) and Qwen3-4B (20.19%)**.
- Total pipeline cost: ~$160.

The strongest framing is no longer "we applied CoT to Cypher and it worked" but mechanistic: *we provide the first systematic decomposition of what CoT distillation actually teaches for graph query generation — compositional generalization, schema grounding, and a latent/active reasoning split, each supported by a targeted diagnostic experiment*.

---

## What's Landed Since the April Review

The April review identified five "deeper angles" to lift the paper above pure empirics. Status as of 2026-05-20:

| Angle | Status | Evidence |
|-------|--------|----------|
| 1. Compositional generalization | **Done** | ZOGRASCOPE fine-tune SOTA on length split; IID-to-length degradation gap of 33pp vs 74-84pp for paper baselines (commit `3a65bb1`) |
| 2. 1-hop schema-grounding mechanism | **Done** | Schema-ambiguity analysis: CoT eliminates baseline's 4.6pp ambiguity penalty entirely (commits `e46b32a`, `b448360`) |
| 3. Latent vs active reasoning | **Done** | Cost-accuracy decomposition in paper §6.8 (commits `e46b32a`, `b448360`) |
| 4. When distillation makes RL redundant | **Open** | Framing only; no GRPO experiment run |
| 5. Graph reasoning primitives taxonomy | **Done** | Reframed from "computational complexity" to "specification difficulty" (commit `7293220`) |

Working title moved toward "Schema Grounding and Pattern Composition: What Chain-of-Thought Actually Teaches for Graph Query Generation" — mechanistic insight rather than technique application.

---

## What's Still Needed

Prioritized by how dangerous each is to acceptance, with effort tags. The top four (#1, #2, #4, #7) are the load-bearing ones; everything else is risk reduction.

### Critical (likely deal-breakers at a strong venue)

1. **Single model family.** Every mechanistic claim (schema grounding, compositional generalization, latent/active split) is supported only on Gemma-2-9B. A reviewer can ask "is this a Gemma artifact?" with no answer. *Fix:* Replicate the CoT pipeline on Llama-3.1-8B or Qwen2.5-7B, at least on the Neo4j benchmark. *Compute: ~$50, ~12 hrs DRAC.*

2. **Compositional claim rests on a single benchmark.** ZOGRASCOPE is the only test of the compositional/length-generalization story. If it doesn't replicate elsewhere, the headline contribution shrinks. *Fix:* Zero-shot evaluation on CypherBench (no training needed). *Cheap: just inference.*

3. **Baseline GLEU discrepancy mixes evidence on the leaderboard.** Reproduced baseline (0.6455) is 9pp above Neo4j's published (0.5560); the leaderboard table mixes our pipeline with their published numbers for other models. A reviewer will call this misleading. *Fix:* Either restrict the leaderboard to our-pipeline numbers and re-run Neo4j's exact inference config to land at 0.5560, or split the table into "our pipeline" vs "published numbers" with explicit annotation.

### Serious (each one is a credible reviewer attack)

4. **ZOGRASCOPE IID gap is unaddressed.** Fine-tune gets 65% IID while paper baselines get 96-98%. Reviewers will read this as "the model is worse at the basic task." Needs explicit argument that the memorization-vs-generalization tradeoff is the right one, with evidence. *Fix:* Land the queued 5-epoch run; show IID climbs without length collapsing, OR cleanly argue the tradeoff is intrinsic to the method.

5. **"SFT matches RL" claim rests on one external data point.** Comparison against Tran et al. only. A reviewer can dismiss it as "their RL was poorly tuned." *Fix:* Run GRPO on the CoT model (closes Angle 4 too), or downgrade the claim from "matches RL" to "competitive with the only published RL Cypher result, without RL or proprietary data."

6. **CoT trace generator robustness untested.** All traces come from GPT-oss-120B. No answer to "does CoT distillation depend on the teacher?" *Fix:* Regenerate ~1000 traces with a different teacher (Claude, Llama-3.1-405B), retrain on this subset, show delta. *Cheap: ~$5-10.*

7. **No statistical significance on the deltas.** Every result is point estimate vs point estimate. *Fix:* Bootstrap 95% CIs on GLEU/EM deltas across all tables. *No compute.*

8. **Self-healing loop is conspicuously missing.** 72 unresolved syntax errors. The fix is well-known (feed back error message, re-prompt). Leaving it on the table invites the reviewer to ask why. *Fix:* Implement minimal one-pass self-healing on the 114 failed queries. *Cheap.*

### Important but lower-stakes

9. **2024 dataset known issues need stronger justification.** Data leakage and paraphrase contamination are real. The "baselines are only on 2024" defense is buried in Limitations. *Fix:* Move to introduction; add a quick validation run on 2025 even without baselines to show the deltas hold qualitatively.

10. **Reasoning quality is never audited.** All metrics are downstream. Nobody has actually looked at whether the reasoning traces are correct or just sound correct. *Fix:* Manual audit on 100 sampled traces. Categorize: correct / plausible-but-wrong / incoherent. *Analysis only.*

11. **Reverse ablation (baseline adapter + CoT prompt) is missing.** Format-mismatch argument is reasonable but not bulletproof. *Fix:* Strengthen the argument with a concrete demonstration of format-mismatch causing degradation, OR just run it and acknowledge the confound. *Cheap.*

12. **Execution eval runs against live `demo.neo4jlabs.com`.** Databases change. Reproducibility risk. *Fix:* Snapshot the database state, or at minimum document the evaluation date and any queries that became invalid.

13. **Cypher match rate of 93.3% in CoT generation is unaudited.** The 6.7% mismatches are called "cosmetic" but never systematically verified. *Fix:* Sample 50 mismatches, categorize. *Analysis only.*

### Easy hygiene

14. Pin seeds, dependencies, container versions in an appendix.
15. Add a reproducibility checklist (ML Reproducibility Checklist).
16. Document the exact `demo.neo4jlabs.com` evaluation date and any queries that fail.
