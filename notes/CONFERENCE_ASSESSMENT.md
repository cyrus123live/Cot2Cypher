# Conference Viability Assessment for Cot2Cypher

## What You Have

A clean, well-executed experiment: CoT distillation on Gemma-2-9B for Text2Cypher, with a strong controlled setup (same QLoRA config as Neo4j baseline, only training data differs). The paper is well-written, properly structured, with good math and thorough analysis.

**Key results:**

| Metric | Baseline | CoT Model | Delta |
|--------|----------|-----------|-------|
| GLEU (n=4,833) | 0.6455 | 0.7682 | +0.1227 |
| String EM (n=4,833) | 0.1924 | 0.3799 | +0.1875 |
| Exec EM (n=2,471) | 0.1862 | 0.2554 | +0.0692 |
| Prediction Errors | 242 | 114 | -53% |

- Rank #4 on the GLEU leaderboard among Neo4j benchmark models (best open-weight on that benchmark); #5 including Tran et al.'s Qwen-3B+GRPO (0.7701), which uses proprietary data
- Nearly matches an SFT+RL approach (0.7682 vs 0.7701) with SFT alone, no proprietary data
- Within 0.0098 GLEU of fine-tuned Gemini-1.5-Flash (0.7780)
- Surpasses unfine-tuned GPT-4o (0.6293) by +0.1389
- Total pipeline cost: ~$75 (data generation + training)
- Clean ablation separating training data vs. inference prompt contributions
- Detailed error analysis showing CoT advantage scales monotonically with query complexity (+0.15 on short queries to +0.63 on 500+ char queries)

---

## Strengths a Reviewer Would Appreciate

1. **Exceptionally clean experimental design.** Same base model, same QLoRA config, same dataset, same everything -- only training data differs. This is rare in the literature and makes the results very convincing. Reviewers love controlled experiments.

2. **The "SFT matches RL" finding is compelling.** Showing that CoT distillation alone (no RL, no proprietary data) matches a GRPO-based approach is a genuinely interesting result with implications beyond this specific benchmark.

3. **Thorough error analysis.** The complexity scaling analysis, per-feature breakdown, improvement-to-regression ratio (8.4:1), and mismatch categorization go well beyond what most papers offer. This is publication-quality analysis.

4. **Execution-based evaluation with fair comparison.** Running both baseline and CoT predictions through identical execution methodology against live databases, rather than relying solely on text-based metrics, adds credibility.

5. **Cost story.** \$75 total, $0.004/query at inference -- practical and reproducible by anyone.

---

## Weaknesses a Reviewer Would Flag

### 1. Limited methodological novelty

CoT distillation for structured query generation is well-explored in Text-to-SQL (STaR-SQL at EMNLP, ExCoT at ACL Findings, Struct-SQL, rationalization models). A reviewer at a top venue will likely frame this as "applying a known technique to a new domain" rather than a methodological contribution. The QDECOMP+InterCOL adaptation from SQL to Cypher is natural but not a significant conceptual leap.

**Mitigation:** Frame the contribution as empirical rather than methodological -- the finding that CoT distillation matches RL on this benchmark is the novel insight, not the technique itself.

### 2. Single benchmark, single model family

All results are on Neo4j text2cypher-2024v1 with Gemma-2-9B. Top venues expect broader validation. There's no evidence the technique transfers to other Cypher benchmarks (CypherBench, ZOGRASCOPE) or other base models (Llama, Qwen, Mistral).

### 3. Incomplete ablation suite

The self-consistency, few-shot, agentic self-healing, and RL experiments are listed as "pending funding." These are exactly the experiments that would differentiate this from a straightforward application paper. A reviewer will ask why these weren't done, especially since self-consistency is essentially free (just run inference multiple times).

### 4. Two confounded variables in the full setup

The full CoT configuration changes both training data AND inference prompt. The ablation partially addresses this (CoT training alone gives +0.0627 GLEU), but the reverse ablation (baseline adapter + CoT prompt) is missing. A reviewer might ask about this, though the justification in the paper (format mismatch makes it confounded) is reasonable.

### 5. Baseline GLEU discrepancy

Your reproduced baseline GLEU (0.6455) is significantly higher than Neo4j's published value (0.5560). The paper acknowledges this and attributes it to input truncation and framework differences, but a reviewer may question whether the comparison to Neo4j's published numbers is meaningful. The delta from YOUR baseline (+0.1227) is the valid comparison, but the leaderboard positioning (which mixes your numbers with Neo4j's published numbers for other models) is slightly awkward.

### 6. Known dataset issues

The 2024 dataset has documented data leakage and paraphrase contamination. While these affect all models equally (and you properly acknowledge them), a reviewer may ask why the cleaner 2025 dataset wasn't used. The answer (no published baselines for 2025) is valid but worth stating more prominently.

---

## Venue-by-Venue Assessment

| Venue | Estimated Chance | Notes |
|-------|-----------------|-------|
| **ACL main** | ~10-15% | Highest novelty bar. Would need a stronger methodological story or broader experiments. |
| **EMNLP main** | ~15-20% | Possible if framed as first systematic CoT study for graph queries with the "SFT matches RL" angle. Still risky on novelty. |
| **ACL/EMNLP Findings** | **~35-45%** | **Best realistic target for the paper as-is.** Findings values solid empirical contributions without requiring breakthrough novelty. |
| **NAACL main** | ~25-30% | Slightly lower bar than ACL/EMNLP. Good fit for the domain. |
| **COLING / LREC-COLING** | ~50%+ | Strong fit. The Neo4j baseline paper itself was at GenAIK@COLING. |
| **Workshops** (GenAIK, KnowledgeNLP, StructuredPrediction, NLP4KGC) | ~60-70% | Very likely accepted. |
| **Domain journals** (Information Processing & Management, Applied Sciences, TKDE) | ~40-50% | Good fit, especially IPM which published Yang et al.'s Cypher pipeline paper. |

---

## What Would Move the Needle Toward Top Tier

These are roughly ordered by impact-to-effort ratio:

### 1. Self-consistency (highest priority, lowest cost)

Run inference N times with temperature sampling, majority-vote on the output. This is essentially free if you already have the inference pipeline. Text-to-SQL papers (CSC-SQL, STaR-SQL) show consistent +3-8% gains from self-consistency. If you get even +0.02-0.03 GLEU from this, it pushes you past Gemini-1.5-Flash (0.7780) and into striking distance of GPT-4o-mini (0.7973). That changes the narrative from "competitive with" to "surpasses."

### 2. Second benchmark (CypherBench or ZOGRASCOPE)

Testing on a second benchmark addresses the generalizability concern directly. ZOGRASCOPE is particularly interesting: they found 98% IID accuracy but only 70-75% compositional generalization and 20-25% length generalization. If CoT specifically helps compositional and length generalization (which your complexity scaling results strongly suggest), that's a powerful story that goes beyond just beating a number.

### 3. Second model family

Run the same CoT distillation on Llama-3.1-8B or Qwen2.5-7B. This addresses "does it only work on Gemma?" and also lets you compare against Tran et al. more directly (they used Qwen2.5-3B).

### 4. RL on top of CoT (GRPO)

Even a preliminary result showing CoT+GRPO > CoT alone > GRPO alone would be a strong contribution. It would validate the claim that CoT distillation and RL are complementary, and could push you into the top 2-3 on the leaderboard.

### 5. Agentic self-healing loop

Re-send failed queries with error messages for iterative correction. Given you already have 114 prediction errors (72 syntax errors), this is a targeted improvement. If you can fix even half of those syntax errors through self-healing, it directly improves exec EM.

---

## Strategic Recommendations

1. **If targeting a deadline in the next 1-2 months:** Submit as-is to EMNLP/ACL Findings or NAACL. The paper is complete and polished. Don't let perfect be the enemy of good.

2. **If you have 2-3 months:** Add self-consistency + one more benchmark. This makes EMNLP main a realistic target.

3. **If you have 3-6 months:** Add self-consistency + second benchmark + second model family + preliminary RL result. This gives you a strong shot at ACL/EMNLP main.

4. **Strongest narrative to lean into:** "We show that teaching a model HOW to think (CoT distillation, $75 SFT) is as effective as teaching it WHAT to extract (RL with proprietary data), and that the two approaches are complementary." This is a cleaner and more interesting story than "we beat a benchmark."

---

## Deeper Angles: What Could Make This a Top-Tier Paper

The current framing — "we applied CoT distillation to Cypher and it worked" — is the weakest possible version of this paper. Reviewers at top conferences have seen CoT distillation applied to SQL many times. They'll shrug and say "of course it worked, we knew it would."

The good news: your results already contain surprising findings that you're not emphasizing. You just need to dig them out and make them the story. Each angle below starts with a plain-English explanation of the concept, then explains how your data supports it and what you'd need to do.

### Angle 1: Compositional Generalization for Graph Pattern Matching

**The concept in plain English:** When humans learn language, we learn small pieces ("the cat," "sat on," "the mat") and combine them into sentences we've never seen before. This ability to use known building blocks in new combinations is called *compositional generalization*. LLMs are famously bad at this — they memorize patterns but struggle when a test query requires combining patterns in ways they haven't seen. This is one of the biggest open problems in NLP right now, and lots of people are working on it.

**What your data shows:** CoT specifically fixes compositional failures for Cypher:

| Feature | Baseline EM | CoT EM | Delta | What it requires |
|---------|------------|--------|-------|-----------------|
| UNION | 0.410 | 0.930 | +0.520 | Composing multiple query branches |
| VAR_LENGTH_PATH | 0.500 | 0.821 | +0.321 | Recursive pattern specification |
| WITH intermediate | 0.063 | 0.244 | +0.181 | Multi-stage sub-query chaining |
| Extra-long (500+) | 0.036 | 0.663 | +0.627 | Multiple composed patterns |

These are all **compositional** features — they require combining sub-patterns into larger structures. ZOGRASCOPE (arXiv:2503.05268) showed this is THE hard problem for Cypher: models get 98% IID accuracy but only 70-75% compositional generalization and 20-25% length generalization.

**The reframing:** Instead of "CoT makes Cypher better," the story becomes "CoT teaches models to *compose* graph patterns they already know individually." The reasoning trace decomposes a complex query into sub-patterns (step 1: sub-questions), links each to schema elements (step 2: InterCOL), identifies the composition pattern (step 3: graph pattern ID), then assembles them (step 4: construction). This mirrors exactly how compositional generalization works in human reasoning — you learn primitives, then learn to compose them. That's a contribution to the compositional generalization literature, not just the Cypher literature. Much bigger audience, much more interesting to reviewers.

**What you'd need:** Run on ZOGRASCOPE with separate IID vs. compositional vs. length-generalization splits. If CoT helps more on compositional and length splits than IID, that's a publishable finding on its own.

### Angle 2: The 1-Hop Paradox — Schema Grounding as Constraint Satisfaction

**The concept in plain English:** A "1-hop" query is the simplest kind of graph traversal — go from one node to a directly connected neighbor. "2-hop" means go through an intermediate node. 3-hop means three steps. Logically, 1-hop should be easiest.

**What your data shows — something genuinely weird:**

| Traversal | Baseline EM | CoT EM | Delta |
|-----------|------------|--------|-------|
| 1-hop | 0.023 | 0.437 | **+0.414** |
| 2-hop | 0.197 | 0.361 | +0.164 |
| 3+-hop | 0.075 | 0.212 | +0.137 |

The baseline is *worst* on the *easiest* queries (2.3% on 1-hop vs 19.7% on 2-hop). And CoT's biggest improvement is on 1-hop (+41.4%), not the harder ones.

**Why this probably happens:** Think of it this way. If a Person node connects to a Movie node through only one relationship type (ACTED_IN), the model can guess the pattern even without thinking. But in real knowledge graphs, two node types are often connected by *multiple* relationship types (Person->DIRECTED->Movie, Person->ACTED_IN->Movie, Person->PRODUCED->Movie). For 1-hop queries, there's no structural complexity to guide the model — it has to pick the right relationship purely from the schema. Without explicit reasoning, it guesses wrong. 2-hop queries give the model more structural clues (the intermediate node type helps disambiguate), so paradoxically, the model does better on them.

CoT fixes this because step 2 of the reasoning trace explicitly says "Relationship: ACTED_IN from Person to Movie." It forces the model to *read the schema and pick the right relationship* before writing the query. In other words: **the baseline doesn't fail because the query is hard — it fails because it doesn't reason about the schema.**

**Why reviewers would care:** This is a *mechanistic* explanation — it tells you *why* CoT helps, not just *that* it helps. Most papers just report numbers. This reframes CoT not as "thinking harder" but as **implicit constraint satisfaction** — the reasoning trace constrains the output space by first identifying valid schema elements, then constructing queries from only those elements.

**What you'd need:** Categorize 1-hop queries by schema ambiguity (how many relationship types connect the two node types in each query). If CoT's improvement correlates with schema ambiguity, you have a mechanistic explanation. This requires no new model runs — just re-analyzing your existing predictions against the schemas.

### Angle 3: Latent vs. Active Reasoning — Two Mechanisms of Knowledge Distillation

**The concept in plain English:** Your ablation tested two things separately:

1. Train on reasoning traces, but at inference time DON'T ask the model to reason (just ask for Cypher directly) — "latent learning"
2. Train on reasoning traces AND ask the model to reason at inference time — "active reasoning"

It's like the difference between a student who absorbed good habits from a textbook (writes decent code instinctively) vs. a student who works through problems step by step on scratch paper (gets exact answers). Both improve performance, but they improve *different things*.

**What your data shows:**

| Configuration | GLEU | String EM |
|--------------|------|-----------|
| Baseline | 0.6455 | 0.1924 |
| CoT training, baseline prompt (latent only) | 0.7082 | 0.2197 |
| CoT training, CoT prompt (latent + active) | 0.7682 | 0.3799 |

GLEU improves roughly equally from both mechanisms (~0.06 each). But exact match is almost entirely driven by active reasoning (+0.0273 from latent vs +0.1602 from active). Training on reasoning traces teaches the model to write *better-shaped* queries (more correct pieces, higher GLEU) even when you don't ask it to think. But getting queries *exactly right* requires the model to actually reason step-by-step at inference time.

**Why reviewers would care:** Nobody has shown this separation before for query generation. It has a practical implication: if you just need approximate queries (e.g., for search or retrieval), skip the reasoning at inference time — it's 3-4x faster since the model doesn't generate ~300 reasoning tokens. If you need exact queries for a production database, use the full reasoning. Most CoT distillation papers treat the technique as one thing. Your ablation shows it's **two separable mechanisms** with different properties.

**What you'd need:** This analysis is already in your data. Just reframe and emphasize it. Maybe add a cost-accuracy tradeoff curve: latent-only inference is 3-4x faster and still gets +0.0627 GLEU.

### Angle 4: When Does Distillation Make RL Redundant?

**The concept in plain English:** Right now, the hottest technique in the field is GRPO (a type of reinforcement learning where the model tries many outputs, sees which ones work, and learns to produce more of the good ones). Everyone is applying it to everything. For SQL, RL gives big gains on top of regular fine-tuning.

But your results show something surprising: for Cypher, plain CoT distillation (no RL) matches a paper that used RL (Tran et al., 0.7682 vs 0.7701). Why would RL help for SQL but not add much for Cypher?

**Intuitive explanation:** Think about SQL vs. Cypher:

- **SQL is procedural** — there are many ways to write the same query (subqueries vs CTEs vs joins, different join orders, EXISTS vs IN, etc.). A reasoning trace shows ONE way. RL can explore MANY ways and learn which ones actually execute correctly. So RL adds value by exploring the large space of valid alternatives.

- **Cypher is more declarative** — you say "match this pattern" and the database figures out how. There are fewer equivalent ways to express the same query. So a reasoning trace already covers most of the solution space, and there's less room for RL to find better alternatives.

**Why reviewers would care:** "When should I use RL vs. distillation?" is a question every practitioner and researcher is asking right now. If you can say "RL helps when the output space is ambiguous (many valid solutions), distillation suffices when it's more constrained" — that's a general principle, not just a Cypher result.

**Prediction this generates:** RL on top of CoT should help MORE on the hard, ambiguous queries (multiple valid MATCH patterns, complex aggregations) and LESS on the easy, unambiguous ones. If you do the GRPO experiment, you can test this directly.

**What you'd need:** Ideally, run GRPO on top of your CoT model and analyze WHERE the additional RL gains come from (which query types, which complexity levels). Even without running RL yourself, you could do a theoretical analysis comparing the ambiguity of SQL vs. Cypher output spaces.

### Angle 5: Graph Reasoning Primitives — A Taxonomy

**The concept in plain English:** Your per-feature results implicitly rank different types of graph reasoning by how much they benefit from being explicitly thought through:

**High CoT benefit (the model needs to "think" about these):**
- Pattern composition (UNION): +0.520
- Schema grounding (1-hop): +0.414
- Recursive patterns (VAR_LENGTH_PATH): +0.321
- Multi-stage (WITH): +0.181

**Medium CoT benefit (somewhat helps):**
- Existence checks (EXISTS): +0.208
- Set operations (DISTINCT): +0.188
- Aggregation (AVG/SUM/MIN/MAX): +0.165

**Low CoT benefit (model can do these on autopilot):**
- Ordering (ORDER BY): +0.088
- Limiting (LIMIT): +0.078
- Zero benefit: UNWIND, OPTIONAL MATCH (too few samples to tell)

This maps onto how conceptually hard each operation is from a graph theory perspective. Composing graph patterns (called "subgraph isomorphism" formally) is fundamentally hard. Counting things (aggregation) is medium. Sorting results (formatting) is trivial — the model just needs to remember the syntax, no reasoning required.

**Why reviewers would care:** This turns your results table into a principled claim about *what kinds of reasoning* benefit from explicit chain-of-thought. It's not just "CoT helps more on hard queries." It's that CoT specifically helps with **graph-structural reasoning** (composing patterns, choosing traversal depth, linking to schema) and NOT with **surface-level formatting** (sorting, limiting). This means CoT is teaching the model something specific about graph structure, not just "how to be more careful." That's useful to anyone designing reasoning pipelines for any structured query language, not just Cypher.

**What you'd need:** Formalize this taxonomy. Connect it to graph theory concepts (subgraph isomorphism for pattern matching, reachability for variable-length paths, graph homomorphism counting for aggregation). This gives the paper theoretical grounding that lifts it above pure empirics.

---

## Recommended Reframing

The current title is: "Teaching Small Models to Think: Chain-of-Thought Distillation for Text-to-Cypher Query Generation"

This frames it as an application paper. Possible reframings that foreground deeper contributions:

1. **Compositional angle:** "Compositional Generalization in Graph Query Generation through Reasoning Distillation"
2. **Distillation-vs-RL angle:** "When Distillation Makes RL Redundant: Chain-of-Thought for Text-to-Cypher"
3. **Graph reasoning angle:** "Schema Grounding and Pattern Composition: What Chain-of-Thought Actually Teaches for Graph Query Generation"

Option 3 is probably strongest because it promises mechanistic insight (WHAT CoT teaches) rather than just a technique (CoT distillation) or a benchmark result (beating Neo4j).

---

## Weaknesses to Fix Before Submission (2026-05-20)

Prioritized by how dangerous each is to acceptance at a strong venue, with effort tags. The top four (#1, #2, #4, #7) move the needle from Findings-tier to main-tier; everything else is risk reduction.

### Critical (likely deal-breakers at a strong venue)

1. **Single model family.** Every mechanistic claim (schema grounding, compositional generalization, latent/active split) is supported only on Gemma-2-9B. A reviewer can ask "is this a Gemma artifact?" with no answer. *Fix:* Replicate the CoT pipeline on Llama-3.1-8B or Qwen2.5-7B, at least on the Neo4j benchmark. *Compute: ~$50, ~12 hrs DRAC.*

2. **Compositional claim rests on a single benchmark.** ZOGRASCOPE is the only test of the compositional/length-generalization story. If it doesn't replicate elsewhere, the headline contribution shrinks. *Fix:* Zero-shot evaluation on CypherBench (no training needed). *Cheap: just inference.*

3. **Baseline GLEU discrepancy mixes evidence on the leaderboard.** His reproduced baseline (0.6455) is 9 pp above Neo4j's published (0.5560), so the leaderboard table mixes his pipeline with their published numbers for other models. A reviewer will call this misleading. *Fix:* Either (a) restrict the leaderboard to his-pipeline numbers and re-run Neo4j's exact inference config to land at 0.5560, or (b) split the table into "our pipeline" vs "published numbers" with explicit annotation.

### Serious (each one is a credible reviewer attack)

4. **ZOGRASCOPE IID gap is unaddressed.** Fine-tune gets 65% IID while paper baselines get 96-98%. Reviewers will read this as "the model is worse at the basic task." Needs explicit argument that the memorization-vs-generalization tradeoff is the right one, with evidence. *Fix:* Land the queued 5-epoch run; show IID climbs without length collapsing, OR cleanly argue the tradeoff is intrinsic to the method.

5. **"SFT matches RL" claim rests on one external data point.** Comparison against Tran et al. only. A reviewer can dismiss it as "their RL was poorly tuned." *Fix:* Either run GRPO on his own CoT model, or downgrade the claim from "matches RL" to "competitive with the only published RL Cypher result, without RL or proprietary data."

6. **CoT trace generator robustness untested.** All traces come from GPT-oss-120B. No answer to "does CoT distillation depend on the teacher?" *Fix:* Regenerate ~1000 traces with a different teacher (Claude, Llama-3.1-405B), retrain on this subset, show delta. *Cheap: ~$5-10.*

7. **No statistical significance on the deltas.** Every result is point estimate vs point estimate. *Fix:* Bootstrap 95% CIs on GLEU/EM deltas across all tables. *No compute.*

8. **Self-healing loop is conspicuously missing.** 72 unresolved syntax errors. The fix is well-known (feed back error message, re-prompt). Leaving it on the table invites the reviewer to ask why. *Fix:* Implement minimal one-pass self-healing on the 114 failed queries. *Cheap.*

### Important but lower-stakes

9. **2024 dataset known issues need stronger justification.** Data leakage and paraphrase contamination are real. Response ("baselines are only on 2024") is defensible but buried in Limitations. *Fix:* Move to introduction; add quick validation run on 2025 even without baselines to show the deltas hold qualitatively.

10. **Reasoning quality is never audited.** All metrics are downstream. Nobody has actually looked at whether the reasoning traces are correct or just sound correct. *Fix:* Manual audit on 100 sampled traces. Categorize: correct / plausible-but-wrong / incoherent. *Analysis only.*

11. **Reverse ablation (baseline adapter + CoT prompt) is missing.** Format-mismatch argument is reasonable but not bulletproof. *Fix:* Strengthen the argument with a concrete demonstration of format-mismatch causing degradation, OR just run it and acknowledge the confound. *Cheap.*

12. **Execution eval runs against live `demo.neo4jlabs.com`.** Databases change. Reproducibility risk. *Fix:* Snapshot the database state, or at minimum document the evaluation date and any queries that became invalid.

13. **Cypher match rate of 93.3% in CoT generation is unaudited.** The 6.7% mismatches are called "cosmetic" but never systematically verified. *Fix:* Sample 50 mismatches, categorize. *Analysis only.*

### Easy hygiene

14. Pin seeds, dependencies, container versions in an appendix.
15. Add a reproducibility checklist (ML Reproducibility Checklist).
16. Document the exact `demo.neo4jlabs.com` evaluation date and any queries that fail.

