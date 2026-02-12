## Overall Verdict: Strong idea, well-researched, very doable for a master's thesis

The core hypothesis -- that chain-of-thought fine-tuning can beat Neo4j's direct-answer fine-tuned baseline -- is well-supported by the literature you've gathered. The STaR-SQL paper showing +18% on Spider, the MDPI paper showing a 3B model beating 9B with intermediate reasoning, and the Chain of Prompts paper hitting 0.94 F1 on Cypher all point in the same direction. This isn't speculative; you're applying a proven technique to an under-explored domain.

Your CLAUDE.md is impressively thorough -- the literature review alone is publication-quality.

## Strengths

1. **Clear, achievable success criteria.** Beating GLEU 0.5560 / EM 0.2104 from Neo4j's QLoRA Gemma-2-9B is realistic given the multiple advantages you're stacking (bf16, CoT, few-shot, self-consistency).

2. **Cost efficiency story is compelling.** ~$30 total budget to potentially match or beat GPT-4o-class performance is a great narrative for the paper.

3. **The baseline reproduction code (`run_neo4j.py`) is solid.** Checkpointing, per-source breakdowns, proper post-processing -- good engineering.

4. **Multiple ablation dimensions.** Few-shot count, self-consistency samples, agentic loops, RL -- gives you plenty of tables for the paper even if some techniques don't pan out.

## Holes and Concerns

### 1. The LoRA vs QLoRA confound is your biggest weakness

You acknowledge this, but reviewers will hammer it. Neo4j explicitly said they didn't tune hyperparameters and minimized resources. If you train in bf16 with LoRA and they trained in int4 with QLoRA, a reviewer can argue your gains come from precision, not CoT. Your framing ("significant gain of X% beyond the typical <1% precision gap") is good, but you need a **direct ablation**: train the same model with bf16 LoRA on the **original direct-answer data** (no CoT) and compare against your CoT version. That isolates the CoT contribution from the precision contribution. Without this, the paper has a clear hole.

### 2. Using the 2024 dataset when 2025 exists

You mention this but it needs stronger justification. A reviewer will ask: "Why not use the cleaner dataset?" Good answers include: (a) the 2024 baseline model/results are the ones published and comparable, (b) 2025 doesn't have published fine-tuned baselines to compare against. But consider running your final model on both datasets if feasible -- it strengthens the paper and addresses the concern pre-emptively.

### 3. CoT reasoning quality is under-specified

The plan says "generate reasoning with GPT-oss-120B" but the reasoning decomposition format matters enormously. The MDPI paper used key-value extraction + triple construction as structured intermediate tasks, not free-form reasoning. You should decide and specify:
- Are you doing **structured step-by-step** (entity extraction -> schema mapping -> pattern ID -> construction)?
- Or **free-form CoT** ("Let me think about this...")?
- How do you **validate** the reasoning traces? If GPT-oss-120B generates bad reasoning that happens to produce the correct Cypher, you're training on noisy signal.
- Do you filter out examples where the generated Cypher doesn't match the reference?

### 4. Few-shot example selection strategy

You mention 9 few-shot examples but don't specify how they're selected. Random? Same schema? Same query pattern? BM25/embedding retrieval? This choice significantly impacts performance (see the CHASE-SQL instance-aware few-shot approach). For 39K training examples, you need an automated selection strategy.

### 5. Evaluation gap: no execution-based eval

Your `run_neo4j.py` computes GLEU and string exact match but not execution-based metrics. Only ~51% of the test set has database access, but that's still 2,471 examples. If you can set up Neo4j instances with the benchmark databases, execution accuracy would substantially strengthen the paper. Without it, you're limited to translation-based metrics which are less meaningful (a query can be textually different but semantically equivalent).

### 6. Model choice uncertainty

The CLAUDE.md mentions Gemma-2-9B as the target but notes Together.ai doesn't support it (only Gemma-3-12b-it). And the Alliance Canada route would let you use Gemma-2-9B directly. You should lock this decision down early -- switching models mid-project creates comparability issues. If you use Gemma-3-12b-it, you're no longer doing an apples-to-apples comparison with Neo4j's Gemma-2-9B baseline.

### 7. Missing from the plan: prompt template alignment

Neo4j used a specific system prompt + user prompt template (Table 3 of their paper). Your `run_neo4j.py` only uses the user template without the system prompt. This could affect your baseline reproduction. Make sure you're using the exact same prompting setup when reproducing their numbers, then clearly document any changes for your approach.

## Suggestions

- **Add the bf16-LoRA-without-CoT ablation.** This is the single most important thing to make the paper airtight.
- **Lock down Gemma-2-9B on Alliance Canada** for apples-to-apples comparison. Use Together.ai only if you want to also test Gemma-3 as a bonus experiment.
- **Build a reasoning trace validation pipeline** -- at minimum, check that the final Cypher in the generated trace matches (or is very close to) the reference Cypher. Discard bad traces.
- **Consider execution-based eval** even if it requires spinning up Neo4j databases. It's what makes the difference between a good and great paper in this space.

## For Your Prof

The knowledge graph angle is strong. You're not just doing text-to-SQL transfer -- you're specifically targeting Cypher/property graphs, which is under-studied compared to SQL. The fact that the best open-source Cypher model (Gemma-2-9B at 0.5560 GLEU) still badly lags GPT-4o (0.6293) means there's clear room for improvement, and your prof's KG expertise gives you domain credibility that a pure NLP group wouldn't have.

The thesis fits well at the intersection of KG + NLP + efficient ML, which is a nice niche.
