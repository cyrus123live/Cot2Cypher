# LLM Thesis: Beating the Neo4j Text2Cypher Benchmark with Chain-of-Thought Fine-Tuned Gemma

## Project Overview

Master's thesis project aiming to beat the Neo4j text2cypher benchmark by training a Gemma-2-9B model with chain-of-thought reasoning. The core hypothesis is that generating reasoning traces with a strong LLM and fine-tuning a smaller model on those traces will significantly outperform Neo4j's direct-answer fine-tuned baseline.

**Success Criteria:** Achieve higher accuracy than Neo4j's fine-tuned `text2cypher-gemma-2-9b-it-finetuned-2024v1` model on the 2024 benchmark dataset. Beating GPT-4o is a stretch goal but not required -- the value proposition is cost-efficient inference.

---

## Thesis Argument

Current open small models (30-35% correct on Neo4j's benchmark) underperform GPT-4o. We improve small model performance through:
1. Chain-of-thought reasoning fine-tuning (generate reasoning with strong LLM, train smaller model)
2. Full-precision LoRA training (bf16) vs Neo4j's QLoRA (int4)
3. Few-shot prompting and self-consistency at inference
4. Agentic self-healing loops
5. Potentially RL on the fine-tuned model

**Key framing note for paper:** "We compare our novel process (trained in bf16) against the baseline (trained in int4/QLoRA). While some performance lift may be attributed to the higher precision, the significant gain of X% suggests our process provides benefits beyond just precision recovery."

---

## Neo4j Text2Cypher Benchmark (2024)

### Dataset
- **Source:** `neo4j/text2cypher-2024v1` on HuggingFace
- **Size:** 44,387 total instances -- 39,554 training, 4,833 testing
- **Format:** Each entry has `(question, schema, cypher)` triplet
- **Paper:** "Text2Cypher: Bridging Natural Language and Graph Databases" (arXiv:2412.10064)

### Known 2024 Dataset Limitations (must acknowledge in paper)
1. **Data leakage:** Same questions may appear with different Cypher outputs across train/test splits
2. **Paraphrase contamination:** Training on paraphrased examples of test set may inflate performance
3. **Distribution shift:** Results may not generalize to different data distributions
4. Neo4j has acknowledged these issues and released a 2025 updated dataset

### Evaluation Metrics
- **Google BLEU (GLEU):** Translation-based evaluation
- **Exact Match:** Execution-based evaluation (valid Cypher returns correct results)
- **HuggingFace evaluate library** used for both metrics

### Baseline Results (2024 Benchmark)

Fine-tuning improvements over base models (from the paper):
| Model | GLEU Gain | ExactMatch Gain |
|-------|-----------|-----------------|
| Gemini-1.0-Pro-002 | ~+0.34 | ~+0.11 |
| Gemini-1.5-Flash | ~+0.27 | ~+0.09 |
| GPT-4o-mini | ~+0.20 | ~+0.06 |
| Gemma-2-9B-it | ~+0.13 | ~+0.07 |
| Llama-3.1-8B | ~+0.14 | ~+0.11 |

- Closed foundational models (GPT-4o, Gemini) achieved best overall performance
- Among open models, Gemma-2-9B-it was best performing
- Fine-tuned models showed improvements in both GLEU and ExactMatch
- GPT-4o and fine-tuned `tomasonjo_text2cypher` both achieved ~30% match ratio on execution-based eval

### Neo4j Baseline Model Details
- **Model:** `neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1`
- **Base:** `google/gemma-2-9b-it`
- **Method:** QLoRA (4-bit NF4 quantization, double quantization, bf16 compute dtype)
- **LoRA config:** r=64, alpha=64, dropout=0.05, target all linear modules
- **Training:** 1 epoch, lr=2e-5, batch=4, grad_accum=8, max_seq_len=1600
- **Optimizer:** `paged_adamw_8bit`
- **Hardware:** 1x A100 PCIe GPU
- **Framework:** PEFT 0.12.0, SFTTrainer

### Resources
- **Dataset:** https://huggingface.co/datasets/neo4j/text2cypher-2024v1
- **Model:** https://huggingface.co/neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1
- **Code/Notebooks:** https://github.com/neo4j-labs/text2cypher (datasets, evaluations, finetuning notebooks)
- **Blog posts:**
  - Benchmarking: https://neo4j.com/blog/developer/benchmarking-neo4j-text2cypher-dataset/
  - Fine-tuning intro: https://medium.com/neo4j/introducing-the-fine-tuned-neo4j-text2cypher-2024-model-b2203d1173b0
  - Dataset improvements: https://medium.com/neo4j/neo4j-text2cypher-analyzing-model-struggles-and-dataset-improvements-0b965fd3ebfa

---

## Research Plan

### Step 1: Reproduce Baseline
Run Neo4j's fine-tuned Gemma-2-9B on their test set to get comparable numbers.
- **Current file:** `run_neo4j.py` (uses Modal for GPU, loads model with 4-bit quantization)
- Uses A10G GPU via Modal
- Need to run full evaluation on 4,833 test examples

### Step 2: Generate Chain-of-Thought Training Data
For each of the 39,554 training examples:
- **Input:** (Few-shot examples, Schema, Question)
- **Output:** (Reasoning trace, Cypher result)
- Few-shot = 9 examples of (Schema, Question, Reasoning, Result)
- **Generation platform:** Together.ai or anything.ai
  - GPT-oss-120B on anything.ai: ~$25 for 35K examples at ~1K tokens in/out
  - HuggingFace inference: ~$9/hour
- Reference for good reasoning decomposition: QDECOMP

### Step 3: Train New Gemma-2-9B with Reasoning
- **Method:** LoRA (NOT QLoRA) in bf16 precision
- **Platform:** Together.ai for LoRA fine-tuning
- **Compute backup:** Digital Research Alliance of Canada
- Train model to output explanation + Cypher (not just Cypher)

### Step 4: Ablation Studies
- Self-consistency (multiple samples, vote on best)
- Few-shot prompting (0, 3, 9 examples)
- Agentic self-healing loop (re-send with error messages)
- RL on fine-tuned model (optional, if time permits)
- Number of self-consistency calls

### Step 5: Write Paper
- Compare against Neo4j baseline, show improvement
- Ablation tables for each technique
- Mathematical formalization of approach
- Discuss limitations of 2024 dataset
- Cost analysis (small model vs GPT-4o for production deployment)

---

## Key Techniques & Literature

### Chain-of-Thought Fine-Tuning (Core Method)
Generate reasoning traces with a strong LLM, train a smaller model on (input, reasoning, output) triples. Shown to yield ~18% improvement over direct-answer fine-tuning on Spider text-to-SQL benchmark. Fine-tuning with CoT is particularly effective on Hard and Extra difficulty queries.

Key papers:
- "Exploring Chain of Thought Style Prompting for Text-to-SQL" (EMNLP 2023)
- "Rationalization Models for Text-to-SQL" (OpenReview)
- "Optimizing Reasoning for Text-to-SQL with Execution..." (ACL Findings 2025)

### Self-Consistency
Sample multiple reasoning paths, select most consistent answer via voting.
- **CSC-SQL:** Combines self-consistency with self-correction + GRPO RL (arXiv:2505.13271)
- **Universal Self-Consistency (USC):** Matches execution-based voting without code execution
- **MBR Decoding:** Uses execution-based similarity for semantic equivalence voting

Key papers:
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (arXiv:2203.11171)
- "CSC-SQL: Corrective Self-Consistency in Text-to-SQL via Reinforcement Learning" (arXiv:2505.13271)
- "Query and Conquer: Execution-Guided SQL Generation" (arXiv:2503.24364)

### Agentic Loops & Self-Healing
Re-send failed queries with error messages for iterative correction.
- "SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction" (arXiv:2509.00581)
- Live-fetched schema + iterative error correction: https://www.mdpi.com/1999-5903/16/12/438 (14 citations, GPT-4 mini, Dec 2024)

### Schema Linking
Map natural language entities/relations to graph schema elements before query generation. Pruning schema slightly helps accuracy but code not typically available.

### LoRA vs QLoRA
- QLoRA: 4x memory reduction, but 66% slower training
- QLoRA with NF4 "fully recovers 16-bit LoRA performance" per original paper (Dettmers et al.)
- In practice: minimal quality difference, but we train in bf16 LoRA for maximum precision
- Our argument: any performance gain beyond precision recovery demonstrates method value

### Reinforcement Learning for Query Generation
- The MDPI Applied Sciences paper (below) showed 3B model + RL outperformed 9B with standard SFT
- GRPO algorithm used in CSC-SQL for fine-tuning both generation and revision models
- Reward signals: execution correctness, semantic similarity, structural validity

---

## Key Papers (Rated by Relevance)

### Directly Relevant to Thesis

| Rating | Paper | URL | Key Finding |
|--------|-------|-----|-------------|
| 9 | Refining Text2Cypher on Small LM with RL Leveraging Semantic Information | https://www.mdpi.com/2076-3417/15/15/8206 | 3B outperformed 9B when trained on intermediate steps (key-value extraction, relationship triple extraction) + RL. Uses Neo4j's dataset. |
| 8 | Enhancing KG Interactions: Comprehensive Text-to-Cypher Pipeline with LLMs | https://www.sciencedirect.com/science/article/pii/S0306457325002213 | Smaller models match GPT-4 with smart fine-tuning + schema filtering. Different dataset. |
| 7 | Robust NL-to-Cypher for KBQA: Chain of Prompts | https://link.springer.com/chapter/10.1007/978-981-99-7224-1_25 | Multi-stage prompted reasoning for Cypher. We internalize this into CoT fine-tuning. |

### Supporting Papers

| Rating | Paper | URL | Key Finding |
|--------|-------|-----|-------------|
| 6 | GraphRAFT: Retrieval Augmented Fine-Tuning for KGs | https://arxiv.org/pdf/2504.05478 | SOTA on STaRK-prime/mag. Generated own training data. RAG + fine-tuning synergy. |
| 5 | Text2Cypher: Data Pruning using Hard Example Selection | https://arxiv.org/abs/2505.05122 | By Neo4J. Hard example selection saves training time without quality loss. |
| 4 | BYOKG-RAG: Multi-Strategy Graph Retrieval for KGQA | https://aclanthology.org/2025.emnlp-main.1417/ | Agentic graph traversal. Multi-strategy retrieval outperforms single-strategy. |

### Text-to-SQL Transfer Papers
- **DIN-SQL:** Decomposed In-Context Learning with Self-Correction (NeurIPS 2023)
- **SDE-SQL:** Exploration-based SQL generation
- **CHESS:** Schema filtering + CoT for text-to-SQL (2024)
- **CodeS:** Small code-focused models outperform larger general models with proper fine-tuning

---

## Project Structure

```
LLM Thesis/
├── CLAUDE.md              # This file
├── run_neo4j.py           # Baseline inference script (Modal + A10G GPU)
└── venv/                  # Python 3.11 virtual environment
```

### Infrastructure
- **Inference:** Modal (A10G GPU, containerized with transformers/peft/bitsandbytes)
- **Training data generation:** Together.ai or anything.ai
- **Fine-tuning:** Together.ai LoRA API or Digital Research Alliance of Canada
- **Evaluation:** HuggingFace evaluate library (GLEU + ExactMatch)

### Prompt Format (Neo4j Baseline)
```
Generate Cypher statement to query a graph database.
Use only the provided relationship types and properties in the schema.
Schema: {schema}
Question: {question}
Cypher output:
```

### Our Enhanced Prompt Format (CoT)
```
Generate Cypher statement to query a graph database.
Use only the provided relationship types and properties in the schema.
Schema: {schema}
Question: {question}

Think step by step:
1. Identify the key entities and relationships in the question
2. Map them to the schema elements
3. Determine the query pattern (single hop, multi-hop, aggregation, etc.)
4. Construct the Cypher query

Reasoning: {reasoning}
Cypher output: {cypher}
```

---

## Important Notes

- We are using the **2024 baseline and dataset** -- must acknowledge known limitations (data leakage, paraphrase contamination) that were posted before the 2025 release
- We are **NOT quantizing** (unlike Neo4j who used QLoRA). The QLoRA paper claims this shouldn't make a difference, but we should frame any gains carefully
- Neo4j used `paged_adamw_8bit` optimizer -- we may use a different optimizer (document which and why)
- The model is for **research purposes** -- Gemma license is non-commercial
- May attempt fine-tuning a more advanced model to find the line to beat GPT-4o
- Target conferences: EMNLP and similar NLP venues
