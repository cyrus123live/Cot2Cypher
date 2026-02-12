# LLM Thesis: Beating the Neo4j Text2Cypher Benchmark with Chain-of-Thought Fine-Tuned Gemma

## Project Overview

Master's thesis project aiming to beat the Neo4j text2cypher benchmark by training a Gemma-2-9B model with chain-of-thought reasoning. The core hypothesis is that generating reasoning traces with a strong LLM and fine-tuning a smaller model on those traces will significantly outperform Neo4j's direct-answer fine-tuned baseline.

**Success Criteria:** Achieve higher accuracy than Neo4j's fine-tuned `text2cypher-gemma-2-9b-it-finetuned-2024v1` model on the 2024 benchmark dataset. Beating GPT-4o is a stretch goal but not required -- the value proposition is cost-efficient inference.

---

## Thesis Argument

Current open small models (30-35% correct on Neo4j's benchmark) underperform GPT-4o. We improve small model performance through:
1. Chain-of-thought reasoning fine-tuning (generate reasoning with strong LLM, train smaller model)
2. Few-shot prompting and self-consistency at inference
3. Agentic self-healing loops
4. Potentially RL on the fine-tuned model

**Key experimental design:** We use the **same QLoRA configuration** as the Neo4j baseline (4-bit NF4, double quantization, bf16 compute dtype, r=64, alpha=64). The ONLY variable between our model and the baseline is the training data: CoT reasoning traces vs direct-answer. This eliminates precision as a confound and ensures any improvement is attributable to chain-of-thought.

### Differentiation from MDPI RL Paper (Paper 1)

The most directly related work is the MDPI Applied Sciences paper ("Refining Text2Cypher on Small LM with RL", July 2025) which uses Qwen2.5-3B + GRPO with structured semantic extraction tasks (key-value pairs + triples) to achieve 0.7701 GLEU. Our approach is fundamentally different:

| Dimension | MDPI Paper (2025) | Our Thesis |
|-----------|-------------------|-------------|
| **Core method** | RL (GRPO) with structured extraction tasks | CoT distillation from strong LLM |
| **Reasoning type** | Structured key-value + triple extraction in tags | Free-form step-by-step reasoning (QDECOMP+InterCOL-style) |
| **Model** | Qwen2.5-3B-Instruct | Gemma-2-9B-it (apples-to-apples with Neo4j baseline) |
| **Training precision** | BF16 (unclear if LoRA or full FT) | QLoRA matching Neo4j's exact config |
| **Training data** | Home-grown (proprietary) + Neo4j | Neo4j + CoT-augmented via GPT-oss-120B (reproducible) |
| **RL component** | GRPO (core method) | Optional (after CoT SFT) |
| **Inference techniques** | Single-pass | Self-consistency + few-shot + self-healing |
| **Evaluation** | GLEU only on Neo4j benchmark | GLEU + Execution accuracy + full ablations |
| **Reproducibility** | Proprietary dataset, sparse hyperparameter details | Fully reproducible on public data |

**Key narrative:** They teach the model **what to extract** (structured key-value pairs and triples via RL reward signals). We teach the model **how to think** (free-form reasoning traces via CoT distillation). These are complementary approaches — their structured extraction is a narrow form of intermediate reasoning, while our QDECOMP+InterCOL-style decomposition provides richer, more generalizable reasoning chains.

**Gaps in their work we exploit:**
1. No CoT distillation from a strong LLM
2. No execution-based evaluation on Neo4j benchmark (GLEU only)
3. No self-consistency / majority voting at inference
4. No few-shot prompting
5. No agentic self-healing loops (they acknowledge this as future work)
6. No execution-based RL rewards (text-matching only)
7. Proprietary dataset makes ablation results non-reproducible
8. Single model family (Qwen only)

---

## Neo4j Text2Cypher Benchmark (2024)

### The Benchmark

The Neo4j Text2Cypher benchmark is a standardized evaluation framework for assessing language models' ability to translate natural language questions into Cypher queries. The foundational paper is **"Text2Cypher: Bridging Natural Language and Graph Databases"** by Makbule Gulcin Ozsoy, Leila Messallem, Jon Besga, and Gianandrea Minneci (all Neo4j). Published at the **GenAIK Workshop (co-located with COLING)**, Abu Dhabi, UAE, January 2025, pages 100-108.

- **ArXiv:** https://arxiv.org/abs/2412.10064
- **ACL Anthology:** https://aclanthology.org/2025.genaik-1.11/

### Dataset: `neo4j/text2cypher-2024v1`

- **URL:** https://huggingface.co/datasets/neo4j/text2cypher-2024v1
- **Total instances:** 44,387
- **Training split:** 39,554 instances (89.1%)
- **Test split:** 4,833 instances (10.9%)
- **Unique schemas:** 966
- **Question length:** 14 - 1,600 characters
- **Cypher query length:** 18 - 2,900 characters
- **Data sources:** 20 distinct sources
- **Database references:** 17 distinct database aliases
- **License:** Apache 2.0
- **Language:** English
- **Format:** Each entry has `(question, schema, cypher)` triplet plus metadata (data_source, database_reference, instance_id, split)

**Data source breakdown (training set):**
- neo4jLabs_functional_cypher (34.3%)
- neo4jLabs_synthetic_gpt4turbo (16.0%)
- neo4jLabs_synthetic_gpt4o (15.4%)
- neo4jLabs_synthetic_gemini (14.9%)
- neo4jLabs_synthetic_claudeopus (8.2%)
- neo4j_text2cypher2023_train (6.5%)
- neo4j_crowdsourced (1.2%)
- Others (3.3% combined)

**Instances with database access (for execution-based evaluation):**
- Training set: 22,093 instances (55.85%)
- Test set: 2,471 instances (51.12%)

**Construction methodology:**
1. Identified 25 publicly available datasets from Neo4j resources, HuggingFace, and academic papers; utilized 16 that met criteria
2. Standardized into a single format
3. Two-stage cleaning: (a) manual checks removing unwanted characters, irrelevant questions, deduplication by [question, cypher] pairs; (b) syntax validation using EXPLAIN clauses in Neo4j, removing queries with syntax errors
4. Split: train-specific files to train, test-specific files to test, remaining split 90:10

### Evaluation Metrics

**A) Translation-Based Evaluation (Lexical)**
Generated Cypher queries compared with reference queries based on textual content. Metrics computed using HuggingFace Evaluate library:
- Text2Text: ROUGE, BLEU, METEOR
- Embedding similarity: BERTScore, FrugalScore
- Text similarity: Cosine and Jaro-Winkler
- **Google-BLEU (GLEU)** -- primary metric: Google's sentence-level BLEU variant. Records all n-gram sub-sequences (1-4 tokens), computes recall and precision, returns minimum. Range: 0-1.
- **Exact Match** -- secondary metric

**B) Execution-Based Evaluation**
Generated and reference Cypher queries executed on target databases. Outputs converted to string representations (ordered lexicographically), then same metrics applied. Only works on instances with database access (51.12% of test set = 2,471 instances).

### Complete Baseline Results (Figure 4 of Paper)

#### Google-BLEU Scores (Translation-Based)

| Model | Google-BLEU |
|-------|------------|
| **Previously Fine-Tuned Models** | |
| hf_finetuned_tomasonjo_text2cypher (Llama-3-8b) | **0.5534** |
| openai_finetuned_neo4j_text2cypher_23_gpt3_5 | **0.5222** |
| hf_finetuned_neo4j_text2cypher_23_codellama (13B) | **0.4552** |
| hf_finetuned_lakkeo_stable_cypher_instruct3B | **0.3767** |
| **Open-Weighted Foundational Models** | |
| hf_foundational_meta_llama3_1_8B_instruct | **0.5104** |
| hf_foundational_codegemma_7B_it | **0.4774** |
| hf_foundational_codeLlama_7B_instruct_hf | **0.4130** |
| hf_foundational_gemma2_9B_it | **0.3986** |
| **Closed Foundational Models** | |
| openai_gpt4_o | **0.6293** |
| openai_gpt4_o_mini | **0.5962** |
| googleaistudio_gemini-1.5-flash-001 | **0.5845** |
| googleaistudio_gemini-1.5-pro-001 | **0.5506** |
| vertexai_gemini-1.0-pro-002 | **0.5033** |
| openai_gpt3_5 | **0.3892** |
| **2024 Fine-Tuned Models (on this dataset)** | |
| openai_finetuned_gpt4_o | **0.8017** |
| openai_finetuned_gpt4_o_mini | **0.7973** |
| googleaistudio_finetuned_gemini_1.5_flash_001 | **0.7780** |
| vertexai_finetuned_gemini_1_pro_002 | **0.7293** |
| hf_finetuned_meta_llama3_1_8B_instruct | **0.6470** |
| hf_finetuned_gemma2_9B_it | **0.5560** |

#### Exact Match Scores (Execution-Based)

| Model | ExactMatch |
|-------|-----------:|
| **Previously Fine-Tuned Models** | |
| hf_finetuned_tomasonjo_text2cypher | **0.2987** |
| openai_finetuned_neo4j_text2cypher_23_gpt3_5 | **0.1574** |
| hf_finetuned_lakkeo_stable_cypher_instruct3B | **0.1522** |
| hf_finetuned_neo4j_text2cypher_23_codellama | **0.1489** |
| **Open-Weighted Foundational Models** | |
| hf_foundational_codegemma_7B_it | **0.1481** |
| hf_foundational_meta_llama3_1_8B_instruct | **0.1449** |
| hf_foundational_codeLlama_7B_instruct_hf | **0.1186** |
| hf_foundational_gemma2_9B_it | **0.0943** |
| **Closed Foundational Models** | |
| openai_gpt4_o | **0.3173** |
| openai_gpt4_o_mini | **0.2582** |
| googleaistudio_gemini-1.5-flash-001 | **0.2307** |
| googleaistudio_gemini-1.5-pro-001 | **0.2052** |
| vertexai_gemini-1.0-pro-002 | **0.1999** |
| openai_gpt3_5 | **0.1186** |
| **2024 Fine-Tuned Models** | |
| openai_finetuned_gpt4_o | **0.3250** |
| openai_finetuned_gpt4_o_mini | **0.3157** |
| googleaistudio_finetuned_gemini_1.5_flash_001 | **0.2768** |
| hf_finetuned_meta_llama3_1_8B_instruct | **0.2299** |
| vertexai_finetuned_gemini_1_pro_002 | **0.2246** |
| hf_finetuned_gemma2_9B_it | **0.2104** |

#### Fine-Tuning Delta Summary

| Model | Delta GLEU | Delta ExactMatch |
|-------|-----------|-----------------|
| Gemma-2-9B-it | +0.1574 (0.3986→0.5560) | +0.1161 (0.0943→0.2104) |
| Llama-3.1-8B | +0.1366 (0.5104→0.6470) | +0.0850 (0.1449→0.2299) |
| Gemini-1.0-Pro | +0.2260 (0.5033→0.7293) | +0.0247 (0.1999→0.2246) |
| Gemini-1.5-Flash | +0.1935 (0.5845→0.7780) | +0.0461 (0.2307→0.2768) |
| GPT-4o-mini | +0.2011 (0.5962→0.7973) | +0.0575 (0.2582→0.3157) |
| GPT-4o | +0.1724 (0.6293→0.8017) | +0.0077 (0.3173→0.3250) |

**KEY OBSERVATIONS:**
1. **Best overall:** Fine-tuned GPT-4o (GLEU 0.8017, EM 0.3250)
2. **Best unfine-tuned:** GPT-4o (GLEU 0.6293, EM 0.3173)
3. **Best open-weight baseline:** Llama-3.1-8B (GLEU 0.5104), NOT Gemma-2-9B (0.3986)
4. **Code-focused models underperformed:** Cypher is "relatively closer to natural language"
5. **tomasonjo_text2cypher had seen 14.4% of test data** during training -- potentially misleading results
6. **GPT-4o ExactMatch barely improved with fine-tuning** (+0.0077), suggesting baseline was near ceiling
7. **Execution-based EM is much lower** than translation-based GLEU across all models

### Neo4j Baseline Model Details

- **Model:** `neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1`
- **Base:** `google/gemma-2-9b-it`
- **Method:** QLoRA (4-bit NF4 quantization, double quantization, bf16 compute dtype)
- **LoRA config:** r=64, alpha=64, dropout=0.05, target all linear modules, bias=none, task=CAUSAL_LM
- **Quantization:** BitsAndBytes -- load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
- **Training:** 1 epoch, lr=2e-5, batch=4, grad_accum=8 (effective batch=32), max_seq_len=1600
- **Optimizer:** `paged_adamw_8bit`
- **Hardware:** 1x NVIDIA A100 PCIe, 31 vCPU, 117 GB RAM, 60 GB disk (RunPod)
- **Framework:** PEFT 0.12.0, PyTorch 2.4.0, Python 3.11, CUDA 12.4.1
- **Container:** runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

**NOTE from authors:** *"During fine-tuning, our goal was to minimize resource usage. With better-tuned parameters, we could potentially achieve even stronger results."* Results are without hyperparameter tuning.

### Prompt Template (Table 3 of Paper)

**System:**
```
Task: Generate Cypher statement to query a graph database.
Instructions: Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided in the schema.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
```

**User:**
```
Generate Cypher statement to query a graph database.
Use only the provided relationship types and properties in the schema.
Schema: {schema}
Question: {question}
Cypher output:
```

### Known 2024 Dataset Limitations (must acknowledge in paper)

From Section 6 "Risks and Pitfalls" of the paper:
1. **Data leakage:** Same questions may appear with different Cypher outputs across train/test splits. Fine-tuned models could encounter the question during training but with a different (incorrect) answer.
2. **Paraphrase contamination:** Paraphrased versions of same questions across train and test -- "may artificially inflate the performance of the fine-tuned model."
3. **Distribution shift:** Both splits drawn from same distribution. "If the data distribution shifts, the results may not hold up."
4. **Data contamination risk:** Dataset from publicly available sources -- foundational models may have seen data during pre-training.

### Model Struggle Analysis (March 2025)

From Ozsoy's analysis (https://neo4j.com/blog/developer/text2cypher-model-struggles-dataset-improvements/):
- Models struggled most with **synthetic datasets**: neo4jLabs/functional, neo4jLabs/gemini, neo4jLabs/gpt4o
- Models struggled most with specific **databases**: recommendations, companies, neoflix
- **Ambiguous questions:** e.g., asking for top-k without specifying ranking field
- **Schema issues:** Questions answerable through different relationships
- **Cypher-specific challenges:** MAX function vs ORDER BY with LIMIT
- **Ground-truth quality issues:** Different clause structures (WHERE vs property conditions) producing equivalent results

---

## 2025 Dataset Update

### Dataset: `neo4j/text2cypher-2025v1`

- **URL:** https://huggingface.co/datasets/neo4j/text2cypher-2025v1
- **Total instances:** 40,400 (down from 44,387)
- **Training split:** 35,900 instances (down from 39,554)
- **Test split:** 4,440 instances (down from 4,833)
- **Unique schemas:** 965
- **Data sources:** 21 (up from 20)
- **Cypher query length:** 19-1,600 chars (max down from 2,900)
- **License:** Apache 2.0

**Key changes:** ~4,000 instances removed via cleaning. Max Cypher length tightened. One additional data source. One fewer database reference (16 vs 17).

### New Models for 2025

- **`neo4j/text-to-cypher-Gemma-3-27B-Instruct-2025.04.0`** -- Base: google/gemma-3-27b-it, BF16 precision, April 2025
- Both Gemma 3-based models deliver superior performance compared to earlier-generation models

### Iterative Refinement (September 2025)

From Ozsoy (https://medium.com/neo4j/exploring-iterative-refinement-for-text2cypher-ea99aeb28949):
- Two-step loop: Verification (check if Cypher is valid) + Correction (refine if invalid)
- Demonstrated "self-healing capability" through iterative prompting on 2025v1 test split

---

## Key Techniques & Literature (Comprehensive)

### 1. Chain-of-Thought Fine-Tuning (Core Method)

Generate reasoning traces with a strong LLM, train a smaller model on (input, reasoning, output) triples. The core insight is that **training on reasoning traces teaches the model HOW to think, not just WHAT to output**.

**Key paper: "Exploring Chain of Thought Style Prompting for Text-to-SQL" (EMNLP 2023)**
- Authors: Chang-You Tai, Ziru Chen, Tianshu Zhang, Xiang Deng, Huan Sun
- URL: https://arxiv.org/abs/2305.14215
- Tested CoT prompting for text-to-SQL with multiple models (Codex, ChatGPT, GPT-4)
- Found CoT significantly helps on **hard and extra-hard queries** in Spider
- **Four CoT methods tested (A-D):**
  - **(A) Chain-of-Thought:** Standard free-form reasoning before SQL generation
  - **(B) Least-to-Most:** Iterative prompting — decompose first, then solve sub-problems sequentially (found unnecessary for text-to-SQL, causes error propagation)
  - **(C) QDecomp (Question Decomposition):** Decomposes the original complex question into a series of sub-questions, each corresponding to a specific SQL component (SELECT, WHERE, JOIN, etc.). Aligns reasoning steps with SQL execution order using natural language templates.
  - **(D) QDecomp+InterCOL (our initial reasoning strategy):** Extends QDecomp by adding **intermediate column identification** — the model explicitly names relevant columns/tables from the schema at each decomposition step. This bridges the gap between natural language sub-questions and schema elements, effectively performing schema linking as part of the reasoning chain.
- **Results (8-shot, Codex, test-suite accuracy):**
  - QDecomp+InterCOL vs standard prompting: **+5.2pp on Spider dev, +6.5pp on Spider Realistic**
  - QDecomp+InterCOL vs Least-to-Most: **+2.4pp on Spider dev, +1.5pp on Spider Realistic**
- **Key finding:** Iterative prompting (Least-to-Most) is unnecessary and introduces error propagation. Single-pass decomposition with schema-grounded intermediate columns (QDecomp+InterCOL) is more effective.
- **Why we adopt this for Cypher:** The QDECOMP+InterCOL structure maps naturally to Cypher generation: sub-questions → node/relationship identification → property mapping → pattern construction. We adapt this from SQL to Cypher as our initial reasoning trace format, generating traces with GPT-oss-120B. May need adaptation since Cypher's graph patterns differ from SQL's tabular joins.

**THE 18% PAPER: STaR-SQL (He et al., ACL 2025)**
- **Full title:** "STaR-SQL: Self-Taught Reasoner for Text-to-SQL"
- **Authors:** Mingqian He, Yongliang Shen, Wenqi Zhang, Qiuying Peng, Jun Wang, Weiming Lu
- **URL:** https://arxiv.org/abs/2502.13550
- **Base model:** Llama-3.1-8B-Instruct
- **Methodology:** Iterative self-improvement loop (bootstrapping):
  1. Generate k candidate rationale-SQL pairs per question using few-shot prompting
  2. Filter rationales whose SQL matches golden queries via execution match
  3. Difficulty-based resampling to address "tail narrowing" bias
  4. For incorrect responses, provide golden SQL as hint for rationalization
  5. Fine-tune via SFT on collected rationales
  6. Re-initialize to original pre-trained model to mitigate overfitting, repeat
  7. Train Outcome-supervised Reward Model (ORM) as verifier; at test time, generate N candidates, select highest ORM score (best-of-N)
- **Results on Spider Dev:**
  | Configuration | Execution Accuracy |
  |---|---|
  | Few-shot baseline | 55.0% |
  | SQL-only fine-tuning (no reasoning) | 68.6% |
  | STaR-SQL (without verifier) | 75.0% |
  | Self-Consistency (majority vote) | 78.8% |
  | **STaR-SQL with ORM@16** | **86.6%** |
- **+18.0 percentage points** over direct-answer fine-tuning (68.6% → 86.6%)
- **+31.6pp** over few-shot baseline
- Outperforms GPT-4 agent-based prompting methods
- **Largest gains on complex queries:** Extra-hard 69.3% (+5.8%), Hard 82.8% (+9.1%)

**Struct-SQL: Structured CoT Distillation (arXiv:2512.17053)**
- Uses query execution plans as formal blueprints for structured reasoning (not free-form CoT)
- **+8.1% absolute improvement** over unstructured CoT distillation
- Marked reduction in syntactic errors as primary driver

**ExCoT: Iterative CoT + DPO (ACL 2025 Findings, arXiv:2503.19988)**
- Authors: Zhewei Yao et al. (Snowflake)
- Combines CoT reasoning with off-policy and on-policy DPO using execution accuracy as feedback
- Pipeline: SFT on GPT-4o CoTs → Off-policy DPO → On-policy iterative DPO
- **Results:** LLaMA-3 70B: BIRD dev 57.37% → **68.51%** (+11.14%), Spider test **86.59%** (+7.78%)
- BIRD test: **68.53%** (SOTA single-model at time)

**CRITICAL FINDING: DPO Requires CoT (Liu et al., ACL 2025, arXiv:2502.11656)**
- "Uncovering the Impact of Chain-of-Thought Reasoning for Direct Preference Optimization"
- **DPO consistently fails or degrades performance** when applied to Text-to-SQL without reasoning chains
- When CoT reasoning is added, DPO achieves "consistent and significant performance improvements" for the first time
- Qwen 1.5B achieved **+18.2% improvement** through DPO with CoT on BIRD
- CoT also mitigates reward hacking and improves scalability

### 2. Self-Consistency

Sample multiple reasoning paths, select most consistent answer via voting. Originally proposed by Wang et al. (2022).

**Self-Consistency (Wang et al., 2022):** arXiv:2203.11171
- Sample multiple chain-of-thought reasoning paths
- Take majority vote over final answers
- Showed consistent improvements across arithmetic, commonsense, and symbolic reasoning benchmarks
- Key insight: diverse reasoning paths that converge on the same answer are more likely correct

**CSC-SQL (arXiv:2505.13271):**
- Combines self-consistency with self-correction + GRPO reinforcement learning
- Two-stage: (1) generate multiple SQL candidates via CoT, (2) use trained correction model to refine
- GRPO trains both generation and revision models with execution-based rewards
- Achieves SOTA on Spider and BIRD benchmarks

**Universal Self-Consistency (USC):**
- Matches execution-based voting performance without code execution
- Uses LLM to select most consistent answer from sampled outputs

**MBR Decoding:**
- Uses execution-based similarity for semantic equivalence voting
- Runs candidate queries and clusters by result equivalence

### 3. Agentic Loops & Self-Healing

Re-send failed queries with error messages for iterative correction.

**SQL-of-Thought (arXiv:2509.00581):**
- Five-stage multi-agent: schema linking → subproblem ID → query plan generation → SQL generation → guided correction
- Taxonomy-guided dynamic error modification: classifies **31 types of SQL errors** and fixes systematically
- **Results: 91.59% on Spider** (SOTA)
- Accuracy drops **at least 5%** without CoT query plan before SQL generation
- Accuracy drops **at least 8-10%** without correction loop

**MAC-SQL (COLING 2025, arXiv:2312.11242):**
- Three agents: Selector (schema pruning), Decomposer (CoT query generation), Refiner (error correction)
- **Results:** 59.59% execution accuracy on BIRD test, 86.75% on Spider dev
- Open-sourced SQL-Llama (fine-tuned Code Llama with agent instruction data)
- 30% of errors were gold standard annotation errors, not model errors

**Live-fetched schema + iterative error correction (MDPI Future Internet, Dec 2024):**
- URL: https://www.mdpi.com/1999-5903/16/12/438
- Dynamic schema retrieval from Neo4j at query time (no hardcoded schema)
- Iterative error correction loop with GPT-4 Turbo
- Schema-agnostic, immediately deployable on any Neo4j database
- 14 citations

### 4. Schema Linking & Filtering

Map natural language entities/relations to graph schema elements before query generation.

**DIN-SQL (NeurIPS 2023, arXiv:2304.11015):**
- Decomposed In-Context Learning with Self-Correction
- Four-module pipeline: schema linking → query classification → SQL generation → self-correction
- Among first to show structured decomposition improves text-to-SQL

**CHESS (arXiv:2405.16755, 2024):**
- Schema filtering + CoT for text-to-SQL
- Dynamic schema pruning based on question content reduces noise for small models

**Enhancing Text2Cypher with Schema Filtering (Neo4j, arXiv:2505.05118, May 2025):**
- Five filtering approaches (2 static, 3 dynamic)
- "Pruned by Exact-Match Schema" achieved highest accuracy
- Token reduction: Up to 62.6% median reduction
- Cost savings: Up to 62.9% across providers
- **Smaller models (Llama-3.1-8B, Qwen2.5-7B) benefit most from schema filtering**

### 5. QDECOMP (Question Decomposition)

**QDMR and Break Dataset (Wolfson et al., TACL 2020):**
- URL: https://arxiv.org/abs/2001.11770
- Question Decomposition Meaning Representation (QDMR): ordered list of natural language steps to answer a question
- Break dataset: 83K+ question-QDMR pairs
- Reference for good reasoning decomposition in our training data generation

**Weakly Supervised Text-to-SQL Parsing through Question Decomposition (NAACL 2022):**
- URL: https://arxiv.org/abs/2112.06311
- Uses QDMR as intermediate representation between NL and SQL
- Weakly-supervised models achieve **91-97% of fully supervised performance**
- Even zero/few-shot QDMR reaches 86-93% of supervised performance

### 6. Decomposed & Two-Stage Approaches

**DTS-SQL (EMNLP 2024 Findings, arXiv:2402.01117):**
- Two-stage fine-tuning: first schema linking, then SQL generation
- Improves execution accuracy by **3-7%** on cross-domain datasets
- **60.31% on BIRD** test set -- highest among 7B parameter methods
- Shows open-source models can match proprietary ones with proper decomposition

**CHASE-SQL (ICLR 2025, arXiv:2410.01943):**
- Multi-path reasoning with preference-optimized candidate selection
- Three candidate generation strategies: divide-and-conquer, query execution plan CoT, instance-aware few-shot
- Selection agent ranks candidates via pairwise comparison using fine-tuned binary LLM
- **73.0% execution accuracy on BIRD test** -- SOTA at time of submission

### 7. Reinforcement Learning for Query Generation

**GRPO (Group Relative Policy Optimization)** has emerged as the dominant RL algorithm for text-to-SQL in 2025:
- From DeepSeekMath (arXiv:2402.03300)
- Groups candidate outputs, computes rewards, optimizes policy based on relative ranking within group
- More stable than PPO for language model fine-tuning

**Reasoning-SQL (arXiv:2503.23157) -- GRPO with Partial Rewards:**
- Authors: Pourreza et al. (including Google Research)
- Five composite reward signals: Execution Accuracy (weight 3), LLM-as-Judge (weight 2), Syntax Check (1), Schema Linking Jaccard (1), N-gram Similarity (1)
- Training: 6 candidates per input, lr=1e-6, batch=32, 3 epochs, 8x H100 GPUs
- **Results:** Qwen2.5-Coder-14B achieves 65.31% BIRD dev; with CHASE-SQL pipeline: **72.78% BIRD test**
- **14B model outperforms o3-mini by 4%** and Gemini-1.5-Pro-002 by 3%
- RL improves base model by **+6.77%** vs **+4.11%** for conventional SFT
- Key finding: "RL fosters robust generalization whereas SFT tends to favor memorization"

**Arctic-Text2SQL-R1 (arXiv:2505.20315) -- Simple Execution Reward:**
- Authors: Yao et al. (Snowflake)
- Uses GRPO with **minimal reward**: just execution correctness + basic syntax validity
- Avoids complex reward shaping; includes inference-time value retrieval + majority voting
- **Results:** 7B model: BIRD dev **68.9%**, BIRD test **68.5%** -- **#1 on BIRD leaderboard**
- **7B model outperforms prior 70B-class systems**
- Average 57.2% across 6 benchmarks (BIRD, Spider, Spider2.0, Spider-DK, EHRSQL, ScienceBenchmark)

**AGRO-SQL (arXiv:2512.23366) -- Agentic GRPO with Data Synthesis:**
- Dual-centric: Data factory (DAG-based DB augmentation, SQL decomposition, tournament selection, K-cycle refinement) + Model training (Diversity-Aware Cold Start SFT → GRPO with environmental feedback)
- **Single-model SOTA on both BIRD and Spider**

**CSC-SQL (arXiv:2505.13271)** uses GRPO for fine-tuning both generation and correction models. 7B: 71.72%, 32B: **73.67%** on BIRD test.

**The MDPI Applied Sciences paper** (see Key Papers below) showed 3B model + GRPO outperformed 9B with standard SFT -- **first application of RL to Text2Cypher**

**PaVeRL-SQL (arXiv:2509.07159):** Partial-match rewards + verbal self-evaluation + CoT RL → **+7.4%** over prior SOTA on Spider2.0-SQLite

**Reward signal design for query RL:**
- Execution correctness (binary: does it run and return correct results?) -- most important signal
- LLM-as-Judge (evaluates incorrect queries for logical/semantic correctness)
- Syntax validity check
- Schema linking Jaccard similarity
- N-gram similarity (bigram Jaccard)
- Key-value extraction accuracy (for Cypher-specific intermediate reasoning)
- Triple relationship accuracy (for Cypher-specific intermediate reasoning)

### 8. Ant Group's Ling Model Family

- **Ling-1T:** Trillion-parameter MoE model for standard language tasks
- **Ring (thinking models):** For complex reasoning, including Ring-1T-preview
- 70.42% on 2025 AIME benchmark
- Relevant as an example of scaling reasoning capabilities, but not directly applicable to our small-model approach

### 9. SDE-SQL and Other Transfer Methods

**SDE-SQL (arXiv:2506.07245):**
- LLM autonomously crafts probe queries to investigate database contents during inference
- Executes probes and incorporates results into reasoning -- zero-shot, no training pairs needed
- **Results with Qwen2.5-72B:** BIRD **+8.02%** over vanilla baseline; SOTA among open-source without SFT

**RSL-SQL (arXiv:2411.00073) -- Robust Schema Linking:**
- Bidirectional schema linking (forward + backward pruning) + contextual augmentation + binary selection + multi-turn self-correction
- **94% schema linking recall** with **83% reduction** in input columns
- BIRD: 67.2%, Spider: 87.9% (with GPT-4o); outperforms GPT-4 systems using DeepSeek

**SQL-PaLM (arXiv:2306.00739):** Google's PaLM-based approach combining few-shot + consistency decoding + execution filtering + instruction fine-tuning. Spider test-suite: 78.2%.

**CodeS:** Small code-focused models outperform larger general models with proper fine-tuning

---

## Key Papers (Comprehensive with Full Details)

### Tier 1: Directly Relevant to Thesis

#### Paper 1: Refining Text2Cypher on Small LM with RL (Relevance: 10/10)

- **Full title:** "Refining Text2Cypher on Small Language Model with Reinforcement Learning Leveraging Semantic Information"
- **URL:** https://www.mdpi.com/2076-3417/15/15/8206
- **Venue:** Applied Sciences (MDPI), Volume 15, Issue 15, Article 8206, July 23, 2025
- **Authors:** Team behind chenggong1995/Qwen2.5-3B-Instruct-grpo on HuggingFace

**Methodology:**
1. **SFT phase:** Qwen2.5-3B-Instruct fine-tuned on combined home-grown + Neo4j Text2Cypher data
2. **GRPO RL phase:** Reinforcement learning with two support tasks as intermediate reasoning:
   - **Key-Value Extraction:** Model learns to extract property pairs (e.g., "name: John", "age: 30")
   - **Triple Relationship Construction:** Model learns subject-predicate-object triples
3. Reward functions incorporate correctness of both final Cypher AND intermediate semantic extractions

**Key Results:**
- 3B model **outperforms Gemma-2 (9B)** with execution accuracy of **56.23%**
- On unseen schemas: **85% execution accuracy**
- RL with support tasks improved execution accuracy by **5.03%** over SFT alone
- Triple relationship improves accuracy more than key-value extraction alone
- GPT-4o achieves Google-BLEU of 0.8017 (confirming our Figure 4 numbers)
- **First application of reinforcement learning to Text2Cypher**

**Relevance:** Most directly relevant paper. Proves (a) 3B can beat 9B with intermediate reasoning + RL, (b) semantic extraction as intermediate steps is effective CoT, (c) GRPO works for Text2Cypher, (d) support tasks = chain-of-thought decomposition.

---

#### Paper 2: Enhancing KG Interactions: Text-to-Cypher Pipeline (Relevance: 9/10)

- **Full title:** "Enhancing knowledge graph interactions: A comprehensive Text-to-Cypher pipeline with large language models"
- **URL:** https://www.sciencedirect.com/science/article/pii/S0306457325002213
- **Venue:** Information Processing and Management, Vol. 63, Issue 1, Article 104280 (Jan 2026)
- **Authors:** Chao Yang (corresponding), Changyi Li, Xiaodu Hu, Hao Yu, Jinzhi Lu

**Methodology:**
1. **Template-Based Synthetic Data Generation:** Creates diverse training samples from target KG schema
2. **SFT + Preference Learning (DPO/RLHF-style):** Two-stage -- basic Cypher generation, then preference learning to prefer accurate queries
3. **Context-Aware Retrieval:** Dynamic schema element incorporation at inference

**Key Results:**
- **ChatGPT-4o baseline (vanilla):** ~48.5% component matching accuracy
- **ChatGPT-4o with context-aware prompting:** 72.1% execution accuracy (+23.6pp)
- **CodeLlama-13B with full pipeline:** 69.2% execution accuracy
- **Gap between 13B open-source and GPT-4o: only 2.9 percentage points**

**Relevance:** Concrete evidence smaller models match GPT-4o with right pipeline. Three-component approach (synthetic data + preference learning + schema-aware retrieval) is a template for our methodology.

---

#### Paper 3: Robust NL-to-Cypher: Chain of Prompts (Relevance: 8/10)

- **Full title:** "Robust NL-to-Cypher translation for KBQA: Harnessing Large Language Model with Chain of Prompts"
- **URL:** https://link.springer.com/chapter/10.1007/978-981-99-7224-1_25
- **Venue:** CCKS 2023, Springer LNCS
- **Authors:** G. Feng, G. Zhu, S. Shi, Y. Sun, Z. Fan, S. Gao, J. Hu

**Methodology:**
Multi-stage "Chain of Prompts" pipeline:
1. Pre-processing (clean/normalize input)
2. Entity and Relation Extraction (LLM prompt)
3. Schema Linking (map to KG schema)
4. Cypher Query Generation (combine all intermediate results)
5. KG Retrieval + LLM-Enhanced Response

**Key Results:**
- **F1 score: 0.94269** on CCKS2023 military unmanned systems KG task

**Relevance:** Foundational precedent for our thesis. Demonstrates decomposing NL-to-Cypher into chained steps dramatically improves accuracy. We internalize this chain-of-prompts into chain-of-thought fine-tuning.

---

#### Paper 4: Auto-Cypher / SynthCypher (Relevance: 8/10)

- **Full title:** "Auto-Cypher: Improving LLMs on Cypher Generation via LLM-Supervised Generation-Verification Framework"
- **URL:** https://arxiv.org/abs/2412.12612
- **Venue:** NAACL 2025 (Short Papers)
- **Authors:** Aman Tiwari, Shiva Krishna Reddy Malay, Vikas Yadav, Masoud Hashemi, Sathwik Tejaswi Madhusudhan (ServiceNow)

**Methodology:**
Five-stage automated pipeline for synthetic Cypher training data:
1. Schema Generation (700 domains via Mixtral-8x22B, validated by GPT-4)
2. NL Question + Ground Truth Answer Generation (109 query taxonomies)
3. **LLM-As-Database-Filler** (novel: generates synthetic databases conditioned on predetermined answers)
4. Cypher Query Generation with chain-of-thought (Llama 3.1 70B / Mixtral 8x22B / GPT-4)
5. Validation (execution checks, up to 5 retries, GPT-4 as semantic judge)

**Dataset produced:** SynthCypher -- 29,838 validated instances, 528 schemas, 700 domains, 109 query types. **100% executable** (vs. Neo4j Labs' ~50%).

**Key Results:**
| Model | SynthCypher Test | SPIDER-Cypher |
|---|---|---|
| Llama-3.1-8B (base) | 40.2% | 37.9% |
| Llama-3.1-8B + SynthCypher | **71.4%** | **62.2%** |
| Mistral-7B + SynthCypher | 69.4% | 61.3% |
| Code-Qwen-2.5-7B + SynthCypher | 70.1% | 62.1% |

Up to **40% absolute improvement** over base models.

**Relevance:** Methodology for high-quality training data generation with guaranteed executability. The CoT query generation step supports our thesis approach.

---

### Tier 2: Supporting Papers

#### Paper 5: Text2Cypher Data Pruning (Relevance: 7/10)

- **Full title:** "Text2Cypher: Data Pruning using Hard Example Selection"
- **URL:** https://arxiv.org/abs/2505.05122
- **Venue:** arXiv, May 2025 (Neo4j)
- **Authors:** Makbule Gulcin Ozsoy et al.

**Key Results:**
| Approach | Steps | GLEU (Translation) | EM (Translation) | GLEU (Execution) | EM (Execution) |
|---|---|---|---|---|---|
| Full dataset | 2,500 | 0.7585 | 0.3642 | 0.2534 | 0.2740 |
| Random sampling | 1,000 | 0.6971 | 0.2048 | 0.2121 | 0.2550 |
| Hard example (best) | 1,000 | 0.7140 | 0.2599 | 0.2473 | 0.2639 |

Reduces training cost by >50% with minimal quality loss. Complexity-based selection best for translation; Cypher-specific selection best for execution.

---

#### Paper 6: GraphRAFT (Relevance: 7/10)

- **Full title:** "GraphRAFT: Retrieval Augmented Fine-Tuning for Knowledge Graphs on Graph Databases"
- **URL:** https://arxiv.org/abs/2504.05478
- **Venue:** arXiv, April 2025 (revised November 2025)
- **Authors:** Alfred Clemedtson, Borun Shi

**Methodology:**
Three-stage retrieve-and-reason:
1. Ground-Truth Cypher Synthesis (GPT-4o-mini + few-shot prompting + vector similarity)
2. Grounded Constrained Decoding (logits-level masking for syntactic correctness)
3. Local Subgraph Reasoning (Llama-3.1-8B-Instruct over retrieved subgraphs)

**Hardware:** Single 40GB A100 GPU. Uses Gemma2-9b-text2cypher for query generation.

**Key Results:**
| Metric | STaRK-Prime | Previous Best | STaRK-Mag | Previous Best |
|---|---|---|---|---|
| Hit@1 | **63.71%** | 40.9% | **71.05%** | 65.40% |
| MRR | **68.99%** | 51.2% | **76.92%** | 69.80% |

+22.81pp and +17.79pp over best baselines on Hit@1 and MRR. With only 10% training data, Hit@1 still reaches 41.07% (vs 44.20% full). First solution designed for KGs in native graph databases.

---

#### Paper 7: BYOKG-RAG (Relevance: 6/10)

- **Full title:** "BYOKG-RAG: Multi-Strategy Graph Retrieval for Knowledge Graph Question Answering"
- **URL:** https://aclanthology.org/2025.emnlp-main.1417/
- **Venue:** EMNLP 2025 (Main), Suzhou, China
- **Authors:** Costas Mavromatis et al. (AWS)

**Methodology:**
Two-stage: KG-Linker (LLM generates entities, paths, OpenCypher queries, draft answers) → Graph Retrieval (4 strategies: entity linking, path retrieval, graph query, triplet retrieval). Iterative refinement (2 iterations with self-termination).

**Key Results:**
| Benchmark | ByoKG-RAG | Best Baseline |
|---|---|---|
| WebQSP | **87.1%** Hit | 84.1% (FiDeLiS) |
| CWQ | **71.1%** Hit | 71.4% (FiDeLiS) |
| CronQuestions | **65.5%** Hit | 60.2% |
| Northwind (Enterprise) | **64.9%** Judge | 55.3% |

Average +4.5pp over strongest baselines. Zero-shot (no task-specific training).

---

#### Paper 8: Live-Fetched Schema with Error Correction (Relevance: 6/10)

- **Full title:** "Real-Time Text-to-Cypher Query Generation with Large Language Models for Graph Databases"
- **URL:** https://www.mdpi.com/1999-5903/16/12/438
- **Venue:** Future Internet (MDPI), Vol. 16, Issue 12, Nov 2024 (14 citations)
- **Authors:** Markus Hornsteiner et al. (University of Regensburg)

**Methodology:** Modular pipeline with live-fetched schema from Neo4j, GPT-4 Turbo for generation, iterative error correction loop (feed error messages back), chat history for multi-turn support. Schema-agnostic, immediately deployable.

---

### Tier 3: Additional Benchmarks & Datasets

#### Mind the Query (EMNLP 2025 Industry Track)
- URL: https://aclanthology.org/2025.emnlp-industry.133/
- By IBM Research (Vashu Chauhan et al.)
- 27,529 NL-Cypher pairs across 11 real-world graph datasets
- Validated through automated + manual review, categorized by complexity

#### ZOGRASCOPE (arXiv, March 2025)
- URL: https://arxiv.org/abs/2503.05268
- First manually annotated benchmark for property graphs/Cypher
- Built on Pole graph (crime KG, 61K nodes, 106K edges), ~5,555 samples
- Results: Fine-tuned 3B-7B models achieve **98% IID accuracy** but only **70-75% compositional** and **20-25% length generalization**
- Critical insight: **CoT should help most with compositional and length generalization**

#### CypherBench (ACL 2025, arXiv:2412.18702)
- 11 large-scale property graphs, 7.8M entities, 10K+ questions
- Best LLM: 60.18% execution accuracy; **no LLM under 10B surpassed 20%**
- Motivates chain-of-thought for small models on complex KGs

#### CoBGT: BERT + GraphSAGE + Transformer (Applied Sciences, 2024)
- Three-module architecture: BERT for key-value extraction, GraphSAGE for relation-property prediction, Transformer for Cypher generation
- Non-LLM decomposition approach -- provides interesting contrast point for thesis
- Each module handles a distinct sub-task (another instantiation of CoT-style reasoning)

#### Text2Cypher Across Languages (arXiv:2506.21445, June 2025)
- Neo4j's multilingual extension (English, Spanish, Turkish)
- English > Spanish > Turkish performance
- Multilingual fine-tuning narrows cross-language gaps

---

## LoRA vs QLoRA: Detailed Comparison

### QLoRA Paper

**Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).** "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023.* arXiv:2305.14314.

### How LoRA Works

LoRA modifies `Y = WX + b` to `Y = (W + BA)X + b`. Freezes all pre-trained weights W, trains only small adapter matrices B and A (low-rank decomposition). Trains 0.5-5% of original parameters.

### How QLoRA Extends LoRA

Backpropagates gradients through frozen, 4-bit quantized pretrained model into LoRA adapters. Three innovations:
1. **4-bit NormalFloat (NF4):** Information-theoretically optimal for normally distributed weights. NF4 consistently outperforms FP4 by ~1pp on MMLU.
2. **Double Quantization (DQ):** Quantizes quantization constants, saving ~0.37 bits/param additional.
3. **Paged Optimizers:** NVIDIA unified memory for memory spike handling.

### The Core Claim: "Quantization Does Not Hurt"

4-bit QLoRA with NF4 **matches 16-bit full fine-tuning and 16-bit LoRA** on established benchmarks:
- On 5-shot MMLU, NF4 QLoRA replicates 16-bit LoRA scores for LLaMA 7B-65B
- NF4 + double quantization matches BFloat16 performance; FP4 is ~1pp behind
- Best model (Guanaco) reached 99.3% of ChatGPT's Vicuna benchmark performance
- Over 1,000 models fine-tuned to validate

### Memory Savings

| Method | VRAM for 7B | VRAM for 65B |
|--------|------------|-------------|
| Full fine-tuning | ~60 GB | ~780 GB |
| LoRA (16-bit) | ~2 GB+ | ~130 GB+ |
| QLoRA (4-bit) | ~0.5 GB+ | ~48 GB |

### Counter-Evidence (2024-2025)

1. **"Accurate and Efficient Fine-Tuning of Quantized LLMs" (TACL 2024/2025):** Combining quantization with LoRA causes imbalance -- overly complex adapter inputs/outputs vs low trainability → underfitting. QLoRA achieves **80-90% of full fine-tuning quality** on some tasks.
2. **CLoQ (arXiv:2501.18475, 2025):** QLoRA "overlooks the critical importance of subsequent LoRA fine-tuning." Quantization discrepancies significantly impact LoRA initialization.
3. **Proposed fixes:** Q-BLoRA, QuAILoRA/IR-QLoRA, QDoRA (2025)

### BF16 vs INT4 Precision ("Give Me BF16 or Give Me Death?" arXiv:2411.02355)

| Model | BF16 (baseline) | W4A16-INT (4-bit) | Recovery % |
|-------|-----------------|-------------------|-----------|
| Llama-3.1-8B | 74.06 | 73.11 | 98.7% |
| Llama-3.1-70B | 84.40 | 84.00 | 99.5% |
| Llama-3.1-405B | 86.79 | 86.78 | ~100% |

**Key patterns:** Larger models more robust. NF4 outperforms FP4 by ~1pp. Training still done in BF16; INT4 for inference quantization. QLoRA bridges: frozen weights INT4, gradients in BF16. Memory: Gemma-3-27B in BF16 = ~54 GB vs INT4 = ~14.1 GB (3.8x smaller). Accuracy difference consistently **under 1%** for modern models.

### Our Approach

Train with QLoRA matching Neo4j's exact configuration. This makes the comparison direct: same model, same precision, same LoRA hyperparameters — only the training data differs (CoT vs direct-answer). Any performance gain is cleanly attributable to chain-of-thought reasoning, with no precision confound for reviewers to challenge.

---

## Infrastructure & Costs

### Training Data Generation: GPT-oss-120B

**GPT-oss-120B** is an open-weight model released by OpenAI (August 2025):
- 117B parameters (MoE), reasoning-capable
- 131K token context window
- Trained via RL + distillation from o3 and other frontier models

**Pricing across providers:**
| Provider | Input (per 1M tokens) | Output (per 1M tokens) |
|----------|----------------------|------------------------|
| Galaxy.ai | **$0.02** | **$0.10** |
| Clarifai | $0.09 | $0.36 |
| DeepInfra | $0.09 | $0.45 |
| OpenAI (direct) | $0.15 | $0.60 |
| Together.ai | Available | Varies |

**Cost estimate for 35,000 training examples:**

| Scenario | Input tokens | Output tokens | Galaxy.ai | OpenAI |
|----------|-------------|--------------|-----------|--------|
| Conservative (500in/200out) | 17.5M | 7.0M | **$1.05** | $6.83 |
| Generous (1000in/500out) | 35M | 17.5M | **$2.45** | $15.75 |

Training data generation is remarkably cheap: **$1-$16** depending on provider and token counts.

### Fine-Tuning: Together.ai

**Note:** Gemma-2-9B is NOT currently supported. Closest alternative: **Gemma-3-12b-it** (12B).

**Pricing for Supervised Fine-Tuning (SFT):**
| Model Size | LoRA (per 1M tokens) | Full (per 1M tokens) |
|------------|---------------------|---------------------|
| Up to 16B | **$0.48** | **$0.54** |
| 17B-69B | **$1.50** | **$1.65** |
| 70B-100B | **$2.90** | **$3.20** |

**DPO Pricing:**
| Model Size | LoRA (per 1M tokens) | Full (per 1M tokens) |
|------------|---------------------|---------------------|
| Up to 16B | **$1.20** | **$1.35** |
| 17B-69B | **$3.75** | **$4.12** |

**Cost estimate (35K examples, ~500 tokens each, 3 epochs):**
- Total tokens: 35K x 500 x 3 = 52.5M
- LoRA SFT (up to 16B): 52.5 x $0.48 = **~$25.20 total**

### Fine-Tuning: Digital Research Alliance of Canada (FREE)

**Available clusters with GPUs:**
| Cluster | GPU Type | VRAM | Notes |
|---------|----------|------|-------|
| **Narval** | NVIDIA A100-40GB | 40 GB | 632+ GPUs |
| **Fir** | NVIDIA H100-80GB | 80 GB | Successor to Cedar |
| **Nibi** | NVIDIA H100-80GB | 80 GB | 288 H100 GPUs |
| **Rorqual** | NVIDIA H100-80GB | 80 GB | No internet on compute nodes |
| **TamIA** | NVIDIA H100 + H200 | 80/141 GB | Quebec-focused |

**Access process:**
1. Create CCDB account at https://ccdb.alliancecan.ca
2. Apply for role with supervisor's CCRI
3. Wait for sponsor approval
4. Set up MFA
5. Annual renewal required

**Two tiers:**
- **Rapid Access Service (RAS):** Immediate, up to 200 core-years CPU, GPU on opportunistic basis. No application needed.
- **Resource Allocation Competition (RAC):** Annual competition (opens September), guaranteed priority GPU access. PI must apply.

**SLURM job submission:**
```bash
sbatch --time=1:00:00 --account=def-supervisor --gres=gpu:1 job.sh
```

**Storage:** $HOME (persistent), $SCRATCH (purged every 3 months if unused), $SLURM_TMPDIR (deleted after job).

**IMPORTANT:** Rorqual compute nodes have NO internet -- pre-download all datasets and model weights.

### Evaluation: HuggingFace Evaluate Library

```python
import evaluate

# Google-BLEU (primary metric)
google_bleu = evaluate.load("google_bleu")
result = google_bleu.compute(
    predictions=["MATCH (n) RETURN n"],
    references=[["MATCH (n) RETURN n"]]
)

# Exact Match
exact_match = evaluate.load("exact_match")
result = exact_match.compute(
    predictions=["MATCH (n) RETURN n"],
    references=["MATCH (n) RETURN n"]
)
```

**GLEU details:** Records n-gram sub-sequences (1-4 tokens), computes recall and precision, returns minimum. Symmetrical. Uses `nltk.translate.gleu_score`. Reference: Wu et al. (2016), arXiv:1609.08144.

**Execution Accuracy:** NOT built into HF Evaluate -- requires custom implementation executing both predicted and reference queries against Neo4j and comparing result sets.

### Inference: HuggingFace Endpoints

| GPU | VRAM | Hourly Rate (AWS) |
|-----|------|-------------------|
| NVIDIA T4 (1x) | 14 GB | $0.50 |
| NVIDIA L4 (1x) | 24 GB | $0.80 |
| NVIDIA A10G (1x) | 24 GB | $1.00 |
| NVIDIA A100 (1x) | 80 GB | $2.50 |
| NVIDIA H100 (1x) | 80 GB | $4.50 |

For Gemma-2-9B in BF16 (~18GB): 1x L4 ($0.80/hr) or 1x A10G ($1.00/hr).

---

## Research Plan

### Step 1: Reproduce Baseline
Run Neo4j's fine-tuned Gemma-2-9B on their test set to get comparable numbers.
- **Current file:** `run_neo4j.py` (uses Modal for GPU, loads model with 4-bit quantization)
- Uses A10G GPU via Modal
- Need to run full evaluation on 4,833 test examples
- **Target numbers to reproduce:** GLEU 0.5560, ExactMatch 0.2104

### Step 2: Generate Chain-of-Thought Training Data
For each of the 39,554 training examples:
- **Input:** (Few-shot examples, Schema, Question)
- **Output:** (Reasoning trace, Cypher result)
- Few-shot = 9 examples of (Schema, Question, Reasoning, Result)
- **Generation platform:** GPT-oss-120B via Galaxy.ai ($1-$3) or Clarifai ($4-$10)
- **Reasoning decomposition — QDECOMP+InterCOL adapted for Cypher (initial strategy, may need iteration):**
  Based on Tai et al. (EMNLP 2023): decompose question into sub-questions, identify relevant schema elements (InterCOL) at each step, then construct query. Adapted from SQL to Cypher:
  1. **Sub-question decomposition:** Break the question into logical sub-questions aligned with Cypher clause order (MATCH → WHERE → RETURN/aggregation)
  2. **Intermediate schema linking (InterCOL):** For each sub-question, explicitly name the relevant node labels, relationship types, and properties from the schema
  3. **Pattern identification:** Determine graph traversal pattern (single hop, multi-hop, variable-length path, aggregation, filtering)
  4. **Step-by-step Cypher construction:** Build MATCH clause from identified patterns, add WHERE conditions from extracted properties (cf. MDPI paper's key-value extraction), construct RETURN with any aggregations
  5. **Final query assembly and validation**

### Step 3: Train New Gemma-2-9B with Reasoning
- **Method:** QLoRA — matching Neo4j's exact config (4-bit NF4, double quant, bf16 compute, r=64, alpha=64, dropout=0.05, paged_adamw_8bit)
- **Rationale:** Same precision as baseline → only variable is CoT training data → clean comparison, no precision confound
- **Platform:** Digital Research Alliance of Canada (free, H100s/A100s) — preferred since QLoRA requires custom training loop
- **Fallback platform:** Together.ai (~$25) if they support QLoRA, otherwise RunPod/Lambda replicating Neo4j's setup
- Train model to output explanation + Cypher (not just Cypher)

### Step 4: Ablation Studies
- Self-consistency (multiple samples, vote on best) -- cf. CSC-SQL, USC
- Few-shot prompting (0, 3, 9 examples)
- Agentic self-healing loop (re-send with error messages) -- cf. MAC-SQL, SQL-of-Thought
- Schema filtering (cf. Neo4j's schema filtering paper)
- RL on fine-tuned model (optional, GRPO following MDPI paper approach)
- Number of self-consistency calls
- **(Optional) bf16 LoRA vs QLoRA on CoT data** -- if time permits, train a bf16 LoRA variant to quantify precision contribution on top of CoT

### Step 5: Write Paper
- Compare against Neo4j baseline (GLEU 0.5560, EM 0.2104), show improvement
- Compare against unfine-tuned GPT-4o (GLEU 0.6293, EM 0.3173) as stretch goal
- Ablation tables for each technique
- Mathematical formalization of approach
- Discuss limitations of 2024 dataset (data leakage, paraphrase contamination, distribution shift)
- Cost analysis (small model vs GPT-4o for production deployment)
- Target conferences: EMNLP and similar NLP venues

---

## Cross-Paper Synthesis: Key Themes

1. **Chain-of-thought decomposition works for Text2Cypher:** Papers 1, 3, and Auto-Cypher all demonstrate that decomposing Cypher generation into intermediate steps significantly improves accuracy. Paper 3 achieves 0.94 F1 with chain-of-prompts; Paper 1 shows 3B outperforming 9B with semantic intermediate steps.

2. **Small models CAN match large models with the right approach:** Paper 2 (CodeLlama-13B at 69.2% vs. GPT-4o at 72.1%), Paper 1 (3B outperforming 9B), Auto-Cypher (7B models reaching 70%+), ZOGRASCOPE (3B achieving 98% IID).

3. **Reinforcement learning with intermediate rewards is promising:** Paper 1's GRPO with key-value and triple-based rewards is the first RL application to Text2Cypher and shows strong results.

4. **Data quality matters more than quantity:** Hard example selection halves training with smart selection; Auto-Cypher generates validated data; Schema filtering reduces noise. Focused, high-quality training > scale.

5. **Constrained decoding and error correction provide safety nets:** GraphRAFT's logits-level masking and live-fetched schema paper's iterative correction both ensure executable outputs.

---

## Project Structure

```
Cot2Cypher/
├── CLAUDE.md              # This file
├── research_prompts.md    # Sub-agent research prompts
├── run_neo4j.py           # Baseline inference script (Modal + A10G GPU)
└── .gitignore
```

### Our Enhanced Prompt Format (QDECOMP+InterCOL adapted for Cypher)

**Note:** This is the initial reasoning format. May need adaptation since Cypher's graph patterns differ from SQL's tabular joins.

```
Generate Cypher statement to query a graph database.
Use only the provided relationship types and properties in the schema.
Schema: {schema}
Question: {question}

Think step by step:
1. Break down the question: What are the sub-questions?
2. For each sub-question, identify the relevant node labels, relationship types, and properties from the schema.
3. Determine the graph pattern (single hop, multi-hop, variable-length path, aggregation, filtering).
4. Construct the Cypher query step by step, building MATCH, WHERE, and RETURN clauses.

Reasoning: {reasoning}
Cypher output: {cypher}
```

**Example reasoning trace (target format for GPT-oss-120B generation):**
```
Reasoning:
1. Sub-questions: (a) Which person has name "John"? (b) What movies did that person act in?
2. Schema elements: (a) Node: Person (name property) (b) Relationship: ACTED_IN (c) Node: Movie (title property)
3. Pattern: Two-hop traversal Person→ACTED_IN→Movie with property filter on Person.name
4. Construction:
   - MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
   - WHERE p.name = "John"
   - RETURN m.title
Cypher output: MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE p.name = "John" RETURN m.title
```

---

## Cost Summary

| Component | Platform | Cost |
|-----------|----------|------|
| Training data generation (35K examples) | GPT-oss-120B via Galaxy.ai | **$1-$3** |
| QLoRA fine-tuning (Gemma-2-9B, 3 epochs) | Together.ai | **~$25** |
| QLoRA fine-tuning (Gemma-2-9B) | Alliance Canada (Narval/Fir) | **Free** |
| Evaluation inference | Modal (A10G) / HF Endpoints | **$1-$5** |
| **Total (Together.ai route)** | | **~$30** |
| **Total (Alliance Canada route)** | | **~$5** |

---

## Important Notes

- We are using the **2024 baseline and dataset** -- must acknowledge known limitations (data leakage, paraphrase contamination) that were posted before the 2025 release
- We **match Neo4j's QLoRA config exactly** (4-bit NF4, double quant, bf16 compute, r=64, alpha=64, paged_adamw_8bit) to eliminate precision as a confound
- The only variable is training data: CoT reasoning traces vs direct-answer Cypher
- The model is for **research purposes** -- Gemma license is non-commercial
- May attempt fine-tuning a more advanced model to find the line to beat GPT-4o
- Target conferences: EMNLP and similar NLP venues
- ZOGRASCOPE results suggest CoT should help most with **compositional generalization** and **length generalization** -- worth testing on that benchmark too
- CypherBench shows no LLM under 10B surpasses 20% on large-scale KGs -- our CoT approach could make a significant impact here

---

## Complete Source URLs

### Primary Benchmark
- Paper: https://arxiv.org/abs/2412.10064
- ACL Anthology: https://aclanthology.org/2025.genaik-1.11/
- 2024 Dataset: https://huggingface.co/datasets/neo4j/text2cypher-2024v1
- 2025 Dataset: https://huggingface.co/datasets/neo4j/text2cypher-2025v1
- Fine-tuned Gemma-2-9B: https://huggingface.co/neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1
- Fine-tuned Gemma-3-27B: https://huggingface.co/neo4j/text-to-cypher-Gemma-3-27B-Instruct-2025.04.0
- GitHub: https://github.com/neo4j-labs/text2cypher

### Neo4j Blog Posts
- Benchmarking: https://neo4j.com/blog/developer/benchmarking-neo4j-text2cypher-dataset/
- Fine-tuned model: https://neo4j.com/blog/developer/fine-tuned-text2cypher-2024-model/
- Model struggles: https://neo4j.com/blog/developer/text2cypher-model-struggles-dataset-improvements/
- Schema filtering: https://arxiv.org/abs/2505.05118
- Hard example selection: https://arxiv.org/abs/2505.05122
- Iterative refinement: https://medium.com/neo4j/exploring-iterative-refinement-for-text2cypher-ea99aeb28949
- Multilingual: https://arxiv.org/abs/2506.21445

### Key Papers
- MDPI RL paper: https://www.mdpi.com/2076-3417/15/15/8206
- Comprehensive pipeline: https://www.sciencedirect.com/science/article/pii/S0306457325002213
- Chain of Prompts: https://link.springer.com/chapter/10.1007/978-981-99-7224-1_25
- Auto-Cypher/SynthCypher: https://arxiv.org/abs/2412.12612
- GraphRAFT: https://arxiv.org/abs/2504.05478
- BYOKG-RAG: https://aclanthology.org/2025.emnlp-main.1417/
- Live schema + error correction: https://www.mdpi.com/1999-5903/16/12/438
- Mind the Query (IBM): https://aclanthology.org/2025.emnlp-industry.133/
- ZOGRASCOPE: https://arxiv.org/abs/2503.05268
- CypherBench: https://arxiv.org/abs/2412.18702

### Text-to-SQL Transfer Papers
- **STaR-SQL (THE 18% paper):** https://arxiv.org/abs/2502.13550
- CSC-SQL: https://arxiv.org/abs/2505.13271
- DIN-SQL: https://arxiv.org/abs/2304.11015
- Self-Consistency: https://arxiv.org/abs/2203.11171
- DTS-SQL: https://arxiv.org/abs/2402.01117
- SQL-PaLM: https://arxiv.org/abs/2306.00739
- CHESS: https://arxiv.org/abs/2405.16755
- CHASE-SQL: https://arxiv.org/abs/2410.01943
- MAC-SQL: https://arxiv.org/abs/2312.11242
- SQL-of-Thought: https://arxiv.org/abs/2509.00581
- QDECOMP/Break: https://arxiv.org/abs/2001.11770
- Weakly supervised QD: https://arxiv.org/abs/2112.06311
- CoT for Text-to-SQL: https://arxiv.org/abs/2305.14215
- ExCoT (iterative CoT+DPO): https://arxiv.org/abs/2503.19988
- Struct-SQL (structured CoT): https://arxiv.org/abs/2512.17053
- Reasoning-SQL (GRPO partial rewards): https://arxiv.org/abs/2503.23157
- Arctic-Text2SQL-R1 (GRPO simple rewards): https://arxiv.org/abs/2505.20315
- AGRO-SQL (agentic GRPO): https://arxiv.org/abs/2512.23366
- DPO+CoT critical finding: https://arxiv.org/abs/2502.11656
- PaVeRL-SQL: https://arxiv.org/abs/2509.07159
- SDE-SQL: https://arxiv.org/abs/2506.07245
- RSL-SQL (schema linking): https://arxiv.org/abs/2411.00073

### Infrastructure
- QLoRA paper: https://arxiv.org/abs/2305.14314
- BF16 vs INT4: https://arxiv.org/abs/2411.02355
- Together.ai fine-tuning: https://docs.together.ai/docs/fine-tuning
- Together.ai pricing: https://www.together.ai/pricing
- Alliance Canada: https://docs.alliancecan.ca/wiki/Getting_started
- HuggingFace Evaluate: https://huggingface.co/docs/evaluate/index
- HuggingFace pricing: https://huggingface.co/pricing