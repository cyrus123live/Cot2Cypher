# Research Prompts for Sub-Agents

Run these as sub-agents with full tool permissions (WebSearch, WebFetch, Bash).

---

## Prompt 1: Neo4j Text2Cypher Benchmark

Research the Neo4j text2cypher benchmark and dataset thoroughly. I need to understand:

1. The Neo4j text2cypher benchmark - what it is, how it works, what metrics it uses (exact match, GLEU, execution accuracy)
2. The 2024 dataset and baseline results - how many examples, what models were tested, what scores were achieved. Get EXACT numbers for GLEU and ExactMatch for each model (GPT-4o, GPT-4o-mini, Gemini-1.5-Pro, Gemma-2-9B base, fine-tuned Gemma, fine-tuned GPT-4o, etc.)
3. The 2025 dataset updates - what changed, what limitations were noted about the 2024 dataset
4. Neo4j's fine-tuned models - specifically Gemma-2-9B results, GPT-4o fine-tuned results
5. The training methodology Neo4j used (QLoRA, int4 quantization, optimizer choices)

Key URLs to check:
- https://neo4j.com/blog/developer/benchmarking-neo4j-text2cypher-dataset/
- https://medium.com/neo4j/benchmarking-using-the-neo4j-text2cypher-2024-dataset-d77be96ab65a
- https://huggingface.co/neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1
- https://huggingface.co/datasets/neo4j/text2cypher-2024v1
- https://arxiv.org/html/2412.10064v1
- https://github.com/neo4j-labs/text2cypher
- https://medium.com/neo4j/introducing-the-fine-tuned-neo4j-text2cypher-2024-model-b2203d1173b0
- https://medium.com/neo4j/neo4j-text2cypher-analyzing-model-struggles-and-dataset-improvements-0b965fd3ebfa

Return a detailed summary with all findings, URLs, and specific numbers.

---

## Prompt 2: Chain-of-Thought and Text-to-SQL Techniques

Research the current state of chain-of-thought reasoning for text-to-SQL and text-to-Cypher generation. I need detailed information on:

1. Chain-of-thought fine-tuning for code/query generation - specifically the technique of generating reasoning traces with a strong LLM (like GPT-4) and training a smaller model on those traces
2. The specific paper that got 18% improvement on Spider benchmark using reasoning-enhanced fine-tuning over direct answer fine-tuning - find this paper, its name, authors, methodology, and results. Search arXiv and Google Scholar.
3. QDECOMP - question decomposition for complex queries
4. Self-consistency techniques applied to code/query generation (especially CSC-SQL)
5. Agentic loops and self-healing for query generation (re-sending queries with error messages)
6. Schema linking research - how it helps with text-to-query tasks
7. The Ant Group model trained on reasoning for code generation
8. SDE-SQL and other relevant text-to-SQL methods that could transfer to Cypher
9. RL (reinforcement learning) applied to fine-tuned models for query generation (GRPO, DPO, PPO)

Key URLs to check:
- https://arxiv.org/abs/2505.13271 (CSC-SQL)
- https://arxiv.org/abs/2304.11015 (DIN-SQL)
- https://arxiv.org/abs/2203.11171 (Self-Consistency)
- https://arxiv.org/abs/2402.01117 (DTS-SQL)
- https://arxiv.org/abs/2306.00739 (SQL-PaLM)
- https://arxiv.org/abs/2405.16755 (CHESS)

Search for recent papers (2023-2025), focusing on methods that improve small model performance on structured query generation. Get paper titles, authors, publication venues, key results with SPECIFIC NUMBERS, and URLs.

Return a comprehensive summary organized by technique.

---

## Prompt 3: Specific Papers from Thesis Notes

Research the following specific papers and resources. For each one, fetch the actual page and extract key details (methodology, results, relevance):

1. "Text2Cypher: Data Pruning using Hard Example Selection" - https://arxiv.org/abs/2505.05122 - by Neo4J, about saving training time
2. "Enhancing knowledge graph interactions: A comprehensive Text-to-Cypher pipeline with large language models" - https://www.sciencedirect.com/science/article/pii/S0306457325002213 - showed smaller models can match GPT-4
3. "BYOKG-RAG: Multi-Strategy Graph Retrieval for Knowledge Graph Question Answering" - https://aclanthology.org/2025.emnlp-main.1417/ - agentic graph traversal
4. "Refining Text2Cypher on Small Language Model with Reinforcement Learning Leveraging Semantic Information" - https://www.mdpi.com/2076-3417/15/15/8206 - 3B outperformed 9B with intermediate steps and RL
5. "GraphRAFT: Retrieval Augmented Fine-Tuning for Knowledge Graphs on Graph Databases" - https://arxiv.org/pdf/2504.05478 - SOTA in STaRK datasets
6. Live-fetched graph schema with iterative error correction - https://www.mdpi.com/1999-5903/16/12/438 (14 citations, GPT-4 mini, Dec 2024)
7. "Robust NL-to-Cypher translation for KBQA: Harnessing Large Language Model with Chain of Prompts" - https://link.springer.com/chapter/10.1007/978-981-99-7224-1_25

For each paper, provide: full title, authors, venue/year, methodology summary, key results/numbers, and how it relates to improving text-to-Cypher generation with chain-of-thought reasoning on small models.

Also search for any other recent (2024-2025) papers on text-to-Cypher that may have been missed.

Return a detailed summary for each paper.

---

## Prompt 4: LoRA Training, Platforms, and Infrastructure

Research the following topics related to fine-tuning small language models for query generation:

1. LoRA vs QLoRA for fine-tuning - what are the differences? The QLoRA paper claims quantization shouldn't make a difference - find specifics on this claim, the paper citation, and any counter-evidence
2. Together.ai's fine-tuning API - what models are supported, pricing for LoRA fine-tuning, specifically for Gemma-2-9B. Check https://docs.together.ai/docs/fine-tuning and https://api.together.ai/pricing
3. bf16 vs int4 training precision differences - what performance gaps exist in practice?
4. Digital Research Alliance of Canada - what compute resources are available for academic research, how to access them for ML training. Check https://docs.alliancecan.ca/wiki/Getting_started
5. HuggingFace evaluate library - specifically for text-to-SQL/Cypher evaluation metrics (GLEU, exact match, execution accuracy). Check https://huggingface.co/docs/evaluate/index
6. anything.ai and GPT-oss-120B - what is this, pricing for generating training data at scale (35000 examples)
7. Hugging Face inference endpoints pricing for running large models. Check https://huggingface.co/pricing

Return detailed findings with specific numbers, pricing, and citations where available.
