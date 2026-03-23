**Cypher Generation Research**

Steps:

1. Find out if anyone’s done this yet   
2. Run current baseline fine-tuned Gemma-2-9B by Neo4J in attempt to get comparable results to their paper   
3. Generate 39,554 training reasoning examples and train new Gemma-2-9B model   
4. Ablation with self-consistency, few shot, agentic loop, maybe RL  
5. Collect sources for each technique, and general state  
6. Write paper, show some math 

Generating chain of thought reasoning examples:

- For each one: (Few Shot, Schema, Question) \-\> (Reasoning, Result)  
  - Few shot is 9 examples of Schema, Question, Reasoning, Result

Training:

- LORA on Together .ai   
- Note: we need to specify that we are not quantizing unlike Neo4J who did. (QLORA paper says shouldn’t make a difference) Also, apparently we are using a different optimizer?  
  - "We compare our novel process (trained in **bf16**) against the baseline (trained in **int4/QLoRA**). While some performance lift may be attributed to the higher precision, the significant gain of X% suggests our process provides benefits beyond just precision recovery."

Note: *current baseline open small models still underperform gpt-4* 

- *Currently 30-35% of questions answered correctly with Neo4J baseline*

Ideas for improvement over 2024 Neo4J baseline:

- Few shot prompting  
- Chain of thought reasoning (finetuning)  
  - Generate reasoning with strong LLM, train smaller LLM on this with *a few thousand examples*. (35000 examples at 1000 tokens in and out would be \~25$ with gpt-oss-120B on [anything.ai](http://anything.ai) or $9 per hour on hugging face (58 iq vs gpt-4o’s 25)) Train model to output an explanation plus the cypher   
    - Method got 18% better in text-to-sql Spider benchmark over direct answer fine-tuned model in [this paper](https://arxiv.org/abs/2502.13550)   
  - Allow better output plus ability see where model is failing  
  - Build on schema linking research  
  - See new ant group model, trained on reasoning  
  - Good reasoning decomposition can be seen here: [QDECOMP](https://arxiv.org/abs/2305.14215)  
- Self-consistency  
- Agentic loop and self-healing  
- Continued RL on fine-tuned model 

- Note: using 2024 baseline and dataset after concerns were posted and 2025 was released, make sure to mention limitations mentioned [here](https://medium.com/neo4j/introducing-the-fine-tuned-neo4j-text2cypher-2024-model-b2203d1173b0) regarding 2024 dataset   
- **Success criteria**: get a better result than the base fine-tuned Neo4J fine-tuned Gemma-2-9B  
  - Note: may be worse than gpt-4o, that’s ok, since this is much cheaper to run  
  - Note: may try to fine-tune more advanced model, and see where line is to beat gpt-4o, try and simultaneously beat Neo4J’s fine-tuned version of that given model 

**Potential Paper Topics**

**SQL**  
Potential Paper Topics:

* Privacy (rag over schema, send SQL back, high-stakes private info)  
  * Business cases  
  * Find out what privacy means in academia and how it’s defined   
* Cost (no fine-tuning, gpt-4o-mini, real deployment)  
* Problem: current sota benchmarks are super high

Potential additions:

* Self-healing loop by re-sending with errors after validation \<- in the works  
  * SDE-SQL  
* Exploration, ie trying out keywords \<- in the works  
* rag over few-shot examples  
* Collecting stats \<- in the works   
  * **Execution Accuracy**: valid SQL returns correct results.  
  * **Syntactic Validity Rate**: parses without errors.  
  * **Turnaround Time**: user input to visual output.  
  * **User Satisfaction**: survey feedback from analysts.  
* Ablation  
  * Which models performed the best  
  * How many examples, what about 0 examples  
  * How many self-consistency calls

Cypher querying knowledge graphs

* [NEO4J uses fine-tuning](%20https://arxiv.org/pdf/2412.10064) (26 citations)  
  * Fine-tuned models beat closed-source foundational models at the time, best performance from Gpt-4o fine-tuned  
  * Train and test dataset [available](https://huggingface.co/datasets/neo4j/text2cypher-2024v1)  
    * Note: [newer dataset available](https://huggingface.co/datasets/neo4j/text2cypher-2025v1), and [blog post by Neo4J](https://medium.com/neo4j/neo4j-text2cypher-analyzing-model-struggles-and-dataset-improvements-0b965fd3ebfa) saying they will improve errors in old dataset  
  * [Models](https://huggingface.co/neo4j/models) available, but only gemma 2024, and two mysterious 2025 gemma models   
    * Instructions for training and using available [here](https://huggingface.co/neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1)  
  * Note: code not available, but hugging face eval available [exact match](https://huggingface.co/spaces/evaluate-metric/exact_match) and [GLEU](https://huggingface.co/spaces/evaluate-metric/google_bleu)  
* [Schema filtering text2cypher for cost](https://arxiv.org/abs/2505.05118)  
  * Pruning schema information only slightly helped, and although explanations of schema adjustments, no code available, likely better as a citation then a foundation for work   
  * Focused on subset of query database with database access (available)  
  * Used 3 models, not fine-tuned  
* [See Gov code](https://github.com/bcgov/unity-ai/tree/main/applications)  
* Current idea: try to beat Neo4J’s results on 2024 dataset using their released fine-tuned model with prompt engineering techniques used in gov project   
  * Few shot, self-consistency, potentially fine-tuning with reasoning built in  
  * Agentic loops: self-healing and exploration  
  * Transferring tech from text-2-sql papers  
  * Use [https://www.together.ai/](https://www.together.ai/) for generating reasoning dataset

* Live-fetched graph schema and optional few-shot examples with iterative error correction and chat functionality: [https://www.mdpi.com/1999-5903/16/12/438](https://www.mdpi.com/1999-5903/16/12/438) (14 citations)  
  * Gpt-4 mini  
  * To improve: async self-consistency, possibly refinement or llm-as-judge, possible cheaper model  
* Robust NL-to-Cypher translation for KBQA: Harnessing Large Language Model with Chain of Prompts: [https://link.springer.com/chapter/10.1007/978-981-99-7224-1\_25](https://link.springer.com/chapter/10.1007/978-981-99-7224-1_25)   
  * We prompt reasoning steps without chain of prompts \- cheaper   
  * Do it with better modern LLM’s  
* Compare each method in Text2Cypher’s dataset or spCQL

Potentially look at SPARQL as well

**Next steps**

- Make enhanced text2cypher work on local for baseline, make changes  
  - If we can make more rigorous, with RL and finetuning, showing the math, we can do better  
  - Hugging face evaluate library   
  - Translation based and execution based evaluation with Google Bleu   
- Idea: Use their 2025 model and dataset, and old method, compare results with text-to-sql methods above and fine-tuned bot with reasoning built in  
  - Possibly use Digital Research Alliance of Canada for compute for fine-tuning

**Summarizer**

Ask ChatGPT \- top conferences EMNLP  
[https://chatgpt.com/c/68c39a86-eb38-8322-a4c5-79513a61e0f5](https://chatgpt.com/c/68c39a86-eb38-8322-a4c5-79513a61e0f5)

* 1\) Sub-questions interesting since we do that    
* 4\) Reducing anchoring interesting 

