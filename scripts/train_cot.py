# Run from project root:
#   modal run scripts/train_cot.py                          # train CoT model
#   modal run scripts/train_cot.py --eval                   # evaluate trained model (greedy)
#   modal run scripts/train_cot.py --eval --no-greedy       # evaluate with sampling
#   modal run scripts/train_cot.py --eval --ablation cot-baseline  # ablation

import json
import os
import time

import modal

app = modal.App("llm-thesis-cot-train")

# Base image with all dependencies (cached after first build)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "peft==0.12.0",
        "accelerate==0.33.0",
        "bitsandbytes==0.43.3",
        "trl==0.9.6",
        "rich",
        "safetensors",
        "sentencepiece",
        "hf-transfer",
        "datasets",
    )
)

# Training image adds the JSONL data file (separate layer, doesn't bust pip cache)
train_image = base_image.add_local_file(
    "data/cot_training_data.jsonl", "/data/cot_training_data.jsonl"
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
model_vol = modal.Volume.from_name("cot-model-output", create_if_missing=True)
results_vol = modal.Volume.from_name("neo4j-eval-results", create_if_missing=True)

BASE_MODEL = "google/gemma-2-9b-it"
NEO4J_ADAPTER = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
COT_ADAPTER_DIR = "/models/cot-gemma-2-9b/final"
LOCAL_RESULTS_DIR = "results"
LOCAL_ADAPTER_DIR = "adapter_weights"

# CoT prompt — only difference from Neo4j's is the "Think step by step" line
COT_INSTRUCTION = (
    "Generate Cypher statement to query a graph database.\n"
    "Use only the provided relationship types and properties in the schema.\n"
    "Schema: {schema}\n"
    "Question: {question}\n\n"
    "Think step by step, then provide the Cypher query."
)

# Baseline prompt — matches Neo4j's model card exactly
BASELINE_INSTRUCTION = (
    "Generate Cypher statement to query a graph database. "
    "Use only the provided relationship types and properties in the schema. \n"
    "Schema: {schema} \n Question: {question}  \n Cypher output: "
)


def _parse_cypher(raw_output: str) -> str:
    """Extract Cypher query from model output containing reasoning + Cypher."""
    marker = "Cypher output:"
    idx = raw_output.rfind(marker)
    if idx != -1:
        cypher = raw_output[idx + len(marker) :]
    else:
        # Fallback: use the whole output (model didn't follow format)
        cypher = raw_output

    # Clean up common artifacts
    cypher = cypher.strip("`\n ")
    if cypher.startswith("cypher\n"):
        cypher = cypher[7:]
    cypher = cypher.strip("`\n ")
    for sep in ["**Explanation:**", "\n\nExplanation:", "\n\nNote:"]:
        cypher, _, _ = cypher.partition(sep)
    return cypher.strip()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100-80GB",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/models": model_vol,
    },
    timeout=43200,  # 12 hours
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
)
def train():
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    # --- Load training data ---
    records = []
    with open("/data/cot_training_data.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("reasoning", "").strip():
                records.append(rec)
    print(f"Loaded {len(records)} training examples with valid reasoning.")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "right"  # Right-pad for training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Format as chat template strings ---
    def make_text(record):
        user_content = COT_INSTRUCTION.format(
            schema=record["schema"], question=record["question"]
        )
        assistant_content = (
            f"Reasoning:\n{record['reasoning']}\n\n"
            f"Cypher output: {record['cypher']}"
        )
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    texts = [make_text(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Dataset: {len(dataset)} examples.")
    print(f"Example (first 500 chars):\n{texts[0][:500]}")

    # --- Model + QLoRA (matching Neo4j's exact config) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        attn_implementation="eager",  # Required for Gemma-2 (soft-capping)
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- Completion-only data collator ---
    # Only compute loss on assistant's response (reasoning + cypher), not the prompt
    response_template = "<start_of_turn>model\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # --- Training config (matching Neo4j's exactly) ---
    training_args = SFTConfig(
        output_dir="/models/cot-gemma-2-9b",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # effective batch = 32
        learning_rate=2e-5,
        optim="paged_adamw_8bit",
        max_seq_length=1600,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        data_collator=collator,
    )

    # --- Train (resume from checkpoint if available) ---
    checkpoint_dir = "/models/cot-gemma-2-9b"
    resume_from = None
    if os.path.isdir(checkpoint_dir):
        checkpoints = sorted(
            [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if checkpoints:
            resume_from = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"Resuming from {resume_from}")

    print("Starting training...")
    print(f"Steps: ~{len(dataset) // (4 * 8)} "
          f"({len(dataset)} examples / effective batch 32)")
    start = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed / 60:.1f} minutes.")

    # --- Save adapter + tokenizer ---
    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}")

    model_vol.commit()
    return f"Training complete. {elapsed / 60:.1f} min."


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@app.cls(
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100-80GB",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/models": model_vol,
        "/results": results_vol,
    },
    timeout=86400,  # 24 hours
    scaledown_window=300,
    env={
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)
class CoTEvaluator:
    @modal.method()
    def run_evaluation(self, greedy: bool = True, batch_size: int = 4,
                       max_length: int = 7680,
                       adapter: str = "cot",
                       prompt_style: str = "cot") -> str:
        import torch
        from datasets import load_dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # --- Select adapter ---
        adapter_path = COT_ADAPTER_DIR if adapter == "cot" else NEO4J_ADAPTER
        print(f"Adapter: {adapter} ({adapter_path})")
        print(f"Prompt: {prompt_style}")

        # --- Select prompt template ---
        instruction = COT_INSTRUCTION if prompt_style == "cot" else BASELINE_INSTRUCTION

        # --- Load model + adapter ---
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        tokenizer.padding_side = "left"  # Left-pad for batched generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        if adapter == "neo4j":
            # Neo4j adapter auto-loads from HuggingFace
            model = AutoModelForCausalLM.from_pretrained(
                adapter_path,
                quantization_config=bnb_config,
                attn_implementation="eager",
                device_map="auto",
            )
        else:
            # CoT adapter: load base + PEFT adapter from volume
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                attn_implementation="eager",
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        print(f"Model loaded: {adapter} adapter (4-bit NF4).")

        # Generation params — more tokens for CoT prompt (reasoning + cypher)
        max_new = 1024 if prompt_style == "cot" else 512
        if greedy:
            gen_params = {
                "do_sample": False,
                "max_new_tokens": max_new,
                "pad_token_id": tokenizer.eos_token_id,
            }
            print(f"Decoding: greedy, max_new_tokens={max_new}.")
        else:
            gen_params = {
                "do_sample": True,
                "temperature": 0.2,
                "top_p": 0.9,
                "max_new_tokens": max_new,
                "pad_token_id": tokenizer.eos_token_id,
            }
            print(f"Decoding: sampling (T=0.2, top_p=0.9), max_new_tokens={max_new}.")

        # --- Load test data ---
        ds = load_dataset("neo4j/text2cypher-2024v1", split="test")
        total = len(ds)
        print(f"Test set: {total} examples.")

        # --- Checkpointing ---
        decoding = "greedy" if greedy else "sampling"
        pred_filename = f"predictions_{adapter}_{prompt_style}_4bit_{decoding}.jsonl"
        output_path = f"/results/{pred_filename}"

        completed_ids = set()
        if os.path.exists(output_path):
            with open(output_path) as f:
                for line in f:
                    completed_ids.add(json.loads(line)["instance_id"])
            print(f"Resuming: {len(completed_ids)} already done.")

        pending = []
        for i, ex in enumerate(ds):
            iid = ex.get("instance_id", str(i))
            if iid not in completed_ids:
                pending.append((iid, ex))

        if not pending:
            print("All examples already completed.")
            return pred_filename

        print(f"{len(pending)} examples to process in batches of {batch_size}.")
        start = time.time()
        processed = 0

        with open(output_path, "a") as out_f:
            for batch_start in range(0, len(pending), batch_size):
                batch = pending[batch_start : batch_start + batch_size]

                # Prepare prompts
                prompts = []
                for _, ex in batch:
                    content = instruction.format(
                        schema=ex["schema"], question=ex["question"]
                    )
                    chat = [{"role": "user", "content": content}]
                    prompt = tokenizer.apply_chat_template(
                        chat, add_generation_prompt=True, tokenize=False
                    )
                    prompts.append(prompt)

                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(model.device)

                with torch.no_grad():
                    tokens = model.generate(**inputs, **gen_params)
                    new_tokens = tokens[:, inputs.input_ids.shape[1] :]
                    raw_outputs = tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )

                for (instance_id, ex), raw_output in zip(batch, raw_outputs):
                    predicted_cypher = _parse_cypher(raw_output)
                    record = {
                        "instance_id": instance_id,
                        "question": ex["question"],
                        "schema": ex["schema"],
                        "predicted_cypher": predicted_cypher,
                        "reference_cypher": ex["cypher"],
                        "raw_output": raw_output,
                        "data_source": ex.get("data_source", "unknown"),
                    }
                    out_f.write(json.dumps(record) + "\n")

                processed += len(batch)
                total_done = len(completed_ids) + processed

                if processed % 100 < batch_size:
                    out_f.flush()
                    results_vol.commit()

                if processed % 50 < batch_size:
                    elapsed = time.time() - start
                    rate = processed / elapsed
                    eta = (len(pending) - processed) / rate / 60 if rate else 0
                    print(f"[{total_done}/{total}] {rate:.2f}/sec, ETA: {eta:.1f}m")

        results_vol.commit()
        elapsed = time.time() - start
        print(f"Eval done. {processed} examples in {elapsed / 60:.1f} min.")
        return pred_filename


# ---------------------------------------------------------------------------
# Metrics (runs locally)
# ---------------------------------------------------------------------------

def compute_metrics(predictions_path: str) -> dict:
    """Compute GLEU and exact match from a predictions JSONL file."""
    import evaluate

    records = []
    with open(predictions_path) as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print("No predictions found.")
        return {}

    predictions = [r["predicted_cypher"] for r in records]
    references = [r["reference_cypher"] for r in records]

    # Corpus GLEU
    gleu = evaluate.load("google_bleu")
    gleu_result = gleu.compute(
        predictions=predictions, references=[[ref] for ref in references]
    )
    gleu_score = gleu_result["google_bleu"]

    # Whitespace-normalized exact match
    def normalize(s: str) -> str:
        return " ".join(s.split())

    exact_matches = sum(
        1 for p, r in zip(predictions, references) if normalize(p) == normalize(r)
    )
    exact_match_ratio = exact_matches / len(records)

    # Per-source breakdown
    source_stats: dict = {}
    for r in records:
        src = r.get("data_source", "unknown")
        if src not in source_stats:
            source_stats[src] = {
                "total": 0,
                "exact_matches": 0,
                "predictions": [],
                "references": [],
            }
        source_stats[src]["total"] += 1
        source_stats[src]["predictions"].append(r["predicted_cypher"])
        source_stats[src]["references"].append(r["reference_cypher"])
        if normalize(r["predicted_cypher"]) == normalize(r["reference_cypher"]):
            source_stats[src]["exact_matches"] += 1

    per_source = {}
    for src, stats in sorted(source_stats.items()):
        src_gleu = gleu.compute(
            predictions=stats["predictions"],
            references=[[ref] for ref in stats["references"]],
        )
        per_source[src] = {
            "count": stats["total"],
            "gleu": src_gleu["google_bleu"],
            "exact_match": stats["exact_matches"] / stats["total"],
        }

    return {
        "total_examples": len(records),
        "gleu_score": gleu_score,
        "exact_match_ratio": exact_match_ratio,
        "exact_match_count": exact_matches,
        "per_source": per_source,
    }


def print_summary(metrics: dict):
    """Print a formatted summary of evaluation metrics."""
    print("\n" + "=" * 60)
    print("COT TEXT2CYPHER EVALUATION")
    print("=" * 60)
    print(f"Total examples:     {metrics['total_examples']}")
    print(f"GLEU score:         {metrics['gleu_score']:.4f}")
    print(f"Exact match ratio:  {metrics['exact_match_ratio']:.4f} "
          f"({metrics['exact_match_count']}/{metrics['total_examples']})")
    print(f"\nBaseline comparison: GLEU 0.6455, ExMatch 0.1924")
    delta_gleu = metrics["gleu_score"] - 0.6455
    delta_em = metrics["exact_match_ratio"] - 0.1924
    print(f"Delta:              GLEU {delta_gleu:+.4f}, ExMatch {delta_em:+.4f}")

    if metrics.get("per_source"):
        print(f"\n{'Source':<40} {'Count':>6} {'GLEU':>8} {'ExMatch':>8}")
        print("-" * 64)
        for src, stats in sorted(metrics["per_source"].items()):
            print(
                f"{src:<40} {stats['count']:>6} "
                f"{stats['gleu']:>8.4f} {stats['exact_match']:>8.4f}"
            )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(eval: bool = False, greedy: bool = True, ablation: str = ""):
    """
    Ablation options (use with --eval):
      --ablation neo4j-cot     Neo4j baseline adapter + CoT prompt
      --ablation cot-baseline  CoT adapter + baseline prompt
    """
    if not eval:
        # --- Training ---
        print("Starting CoT fine-tuning on Modal (A100-80GB)...")
        result = train.remote()
        print(result)

        # Download adapter weights
        os.makedirs(LOCAL_ADAPTER_DIR, exist_ok=True)
        print(f"\nDownloading adapter weights to {LOCAL_ADAPTER_DIR}/...")
        for filename in [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]:
            remote_path = f"cot-gemma-2-9b/final/{filename}"
            local_path = os.path.join(LOCAL_ADAPTER_DIR, filename)
            try:
                data = model_vol.read_file(remote_path)
                with open(local_path, "wb") as f:
                    f.write(data)
                size_mb = len(data) / (1024 * 1024)
                print(f"  {filename} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  {filename}: skipped ({e})")
        print("Adapter weights downloaded.")

    else:
        # --- Evaluation ---
        # Determine adapter and prompt style
        if ablation == "neo4j-cot":
            adapter, prompt_style = "neo4j", "cot"
        elif ablation == "cot-baseline":
            adapter, prompt_style = "cot", "baseline"
        else:
            adapter, prompt_style = "cot", "cot"

        decoding = "greedy" if greedy else "sampling"
        pred_filename = f"predictions_{adapter}_{prompt_style}_4bit_{decoding}.jsonl"
        metrics_filename = pred_filename.replace("predictions_", "metrics_").replace(
            ".jsonl", ".json"
        )
        local_predictions = os.path.join(LOCAL_RESULTS_DIR, pred_filename)
        local_metrics = os.path.join(LOCAL_RESULTS_DIR, metrics_filename)

        if os.path.exists(local_predictions) and os.path.getsize(local_predictions) > 0:
            print(f"Found existing predictions at {local_predictions}, skipping inference.")
        else:
            print(f"Starting evaluation on Modal (adapter={adapter}, "
                  f"prompt={prompt_style}, greedy={greedy})...")
            evaluator = CoTEvaluator()
            result = evaluator.run_evaluation.remote(
                greedy=greedy, adapter=adapter, prompt_style=prompt_style
            )
            print(f"Remote returned: {result}")

            # Download predictions
            os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
            print("Downloading predictions from Modal volume...")
            data = results_vol.read_file(pred_filename)
            with open(local_predictions, "wb") as f:
                f.write(data)
            print(f"Saved predictions to {local_predictions}")

        # Compute metrics locally
        print("\nComputing metrics...")
        metrics = compute_metrics(local_predictions)
        if not metrics:
            print("No predictions to evaluate.")
            return

        print_summary(metrics)

        os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
        with open(local_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {local_metrics}")
