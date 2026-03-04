# Run with:
#   modal run run_neo4j.py              # bf16 inference (default)
#   modal run run_neo4j.py --quantize   # 4-bit NF4 inference (matching Neo4j's eval)

import json
import os
import time

import modal

app = modal.App("llm-thesis-neo4j")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "peft==0.12.0",
        "accelerate==0.33.0",
        "bitsandbytes==0.43.3",
        "safetensors",
        "sentencepiece",
        "hf-transfer",
        "datasets",
    )
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("neo4j-eval-results", create_if_missing=True)

MODEL_NAME = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
BATCH_SIZE = 4

SYSTEM_PROMPT = (
    "Task: Generate Cypher statement to query a graph database.\n"
    "Instructions: Use only the provided relationship types and properties in the schema.\n"
    "Do not use any other relationship types or properties that are not provided in the schema.\n"
    "Do not include any explanations or apologies in your responses.\n"
    "Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.\n"
    "Do not include any text except the generated Cypher statement."
)

USER_PROMPT = (
    "Generate Cypher statement to query a graph database.\n"
    "Use only the provided relationship types and properties in the schema.\n"
    "Schema: {schema}\n"
    "Question: {question}\n"
    "Cypher output:"
)

PREDICTIONS_BF16_FILENAME = "predictions.jsonl"
PREDICTIONS_4BIT_FILENAME = "predictions_4bit.jsonl"
LOCAL_RESULTS_DIR = "results"


def prepare_chat_prompt(question: str, schema: str) -> list[dict]:
    chat = [
        {
            "role": "user",
            "content": SYSTEM_PROMPT + "\n\n" + USER_PROMPT.format(schema=schema, question=question),
        },
    ]
    return chat


def _postprocess_output_cypher(output_cypher: str) -> str:
    partition_by = "**Explanation:**"
    output_cypher, _, _ = output_cypher.partition(partition_by)
    output_cypher = output_cypher.strip("`\n")
    output_cypher = output_cypher.lstrip("cypher\n")
    output_cypher = output_cypher.strip("`\n ")
    return output_cypher


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100-80GB",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
    timeout=21600,  # 6 hours (4-bit inference is ~4x slower than bf16)
    scaledown_window=300,
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
)
class Text2CypherEvaluator:
    def _load_model(self, quantize: bool = False):
        """Load model with specified precision. Called once at start of evaluation."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Left-pad for batched generation with decoder-only models
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if quantize:
            from transformers import BitsAndBytesConfig

            # Match Neo4j's exact quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            print("Model loaded in 4-bit NF4 quantization (matching Neo4j config).")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            print("Model loaded in bf16 precision.")

        self.model.eval()
        self.generate_params = {
            "top_p": 0.9,
            "temperature": 0.2,
            "max_new_tokens": 512,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    @modal.method()
    def run_evaluation(self, quantize: bool = False) -> str:
        import torch
        from datasets import load_dataset

        # Load model with specified precision
        self._load_model(quantize=quantize)

        # Load test split
        ds = load_dataset("neo4j/text2cypher-2024v1", split="test")
        total = len(ds)
        print(f"Loaded {total} test examples.")

        # Use appropriate output file based on quantization mode
        predictions_filename = PREDICTIONS_4BIT_FILENAME if quantize else PREDICTIONS_BF16_FILENAME
        mode_label = "4-bit NF4" if quantize else "bf16"
        print(f"Evaluation mode: {mode_label}, output: {predictions_filename}")

        # Load checkpoint: collect already-completed instance_ids
        output_path = f"/results/{predictions_filename}"
        completed_ids = set()
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    completed_ids.add(rec["instance_id"])
            print(f"Resuming: {len(completed_ids)} examples already completed.")

        # Collect pending examples
        pending = []
        for i, example in enumerate(ds):
            instance_id = example.get("instance_id", str(i))
            if instance_id not in completed_ids:
                pending.append((instance_id, example))

        if not pending:
            print("All examples already completed.")
            return "Done. Nothing to process."

        print(f"{len(pending)} examples to process in batches of {BATCH_SIZE}.")
        start_time = time.time()
        processed = 0

        with open(output_path, "a") as out_f:
            for batch_start in range(0, len(pending), BATCH_SIZE):
                batch = pending[batch_start : batch_start + BATCH_SIZE]

                # Prepare batch prompts
                prompts = []
                for _, example in batch:
                    chat = prepare_chat_prompt(
                        question=example["question"], schema=example["schema"]
                    )
                    prompt = self.tokenizer.apply_chat_template(
                        chat, add_generation_prompt=True, tokenize=False
                    )
                    prompts.append(prompt)

                # Tokenize with left-padding for batch
                inputs = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=7680,
                ).to(self.model.device)

                # Generate batch
                with torch.no_grad():
                    tokens = self.model.generate(**inputs, **self.generate_params)
                    # Slice off input tokens to get only generated output
                    new_tokens = tokens[:, inputs.input_ids.shape[1] :]
                    raw_outputs = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )

                # Write results
                for (instance_id, example), raw_output in zip(batch, raw_outputs):
                    predicted_cypher = _postprocess_output_cypher(raw_output)
                    record = {
                        "instance_id": instance_id,
                        "question": example["question"],
                        "schema": example["schema"],
                        "predicted_cypher": predicted_cypher,
                        "reference_cypher": example["cypher"],
                        "data_source": example.get("data_source", "unknown"),
                    }
                    out_f.write(json.dumps(record) + "\n")

                processed += len(batch)
                total_done = len(completed_ids) + processed

                # Checkpoint every 100 examples
                if processed % 100 < BATCH_SIZE:
                    out_f.flush()
                    results_vol.commit()

                # Progress logging
                if processed % 50 < BATCH_SIZE:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = len(pending) - processed
                    eta_min = (remaining / rate / 60) if rate > 0 else 0
                    print(
                        f"[{total_done}/{total}] "
                        f"{rate:.2f} examples/sec, "
                        f"ETA: {eta_min:.1f} min"
                    )

        # Final commit
        results_vol.commit()
        elapsed_total = time.time() - start_time
        print(
            f"Evaluation complete. {processed} new examples in "
            f"{elapsed_total / 60:.1f} min. Total: {len(completed_ids) + processed}/{total}"
        )
        return f"Done. Processed {processed} new examples in {elapsed_total / 60:.1f} min."


def compute_metrics(predictions_path: str) -> dict:
    """Compute GLEU and exact match from a predictions JSONL file."""
    import evaluate

    records = []
    with open(predictions_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print("No predictions found.")
        return {}

    predictions = [r["predicted_cypher"] for r in records]
    references = [r["reference_cypher"] for r in records]

    # Corpus GLEU
    gleu = evaluate.load("google_bleu")
    # google_bleu expects list of str for predictions, list of list of str for references
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
    source_stats = {}
    for r in records:
        src = r.get("data_source", "unknown")
        if src not in source_stats:
            source_stats[src] = {"total": 0, "exact_matches": 0, "predictions": [], "references": []}
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

    metrics = {
        "total_examples": len(records),
        "gleu_score": gleu_score,
        "exact_match_ratio": exact_match_ratio,
        "exact_match_count": exact_matches,
        "per_source": per_source,
    }
    return metrics


def print_summary(metrics: dict):
    """Print a formatted summary of evaluation metrics."""
    print("\n" + "=" * 60)
    print("NEO4J TEXT2CYPHER BASELINE EVALUATION")
    print("=" * 60)
    print(f"Total examples:     {metrics['total_examples']}")
    print(f"GLEU score:         {metrics['gleu_score']:.4f}")
    print(f"Exact match ratio:  {metrics['exact_match_ratio']:.4f} "
          f"({metrics['exact_match_count']}/{metrics['total_examples']})")
    print("\nNote: Exact match is whitespace-normalized string comparison")
    print("(lower bound — execution-based match not available).")

    if metrics.get("per_source"):
        print(f"\n{'Source':<30} {'Count':>6} {'GLEU':>8} {'ExMatch':>8}")
        print("-" * 54)
        for src, stats in sorted(metrics["per_source"].items()):
            print(
                f"{src:<30} {stats['count']:>6} "
                f"{stats['gleu']:>8.4f} {stats['exact_match']:>8.4f}"
            )
    print("=" * 60)


@app.local_entrypoint()
def main(quantize: bool = False):
    predictions_filename = PREDICTIONS_4BIT_FILENAME if quantize else PREDICTIONS_BF16_FILENAME
    metrics_filename = "metrics_4bit.json" if quantize else "metrics.json"
    mode_label = "4-bit NF4" if quantize else "bf16"

    local_predictions = os.path.join(LOCAL_RESULTS_DIR, predictions_filename)
    local_metrics = os.path.join(LOCAL_RESULTS_DIR, metrics_filename)

    print(f"Mode: {mode_label}")

    # Step 1: Check if predictions already exist locally
    if os.path.exists(local_predictions) and os.path.getsize(local_predictions) > 0:
        print(f"Found existing predictions at {local_predictions}, skipping inference.")
    else:
        # Step 2: Run evaluation on Modal
        print(f"Starting {mode_label} evaluation on Modal...")
        evaluator = Text2CypherEvaluator()
        result = evaluator.run_evaluation.remote(quantize=quantize)
        print(result)

        # Step 3: Download predictions from Modal volume
        os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
        print("Downloading predictions from Modal volume...")
        data = results_vol.read_file(predictions_filename)
        with open(local_predictions, "wb") as f:
            f.write(data)
        print(f"Saved predictions to {local_predictions}")

    # Step 4: Compute metrics locally
    print("\nComputing metrics...")
    metrics = compute_metrics(local_predictions)

    if not metrics:
        print("No predictions to evaluate.")
        return

    # Step 5: Print summary and save metrics
    print_summary(metrics)

    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    with open(local_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {local_metrics}")
