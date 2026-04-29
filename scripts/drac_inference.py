#!/usr/bin/env python3
"""
Standalone inference script for DRAC / Compute Canada clusters.
No Modal dependency. Runs evaluation and self-consistency on a single GPU.

Usage (called by drac_eval.sh, or directly):
    python drac_inference.py --adapter-path /path/to/adapter --output-dir /path/to/results
    python drac_inference.py --adapter-path /path/to/adapter --output-dir /path/to/results --self-consistency 5
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL = "google/gemma-2-9b-it"

COT_INSTRUCTION = (
    "Generate Cypher statement to query a graph database.\n"
    "Use only the provided relationship types and properties in the schema.\n"
    "Schema: {schema}\n"
    "Question: {question}\n\n"
    "Think step by step, then provide the Cypher query."
)


def parse_cypher(raw_output: str) -> str:
    """Extract Cypher query from model output containing reasoning + Cypher."""
    marker = "Cypher output:"
    idx = raw_output.rfind(marker)
    if idx != -1:
        cypher = raw_output[idx + len(marker) :]
    else:
        cypher = raw_output

    cypher = cypher.strip("`\n ")
    if cypher.startswith("cypher\n"):
        cypher = cypher[7:]
    cypher = cypher.strip("`\n ")
    for sep in ["**Explanation:**", "\n\nExplanation:", "\n\nNote:"]:
        cypher, _, _ = cypher.partition(sep)
    return cypher.strip()


def load_model(adapter_path: str, hf_cache: str = None):
    """Load base model with QLoRA quantization and CoT adapter."""
    kwargs = {}
    if hf_cache:
        kwargs["cache_dir"] = hf_cache

    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, **kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading base model {BASE_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        attn_implementation="eager",
        device_map="auto",
        **kwargs,
    )

    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def run_inference(model, tokenizer, examples, batch_size=4, max_length=7680,
                  max_new_tokens=1024, temperature=0.0, do_sample=False):
    """Run inference on a list of examples. Returns list of (raw_output, predicted_cypher)."""
    gen_params = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_params["do_sample"] = True
        gen_params["temperature"] = temperature
        gen_params["top_p"] = 0.95
    else:
        gen_params["do_sample"] = False

    results = []
    total = len(examples)

    for batch_start in range(0, total, batch_size):
        batch = examples[batch_start : batch_start + batch_size]

        prompts = []
        for ex in batch:
            content = COT_INSTRUCTION.format(
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

        for raw_output in raw_outputs:
            predicted_cypher = parse_cypher(raw_output)
            results.append((raw_output, predicted_cypher))

        done = min(batch_start + batch_size, total)
        if done % 50 < batch_size or done == total:
            print(f"  [{done}/{total}]")

    return results


def self_consistency_vote(candidates: list[str]) -> str:
    """Majority vote over candidate Cypher queries (whitespace-normalized)."""
    normalized = [" ".join(c.split()) for c in candidates]
    counter = Counter(normalized)
    winner = counter.most_common(1)[0][0]
    return winner


def main():
    parser = argparse.ArgumentParser(description="CoT Text2Cypher Inference (DRAC)")
    parser.add_argument("--adapter-path", required=True, help="Path to CoT LoRA adapter")
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    parser.add_argument("--hf-cache", default=None, help="HuggingFace cache directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=7680)
    parser.add_argument("--self-consistency", type=int, default=0,
                        help="Number of samples for self-consistency (0 = greedy only)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for self-consistency sampling")
    parser.add_argument("--zograscope", type=str, default=None,
                        help="Path to ZOGRASCOPE formatted JSONL (instead of Neo4j dataset)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.adapter_path, args.hf_cache)

    # Load test data
    if args.zograscope:
        print(f"Loading ZOGRASCOPE data from {args.zograscope}...")
        with open(args.zograscope) as f:
            examples = [json.loads(line) for line in f]
        # ZOGRASCOPE uses 'reference_cypher' not 'cypher'
        for ex in examples:
            ex["cypher"] = ex.get("reference_cypher", "")
            ex["instance_id"] = ex.get("id", "")
            ex["data_source"] = f"zograscope_{ex.get('split', 'unknown')}"
    else:
        print("Loading test dataset...")
        if args.hf_cache:
            os.environ["HF_DATASETS_CACHE"] = args.hf_cache
        ds = load_dataset("neo4j/text2cypher-2024v1", split="test")
        examples = list(ds)
    print(f"Test set: {len(examples)} examples.")

    # ================================================================
    # Greedy evaluation (always run, with checkpointing)
    # ================================================================
    prefix = "predictions_zograscope" if args.zograscope else "predictions_cot"
    greedy_path = os.path.join(args.output_dir, f"{prefix}_greedy.jsonl")

    # Resume from checkpoint
    completed_greedy = 0
    if os.path.exists(greedy_path):
        with open(greedy_path) as f:
            completed_greedy = sum(1 for _ in f)
        print(f"Greedy: resuming from {completed_greedy}/{len(examples)}")

    if completed_greedy >= len(examples):
        print(f"Greedy predictions complete ({completed_greedy}), skipping.")
    else:
        print(f"\n=== Greedy Evaluation ({len(examples) - completed_greedy} remaining) ===")
        start = time.time()
        remaining = examples[completed_greedy:]

        with open(greedy_path, "a") as f:
            for batch_start in range(0, len(remaining), args.batch_size):
                batch = remaining[batch_start : batch_start + args.batch_size]
                results = run_inference(
                    model, tokenizer, batch,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    do_sample=False,
                )
                for ex, (raw_output, predicted_cypher) in zip(batch, results):
                    record = {
                        "instance_id": ex.get("instance_id", ""),
                        "question": ex["question"],
                        "schema": ex["schema"],
                        "predicted_cypher": predicted_cypher,
                        "reference_cypher": ex["cypher"],
                        "raw_output": raw_output,
                        "data_source": ex.get("data_source", "unknown"),
                    }
                    f.write(json.dumps(record) + "\n")

                done = completed_greedy + batch_start + len(batch)
                if done % 50 < args.batch_size or done >= len(examples):
                    f.flush()
                    elapsed = time.time() - start
                    rate = (batch_start + len(batch)) / elapsed
                    eta = (len(remaining) - batch_start - len(batch)) / rate / 60 if rate > 0 else 0
                    print(f"  [{done}/{len(examples)}] {rate:.2f}/sec, ETA: {eta:.1f}m")

        elapsed = time.time() - start
        print(f"Greedy done in {elapsed / 60:.1f} min.")

    # ================================================================
    # Self-consistency evaluation (if requested, with checkpointing)
    # ================================================================
    if args.self_consistency > 0:
        n_samples = args.self_consistency
        print(f"\n=== Self-Consistency (n={n_samples}, T={args.temperature}) ===")
        start = time.time()

        # Run each sample as a separate file for resumability
        sample_files = []
        for sample_idx in range(n_samples):
            sample_path = os.path.join(
                args.output_dir, f"{prefix}_sc_sample_{sample_idx}.jsonl"
            )
            sample_files.append(sample_path)

            # Check if this sample is already complete
            completed = 0
            if os.path.exists(sample_path):
                with open(sample_path) as f:
                    completed = sum(1 for _ in f)

            if completed >= len(examples):
                print(f"Sample {sample_idx + 1}/{n_samples}: complete ({completed}), skipping.")
                continue

            print(f"\nSample {sample_idx + 1}/{n_samples}: {len(examples) - completed} remaining")
            remaining = examples[completed:]

            with open(sample_path, "a") as f:
                for batch_start in range(0, len(remaining), args.batch_size):
                    batch = remaining[batch_start : batch_start + args.batch_size]
                    results = run_inference(
                        model, tokenizer, batch,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        do_sample=True,
                        temperature=args.temperature,
                    )
                    for (raw_output, predicted_cypher) in results:
                        f.write(json.dumps({"cypher": predicted_cypher}) + "\n")

                    done = completed + batch_start + len(batch)
                    if done % 50 < args.batch_size or done >= len(examples):
                        f.flush()
                        elapsed_sample = time.time() - start
                        print(f"  [{done}/{len(examples)}]")

        # Majority vote across all samples
        sc_path = os.path.join(
            args.output_dir,
            f"{prefix}_sc{n_samples}_t{args.temperature}.jsonl",
        )

        print(f"\nComputing majority vote...")
        all_samples = []
        for sf in sample_files:
            with open(sf) as f:
                all_samples.append([json.loads(line)["cypher"] for line in f])

        with open(sc_path, "w") as f:
            for i, ex in enumerate(examples):
                candidates = [all_samples[s][i] for s in range(n_samples)]
                voted_cypher = self_consistency_vote(candidates)
                record = {
                    "instance_id": ex.get("instance_id", ""),
                    "question": ex["question"],
                    "schema": ex["schema"],
                    "predicted_cypher": voted_cypher,
                    "reference_cypher": ex["cypher"],
                    "candidates": candidates,
                    "data_source": ex.get("data_source", "unknown"),
                }
                f.write(json.dumps(record) + "\n")

        elapsed = time.time() - start
        print(f"Self-consistency done in {elapsed / 60:.1f} min. Saved to {sc_path}")

    # ================================================================
    # Compute metrics
    # ================================================================
    print("\n=== Computing Metrics ===")
    import evaluate

    for pred_file in sorted(os.listdir(args.output_dir)):
        if not pred_file.startswith("predictions_") or not pred_file.endswith(".jsonl"):
            continue

        pred_path = os.path.join(args.output_dir, pred_file)
        with open(pred_path) as f:
            records = [json.loads(line) for line in f]

        predictions = [r["predicted_cypher"] for r in records]
        references = [r["reference_cypher"] for r in records]

        gleu = evaluate.load("google_bleu")
        gleu_score = gleu.compute(
            predictions=predictions, references=[[ref] for ref in references]
        )["google_bleu"]

        normalize = lambda s: " ".join(s.split())
        em = sum(
            1 for p, r in zip(predictions, references)
            if normalize(p) == normalize(r)
        )

        print(f"\n{pred_file}:")
        print(f"  GLEU:      {gleu_score:.4f}")
        print(f"  String EM: {em / len(records):.4f} ({em}/{len(records)})")

        # Save metrics
        metrics_file = pred_file.replace("predictions_", "metrics_").replace(
            ".jsonl", ".json"
        )
        metrics_path = os.path.join(args.output_dir, metrics_file)
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "total_examples": len(records),
                    "gleu_score": gleu_score,
                    "exact_match_ratio": em / len(records),
                    "exact_match_count": em,
                },
                f,
                indent=2,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
