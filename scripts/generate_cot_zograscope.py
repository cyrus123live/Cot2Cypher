"""Generate CoT reasoning traces for ZOGRASCOPE training examples.

Reuses the same provider-agnostic config from generate_cot.config (Cerebras + GPT-oss-120B by default).

Usage (from project root):
    COT_PROVIDER=cerebras COT_API_KEY=... ./venv/bin/python scripts/generate_cot_zograscope.py --pilot
    COT_PROVIDER=cerebras COT_API_KEY=... ./venv/bin/python scripts/generate_cot_zograscope.py
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time

# Allow imports from generate_cot package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI

from generate_cot.config import (
    MAX_CONCURRENCY,
    MAX_RETRIES,
    MAX_TOKENS,
    RETRY_BASE_DELAY,
    TEMPERATURE,
    get_api_key,
    get_base_url,
    get_model,
    get_provider,
)
from generate_cot.exemplars_zograscope import POLE_SCHEMA_TEXT
from generate_cot.parse import parse_response, validate_result
from generate_cot.prompts_zograscope import build_messages_zograscope


ZOG_TRAIN_CSV = "data/zograscope/zograscope_train_v1.csv"
ZOG_LENGTH_TRAIN_CSV = "data/zograscope/zograscope_length_train_v1.csv"
OUTPUT_PATH = "data/zograscope/zograscope_cot_traces.jsonl"


def load_zograscope_training(include_length: bool = True) -> list[dict]:
    """Load ZOGRASCOPE training data (main + length splits)."""
    examples = []
    seen_ids = set()

    for csv_path in [ZOG_TRAIN_CSV] + ([ZOG_LENGTH_TRAIN_CSV] if include_length else []):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["id"] in seen_ids:
                    continue
                seen_ids.add(row["id"])
                examples.append({
                    "instance_id": row["id"],
                    "question": row["nl_gold_linked"],
                    "cypher": row["mr"],
                    "num_nodes": int(row["num_nodes"]),
                    "template_id": row["template_id"],
                    "query_type": row["type"],
                    "data_source": "zograscope_train",
                })
    return examples


async def generate_one(
    client: AsyncOpenAI,
    model: str,
    example: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate CoT reasoning for a single ZOGRASCOPE example."""
    messages = build_messages_zograscope(
        question=example["question"],
        cypher=example["cypher"],
    )

    async with semaphore:
        last_error = None
        response = None
        max_attempts = 8  # More attempts for rate-limited backend
        for attempt in range(max_attempts):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                break
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # Aggressive backoff on rate limits
                is_rate_limit = "429" in err_str or "queue" in err_str or "rate" in err_str
                if attempt < max_attempts - 1:
                    if is_rate_limit:
                        delay = min(60, 5 * (1.5 ** attempt)) + random.uniform(0, 2)
                    else:
                        delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                else:
                    return _error_record(
                        example, f"API error after {max_attempts} retries: {last_error}"
                    )

    choice = response.choices[0]
    raw_text = choice.message.content or ""

    result = validate_result(parse_response(raw_text))

    usage = response.usage
    metadata = {
        "model": model,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }

    base_record = {
        "instance_id": example["instance_id"],
        "question": example["question"],
        "schema": POLE_SCHEMA_TEXT,
        "cypher": example["cypher"],
        "num_nodes": example["num_nodes"],
        "template_id": example["template_id"],
        "query_type": example["query_type"],
        "data_source": example["data_source"],
        "generation_metadata": metadata,
    }

    if not result.success:
        return {
            **base_record,
            "reasoning": "",
            "generated_cypher": "",
            "parse_error": result.error,
            "raw_response": raw_text[:2000],
        }

    return {
        **base_record,
        "reasoning": result.reasoning,
        "generated_cypher": result.cypher,
    }


def _error_record(example: dict, error: str) -> dict:
    return {
        "instance_id": example["instance_id"],
        "question": example["question"],
        "schema": POLE_SCHEMA_TEXT,
        "cypher": example["cypher"],
        "num_nodes": example["num_nodes"],
        "template_id": example["template_id"],
        "query_type": example["query_type"],
        "data_source": example["data_source"],
        "reasoning": "",
        "generated_cypher": "",
        "parse_error": error,
        "raw_response": "",
        "generation_metadata": {},
    }


def load_completed_ids(output_path: str) -> set[str]:
    completed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add(rec["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


async def main():
    parser = argparse.ArgumentParser(description="Generate ZOGRASCOPE CoT training data")
    parser.add_argument("--pilot", action="store_true", help="Run on 20 examples only")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--no-length", action="store_true", help="Skip length-generalization training set")
    args = parser.parse_args()

    examples = load_zograscope_training(include_length=not args.no_length)
    total = len(examples)

    provider = get_provider()
    base_url = get_base_url()
    model = get_model()
    api_key = get_api_key()

    print(f"Provider: {provider} ({base_url})")
    print(f"Model: {model}")
    print(f"Dataset: {total} ZOGRASCOPE training examples (main + length)")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {args.output}")

    completed_ids = load_completed_ids(args.output)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} already completed")

    pending = [ex for ex in examples if ex["instance_id"] not in completed_ids]

    if args.pilot:
        random.seed(42)
        # Stratify by num_nodes to test full range
        by_nn = {}
        for ex in pending:
            by_nn.setdefault(ex["num_nodes"], []).append(ex)
        pilot = []
        for nn in sorted(by_nn.keys()):
            random.shuffle(by_nn[nn])
            pilot.extend(by_nn[nn][:5])
        pending = pilot[:20]
        print(f"Pilot mode: {len(pending)} examples (stratified by num_nodes)")
    elif args.limit > 0:
        pending = pending[:args.limit]
        print(f"Batch mode: {len(pending)} examples (limit={args.limit})")
    else:
        print(f"Pending: {len(pending)} examples")

    if not pending:
        print("Nothing to do.")
        return

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    batch_size = 10  # Smaller batches: more frequent flushes/checkpoints under rate limits
    start_time = time.time()
    processed = 0
    successes = 0
    failures = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    out_f = open(args.output, "a")

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        tasks = [generate_one(client, model, example, semaphore) for example in batch]
        results = await asyncio.gather(*tasks)

        for record in results:
            out_f.write(json.dumps(record) + "\n")
            if record.get("reasoning"):
                successes += 1
            else:
                failures += 1
            meta = record.get("generation_metadata", {})
            total_prompt_tokens += meta.get("prompt_tokens", 0)
            total_completion_tokens += meta.get("completion_tokens", 0)
        out_f.flush()

        processed += len(batch)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = len(pending) - processed
        eta_min = (remaining / rate / 60) if rate > 0 else 0

        print(
            f"[{len(completed_ids) + processed}/{total}] "
            f"{rate:.2f}/s | ok={successes} fail={failures} | "
            f"tokens={total_prompt_tokens + total_completion_tokens:,} | "
            f"ETA: {eta_min:.1f}m",
            flush=True,
        )

    out_f.close()

    elapsed_total = time.time() - start_time
    print(f"\nDone. {processed} examples in {elapsed_total / 60:.1f} min.")
    print(f"Successes: {successes}, Failures: {failures}")
    print(f"Tokens: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion")

    rates = {
        "cerebras": (0.25, 0.69),
        "galaxy": (0.02, 0.10),
        "openrouter": (0.039, 0.19),
        "deepinfra": (0.04, 0.19),
        "openai": (0.15, 0.60),
    }
    if provider in rates:
        in_rate, out_rate = rates[provider]
        cost = (total_prompt_tokens / 1e6 * in_rate) + (total_completion_tokens / 1e6 * out_rate)
        print(f"Estimated cost ({provider}): ${cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
