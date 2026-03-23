"""Main async script for generating CoT training data.

Usage:
    # Pilot run (50 examples)
    COT_API_KEY=... python -m generate_cot.generate --pilot

    # Full run
    COT_API_KEY=... python -m generate_cot.generate

    # Custom provider/concurrency
    COT_PROVIDER=openrouter COT_API_KEY=... python -m generate_cot.generate --concurrency 30

    # Resume after interruption (automatic — reads checkpoint)
    COT_API_KEY=... python -m generate_cot.generate
"""

import argparse
import asyncio
import json
import os
import random
import time

from openai import AsyncOpenAI

from generate_cot.config import (
    MAX_CONCURRENCY,
    MAX_RETRIES,
    MAX_TOKENS,
    OUTPUT_PATH,
    RETRY_BASE_DELAY,
    TEMPERATURE,
    get_api_key,
    get_base_url,
    get_model,
    get_provider,
)
from generate_cot.parse import parse_response, validate_result
from generate_cot.prompts import build_messages


async def generate_one(
    client: AsyncOpenAI,
    model: str,
    example: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate CoT reasoning for a single example."""
    messages = build_messages(
        schema=example["schema"],
        question=example["question"],
        cypher=example["cypher"],
    )

    async with semaphore:
        last_error = None
        for attempt in range(MAX_RETRIES):
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
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                else:
                    return _error_record(example, f"API error after {MAX_RETRIES} retries: {last_error}")

    choice = response.choices[0]
    raw_text = choice.message.content or ""

    result = validate_result(parse_response(raw_text))

    usage = response.usage
    metadata = {
        "model": model,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }

    if not result.success:
        return {
            "instance_id": example["instance_id"],
            "question": example["question"],
            "schema": example["schema"],
            "cypher": example["cypher"],
            "reasoning": "",
            "generated_cypher": "",
            "parse_error": result.error,
            "raw_response": raw_text[:2000],
            "data_source": example.get("data_source", ""),
            "database_reference_alias": example.get("database_reference_alias", ""),
            "generation_metadata": metadata,
        }

    return {
        "instance_id": example["instance_id"],
        "question": example["question"],
        "schema": example["schema"],
        "cypher": example["cypher"],
        "reasoning": result.reasoning,
        "generated_cypher": result.cypher,
        "data_source": example.get("data_source", ""),
        "database_reference_alias": example.get("database_reference_alias", ""),
        "generation_metadata": metadata,
    }


def _error_record(example: dict, error: str) -> dict:
    return {
        "instance_id": example["instance_id"],
        "question": example["question"],
        "schema": example["schema"],
        "cypher": example["cypher"],
        "reasoning": "",
        "generated_cypher": "",
        "parse_error": error,
        "raw_response": "",
        "data_source": example.get("data_source", ""),
        "database_reference_alias": example.get("database_reference_alias", ""),
        "generation_metadata": {},
    }


def load_completed_ids(output_path: str) -> set[str]:
    """Load instance IDs already in the checkpoint file."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add(rec["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


async def main():
    parser = argparse.ArgumentParser(description="Generate CoT training data")
    parser.add_argument("--pilot", action="store_true", help="Run on 50 diverse examples only")
    parser.add_argument("--limit", type=int, default=0, help="Max examples to process this run (0=all)")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    # Load dataset
    from datasets import load_dataset
    ds = load_dataset("neo4j/text2cypher-2024v1", split="train")
    total = len(ds)

    # Config
    provider = get_provider()
    base_url = get_base_url()
    model = get_model()
    api_key = get_api_key()

    print(f"Provider: {provider} ({base_url})")
    print(f"Model: {model}")
    print(f"Dataset: {total} training examples")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {args.output}")

    # Load checkpoint
    completed_ids = load_completed_ids(args.output)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} already completed")

    # Build pending list
    all_examples = list(ds)
    pending = [ex for ex in all_examples if ex["instance_id"] not in completed_ids]

    if args.pilot:
        # Select 50 diverse examples across sources
        pending = _select_pilot(pending, n=50)
        print(f"Pilot mode: {len(pending)} examples selected")
    elif args.limit > 0:
        pending = pending[:args.limit]
        print(f"Batch mode: {len(pending)} examples (limit={args.limit})")
    else:
        print(f"Pending: {len(pending)} examples")

    if not pending:
        print("Nothing to do.")
        return

    # Init client
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Process in batches for progress reporting and checkpointing
    batch_size = 100
    start_time = time.time()
    processed = 0
    successes = 0
    failures = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]

        tasks = [
            generate_one(client, model, example, semaphore)
            for example in batch
        ]
        results = await asyncio.gather(*tasks)

        # Write results
        with open(args.output, "a") as f:
            for record in results:
                f.write(json.dumps(record) + "\n")
                if record.get("reasoning"):
                    successes += 1
                else:
                    failures += 1
                meta = record.get("generation_metadata", {})
                total_prompt_tokens += meta.get("prompt_tokens", 0)
                total_completion_tokens += meta.get("completion_tokens", 0)

        processed += len(batch)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = len(pending) - processed
        eta_min = (remaining / rate / 60) if rate > 0 else 0

        print(
            f"[{len(completed_ids) + processed}/{total}] "
            f"{rate:.1f}/s | "
            f"ok={successes} fail={failures} | "
            f"tokens={total_prompt_tokens + total_completion_tokens:,} | "
            f"ETA: {eta_min:.1f}m"
        )

    elapsed_total = time.time() - start_time
    print(f"\nDone. {processed} examples in {elapsed_total / 60:.1f} min.")
    print(f"Successes: {successes}, Failures: {failures}")
    print(f"Tokens: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion")

    # Cost estimate
    provider = get_provider()
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


def _select_pilot(examples: list[dict], n: int = 50) -> list[dict]:
    """Select n diverse examples stratified by data_source."""
    by_source: dict[str, list] = {}
    for ex in examples:
        src = ex.get("data_source", "unknown")
        by_source.setdefault(src, []).append(ex)

    selected = []
    # Round-robin from each source
    sources = sorted(by_source.keys())
    per_source = max(1, n // len(sources))

    for src in sources:
        pool = by_source[src]
        random.seed(42)
        sample = random.sample(pool, min(per_source, len(pool)))
        selected.extend(sample)

    # Fill remaining if needed
    if len(selected) < n:
        remaining = [ex for ex in examples if ex not in selected]
        random.seed(42)
        selected.extend(random.sample(remaining, min(n - len(selected), len(remaining))))

    return selected[:n]


if __name__ == "__main__":
    asyncio.run(main())
