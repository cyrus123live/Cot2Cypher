"""Generate FORWARD CoT candidates (STaR-style) via a local vLLM endpoint.

For each training example WITH database access, generate k candidate
(reasoning, cypher) pairs from schema+question only (no reference Cypher).
These are later execution-filtered (filter_traces_by_execution.py) to keep
only candidates whose Cypher executes to the reference result.

Designed to run on DRAC with a local vLLM server (OpenAI-compatible) serving
the teacher model (default gpt-oss-120b). Point COT_API_BASE at the vLLM
endpoint (e.g. http://localhost:8000/v1).

Usage (on DRAC, after vLLM is up):
    COT_API_BASE=http://localhost:8000/v1 COT_API_KEY=dummy \\
        python scripts/generate_forward_traces.py \\
        --model openai/gpt-oss-120b --k 4 --temperature 0.7 \\
        --output data/forward_traces.jsonl
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI
from datasets import load_dataset

from generate_cot.parse import parse_response, validate_result
from generate_cot.prompts_forward import build_messages_forward


def load_db_accessible_training(db_mapping_path: str, limit: int = 0) -> list[dict]:
    """Load training-split examples that have database access (for exec filtering)."""
    with open(db_mapping_path) as f:
        db_map = json.load(f)  # instance_id -> alias (built from test set; see note)

    ds = load_dataset("neo4j/text2cypher-2024v1", split="train")
    examples = []
    for row in ds:
        # We only keep examples whose database_reference_alias indicates DB access.
        alias = row.get("database_reference_alias") or ""
        if not alias:
            continue
        examples.append({
            "instance_id": row["instance_id"],
            "question": row["question"],
            "schema": row["schema"],
            "cypher": row["cypher"],
            "data_source": row.get("data_source", ""),
            "database_reference_alias": alias,
        })
    if limit:
        examples = examples[:limit]
    return examples


async def generate_candidates(client, model, example, k, temperature, max_tokens, sem):
    """Generate k forward candidates for one example."""
    messages = build_messages_forward(example["schema"], example["question"])
    candidates = []
    async with sem:
        for _ in range(k):
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                raw = resp.choices[0].message.content or ""
                result = validate_result(parse_response(raw))
                if result.success:
                    candidates.append({"reasoning": result.reasoning, "cypher": result.cypher})
                else:
                    candidates.append({"reasoning": "", "cypher": "", "parse_error": result.error})
            except Exception as e:
                candidates.append({"reasoning": "", "cypher": "", "error": str(e)[:200]})
    return {
        "instance_id": example["instance_id"],
        "question": example["question"],
        "schema": example["schema"],
        "cypher": example["cypher"],  # reference, for execution filtering downstream
        "data_source": example["data_source"],
        "database_reference_alias": example["database_reference_alias"],
        "forward_candidates": candidates,
    }


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--k", type=int, default=4, help="candidates per example")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--db-mapping", default="data/test_db_mapping.json")
    ap.add_argument("--output", default="data/forward_traces.jsonl")
    args = ap.parse_args()

    base_url = os.environ.get("COT_API_BASE", "http://localhost:8000/v1")
    api_key = os.environ.get("COT_API_KEY", "dummy")

    examples = load_db_accessible_training(args.db_mapping, args.limit)
    done = load_done(args.output)
    pending = [e for e in examples if e["instance_id"] not in done]
    print(f"Teacher: {args.model} @ {base_url}")
    print(f"DB-accessible training examples: {len(examples)}; already done: {len(done)}; pending: {len(pending)}")
    print(f"k={args.k} candidates each, temperature={args.temperature}")
    if not pending:
        print("Nothing to do.")
        return

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    start = time.time()
    batch_size = 50
    out = open(args.output, "a")
    processed = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        tasks = [generate_candidates(client, args.model, e, args.k,
                                     args.temperature, args.max_tokens, sem) for e in batch]
        for rec in await asyncio.gather(*tasks):
            out.write(json.dumps(rec) + "\n")
        out.flush()
        processed += len(batch)
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed else 0
        eta = (len(pending) - processed) / rate / 60 if rate else 0
        print(f"[{len(done)+processed}/{len(examples)}] {rate:.1f} ex/s, ETA {eta:.0f}m", flush=True)
    out.close()
    print(f"Done. {processed} examples in {(time.time()-start)/60:.1f} min.")


if __name__ == "__main__":
    asyncio.run(main())
