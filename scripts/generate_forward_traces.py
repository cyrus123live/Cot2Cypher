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

from generate_cot.parse import parse_response, validate_result
from generate_cot.prompts_forward import build_messages_forward


def load_db_accessible_training(source: str, limit: int = 0) -> list[dict]:
    """Load training examples that have database access (for execution filtering).

    Reads from the local cot_training_data.jsonl (which carries
    database_reference_alias) so this runs anywhere with no HF download and no
    dataset-field-name guessing.
    """
    examples = []
    with open(source) as f:
        for line in f:
            row = json.loads(line)
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


async def _one_completion(client, model, messages, temperature, max_tokens, max_attempts=8):
    """One chat completion with rate-limit-aware backoff."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content or ""
            result = validate_result(parse_response(raw))
            if result.success:
                return {"reasoning": result.reasoning, "cypher": result.cypher}
            return {"reasoning": "", "cypher": "", "parse_error": result.error}
        except Exception as e:
            last_error = e
            err = str(e).lower()
            is_rl = "429" in err or "rate" in err or "quota" in err or "too many" in err
            if attempt < max_attempts - 1:
                # Aggressive backoff on rate limits, gentler on transient errors
                delay = (min(60, 5 * (1.5 ** attempt)) if is_rl
                         else 2.0 * (2 ** attempt)) + random.uniform(0, 2)
                await asyncio.sleep(delay)
    return {"reasoning": "", "cypher": "", "error": str(last_error)[:200]}


async def generate_candidates(client, model, example, k, temperature, max_tokens, sem):
    """Generate k forward candidates for one example."""
    messages = build_messages_forward(example["schema"], example["question"])
    candidates = []
    async with sem:
        for _ in range(k):
            candidates.append(
                await _one_completion(client, model, messages, temperature, max_tokens)
            )
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
    ap.add_argument("--source", default="data/cot_training_data.jsonl",
                    help="JSONL with question/schema/cypher/database_reference_alias")
    ap.add_argument("--output", default="data/forward_traces.jsonl")
    args = ap.parse_args()

    base_url = os.environ.get("COT_API_BASE", "https://api.groq.com/openai/v1")
    api_key = os.environ.get("COT_API_KEY", "")
    if not api_key:
        raise SystemExit("Set COT_API_KEY. Set COT_API_BASE to override the endpoint "
                         "(default Groq: https://api.groq.com/openai/v1).")

    examples = load_db_accessible_training(args.source, args.limit)
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
