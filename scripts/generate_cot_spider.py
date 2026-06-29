"""Generate CoT reasoning traces for SQL on SPIDER (the canonical benchmark).

The DEFINITIVE positive control. The gretel run used string EM (wrong metric for
SQL) on synthetic data. This run uses Spider (real databases -> execution accuracy,
the metric the CoT-helps-SQL literature actually reports) so the positive control
gets a fair test. Reuses the SQL distillation prompt from generate_cot_sql.

QA pairs: xlangai/spider (parquet). Schemas: richardr1126/spider-schema.
Databases (for execution eval, separate step): the Spider `database/` zip.

Usage (from project root):
    ./venv/bin/python scripts/generate_cot_spider.py --prepare-test
    COT_PROVIDER=groq COT_API_KEY=... ./venv/bin/python scripts/generate_cot_spider.py
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

from generate_cot.config import (
    MAX_CONCURRENCY, MAX_TOKENS, RETRY_BASE_DELAY, TEMPERATURE,
    get_api_key, get_base_url, get_model, get_provider,
)
# reuse the SQL distillation prompt + parser
from generate_cot_sql import build_messages_sql, parse_sql_trace, load_completed_ids

TRAIN_OUTPUT = "data/spider/spider_cot_traces.jsonl"
TEST_OUTPUT = "data/spider/spider_test.jsonl"


def load_spider_schemas() -> dict:
    """db_id -> serialized schema string (from richardr1126/spider-schema)."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("richardr1126/spider-schema",
                        "spider_schema_rows_v2.json", repo_type="dataset")
    schemas = {}
    for row in json.load(open(p)):
        db = row["db_id"]
        text = row.get("Schema (values (type))", "")
        fk = (row.get("Foreign Keys") or "").strip()
        if fk and fk.lower() != "none":
            text += " | foreign keys: " + fk
        schemas[db] = text
    return schemas


def load_spider(split: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("xlangai/spider", split=split)
    schemas = load_spider_schemas()
    examples = []
    for i, row in enumerate(ds):
        db = row["db_id"]
        q = (row.get("question") or "").strip()
        sql = (row.get("query") or "").strip()
        schema = schemas.get(db, "")
        if not q or not sql or not schema:
            continue
        examples.append({
            "instance_id": f"{split}_{i}", "db_id": db, "question": q,
            "schema": schema, "sql": sql, "data_source": f"spider_{split}",
        })
    return examples


async def generate_one(client, model, ex, semaphore) -> dict:
    messages = build_messages_sql(ex["schema"], ex["question"], ex["sql"])
    base = {k: ex[k] for k in ("instance_id", "db_id", "question", "schema", "sql", "data_source")}
    async with semaphore:
        last_error = None
        for attempt in range(8):
            try:
                response = await client.chat.completions.create(
                    model=model, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
                break
            except Exception as e:
                last_error = e
                err = str(e).lower()
                is_rl = "429" in err or "queue" in err or "rate" in err
                if attempt < 7:
                    delay = (min(60, 5 * (1.5 ** attempt)) if is_rl
                             else RETRY_BASE_DELAY * (2 ** attempt)) + random.uniform(0, 2)
                    await asyncio.sleep(delay)
                else:
                    return {**base, "reasoning": "", "generated_sql": "", "parse_error": str(last_error)}
    raw = response.choices[0].message.content or ""
    reasoning, generated = parse_sql_trace(raw)
    usage = response.usage
    meta = {"prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0}
    if not reasoning:
        return {**base, "reasoning": "", "generated_sql": "",
                "parse_error": "no reasoning", "raw_response": raw[:1500], "generation_metadata": meta}
    return {**base, "reasoning": reasoning, "generated_sql": generated, "generation_metadata": meta}


async def run_generation(args):
    examples = load_spider("train")
    provider, base_url, model, api_key = get_provider(), get_base_url(), get_model(), get_api_key()
    print(f"Provider: {provider} ({base_url}) | Model: {model}")
    print(f"Spider train: {len(examples)} valid (q, schema, sql) triples")

    completed = load_completed_ids(args.output)
    if completed:
        print(f"Resuming: {len(completed)} done")
    pending = [e for e in examples if e["instance_id"] not in completed]
    if args.limit > 0:
        random.seed(42); random.shuffle(pending); pending = pending[:args.limit]
        print(f"Limit: {len(pending)} traces")
    if not pending:
        print("Nothing to do."); return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    start, processed, ok, fail = time.time(), 0, 0, 0
    with open(args.output, "a") as out_f:
        for bs in range(0, len(pending), 10):
            batch = pending[bs:bs + 10]
            for rec in await asyncio.gather(*[generate_one(client, model, e, semaphore) for e in batch]):
                out_f.write(json.dumps(rec) + "\n")
                ok += 1 if rec.get("reasoning") else 0
                fail += 0 if rec.get("reasoning") else 1
            out_f.flush()
            processed += len(batch)
            rate = processed / (time.time() - start) if time.time() > start else 0
            eta = (len(pending) - processed) / rate / 60 if rate else 0
            print(f"[{processed}/{len(pending)}] {rate:.2f}/s | ok={ok} fail={fail} | ETA {eta:.1f}m", flush=True)
    print(f"\nDone. {processed} in {(time.time() - start) / 60:.1f}m. ok={ok} fail={fail}")


def prepare_test():
    examples = load_spider("validation")  # Spider's dev set is the standard eval split
    os.makedirs(os.path.dirname(TEST_OUTPUT) or ".", exist_ok=True)
    with open(TEST_OUTPUT, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {len(examples)} test (dev) examples to {TEST_OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description="Generate Spider CoT traces (positive control, execution-evaluable)")
    parser.add_argument("--prepare-test", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--output", type=str, default=TRAIN_OUTPUT)
    args = parser.parse_args()
    if args.prepare_test:
        prepare_test()
        return
    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
