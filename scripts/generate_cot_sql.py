"""Generate CoT reasoning traces for SQL (gretelai/synthetic_text_to_sql).

The POSITIVE CONTROL for the CoT-as-compositional-prior hypothesis. SQL composes
(joins, sub-queries) so decomposition is the right inductive bias -> we predict
CoT HELPS, the opposite of Cypher/SPARQL. If CoT does NOT help SQL in our own
matched pipeline, the theory is dead regardless of SPARQL.

Dataset chosen for: inline schema (CREATE TABLE in `sql_context` — mirrors our
Cypher setup), compositional queries (joins/subqueries), and complexity labels
(`sql_complexity` — lets us show the CoT gain scales with complexity).

Distillation: teacher SEES the reference SQL and explains how to derive it; we
train on the reference query (identical to the Cypher/SPARQL CoT pipeline).

Usage (from project root):
    ./venv/bin/python scripts/generate_cot_sql.py --prepare-test
    COT_PROVIDER=groq COT_API_KEY=... ./venv/bin/python scripts/generate_cot_sql.py --limit 5000
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
    MAX_CONCURRENCY,
    MAX_TOKENS,
    RETRY_BASE_DELAY,
    TEMPERATURE,
    get_api_key,
    get_base_url,
    get_model,
    get_provider,
)

DATASET = "gretelai/synthetic_text_to_sql"
TRAIN_OUTPUT = "data/sql/sql_cot_traces.jsonl"
TEST_OUTPUT = "data/sql/sql_test.jsonl"

SYSTEM_MESSAGE = (
    "You are an expert in SQL. You are given a database schema, a natural-language "
    "question, and the reference SQL query. Produce clear, step-by-step reasoning "
    "that explains how to derive the query, then restate the query verbatim.\n"
    "Use this four-step structure:\n"
    "1. Decompose the question into sub-questions.\n"
    "2. Identify the tables and columns each sub-question needs (from the schema).\n"
    "3. Determine the structure: which joins, filters, grouping, or sub-queries connect them.\n"
    "4. Assemble the SQL query.\n"
    "Format your answer EXACTLY as:\n"
    "Reasoning:\n<your four steps>\n\nSQL output: <the query on one line>"
)

_EXEMPLARS = [
    (
        "CREATE TABLE head (head_id INT, name TEXT, age INT);",
        "How many heads of departments are older than 56?",
        "SELECT COUNT(*) FROM head WHERE age > 56",
        "Reasoning:\n"
        "1. Sub-questions: (a) Which heads are older than 56? (b) How many are there?\n"
        "2. Schema elements: table head, column age (filter), COUNT over rows.\n"
        "3. Structure: a single-table filter age > 56 plus a COUNT aggregation; no joins.\n"
        "4. Assemble: count rows of head where age > 56.\n\n"
        "SQL output: SELECT COUNT(*) FROM head WHERE age > 56",
    ),
    (
        "CREATE TABLE department (id INT, name TEXT); "
        "CREATE TABLE head (head_id INT, department_id INT, name TEXT);",
        "List the names of heads in the 'Treasury' department.",
        "SELECT h.name FROM head h JOIN department d ON h.department_id = d.id "
        "WHERE d.name = 'Treasury'",
        "Reasoning:\n"
        "1. Sub-questions: (a) Which department is 'Treasury'? (b) Which heads belong to it? (c) Their names?\n"
        "2. Schema elements: department(id, name), head(department_id, name); join key head.department_id = department.id.\n"
        "3. Structure: a two-table JOIN of head and department, filtered on department.name; "
        "this decomposes naturally into a join of two sub-results.\n"
        "4. Assemble: join the tables on the key and filter on the department name.\n\n"
        "SQL output: SELECT h.name FROM head h JOIN department d ON h.department_id = d.id WHERE d.name = 'Treasury'",
    ),
]


def build_messages_sql(schema: str, question: str, sql: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for ex_schema, ex_q, ex_sql, ex_answer in _EXEMPLARS:
        messages.append({
            "role": "user",
            "content": f"Schema: {ex_schema}\nQuestion: {ex_q}\nReference SQL: {ex_sql}\n\n"
                       "Produce the reasoning and restate the SQL query in the required format.",
        })
        messages.append({"role": "assistant", "content": ex_answer})
    messages.append({
        "role": "user",
        "content": f"Schema: {schema}\nQuestion: {question}\nReference SQL: {sql}\n\n"
                   "Produce the reasoning and restate the SQL query in the required format.",
    })
    return messages


def parse_sql_trace(raw_text: str) -> tuple[str, str]:
    marker = "SQL output:"
    idx = raw_text.rfind(marker)
    if idx == -1:
        return "", ""
    reasoning = raw_text[:idx].strip()
    if reasoning.lower().startswith("reasoning:"):
        reasoning = reasoning[len("reasoning:"):].strip()
    generated = raw_text[idx + len(marker):].strip().strip("`").strip()
    return reasoning, generated


def load_sql(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    examples = []
    for i, row in enumerate(ds):
        q = (row.get("sql_prompt") or "").strip()
        schema = (row.get("sql_context") or "").strip()
        sql = (row.get("sql") or "").strip()
        if not q or not schema or not sql:
            continue
        examples.append({
            "instance_id": str(row.get("id", i)),
            "question": q,
            "schema": schema,
            "sql": sql,
            "sql_complexity": row.get("sql_complexity", "unknown"),
            "data_source": f"gretel_sql_{split}",
        })
    return examples


def load_completed_ids(path: str) -> set[str]:
    completed = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


async def generate_one(client, model, ex, semaphore) -> dict:
    messages = build_messages_sql(ex["schema"], ex["question"], ex["sql"])
    base = {k: ex[k] for k in ("instance_id", "question", "schema", "sql",
                               "sql_complexity", "data_source")}
    async with semaphore:
        last_error = None
        for attempt in range(8):
            try:
                response = await client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                )
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
                    return {**base, "reasoning": "", "generated_sql": "",
                            "parse_error": f"API error: {last_error}"}
    raw = response.choices[0].message.content or ""
    reasoning, generated = parse_sql_trace(raw)
    usage = response.usage
    meta = {"model": model,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0}
    if not reasoning:
        return {**base, "reasoning": "", "generated_sql": "",
                "parse_error": "no reasoning parsed", "raw_response": raw[:1500],
                "generation_metadata": meta}
    return {**base, "reasoning": reasoning, "generated_sql": generated,
            "generation_metadata": meta}


async def run_generation(args):
    examples = load_sql("train")
    provider, base_url, model, api_key = (
        get_provider(), get_base_url(), get_model(), get_api_key())
    print(f"Provider: {provider} ({base_url}) | Model: {model}")
    print(f"{DATASET} train: {len(examples)} valid (q, schema, sql) triples")

    completed = load_completed_ids(args.output)
    if completed:
        print(f"Resuming: {len(completed)} already done")
    pending = [e for e in examples if e["instance_id"] not in completed]
    if args.limit > 0:
        random.seed(42)
        random.shuffle(pending)
        pending = pending[:args.limit]
        print(f"Limit: generating {len(pending)} traces")
    if not pending:
        print("Nothing to do.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    start, processed, ok, fail, ptok, ctok = time.time(), 0, 0, 0, 0, 0
    with open(args.output, "a") as out_f:
        for bs in range(0, len(pending), 10):
            batch = pending[bs:bs + 10]
            results = await asyncio.gather(
                *[generate_one(client, model, e, semaphore) for e in batch])
            for rec in results:
                out_f.write(json.dumps(rec) + "\n")
                ok += 1 if rec.get("reasoning") else 0
                fail += 0 if rec.get("reasoning") else 1
                m = rec.get("generation_metadata", {})
                ptok += m.get("prompt_tokens", 0)
                ctok += m.get("completion_tokens", 0)
            out_f.flush()
            processed += len(batch)
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed else 0
            eta = (len(pending) - processed) / rate / 60 if rate else 0
            print(f"[{processed}/{len(pending)}] {rate:.2f}/s | ok={ok} fail={fail} "
                  f"| tok={ptok + ctok:,} | ETA {eta:.1f}m", flush=True)
    print(f"\nDone. {processed} in {(time.time() - start) / 60:.1f}m. ok={ok} fail={fail}")


def prepare_test():
    examples = load_sql("test")
    os.makedirs(os.path.dirname(TEST_OUTPUT) or ".", exist_ok=True)
    with open(TEST_OUTPUT, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {len(examples)} test examples to {TEST_OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description="Generate SQL CoT traces (positive control)")
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
