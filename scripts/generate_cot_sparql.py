"""Generate CoT reasoning traces for SPARQL (LC-QuAD 2.0) training examples.

First cross-formalism test of the "CoT-as-compositional-prior" hypothesis:
SPARQL's basic graph pattern is a connected triple pattern (holistic, like
Cypher), so we predict CoT will HURT — same as Cypher, unlike SQL.

Reuses the same provider-agnostic config from generate_cot.config
(Cerebras/Groq + GPT-oss-120B by default). Distillation: the teacher SEES the
reference SPARQL and explains how to derive it (we train on the reference query,
not the generated one — identical to the Cypher/ZOG CoT pipeline).

Usage (from project root):
    # one-time: dump the test split (no API needed)
    ./venv/bin/python scripts/generate_cot_sparql.py --prepare-test

    # generate train reasoning traces (quick first pass: 5k)
    COT_PROVIDER=groq COT_API_KEY=... ./venv/bin/python scripts/generate_cot_sparql.py --limit 5000
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

TRAIN_OUTPUT = "data/sparql/sparql_cot_traces.jsonl"
TEST_OUTPUT = "data/sparql/sparql_test.jsonl"

# ---------------------------------------------------------------------------
# SPARQL distillation prompt (QDecomp+InterCOL style, adapted to triples)
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are an expert in SPARQL and Wikidata. You are given a natural-language "
    "question and its reference SPARQL query over Wikidata. Produce clear, "
    "step-by-step reasoning that explains how to derive the query, then restate "
    "the query verbatim.\n"
    "Use this four-step structure:\n"
    "1. Decompose the question into sub-questions.\n"
    "2. Identify the Wikidata entities (wd:Q...) and properties (wdt:P.../p:.../ps:...) each needs.\n"
    "3. Describe the triple pattern (how the triples connect into one graph pattern).\n"
    "4. Assemble the SPARQL query.\n"
    "Format your answer EXACTLY as:\n"
    "Reasoning:\n<your four steps>\n\nSPARQL output: <the query on one line>"
)

_EXEMPLARS = [
    (
        "What is the capital of France?",
        "SELECT ?o WHERE { wd:Q142 wdt:P36 ?o }",
        "Reasoning:\n"
        "1. Sub-questions: (a) Which entity is France? (b) What is its capital?\n"
        "2. Schema elements: France = wd:Q142; capital property = wdt:P36.\n"
        "3. Triple pattern: a single connected triple France -P36-> ?capital.\n"
        "4. Assemble: select the object of (France, capital, ?o).\n\n"
        "SPARQL output: SELECT ?o WHERE { wd:Q142 wdt:P36 ?o }",
    ),
    (
        "Who are the children of the director of Inception?",
        "SELECT ?child WHERE { wd:Q25188 wdt:P57 ?director . ?director wdt:P40 ?child }",
        "Reasoning:\n"
        "1. Sub-questions: (a) Who directed Inception? (b) Who are that director's children?\n"
        "2. Schema elements: Inception = wd:Q25188; director = wdt:P57; child = wdt:P40.\n"
        "3. Triple pattern: a connected 2-hop path Inception -P57-> ?director -P40-> ?child; "
        "?director is the shared join variable.\n"
        "4. Assemble: chain the two triples sharing ?director.\n\n"
        "SPARQL output: SELECT ?child WHERE { wd:Q25188 wdt:P57 ?director . ?director wdt:P40 ?child }",
    ),
]


def build_messages_sparql(question: str, sparql: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for ex_q, ex_sparql, ex_answer in _EXEMPLARS:
        messages.append({
            "role": "user",
            "content": f"Question: {ex_q}\nReference SPARQL: {ex_sparql}\n\n"
                       "Produce the reasoning and restate the SPARQL query in the required format.",
        })
        messages.append({"role": "assistant", "content": ex_answer})
    messages.append({
        "role": "user",
        "content": f"Question: {question}\nReference SPARQL: {sparql}\n\n"
                   "Produce the reasoning and restate the SPARQL query in the required format.",
    })
    return messages


def parse_sparql_trace(raw_text: str) -> tuple[str, str]:
    """Return (reasoning, generated_sparql) from the teacher output."""
    marker = "SPARQL output:"
    idx = raw_text.rfind(marker)
    if idx == -1:
        return "", ""
    reasoning = raw_text[:idx]
    generated = raw_text[idx + len(marker):].strip().strip("`").strip()
    # strip a leading "Reasoning:" label if present
    reasoning = reasoning.strip()
    if reasoning.lower().startswith("reasoning:"):
        reasoning = reasoning[len("reasoning:"):].strip()
    return reasoning, generated


# ---------------------------------------------------------------------------
# Data loading (LC-QuAD 2.0)
# ---------------------------------------------------------------------------

def _clean_nl(row: dict) -> str | None:
    for field in ("question", "paraphrased_question", "NNQT_question"):
        v = (row.get(field) or "").strip()
        if v and v.lower() not in ("n/a", "na", "null", "none", "{}"):
            return v
    return None


def load_lcquad(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("lc_quad", split=split)
    examples = []
    for i, row in enumerate(ds):
        nl = _clean_nl(row)
        sparql = (row.get("sparql_wikidata") or "").strip()
        if not nl or not sparql:
            continue
        examples.append({
            "instance_id": str(row.get("uid", i)),
            "question": nl,
            "sparql": sparql,
            "data_source": f"lcquad2_{split}",
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_one(client, model, example, semaphore) -> dict:
    messages = build_messages_sparql(example["question"], example["sparql"])
    base = {
        "instance_id": example["instance_id"],
        "question": example["question"],
        "sparql": example["sparql"],
        "data_source": example["data_source"],
    }
    async with semaphore:
        last_error = None
        max_attempts = 8
        for attempt in range(max_attempts):
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
                if attempt < max_attempts - 1:
                    delay = (min(60, 5 * (1.5 ** attempt)) if is_rl
                             else RETRY_BASE_DELAY * (2 ** attempt)) + random.uniform(0, 2)
                    await asyncio.sleep(delay)
                else:
                    return {**base, "reasoning": "", "generated_sparql": "",
                            "parse_error": f"API error: {last_error}"}

    raw = response.choices[0].message.content or ""
    reasoning, generated = parse_sparql_trace(raw)
    usage = response.usage
    meta = {
        "model": model,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }
    if not reasoning:
        return {**base, "reasoning": "", "generated_sparql": "",
                "parse_error": "no reasoning parsed", "raw_response": raw[:1500],
                "generation_metadata": meta}
    return {**base, "reasoning": reasoning, "generated_sparql": generated,
            "generation_metadata": meta}


async def run_generation(args):
    examples = load_lcquad("train")
    total_available = len(examples)

    provider, base_url, model, api_key = (
        get_provider(), get_base_url(), get_model(), get_api_key()
    )
    print(f"Provider: {provider} ({base_url}) | Model: {model}")
    print(f"LC-QuAD 2.0 train: {total_available} valid (question, sparql) pairs")

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

    start, processed, ok, fail = time.time(), 0, 0, 0
    ptok = ctok = 0
    batch_size = 10
    with open(args.output, "a") as out_f:
        for bs in range(0, len(pending), batch_size):
            batch = pending[bs:bs + batch_size]
            results = await asyncio.gather(
                *[generate_one(client, model, e, semaphore) for e in batch]
            )
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
    examples = load_lcquad("test")
    os.makedirs(os.path.dirname(TEST_OUTPUT) or ".", exist_ok=True)
    with open(TEST_OUTPUT, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {len(examples)} test examples to {TEST_OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description="Generate SPARQL (LC-QuAD 2.0) CoT traces")
    parser.add_argument("--prepare-test", action="store_true",
                        help="Dump the LC-QuAD test split to data/sparql/sparql_test.jsonl (no API)")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of train traces (0 = all)")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--output", type=str, default=TRAIN_OUTPUT)
    args = parser.parse_args()

    if args.prepare_test:
        prepare_test()
        return
    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
