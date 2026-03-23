"""Post-generation analysis: stats, failure review, quality checks.

Usage:
    python -m generate_cot.analyze [--path data/cot_training_data.jsonl] [--sample 50]
"""

import argparse
import json
import random
import statistics
from collections import Counter


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def analyze(records: list[dict]):
    total = len(records)
    successes = [r for r in records if r.get("reasoning")]
    failures = [r for r in records if not r.get("reasoning")]

    print(f"{'=' * 60}")
    print(f"COT GENERATION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Total records:     {total}")
    print(f"Successes:         {len(successes)} ({len(successes)/total*100:.1f}%)")
    print(f"Failures:          {len(failures)} ({len(failures)/total*100:.1f}%)")

    # Reasoning length distribution
    if successes:
        lengths = [len(r["reasoning"]) for r in successes]
        print(f"\nReasoning length (chars):")
        print(f"  Min:    {min(lengths)}")
        print(f"  Median: {statistics.median(lengths):.0f}")
        print(f"  Mean:   {statistics.mean(lengths):.0f}")
        print(f"  Max:    {max(lengths)}")
        print(f"  Stdev:  {statistics.stdev(lengths):.0f}" if len(lengths) > 1 else "")

    # Cypher match rate (generated_cypher vs reference cypher)
    if successes:
        def normalize(s: str) -> str:
            return " ".join(s.split())
        matches = sum(
            1 for r in successes
            if normalize(r.get("generated_cypher", "")) == normalize(r["cypher"])
        )
        print(f"\nCypher match rate: {matches}/{len(successes)} ({matches/len(successes)*100:.1f}%)")

    # Per-source breakdown
    source_counts: dict[str, dict] = {}
    for r in records:
        src = r.get("data_source", "unknown")
        if src not in source_counts:
            source_counts[src] = {"total": 0, "success": 0, "fail": 0}
        source_counts[src]["total"] += 1
        if r.get("reasoning"):
            source_counts[src]["success"] += 1
        else:
            source_counts[src]["fail"] += 1

    print(f"\n{'Source':<40} {'Total':>6} {'OK':>6} {'Fail':>6} {'Rate':>7}")
    print("-" * 67)
    for src in sorted(source_counts, key=lambda s: -source_counts[s]["total"]):
        s = source_counts[src]
        rate = s["success"] / s["total"] * 100
        print(f"{src:<40} {s['total']:>6} {s['success']:>6} {s['fail']:>6} {rate:>6.1f}%")

    # Failure error breakdown
    if failures:
        error_counts = Counter(r.get("parse_error", "unknown") for r in failures)
        print(f"\nFailure reasons:")
        for error, count in error_counts.most_common(10):
            print(f"  {count:>5}  {error[:80]}")

    # Token usage
    total_prompt = sum(r.get("generation_metadata", {}).get("prompt_tokens", 0) for r in records)
    total_completion = sum(r.get("generation_metadata", {}).get("completion_tokens", 0) for r in records)
    print(f"\nToken usage:")
    print(f"  Prompt:     {total_prompt:>12,}")
    print(f"  Completion: {total_completion:>12,}")
    print(f"  Total:      {total_prompt + total_completion:>12,}")

    if total_prompt > 0:
        avg_prompt = total_prompt / total
        avg_completion = total_completion / total
        print(f"  Avg prompt/example:     {avg_prompt:,.0f}")
        print(f"  Avg completion/example: {avg_completion:,.0f}")

    print(f"{'=' * 60}")


def show_samples(records: list[dict], n: int = 10, failures_only: bool = False):
    """Print random samples for manual review."""
    if failures_only:
        pool = [r for r in records if not r.get("reasoning")]
        label = "FAILURE"
    else:
        pool = [r for r in records if r.get("reasoning")]
        label = "SUCCESS"

    if not pool:
        print(f"No {label.lower()} records to sample.")
        return

    random.seed(42)
    samples = random.sample(pool, min(n, len(pool)))

    for i, r in enumerate(samples):
        print(f"\n{'─' * 60}")
        print(f"[{label} SAMPLE {i+1}] {r['instance_id']} ({r.get('data_source', '')})")
        print(f"Q: {r['question']}")
        print(f"Reference Cypher: {r['cypher'][:200]}")
        if r.get("reasoning"):
            print(f"Reasoning:\n{r['reasoning']}")
            print(f"Generated Cypher: {r.get('generated_cypher', '')[:200]}")
        else:
            print(f"Error: {r.get('parse_error', 'unknown')}")
            if r.get("raw_response"):
                print(f"Raw (first 500): {r['raw_response'][:500]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/cot_training_data.jsonl")
    parser.add_argument("--sample", type=int, default=10, help="Number of samples to show")
    parser.add_argument("--failures", action="store_true", help="Show failure samples")
    args = parser.parse_args()

    records = load_records(args.path)
    analyze(records)
    show_samples(records, n=args.sample, failures_only=args.failures)


if __name__ == "__main__":
    main()
