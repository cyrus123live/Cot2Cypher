#!/usr/bin/env python3
"""Execution-based candidate selection (MBR / result-clustering) for Text2Cypher.

Tests the CSC-SQL insight: instead of voting on candidate query STRINGS
(which failed for us — null +0.9%, because Cypher has ~one canonical surface
form), vote on candidate EXECUTION RESULTS. Group the N candidates by what
they return against the database, pick the largest result-cluster, and use a
representative query from that cluster.

This is the correct form of self-consistency for a constrained output space:
semantically-equivalent-but-syntactically-different queries collapse into the
same result cluster and reinforce each other.

Reuses the SC@5 candidates already generated (results/predictions_cot_sc5_t0.7.jsonl,
the `candidates` field) — NO new model inference required, only DB execution.

Run from project root (Pole Docker not needed; uses demo.neo4jlabs.com):
    ./venv/bin/python scripts/eval_execution_selection.py results/predictions_cot_sc5_t0.7.jsonl
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict

# Reuse the hardened execution machinery (driver timeouts, hard wall-clock kill)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_execution import execute_cypher, get_driver, load_predictions_with_db


def normalize_cypher(s: str) -> str:
    return " ".join((s or "").split()).strip().rstrip(";")


def select_by_execution(candidates, exec_results):
    """Given candidate queries and their execution result-strings, return the
    query from the largest result-cluster (ties broken by first occurrence).

    Candidates whose execution errored (result is None) are excluded from
    voting. If ALL candidates error, fall back to the first candidate.
    """
    # Cluster candidate indices by their execution result string
    clusters = defaultdict(list)
    for i, res in enumerate(exec_results):
        if res is not None:
            clusters[res].append(i)

    if not clusters:
        # every candidate errored — nothing executed; fall back to first
        return candidates[0], "all_errored"

    # Largest cluster wins; tie -> cluster containing the earliest candidate
    best_result = max(clusters, key=lambda r: (len(clusters[r]), -min(clusters[r])))
    rep_idx = min(clusters[best_result])  # representative = earliest candidate in cluster
    return candidates[rep_idx], best_result


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_execution_selection.py <sc_predictions.jsonl>")
        sys.exit(1)
    predictions_path = sys.argv[1]

    records = load_predictions_with_db(predictions_path)
    print(f"DB-eligible records with candidates: {len(records)}")

    # Group by database to reuse one driver per DB
    by_db = defaultdict(list)
    for r in records:
        if r.get("candidates"):
            by_db[r["db_name"]].append(r)

    out_path = predictions_path.replace("predictions_", "exec_select_").replace(
        ".jsonl", "_select.jsonl"
    )
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                results[r["instance_id"]] = r
        print(f"Resuming: {len(results)} already done")

    total = len(records)
    processed = len(results)
    start = time.time()

    with open(out_path, "a") as out_f:
        for db_name, recs in sorted(by_db.items(), key=lambda x: -len(x[1])):
            pending = [r for r in recs if r["instance_id"] not in results]
            if not pending:
                continue
            print(f"\nConnecting to {db_name} ({len(pending)} pending)...", flush=True)
            try:
                driver = get_driver(db_name)
                driver.execute_query("RETURN 1 AS test", database_=db_name, timeout=10)
            except Exception as e:
                print(f"  connect failed: {str(e)[:80]} — skipping", flush=True)
                continue

            for i, rec in enumerate(pending):
                ref_result, ref_error = execute_cypher(
                    driver, db_name, rec["reference_cypher"]
                )
                cands = rec["candidates"]
                cand_results = []
                for c in cands:
                    res, err = execute_cypher(driver, db_name, c)
                    cand_results.append(res)

                selected_query, selected_result = select_by_execution(cands, cand_results)

                # Exec match: selected query's result matches reference's result
                exec_match = (
                    ref_result is not None
                    and selected_result not in (None, "all_errored")
                    and selected_result == ref_result
                )

                # For comparison: how many of the 5 candidates individually matched?
                n_cand_match = sum(
                    1 for cr in cand_results
                    if ref_result is not None and cr is not None and cr == ref_result
                )

                row = {
                    "instance_id": rec["instance_id"],
                    "db_name": db_name,
                    "exec_match": exec_match,
                    "selected_query": selected_query,
                    "ref_error": ref_error,
                    "n_candidates_matching_ref": n_cand_match,
                    "n_candidates_executed": sum(1 for cr in cand_results if cr is not None),
                }
                out_f.write(json.dumps(row) + "\n")
                results[rec["instance_id"]] = row
                processed += 1

                if (i + 1) % 20 == 0:
                    out_f.flush()
                    elapsed = time.time() - start
                    print(f"  [{processed}/{total}] {db_name}: {i+1}/{len(pending)} "
                          f"({elapsed/60:.1f}m)", flush=True)
            driver.close()

    # Final metrics
    rows = list(results.values())
    valid = [r for r in rows if not r.get("ref_error")]
    matches = sum(1 for r in valid if r["exec_match"])
    n = len(valid)
    # "oracle" upper bound: instances where ANY candidate matched the reference
    oracle = sum(1 for r in valid if r["n_candidates_matching_ref"] > 0)

    print("\n" + "=" * 60)
    print("EXECUTION-BASED SELECTION RESULTS (MBR / result-clustering)")
    print("=" * 60)
    print(f"Valid instances (ref executed):     {n}")
    print(f"Execution-selected exec EM:         {matches}/{n} = {matches/n:.4f}")
    print(f"Oracle (any candidate correct):     {oracle}/{n} = {oracle/n:.4f}")
    print(f"  -> selection captures {matches/oracle*100:.1f}% of the oracle ceiling" if oracle else "")
    print("\nCompare to (from prior runs, same 2,471-DB subset):")
    print("  Greedy CoT exec EM:        ~0.2554")
    print("  String-voted SC@5 exec EM: ~0.2509")
    print("=" * 60)

    metrics_path = out_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "valid_instances": n,
            "exec_select_em": matches / n if n else 0,
            "exec_select_match_count": matches,
            "oracle_any_candidate": oracle / n if n else 0,
            "oracle_count": oracle,
        }, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
