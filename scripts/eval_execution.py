#!/usr/bin/env python3
"""
Execution-based evaluation for Text2Cypher predictions.

Runs predicted and reference Cypher queries against Neo4j demo databases
at demo.neo4jlabs.com and compares results.

Run from project root:
    ./venv/bin/python scripts/eval_execution.py results/predictions_cot_4bit_greedy.jsonl
    ./venv/bin/python scripts/eval_execution.py results/predictions_4bit_greedy.jsonl
    ./venv/bin/python scripts/eval_execution.py results/predictions_cot_4bit_greedy.jsonl --gleu
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict

# Suppress verbose Neo4j warnings (property not found, deprecations)
logging.getLogger("neo4j").setLevel(logging.ERROR)

from neo4j import GraphDatabase
from neo4j.exceptions import (
    ClientError,
    CypherSyntaxError,
    DatabaseError,
    ServiceUnavailable,
    TransientError,
)

# Map dataset alias -> actual database name on demo.neo4jlabs.com
ALIAS_TO_DB = {
    "neo4jlabs_demo_db_recommendations": "recommendations",
    "neo4jlabs_demo_db_companies": "companies",
    "neo4jlabs_demo_db_eoflix": "neoflix",  # typo in dataset
    "neo4jlabs_demo_db_movies": "movies",
    "neo4jlabs_demo_db_northwind": "northwind",
    "neo4jlabs_demo_db_twitch": "twitch",
    "neo4jlabs_demo_db_grandstack": "grandstack",
    "neo4jlabs_demo_db_gameofthrones": "gameofthrones",
    "neo4jlabs_demo_db_fincen": "fincen",
    "neo4jlabs_demo_db_twitter": "twitter",
    "neo4jlabs_demo_db_buzzoverflow": "buzzoverflow",
    "neo4jlabs_demo_db_offshoreleaks": "offshoreleaks",
    "neo4jlabs_demo_db_network": "network",
    "neo4jlabs_demo_db_stackoverflow2": "stackoverflow2",
    "neo4jlabs_demo_db_bluesky": "bluesky",
    "neo4jlabs_demo_db_openstreetmap": "openstreetmap",
}

NEO4J_URI = "neo4j+s://demo.neo4jlabs.com"
QUERY_TIMEOUT = 30  # seconds


def get_driver(db_name: str) -> GraphDatabase.driver:
    """Create a Neo4j driver for a demo database."""
    return GraphDatabase.driver(NEO4J_URI, auth=(db_name, db_name))


MAX_RESULT_ROWS = 10000  # Cap results to avoid OOM on runaway queries


def execute_cypher(driver, db_name: str, cypher: str) -> tuple[str | None, str | None]:
    """Execute a Cypher query and return (result_string, error).

    Result string is a lexicographically sorted string representation
    of all records, matching Neo4j's evaluation methodology.
    Uses a session with explicit transaction timeout for reliability.
    """
    try:
        with driver.session(database=db_name) as session:
            result = session.run(cypher, timeout=QUERY_TIMEOUT)
            rows = []
            count = 0
            for record in result:
                count += 1
                if count > MAX_RESULT_ROWS:
                    # Consume remaining to avoid connection issues
                    result.consume()
                    return None, f"too_many_rows: >{MAX_RESULT_ROWS}"
                row = {k: record[k] for k in record.keys()}
                rows.append(json.dumps(row, sort_keys=True, default=str))
            rows.sort()
            return "\n".join(rows), None
    except (CypherSyntaxError, ClientError) as e:
        return None, f"syntax: {str(e)[:100]}"
    except (DatabaseError, TransientError) as e:
        return None, f"db: {str(e)[:100]}"
    except ServiceUnavailable as e:
        return None, f"unavailable: {str(e)[:100]}"
    except MemoryError:
        return None, "memory_error"
    except Exception as e:
        return None, f"other: {type(e).__name__}: {str(e)[:100]}"


DB_MAPPING_PATH = "data/test_db_mapping.json"


def load_predictions_with_db(predictions_path: str) -> list[dict]:
    """Load predictions and join with pre-extracted database mapping."""
    with open(DB_MAPPING_PATH) as f:
        db_map = json.load(f)  # instance_id -> database_reference_alias

    records = []
    with open(predictions_path) as f:
        for line in f:
            rec = json.loads(line)
            alias = db_map.get(rec["instance_id"])
            if alias and alias in ALIAS_TO_DB:
                rec["database_reference_alias"] = alias
                rec["db_name"] = ALIAS_TO_DB[alias]
                records.append(rec)

    return records


def run_execution_eval(predictions_path: str):
    """Run execution-based evaluation."""
    print(f"Loading predictions from {predictions_path}...")
    records = load_predictions_with_db(predictions_path)
    print(f"  {len(records)} instances with database access")

    # Group by database for connection reuse
    by_db = defaultdict(list)
    for rec in records:
        by_db[rec["db_name"]].append(rec)

    print(f"  {len(by_db)} databases")
    for db, recs in sorted(by_db.items(), key=lambda x: -len(x[1])):
        print(f"    {db}: {len(recs)} instances")

    # Checkpoint file
    base = os.path.basename(predictions_path).replace("predictions_", "exec_eval_")
    checkpoint_path = os.path.join(os.path.dirname(predictions_path), base)
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                r = json.loads(line)
                completed[r["instance_id"]] = r
        print(f"  Resuming: {len(completed)} already evaluated")

    # Evaluate database by database
    total = len(records)
    processed = len(completed)
    start_time = time.time()

    with open(checkpoint_path, "a") as out_f:
        for db_name, db_records in sorted(by_db.items(), key=lambda x: -len(x[1])):
            pending = [r for r in db_records if r["instance_id"] not in completed]
            if not pending:
                continue

            print(f"\n  Connecting to {db_name} ({len(pending)} pending)...")
            try:
                driver = get_driver(db_name)
                # Quick connectivity test
                driver.execute_query(
                    "RETURN 1 AS test", database_=db_name, timeout=10
                )
            except Exception as e:
                print(f"  ERROR connecting to {db_name}: {e}")
                print(f"  Skipping {len(pending)} instances")
                for rec in pending:
                    result = {
                        "instance_id": rec["instance_id"],
                        "db_name": db_name,
                        "exec_match": False,
                        "pred_error": f"connection_failed: {str(e)[:80]}",
                        "ref_error": None,
                        "skipped": True,
                    }
                    out_f.write(json.dumps(result) + "\n")
                    completed[rec["instance_id"]] = result
                    processed += 1
                out_f.flush()
                continue

            for i, rec in enumerate(pending):
                # Execute reference query
                ref_result, ref_error = execute_cypher(
                    driver, db_name, rec["reference_cypher"]
                )

                # Execute predicted query
                pred_result, pred_error = execute_cypher(
                    driver, db_name, rec["predicted_cypher"]
                )

                # Compare
                exec_match = (
                    ref_result is not None
                    and pred_result is not None
                    and ref_result == pred_result
                )

                result = {
                    "instance_id": rec["instance_id"],
                    "db_name": db_name,
                    "exec_match": exec_match,
                    "pred_error": pred_error,
                    "ref_error": ref_error,
                }
                out_f.write(json.dumps(result) + "\n")
                completed[rec["instance_id"]] = result
                processed += 1

                if (i + 1) % 20 == 0:
                    out_f.flush()
                    elapsed = time.time() - start_time
                    rate = (processed - len(completed) + len(pending)) / elapsed if elapsed > 0 else 0
                    print(f"    [{processed}/{total}] {db_name}: {i+1}/{len(pending)}")

            driver.close()
            out_f.flush()
            print(f"    {db_name} done.")

    # Compute metrics
    print("\n" + "=" * 64)
    print("EXECUTION-BASED EVALUATION RESULTS")
    print("=" * 64)

    eval_results = list(completed.values())
    non_skipped = [r for r in eval_results if not r.get("skipped")]
    matches = sum(1 for r in non_skipped if r["exec_match"])
    pred_errors = sum(1 for r in non_skipped if r["pred_error"])
    ref_errors = sum(1 for r in non_skipped if r["ref_error"])

    print(f"Total instances with DB:  {len(eval_results)}")
    print(f"Skipped (connection):     {len(eval_results) - len(non_skipped)}")
    print(f"Evaluated:                {len(non_skipped)}")
    print(f"Execution exact match:    {matches}/{len(non_skipped)} "
          f"= {matches/len(non_skipped):.4f}" if non_skipped else "N/A")
    print(f"Prediction errors:        {pred_errors}/{len(non_skipped)}")
    print(f"Reference errors:         {ref_errors}/{len(non_skipped)}")

    # Per-database breakdown
    db_stats = defaultdict(lambda: {"total": 0, "matches": 0, "pred_err": 0, "ref_err": 0, "skipped": 0})
    for r in eval_results:
        db = r["db_name"]
        db_stats[db]["total"] += 1
        if r.get("skipped"):
            db_stats[db]["skipped"] += 1
        else:
            if r["exec_match"]:
                db_stats[db]["matches"] += 1
            if r["pred_error"]:
                db_stats[db]["pred_err"] += 1
            if r["ref_error"]:
                db_stats[db]["ref_err"] += 1

    print(f"\n{'Database':<20} {'Total':>6} {'Match':>6} {'EM':>8} {'PredErr':>8} {'RefErr':>8}")
    print("-" * 58)
    for db, s in sorted(db_stats.items(), key=lambda x: -x[1]["total"]):
        evaluated = s["total"] - s["skipped"]
        em = s["matches"] / evaluated if evaluated > 0 else 0
        print(f"{db:<20} {s['total']:>6} {s['matches']:>6} {em:>8.4f} "
              f"{s['pred_err']:>8} {s['ref_err']:>8}")

    print("=" * 64)

    # Save summary metrics
    metrics_path = checkpoint_path.replace(".jsonl", "_metrics.json")
    metrics = {
        "total_with_db": len(eval_results),
        "evaluated": len(non_skipped),
        "exec_exact_match": matches / len(non_skipped) if non_skipped else 0,
        "exec_match_count": matches,
        "pred_errors": pred_errors,
        "ref_errors": ref_errors,
        "per_database": {
            db: {
                "total": s["total"],
                "matches": s["matches"],
                "exec_em": s["matches"] / (s["total"] - s["skipped"])
                if (s["total"] - s["skipped"]) > 0 else 0,
            }
            for db, s in db_stats.items()
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Detailed results in {checkpoint_path}")


def run_execution_gleu(predictions_path: str):
    """Run execution-based GLEU: execute queries, save result strings, compute GLEU."""
    import evaluate

    print(f"Loading predictions from {predictions_path}...")
    records = load_predictions_with_db(predictions_path)
    print(f"  {len(records)} instances with database access")

    by_db = defaultdict(list)
    for rec in records:
        by_db[rec["db_name"]].append(rec)

    # Checkpoint file for result strings
    base = os.path.basename(predictions_path).replace("predictions_", "exec_results_")
    checkpoint_path = os.path.join(os.path.dirname(predictions_path), base)
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                r = json.loads(line)
                completed[r["instance_id"]] = r
        print(f"  Resuming: {len(completed)} already evaluated")

    total = len(records)
    processed = len(completed)
    start_time = time.time()

    with open(checkpoint_path, "a") as out_f:
        for db_name, db_records in sorted(by_db.items(), key=lambda x: -len(x[1])):
            pending = [r for r in db_records if r["instance_id"] not in completed]
            if not pending:
                continue

            print(f"\n  Connecting to {db_name} ({len(pending)} pending)...")
            try:
                driver = get_driver(db_name)
                driver.execute_query("RETURN 1 AS test", database_=db_name, timeout=10)
            except Exception as e:
                print(f"  ERROR connecting to {db_name}: {e}")
                for rec in pending:
                    result = {
                        "instance_id": rec["instance_id"],
                        "db_name": db_name,
                        "pred_result": "",
                        "ref_result": "",
                        "pred_error": f"connection_failed: {str(e)[:80]}",
                        "ref_error": None,
                    }
                    out_f.write(json.dumps(result) + "\n")
                    completed[rec["instance_id"]] = result
                    processed += 1
                out_f.flush()
                continue

            for i, rec in enumerate(pending):
                ref_result, ref_error = execute_cypher(driver, db_name, rec["reference_cypher"])
                pred_result, pred_error = execute_cypher(driver, db_name, rec["predicted_cypher"])

                result = {
                    "instance_id": rec["instance_id"],
                    "db_name": db_name,
                    "pred_result": pred_result or "",
                    "ref_result": ref_result or "",
                    "pred_error": pred_error,
                    "ref_error": ref_error,
                }
                out_f.write(json.dumps(result) + "\n")
                completed[rec["instance_id"]] = result
                processed += 1

                if (i + 1) % 20 == 0:
                    out_f.flush()
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"    [{processed}/{total}] {db_name}: {i+1}/{len(pending)} "
                          f"({rate:.2f}/sec)")

            driver.close()
            out_f.flush()
            print(f"    {db_name} done.")

    # Compute metrics
    all_results = list(completed.values())
    # Only include instances where both queries returned results
    valid = [r for r in all_results if r["ref_result"] and not r["ref_error"]]
    pred_strs = []
    ref_strs = []
    matches = 0
    for r in valid:
        p = r["pred_result"] if r["pred_result"] and not r["pred_error"] else ""
        pred_strs.append(p)
        ref_strs.append(r["ref_result"])
        if p and p == r["ref_result"]:
            matches += 1

    gleu = evaluate.load("google_bleu")
    gleu_score = gleu.compute(
        predictions=pred_strs, references=[[ref] for ref in ref_strs]
    )["google_bleu"]

    print("\n" + "=" * 64)
    print("EXECUTION-BASED GLEU RESULTS")
    print("=" * 64)
    print(f"Total with DB:        {len(all_results)}")
    print(f"Valid (ref executed):  {len(valid)}")
    print(f"Execution GLEU:       {gleu_score:.4f}")
    print(f"Execution EM:         {matches}/{len(valid)} = {matches/len(valid):.4f}")

    # Per-database
    db_stats = defaultdict(lambda: {"preds": [], "refs": [], "matches": 0})
    for r in valid:
        db = r["db_name"]
        p = r["pred_result"] if r["pred_result"] and not r["pred_error"] else ""
        db_stats[db]["preds"].append(p)
        db_stats[db]["refs"].append(r["ref_result"])
        if p and p == r["ref_result"]:
            db_stats[db]["matches"] += 1

    print(f"\n{'Database':<20} {'Count':>6} {'GLEU':>8} {'EM':>8}")
    print("-" * 44)
    for db, s in sorted(db_stats.items(), key=lambda x: -len(x[1]["preds"])):
        n = len(s["preds"])
        db_gleu = gleu.compute(
            predictions=s["preds"], references=[[r] for r in s["refs"]]
        )["google_bleu"]
        print(f"{db:<20} {n:>6} {db_gleu:>8.4f} {s['matches']/n:>8.4f}")

    print("=" * 64)

    metrics_path = checkpoint_path.replace(".jsonl", "_metrics.json")
    metrics = {
        "total_with_db": len(all_results),
        "valid": len(valid),
        "exec_gleu": gleu_score,
        "exec_em": matches / len(valid) if valid else 0,
        "exec_match_count": matches,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_execution.py <predictions.jsonl> [--gleu]")
        sys.exit(1)

    use_gleu = "--gleu" in sys.argv
    predictions_path = [a for a in sys.argv[1:] if not a.startswith("--")][0]

    if use_gleu:
        run_execution_gleu(predictions_path)
    else:
        run_execution_eval(predictions_path)
