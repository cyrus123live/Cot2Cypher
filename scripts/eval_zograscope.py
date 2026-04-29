#!/usr/bin/env python3
"""
ZOGRASCOPE benchmark evaluation for our CoT Text2Cypher model.

Three steps:
  1. Format ZOGRASCOPE test examples with Pole schema for our model
  2. Run inference (via Modal or local)
  3. Execute predicted + reference Cypher against local Neo4j Pole database

Prerequisites:
  - Docker running with Neo4j + Pole graph (see setup instructions below)
  - Modal configured (for GPU inference) OR adapter weights local

Setup Neo4j with Pole graph:
  docker run -d --name pole-neo4j \
    -p 7687:7687 -p 7474:7474 \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    neo4j:5.0-community
  # Wait for startup, then load dump:
  docker exec pole-neo4j neo4j-admin database load --from-stdin neo4j < data/zograscope/pole-50.dump
  # Or use the data importer approach from the ZOGRASCOPE repo

Usage (run from project root):
  ./venv/bin/python scripts/eval_zograscope.py --format-only           # just create formatted test file
  ./venv/bin/python scripts/eval_zograscope.py --eval-predictions FILE  # evaluate existing predictions
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Schema formatting
# ---------------------------------------------------------------------------

POLE_SCHEMA_PATH = "data/zograscope/graph_schema.json"


def load_pole_schema_text():
    """Format the Pole graph schema as text for our model's prompt."""
    with open(POLE_SCHEMA_PATH) as f:
        schema = json.load(f)

    lines = ["Node properties:"]
    # Build property-to-class mapping from relations (domain/range)
    class_props = defaultdict(set)

    # From the schema, we know which properties belong to which classes
    # based on the Pole graph documentation
    POLE_CLASS_PROPERTIES = {
        "Person": ["name", "surname", "nhs_no", "age"],
        "Location": ["address"],
        "Phone": ["phoneNo"],
        "Email": ["email_address"],
        "Officer": ["name", "surname", "badge_no", "rank"],
        "PostCode": ["postcode"],
        "Area": ["areaCode"],
        "PhoneCall": ["call_time", "call_date", "call_duration"],
        "Crime": ["date", "type", "last_outcome"],
        "Object": ["type"],
        "Vehicle": ["make", "model", "year"],
    }

    for cls, desc in schema["classes"].items():
        props = POLE_CLASS_PROPERTIES.get(cls, [])
        lines.append(f"- **{cls}** ({desc['description']})")
        for prop in props:
            prop_info = schema["properties"].get(prop, {})
            prop_type = prop_info.get("type", "STRING")
            lines.append(f"  - `{prop}`: {prop_type}")

    lines.append("\nRelationship types:")
    for rel, info in schema["relations"].items():
        domain = info["domain"]
        range_ = info["range"]
        desc = info["description"]
        lines.append(f"- **{domain}** -[:{rel}]-> **{range_}** ({desc})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zograscope_test():
    """Load ZOGRASCOPE test data with IID/compositional split labels."""
    # Load split IDs
    with open("data/zograscope/ids_iid_test.txt") as f:
        iid_ids = set(line.strip() for line in f if line.strip())
    with open("data/zograscope/ids_compositional_test.txt") as f:
        comp_ids = set(line.strip() for line in f if line.strip())

    examples = []
    with open("data/zograscope/zograscope_test_v1.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = "iid" if row["id"] in iid_ids else "compositional" if row["id"] in comp_ids else "unknown"
            examples.append({
                "id": row["id"],
                "question": row["nl"],
                "question_linked": row["nl_gold_linked"],
                "reference_cypher": row["mr"],
                "num_nodes": int(row["num_nodes"]),
                "template_id": row["template_id"],
                "query_type": row["type"],
                "split": split,
            })
    return examples


def load_zograscope_length_test():
    """Load ZOGRASCOPE length generalization test data."""
    examples = []
    with open("data/zograscope/zograscope_length_test_v1.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append({
                "id": row["id"],
                "question": row["nl"],
                "question_linked": row["nl_gold_linked"],
                "reference_cypher": row["mr"],
                "num_nodes": int(row["num_nodes"]),
                "template_id": row["template_id"],
                "query_type": row["type"],
                "split": "length",
            })
    return examples


# ---------------------------------------------------------------------------
# Format for our model
# ---------------------------------------------------------------------------

def format_for_model(examples, schema_text):
    """Format ZOGRASCOPE examples for our CoT model's prompt."""
    formatted = []
    for ex in examples:
        # Use gold-linked question (entity values pre-identified)
        question = ex["question_linked"]
        formatted.append({
            "id": ex["id"],
            "schema": schema_text,
            "question": question,
            "reference_cypher": ex["reference_cypher"],
            "num_nodes": ex["num_nodes"],
            "split": ex["split"],
            "query_type": ex["query_type"],
        })
    return formatted


# ---------------------------------------------------------------------------
# Execution-based evaluation against local Neo4j
# ---------------------------------------------------------------------------

def evaluate_execution(predictions_path, neo4j_uri="bolt://localhost:7687",
                       neo4j_user="neo4j", neo4j_password="password"):
    """Evaluate predictions by executing against local Neo4j Pole database."""
    from neo4j import GraphDatabase
    from neo4j.exceptions import CypherSyntaxError, ClientError, DatabaseError

    with open(predictions_path) as f:
        records = [json.loads(line) for line in f]

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # Quick connectivity test
    try:
        driver.execute_query("RETURN 1 AS test", database_="neo4j")
        print("Connected to Neo4j Pole database.")
    except Exception as e:
        print(f"ERROR connecting to Neo4j: {e}")
        print("Make sure Docker is running with the Pole graph loaded.")
        return

    results = []
    for i, rec in enumerate(records):
        # Execute reference
        try:
            ref_records, _, _ = driver.execute_query(
                rec["reference_cypher"], database_="neo4j", timeout=30
            )
            ref_rows = sorted(
                json.dumps({k: v for k, v in r.items()}, sort_keys=True, default=str)
                for r in ref_records
            )
            ref_result = "\n".join(ref_rows)
            ref_error = None
        except Exception as e:
            ref_result = None
            ref_error = str(e)[:100]

        # Execute prediction
        try:
            pred_records, _, _ = driver.execute_query(
                rec["predicted_cypher"], database_="neo4j", timeout=30
            )
            pred_rows = sorted(
                json.dumps({k: v for k, v in r.items()}, sort_keys=True, default=str)
                for r in pred_records
            )
            pred_result = "\n".join(pred_rows)
            pred_error = None
        except Exception as e:
            pred_result = None
            pred_error = str(e)[:100]

        match = (
            ref_result is not None
            and pred_result is not None
            and ref_result == pred_result
        )

        results.append({
            "id": rec["id"],
            "split": rec.get("split", "unknown"),
            "num_nodes": rec.get("num_nodes"),
            "exec_match": match,
            "pred_error": pred_error,
            "ref_error": ref_error,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(records)}]")

    driver.close()

    # Compute metrics by split
    print("\n" + "=" * 60)
    print("ZOGRASCOPE EXECUTION ACCURACY")
    print("=" * 60)

    splits = defaultdict(lambda: {"total": 0, "correct": 0, "pred_err": 0, "ref_err": 0})
    for r in results:
        s = r["split"]
        splits[s]["total"] += 1
        if r["exec_match"]:
            splits[s]["correct"] += 1
        if r["pred_error"]:
            splits[s]["pred_err"] += 1
        if r["ref_error"]:
            splits[s]["ref_err"] += 1

    for split_name in ["iid", "compositional", "length", "unknown"]:
        s = splits[split_name]
        if s["total"] == 0:
            continue
        acc = s["correct"] / s["total"]
        print(f"\n{split_name.upper()} (n={s['total']}):")
        print(f"  Execution Accuracy: {acc:.4f} ({s['correct']}/{s['total']})")
        print(f"  Prediction Errors:  {s['pred_err']}")
        print(f"  Reference Errors:   {s['ref_err']}")

    # By num_nodes
    print(f"\nBy query complexity (num_nodes):")
    node_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        nn = r.get("num_nodes", "?")
        node_stats[nn]["total"] += 1
        if r["exec_match"]:
            node_stats[nn]["correct"] += 1

    for nn in sorted(node_stats.keys()):
        s = node_stats[nn]
        print(f"  {nn}-node: {s['correct']}/{s['total']} = {s['correct']/s['total']:.4f}")

    print("=" * 60)

    # Save results
    results_path = predictions_path.replace("predictions_", "zog_exec_").replace(".jsonl", "_exec.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDetailed results saved to {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ZOGRASCOPE benchmark evaluation")
    parser.add_argument("--format-only", action="store_true",
                        help="Just format test data for model input, don't run inference")
    parser.add_argument("--include-length", action="store_true",
                        help="Also include length generalization test set")
    parser.add_argument("--eval-predictions", type=str,
                        help="Evaluate an existing predictions JSONL file against Neo4j")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-password", default="password")
    args = parser.parse_args()

    if args.eval_predictions:
        evaluate_execution(
            args.eval_predictions,
            neo4j_uri=args.neo4j_uri,
            neo4j_password=args.neo4j_password,
        )
        return

    # Load schema and data
    print("Loading Pole graph schema...")
    schema_text = load_pole_schema_text()
    print(f"Schema: {len(schema_text)} chars")
    print(schema_text[:500])
    print("...")

    print("\nLoading ZOGRASCOPE test data...")
    examples = load_zograscope_test()
    iid = [e for e in examples if e["split"] == "iid"]
    comp = [e for e in examples if e["split"] == "compositional"]
    print(f"  IID: {len(iid)}, Compositional: {len(comp)}, Total: {len(examples)}")

    if args.include_length:
        length_examples = load_zograscope_length_test()
        print(f"  Length generalization: {len(length_examples)}")
        examples.extend(length_examples)

    # Format for our model
    formatted = format_for_model(examples, schema_text)

    # Save formatted data
    output_path = "data/zograscope/zograscope_formatted.jsonl"
    with open(output_path, "w") as f:
        for ex in formatted:
            f.write(json.dumps(ex) + "\n")

    print(f"\nFormatted {len(formatted)} examples saved to {output_path}")
    print(f"Schema included in each example ({len(schema_text)} chars)")
    print("\nNext steps:")
    print("  1. Run inference via Modal or DRAC")
    print("  2. Start Neo4j: docker run -d --name pole-neo4j \\")
    print("       -p 7687:7687 -p 7474:7474 \\")
    print("       -e NEO4J_AUTH=neo4j/password \\")
    print("       neo4j:5.0-community")
    print("  3. Load Pole graph dump")
    print("  4. Evaluate: ./venv/bin/python scripts/eval_zograscope.py --eval-predictions <file>")


if __name__ == "__main__":
    main()
