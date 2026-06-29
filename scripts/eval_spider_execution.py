#!/usr/bin/env python3
"""Execution accuracy for Spider predictions — the metric the CoT-helps-SQL
literature actually uses (and the one our gretel string-EM run was missing).

Runs the predicted and gold SQL against each question's SQLite database and
compares result sets. This credits semantically-equivalent-but-syntactically-
different SQL (aliasing, JOIN vs subquery, clause order) that string EM penalizes.

Comparison rule (a standard approximation of Spider execution accuracy):
  - if the gold query has ORDER BY -> compare rows as an ordered list;
  - otherwise -> compare as a multiset (sorted rows).
Column order within a row is kept (same as gold). This may undercount absolute
accuracy vs the official permutation-aware eval, but it is applied identically to
both arms, so the direct-vs-CoT DELTA is valid.

Pass the direct and cot prediction files together to get the CoT delta.

Usage:
  python eval_spider_execution.py --db-dir ~/scratch/spider_database \\
      results/predictions_spider_direct.jsonl results/predictions_spider_cot.jsonl
"""

import argparse
import json
import os
import sqlite3
import time


def execute(db_path: str, query: str, timeout: float = 15.0):
    """Run query against the sqlite db; return (rows, error)."""
    if not os.path.exists(db_path):
        return None, "db missing"
    con = sqlite3.connect(db_path)
    con.text_factory = lambda b: b.decode("utf-8", "ignore")
    start = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - start > timeout else 0, 2000)
    try:
        rows = con.execute(query).fetchall()
        return rows, None
    except Exception as e:
        return None, str(e)[:120]
    finally:
        con.close()


def rows_match(pred_rows, gold_rows, order_matters: bool) -> bool:
    if pred_rows is None or gold_rows is None:
        return False
    p = [tuple(str(c) for c in r) for r in pred_rows]
    g = [tuple(str(c) for c in r) for r in gold_rows]
    return (p == g) if order_matters else (sorted(p) == sorted(g))


def db_path_for(db_dir: str, db_id: str) -> str:
    return os.path.join(db_dir, db_id, f"{db_id}.sqlite")


def score_file(path: str, db_dir: str) -> dict:
    recs = [json.loads(l) for l in open(path)]
    n_total = len(recs)
    valid = 0          # gold executed
    correct = 0        # pred result == gold result
    pred_errors = 0
    for r in recs:
        db = db_path_for(db_dir, r.get("db_id", ""))
        gold = r.get("reference_sql", "")
        pred = r.get("predicted_sql", "")
        gold_rows, gold_err = execute(db, gold)
        if gold_err is not None:
            continue  # skip instances whose gold doesn't execute (like Cypher ref-errors)
        valid += 1
        pred_rows, pred_err = execute(db, pred)
        if pred_err is not None:
            pred_errors += 1
            continue
        order_matters = "order by" in gold.lower()
        if rows_match(pred_rows, gold_rows, order_matters):
            correct += 1
    return {"path": path, "n_total": n_total, "valid": valid,
            "exec_acc": correct / valid if valid else 0.0,
            "correct": correct, "pred_errors": pred_errors}


def main():
    ap = argparse.ArgumentParser(description="Spider execution accuracy")
    ap.add_argument("--db-dir", required=True, help="Spider database/ dir ({db_id}/{db_id}.sqlite)")
    ap.add_argument("files", nargs="+", help="prediction JSONL file(s) (direct then cot)")
    args = ap.parse_args()

    if not os.path.isdir(args.db_dir):
        print(f"FATAL: --db-dir {args.db_dir} not found. Need the Spider database/ folder.")
        raise SystemExit(1)

    results = [score_file(p, args.db_dir) for p in args.files]
    print(f"\n{'file':<42}{'valid':>8}{'exec_acc':>10}{'pred_err':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['path'].split('/')[-1]:<42}{r['valid']:>8}{r['exec_acc']:>10.4f}{r['pred_errors']:>10}")

    if len(results) == 2:
        d, c = results
        print(f"\nCoT effect (file2 - file1) on EXECUTION accuracy: {c['exec_acc'] - d['exec_acc']:+.4f}")
        print("PREDICTION (compositional-prior theory): POSITIVE (CoT helps SQL on execution).")
        print("If positive here but negative for Cypher/SPARQL -> the theory is revived.")


if __name__ == "__main__":
    main()
