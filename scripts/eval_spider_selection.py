#!/usr/bin/env python3
"""Execution-grounded selection over sampled Spider candidates (direct vs CoT).

Scores the files written by `drac_train_sql.py --eval --num-samples N`
(predictions_spider_{variant}_scN.jsonl, one 'candidates' list per example).
Every candidate and the gold query are executed against the Spider SQLite
databases, and for each file we report:
  - string-vote : majority vote on whitespace-normalized SQL text, then executed
  - MBR         : cluster executable candidates by result set, pick the largest
                  cluster (ties -> the cluster holding the earliest sample)
  - oracle      : any candidate's result matches gold (the selection ceiling)
plus the oracle partition used in the paper for Cypher: generation-bound (no
correct candidate) / MBR-solved / verifier-addressable (a correct candidate
exists but loses the vote), and how many of the last are lone-correct.

With two files (direct FIRST, CoT second) it also prints paired-bootstrap 95%
CIs on the CoT-minus-direct deltas. Instances whose gold query fails to execute
are skipped, the same rule as eval_spider_execution.py.

Usage (from project root):
  ./venv/bin/python scripts/eval_spider_selection.py \\
      --db-dir data/spider/spider_data/database \\
      results/results_spider_sampling/predictions_spider_direct_sc5.jsonl \\
      results/results_spider_sampling/predictions_spider_cot_sc5.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_spider_execution import db_path_for, execute  # noqa: E402

B = 10_000
SEED = 42


def result_key(rows, order_matters: bool):
    """Canonical form of a result set; equal keys <=> rows_match() is True."""
    norm = [tuple(str(c) for c in r) for r in rows]
    return tuple(norm) if order_matters else tuple(sorted(norm))


def score_file(path: str, db_dir: str, timeout: float) -> dict:
    recs = [json.loads(line) for line in open(path)]
    per = {}  # instance_id -> {"sv": bool, "mbr": bool, "oracle": bool, "n_correct": int}
    n_cand_total = n_cand_err = 0
    for i, r in enumerate(recs):
        db = db_path_for(db_dir, r.get("db_id", ""))
        gold = r["reference_sql"]
        gold_rows, gold_err = execute(db, gold, timeout)
        if gold_err is not None:
            continue
        order = "order by" in gold.lower()
        gkey = result_key(gold_rows, order)

        cands = r["candidates"]
        keys = []  # per candidate: result key, or None if it failed to execute
        for c in cands:
            rows, err = execute(db, c, timeout)
            n_cand_total += 1
            if err is not None:
                n_cand_err += 1
                keys.append(None)
            else:
                keys.append(result_key(rows, order))

        correct = [k is not None and k == gkey for k in keys]

        # String vote: most common normalized text; Counter keeps first-seen order on ties.
        norm_txt = [" ".join(c.split()) for c in cands]
        winner_txt = Counter(norm_txt).most_common(1)[0][0]
        sv = correct[norm_txt.index(winner_txt)]

        # MBR: largest result cluster among executable candidates; ties -> earliest sample.
        counts = Counter(k for k in keys if k is not None)
        if counts:
            best = max(counts.values())
            mbr_key = next(k for k in keys if k is not None and counts[k] == best)
            mbr = mbr_key == gkey
        else:
            mbr = False

        per[r["instance_id"]] = {"sv": sv, "mbr": mbr, "oracle": any(correct),
                                 "n_correct": sum(correct)}
        if (i + 1) % 200 == 0:
            print(f"  [{os.path.basename(path)}] {i + 1}/{len(recs)}", file=sys.stderr)

    n = len(per)
    oracle = sum(v["oracle"] for v in per.values())
    mbr = sum(v["mbr"] for v in per.values())
    verif = [v for v in per.values() if v["oracle"] and not v["mbr"]]
    return {
        "path": path,
        "n_total": len(recs),
        "valid": n,
        "k": len(recs[0]["candidates"]) if recs else 0,
        "string_vote": sum(v["sv"] for v in per.values()) / n,
        "mbr": mbr / n,
        "oracle": oracle / n,
        "generation_bound": (n - oracle) / n,
        "mbr_solved": mbr / n,
        "verifier_addressable": len(verif) / n,
        "lone_correct_share": (sum(v["n_correct"] == 1 for v in verif) / len(verif)) if verif else 0.0,
        "candidate_exec_error_rate": n_cand_err / n_cand_total if n_cand_total else 0.0,
        "_per": per,
    }


def paired_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple:
    """Point + 95% percentile CI for mean(b) - mean(a), resampling paired instances."""
    idx = rng.integers(0, len(a), size=(B, len(a)))
    deltas = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return b.mean() - a.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser(description="Spider execution-grounded selection (MBR / oracle)")
    ap.add_argument("--db-dir", required=True, help="Spider database/ dir ({db_id}/{db_id}.sqlite)")
    ap.add_argument("--timeout", type=float, default=15.0, help="Per-query timeout (s)")
    ap.add_argument("--out", default=None,
                    help="Metrics JSON (default: selection_metrics.json next to the first file)")
    ap.add_argument("files", nargs="+", help="Sampled prediction files; direct first, CoT second")
    args = ap.parse_args()
    if not os.path.isdir(args.db_dir):
        sys.exit(f"FATAL: --db-dir {args.db_dir} not found.")

    results = [score_file(p, args.db_dir, args.timeout) for p in args.files]

    print(f"\n{'file':<40}{'valid':>6}{'k':>3}{'str-vote':>10}{'MBR':>8}{'oracle':>8}"
          f"{'gen-bound':>11}{'verif-addr':>12}")
    print("-" * 98)
    for r in results:
        print(f"{os.path.basename(r['path']):<40}{r['valid']:>6}{r['k']:>3}"
              f"{r['string_vote']:>10.4f}{r['mbr']:>8.4f}{r['oracle']:>8.4f}"
              f"{r['generation_bound']:>11.3f}{r['verifier_addressable']:>12.3f}")

    report = {"files": [{k: v for k, v in r.items() if k != "_per"} for r in results]}

    if len(results) == 2:
        d, c = results[0]["_per"], results[1]["_per"]
        ids = sorted(set(d) & set(c))
        rng = np.random.default_rng(SEED)
        print(f"\nCoT - direct on {len(ids)} paired instances (paired bootstrap 95% CI, B={B}):")
        report["cot_minus_direct"] = {}
        for metric in ("sv", "mbr", "oracle"):
            a = np.array([d[i][metric] for i in ids], dtype=float)
            b = np.array([c[i][metric] for i in ids], dtype=float)
            p, lo, hi = paired_ci(a, b, rng)
            name = {"sv": "string-vote", "mbr": "MBR", "oracle": "oracle"}[metric]
            print(f"  {name:<12} {p:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
            report["cot_minus_direct"][name] = {"delta": p, "lo": lo, "hi": hi, "n": len(ids)}

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.files[0])),
                                   "selection_metrics.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nMetrics saved to {out}")


if __name__ == "__main__":
    main()
