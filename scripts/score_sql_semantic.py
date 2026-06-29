#!/usr/bin/env python3
"""Re-score SQL predictions with SQL-AWARE metrics (not raw string EM).

Why: our SQL CoT "negative" was measured with string EM/GLEU, but SQL has many
semantically-equivalent surface forms (aliasing, JOIN vs subquery, clause order),
so string match undercounts correctness and penalizes the CoT arm — which produces
valid-but-different SQL. This is exactly what the constrained-output-space
hypothesis predicts: string metrics work for Cypher (one canonical form) but FAIL
for SQL. This script tests whether CoT's apparent deficit shrinks as the metric
becomes more semantic.

Metrics per file:
  - raw_em    : exact string match
  - norm_em   : lowercase + whitespace-collapsed + trailing-; stripped
  - canon_em  : sqlglot canonical AST match — credits case/whitespace/quoting/
                function-casing and structural normalization, but NOT alias-rename
                or full semantic equivalence (that needs a schema + execution; see
                eval_spider_execution.py). So canon_em is a *lower bound* on the
                semantic match rate — a cheap directional check, not the last word.

Pass BOTH the direct and cot prediction files to see the CoT delta at each metric
level. If the delta closes from raw -> canon, surface form was inflating the gap;
if it does NOT close, the deficit is likely real and Spider execution is the test.

Setup + usage:
  ./venv/bin/pip install sqlglot
  ./venv/bin/python scripts/score_sql_semantic.py \\
      results/predictions_sql_direct.jsonl results/predictions_sql_cot.jsonl
"""

import json
import re
import sys
from collections import defaultdict

try:
    import sqlglot
    HAVE_SQLGLOT = True
except ImportError:
    HAVE_SQLGLOT = False


def _pred(r):
    return r.get("predicted_sql") or r.get("predicted") or r.get("prediction") or ""


def _ref(r):
    return r.get("reference_sql") or r.get("sql") or r.get("reference") or ""


def norm(s: str) -> str:
    s = (s or "").strip().rstrip(";")
    return re.sub(r"\s+", " ", s).lower()


def canon(s: str):
    """sqlglot canonical form, or None if it can't be parsed."""
    if not HAVE_SQLGLOT or not (s or "").strip():
        return None
    try:
        tree = sqlglot.parse_one(s, read="sqlite")
        return tree.sql(normalize=True, normalize_functions="upper", comments=False, pretty=False).lower()
    except Exception:
        return None


def score_file(path: str) -> dict:
    recs = [json.loads(l) for l in open(path)]
    n = len(recs)
    raw = norm_em = canon_em = ref_parsed = 0
    # per-complexity (if present)
    by_cx = defaultdict(lambda: [0, 0, 0])  # [raw, canon, total]
    for r in recs:
        p, g = _pred(r), _ref(r)
        cx = r.get("sql_complexity", "all")
        rmatch = p.strip().rstrip(";") == g.strip().rstrip(";")
        nmatch = norm(p) == norm(g)
        cp, cg = canon(p), canon(g)
        cmatch = cg is not None and cp is not None and cp == cg
        if cg is not None:
            ref_parsed += 1
        raw += rmatch
        norm_em += nmatch
        canon_em += cmatch
        by_cx[cx][0] += rmatch
        by_cx[cx][1] += cmatch
        by_cx[cx][2] += 1
    return {
        "path": path, "n": n,
        "raw_em": raw / n, "norm_em": norm_em / n,
        "canon_em": (canon_em / n) if HAVE_SQLGLOT else None,
        "ref_parse_rate": (ref_parsed / n) if HAVE_SQLGLOT else None,
        "by_cx": {cx: {"raw": v[0] / v[2], "canon": v[1] / v[2], "n": v[2]}
                  for cx, v in sorted(by_cx.items())},
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if not HAVE_SQLGLOT:
        print("WARNING: sqlglot not installed — canon_em unavailable. "
              "Install with: ./venv/bin/pip install sqlglot\n")
    results = [score_file(p) for p in sys.argv[1:]]

    print(f"{'file':<45}{'raw_em':>9}{'norm_em':>9}{'canon_em':>10}{'ref_parse':>11}")
    print("-" * 84)
    for r in results:
        ce = f"{r['canon_em']:.4f}" if r["canon_em"] is not None else "n/a"
        rp = f"{r['ref_parse_rate']:.3f}" if r["ref_parse_rate"] is not None else "n/a"
        print(f"{r['path'].split('/')[-1]:<45}{r['raw_em']:>9.4f}{r['norm_em']:>9.4f}{ce:>10}{rp:>11}")

    # CoT delta at each metric level (direct assumed first, cot second)
    if len(results) == 2:
        d, c = results
        print("\nCoT effect (file2 - file1) at each metric — does the deficit close as the metric gets semantic?")
        print(f"  raw_em  : {c['raw_em']-d['raw_em']:+.4f}")
        print(f"  norm_em : {c['norm_em']-d['norm_em']:+.4f}")
        if d["canon_em"] is not None and c["canon_em"] is not None:
            print(f"  canon_em: {c['canon_em']-d['canon_em']:+.4f}  <- the SQL-aware one")
        # per-complexity canonical delta
        cxs = sorted(set(d["by_cx"]) | set(c["by_cx"]))
        if cxs and cxs != ["all"]:
            print("\n  per-complexity canon_em (direct -> cot):")
            for cx in cxs:
                dd = d["by_cx"].get(cx, {}).get("canon", 0.0)
                cc = c["by_cx"].get(cx, {}).get("canon", 0.0)
                nn = d["by_cx"].get(cx, {}).get("n", c["by_cx"].get(cx, {}).get("n", 0))
                print(f"    {cx:<26} {dd:.3f} -> {cc:.3f}  (delta {cc-dd:+.3f}, n={nn})")


if __name__ == "__main__":
    main()
