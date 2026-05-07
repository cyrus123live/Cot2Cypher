"""Angle 2 (1-hop paradox): does CoT's 1-hop improvement correlate with schema ambiguity?

Hypothesis: 1-hop queries with multiple relationship types between the same
(NodeA, NodeB) pair force the model to *read* the schema. CoT's InterCOL step
(linking sub-questions to specific relationship types) makes that explicit, so
gains should scale with ambiguity.

Ambiguity := |distinct relationship types in schema between the unordered pair
              of node labels used in the reference query|

Run:
    ./venv/bin/python scripts/analyze_one_hop.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "predictions_4bit_greedy.jsonl"
COT = ROOT / "results" / "predictions_cot_4bit_greedy.jsonl"
EXEC_BASELINE = ROOT / "results" / "exec_eval_4bit_greedy.jsonl"
EXEC_COT = ROOT / "results" / "exec_eval_cot_4bit_greedy.jsonl"

REL_LINE = re.compile(r"\(:([A-Za-z_][\w]*)\)-\[:([A-Za-z_][\w]*)\]->\(:([A-Za-z_][\w]*)\)")
# 1-hop reference: exactly ONE relationship marker `-[`, ONE MATCH clause,
# no variable-length `*`, no `OPTIONAL MATCH` chains beyond the first.
ONE_HOP_REL = re.compile(r"-\[(?::|[a-zA-Z_]\w*:)([A-Za-z_]\w*)\s*(?:\*[^\]]*)?\]")
NODE_LABEL = re.compile(r"\(\s*[a-zA-Z_]?\w*\s*:\s*([A-Za-z_]\w*)")


def parse_schema_relationships(schema: str) -> dict:
    """Return {frozenset({A,B}): set(rel_types)} for all schema-declared rels."""
    pairs = defaultdict(set)
    for m in REL_LINE.finditer(schema):
        a, rel, b = m.group(1), m.group(2), m.group(3)
        pairs[frozenset({a, b})].add(rel)
    return pairs


def is_one_hop(cypher: str) -> bool:
    """One MATCH clause, exactly one relationship marker, no variable-length, no UNION."""
    cy = cypher.strip()
    if cy.upper().count(" UNION ") > 0 or cy.upper().count("UNION ALL") > 0:
        return False
    matches = re.findall(r"\bMATCH\b", cy, flags=re.IGNORECASE)
    optional = re.findall(r"\bOPTIONAL\s+MATCH\b", cy, flags=re.IGNORECASE)
    if len(matches) - len(optional) != 1 or optional:
        return False
    rels = re.findall(r"-\[", cy)
    if len(rels) != 1:
        return False
    if "*" in re.findall(r"-\[[^\]]*\]", cy)[0]:
        return False
    return True


def extract_one_hop_signature(cypher: str):
    """Return (label_a, label_b, rel_type) from a 1-hop reference, or None."""
    rel = ONE_HOP_REL.search(cypher)
    if not rel:
        return None
    rel_type = rel.group(1)
    labels = NODE_LABEL.findall(cypher)
    if len(labels) < 2:
        return None
    return (labels[0], labels[1], rel_type)


def normalize(s: str) -> str:
    return " ".join((s or "").split()).strip().rstrip(";")


def load(p: Path) -> dict:
    by_id = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            by_id[r["instance_id"]] = r
    return by_id


REL_IN_PRED = re.compile(r"-\[(?::|[a-zA-Z_]\w*:)([A-Za-z_]\w*)\s*(?:\*[^\]]*)?\]")


def relationships_used(cypher: str) -> set:
    """Return set of relationship types referenced in a Cypher query."""
    return {m.group(1) for m in REL_IN_PRED.finditer(cypher or "")}


def load_exec(p: Path) -> dict:
    by_id = {}
    if not p.exists():
        return by_id
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            by_id[r["instance_id"]] = r
    return by_id


def main():
    base = load(BASELINE)
    cot = load(COT)
    base_exec = load_exec(EXEC_BASELINE)
    cot_exec = load_exec(EXEC_COT)
    common_ids = set(base) & set(cot)

    rows = []
    for iid in common_ids:
        b = base[iid]
        c = cot[iid]
        ref = b.get("reference_cypher", "")
        if not is_one_hop(ref):
            continue
        sig = extract_one_hop_signature(ref)
        if sig is None:
            continue
        la, lb, rel = sig
        schema_rels = parse_schema_relationships(b.get("schema", ""))
        if not schema_rels:
            continue
        pair_rels = schema_rels.get(frozenset({la, lb}), set())
        if rel not in pair_rels:
            # reference uses a relationship not declared between this pair —
            # likely a parsing artifact (e.g., abstract node label). Skip.
            continue
        ambiguity = len(pair_rels)
        ref_n = normalize(ref)
        b_pred = b.get("predicted_cypher", "")
        c_pred = c.get("predicted_cypher", "")
        # "Used correct relationship type" — direct test of schema grounding
        b_rels_used = relationships_used(b_pred)
        c_rels_used = relationships_used(c_pred)
        bx = base_exec.get(iid, {})
        cx = cot_exec.get(iid, {})
        rows.append({
            "id": iid,
            "ambiguity": ambiguity,
            "rel_type": rel,
            "labels": tuple(sorted([la, lb])),
            "baseline_em": int(normalize(b_pred) == ref_n),
            "cot_em": int(normalize(c_pred) == ref_n),
            "baseline_correct_rel": int(rel in b_rels_used),
            "cot_correct_rel": int(rel in c_rels_used),
            "baseline_exec": bx.get("exec_match"),
            "cot_exec": cx.get("exec_match"),
            "has_exec": iid in base_exec and iid in cot_exec,
            "data_source": b.get("data_source", ""),
        })

    print(f"\n=== Angle 2: 1-Hop Schema Ambiguity Analysis ===")
    print(f"Total 1-hop queries with parseable schema: {len(rows)}")

    # Bucket by ambiguity
    buckets = defaultdict(list)
    for r in rows:
        # Cap at 4+ to avoid sparse tail
        b = min(r["ambiguity"], 4)
        buckets[b].append(r)

    print(f"\nString EM by schema ambiguity:")
    print(f"{'ambiguity':>10} | {'n':>5} | {'base EM':>8} | {'CoT EM':>8} | {'Delta':>7}")
    print("-" * 60)
    for amb in sorted(buckets):
        rs = buckets[amb]
        b_em = mean(r["baseline_em"] for r in rs)
        c_em = mean(r["cot_em"] for r in rs)
        label = f"{amb}" if amb < 4 else "4+"
        print(f"{label:>10} | {len(rs):>5} | {b_em:>8.4f} | {c_em:>8.4f} | {c_em - b_em:>+7.4f}")

    print(f"\nCorrect relationship type used by ambiguity (direct schema-grounding test):")
    print(f"{'ambiguity':>10} | {'n':>5} | {'base':>8} | {'CoT':>8} | {'Delta':>7}")
    print("-" * 60)
    for amb in sorted(buckets):
        rs = buckets[amb]
        b_r = mean(r["baseline_correct_rel"] for r in rs)
        c_r = mean(r["cot_correct_rel"] for r in rs)
        label = f"{amb}" if amb < 4 else "4+"
        print(f"{label:>10} | {len(rs):>5} | {b_r:>8.4f} | {c_r:>8.4f} | {c_r - b_r:>+7.4f}")

    # Exec EM (only on instances that have DB access)
    exec_rows = [r for r in rows if r["has_exec"]
                 and r["baseline_exec"] is not None and r["cot_exec"] is not None]
    if exec_rows:
        exec_buckets = defaultdict(list)
        for r in exec_rows:
            exec_buckets[min(r["ambiguity"], 4)].append(r)
        print(f"\nExecution EM by ambiguity (DB-eligible 1-hop subset, n={len(exec_rows)}):")
        print(f"{'ambiguity':>10} | {'n':>5} | {'base':>8} | {'CoT':>8} | {'Delta':>7}")
        print("-" * 60)
        for amb in sorted(exec_buckets):
            rs = exec_buckets[amb]
            b_x = mean(int(r["baseline_exec"]) for r in rs)
            c_x = mean(int(r["cot_exec"]) for r in rs)
            label = f"{amb}" if amb < 4 else "4+"
            print(f"{label:>10} | {len(rs):>5} | {b_x:>8.4f} | {c_x:>8.4f} | {c_x - b_x:>+7.4f}")

    # Headline framing: ambiguity penalty
    unamb = [r for r in rows if r["ambiguity"] == 1]
    ambig = [r for r in rows if r["ambiguity"] >= 2]
    if unamb and ambig:
        b_unamb = mean(r["baseline_correct_rel"] for r in unamb)
        b_ambig = mean(r["baseline_correct_rel"] for r in ambig)
        c_unamb = mean(r["cot_correct_rel"] for r in unamb)
        c_ambig = mean(r["cot_correct_rel"] for r in ambig)
        print(f"\n=== Schema-grounding ambiguity penalty (correct rel type) ===")
        print(f"           {'unambig (=1)':>14} | {'ambig (≥2)':>14} | {'penalty':>8}")
        print(f"  Baseline {b_unamb:>14.4f} | {b_ambig:>14.4f} | {b_ambig - b_unamb:>+8.4f}")
        print(f"  CoT      {c_unamb:>14.4f} | {c_ambig:>14.4f} | {c_ambig - c_unamb:>+8.4f}")
        print(f"  ⇒ CoT collapses the schema-ambiguity penalty from "
              f"{(b_unamb - b_ambig)*100:.1f}pp to {(c_unamb - c_ambig)*100:.1f}pp")

    # Spearman correlation between ambiguity and CoT improvement (per-instance)
    n = len(rows)
    if n > 1:
        # Per-instance delta
        amb_arr = [r["ambiguity"] for r in rows]
        delta_arr = [r["cot_em"] - r["baseline_em"] for r in rows]
        # Pearson (continuous, simple)
        mx, my = mean(amb_arr), mean(delta_arr)
        num = sum((a - mx) * (d - my) for a, d in zip(amb_arr, delta_arr))
        denx = sum((a - mx) ** 2 for a in amb_arr) ** 0.5
        deny = sum((d - my) ** 2 for d in delta_arr) ** 0.5
        pearson = num / (denx * deny) if denx > 0 and deny > 0 else 0
        print(f"\nPearson(ambiguity, Δ-EM) = {pearson:+.4f}  over n={n} 1-hop queries")

        # Ambiguous (>=2) vs unambiguous (=1)
        ambig = [r for r in rows if r["ambiguity"] >= 2]
        unamb = [r for r in rows if r["ambiguity"] == 1]
        if ambig and unamb:
            amb_delta = mean(r["cot_em"] - r["baseline_em"] for r in ambig)
            unamb_delta = mean(r["cot_em"] - r["baseline_em"] for r in unamb)
            print(
                f"  Ambiguous (≥2 rels):   n={len(ambig):>4}  baseline={mean(r['baseline_em'] for r in ambig):.4f}  "
                f"cot={mean(r['cot_em'] for r in ambig):.4f}  Δ={amb_delta:+.4f}"
            )
            print(
                f"  Unambiguous (=1 rel):  n={len(unamb):>4}  baseline={mean(r['baseline_em'] for r in unamb):.4f}  "
                f"cot={mean(r['cot_em'] for r in unamb):.4f}  Δ={unamb_delta:+.4f}"
            )
            print(f"  Gap (ambig - unambig CoT gain): {amb_delta - unamb_delta:+.4f}")

    # Save raw for the paper
    out = ROOT / "results" / "one_hop_ambiguity.jsonl"
    with open(out, "w") as f:
        for r in rows:
            r["labels"] = list(r["labels"])
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote per-query data to {out}")


if __name__ == "__main__":
    main()
