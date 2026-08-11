#!/usr/bin/env python3
"""Paired bootstrap 95% CIs for the paper's main CoT-vs-direct deltas.

For each comparison we build per-instance PAIRED data (joined on instance id),
resample instances with replacement (B=10,000), recompute the delta on each
resample, and report the 2.5/97.5 percentile interval.

Metrics:
  - String EM: whitespace-collapsed exact match (matches train_cot.py).
  - GLEU: corpus-level Google-BLEU. Per-instance sufficient statistics
    (match_i, all_i) with all_i = max(#hyp ngrams, #ref ngrams), n=1..4,
    under Tokenizer13a (mteval-v13a), so corpus GLEU = sum(m)/sum(a) and each
    bootstrap resample is two vector sums. Validated against the pipeline's
    reported HF-evaluate numbers before use (script asserts within 1e-3).
  - Exec EM: exec_match booleans from the exec_eval jsonl files.
  - SQL canonical-AST EM: sqlglot-normalized match (matches score_sql_semantic.py).

Run from project root: ./venv/bin/python scripts/bootstrap_ci.py
Writes results/bootstrap_cis.json and prints a table.
"""

import json
import re
import sys
from collections import Counter

import numpy as np

B = 10_000
SEED = 42


# ---------------- Tokenizer13a (mteval-v13a, as used by HF google_bleu) ------

def tokenize_13a(line: str) -> list:
    norm = line
    norm = norm.replace("<skipped>", "")
    norm = norm.replace("-\n", "")
    norm = norm.replace("\n", " ")
    norm = norm.replace("&quot;", '"')
    norm = norm.replace("&amp;", "&")
    norm = norm.replace("&lt;", "<")
    norm = norm.replace("&gt;", ">")
    norm = f" {norm} "
    norm = re.sub(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 ", norm)
    norm = re.sub(r"([^0-9])([\.,])", r"\1 \2 ", norm)
    norm = re.sub(r"([\.,])([^0-9])", r" \1 \2", norm)
    norm = re.sub(r"([0-9])(-)", r"\1 \2 ", norm)
    return norm.split()


def gleu_stats(pred: str, ref: str, max_n: int = 4):
    """Per-instance sufficient statistics for corpus GLEU (single reference)."""
    p, r = tokenize_13a(pred), tokenize_13a(ref)

    def ngrams(toks):
        c = Counter()
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1):
                c[tuple(toks[i:i + n])] += 1
        return c

    cp, cr = ngrams(p), ngrams(r)
    match = sum((cp & cr).values())
    return match, max(sum(cp.values()), sum(cr.values()))


def norm_ws(s: str) -> str:
    return " ".join((s or "").split())


# ---------------- data loaders ----------------------------------------------

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def paired_translation(direct_path, cot_path):
    """Return arrays (m_d, a_d, m_c, a_c, em_d, em_c) joined on instance_id."""
    d = {r["instance_id"]: r for r in load_jsonl(direct_path)}
    c = {r["instance_id"]: r for r in load_jsonl(cot_path)}
    ids = sorted(set(d) & set(c))
    assert len(ids) == len(d) == len(c), f"join mismatch: {len(d)} vs {len(c)} vs {len(ids)}"
    md, ad, mc, ac, ed, ec = [], [], [], [], [], []
    for i in ids:
        pd_, rd = d[i]["predicted_cypher"], d[i]["reference_cypher"]
        pc, rc = c[i]["predicted_cypher"], c[i]["reference_cypher"]
        m, a = gleu_stats(pd_, rd)
        md.append(m); ad.append(a)
        m, a = gleu_stats(pc, rc)
        mc.append(m); ac.append(a)
        ed.append(norm_ws(pd_) == norm_ws(rd))
        ec.append(norm_ws(pc) == norm_ws(rc))
    return tuple(np.array(x) for x in (md, ad, mc, ac, ed, ec))


def paired_exec(direct_path, cot_path, key="instance_id"):
    d = {r[key]: r["exec_match"] for r in load_jsonl(direct_path)}
    c = {r[key]: r["exec_match"] for r in load_jsonl(cot_path)}
    ids = sorted(set(d) & set(c))
    xd = np.array([bool(d[i]) if isinstance(d[i], bool) else d[i] == "True" or d[i] is True for i in ids])
    xc = np.array([bool(c[i]) if isinstance(c[i], bool) else c[i] == "True" or c[i] is True for i in ids])
    # exec files store real booleans in json; handle both
    xd = np.array([d[i] if isinstance(d[i], bool) else str(d[i]) == "True" for i in ids])
    xc = np.array([c[i] if isinstance(c[i], bool) else str(c[i]) == "True" for i in ids])
    return xd, xc, ids


# ---------------- bootstrap -------------------------------------------------

def boot_binary(x_direct, x_cot, rng):
    """CI for mean(cot) - mean(direct)."""
    n = len(x_direct)
    idx = rng.integers(0, n, size=(B, n))
    deltas = x_cot[idx].mean(axis=1) - x_direct[idx].mean(axis=1)
    point = x_cot.mean() - x_direct.mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, lo, hi


def boot_gleu(md, ad, mc, ac, rng):
    n = len(md)
    idx = rng.integers(0, n, size=(B, n))
    g_d = md[idx].sum(axis=1) / ad[idx].sum(axis=1)
    g_c = mc[idx].sum(axis=1) / ac[idx].sum(axis=1)
    deltas = g_c - g_d
    point = mc.sum() / ac.sum() - md.sum() / ad.sum()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, lo, hi


# ---------------- SQL canonical EM ------------------------------------------

def sql_canon_pairs(direct_path, cot_path):
    import sqlglot

    def canon(s):
        if not (s or "").strip():
            return None
        try:
            t = sqlglot.parse_one(s, read="sqlite")
            return t.sql(normalize=True, normalize_functions="upper",
                         comments=False, pretty=False).lower()
        except Exception:
            return None

    d = {r["instance_id"]: r for r in load_jsonl(direct_path)}
    c = {r["instance_id"]: r for r in load_jsonl(cot_path)}
    ids = sorted(set(d) & set(c))
    xd, xc = [], []
    for i in ids:
        for rec, out in ((d[i], xd), (c[i], xc)):
            cp = canon(rec["predicted_sql"])
            cg = canon(rec["reference_sql"])
            out.append(cp is not None and cg is not None and cp == cg)
    return np.array(xd), np.array(xc)


# ---------------- main ------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)
    out = {}

    print("Building paired data (GLEU stats take a minute)...", file=sys.stderr)

    # --- Neo4j translation metrics, Gemma and Llama
    # NOTE: the matched-prompt Llama baseline predictions (A6: GLEU 0.7680) live on
    # DRAC at ~/scratch/results_llama_baseline_matched/ and have not been scp'd back;
    # the local predictions_llama_baseline_greedy.jsonl is the MISMATCHED-prompt A7
    # run (GLEU 0.7024). Fetch the matched file to the path below to enable that CI.
    import os
    comparisons = [
        ("gemma", "results/predictions_gemma_baseline_greedy.jsonl",
         "results/predictions_cot_4bit_greedy.jsonl", 0.7854, 0.7682),
    ]
    llama_matched = "results/predictions_llama_baseline_matched_greedy.jsonl"
    if os.path.exists(llama_matched):
        comparisons.append(("llama", llama_matched,
                            "results/predictions_cot_llama_greedy.jsonl", 0.7680, 0.7416))
    else:
        print(f"SKIP llama translation CIs: {llama_matched} not found "
              "(matched A6 predictions still on DRAC)", file=sys.stderr)
    for fam, dpath, cpath, want_d, want_c in comparisons:
        md, ad, mc, ac, ed, ec = paired_translation(dpath, cpath)
        g_d, g_c = md.sum() / ad.sum(), mc.sum() / ac.sum()
        assert abs(g_d - want_d) < 1e-3, f"{fam} direct GLEU {g_d:.4f} != reported {want_d}"
        assert abs(g_c - want_c) < 1e-3, f"{fam} cot GLEU {g_c:.4f} != reported {want_c}"
        p, lo, hi = boot_gleu(md, ad, mc, ac, rng)
        out[f"{fam}_gleu"] = {"delta": p, "lo": lo, "hi": hi, "n": len(md)}
        p, lo, hi = boot_binary(ed, ec, rng)
        out[f"{fam}_string_em"] = {"delta": p, "lo": lo, "hi": hi, "n": len(ed)}

    # --- Neo4j exec EM
    for fam, dpath, cpath in [
        ("gemma", "results/exec_eval_gemma_baseline_greedy.jsonl",
         "results/exec_eval_cot_4bit_greedy.jsonl"),
        ("llama", "results/exec_eval_llama_baseline_greedy.jsonl",
         "results/exec_eval_cot_llama_greedy.jsonl"),
    ]:
        xd, xc, ids = paired_exec(dpath, cpath)
        p, lo, hi = boot_binary(xd, xc, rng)
        out[f"{fam}_exec_em"] = {"delta": p, "lo": lo, "hi": hi, "n": len(ids)}

    # --- ZOGRASCOPE clean splits (exec)
    iid_ids = set(open("data/zograscope/ids_iid_test.txt").read().split())
    comp_ids = set(open("data/zograscope/ids_compositional_test.txt").read().split())
    xd, xc, ids = paired_exec("results/pred_zog_length_baseline_exec.jsonl",
                              "results/pred_zog_length_cot_exec.jsonl", key="id")
    p, lo, hi = boot_binary(xd, xc, rng)
    out["zog_length_exec"] = {"delta": p, "lo": lo, "hi": hi, "n": len(ids)}
    xd, xc, ids = paired_exec("results/pred_zog_regular_baseline_exec.jsonl",
                              "results/pred_zog_regular_cot_exec.jsonl", key="id")
    ids = np.array(ids)
    for name, idset in (("zog_iid_exec", iid_ids), ("zog_comp_exec", comp_ids)):
        mask = np.array([i in idset for i in ids])
        p, lo, hi = boot_binary(xd[mask], xc[mask], rng)
        out[name] = {"delta": p, "lo": lo, "hi": hi, "n": int(mask.sum())}

    # --- SQL canonical-AST EM
    xd, xc = sql_canon_pairs("results/predictions_sql_direct.jsonl",
                             "results/predictions_sql_cot.jsonl")
    p, lo, hi = boot_binary(xd, xc, rng)
    out["sql_canon_em"] = {"delta": p, "lo": lo, "hi": hi, "n": len(xd)}

    with open("results/bootstrap_cis.json", "w") as f:
        json.dump({k: {kk: (float(vv) if not isinstance(vv, int) else vv)
                       for kk, vv in v.items()} for k, v in out.items()}, f, indent=2)

    print(f"\nPaired bootstrap 95% CIs (B={B}, seed={SEED}); delta = CoT - direct")
    print(f"{'comparison':<20}{'delta':>10}{'95% CI':>22}{'n':>7}{'  sig?':>7}")
    print("-" * 66)
    for k, v in out.items():
        sig = "yes" if (v["lo"] > 0 or v["hi"] < 0) else "NO"
        print(f"{k:<20}{v['delta']:>+10.4f}   [{v['lo']:+.4f}, {v['hi']:+.4f}]{v['n']:>7}{sig:>7}")


if __name__ == "__main__":
    main()
