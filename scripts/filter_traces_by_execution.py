"""STaR step 2: execution-filter forward-generated CoT candidates.

For each training example, execute the reference Cypher and each forward
candidate against the demo Neo4j databases. Keep ONE verified trace per
example: the first candidate whose execution result matches the reference's
result. These verified-correct (reasoning, cypher) pairs become the training
set — the ingredient the successful Text-to-SQL CoT papers (STaR-SQL) used and
our original post-hoc rationalization pipeline lacked.

Output records are in the same shape train_cot expects: instance_id, question,
schema, cypher (the verified candidate's query), reasoning, data_source.

Run from project root (uses demo.neo4jlabs.com; the hardened execute_cypher
with per-query hard timeout is reused):
    ./venv/bin/python scripts/filter_traces_by_execution.py data/forward_traces.jsonl
"""

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_execution import ALIAS_TO_DB, execute_cypher, get_driver


def main():
    if len(sys.argv) < 2:
        print("Usage: python filter_traces_by_execution.py <forward_traces.jsonl> [output.jsonl]")
        sys.exit(1)
    src = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else src.replace(
        ".jsonl", "_filtered.jsonl"
    )

    records = [json.loads(l) for l in open(src)]
    # Map alias -> demo db name; skip examples whose alias we can't reach
    by_db = defaultdict(list)
    skipped_alias = 0
    for r in records:
        alias = r.get("database_reference_alias", "")
        if alias in ALIAS_TO_DB:
            by_db[ALIAS_TO_DB[alias]].append(r)
        else:
            skipped_alias += 1
    print(f"Forward records: {len(records)} | mapped to {len(by_db)} DBs | "
          f"skipped (unmapped alias): {skipped_alias}")

    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} already filtered")

    kept = 0
    examined = 0
    no_candidate_correct = 0
    ref_failed = 0
    start = time.time()

    out = open(out_path, "a")
    log = open(out_path.replace(".jsonl", "_audit.jsonl"), "a")

    for db_name, recs in sorted(by_db.items(), key=lambda x: -len(x[1])):
        pending = [r for r in recs if r["instance_id"] not in done_ids]
        if not pending:
            continue
        print(f"\n{db_name}: {len(pending)} pending", flush=True)
        try:
            driver = get_driver(db_name)
            driver.execute_query("RETURN 1 AS t", database_=db_name, timeout=10)
        except Exception as e:
            print(f"  connect failed: {str(e)[:80]} — skipping", flush=True)
            continue

        for i, r in enumerate(pending):
            examined += 1
            # values_only=True: ignore column aliases so a forward candidate
            # that returns the right VALUES under different names still matches.
            ref_result, ref_err = execute_cypher(driver, db_name, r["cypher"], values_only=True)
            if ref_result is None:
                ref_failed += 1
                log.write(json.dumps({"instance_id": r["instance_id"],
                                      "status": "ref_failed", "ref_err": ref_err}) + "\n")
                continue

            chosen = None
            for c in r.get("forward_candidates", []):
                cy = c.get("cypher", "")
                if not cy:
                    continue
                res, _ = execute_cypher(driver, db_name, cy, values_only=True)
                if res is not None and res == ref_result:
                    chosen = c
                    break

            if chosen is None:
                no_candidate_correct += 1
                log.write(json.dumps({"instance_id": r["instance_id"],
                                      "status": "no_correct_candidate"}) + "\n")
                continue

            # Keep the verified trace. Train on the VERIFIED candidate's own cypher
            # (which executes to the reference result) + its reasoning.
            out.write(json.dumps({
                "instance_id": r["instance_id"],
                "question": r["question"],
                "schema": r["schema"],
                "cypher": chosen["cypher"],
                "reasoning": chosen["reasoning"],
                "data_source": r.get("data_source", ""),
                "database_reference_alias": r.get("database_reference_alias", ""),
            }) + "\n")
            kept += 1

            if (i + 1) % 50 == 0:
                out.flush(); log.flush()
                el = time.time() - start
                print(f"  [{examined}] kept={kept} no_correct={no_candidate_correct} "
                      f"ref_failed={ref_failed} ({el/60:.1f}m)", flush=True)
        driver.close()

    out.close(); log.close()
    print("\n" + "=" * 60)
    print("EXECUTION-FILTER SUMMARY")
    print("=" * 60)
    print(f"Examined:                 {examined}")
    print(f"Kept (verified traces):   {kept}")
    print(f"No correct candidate:     {no_candidate_correct}")
    print(f"Reference query failed:    {ref_failed}")
    if examined:
        print(f"Keep rate (of ref-valid): "
              f"{kept}/{examined - ref_failed} = "
              f"{kept/max(1, examined - ref_failed)*100:.1f}%")
    print(f"\nFiltered traces -> {out_path}")


if __name__ == "__main__":
    main()
