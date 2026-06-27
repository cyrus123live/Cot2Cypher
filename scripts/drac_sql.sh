#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=16:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=sql-cpp

# ==========================================================
# SQL matched direct-vs-CoT experiment (gretelai/synthetic_text_to_sql).
# The POSITIVE CONTROL for the CoT-as-compositional-prior hypothesis.
#
# SQL composes (joins/sub-queries) -> PREDICT: CoT HELPS (positive delta),
# the opposite of Cypher/SPARQL. If CoT does NOT help SQL in OUR pipeline,
# the theory is dead regardless of SPARQL.
#
# Trains BOTH arms, evals BOTH (with per-complexity EM breakdown), prints
# the CoT delta. Only the training target differs.
#
# PREREQUISITES (machine/login node with internet + API key):
#   ./venv/bin/python scripts/generate_cot_sql.py --prepare-test
#   COT_PROVIDER=groq COT_API_KEY=... ./venv/bin/python scripts/generate_cot_sql.py --limit 5000
#   scp data/sql/sql_cot_traces.jsonl data/sql/sql_test.jsonl cyrusp@fir.alliancecan.ca:~/scratch/
#
# Outputs:
#   ~/scratch/sql_direct_adapter/final/   ~/scratch/sql_cot_adapter/final/
#   ~/scratch/results_sql/metrics_sql_{direct,cot}.json
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

module load python/3.11
module load scipy-stack
module load gcc arrow

virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch torchvision
pip install --no-index pyarrow
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf
pip install --no-index nltk
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export BNB_CUDA_VERSION=129
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRACES=~/scratch/sql_cot_traces.jsonl
TEST=~/scratch/sql_test.jsonl
for f in "$TRACES" "$TEST"; do
    if [ ! -f "$f" ]; then
        echo "FATAL: $f not found. Generate + upload it first (see header)." >&2
        exit 1
    fi
done
echo "Traces: $(wc -l < $TRACES) | Test: $(wc -l < $TEST)"

SCRIPT=$PROJECT/thesis/scripts/drac_train_sql.py
mkdir -p ~/scratch/results_sql

echo "==================== TRAIN: direct ===================="
python $SCRIPT --variant direct --train-data $TRACES \
    --output-dir ~/scratch/sql_direct_adapter --hf-cache $HF_CACHE

echo "==================== TRAIN: cot ===================="
python $SCRIPT --variant cot --train-data $TRACES \
    --output-dir ~/scratch/sql_cot_adapter --hf-cache $HF_CACHE

echo "==================== EVAL: direct ===================="
python $SCRIPT --eval --variant direct --test-data $TEST \
    --adapter-path ~/scratch/sql_direct_adapter/final \
    --output-dir ~/scratch/results_sql --hf-cache $HF_CACHE

echo "==================== EVAL: cot ===================="
python $SCRIPT --eval --variant cot --test-data $TEST \
    --adapter-path ~/scratch/sql_cot_adapter/final \
    --output-dir ~/scratch/results_sql --hf-cache $HF_CACHE

echo ""
echo "==================== SQL CoT DELTA ===================="
python - <<'PY'
import json, os
d = json.load(open(os.path.expanduser("~/scratch/results_sql/metrics_sql_direct.json")))
c = json.load(open(os.path.expanduser("~/scratch/results_sql/metrics_sql_cot.json")))
print(f"direct : GLEU {d['gleu']:.4f}  EM {d['string_em']:.4f}")
print(f"cot    : GLEU {c['gleu']:.4f}  EM {c['string_em']:.4f}")
print(f"CoT effect (cot - direct): GLEU {c['gleu']-d['gleu']:+.4f}  EM {c['string_em']-d['string_em']:+.4f}")
print("PREDICTION: positive (CoT helps), unlike Cypher/SPARQL. Negative would kill the theory.")
print("\nPer-complexity EM (CoT should help MOST on the compositional strata):")
dc, cc = d.get("per_complexity", {}), c.get("per_complexity", {})
for cx in sorted(set(dc) | set(cc)):
    de = dc.get(cx, {}).get("em", 0.0); ce = cc.get(cx, {}).get("em", 0.0)
    n = dc.get(cx, {}).get("n", cc.get(cx, {}).get("n", 0))
    print(f"  {cx:<28} direct {de:.3f} -> cot {ce:.3f}  (delta {ce-de:+.3f}, n={n})")
PY
echo "Done. Metrics in ~/scratch/results_sql/"
