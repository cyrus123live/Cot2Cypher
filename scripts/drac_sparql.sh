#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=16:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=sparql-cpp

# ==========================================================
# SPARQL matched direct-vs-CoT experiment (LC-QuAD 2.0).
# First cross-formalism test of the CoT-as-compositional-prior hypothesis.
#
# SPARQL's basic graph pattern is a connected triple pattern (holistic, like
# Cypher) -> PREDICT: CoT HURTS, same as Cypher, unlike SQL.
#
# This one job trains BOTH arms and evals BOTH, then prints the CoT delta.
# Only the training target differs (both train on the same instances).
#
# PREREQUISITES (run on a machine/login node with internet + API key first):
#   ./venv/bin/python scripts/generate_cot_sparql.py --prepare-test
#   COT_PROVIDER=groq COT_API_KEY=... ./venv/bin/python scripts/generate_cot_sparql.py --limit 5000
#   scp data/sparql/sparql_cot_traces.jsonl data/sparql/sparql_test.jsonl \
#       cyrusp@fir.alliancecan.ca:~/scratch/
#
# Outputs:
#   ~/scratch/sparql_direct_adapter/final/   ~/scratch/sparql_cot_adapter/final/
#   ~/scratch/results_sparql/metrics_sparql_{direct,cot}.json
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

TRACES=~/scratch/sparql_cot_traces.jsonl
TEST=~/scratch/sparql_test.jsonl
for f in "$TRACES" "$TEST"; do
    if [ ! -f "$f" ]; then
        echo "FATAL: $f not found. Generate + upload it first (see header)." >&2
        exit 1
    fi
done
echo "Traces: $(wc -l < $TRACES) | Test: $(wc -l < $TEST)"

SCRIPT=$PROJECT/thesis/scripts/drac_train_sparql.py
mkdir -p ~/scratch/results_sparql

echo "==================== TRAIN: direct ===================="
python $SCRIPT --variant direct --train-data $TRACES \
    --output-dir ~/scratch/sparql_direct_adapter --hf-cache $HF_CACHE

echo "==================== TRAIN: cot ===================="
python $SCRIPT --variant cot --train-data $TRACES \
    --output-dir ~/scratch/sparql_cot_adapter --hf-cache $HF_CACHE

echo "==================== EVAL: direct ===================="
python $SCRIPT --eval --variant direct --test-data $TEST \
    --adapter-path ~/scratch/sparql_direct_adapter/final \
    --output-dir ~/scratch/results_sparql --hf-cache $HF_CACHE

echo "==================== EVAL: cot ===================="
python $SCRIPT --eval --variant cot --test-data $TEST \
    --adapter-path ~/scratch/sparql_cot_adapter/final \
    --output-dir ~/scratch/results_sparql --hf-cache $HF_CACHE

echo ""
echo "==================== SPARQL CoT DELTA ===================="
python - <<'PY'
import json, os
d = json.load(open(os.path.expanduser("~/scratch/results_sparql/metrics_sparql_direct.json")))
c = json.load(open(os.path.expanduser("~/scratch/results_sparql/metrics_sparql_cot.json")))
print(f"direct : GLEU {d['gleu']:.4f}  EM {d['string_em']:.4f}")
print(f"cot    : GLEU {c['gleu']:.4f}  EM {c['string_em']:.4f}")
print(f"CoT effect (cot - direct): GLEU {c['gleu']-d['gleu']:+.4f}  EM {c['string_em']-d['string_em']:+.4f}")
print("PREDICTION: negative (CoT hurts), like Cypher. Positive would challenge the theory.")
PY
echo "Done. Metrics in ~/scratch/results_sparql/"
