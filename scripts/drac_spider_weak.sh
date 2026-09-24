#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=16:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=spider-weak

# ==========================================================
# E-A, SQL CELL — WEAK-RECIPE SPIDER: the causal baseline-strength test.
#
# Identical to drac_spider.sh (strong recipe: direct 0.7669 / CoT 0.6683 exec
# acc, CoT -0.0986) EXCEPT --full-sequence on both arms: loss on every token,
# prompt included -- the literature-typical recipe. Same traces, same instances,
# same QLoRA config; only the loss mask differs from the strong run.
#
# PRE-REGISTERED PREDICTION (notes/PROPOSED_EXPERIMENTS.md): the weak recipe
# lowers the direct arm below 0.7669 and CoT FLIPS POSITIVE on execution
# accuracy -- reproducing the literature's CoT-helps-SQL result in-pipeline.
# FIRST CHECK: if weak direct stays near 0.7669, the weak recipe opened no
# headroom and the test is uninformative.
#
# Prereqs (already on Fir from drac_spider.sh): ~/scratch/spider_cot_traces.jsonl,
# ~/scratch/spider_test.jsonl, ~/scratch/spider_database/
# Output: ~/scratch/results_spider_weak/ (+ execution delta at the end of the .out)
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

module load python/3.11
module load scipy-stack
module load gcc arrow

set -eo pipefail  # fail LOUDLY -- a silent venv/CVMFS failure previously printed "Done"

for attempt in 1 2 3; do
    if virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env; then break; fi
    echo "virtualenv attempt $attempt failed; retrying in 30s..." >&2
    rm -rf $SLURM_TMPDIR/env
    sleep 30
    if [ "$attempt" = 3 ]; then echo "FATAL: virtualenv failed 3x (CVMFS?)" >&2; exit 1; fi
done
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision pyarrow
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf nltk
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export BNB_CUDA_VERSION=129
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRACES=~/scratch/spider_cot_traces.jsonl
TEST=~/scratch/spider_test.jsonl
DB=~/scratch/spider_database
for f in "$TRACES" "$TEST"; do
    [ -f "$f" ] || { echo "FATAL: $f not found (upload as for drac_spider.sh)." >&2; exit 1; }
done
[ -d "$DB" ] || { echo "FATAL: $DB not found (upload the Spider database/ folder)." >&2; exit 1; }
echo "Traces: $(wc -l < $TRACES) | Test: $(wc -l < $TEST)"

TRAIN=$PROJECT/thesis/scripts/drac_train_sql.py
RES=~/scratch/results_spider_weak
mkdir -p $RES

echo "==================== TRAIN (weak recipe): direct ===================="
python $TRAIN --variant direct --tag spider --train-data $TRACES --full-sequence \
    --output-dir ~/scratch/spider_weak_direct_adapter --hf-cache $HF_CACHE
echo "==================== TRAIN (weak recipe): cot ===================="
python $TRAIN --variant cot --tag spider --train-data $TRACES --full-sequence \
    --output-dir ~/scratch/spider_weak_cot_adapter --hf-cache $HF_CACHE

echo "==================== EVAL: weak direct ===================="
python $TRAIN --eval --variant direct --tag spider --test-data $TEST \
    --adapter-path ~/scratch/spider_weak_direct_adapter/final --output-dir $RES --hf-cache $HF_CACHE
echo "==================== EVAL: weak cot ===================="
python $TRAIN --eval --variant cot --tag spider --test-data $TEST \
    --adapter-path ~/scratch/spider_weak_cot_adapter/final --output-dir $RES --hf-cache $HF_CACHE

echo ""
echo "==================== EXECUTION ACCURACY (weak recipe) ===================="
python $PROJECT/thesis/scripts/eval_spider_execution.py --db-dir $DB \
    $RES/predictions_spider_direct.jsonl $RES/predictions_spider_cot.jsonl
echo ""
echo "Strong-recipe reference (drac_spider.sh): direct 0.7669 / CoT 0.6683 / CoT effect -0.0986"
echo "Done. Results in $RES/"
