#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=spider-sc5

# ==========================================================
# E-B ON SPIDER — CANDIDATE DIVERSITY FOR SELECTION (strong-recipe arms).
#
# STaR-SQL's +18pp came from best-of-16 with a trained verifier; CoT-SFT alone
# gave +6.4. So CoT's real value may be a better candidate POOL even when its
# greedy answer is worse (ours: CoT 0.6683 vs direct 0.7669). This samples 5
# candidates (T=0.7, top_p=0.95) from each EXISTING strong-recipe Spider adapter
# and scores string-vote / execution-MBR / oracle, with paired-bootstrap CIs
# on the CoT-minus-direct deltas. Inference only -- no training.
#
# PRE-REGISTERED PREDICTION: direct's oracle and MBR stay at or above CoT's
# (the negative result extends to selection). If CoT's oracle/MBR is clearly
# higher, CoT keeps a defensible role as a candidate generator for best-of-N.
#
# Prereqs: ~/scratch/spider_{direct,cot}_adapter/final (from drac_spider.sh),
#          ~/scratch/spider_test.jsonl, ~/scratch/spider_database/
# Output: ~/scratch/results_spider_sampling/ (+ selection table at end of .out)
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

module load python/3.11
module load scipy-stack
module load gcc arrow

set -eo pipefail

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

TEST=~/scratch/spider_test.jsonl
DB=~/scratch/spider_database
[ -f "$TEST" ] || { echo "FATAL: $TEST not found." >&2; exit 1; }
[ -d "$DB" ] || { echo "FATAL: $DB not found." >&2; exit 1; }
for v in direct cot; do
    [ -f ~/scratch/spider_${v}_adapter/final/adapter_config.json ] || {
        echo "FATAL: ~/scratch/spider_${v}_adapter/final missing (rerun drac_spider.sh?)" >&2; exit 1; }
done

TRAIN=$PROJECT/thesis/scripts/drac_train_sql.py
RES=~/scratch/results_spider_sampling
mkdir -p $RES

for v in direct cot; do
    echo "==================== SAMPLE x5: $v ===================="
    python $TRAIN --eval --variant $v --tag spider --test-data $TEST \
        --adapter-path ~/scratch/spider_${v}_adapter/final --output-dir $RES \
        --hf-cache $HF_CACHE --num-samples 5 --temperature 0.7 --batch-size 4
done

echo ""
echo "==================== SELECTION (string-vote / MBR / oracle) ===================="
python $PROJECT/thesis/scripts/eval_spider_selection.py --db-dir $DB \
    $RES/predictions_spider_direct_sc5.jsonl $RES/predictions_spider_cot_sc5.jsonl
echo ""
echo "Greedy reference (strong recipe): direct 0.7669 / CoT 0.6683"
echo "Done. Results in $RES/"
