#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=06:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=star-train

# ==========================================================
# STaR retrain — clean reasoning-vs-direct comparison on the SAME
# execution-verified instances (the STaR-SQL design).
#
# Both arms train on forward_traces_filtered.jsonl (3,938 verified traces).
# Only difference: reasoning prefix.
#   MODE=cot      -> drac_train_zograscope.py   (reasoning + verified cypher)
#   MODE=baseline -> drac_train_gemma_baseline.py (verified cypher only)
#
#   sbatch --export=ALL,MODE=cot      scripts/drac_train_star.sh
#   sbatch --export=ALL,MODE=baseline scripts/drac_train_star.sh
#
# Prereq on Fir: ~/scratch/forward_traces_filtered.jsonl  (scp from local)
# Output: ~/scratch/star_${MODE}_adapter/final/
# ==========================================================

set -e
: "${MODE:?set MODE=cot|baseline}"

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

OUT_DIR=~/scratch/star_${MODE}_adapter
if [ "$MODE" = "cot" ]; then PY=drac_train_zograscope.py; else PY=drac_train_gemma_baseline.py; fi

module load python/3.11; module load scipy-stack; module load gcc arrow
virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export BNB_CUDA_VERSION=129
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f ~/scratch/forward_traces_filtered.jsonl ]; then
    echo "FATAL: ~/scratch/forward_traces_filtered.jsonl not found. scp from local." >&2
    exit 1
fi
cp ~/scratch/forward_traces_filtered.jsonl $SLURM_TMPDIR/train.jsonl
echo "[star/$MODE] training on $(wc -l < $SLURM_TMPDIR/train.jsonl) verified traces via $PY"
mkdir -p "$OUT_DIR"
python $PROJECT/thesis/scripts/$PY \
    --train-data $SLURM_TMPDIR/train.jsonl \
    --output-dir "$OUT_DIR" \
    --hf-cache $HF_CACHE

echo "Done. Adapter at $OUT_DIR/final/"
