#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=04:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=zog-base

# ==========================================================
# THE ZOGRASCOPE DISTRIBUTION-SHIFT CONTROL.
#
# Trains a Gemma-2-9B DIRECT-ANSWER baseline on the ZOGRASCOPE training set
# (no CoT) with our pipeline. Pairs with the existing CoT ZOGRASCOPE model
# (32.24% length-split exec acc) to answer the decisive question:
#   does CoT actually help under length/compositional distribution shift,
#   or was the ZOGRASCOPE "SOTA" also a pipeline artifact (like in-distribution)?
#
# Reuses drac_train_gemma_baseline.py (trains on (question,schema)->cypher,
# ignoring the reasoning field) — just points it at the ZOGRASCOPE traces.
#
# Prereq: ~/scratch/zograscope_cot_traces.jsonl (already on Fir from the
# CoT ZOGRASCOPE run).
#
# Output: ~/scratch/zograscope_baseline_adapter/final/
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
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export BNB_CUDA_VERSION=129
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f ~/scratch/zograscope_cot_traces.jsonl ]; then
    echo "FATAL: ~/scratch/zograscope_cot_traces.jsonl not found." >&2
    exit 1
fi
cp ~/scratch/zograscope_cot_traces.jsonl $SLURM_TMPDIR/
echo "Data copied: $(wc -l < $SLURM_TMPDIR/zograscope_cot_traces.jsonl) records"

echo "Training ZOGRASCOPE direct-answer baseline (no CoT)..."
mkdir -p ~/scratch/zograscope_baseline_adapter
python $PROJECT/thesis/scripts/drac_train_gemma_baseline.py \
    --train-data $SLURM_TMPDIR/zograscope_cot_traces.jsonl \
    --output-dir ~/scratch/zograscope_baseline_adapter \
    --hf-cache $HF_CACHE

echo "Done. Adapter at ~/scratch/zograscope_baseline_adapter/"
