#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=zog-be

# ==========================================================
# Evaluate the ZOGRASCOPE direct-answer baseline on IID/comp/length splits,
# using the matched DIRECT-ANSWER prompt (--no-cot-prompt).
#
# Compare against the CoT ZOGRASCOPE model (IID 64.71 / comp 53.08 / length 32.24).
# Output: ~/scratch/results_zograscope_baseline/predictions_zograscope_greedy.jsonl
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

cp -r ~/scratch/zograscope_baseline_adapter/final/ $SLURM_TMPDIR/adapter/
cp $PROJECT/thesis/data/zograscope/zograscope_formatted.jsonl $SLURM_TMPDIR/

echo "Evaluating ZOGRASCOPE baseline with matched direct-answer prompt..."
mkdir -p ~/scratch/results_zograscope_baseline
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir ~/scratch/results_zograscope_baseline \
    --hf-cache $HF_CACHE \
    --base-model google/gemma-2-9b-it \
    --zograscope $SLURM_TMPDIR/zograscope_formatted.jsonl \
    --no-cot-prompt

echo "Done. Results in ~/scratch/results_zograscope_baseline/"
