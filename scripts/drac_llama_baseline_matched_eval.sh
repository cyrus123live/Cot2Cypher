#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=llama-be-m

# ==========================================================
# Re-evaluate the Llama-3.1-8B baseline adapter with the MATCHED prompt
# (no "think step by step") at inference, fixing the training/inference
# prompt mismatch in the first baseline eval (drac_llama_baseline_eval.sh).
#
# Sanity check: if Llama baseline numbers stay at ~0.39 EM / 0.29 exec EM
# (or climb), the cross-model finding (CoT hurts Llama) is robust.
# If they drop toward 0.30 / 0.21, the first result was inflated by
# the prompt mismatch in a weird way.
#
# Output: ~/scratch/results_llama_baseline_matched/predictions_cot_greedy.jsonl
# (filename prefix stays "predictions_cot_greedy" because drac_inference.py
#  determines the prefix; the directory disambiguates.)
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

echo "Copying Llama baseline adapter..."
cp -r ~/scratch/llama_baseline_adapter/final/ $SLURM_TMPDIR/adapter/
echo "Adapter copied."

echo "Starting evaluation of Llama baseline with MATCHED prompt (--no-cot-prompt)..."
mkdir -p ~/scratch/results_llama_baseline_matched
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir ~/scratch/results_llama_baseline_matched \
    --hf-cache $HF_CACHE \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --no-cot-prompt

echo "Done. Results in ~/scratch/results_llama_baseline_matched/"
