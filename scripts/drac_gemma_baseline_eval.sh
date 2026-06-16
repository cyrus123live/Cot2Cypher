#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=gemma-be

# ==========================================================
# Evaluate the Gemma-2-9B *baseline* (no-CoT, our pipeline) adapter on the
# Neo4j test set, using the MATCHED direct-answer prompt (--no-cot-prompt).
#
# Companion to drac_train_gemma_baseline.sh. This produces the decisive
# number: our-pipeline Gemma direct-answer GLEU/EM, to compare against our
# Gemma+CoT (0.7682 / 0.3799). If the gap is ~0, the headline was the
# pipeline, not CoT.
#
# Output: ~/scratch/results_gemma_baseline/predictions_cot_greedy.jsonl
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

echo "Copying Gemma baseline adapter..."
cp -r ~/scratch/gemma_baseline_adapter/final/ $SLURM_TMPDIR/adapter/
echo "Adapter copied."

echo "Starting evaluation of Gemma baseline with MATCHED prompt (--no-cot-prompt)..."
mkdir -p ~/scratch/results_gemma_baseline
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir ~/scratch/results_gemma_baseline \
    --hf-cache $HF_CACHE \
    --base-model google/gemma-2-9b-it \
    --no-cot-prompt

echo "Done. Results in ~/scratch/results_gemma_baseline/"
