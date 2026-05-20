#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00              # 12 hrs (greedy on 3370 examples)
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=zog-eval-5ep

# ==========================================================
# Evaluate the 5-EPOCH ZOGRASCOPE-fine-tuned adapter.
#
# Companion to drac_train_zograscope_5ep.sh. Outputs to a dedicated
# directory so it does not clobber the 1-epoch results.
#
# Output: ~/scratch/results_finetuned_zog_5ep/predictions_zograscope_greedy.jsonl
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

# bitsandbytes <-> CUDA version pin: torch wheel now uses CUDA 13.2 but bnb
# only ships 12.2/12.6/12.9 binaries. Force the 12.9 binary (forward-compatible).
export BNB_CUDA_VERSION=129

echo "Copying 5-epoch adapter..."
cp -r ~/scratch/zograscope_adapter_5ep/final/ $SLURM_TMPDIR/adapter/
echo "Adapter copied."

echo "Copying ZOGRASCOPE test data..."
cp $PROJECT/thesis/data/zograscope/zograscope_formatted.jsonl $SLURM_TMPDIR/
echo "Data copied."

echo "Starting evaluation of 5-epoch model..."
mkdir -p ~/scratch/results_finetuned_zog_5ep
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir ~/scratch/results_finetuned_zog_5ep \
    --hf-cache $HF_CACHE \
    --zograscope $SLURM_TMPDIR/zograscope_formatted.jsonl

echo "Done. Results in ~/scratch/results_finetuned_zog_5ep/"
