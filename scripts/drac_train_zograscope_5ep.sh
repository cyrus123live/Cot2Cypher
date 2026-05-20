#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=04:00:00              # 4 hrs (1 epoch was 20 min; 5 epochs ~2 hrs + setup)
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=zog-train-5ep

# ==========================================================
# Fine-tune Gemma-2-9B-it on ZOGRASCOPE CoT traces — 5 epochs
#
# Tests whether IID accuracy climbs while length advantage holds —
# defuses the reviewer concern that 1-epoch is just undertraining.
#
# Output: ~/scratch/zograscope_adapter_5ep/final/
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
export BNB_CUDA_VERSION=129  # bnb ships 12.2/12.6/12.9; force 12.9 if torch is on 13.x

echo "Copying training data..."
cp ~/scratch/zograscope_cot_traces.jsonl $SLURM_TMPDIR/
echo "Data copied."

echo "Starting ZOGRASCOPE CoT fine-tuning (5 epochs)..."
mkdir -p ~/scratch/zograscope_adapter_5ep
python $PROJECT/thesis/scripts/drac_train_zograscope.py \
    --train-data $SLURM_TMPDIR/zograscope_cot_traces.jsonl \
    --output-dir ~/scratch/zograscope_adapter_5ep \
    --hf-cache $HF_CACHE \
    --num-epochs 5

echo "Done. Adapter saved to ~/scratch/zograscope_adapter_5ep/"
