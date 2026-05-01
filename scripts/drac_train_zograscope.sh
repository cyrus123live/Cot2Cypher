#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=24:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=zog-train

# ==========================================================
# Fine-tune Gemma-2-9B-it on ZOGRASCOPE CoT traces
#
# Prerequisites:
#   - ~/scratch/zograscope_cot_traces.jsonl (generated locally with Cerebras)
#   - HuggingFace token cached at ~/.cache/huggingface/token
#
# Usage:
#   sbatch scripts/drac_train_zograscope.sh
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

# 1. Load modules
module load python/3.11
module load scipy-stack
module load gcc arrow

# 2. Create virtualenv
virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# Install packages
pip install --no-index torch torchvision
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

# 3. HF settings (Fir has internet)
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

# 4. Copy training data to local SSD
echo "Copying training data..."
cp ~/scratch/zograscope_cot_traces.jsonl $SLURM_TMPDIR/
echo "Data copied."

# 5. Run training
echo "Starting ZOGRASCOPE CoT fine-tuning..."
mkdir -p ~/scratch/zograscope_adapter
python $PROJECT/thesis/scripts/drac_train_zograscope.py \
    --train-data $SLURM_TMPDIR/zograscope_cot_traces.jsonl \
    --output-dir ~/scratch/zograscope_adapter \
    --hf-cache $HF_CACHE

echo "Done. Adapter saved to ~/scratch/zograscope_adapter/"
