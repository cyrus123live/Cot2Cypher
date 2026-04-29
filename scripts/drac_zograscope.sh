#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1       # 1x H100-80GB on Fir
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=24:00:00              # 24 hours (3370 examples, greedy only)
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=zograscope

# ==========================================================
# ZOGRASCOPE benchmark evaluation on DRAC
#
# Usage:
#   sbatch scripts/drac_zograscope.sh
# ==========================================================

export PROJECT=~/projects/def-thomo/cyrusp
export HF_CACHE=~/scratch/hf_cache

# 1. Load modules (arrow before virtualenv for pyarrow)
module load python/3.11
module load scipy-stack
module load gcc arrow

# 2. Create virtualenv
virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# Install packages
pip install --no-index torch torchvision
pip install --no-index pyarrow
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf
pip install --no-index nltk
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes
pip install --no-index --find-links $HOME/wheels datasets evaluate pyarrow sentencepiece 2>/dev/null || true

# 3. HuggingFace settings (Fir has internet)
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

# 4. Copy adapter to local SSD
echo "Copying adapter to local SSD..."
cp -r ~/scratch/adapter_weights/final/ $SLURM_TMPDIR/adapter/
echo "Adapter copied."

# 5. Copy ZOGRASCOPE data to local SSD
echo "Copying ZOGRASCOPE data..."
cp $PROJECT/thesis/data/zograscope/zograscope_formatted.jsonl $SLURM_TMPDIR/
echo "Data copied."

# 6. Run inference (output directly to scratch)
echo "Starting ZOGRASCOPE evaluation..."
mkdir -p ~/scratch/results
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir ~/scratch/results \
    --hf-cache $HF_CACHE \
    --zograscope $SLURM_TMPDIR/zograscope_formatted.jsonl

echo "Done. Results in ~/scratch/results/"
