#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1       # 1x H100-80GB on Fir
#SBATCH --cpus-per-task=12           # 12 cores per GPU (Narval max)
#SBATCH --mem=48000M                 # 48 GB system RAM
#SBATCH --time=24:00:00              # 24 hours
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=cot-eval

# ==========================================================
# CoT Text2Cypher Evaluation on DRAC
#
# Usage:
#   sbatch scripts/drac_eval.sh                          # greedy eval
#   sbatch scripts/drac_eval.sh --self-consistency 5     # 5-sample majority vote
#
# Before first run, execute the setup script:
#   bash scripts/drac_setup.sh
# ==========================================================

# Parse arguments passed via sbatch
EXTRA_ARGS="${@}"

# PROJECT on DRAC = ~/projects/def-thomo/cyrusp
export PROJECT=~/projects/def-thomo/cyrusp
export HF_CACHE=~/scratch/hf_cache

# 1. Load modules (arrow MUST be loaded before virtualenv for pyarrow)
module load python/3.11
module load scipy-stack
module load gcc arrow

# 2. Create virtualenv on fast local SSD
virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# Install from DRAC wheels (pyarrow comes from arrow module loaded above)
pip install --no-index torch torchvision
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf
pip install --no-index nltk  # needed by evaluate's google_bleu

# Install pre-downloaded packages (run drac_setup.sh first)
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

# Fallback: if any package failed from wheelhouse, try from pre-downloaded wheels
pip install --no-index --find-links $HOME/wheels datasets evaluate pyarrow sentencepiece 2>/dev/null || true

# 3. HuggingFace cache on scratch (Fir has internet, no need for offline mode)
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
# Uncomment these for clusters WITHOUT internet (Narval, Rorqual):
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

# 4. Copy data to fast local storage
echo "Copying data to local SSD..."
cp $PROJECT/thesis/data/test_db_mapping.json $SLURM_TMPDIR/ 2>/dev/null || true
cp -r ~/scratch/adapter_weights/final/ $SLURM_TMPDIR/adapter/ 2>/dev/null || \
    cp -r $PROJECT/thesis/adapter_weights/final/ $SLURM_TMPDIR/adapter/ 2>/dev/null || true
echo "Data copied."

# 5. Run evaluation
echo "Starting evaluation..."
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir $SLURM_TMPDIR/results \
    --hf-cache $PROJECT/hf_cache \
    $EXTRA_ARGS

# 6. Copy results back to persistent storage
echo "Copying results to PROJECT..."
mkdir -p $PROJECT/thesis/results
cp -r $SLURM_TMPDIR/results/* $PROJECT/thesis/results/
echo "Done. Results in $PROJECT/thesis/results/"
