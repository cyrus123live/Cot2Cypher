#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=gemma-pke

# ==========================================================
# Evaluate the Gemma-2-9B PACKING-ablation adapter on the Neo4j test set,
# MATCHED direct-answer prompt (--no-cot-prompt) — identical inference to the
# A5 baseline eval and the 1c full-sequence eval.
#
# Decisive ladder (only the training recipe differs):
#   A5 completion-only            : GLEU 0.7854 / EM 0.4331
#   1c full-sequence (no packing) : GLEU 0.7415
#   packing (Neo4j full recipe)   : this run
#   A2 published adapter          : GLEU 0.6455
#
# Output: ~/scratch/results_gemma_packing/predictions_cot_greedy.jsonl
# (prefix is fixed by drac_inference.py; the directory disambiguates)
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

module load python/3.11
module load scipy-stack
module load gcc arrow

set -eo pipefail  # fail LOUDLY -- a silent venv/CVMFS failure previously printed "Done"

# CVMFS can throw transient Errno-5 I/O errors seeding the venv (seen on fc10620);
# retry a few times before giving up.
for attempt in 1 2 3; do
    if virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env; then break; fi
    echo "virtualenv attempt $attempt failed; retrying in 30s..." >&2
    rm -rf $SLURM_TMPDIR/env
    sleep 30
    if [ "$attempt" = 3 ]; then echo "FATAL: virtualenv failed 3x (CVMFS?)" >&2; exit 1; fi
done
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

echo "Copying Gemma packing-ablation adapter..."
if [ ! -d ~/scratch/gemma_packing_adapter/final ]; then
    echo "FATAL: ~/scratch/gemma_packing_adapter/final not found — run drac_train_gemma_packing.sh first." >&2
    exit 1
fi
cp -r ~/scratch/gemma_packing_adapter/final/ $SLURM_TMPDIR/adapter/
echo "Adapter copied."

echo "Starting evaluation of Gemma packing ablation with MATCHED prompt (--no-cot-prompt)..."
mkdir -p ~/scratch/results_gemma_packing
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir ~/scratch/results_gemma_packing \
    --hf-cache $HF_CACHE \
    --base-model google/gemma-2-9b-it \
    --no-cot-prompt

echo "Done. Results in ~/scratch/results_gemma_packing/"
