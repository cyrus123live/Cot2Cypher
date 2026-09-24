#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=16:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=gemma-fscot

# ==========================================================
# E-A, CYPHER CELL — WEAK-RECIPE CoT on Neo4j.
#
# The CoT target in exactly the A3 format (same prompt, "Reasoning: ... Cypher
# output: ..." target, same rows-with-reasoning filter) trained with FULL-SEQUENCE
# loss. Its matched control already exists: the 1c full-sequence DIRECT arm
# (GLEU 0.7415). vs that arm the ONLY variable is the training target.
#
# PRE-REGISTERED PREDICTION: CoT still loses to weak direct (GLEU < 0.7415),
# by less than the strong recipe's -0.017 -- the fragmentation / value-
# corruption mechanism is a structural cost the weak recipe does not remove.
# Read together with the Spider cell (drac_spider_weak.sh): Spider flips but
# Cypher stays negative -> baseline strength AND output structure both matter.
#
# Output: ~/scratch/gemma_fullseq_cot_adapter/final/
# Eval:   drac_gemma_fullseq_cot_eval.sh (submit with --dependency=afterok)
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

module load python/3.11
module load scipy-stack
module load gcc arrow

set -eo pipefail

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
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export BNB_CUDA_VERSION=129
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Copying training data to local SSD..."
if [ ! -f ~/scratch/cot_training_data.jsonl ]; then
    echo "FATAL: ~/scratch/cot_training_data.jsonl not found." >&2
    echo "       Upload it from your Mac with:" >&2
    echo "       scp data/cot_training_data.jsonl cyrusp@fir.alliancecan.ca:~/scratch/" >&2
    exit 1
fi
cp ~/scratch/cot_training_data.jsonl $SLURM_TMPDIR/
echo "Data copied: $(wc -l < $SLURM_TMPDIR/cot_training_data.jsonl) records"

echo "Starting Gemma-2-9B FULL-SEQUENCE CoT fine-tuning (weak-recipe CoT arm)..."
mkdir -p ~/scratch/gemma_fullseq_cot_adapter
python $PROJECT/thesis/scripts/drac_train_gemma_baseline.py \
    --train-data $SLURM_TMPDIR/cot_training_data.jsonl \
    --output-dir ~/scratch/gemma_fullseq_cot_adapter \
    --hf-cache $HF_CACHE \
    --full-sequence --cot

echo "Done. Adapter saved to ~/scratch/gemma_fullseq_cot_adapter/"
