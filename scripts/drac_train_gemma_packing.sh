#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=14:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=gemma-packing

# ==========================================================
# PACKING ABLATION — the last piece of the +0.14 training-gap decomposition.
#
# Ladder so far (same data, same QLoRA config, same prompt; Neo4j test GLEU):
#   A5  completion-only masking, no packing : 0.7854
#   1c  full-sequence loss,     no packing : 0.7415   (masking explains +0.044)
#   ??  full-sequence loss,     packing    : this run (Neo4j's FULL recipe:
#       SFTTrainer(dataset_text_field="text", packing=True), verified from
#       neo4j-labs/text2cypher notebooks)
#   A2  Neo4j's published adapter, our inference : 0.6455
#
# The ONLY variable vs the 1c arm is packing. If this lands near A2's 0.6455,
# the +0.14 training gap is FULLY explained (masking +0.044, packing the rest)
# and the recipe section's "remainder unexplained" caveat can be retired.
# If it stays near 0.74, the residual is framework/version and stays flagged.
#
# Output: ~/scratch/gemma_packing_adapter/final/
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

echo "Starting Gemma-2-9B PACKING-ablation fine-tuning (Neo4j full recipe)..."
mkdir -p ~/scratch/gemma_packing_adapter
python $PROJECT/thesis/scripts/drac_train_gemma_baseline.py \
    --train-data $SLURM_TMPDIR/cot_training_data.jsonl \
    --output-dir ~/scratch/gemma_packing_adapter \
    --hf-cache $HF_CACHE \
    --packing

echo "Done. Adapter saved to ~/scratch/gemma_packing_adapter/"
