#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=24:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=gemma-fscot-e

# ==========================================================
# Evaluate the weak-recipe (full-sequence) CoT adapter on the Neo4j test set
# with the CoT prompt -- identical inference to the A3 eval.
#
# Decisive comparison (only the training target differs; both full-sequence):
#   1c full-sequence direct : GLEU 0.7415
#   full-sequence CoT       : GLEU ??????  <- this run
# Strong-recipe reference: direct 0.7854 / CoT 0.7682 (CoT effect -0.017).
#
# 24h wall: CoT inference over 4,833 long-schema prompts is slow (the 1b eval
# timed out at 12h). drac_inference.py checkpoints and resumes, so a TIMEOUT
# just needs a resubmit.
#
# Output: ~/scratch/results_gemma_fullseq_cot/predictions_cot_greedy.jsonl
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

[ -f ~/scratch/gemma_fullseq_cot_adapter/final/adapter_config.json ] || {
    echo "FATAL: ~/scratch/gemma_fullseq_cot_adapter/final missing -- did the train job finish?" >&2
    exit 1; }

echo "Copying full-sequence CoT adapter..."
cp -r ~/scratch/gemma_fullseq_cot_adapter/final/ $SLURM_TMPDIR/adapter/
echo "Adapter copied."

echo "Starting eval of full-sequence CoT with the CoT prompt..."
mkdir -p ~/scratch/results_gemma_fullseq_cot
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir ~/scratch/results_gemma_fullseq_cot \
    --hf-cache $HF_CACHE \
    --base-model google/gemma-2-9b-it

echo "Done. Results in ~/scratch/results_gemma_fullseq_cot/"
echo "Compare GLEU to 1c full-sequence direct 0.7415 (strong recipe: direct 0.7854 / CoT 0.7682)"
