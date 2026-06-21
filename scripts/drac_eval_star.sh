#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=14:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=star-eval

# ==========================================================
# Evaluate a STaR arm on the FULL Neo4j test set (4,833) so we can split
# by seen/unseen locally afterward.
#
#   sbatch --export=ALL,MODE=cot      scripts/drac_eval_star.sh
#   sbatch --export=ALL,MODE=baseline scripts/drac_eval_star.sh
#
# drac_inference.py with no --zograscope loads the Neo4j 2024 test split.
# baseline arm uses --no-cot-prompt (matched direct-answer prompt).
#
# Output: ~/scratch/results_star_${MODE}/predictions_cot_greedy.jsonl
# ==========================================================

set -e
: "${MODE:?set MODE=cot|baseline}"

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

ADAPTER=~/scratch/star_${MODE}_adapter/final
OUT_DIR=~/scratch/results_star_${MODE}
PROMPT_FLAG=""
[ "$MODE" = "baseline" ] && PROMPT_FLAG="--no-cot-prompt"

module load python/3.11; module load scipy-stack; module load gcc arrow
virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision pyarrow
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf nltk
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export BNB_CUDA_VERSION=129
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cp -r "$ADAPTER" $SLURM_TMPDIR/adapter
echo "[star/$MODE] eval on Neo4j test set, prompt='$PROMPT_FLAG'"
mkdir -p "$OUT_DIR"
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir "$OUT_DIR" \
    --hf-cache $HF_CACHE \
    --base-model google/gemma-2-9b-it \
    $PROMPT_FLAG

echo "Done. Predictions in $OUT_DIR/"
