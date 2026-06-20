#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=zog-ceval

# ==========================================================
# CLEAN ZOGRASCOPE eval — each model on its OWN experiment's test set.
#
# Parameterized (sbatch --export):
#   EXPERIMENT = length | regular
#   MODE       = cot | baseline
#
#   sbatch --export=ALL,EXPERIMENT=length,MODE=cot      scripts/drac_eval_zog_clean.sh
#   sbatch --export=ALL,EXPERIMENT=length,MODE=baseline scripts/drac_eval_zog_clean.sh
#   sbatch --export=ALL,EXPERIMENT=regular,MODE=cot     scripts/drac_eval_zog_clean.sh
#   sbatch --export=ALL,EXPERIMENT=regular,MODE=baseline scripts/drac_eval_zog_clean.sh
#
# Prereqs on Fir (scp from local):
#   ~/scratch/test_length.jsonl  (length_test_v1, 1253)
#   ~/scratch/test_regular.jsonl (test_v1 iid+comp, 2117)
#   plus the trained adapter from drac_train_zog_clean.sh
#
# Output: ~/scratch/results_zog_${EXPERIMENT}_${MODE}/predictions_zograscope_greedy.jsonl
# ==========================================================

set -e
: "${EXPERIMENT:?set EXPERIMENT=length|regular}"
: "${MODE:?set MODE=cot|baseline}"

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

ADAPTER=~/scratch/zog_${EXPERIMENT}_${MODE}_adapter/final
TEST_FILE=~/scratch/test_${EXPERIMENT}.jsonl
OUT_DIR=~/scratch/results_zog_${EXPERIMENT}_${MODE}
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
cp "$TEST_FILE" $SLURM_TMPDIR/test.jsonl
echo "[$EXPERIMENT/$MODE] eval on $(wc -l < $SLURM_TMPDIR/test.jsonl) examples, prompt='$PROMPT_FLAG'"
mkdir -p "$OUT_DIR"
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir "$OUT_DIR" \
    --hf-cache $HF_CACHE \
    --base-model google/gemma-2-9b-it \
    --zograscope $SLURM_TMPDIR/test.jsonl \
    $PROMPT_FLAG

echo "Done. Predictions in $OUT_DIR/"
