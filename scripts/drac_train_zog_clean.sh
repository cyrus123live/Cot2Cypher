#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=04:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=zog-clean

# ==========================================================
# CLEAN ZOGRASCOPE training — one experiment at a time (no merged splits).
#
# Parameterized by env vars (pass via sbatch --export):
#   EXPERIMENT = length | regular
#   MODE       = cot | baseline
#
# Examples (launch all four in parallel):
#   sbatch --export=ALL,EXPERIMENT=length,MODE=cot      scripts/drac_train_zog_clean.sh
#   sbatch --export=ALL,EXPERIMENT=length,MODE=baseline scripts/drac_train_zog_clean.sh
#   sbatch --export=ALL,EXPERIMENT=regular,MODE=cot     scripts/drac_train_zog_clean.sh
#   sbatch --export=ALL,EXPERIMENT=regular,MODE=baseline scripts/drac_train_zog_clean.sh
#
# Prereqs on Fir (scp from local):
#   ~/scratch/cot_train_length.jsonl   (length_train_v1 traces, 3769)
#   ~/scratch/cot_train_regular.jsonl  (train_v1 traces, 2905)
#
# Output: ~/scratch/zog_${EXPERIMENT}_${MODE}_adapter/final/
# ==========================================================

set -e
: "${EXPERIMENT:?set EXPERIMENT=length|regular}"
: "${MODE:?set MODE=cot|baseline}"
# Optional VARIANT (e.g. "holistic" for Test D). Empty = default QDecomp CoT.
VARIANT="${VARIANT:-}"
SUF=""; [ -n "$VARIANT" ] && SUF="_${VARIANT}"

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

TRAIN_FILE=~/scratch/cot_train_${EXPERIMENT}${SUF}.jsonl
OUT_DIR=~/scratch/zog_${EXPERIMENT}_${MODE}${SUF}_adapter
if [ "$MODE" = "cot" ]; then
    PY=drac_train_zograscope.py        # trains reasoning+cypher, CoT prompt
else
    PY=drac_train_gemma_baseline.py    # trains cypher only, direct-answer prompt
fi

module load python/3.11; module load scipy-stack; module load gcc arrow
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
export BNB_CUDA_VERSION=129
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$TRAIN_FILE" ]; then
    echo "FATAL: $TRAIN_FILE not found. scp it from local data/zograscope/." >&2
    exit 1
fi
cp "$TRAIN_FILE" $SLURM_TMPDIR/train.jsonl
echo "[$EXPERIMENT/$MODE] training on $(wc -l < $SLURM_TMPDIR/train.jsonl) examples via $PY"
mkdir -p "$OUT_DIR"
python $PROJECT/thesis/scripts/$PY \
    --train-data $SLURM_TMPDIR/train.jsonl \
    --output-dir "$OUT_DIR" \
    --hf-cache $HF_CACHE

echo "Done. Adapter at $OUT_DIR/final/"
