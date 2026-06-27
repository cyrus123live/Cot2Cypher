#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=14:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=gemma-fullseq

# ==========================================================
# ITEM 1c — LOSS-MASKING ABLATION.
#
# Identical to drac_train_gemma_baseline.sh (which produced the A5 direct-answer
# baseline, GLEU 0.7854 / EM 0.4331) EXCEPT for a single flag: --full-sequence.
# That flips the loss target from completion-only (answer tokens only) to
# FULL-SEQUENCE (every non-pad token, schema-heavy prompt included) — matching
# Neo4j's published recipe (SFTTrainer + dataset_text_field + packing=True, no
# completion-only collator; verified from neo4j-labs/text2cypher notebooks).
#
# Same training data, same QLoRA config, same lr/batch/epochs/max_seq_len, same
# eager attention, same prompt. The ONLY variable is the loss mask.
#
# Hypothesis: full-sequence loss collapses GLEU from 0.7854 toward ~0.64,
# confirming that completion-only masking — NOT CoT — is what produced the
# apparent improvement over Neo4j's published 0.5560.
#
# Output: ~/scratch/gemma_fullseq_adapter/final/
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

module load python/3.11
module load scipy-stack
module load gcc arrow

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

echo "Copying training data to local SSD..."
if [ ! -f ~/scratch/cot_training_data.jsonl ]; then
    echo "FATAL: ~/scratch/cot_training_data.jsonl not found." >&2
    echo "       Upload it from your Mac with:" >&2
    echo "       scp data/cot_training_data.jsonl cyrusp@fir.alliancecan.ca:~/scratch/" >&2
    exit 1
fi
cp ~/scratch/cot_training_data.jsonl $SLURM_TMPDIR/
echo "Data copied: $(wc -l < $SLURM_TMPDIR/cot_training_data.jsonl) records, $(du -h $SLURM_TMPDIR/cot_training_data.jsonl | cut -f1)"

echo "Starting Gemma-2-9B FULL-SEQUENCE (1c ablation) fine-tuning..."
mkdir -p ~/scratch/gemma_fullseq_adapter
python $PROJECT/thesis/scripts/drac_train_gemma_baseline.py \
    --train-data $SLURM_TMPDIR/cot_training_data.jsonl \
    --output-dir ~/scratch/gemma_fullseq_adapter \
    --hf-cache $HF_CACHE \
    --full-sequence

echo "Done. Adapter saved to ~/scratch/gemma_fullseq_adapter/"
