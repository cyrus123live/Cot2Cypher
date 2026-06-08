#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=14:00:00              # 14 hrs (Gemma was ~10 hr at 1 epoch on A100; Llama-3.1-8B is smaller)
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=llama-cot

# ==========================================================
# Fine-tune Llama-3.1-8B-Instruct on Neo4j CoT traces.
#
# Same QLoRA config and same CoT training data as the Gemma-2-9B headline
# experiment — the ONLY change is the base model. This is Alex's #1 critical
# gap: a second model family to defuse "is this just Gemma?" reviewer attack.
#
# Prerequisites:
#   - ~/scratch/cot_training_data.jsonl  (scp from local data/ — ~130MB)
#   - HuggingFace token at ~/.cache/huggingface/token (Llama-3.1 requires accepting license)
#
# Usage:
#   sbatch scripts/drac_train_llama_cot.sh
#
# Output: ~/scratch/llama_cot_adapter/final/
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
export BNB_CUDA_VERSION=129  # bnb ships 12.2/12.6/12.9; force 12.9 if torch is on 13.x

echo "Copying training data to local SSD..."
if [ ! -f ~/scratch/cot_training_data.jsonl ]; then
    echo "FATAL: ~/scratch/cot_training_data.jsonl not found." >&2
    echo "       Upload it from your Mac with:" >&2
    echo "       scp data/cot_training_data.jsonl cyrusp@fir.alliancecan.ca:~/scratch/" >&2
    exit 1
fi
cp ~/scratch/cot_training_data.jsonl $SLURM_TMPDIR/
echo "Data copied: $(wc -l < $SLURM_TMPDIR/cot_training_data.jsonl) records, $(du -h $SLURM_TMPDIR/cot_training_data.jsonl | cut -f1)"

echo "Starting Llama-3.1-8B CoT fine-tuning..."
mkdir -p ~/scratch/llama_cot_adapter
python $PROJECT/thesis/scripts/drac_train_llama_cot.py \
    --train-data $SLURM_TMPDIR/cot_training_data.jsonl \
    --output-dir ~/scratch/llama_cot_adapter \
    --hf-cache $HF_CACHE

echo "Done. Adapter saved to ~/scratch/llama_cot_adapter/"
