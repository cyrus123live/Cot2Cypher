#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=14:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=llama-base

# ==========================================================
# Fine-tune Llama-3.1-8B-Instruct on the Neo4j *direct-answer* baseline
# (no CoT). Companion to drac_train_llama_cot.sh.
#
# Why: the Llama+CoT comparison vs Neo4j's published 0.6470 GLEU baseline
# mixes pipelines. Training our own Llama baseline with the same QLoRA
# config gives us a clean matched comparison and removes the cross-pipeline
# reviewer attack.
#
# Same training data file as the CoT run (cot_training_data.jsonl);
# the script ignores the `reasoning` field and trains on (q, schema) -> cypher.
#
# Output: ~/scratch/llama_baseline_adapter/final/
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

echo "Copying training data to local SSD..."
if [ ! -f ~/scratch/cot_training_data.jsonl ]; then
    echo "FATAL: ~/scratch/cot_training_data.jsonl not found." >&2
    echo "       Upload it from your Mac with:" >&2
    echo "       scp data/cot_training_data.jsonl cyrusp@fir.alliancecan.ca:~/scratch/" >&2
    exit 1
fi
cp ~/scratch/cot_training_data.jsonl $SLURM_TMPDIR/
echo "Data copied: $(wc -l < $SLURM_TMPDIR/cot_training_data.jsonl) records, $(du -h $SLURM_TMPDIR/cot_training_data.jsonl | cut -f1)"

echo "Starting Llama-3.1-8B baseline (direct-answer) fine-tuning..."
mkdir -p ~/scratch/llama_baseline_adapter
python $PROJECT/thesis/scripts/drac_train_llama_baseline.py \
    --train-data $SLURM_TMPDIR/cot_training_data.jsonl \
    --output-dir ~/scratch/llama_baseline_adapter \
    --hf-cache $HF_CACHE

echo "Done. Adapter saved to ~/scratch/llama_baseline_adapter/"
