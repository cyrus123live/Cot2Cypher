#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00              # 12 hrs (4833 examples, greedy)
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=llama-eval

# ==========================================================
# Evaluate the Llama-3.1-8B CoT adapter on the Neo4j test set.
#
# Uses the adapter trained by drac_train_llama_cot.sh.
# Output: ~/scratch/results_llama/predictions_cot_greedy.jsonl
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
pip install --no-index pyarrow
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf
pip install --no-index nltk
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export BNB_CUDA_VERSION=129

echo "Copying Llama CoT adapter..."
cp -r ~/scratch/llama_cot_adapter/final/ $SLURM_TMPDIR/adapter/
echo "Adapter copied."

echo "Starting evaluation of Llama-3.1-8B CoT model..."
mkdir -p ~/scratch/results_llama
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir ~/scratch/results_llama \
    --hf-cache $HF_CACHE \
    --base-model meta-llama/Llama-3.1-8B-Instruct

echo "Done. Results in ~/scratch/results_llama/"
