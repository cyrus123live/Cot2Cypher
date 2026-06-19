#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=06:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=star-cot

# ==========================================================
# STaR step 3: retrain Gemma-2-9B on EXECUTION-FILTERED CoT traces.
#
# The verified-correct traces (forward-generated, kept only if the query
# executes to the reference result) are the ingredient our original post-hoc
# rationalization pipeline lacked and that STaR-SQL used. This tests whether
# execution-filtered CoT recovers the SQL-style gain for Cypher.
#
# drac_train_zograscope.py is reused unchanged: it trains on the
# (reasoning, cypher) traces with the CoT prompt + completion-only masking.
# The filtered file has the same fields (question/schema/cypher/reasoning).
#
# Prereq: ~/scratch/forward_traces_filtered.jsonl (scp from local after the
#         execution-filter step completes).
#
# Output: ~/scratch/star_cot_adapter/final/
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

if [ ! -f ~/scratch/forward_traces_filtered.jsonl ]; then
    echo "FATAL: ~/scratch/forward_traces_filtered.jsonl not found." >&2
    echo "       scp it from local: scp data/forward_traces_filtered.jsonl cyrusp@fir.alliancecan.ca:~/scratch/" >&2
    exit 1
fi
cp ~/scratch/forward_traces_filtered.jsonl $SLURM_TMPDIR/
echo "Filtered traces: $(wc -l < $SLURM_TMPDIR/forward_traces_filtered.jsonl) records"

echo "Training STaR (execution-filtered) CoT model..."
mkdir -p ~/scratch/star_cot_adapter
python $PROJECT/thesis/scripts/drac_train_zograscope.py \
    --train-data $SLURM_TMPDIR/forward_traces_filtered.jsonl \
    --output-dir ~/scratch/star_cot_adapter \
    --hf-cache $HF_CACHE

echo "Done. Adapter at ~/scratch/star_cot_adapter/"
