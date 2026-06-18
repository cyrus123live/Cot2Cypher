#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64000M
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=fwd-gen

# ==========================================================
# FORWARD trace generation (STaR-style) on DRAC.
#
# Serves the teacher (gpt-oss-120b) locally with vLLM on one H100, then
# generates k forward candidates (reasoning -> Cypher, NO reference answer)
# for every DB-accessible training example. Output is execution-filtered in
# a later step.
#
# ⚠️ VERIFY BEFORE RELYING ON A LONG RUN (these can't be checked from laptop):
#   1. vLLM with gpt-oss support is installable on Fir (check $HOME/wheels or
#      `pip install vllm` with internet on the compute node). gpt-oss needs a
#      recent vLLM (>=0.6.x with gpt-oss kernels).
#   2. The dataset field name for DB access: generate_forward_traces.py reads
#      row["database_reference_alias"]. Confirm that's the field in
#      neo4j/text2cypher-2024v1 (could be "database_reference"). Adjust if needed.
#   3. Fir compute nodes can reach demo.neo4jlabs.com (needed for the LATER
#      execution-filter step, not this one). Fir is documented as having internet.
#
# Run a --limit 50 pilot first to validate end-to-end before the full ~22k.
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

module load python/3.11
module load scipy-stack
module load gcc arrow

virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# vLLM + deps. If --no-index lacks vllm, allow internet install on Fir:
pip install --no-index torch torchvision 2>/dev/null || true
pip install vllm openai datasets 2>&1 | tail -5

MODEL="openai/gpt-oss-120b"
PORT=8000

# 1. Start vLLM server in the background (MXFP4 fits gpt-oss-120b on one 80GB H100)
echo "Starting vLLM server for $MODEL ..."
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --download-dir $HF_CACHE \
    > $PROJECT/vllm_server.log 2>&1 &
VLLM_PID=$!

# 2. Wait for the server to come up (up to ~20 min for first-time weight download)
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/v1/models >/dev/null 2>&1; then
        echo "vLLM is up after ~$((i*10))s."
        break
    fi
    sleep 10
done

if ! curl -s http://localhost:$PORT/v1/models >/dev/null 2>&1; then
    echo "FATAL: vLLM did not come up. Check $PROJECT/vllm_server.log" >&2
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# 3. Generate forward traces against the local endpoint
export COT_API_BASE="http://localhost:$PORT/v1"
export COT_API_KEY="dummy"

echo "Starting forward generation..."
python $PROJECT/thesis/scripts/generate_forward_traces.py \
    --model $MODEL \
    --k 4 \
    --temperature 0.7 \
    --concurrency 32 \
    --output ~/scratch/forward_traces.jsonl \
    "${EXTRA_ARGS:-}"

echo "Generation done. Shutting down vLLM."
kill $VLLM_PID 2>/dev/null

echo "Forward traces at ~/scratch/forward_traces.jsonl"
