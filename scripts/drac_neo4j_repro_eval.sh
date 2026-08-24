#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=neo4j-repro

# ==========================================================
# ITEM 1b — HARNESS CALIBRATION: reproduce Neo4j's published GLEU 0.5560.
#
# Runs Neo4j's PUBLISHED adapter (neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1)
# through our inference with ONE change vs the A2 run: --max-length 1600, i.e.
# inputs hard-truncated at Neo4j's training max_seq_len (right-truncation, the
# tokenizer default — long schemas lose their tail, including the question for
# the longest ones). Prompt is the exact model-card format (--no-cot-prompt);
# greedy decoding; 4-bit NF4 — all identical to A2.
#
#   A1 published number            : GLEU 0.5560
#   A2 our inference, full schema  : GLEU 0.6455  (max_length=7680)
#   1b our inference, truncated    : this run     (max_length=1600)
#
# If 1b lands near 0.5560 -> the +0.09 harness component of the +0.23 gap is
# confirmed as inference truncation and the harness is calibrated to the
# leaderboard scale. If not -> the residual is framework/version and stays
# documented as such. Neither outcome changes any conclusion.
#
# PREREQUISITE (login node, has internet) — provide the published adapter by EITHER:
#   (a) hf_hub download into the cache:
#       module load python/3.11
#       python -m pip install --no-index --user huggingface_hub
#       HF_HOME=~/scratch/hf_cache python -c "from huggingface_hub import snapshot_download; \
#           print(snapshot_download('neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1'))"
#       python -m pip uninstall -y huggingface_hub   # avoid shadowing job venvs
#   (b) a plain git-lfs clone (no python needed):
#       module load git-lfs && cd ~/scratch && \
#       git clone https://huggingface.co/neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1 \
#           neo4j_published_adapter
#   The job checks ~/scratch/neo4j_published_adapter first, then the HF cache.
#
# Output: ~/scratch/results_neo4j_repro/predictions_cot_greedy.jsonl
# (prefix fixed by drac_inference.py; the directory disambiguates)
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

module load python/3.11
module load scipy-stack
module load gcc arrow

set -eo pipefail  # fail LOUDLY -- a silent venv/CVMFS failure previously printed "Done"

# CVMFS can throw transient Errno-5 I/O errors seeding the venv (seen on fc10620);
# retry a few times before giving up.
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

echo "Resolving the published Neo4j adapter..."
if [ -f ~/scratch/neo4j_published_adapter/adapter_config.json ]; then
    ADAPTER_DIR=~/scratch/neo4j_published_adapter
else
    ADAPTER_DIR=$(python - <<'EOF'
from huggingface_hub import snapshot_download
repo = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
try:
    p = snapshot_download(repo, local_files_only=True)
except Exception:
    p = snapshot_download(repo)  # compute node may lack internet; see header
print(p)
EOF
    )
fi
if [ -z "$ADAPTER_DIR" ] || [ ! -d "$ADAPTER_DIR" ]; then
    echo "FATAL: could not resolve the published adapter. Pre-download it on the" >&2
    echo "       login node (see the PREREQUISITE in this script's header)." >&2
    exit 1
fi
echo "Adapter: $ADAPTER_DIR"

echo "Starting 1b calibration eval (model-card prompt, max_length=1600, greedy)..."
mkdir -p ~/scratch/results_neo4j_repro
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path "$ADAPTER_DIR" \
    --output-dir ~/scratch/results_neo4j_repro \
    --hf-cache $HF_CACHE \
    --base-model google/gemma-2-9b-it \
    --no-cot-prompt \
    --max-length 1600

echo "Done. Results in ~/scratch/results_neo4j_repro/"
echo "Compare metrics_cot_greedy.json GLEU to: published 0.5560 / A2 full-schema 0.6455"
