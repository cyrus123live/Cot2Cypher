#!/bin/bash
# ==========================================================
# One-time setup for DRAC cluster (run on login node)
#
# Downloads model weights, datasets, and pip packages
# that aren't available in DRAC's wheelhouse.
#
# Usage:
#   bash scripts/drac_setup.sh
# ==========================================================

set -e

# PROJECT on DRAC = ~/projects/def-thomo/cyrusp
export PROJECT=~/projects/def-thomo/cyrusp
export HF_CACHE=~/scratch/hf_cache

echo "=== DRAC Setup for CoT Text2Cypher ==="
echo "PROJECT=$PROJECT"
echo "HOME=$HOME"

# 1. Create directories
mkdir -p $PROJECT/thesis/data
mkdir -p $PROJECT/thesis/adapter_weights
mkdir -p $PROJECT/thesis/results
mkdir -p $HF_CACHE
mkdir -p $HOME/wheels

# 2. Download pip packages not in DRAC wheelhouse
echo "Downloading pip packages..."
module load python/3.11
TMPENV=/tmp/setup_env_$$
virtualenv --no-download $TMPENV
source $TMPENV/bin/activate
pip install --no-index --upgrade pip

# Check what's available in wheelhouse
echo "Checking DRAC wheelhouse availability..."
avail_wheels peft 2>/dev/null || echo "peft: not in wheelhouse"
avail_wheels trl 2>/dev/null || echo "trl: not in wheelhouse"
avail_wheels bitsandbytes 2>/dev/null || echo "bitsandbytes: not in wheelhouse"

# Download packages not in wheelhouse (or that have build issues)
pip download --no-deps -d $HOME/wheels peft trl bitsandbytes 2>/dev/null || true
pip download -d $HOME/wheels datasets evaluate pyarrow nltk 2>/dev/null || true
echo "Downloaded wheels to $HOME/wheels"

deactivate
rm -rf $TMPENV

# 3. Download HuggingFace model weights
echo "Downloading Gemma-2-9B-it weights (this takes a while)..."
module load python/3.11
TMPENV2=/tmp/hf_download_$$
virtualenv --no-download $TMPENV2
source $TMPENV2/bin/activate
pip install --no-index --upgrade pip
pip install --no-index transformers
pip install huggingface_hub 2>/dev/null || pip download --no-deps -d /tmp huggingface_hub && pip install --no-index --find-links /tmp huggingface_hub

python -c "
from huggingface_hub import snapshot_download
import os

cache_dir = os.environ.get('HF_CACHE', os.path.expanduser('~/scratch/hf_cache'))

# Download base model
print('Downloading google/gemma-2-9b-it...')
snapshot_download('google/gemma-2-9b-it', cache_dir=cache_dir)

# Download dataset
print('Downloading neo4j/text2cypher-2024v1...')
snapshot_download('neo4j/text2cypher-2024v1', repo_type='dataset', cache_dir=cache_dir)

print('Done downloading.')
"

deactivate
rm -rf $TMPENV2

# 4. Remind about manual steps
echo ""
echo "=== Setup complete ==="
echo ""
echo "Manual steps remaining:"
echo "  1. Copy adapter weights:  scp -r adapter_weights/final/ <cluster>:\$PROJECT/thesis/adapter_weights/final/"
echo "  2. Copy data files:       scp data/test_db_mapping.json <cluster>:\$PROJECT/thesis/data/"
echo "  3. Copy scripts:          scp -r scripts/ <cluster>:\$PROJECT/thesis/scripts/"
echo "  4. Update SBATCH account: edit scripts/drac_eval.sh, replace def-SUPERVISOR with your account"
echo "  5. Submit job:            cd \$PROJECT/thesis && sbatch scripts/drac_eval.sh"
echo ""
echo "For self-consistency (5 samples):"
echo "  sbatch scripts/drac_eval.sh --self-consistency 5"
