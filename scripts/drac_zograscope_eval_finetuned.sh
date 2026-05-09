#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1       # 1x H100-80GB on Fir
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00              # 12 hours (3370 examples, greedy only)
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=zog-eval-ft

# ==========================================================
# Evaluate the ZOGRASCOPE-FINE-TUNED adapter on the ZOGRASCOPE test sets.
#
# Uses the adapter trained by drac_train_zograscope.sh (saved to
# ~/scratch/zograscope_adapter/final/), as opposed to the original
# Neo4j-trained adapter at ~/scratch/adapter_weights/final/.
#
# Output: ~/scratch/results_finetuned_zog/predictions_zograscope_greedy.jsonl
#
# Usage:
#   sbatch scripts/drac_zograscope_eval_finetuned.sh
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

# 1. Load modules
module load python/3.11
module load scipy-stack
module load gcc arrow

# 2. Create virtualenv
virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch torchvision
pip install --no-index pyarrow
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf
pip install --no-index nltk
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

# 3. HF settings
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

# 4. Copy ZOGRASCOPE-fine-tuned adapter to local SSD
echo "Copying ZOGRASCOPE fine-tuned adapter..."
cp -r ~/scratch/zograscope_adapter/final/ $SLURM_TMPDIR/adapter/
echo "Adapter copied."

# 5. Copy ZOGRASCOPE test data to local SSD
echo "Copying ZOGRASCOPE test data..."
cp $PROJECT/thesis/data/zograscope/zograscope_formatted.jsonl $SLURM_TMPDIR/
echo "Data copied."

# 6. Run inference, write to a NEW output dir so we don't clobber the zero-shot results
echo "Starting evaluation of ZOGRASCOPE fine-tuned model..."
mkdir -p ~/scratch/results_finetuned_zog
python $PROJECT/thesis/scripts/drac_inference.py \
    --adapter-path $SLURM_TMPDIR/adapter \
    --output-dir ~/scratch/results_finetuned_zog \
    --hf-cache $HF_CACHE \
    --zograscope $SLURM_TMPDIR/zograscope_formatted.jsonl

echo "Done. Results in ~/scratch/results_finetuned_zog/"
