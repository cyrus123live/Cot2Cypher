#!/bin/bash
#SBATCH --account=def-thomo
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=16:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --job-name=spider-cpp

# ==========================================================
# SPIDER matched direct-vs-CoT experiment — the DEFINITIVE SQL positive control.
#
# The gretel run used string EM (wrong metric for SQL). This uses Spider with
# EXECUTION ACCURACY, the metric the CoT-helps-SQL literature reports. If CoT
# helps SQL here but hurts Cypher/SPARQL, the compositional-prior theory is
# revived. Reuses drac_train_sql.py (SQL prompts) via --tag spider.
#
# PREREQUISITES (login node w/ internet + API key):
#   ./venv/bin/python scripts/generate_cot_spider.py --prepare-test
#   COT_PROVIDER=groq COT_API_KEY=... ./venv/bin/python scripts/generate_cot_spider.py
#   scp data/spider/spider_cot_traces.jsonl data/spider/spider_test.jsonl cyrusp@fir.alliancecan.ca:~/scratch/
#   # for execution accuracy, also upload the Spider databases:
#   scp -r spider/database cyrusp@fir.alliancecan.ca:~/scratch/spider_database
#
# Outputs: ~/scratch/results_spider/{predictions,metrics}_spider_{direct,cot}.* + execution deltas
# ==========================================================

export PROJECT=~/scratch
export HF_CACHE=~/scratch/hf_cache

module load python/3.11
module load scipy-stack
module load gcc arrow

virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision pyarrow
pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf nltk
pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export BNB_CUDA_VERSION=129
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRACES=~/scratch/spider_cot_traces.jsonl
TEST=~/scratch/spider_test.jsonl
for f in "$TRACES" "$TEST"; do
    [ -f "$f" ] || { echo "FATAL: $f not found (see header)." >&2; exit 1; }
done
echo "Traces: $(wc -l < $TRACES) | Test: $(wc -l < $TEST)"

TRAIN=$PROJECT/thesis/scripts/drac_train_sql.py
RES=~/scratch/results_spider
mkdir -p $RES

echo "==================== TRAIN: direct ===================="
python $TRAIN --variant direct --tag spider --train-data $TRACES \
    --output-dir ~/scratch/spider_direct_adapter --hf-cache $HF_CACHE
echo "==================== TRAIN: cot ===================="
python $TRAIN --variant cot --tag spider --train-data $TRACES \
    --output-dir ~/scratch/spider_cot_adapter --hf-cache $HF_CACHE

echo "==================== EVAL (predictions + string metrics): direct ===================="
python $TRAIN --eval --variant direct --tag spider --test-data $TEST \
    --adapter-path ~/scratch/spider_direct_adapter/final --output-dir $RES --hf-cache $HF_CACHE
echo "==================== EVAL (predictions + string metrics): cot ===================="
python $TRAIN --eval --variant cot --tag spider --test-data $TEST \
    --adapter-path ~/scratch/spider_cot_adapter/final --output-dir $RES --hf-cache $HF_CACHE

echo ""
echo "==================== EXECUTION ACCURACY (the real metric) ===================="
if [ -d ~/scratch/spider_database ]; then
    python $PROJECT/thesis/scripts/eval_spider_execution.py --db-dir ~/scratch/spider_database \
        $RES/predictions_spider_direct.jsonl $RES/predictions_spider_cot.jsonl
else
    echo "~/scratch/spider_database not found — skipping execution accuracy."
    echo "Upload the Spider database/ folder there and rerun eval_spider_execution.py:"
    echo "  python scripts/eval_spider_execution.py --db-dir ~/scratch/spider_database \\"
    echo "      $RES/predictions_spider_direct.jsonl $RES/predictions_spider_cot.jsonl"
fi
echo "Done. Results in $RES/"
