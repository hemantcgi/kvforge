#!/usr/bin/env bash
# Phase 3.2a,b — kv_fulltoken accuracy sweep on UC4 Bedrock
set -e
cd ~/kvforge
source venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

FIREWORKS_KEY="fw_3ZFs7JStEVYvrQdK7FhwC9qw"
JUDGE_MODEL="accounts/fireworks/models/deepseek-v4-flash"
CFG=examples/usecase4_bedrock_userguide/config.json
DATA_DIR=examples/usecase4_bedrock_userguide
RESULTS_DIR=results/fulltoken_sweep_$(date +%Y%m%d_%H%M)

mkdir -p "$RESULTS_DIR" logs
echo "Results dir: $RESULTS_DIR"

for N in 500 1000 2000 4000 6000; do
    EVAL_SET=$DATA_DIR/eval_tier_${N}.json
    RUN_DIR=$RESULTS_DIR/N${N}
    mkdir -p "$RUN_DIR"
    echo "[$(date)] N${N}: kv_fulltoken evaluation..."
    python3 -u -m tools.measure_baseline_fkds \
        --config "$CFG" \
        --eval-set "$EVAL_SET" \
        --modes kv_fulltoken \
        --judge-model "$JUDGE_MODEL" \
        --judge-provider openai \
        --judge-api-key "$FIREWORKS_KEY" \
        --judge-base-url "https://api.fireworks.ai/inference/v1" \
        --output-dir "$RUN_DIR" 2>&1 | tee -a logs/fulltoken_sweep_${N}.log
    echo "[$(date)] N${N}: Done"
done

echo ""
echo "=== kv_fulltoken Sweep Results ==="
python3 -c "
import json, glob
print('Tier    kv_fulltoken')
print('-'*30)
for f in sorted(glob.glob('results/fulltoken_sweep_*/N*/summary.json')):
    label = f.split('/')[-2]
    d = json.load(open(f))
    m = d.get('modes', {})
    val = m.get('kv_fulltoken', {}).get('factual_accuracy', {}).get('mean', 0)
    print(f'{label:<7} {val:.4f}')
" 2>&1 | tee "$RESULTS_DIR/aggregate.txt"
echo "Sweep complete. Results in $RESULTS_DIR"
