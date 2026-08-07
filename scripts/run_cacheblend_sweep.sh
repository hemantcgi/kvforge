#!/usr/bin/env bash
# CacheBlend partial recompute sweep (C2) on UC4 Bedrock
set -e
cd ~/kvforge
source venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

RESULTS_DIR=results/cacheblend_sweep_$(date +%Y%m%d_%H%M)
mkdir -p "$RESULTS_DIR" logs

for RATIO in 0.0 0.1 0.25 0.5 0.75 1.0; do
    echo "[$(date)] recompute_ratio=$RATIO"
    CFG=$(mktemp /tmp/cfg_cacheblend_XXXXXX.json)
    python3 -c "
import json
with open('examples/usecase4_bedrock_userguide/config.json') as f:
    d = json.load(f)
d['recompute_ratio'] = $RATIO
json.dump(d, open('$CFG', 'w'))
"
    python3 -u -m tools.measure_baseline_fkds \
        --config "$CFG" \
        --eval-set examples/usecase4_bedrock_userguide/eval_tier_500.json \
        --modes kv_fulltoken \
        --judge-model accounts/fireworks/models/deepseek-v4-flash \
        --judge-provider openai \
        --judge-api-key fw_3ZFs7JStEVYvrQdK7FhwC9qw \
        --judge-base-url "https://api.fireworks.ai/inference/v1" \
        --output-dir "$RESULTS_DIR/ratio_${RATIO}" 2>&1 | tee -a logs/cacheblend_ratio_${RATIO}.log
    rm -f "$CFG"
    echo "[$(date)] ratio=$RATIO done"
done

echo ""
echo "=== CacheBlend Sweep Results ==="
python3 -c "
import json, glob
for f in sorted(glob.glob('$RESULTS_DIR/*/summary.json')):
    label = f.split('/')[-2]
    d = json.load(open(f))
    m = d.get('modes', {})
    val = m.get('kv_fulltoken', {}).get('factual_accuracy', {}).get('mean', 0)
    print(f'{label:<15} {val:.4f}')
" | tee "$RESULTS_DIR/aggregate.txt"
echo "Done. Results in $RESULTS_DIR"
