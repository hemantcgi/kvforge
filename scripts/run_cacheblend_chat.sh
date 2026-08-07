#!/usr/bin/env bash
# CacheBlend C2 — recompute_ratio sweep with chat_template prompt format
set -e
cd ~/kvforge
source venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

TIMESTAMP=$(date +%Y%m%d_%H%M)
RESULTS_DIR=results/cacheblend_chat_${TIMESTAMP}
mkdir -p "$RESULTS_DIR"

JUDGE_MODEL="accounts/fireworks/models/deepseek-v4-flash"
JUDGE_KEY="fw_3ZFs7JStEVYvrQdK7FhwC9qw"
EVAL_SET="examples/usecase4_bedrock_userguide/eval_tier_500.json"
BASE_CFG="examples/usecase4_bedrock_userguide/config.json"

for RATIO in 0.0 0.1 0.25 0.5 0.75 1.0; do
    CFG=$(mktemp /tmp/cfg_cb_XXXXXX.json)
    python3 -c "
import json
with open('$BASE_CFG') as f:
    d = json.load(f)
d['recompute_ratio'] = $RATIO
d['prompt_format'] = 'chat_template'
json.dump(d, open('$CFG', 'w'))
"
    echo "[$(date)] recompute_ratio=$RATIO (chat_template)"
    python3 -u -m tools.measure_baseline_fkds \
        --config "$CFG" \
        --eval-set "$EVAL_SET" \
        --modes kv_fulltoken \
        --judge-model "$JUDGE_MODEL" \
        --judge-provider openai \
        --judge-api-key "$JUDGE_KEY" \
        --judge-base-url "https://api.fireworks.ai/inference/v1" \
        --output-dir "$RESULTS_DIR/ratio_${RATIO}"
    rm -f "$CFG"
    echo "[$(date)] ratio=$RATIO done"
done

python3 -c "
import json, glob
for f in sorted(glob.glob('$RESULTS_DIR/*/summary.json')):
    label = f.split('/')[-2].replace('ratio_', '')
    d = json.load(open(f))
    kv = d['modes']['kv_fulltoken']['factual_accuracy']
    print(f'{label:<10} {kv[\"mean\"]:.4f}')
" | tee "$RESULTS_DIR/aggregate.txt"
echo "Done. Results in $RESULTS_DIR"
