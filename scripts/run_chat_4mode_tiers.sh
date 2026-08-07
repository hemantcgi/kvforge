#!/usr/bin/env bash
# 4-mode comparison tiers N1000-N6000 with chat_template prompt format
set -e
cd ~/kvforge
source venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

CFG=$(mktemp /tmp/cfg_chat_tiers_XXXXXX.json)
python3 -c "
import json
with open('examples/usecase4_bedrock_userguide/config.json') as f:
    d = json.load(f)
d['prompt_format'] = 'chat_template'
json.dump(d, open('$CFG', 'w'))
"
for N in 1000 2000 4000 6000; do
    echo "[$(date)] N${N}"
    python3 -u -m tools.measure_baseline_fkds \
        --config "$CFG" \
        --eval-set examples/usecase4_bedrock_userguide/eval_tier_${N}.json \
        --modes kv_fulltoken kv_meanpool text_rag parametric \
        --judge-model accounts/fireworks/models/deepseek-v4-flash \
        --judge-provider openai \
        --judge-api-key fw_3ZFs7JStEVYvrQdK7FhwC9qw \
        --judge-base-url https://api.fireworks.ai/inference/v1 \
        --output-dir results/chat_4mode_tiers_$(date +%Y%m%d_%H%M)/N${N}
    echo "[$(date)] N${N} done"
done
rm -f "$CFG"
echo "All tiers done"
