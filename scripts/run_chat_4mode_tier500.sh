#!/usr/bin/env bash
# 4-mode comparison N500 with chat_template prompt format
set -e
cd ~/kvforge
source venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

CFG=$(mktemp /tmp/cfg_chat_XXXXXX.json)
python3 -c "
import json
with open('examples/usecase4_bedrock_userguide/config.json') as f:
    d = json.load(f)
d['prompt_format'] = 'chat_template'
json.dump(d, open('$CFG', 'w'))
"
python3 -u -m tools.measure_baseline_fkds \
    --config "$CFG" \
    --eval-set examples/usecase4_bedrock_userguide/eval_tier_500.json \
    --modes kv_fulltoken kv_meanpool text_rag parametric \
    --judge-model accounts/fireworks/models/deepseek-v4-flash \
    --judge-provider openai \
    --judge-api-key fw_3ZFs7JStEVYvrQdK7FhwC9qw \
    --judge-base-url https://api.fireworks.ai/inference/v1 \
    --output-dir results/chat_4mode_$(date +%Y%m%d_%H%M)/N500
rm -f "$CFG"
echo "Done"
