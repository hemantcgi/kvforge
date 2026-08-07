#!/usr/bin/env bash
# Prompt format ablation: system_prompt vs simple vs chat_template
set -e
cd ~/kvforge
source venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

TIMESTAMP=$(date +%Y%m%d_%H%M)
RESULTS_DIR=results/ablation_prompt_${TIMESTAMP}
mkdir -p "$RESULTS_DIR" logs

JUDGE_MODEL="accounts/fireworks/models/deepseek-v4-flash"
JUDGE_KEY="fw_3ZFs7JStEVYvrQdK7FhwC9qw"
EVAL_SET="examples/usecase4_bedrock_userguide/eval_tier_500.json"
BASE_CFG="examples/usecase4_bedrock_userguide/config.json"

for FORMAT in system_prompt simple chat_template; do
    echo "[$(date)] Testing prompt_format=$FORMAT"
    CFG=$(mktemp /tmp/cfg_ablate_XXXXXX.json)
    python3 -c "
import json
with open('$BASE_CFG') as f:
    d = json.load(f)
d['prompt_format'] = '$FORMAT'
json.dump(d, open('$CFG', 'w'))
"
    python3 -u -m tools.measure_baseline_fkds \
        --config "$CFG" \
        --eval-set "$EVAL_SET" \
        --modes kv_fulltoken \
        --judge-model "$JUDGE_MODEL" \
        --judge-provider openai \
        --judge-api-key "$JUDGE_KEY" \
        --judge-base-url "https://api.fireworks.ai/inference/v1" \
        --output-dir "$RESULTS_DIR/$FORMAT" 2>&1 | tee -a "logs/ablation_${FORMAT}.log"
    rm -f "$CFG"
    echo "[$(date)] $FORMAT done"
done

echo ""
echo "=== Prompt Format Ablation Results ==="
python3 -c "
import json, glob
for f in sorted(glob.glob('$RESULTS_DIR/*/summary.json')):
    fmt = f.split('/')[-2]
    d = json.load(open(f))
    m = d.get('modes', {})
    kv = m.get('kv_fulltoken', {})
    fa = kv.get('factual_accuracy', {})
    print(f'{fmt:<20} fa={fa.get(\"mean\",0):.4f} em={fa.get(\"em\",0):.4f} f1={fa.get(\"f1\",0):.4f}')
" | tee "$RESULTS_DIR/aggregate.txt"
echo "Done. Results in $RESULTS_DIR"
