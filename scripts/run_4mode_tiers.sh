#!/usr/bin/env bash
# Phase 3.2 — remaining 4-mode tiers: N1000, N2000, N4000, N6000
set -e
cd ~/kvforge
source venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

for N in 1000 2000 4000 6000; do
    echo "[$(date)] N${N}: 4-mode comparison"
    python3 -u -m tools.measure_baseline_fkds \
        --config examples/usecase4_bedrock_userguide/config.json \
        --eval-set examples/usecase4_bedrock_userguide/eval_tier_${N}.json \
        --modes kv_fulltoken kv_meanpool text_rag parametric \
        --judge-model accounts/fireworks/models/deepseek-v4-flash \
        --judge-provider openai \
        --judge-api-key fw_3ZFs7JStEVYvrQdK7FhwC9qw \
        --judge-base-url https://api.fireworks.ai/inference/v1 \
        --output-dir results/phase3_tiers_$(date +%Y%m%d_%H%M)/N${N}
    echo "[$(date)] N${N}: Done"
done
echo "All tiers complete"
