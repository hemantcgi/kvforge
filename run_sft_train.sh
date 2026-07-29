#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/kvforge
source venv/bin/activate
echo "=== SFT TRAIN B START $(date) ==="
python -m pipeline.lora_trainer --config examples/usecase1_customer_support/config_sft.json --faqs examples/usecase1_customer_support/faqs.json > before_after_eval/uc1_sft_train.log 2>&1
echo "=== SFT TRAIN B DONE $(date) exit=$? ==="
