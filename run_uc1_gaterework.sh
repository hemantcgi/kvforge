#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/kvforge
source venv/bin/activate
echo "=== UC1 gate-rework spot-check START $(date) ==="
python -m pipeline.prs_evaluator --config examples/usecase1_customer_support/config.json --faqs examples/usecase1_customer_support/eval_subset_50.json --sample 50 --skip-version-update > before_after_eval/usecase1_gaterework.log 2>&1
echo "=== DONE $(date) exit=$? ==="
