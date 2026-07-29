#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/kvforge
source venv/bin/activate
for uc in usecase1_customer_support usecase2_pubmedqa usecase3_squad usecase4_bedrock_userguide; do
  echo "=== AFTER: $uc START $(date) ==="
  python -m pipeline.prs_evaluator     --config examples/$uc/config.json     --faqs examples/$uc/eval_subset_50.json     --sample 50     --skip-version-update     > before_after_eval/${uc}_after.log 2>&1
  echo "=== AFTER: $uc DONE $(date) exit=$? ==="
done
echo AFTER_ALL_DONE
