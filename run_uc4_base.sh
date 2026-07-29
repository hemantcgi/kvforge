#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/kvforge
source venv/bin/activate
UC=examples/usecase4_bedrock_userguide
echo "=== BASE (no-adapter) HELDOUT START $(date) ===" >> before_after_eval/run_uc4_base.driver.log
python -m pipeline.prs_evaluator --config $UC/config_aug_P.json --faqs $UC/eval_heldout_paraphrase.json --sample 50 --skip-version-update > before_after_eval/uc4_base_heldout.log 2>&1
echo "=== BASE HELDOUT DONE $(date) exit=$? ===" >> before_after_eval/run_uc4_base.driver.log
