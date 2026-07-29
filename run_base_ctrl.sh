#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/kvforge
source venv/bin/activate
UC=examples/usecase4_bedrock_userguide
echo "=== BASE-CTRL START $(date) ===" >> before_after_eval/run_base_ctrl.driver.log
python -m pipeline.prs_evaluator --config $UC/config_base_ctrl.json --faqs $UC/eval_heldout_paraphrase.json --sample 50 --skip-version-update > before_after_eval/uc4_base_ctrl.log 2>&1
echo "=== BASE-CTRL DONE $(date) exit=$? ===" >> before_after_eval/run_base_ctrl.driver.log
