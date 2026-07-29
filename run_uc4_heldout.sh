#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/kvforge
source venv/bin/activate
echo "=== UC4 HELDOUT EVAL START $(date) ==="
python -m pipeline.prs_evaluator --config examples/usecase4_bedrock_userguide/config_sft_hi.json --faqs examples/usecase4_bedrock_userguide/eval_heldout_paraphrase.json --sample 50 --checkpoint examples/usecase4_bedrock_userguide/lora_checkpoints_sft_hi/v2/ --skip-version-update > before_after_eval/uc4_heldout_hi.log 2>&1
echo "=== UC4 HELDOUT EVAL DONE $(date) exit=$? ===" >> before_after_eval/run_uc4_heldout.driver.log
