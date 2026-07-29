#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/kvforge
source venv/bin/activate
echo "=== UC4 HI-CAP TRAIN START $(date) ==="
python -m pipeline.lora_trainer --config examples/usecase4_bedrock_userguide/config_sft_hi.json --faqs examples/usecase4_bedrock_userguide/faqs.json > before_after_eval/uc4_hi_train.log 2>&1
echo "=== UC4 HI-CAP TRAIN DONE $(date) exit=$? ==="
python -m pipeline.prs_evaluator --config examples/usecase4_bedrock_userguide/config_sft_hi.json --faqs examples/usecase4_bedrock_userguide/eval_subset_50.json --sample 50 --checkpoint examples/usecase4_bedrock_userguide/lora_checkpoints_sft_hi/v2/ --skip-version-update > before_after_eval/uc4_hi_eval.log 2>&1
echo "=== UC4 HI-CAP EVAL DONE $(date) exit=$? ===" >> before_after_eval/run_uc4_hi.driver.log
