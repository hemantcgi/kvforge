#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/kvforge
source venv/bin/activate
echo "=== UC4 chat-SFT TRAIN START $(date) ==="
python -m pipeline.lora_trainer --config examples/usecase4_bedrock_userguide/config_sft.json --faqs examples/usecase4_bedrock_userguide/faqs.json > before_after_eval/uc4_sft_train.log 2>&1
echo "=== UC4 TRAIN DONE $(date) exit=$? ==="
echo "=== UC4 chat-SFT EVAL START $(date) ==="
python -m pipeline.prs_evaluator --config examples/usecase4_bedrock_userguide/config_sft.json --faqs examples/usecase4_bedrock_userguide/eval_subset_50.json --sample 50 --checkpoint examples/usecase4_bedrock_userguide/lora_checkpoints_sft/v2/ --skip-version-update > before_after_eval/uc4_sft_eval.log 2>&1
echo "=== UC4 EVAL DONE $(date) exit=$? ===" >> before_after_eval/run_uc4_sft.driver.log
