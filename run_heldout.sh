#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/kvforge
source venv/bin/activate
echo "=== B (chat-SFT) held-out START $(date) ==="
python -m pipeline.prs_evaluator --config examples/usecase1_customer_support/config_sft.json --faqs examples/usecase1_customer_support/eval_heldout_paraphrase.json --sample 15 --checkpoint examples/usecase1_customer_support/lora_checkpoints_sft/v2/ --skip-version-update > before_after_eval/uc1_heldout_B.log 2>&1
echo "=== A (bare) held-out START $(date) ==="
python -m pipeline.prs_evaluator --config examples/usecase1_customer_support/config_bare.json --faqs examples/usecase1_customer_support/eval_heldout_paraphrase.json --sample 15 --checkpoint examples/usecase1_customer_support/lora_checkpoints/v1/ --skip-version-update > before_after_eval/uc1_heldout_A.log 2>&1
echo "=== HELDOUT DONE $(date) ===" >> before_after_eval/run_heldout.driver.log
