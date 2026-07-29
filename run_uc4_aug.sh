#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/kvforge
source venv/bin/activate
UC=examples/usecase4_bedrock_userguide
HELD=$UC/eval_heldout_paraphrase.json
run_arm () {
  local tag=$1 faqs=$2
  echo "=== AUG $tag TRAIN START $(date) ===" >> before_after_eval/run_uc4_aug.driver.log
  python -m pipeline.lora_trainer --config $UC/config_aug_$tag.json --faqs $faqs > before_after_eval/uc4_aug_${tag}_train.log 2>&1
  echo "=== AUG $tag TRAIN DONE $(date) exit=$? ===" >> before_after_eval/run_uc4_aug.driver.log
  local ck=$(ls -d $UC/lora_ckpt_aug_$tag/v*/ 2>/dev/null | sort -V | tail -1)
  echo "AUG $tag checkpoint=$ck" >> before_after_eval/run_uc4_aug.driver.log
  python -m pipeline.prs_evaluator --config $UC/config_aug_$tag.json --faqs $HELD --sample 50 --checkpoint $ck --skip-version-update > before_after_eval/uc4_aug_${tag}_eval.log 2>&1
  echo "=== AUG $tag EVAL DONE $(date) exit=$? ===" >> before_after_eval/run_uc4_aug.driver.log
}
run_arm D $UC/faqs_aug_dupes.json
run_arm P $UC/faqs_aug_paraphrase.json
echo "=== AUG ALL DONE $(date) ===" >> before_after_eval/run_uc4_aug.driver.log
