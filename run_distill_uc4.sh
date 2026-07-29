#!/bin/bash
# Queue Sprint 2 distillation for UC4 on EC2 g5.xlarge.
# Waits for the current GPU job (PID 513545) to exit, then runs distillation.
set -euo pipefail

KWFORGE=/home/ubuntu/kvforge
VENV=$KWFORGE/venv_gpu/bin
CONFIG=$KWFORGE/examples/usecase4_bedrock_userguide/config_distill.json
DISTILL_PAIRS=$KWFORGE/examples/usecase4_bedrock_userguide/distill_pairs_v1.json
LOG=$KWFORGE/logs/distill_uc4_v1_$(date +%Y%m%d_%H%M%S).log
WAIT_PID=513545

echo "[$(date)] Waiting for PID ${WAIT_PID} to exit..."
while kill -0 ${WAIT_PID} 2>/dev/null; do
    sleep 30
done
echo "[$(date)] GPU freed. Starting distillation..." | tee "$LOG"

cd $KWFORGE
$VENV/python -u $KWFORGE/pipeline/lora_trainer.py \
    --config "$CONFIG" \
    --distill-pairs "$DISTILL_PAIRS" \
    --replay-ratio 0.5 \
    --from-base \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Distillation complete." | tee -a "$LOG"
ls -la /home/ubuntu/kvforge/examples/usecase4_bedrock_userguide/lora_checkpoints/ | tee -a "$LOG"
