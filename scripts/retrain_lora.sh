#!/usr/bin/env bash
# Two-pass LoRA retraining for UC4 (Amazon Bedrock User Guide):
#
#   Pass 1 — Raw chunk mode: trains on all indexed Qdrant chunks
#             → teaches the model the document's domain content
#
#   Pass 2 — Instruction mode: fine-tunes on 1,027 Q&A pairs
#             → aligns model output with the expected answer format
#
# GPUs: training uses GPU 0 (single-GPU 4-bit, ~18GB).
#       GPUs 1-3 stay idle during training.
#
# After both passes the script:
#   - updates start_vllm_pool.sh to point at the new checkpoint
#   - restarts the vLLM pool
#
# Run from repo root:
#   bash scripts/retrain_lora.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(pwd)/venv/bin/python"
CONFIG="examples/usecase4_bedrock_userguide/config.json"
FAQS="examples/usecase4_bedrock_userguide/faqs_train.json"
SOURCE_FILE="Amazon Bedrock Dataset.pdf"
LOG_DIR="logs/lora_train"

mkdir -p "$LOG_DIR"

# ── Step 1: Stop vLLM pool ────────────────────────────────────────────────────
echo "=== [1/5] Stopping vLLM pool ==="
bash scripts/stop_vllm_pool.sh || true
sleep 5
echo ""

# ── Step 2: Pass 1 — raw chunk training ──────────────────────────────────────
echo "=== [2/5] Pass 1: Raw chunk training on all Qdrant chunks ==="
echo "    Source: '$SOURCE_FILE'  |  config: $CONFIG"
echo "    Logs: $LOG_DIR/pass1.log"
echo ""

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m pipeline.lora_trainer \
    --config      "$CONFIG" \
    --source-file "$SOURCE_FILE" \
    2>&1 | tee "$LOG_DIR/pass1.log"

echo ""
echo "✅ Pass 1 complete."

# Capture new checkpoint path from version.json
CKPT_AFTER_P1=$(python3 -c "
import json
v = json.load(open('$CONFIG'))
import json as j2
ver = j2.load(open(v.get('version_file','examples/usecase4_bedrock_userguide/version.json')))
print(ver.get('checkpoint_path',''))
")
echo "    Checkpoint: $CKPT_AFTER_P1"
echo ""

# ── Step 3: Pass 2 — instruction fine-tuning ─────────────────────────────────
echo "=== [3/5] Pass 2: Instruction fine-tuning on $FAQS ==="
echo "    Logs: $LOG_DIR/pass2.log"
echo ""

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m pipeline.lora_trainer \
    --config "$CONFIG" \
    --faqs   "$FAQS" \
    2>&1 | tee "$LOG_DIR/pass2.log"

echo ""
echo "✅ Pass 2 complete."

# Capture final checkpoint path
FINAL_CKPT=$(python3 -c "
import json
cfg = json.load(open('$CONFIG'))
ver = json.load(open(cfg.get('version_file','examples/usecase4_bedrock_userguide/version.json')))
print(ver.get('checkpoint_path',''))
")
echo "    Final checkpoint: $FINAL_CKPT"
echo ""

# ── Step 4: Update start_vllm_pool.sh to new checkpoint ──────────────────────
echo "=== [4/5] Updating start_vllm_pool.sh → $FINAL_CKPT ==="
# Replace the LORA_PATH line
sed -i "s|^LORA_PATH=.*|LORA_PATH=\"${FINAL_CKPT}\"|" scripts/start_vllm_pool.sh
grep "^LORA_PATH=" scripts/start_vllm_pool.sh
echo ""

# ── Step 5: Restart vLLM pool ─────────────────────────────────────────────────
echo "=== [5/5] Restarting vLLM pool with new LoRA checkpoint ==="
bash scripts/start_vllm_pool.sh

echo ""
echo "================================================================"
echo " LoRA retraining complete!"
echo " New checkpoint: $FINAL_CKPT"
echo ""
echo " Next step — evaluate PRS:"
echo "   python -m pipeline.prs_evaluator \\"
echo "     --config $CONFIG \\"
echo "     --faqs examples/usecase4_bedrock_userguide/faqs.json \\"
echo "     --sample 30"
echo "================================================================"
