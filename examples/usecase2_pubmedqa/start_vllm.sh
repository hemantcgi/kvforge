#!/usr/bin/env bash
# Start vLLM inference server for UC2 on GPU 1 (port 8092).
#
# Reads the latest LoRA checkpoint path from version.json so it works
# regardless of which training round last completed.
#
# Usage:
#   cd ~/kvforge && bash examples/usecase2_pubmedqa/start_vllm.sh
#
# Logs: examples/usecase2_pubmedqa/vllm.log

set -e

UC_DIR="examples/usecase2_pubmedqa"
CONFIG="$UC_DIR/config.json"
VERSION_FILE="$UC_DIR/version.json"
LOG="$UC_DIR/vllm.log"
PORT=8092
GPU=1
MODEL="meta-llama/Llama-3.2-3B-Instruct"
LORA_NAME="uc2"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found at $CONFIG"; exit 1
fi

# Resolve LoRA checkpoint: prefer version.json, fall back to latest versioned dir
if [ -f "$VERSION_FILE" ]; then
  LORA_DIR=$(python3 -c "import json; v=json.load(open('$VERSION_FILE')); print(v.get('checkpoint_path',''))" 2>/dev/null || true)
fi

if [ -z "$LORA_DIR" ] || [ ! -d "$LORA_DIR" ]; then
  # Scan for latest vN directory
  LORA_DIR=$(ls -d "$UC_DIR/lora_checkpoints/v"* 2>/dev/null | sort -V | tail -1 || true)
fi

if [ -z "$LORA_DIR" ] || [ ! -d "$LORA_DIR" ]; then
  echo "ERROR: No LoRA checkpoint found — run the pipeline first"
  echo "  Checked version.json and $UC_DIR/lora_checkpoints/v*"
  exit 1
fi

command -v python3 &>/dev/null || { echo "ERROR: python3 not found"; exit 1; }
python3 -c "import vllm" 2>/dev/null || { echo "ERROR: vllm not installed (pip install vllm)"; exit 1; }

echo "[UC2 vLLM] starting on GPU $GPU, port $PORT, model=$MODEL, lora=$LORA_NAME" | tee -a "$LOG"
echo "[UC2 vLLM] checkpoint: $LORA_DIR" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --enable-lora \
  --lora-modules "${LORA_NAME}=${LORA_DIR}" \
  --max-lora-rank 16 \
  --port "$PORT" \
  --host "0.0.0.0" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --dtype float16 \
  2>&1 | tee -a "$LOG"
