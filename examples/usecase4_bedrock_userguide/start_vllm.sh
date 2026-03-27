#!/usr/bin/env bash
# Start vLLM inference server for UC4 on GPU 3 (port 8090).
#
# vLLM achieves 6-10x higher decode throughput than HuggingFace transformers
# by using CUDA graphs, continuous batching, and PagedAttention.
#
# Usage:
#   cd ~/kvforge && bash examples/usecase4_bedrock_userguide/start_vllm.sh
#
# Logs: examples/usecase4_bedrock_userguide/vllm.log

set -e

UC_DIR="examples/usecase4_bedrock_userguide"
CONFIG="$UC_DIR/config.json"
LOG="$UC_DIR/vllm.log"
PORT=8090
GPU=3

# Paths
MODEL="meta-llama/Llama-3.2-3B-Instruct"
LORA_DIR="$UC_DIR/lora_checkpoints/v3"
LORA_NAME="uc4"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found at $CONFIG"
  exit 1
fi

if [ ! -d "$LORA_DIR" ]; then
  echo "ERROR: LoRA checkpoint not found at $LORA_DIR — run the pipeline first"
  exit 1
fi

command -v python3 &>/dev/null || { echo "ERROR: python3 not found"; exit 1; }
python3 -c "import vllm" 2>/dev/null || { echo "ERROR: vllm not installed (pip install vllm)"; exit 1; }

echo "[UC4 vLLM] starting on GPU $GPU, port $PORT, model=$MODEL, lora=$LORA_NAME" | tee -a "$LOG"

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
