#!/usr/bin/env bash
# start_uc4_dashboard.sh — UC4 fast-path: load existing LoRA weights and start dashboard.
#
# UC4 (Amazon Bedrock User Guide) has pre-trained weights at v3 (PRS 0.82).
# This script skips all pipeline steps and starts the dashboard directly.
#
# Expects:
#   - Running from ~/kvforge/ as working directory
#   - Checkpoint at examples/usecase4_bedrock_userguide/lora_checkpoints/v3/
#   - version.json already written with phase=3 and checkpoint_path set
#   - CUDA_VISIBLE_DEVICES=3 set by caller
#
# Usage (called by deploy_and_run.sh):
#   CUDA_VISIBLE_DEVICES=3 bash examples/usecase4_bedrock_userguide/start_uc4_dashboard.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UC_DIR="$REPO_ROOT/examples/usecase4_bedrock_userguide"
CHECKPOINT_DIR="$UC_DIR/lora_checkpoints/v3"
CONFIG="$UC_DIR/config.json"
LOG="$UC_DIR/dashboard.log"

cd "$REPO_ROOT"

if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo "Error: config not found at $CONFIG" >&2; exit 1
fi

echo "================================================================"
echo " KVForge — Use Case 4: Bedrock User Guide (fast-path)"
echo " Loading existing LoRA v3 checkpoint (PRS 0.82)"
echo "================================================================"

# Verify checkpoint exists before attempting to start
if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "ERROR: UC4 checkpoint not found at $CHECKPOINT_DIR" >&2
  echo "Run deploy_and_run.sh to migrate weights from ~/qdrant/lora_checkpoints/bedrock/v3/" >&2
  exit 1
fi

echo "Checkpoint found: $CHECKPOINT_DIR"

PORT=$(python -c "import json; print(json.load(open('$CONFIG')).get('dashboard_port', 8084))")

echo "Starting KVForge dashboard for Use Case 4: Bedrock User Guide"
echo "  Config  : $CONFIG"
echo "  Port    : $PORT"
echo "  Log     : $LOG"
echo "  URL     : http://localhost:$PORT"
echo ""

exec python -m pipeline.monitoring_dashboard --config "$CONFIG" --port "$PORT" 2>&1 | tee "$LOG"
