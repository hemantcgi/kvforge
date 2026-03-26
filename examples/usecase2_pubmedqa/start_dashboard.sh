#!/usr/bin/env bash
# start_dashboard.sh — Start the KVForge dashboard for Use Case 2: Biomedical Q&A (PubMedQA)
#
# Loads the pre-trained LoRA checkpoint and KV tensors for this corpus
# and starts the monitoring dashboard on port 8082.
#
# Usage (from repo root):
#   bash examples/usecase2_pubmedqa/start_dashboard.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UC_DIR="$REPO_ROOT/examples/usecase2_pubmedqa"
CONFIG="$UC_DIR/config.json"
LOG="$UC_DIR/dashboard.log"

cd "$REPO_ROOT"

if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo "Error: config not found at $CONFIG" >&2; exit 1
fi

if [ ! -d "$UC_DIR/lora_checkpoints" ]; then
  echo "Warning: No LoRA checkpoint found — dashboard will run in Phase 1 (retrieval only)."
  echo "Run the full pipeline first: bash examples/usecase2_pubmedqa/run_pipeline.sh"
fi

PORT=$(python -c "import json; print(json.load(open('$CONFIG')).get('dashboard_port', 8082))")

echo "Starting KVForge dashboard for Use Case 2: Biomedical Q&A (PubMedQA)"
echo "  Config  : $CONFIG"
echo "  Port    : $PORT"
echo "  Log     : $UC_DIR/dashboard.log"
echo "  URL     : http://localhost:$PORT"
echo ""

exec python -m pipeline.monitoring_dashboard --config "$CONFIG" --port "$PORT" 2>&1 | tee "$LOG"
