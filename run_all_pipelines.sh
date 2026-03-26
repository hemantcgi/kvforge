#!/usr/bin/env bash
# run_all_pipelines.sh — Run all 3 KVForge use-case pipelines SEQUENTIALLY.
#
# Runs UC1 → UC2 → UC3 one at a time to avoid GPU OOM on single-GPU machines.
# UC4 (Bedrock) is assumed to be already trained; only its dashboard is started.
#
# Each pipeline writes model weights + KV tensors to its own use-case folder.
# After all pipelines complete, all 4 dashboards are started.
#
# Usage (from repo root, on GPU machine):
#   bash run_all_pipelines.sh
#
# Requirements:
#   - Qdrant running on localhost:6333
#   - HF_TOKEN set for gated model access
#   - GPU with ≥ 16 GB VRAM
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/venv/bin/python3"
LOGS="${REPO_ROOT}/logs"
mkdir -p "$LOGS"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ ! -f "$PYTHON" ]; then
  PYTHON="$(command -v python3 || command -v python)"
fi

log "Using python: $PYTHON"
log "Logs: $LOGS"

# ── Helper: run one full pipeline ─────────────────────────────────────────────
run_pipeline() {
  local uc_num="$1"
  local dir="$2"
  local name="$3"
  local CONFIG="examples/$dir/config.json"
  local FAQS="examples/$dir/faqs.json"
  local LOG="$LOGS/pipeline_uc${uc_num}.log"

  log "════════════════════════════════════════"
  log "UC${uc_num}: $name"
  log "════════════════════════════════════════"

  # Step 0: Download / prepare dataset
  if [ -f "examples/$dir/setup.py" ]; then
    log "[UC${uc_num}] Preparing dataset..."
    "$PYTHON" "examples/$dir/setup.py" 2>&1 | tee -a "$LOG"
  fi

  # Step 1: Index corpus
  local corpus
  corpus="$(ls examples/$dir/data/*.jsonl 2>/dev/null | head -1 || true)"
  if [ -z "$corpus" ]; then
    log "[UC${uc_num}] ERROR: no corpus.jsonl found in examples/$dir/data/" >&2
    return 1
  fi
  log "[UC${uc_num}] Indexing corpus: $corpus"
  "$PYTHON" kvforge.py index --config "$CONFIG" --source "$corpus" 2>&1 | tee -a "$LOG"

  # Step 2: Compute KV tensors
  log "[UC${uc_num}] Computing KV tensors (GPU)..."
  "$PYTHON" -m pipeline.kv_indexer --config "$CONFIG" compute-kv 2>&1 | tee -a "$LOG"

  # Step 3: Generate FAQs (if not present)
  if [ ! -f "$FAQS" ]; then
    log "[UC${uc_num}] Generating FAQs..."
    "$PYTHON" tools/generate_faqs.py \
      --config "$CONFIG" --count 50 --output "$FAQS" 2>&1 | tee -a "$LOG"
  else
    log "[UC${uc_num}] FAQs already exist at $FAQS, skipping."
  fi

  # Step 4: LoRA fine-tuning
  log "[UC${uc_num}] Fine-tuning LoRA (GPU)..."
  "$PYTHON" -m pipeline.lora_trainer --config "$CONFIG" --faqs "$FAQS" 2>&1 | tee -a "$LOG"

  # Step 5: Recompute KV with updated weights
  log "[UC${uc_num}] Recomputing KV tensors with updated LoRA weights..."
  "$PYTHON" -m pipeline.kv_indexer --config "$CONFIG" compute-kv 2>&1 | tee -a "$LOG"

  # Step 6: PRS evaluation
  log "[UC${uc_num}] Evaluating PRS..."
  "$PYTHON" -m pipeline.prs_evaluator \
    --config "$CONFIG" --faqs "$FAQS" --sample 30 2>&1 | tee -a "$LOG"

  log "[UC${uc_num}] ✓ Pipeline complete. Checkpoint: examples/$dir/lora_checkpoints/"
}

# ── Run pipelines sequentially ────────────────────────────────────────────────
run_pipeline 1 "usecase1_customer_support" "Customer Support Q&A"
run_pipeline 2 "usecase2_pubmedqa"         "Biomedical Q&A (PubMedQA)"
run_pipeline 3 "usecase3_squad"            "Reading Comprehension (SQuAD v2)"

log ""
log "════════════════════════════════════════"
log "All pipelines complete. Starting dashboards..."
log "════════════════════════════════════════"

# ── Start all 4 dashboards ────────────────────────────────────────────────────
start_dashboard() {
  local dir="$1"
  local port="$2"
  local logfile="$LOGS/dashboard_$(basename $dir).log"
  nohup "$PYTHON" -m pipeline.monitoring_dashboard \
    --config "examples/$dir/config.json" \
    --port "$port" > "$logfile" 2>&1 &
  log "Dashboard started: examples/$dir → http://localhost:$port  (log: $logfile)"
}

start_dashboard "usecase1_customer_support"   8081
start_dashboard "usecase2_pubmedqa"           8082
start_dashboard "usecase3_squad"              8083
start_dashboard "usecase4_bedrock_userguide"  8084

# Portal at 8080
nohup "$PYTHON" kvforge_portal.py --port 8080 > "$LOGS/portal.log" 2>&1 &
log "KVForge portal started: http://localhost:8080"

log ""
log "All services running. Access the main portal at: http://$(hostname -I | awk '{print $1}'):8080"
