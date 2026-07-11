#!/usr/bin/env bash
# Start 4 vLLM instances for UC4, one per GPU, with a round-robin router on port 8090.
#
# Layout:
#   GPU 0  →  port 8091  (worker)
#   GPU 1  →  port 8092  (worker)
#   GPU 2  →  port 8093  (worker)
#   GPU 3  →  port 8094  (worker, shares GPU with monitoring dashboard)
#   Router →  port 8090  (what UC4 config points to)
#
# Stop with: ./scripts/stop_vllm_pool.sh

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
LORA_PATH="examples/usecase4_bedrock_userguide/lora_checkpoints/v1"
LOG_DIR="logs/vllm_pool"
PYTHON="$(pwd)/venv/bin/python"
GPU_MEM=0.82      # GPU 3 shares with monitoring dashboard (~3GB used)
MAX_LEN=4096
PORTS=(8091 8092 8093 8094)

mkdir -p "$LOG_DIR"

# ── Stop anything already on pool ports ──────────────────────────────────────
for port in "${PORTS[@]}" 8090; do
    pid=$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    if [[ -n "$pid" ]]; then
        echo "Killing existing process on port $port (PID $pid)"
        kill "$pid" 2>/dev/null || true
    fi
done
sleep 3

# ── Start 4 worker vLLM instances ────────────────────────────────────────────
for i in 0 1 2 3; do
    port="${PORTS[$i]}"
    log="$LOG_DIR/worker_gpu${i}.log"
    echo "[GPU $i] starting worker on port $port → $log"

    CUDA_VISIBLE_DEVICES=$i \
    nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model               "$BASE_MODEL" \
        --served-model-name   "uc4" \
        --port                "$port" \
        --gpu-memory-utilization "$GPU_MEM" \
        --max-model-len       "$MAX_LEN" \
        --tensor-parallel-size 1 \
        --dtype               float16 \
        --enable-lora \
        --lora-modules        "uc4=${LORA_PATH}" \
        > "$log" 2>&1 &

    echo $! > "$LOG_DIR/worker_gpu${i}.pid"
    echo "[GPU $i] PID $!"
done

# ── Wait for all workers to be healthy ───────────────────────────────────────
echo ""
echo "Waiting for all 4 workers (CUDA graph compilation may take 2-5 min)..."
for port in "${PORTS[@]}"; do
    echo -n "  port $port "
    deadline=$((SECONDS + 600))
    while [[ $SECONDS -lt $deadline ]]; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "✓"
            break
        fi
        echo -n "."
        sleep 10
    done
    if [[ $SECONDS -ge $deadline ]]; then
        echo " TIMEOUT — check $LOG_DIR/worker_gpu*.log"
        exit 1
    fi
done

# ── Start router on port 8090 ─────────────────────────────────────────────────
router_log="$LOG_DIR/router.log"
echo ""
echo "Starting round-robin router on port 8090 → $router_log"

nohup "$PYTHON" scripts/vllm_router.py \
    > "$router_log" 2>&1 &
echo $! > "$LOG_DIR/router.pid"
echo "Router PID $!"

# Brief wait for router to bind
sleep 3
if curl -sf http://localhost:8090/health > /dev/null 2>&1; then
    echo "Router healthy ✓"
else
    echo "Router not yet responding — check $router_log"
fi

echo ""
echo "Pool ready. UC4 dashboard (port 8084) will route through 8090 → 8091-8094."
echo "Stop with: ./scripts/stop_vllm_pool.sh"
