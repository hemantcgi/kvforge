#!/usr/bin/env bash
# Stop all vLLM pool workers and router started by start_vllm_pool.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs/vllm_pool"

stop_pid() {
    local label="$1" pidfile="$2"
    if [[ -f "$pidfile" ]]; then
        local pid; pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $label (PID $pid)"
            kill "$pid" 2>/dev/null || true
        else
            echo "$label (PID $pid) already stopped"
        fi
        rm -f "$pidfile"
    else
        echo "No pidfile for $label"
    fi
}

stop_pid "router"    "$LOG_DIR/router.pid"
stop_pid "worker_gpu0" "$LOG_DIR/worker_gpu0.pid"
stop_pid "worker_gpu1" "$LOG_DIR/worker_gpu1.pid"
stop_pid "worker_gpu2" "$LOG_DIR/worker_gpu2.pid"
stop_pid "worker_gpu3" "$LOG_DIR/worker_gpu3.pid"

# Kill any orphaned vLLM processes
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null && \
    echo "Killed orphaned vllm.entrypoints processes" || true
pkill -f 'vllm_router.py' 2>/dev/null && \
    echo "Killed orphaned router process" || true

echo "Done."
