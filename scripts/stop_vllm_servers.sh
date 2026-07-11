#!/usr/bin/env bash
# Stop all vLLM servers started by start_vllm_servers.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs/vllm"
FILTER="${1:-all}"

stop_uc() {
    local uc_name="$1"
    local pidfile="$LOG_DIR/${uc_name}.pid"

    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[${uc_name}] stopping PID $pid"
            kill "$pid"
            rm -f "$pidfile"
        else
            echo "[${uc_name}] PID $pid not running (already stopped)"
            rm -f "$pidfile"
        fi
    else
        echo "[${uc_name}] no pidfile found"
    fi
}

[[ "$FILTER" == "all" || "$FILTER" == "uc1" ]] && stop_uc uc1
[[ "$FILTER" == "all" || "$FILTER" == "uc2" ]] && stop_uc uc2
[[ "$FILTER" == "all" || "$FILTER" == "uc3" ]] && stop_uc uc3
[[ "$FILTER" == "all" || "$FILTER" == "uc4" ]] && stop_uc uc4

# Belt-and-suspenders: kill any orphaned vLLM workers
if [[ "$FILTER" == "all" ]]; then
    pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null && \
        echo "Killed remaining vllm.entrypoints processes" || true
fi

echo "Done."
