#!/usr/bin/env bash
# Start 4 independent vLLM servers — one per A10G GPU, one per use-case.
#
# Layout:
#   GPU 0  →  UC1 customer_support   port 8091  LoRA: uc1
#   GPU 1  →  UC2 pubmedqa           port 8092  LoRA: uc2
#   GPU 2  →  UC3 squad              port 8093  LoRA: uc3
#   GPU 3  →  UC4 bedrock_userguide  port 8090  LoRA: uc4
#
# Each UC config already sets vllm_model to the matching LoRA module name,
# so monitoring dashboards and kv_inference pick up the right endpoint.
#
# Usage:
#   ./scripts/start_vllm_servers.sh          # start all
#   ./scripts/start_vllm_servers.sh uc1      # start one UC
#   ./scripts/stop_vllm_servers.sh           # stop all

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
GPU_MEM=0.85
MAX_LEN=4096
LOG_DIR="logs/vllm"
PYTHON="$(pwd)/venv/bin/python"

mkdir -p "$LOG_DIR"

start_uc() {
    local uc_name="$1"      # e.g. uc1
    local gpu_id="$2"       # e.g. 0
    local port="$3"         # e.g. 8091
    local lora_path="$4"    # e.g. examples/usecase1.../lora_checkpoints/v1

    local log="$LOG_DIR/vllm_${uc_name}.log"

    if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
        echo "[${uc_name}] port ${port} already in use — skipping"
        return
    fi

    echo "[${uc_name}] starting on GPU ${gpu_id}, port ${port} → ${log}"

    CUDA_VISIBLE_DEVICES=$gpu_id \
    nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model               "$BASE_MODEL" \
        --served-model-name   "$uc_name" \
        --port                "$port" \
        --gpu-memory-utilization "$GPU_MEM" \
        --max-model-len       "$MAX_LEN" \
        --tensor-parallel-size 1 \
        --dtype               float16 \
        --enforce-eager \
        --enable-lora \
        --lora-modules        "${uc_name}=${lora_path}" \
        > "$log" 2>&1 &

    echo $! > "$LOG_DIR/${uc_name}.pid"
    echo "[${uc_name}] PID $!"
}

FILTER="${1:-all}"

[[ "$FILTER" == "all" || "$FILTER" == "uc1" ]] && \
    start_uc uc1 0 8091 "examples/usecase1_customer_support/lora_checkpoints/v1"

[[ "$FILTER" == "all" || "$FILTER" == "uc2" ]] && \
    start_uc uc2 1 8092 "examples/usecase2_pubmedqa/lora_checkpoints/v1"

[[ "$FILTER" == "all" || "$FILTER" == "uc3" ]] && \
    start_uc uc3 2 8093 "examples/usecase3_squad/lora_checkpoints/v1"

[[ "$FILTER" == "all" || "$FILTER" == "uc4" ]] && \
    start_uc uc4 3 8090 "examples/usecase4_bedrock_userguide/lora_checkpoints/v1"

echo ""
echo "Servers starting. Check health with:"
echo "  curl http://localhost:8091/health  # uc1"
echo "  curl http://localhost:8092/health  # uc2"
echo "  curl http://localhost:8093/health  # uc3"
echo "  curl http://localhost:8090/health  # uc4"
echo ""
echo "Logs: $LOG_DIR/vllm_uc{1..4}.log"
echo "Stop: ./scripts/stop_vllm_servers.sh"
