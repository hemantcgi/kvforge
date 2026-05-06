#!/usr/bin/env bash
# Full FAQ generation pipeline on EC2:
#   1. Stop Llama 3.2-3B vLLM pool (frees all 4 GPUs)
#   2. Start 4× Qwen3-4B workers (one per GPU) + round-robin router on 8090
#   3. Pull chunks from Qdrant bedrock-user-guide collection
#   4. Generate 2000 Q&A pairs via generate_large_faqs.py
#   5. Stop Qwen3-4B pool
#   6. Restart original Llama 3.2-3B pool for serving/training
#
# Run from repo root:
#   bash scripts/generate_faqs_ec2.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(pwd)/venv/bin/python"
LOG_DIR="logs/faq_gen"
QWEN_MODEL="Qwen/Qwen3-4B"
PORTS=(8091 8092 8093 8094)
GPU_MEM=0.90   # 7.6GB model on 22GB GPU — plenty of headroom
MAX_LEN=4096
OUTPUT="examples/usecase4_bedrock_userguide/faqs_train.json"
TARGET=2000
N_PER_CHUNK=4
CONCURRENCY=16   # 4 GPUs, 4 concurrent each

mkdir -p "$LOG_DIR"

# ── Step 1: Stop existing Llama pool ────────────────────────────────────────
echo "=== [1/6] Stopping Llama 3.2-3B vLLM pool ==="
bash scripts/stop_vllm_pool.sh || true
sleep 5

# ── Step 2: Start 4× Qwen3-4B (one per GPU) ─────────────────────────────────
echo ""
echo "=== [2/6] Starting 4× Qwen3-4B workers ==="
for i in 0 1 2 3; do
    port="${PORTS[$i]}"
    log="$LOG_DIR/qwen_gpu${i}.log"
    # GPU3 shares with monitoring dashboard (~2.6GB), needs lower util
    mem=$( [[ $i -eq 3 ]] && echo "0.82" || echo "$GPU_MEM" )
    echo "[GPU $i] port $port  gpu_mem=$mem → $log"
    CUDA_VISIBLE_DEVICES=$i \
    nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model               "$QWEN_MODEL" \
        --served-model-name   "gen" \
        --port                "$port" \
        --gpu-memory-utilization "$mem" \
        --max-model-len       "$MAX_LEN" \
        --tensor-parallel-size 1 \
        --dtype               bfloat16 \
        > "$log" 2>&1 &
    echo $! > "$LOG_DIR/qwen_gpu${i}.pid"
    echo "[GPU $i] PID $!"
done

echo ""
echo "Waiting for all 4 Qwen3-4B workers to be healthy (may take 3-5 min) ..."
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
        echo " TIMEOUT — check $LOG_DIR/qwen_gpu*.log"
        exit 1
    fi
done

# ── Step 3: Start router on 8090 ─────────────────────────────────────────────
echo ""
echo "=== [3/6] Starting round-robin router on 8090 ==="
nohup "$PYTHON" scripts/vllm_router.py \
    > "$LOG_DIR/router.log" 2>&1 &
echo $! > "$LOG_DIR/router.pid"
sleep 3
if curl -sf http://localhost:8090/health > /dev/null 2>&1; then
    echo "Router healthy ✓"
else
    echo "Router not responding — check $LOG_DIR/router.log"
    exit 1
fi

# ── Step 4: Generate Q&A from Qdrant chunks ──────────────────────────────────
echo ""
echo "=== [4/6] Generating $TARGET Q&A pairs from Qdrant (bedrock-user-guide) ==="
echo "    Model: $QWEN_MODEL | Concurrency: $CONCURRENCY | N/chunk: $N_PER_CHUNK"
echo ""

"$PYTHON" scripts/generate_large_faqs.py \
    --provider     vllm \
    --model        gen \
    --vllm-base    http://localhost:8090/v1 \
    --qdrant-source \
    --qdrant-collection bedrock-user-guide \
    --output       "$OUTPUT" \
    --count        "$TARGET" \
    --n-per-chunk  "$N_PER_CHUNK" \
    --concurrency  "$CONCURRENCY" \
    --save-every   50

echo ""
echo "=== Q&A generation complete ==="
python3 -c "
import json
d = json.load(open('$OUTPUT'))
print(f'  Total records: {len(d)}')
print(f'  Sample Q: {d[0][\"question\"][:100]}')
"

# ── Step 5: Stop Qwen3-4B pool ───────────────────────────────────────────────
echo ""
echo "=== [5/6] Stopping Qwen3-4B pool ==="
for f in "$LOG_DIR"/qwen_gpu*.pid "$LOG_DIR"/router.pid; do
    [[ -f "$f" ]] && kill "$(cat "$f")" 2>/dev/null || true
    rm -f "$f"
done
pkill -f "Qwen3-4B\|Qwen/Qwen3" 2>/dev/null || true
sleep 5

# ── Step 6: Restart original Llama 3.2-3B pool ───────────────────────────────
echo ""
echo "=== [6/6] Restarting Llama 3.2-3B vLLM pool ==="
bash scripts/start_vllm_pool.sh

echo ""
echo "================================================================"
echo " Done! FAQs at: $OUTPUT"
echo " Next step — retrain LoRA:"
echo "   python -m pipeline.lora_trainer \\"
echo "     --config examples/usecase4_bedrock_userguide/config.json \\"
echo "     --source-file 'Amazon Bedrock Dataset.pdf'"
echo ""
echo "   python -m pipeline.lora_trainer \\"
echo "     --config examples/usecase4_bedrock_userguide/config.json \\"
echo "     --faqs $OUTPUT"
echo "================================================================"
