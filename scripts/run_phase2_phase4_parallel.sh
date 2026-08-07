#!/usr/bin/env bash
set -e
cd /home/ubuntu/kvforge

export PYTHONUNBUFFERED=1
FIREWORKS_KEY="fw_3ZFs7JStEVYvrQdK7FhwC9qw"
FIREWORKS_URL="https://api.fireworks.ai/inference/v1"
JUDGE_MODEL="accounts/fireworks/models/deepseek-v4-flash"

mkdir -p results/absorption/phase2 results/absorption/phase4 logs

# ============================================================
# GPU0: Phase 2 re-run (2WikiMQA + HotpotQA, corrected LoRA)
# ============================================================
run_phase2() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[GPU0] Phase 2 re-run with corrected LoRA targets"

    for DS in longbench_2wikimqa longbench_hotpotqa; do
        case $DS in
            longbench_2wikimqa) LABEL="2wikimqa" ;;
            longbench_hotpotqa) LABEL="hotpotqa" ;;
        esac
        TOTAL=$(python3 -c "import json; print(len(json.load(open('examples/$DS/chunks.json'))))")

        for N in 500 1000 $TOTAL; do
            EVAL_SET="examples/$DS/eval_tier_${N}.json"
            CKPT_DIR="examples/$DS/lora_checkpoints/phase2_N${N}"
            mkdir -p "$CKPT_DIR"

            for SEED in 42 43 44; do
                RUN_LABEL="${LABEL}_N${N}_seed${SEED}"
                RUN_DIR="results/absorption/phase2/${RUN_LABEL}"
                [ -f "$RUN_DIR/summary.json" ] && echo "  [GPU0] $RUN_LABEL: already done" && continue
                mkdir -p "$RUN_DIR"

                echo "  [GPU0][$(date)] $RUN_LABEL: Training from base..."
                ./venv/bin/python3 -u -m pipeline.lora_trainer \
                    --config examples/$DS/config.json \
                    --faqs "examples/$DS/faqs_500.json" \
                    --seed $SEED --from-base \
                    --checkpoint-dir "$CKPT_DIR/" 2>&1 | tail -2

                echo "  [GPU0][$(date)] $RUN_LABEL: Evaluating (text_rag + parametric)..."
                ./venv/bin/python3 -u -m tools.measure_baseline_fkds \
                    --config examples/$DS/config.json \
                    --eval-set "$EVAL_SET" \
                    --checkpoint "$CKPT_DIR" \
                    --modes text_rag parametric \
                    --judge-model "$JUDGE_MODEL" --judge-provider openai \
                    --judge-api-key "$FIREWORKS_KEY" --judge-base-url "$FIREWORKS_URL" \
                    --output-dir "$RUN_DIR" 2>&1 | tail -3

                echo "  [GPU0][$(date)] $RUN_LABEL: Done"
            done
        done
    done
    echo "[GPU0] Phase 2 re-run COMPLETE"
}

# ============================================================
# GPU1: Phase 4 — Embedding ablation
# ============================================================
run_phase4() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[GPU1] Phase 4: Embedding ablation"

    BASE_CFG=examples/usecase4_bedrock_userguide/config.json
    EVAL_SET=examples/usecase4_bedrock_userguide/eval_heldout.json
    BASE_COLL=bedrock-userguide

    for EMBED in "BAAI/bge-large-en-v1.5" "mixedbread-ai/mxbai-embed-large-v1"; do
        EMBED_NAME=$(echo $EMBED | tr '/' '_')
        COLL="${BASE_COLL}_${EMBED_NAME}"
        DIM=1024

        echo "  [GPU1][$(date)] Re-indexing with $EMBED ($DIM-dim)..."
        python3 -c "
import json; cfg=json.load(open('$BASE_CFG'))
cfg['embed_model']='$EMBED'; cfg['vector_dim']=$DIM; cfg['collection']='$COLL'
cfg.setdefault('addon_config',{}).setdefault('indexing',{})['embed_model']='$EMBED'
cfg['addon_config']['indexing']['vector_dim']=$DIM
json.dump(cfg, open('/tmp/cfg_$EMBED_NAME.json','w'), indent=2)
"

        curl -s -X DELETE "http://localhost:6333/collections/$COLL" > /dev/null 2>&1
        ./venv/bin/python3 -m pipeline.kv_indexer \
            --config /tmp/cfg_$EMBED_NAME.json \
            index 2>&1 | tail -2

        RUN_DIR="results/absorption/phase4/${EMBED_NAME}"
        mkdir -p "$RUN_DIR"

        echo "  [GPU1][$(date)] Evaluating text_rag with $EMBED..."
        ./venv/bin/python3 -u -m tools.measure_baseline_fkds \
            --config /tmp/cfg_$EMBED_NAME.json \
            --eval-set "$EVAL_SET" \
            --modes text_rag \
            --judge-model "$JUDGE_MODEL" --judge-provider openai \
            --judge-api-key "$FIREWORKS_KEY" --judge-base-url "$FIREWORKS_URL" \
            --output-dir "$RUN_DIR" 2>&1 | tail -3
    done

    echo "[GPU1] Phase 4 COMPLETE"
}

# Launch both in parallel
run_phase2 > logs/phase2_corrected.log 2>&1 &
PID2=$!
echo "Phase 2 PID: $PID2 (GPU0)"

run_phase4 > logs/phase4.log 2>&1 &
PID4=$!
echo "Phase 4 PID: $PID4 (GPU1)"

wait $PID2 $PID4
echo ""
echo "============================================"
echo " ALL DONE"
echo "============================================"
