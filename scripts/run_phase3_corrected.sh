#!/usr/bin/env bash
set -e
cd /home/ubuntu/kvforge

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0

FIREWORKS_KEY="fw_3ZFs7JStEVYvrQdK7FhwC9qw"
FIREWORKS_URL="https://api.fireworks.ai/inference/v1"
JUDGE_MODEL="accounts/fireworks/models/deepseek-v4-flash"
CFG=examples/usecase4_bedrock_userguide/config.json
DATA_DIR=examples/usecase4_bedrock_userguide
PHASE3_DIR=results/absorption/phase3

rm -rf $PHASE3_DIR
mkdir -p $PHASE3_DIR logs

# The FAQs are the same across tiers (337 FAQs for UC4 Bedrock).
# Each tier trains from scratch with --from-base and saves to its own directory.
TRAIN_FAQS=$DATA_DIR/faqs_train.json

echo "============================================"
echo " Phase 3 — Training + Evaluation (corrected)"
echo "============================================"

for N in 500 1000 2000 4000 6000; do
    CKPT_DIR=$DATA_DIR/lora_checkpoints/phase3_N${N}
    EVAL_SET=$DATA_DIR/eval_tier_${N}.json
    RUN_DIR=$PHASE3_DIR/N${N}_4mode
    mkdir -p "$CKPT_DIR" "$RUN_DIR"

    echo ""
    echo "[$(date)] N${N}: Training from base (fresh LoRA)..."
    ./venv/bin/python3 -u -m pipeline.lora_trainer \
        --config "$CFG" \
        --faqs "$TRAIN_FAQS" \
        --seed 42 \
        --from-base \
        --checkpoint-dir "$CKPT_DIR/" 2>&1 | tail -3

    echo "[$(date)] N${N}: Evaluating text_rag + kv_meanpool + parametric..."
    ./venv/bin/python3 -u -m tools.measure_baseline_fkds \
        --config "$CFG" \
        --eval-set "$EVAL_SET" \
        --checkpoint "$CKPT_DIR" \
        --modes text_rag kv_meanpool parametric \
        --judge-model "$JUDGE_MODEL" \
        --judge-provider openai \
        --judge-api-key "$FIREWORKS_KEY" \
        --judge-base-url "$FIREWORKS_URL" \
        --output-dir "$RUN_DIR" 2>&1 | tail -5

    echo "[$(date)] N${N}: Done"
done

echo ""
echo "============================================"
echo " Phase 3 Results"
echo "============================================"
python3 -c "
import json, glob
print(f'{\"Tier\":<10} {\"text_rag\":<10} {\"kv_meanpool\":<12} {\"parametric\":<10} {\"best\":<12}')
print('-'*55)
for f in sorted(glob.glob('results/absorption/phase3/*/summary.json')):
    label = f.split('/')[-2].replace('_4mode','')
    d=json.load(open(f));m=d.get('modes',{})
    vals={}
    for mode in ['text_rag','kv_meanpool','parametric']:
        vals[mode]=m.get(mode,{}).get('factual_accuracy',{}).get('mean',0)
    best=max(vals,key=vals.get)
    print(f'{label:<10} {vals[\"text_rag\"]:<10.4f} {vals[\"kv_meanpool\"]:<12.4f} {vals[\"parametric\"]:<10.4f} {best:<12}')
" 2>&1 | tee $PHASE3_DIR/aggregate.txt
