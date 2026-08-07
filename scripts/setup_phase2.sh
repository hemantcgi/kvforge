#!/usr/bin/env bash
set -e
cd /home/ubuntu/kvforge

export OPENAI_API_KEY='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'

echo "============================================"
echo " Phase 2 Setup — $(date)"
echo "============================================"

declare -A DATASETS
DATASETS[usecase3_squad]="squad"
DATASETS[longbench_2wikimqa]="2wikimqa"
DATASETS[longbench_hotpotqa]="hotpotqa"

# 1. Extract chunks + embeddings from Qdrant
for DS in "${!DATASETS[@]}"; do
    COLL=${DATASETS[$DS]}
    echo "[$COLL] Extracting chunks + embeddings from Qdrant..."

    # Build a mapping from point_id to chunk (for LongBench UUIDs)
    ./venv/bin/python3 -c "
import json, numpy as np
from vectorstore.registry import get_store

cfg = json.load(open('examples/$DS/config.json'))
store = get_store(cfg)
coll = cfg['collection']

all_points = []
offset = None
while True:
    points, offset = store.scroll(coll, limit=100, with_payload=True, with_vectors=True, offset=offset)
    if not points:
        break
    all_points.extend(points)

texts = [p.payload['text'] for p in all_points]
vecs = np.array([p.vector for p in all_points], dtype=np.float32)
chunks = [{'chunk_id': p.id, 'text': t} for p, t in zip(all_points, texts)]

np.save('examples/$DS/embeddings.npy', vecs)
json.dump(chunks, open('examples/$DS/chunks.json', 'w'), indent=2)
print(f'  Saved {len(chunks)} chunks, embeddings shape {vecs.shape}')
"
done

# 2. Create size tiers
echo ""
echo "--- Creating size tiers ---"
for DS in "${!DATASETS[@]}"; do
    COLL=${DATASETS[$DS]}
    ./venv/bin/python3 -c "
import json, numpy as np
from tools.corpus_slicer import create_size_tiers

chunks = json.load(open('examples/$DS/chunks.json'))
embs = np.load('examples/$DS/embeddings.npy')
total = len(chunks)

tiers = create_size_tiers(chunks, embs, tiers=[500, 1000, total])
for N, tier_list in tiers.items():
    json.dump([{'chunk_id': c['chunk_id']} for c in tier_list],
              open(f'examples/$DS/tier_{N}.json', 'w'))
    print(f'  [$COLL] tier_{N}.json: {len(tier_list)} chunks')
"
done

# 3. Generate FAQs — one unified set per dataset (measure_baseline_fkds will filter by tier later)
echo ""
echo "--- Generating FAQs (500 per dataset) ---"
for DS in "${!DATASETS[@]}"; do
    COLL=${DATASETS[$DS]}
    echo "[$COLL] Generating FAQs..."

    ./venv/bin/python3 -m pipeline.sleep_faq_generator \
        --config examples/$DS/config.json \
        --output examples/$DS/faqs_500.json \
        --count 500 2>&1 | tail -3

    FAQ_COUNT=$(./venv/bin/python3 -c "import json; print(len(json.load(open('examples/$DS/faqs_500.json'))))" 2>/dev/null)
    echo "  [$COLL] Generated $FAQ_COUNT FAQs"
done

# 4. Wrap FAQs in eval-like format for measure_baseline_fkds (which expects 'items' key)
echo ""
echo "--- Preparing eval wrappers ---"
for DS in "${!DATASETS[@]}"; do
    COLL=${DATASETS[$DS]}
    if [ -f "examples/$DS/faqs_500.json" ]; then
        ./venv/bin/python3 -c "
import json

faqs = json.load(open('examples/$DS/faqs_500.json'))
chunks = json.load(open('examples/$DS/chunks.json'))
total = len(chunks)

# Wrap FAQs in eval format for each tier
for N in [500, 1000, total]:
    tier = json.load(open(f'examples/$DS/tier_{N}.json'))
    tier_qids = set(c['chunk_id'] for c in tier)
    tier_faqs = [q for q in faqs if q.get('source') in tier_qids or q.get('chunk_id') in tier_qids]
    
    # If no FAQs match by direct ID, just use all FAQs
    if not tier_faqs:
        tier_faqs = faqs
    
    eval_wrapper = {
        'items': [{'question': q.get('question', q.get('user', '')),
                    'answer': q.get('answer', ''),
                    'source_chunk_ids': [q.get('source', str(q.get('chunk_id', '')))]}
                   for q in tier_faqs]
    }
    json.dump(eval_wrapper, open(f'examples/$DS/eval_tier_{N}.json', 'w'))
    print(f'  [$COLL] eval_tier_{N}.json: {len(eval_wrapper[\"items\"])} items')
"
    fi
done

# 5. Check lora_checkpoints
for DS in "${!DATASETS[@]}"; do
    mkdir -p examples/$DS/lora_checkpoints
done

# 6. Write the Phase 2 sweep script
echo ""
echo "--- Writing Phase 2 sweep script ---"
cat > scripts/run_phase2_multidataset.sh << 'SWEEPSCRIPT'
#!/usr/bin/env bash
set -e
cd /home/ubuntu/kvforge

export OPENAI_API_KEY='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'
export CUDA_VISIBLE_DEVICES=0
export PHASE2_DIR=results/absorption/phase2
mkdir -p $PHASE2_DIR
mkdir -p logs

BATCH_SIZE=1

LOOP_DATASETS="usecase3_squad longbench_2wikimqa longbench_hotpotqa"
LOOP_SEEDS="42 43 44"
LOOP_TIERS="500 1000"

for DS in $LOOP_DATASETS; do
    case $DS in
        usecase3_squad) LABEL="squad" ;;
        longbench_2wikimqa) LABEL="2wikimqa" ;;
        longbench_hotpotqa) LABEL="hotpotqa" ;;
    esac

    # Get total chunk count for full_corpus tier
    TOTAL=$(./venv/bin/python3 -c "import json; print(len(json.load(open('examples/$DS/chunks.json'))))")

    for N in $LOOP_TIERS $TOTAL; do
        TIER_FAQS="examples/$DS/faqs_500.json"
        EVAL_SET="examples/$DS/eval_tier_${N}.json"

        if [ ! -f "$EVAL_SET" ]; then
            echo "[$LABEL] No eval set for tier $N — skipping"
            continue
        fi

        for SEED in $LOOP_SEEDS; do
            RUN_LABEL="${LABEL}_N${N}_seed${SEED}"
            RUN_DIR="${PHASE2_DIR}/${RUN_LABEL}"
            mkdir -p "$RUN_DIR"

            echo "[$(date)] $RUN_LABEL: Training..."
            ./venv/bin/python3 -m pipeline.lora_trainer \
                --config examples/$DS/config.json \
                --faqs "$TIER_FAQS" \
                --seed $SEED \
                --batch-size $BATCH_SIZE \
                --output-dir "$RUN_DIR/train" 2>&1 | tail -3

            echo "[$(date)] $RUN_LABEL: Evaluating Path A (text_rag)..."
            ./venv/bin/python3 -m tools.measure_baseline_fkds \
                --config examples/$DS/config.json \
                --eval-set "$EVAL_SET" \
                --modes text_rag \
                --judge-model gpt-4o-mini \
                --output "$RUN_DIR/text_rag.json" 2>&1 | tail -5

            echo "[$(date)] $RUN_LABEL: Evaluating Path B (parametric)..."
            ./venv/bin/python3 -m tools.measure_baseline_fkds \
                --config examples/$DS/config.json \
                --eval-set "$EVAL_SET" \
                --modes parametric \
                --judge-model gpt-4o-mini \
                --output "$RUN_DIR/parametric.json" 2>&1 | tail -5

            # Compute summary delta
            ./venv/bin/python3 -c "
import json
ta = json.load(open('$RUN_DIR/text_rag.json'))
par = json.load(open('$RUN_DIR/parametric.json'))
ta_fkds = ta.get('fkds', ta.get('accuracy', 0))
par_fkds = par.get('fkds', par.get('accuracy', 0))
delta = par_fkds - ta_fkds
summary = {'text_rag_fkds': ta_fkds, 'parametric_fkds': par_fkds, 'delta': delta, 'crossover': delta > 0}
json.dump(summary, open('$RUN_DIR/summary.json', 'w'), indent=2)
print(f'  text_rag={ta_fkds:.4f}  parametric={par_fkds:.4f}  delta={delta:.4f}  crossover={delta > 0}')
"

            echo "[$(date)] $RUN_LABEL: Done"
        done
    done
done

echo "============================================"
echo " Phase 2 Complete — $(date)"
echo "============================================"

# Aggregate
./venv/bin/python3 -c "
import json, glob
results = {}
for f in glob.glob('${PHASE2_DIR}/*/summary.json'):
    label = f.split('/')[-2]
    data = json.load(open(f))
    results[label] = data

rows = []
for label in sorted(results):
    parts = label.split('_')
    ds = parts[0]
    n = parts[1]
    seed = parts[2] if len(parts) > 2 else '?'
    r = results[label]
    rows.append({
        'dataset': ds, 'tier': n, 'seed': seed,
        'text_rag': r.get('text_rag_fkds', '?'),
        'parametric': r.get('parametric_fkds', '?'),
        'delta': r.get('delta', '?'),
        'crossover': r.get('crossover', '?')
    })

print(f'{\"dataset\":<12} {\"tier\":<8} {\"seed\":<6} {\"delta\":<10} {\"crossover\":<10}')
print('-' * 50)
for row in rows:
    print(f\"{row['dataset']:<12} {row['tier']:<8} {row['seed']:<6} {str(row['delta']):<10} {str(row['crossover']):<10}\")
" 2>&1 | tee $PHASE2_DIR/aggregate.txt
SWEEPSCRIPT

chmod +x scripts/run_phase2_multidataset.sh
echo "Script written: scripts/run_phase2_multidataset.sh"

echo ""
echo "=== Phase 2 Setup Complete ==="
echo "To launch: screen -dmS phase2 bash scripts/run_phase2_multidataset.sh"