#!/usr/bin/env bash
# Phase 3 — Cross-dataset preparation on EC2
# Sets up configs, held-out questions, and embeddings for all 4 cross-datasets.
set -euo pipefail
cd /home/ubuntu/kvforge

VENV="$(pwd)/venv/bin/python3"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

gemma_config() {
    local name="$1" collection="$2" dir="$3" chunks="$4"
    mkdir -p "$dir"
    echo '{"current_lora_version":0,"checkpoint_path":null,"phase":1,"prs_history":[]}' > "$dir/version.json"
    cat > "$dir/config.json" << JSONEOF
{
  "use_case_name": "$name",
  "collection": "$collection",
  "version_file": "$dir/version.json",
  "llm_model": "google/gemma-4-E2B-it",
  "quantization": "4bit",
  "addons": ["indexing", "inference", "training", "background"],
  "addon_config": {
    "indexing": {
      "loader": "jsonl", "embed_model": "BAAI/bge-small-en-v1.5",
      "vector_dim": 384, "vector_store": "qdrant",
      "qdrant_host": "localhost", "qdrant_port": 6333
    },
    "inference": {
      "llm_model": "google/gemma-4-E2B-it",
      "top_k": 5, "max_new_tokens": 128
    },
    "training": {
      "lora_rank": 16, "lora_alpha": 32, "lora_lr": 2e-4,
      "lora_epochs": 1,
      "checkpoint_dir": "$dir/lora_checkpoints/",
      "replay_db": "$dir/replay.db",
      "sft_format": "chat"
    },
    "background": {"flush_seconds": 300, "flush_queries": 50}
  },
  "model_library": {
    "google/gemma-4-E2B-it": {"num_layers": 35, "num_kv_heads": 1, "head_dim": 256}
  }
}
JSONEOF
}

# ─── SQuAD ──────────────────────────────────────────────────────────────
log "Setting up SQuAD..."
gemma_config "SQuAD Reading Comp" "squad" "examples/usecase3_squad" "usecase3_squad"

# Generate held-out questions from SQuAD validation set
if [ ! -f "examples/usecase3_squad/eval_heldout.json" ]; then
    $VENV -c "
import json, random
from datasets import load_dataset
ds = load_dataset('rajpurkar/squad_v2', split='validation', trust_remote_code=False)
items = []
rng = random.Random(42)
samples = rng.sample(range(len(ds)), min(200, len(ds)))
for idx in samples:
    item = ds[int(idx)]
    items.append({
        'question': item['question'],
        'answer': item['answers']['text'][0] if item['answers']['text'] else '',
        'source_chunk_ids': [str(hash(item['context']) % 100000)],
    })
output = {'items': items}
open('examples/usecase3_squad/eval_heldout.json', 'w').write(json.dumps(output, indent=2))
print(f'SQuAD: {len(items)} held-out questions')
" 2>&1
fi

# Compute embeddings for SQuAD
if [ ! -f "examples/usecase3_squad/embeddings.npy" ]; then
    $VENV -c "
import json, numpy as np
from pathlib import Path
from fastembed import TextEmbedding
texts = []
with open('examples/usecase3_squad/data/corpus.jsonl') as f:
    for i, line in enumerate(f):
        try:
            d = json.loads(line)
            t = d.get('text', '')
            if t: texts.append(t)
        except: pass
print(f'SQuAD: {len(texts)} text chunks')
embedder = TextEmbedding(model_name='BAAI/bge-small-en-v1.5', show_download_progress=False)
embs = list(embedder.embed(texts))
np.save('examples/usecase3_squad/embeddings.npy', np.array(embs, dtype=np.float32))
chunks_out = [{'chunk_id': i, 'text': t} for i, t in enumerate(texts)]
Path('examples/usecase3_squad/chunks.json').write_text(json.dumps(chunks_out))
print(f'SQuAD embeddings: {len(embs)}')
" 2>&1
fi

# ─── TriviaQA ──────────────────────────────────────────────────────────
log "Setting up TriviaQA..."
gemma_config "TriviaQA General Knowledge" "triviaqa" "examples/usecase5_triviaqa" "usecase5_triviaqa"

# Copy eval from existing if available, or generate from dataset
if [ ! -f "examples/usecase5_triviaqa/eval_heldout.json" ]; then
    $VENV -c "
import json
# Sample 200 questions from the corpus line by line
lines = open('examples/usecase5_triviaqa/data/corpus.jsonl').readlines()
all_lines = lines[:200]
items = []
for i, line in enumerate(all_lines):
    try:
        d = json.loads(line)
        q = d.get('question', d.get('text', ''))[:200]
        a = d.get('answer', d.get('target', ''))[:200]
        if q and a:
            items.append({'question': q, 'answer': a, 'source_chunk_ids': [i]})
    except: pass
output = {'items': items}
open('examples/usecase5_triviaqa/eval_heldout.json', 'w').write(json.dumps(output, indent=2))
print(f'TriviaQA: {len(items)} held-out questions')
" 2>&1
fi

# Compute embeddings for TriviaQA
if [ ! -f "examples/usecase5_triviaqa/embeddings.npy" ]; then
    $VENV -c "
import json, numpy as np
from pathlib import Path
from fastembed import TextEmbedding
texts = []
with open('examples/usecase5_triviaqa/data/corpus.jsonl') as f:
    for i, line in enumerate(f):
        try:
            d = json.loads(line)
            t = d.get('text', '')
            if t: texts.append(t)
        except: pass
print(f'TriviaQA: {len(texts)} text chunks')
embedder = TextEmbedding(model_name='BAAI/bge-small-en-v1.5', show_download_progress=False)
embs = list(embedder.embed(texts))
np.save('examples/usecase5_triviaqa/embeddings.npy', np.array(embs, dtype=np.float32))
chunks_out = [{'chunk_id': i, 'text': t} for i, t in enumerate(texts)]
Path('examples/usecase5_triviaqa/chunks.json').write_text(json.dumps(chunks_out))
print(f'TriviaQA embeddings: {len(embs)}')
" 2>&1
fi

# ─── 2WikiMQA ──────────────────────────────────────────────────────────
log "Setting up 2WikiMQA..."
gemma_config "2Wiki Multi-Hop QA" "2wikimqa" "examples/longbench_2wikimqa" "longbench_2wikimqa"

# 2WikiMQA already has eval_2wikimqa.json with 200 items
if [ ! -f "examples/longbench_2wikimqa/eval_heldout.json" ]; then
    cp examples/longbench_2wikimqa/eval_2wikimqa.json examples/longbench_2wikimqa/eval_heldout.json
    log "2WikiMQA: eval copied"
fi

# Compute embeddings for 2WikiMQA
if [ ! -f "examples/longbench_2wikimqa/embeddings.npy" ]; then
    $VENV -c "
import json, numpy as np
from pathlib import Path
from fastembed import TextEmbedding
data = json.loads(Path('examples/longbench_2wikimqa/data/2wikimqa.chunks.json').read_text())
chunks_list = data.get('chunks', [])
texts = [c if isinstance(c, str) else c.get('text', '') for c in chunks_list if c]
print(f'2WikiMQA: {len(texts)} text chunks')
embedder = TextEmbedding(model_name='BAAI/bge-small-en-v1.5', show_download_progress=False)
embs = list(embedder.embed(texts))
np.save('examples/longbench_2wikimqa/embeddings.npy', np.array(embs, dtype=np.float32))
chunks_out = [{'chunk_id': i, 'text': t} for i, t in enumerate(texts)]
Path('examples/longbench_2wikimqa/chunks.json').write_text(json.dumps(chunks_out))
print(f'2WikiMQA embeddings: {len(embs)}')
" 2>&1
fi

# ─── HotpotQA ──────────────────────────────────────────────────────────
log "Setting up HotpotQA..."
gemma_config "HotpotQA Multi-Hop" "hotpotqa" "examples/longbench_hotpotqa" "longbench_hotpotqa"

if [ ! -f "examples/longbench_hotpotqa/eval_heldout.json" ]; then
    cp examples/longbench_hotpotqa/eval_hotpotqa.json examples/longbench_hotpotqa/eval_heldout.json
    log "HotpotQA: eval copied"
fi

if [ ! -f "examples/longbench_hotpotqa/embeddings.npy" ]; then
    $VENV -c "
import json, numpy as np
from pathlib import Path
from fastembed import TextEmbedding
data = json.loads(Path('examples/longbench_hotpotqa/data/hotpotqa.chunks.json').read_text())
chunks_list = data.get('chunks', [])
texts = [c if isinstance(c, str) else c.get('text', '') for c in chunks_list if c]
print(f'HotpotQA: {len(texts)} text chunks')
embedder = TextEmbedding(model_name='BAAI/bge-small-en-v1.5', show_download_progress=False)
embs = list(embedder.embed(texts))
np.save('examples/longbench_hotpotqa/embeddings.npy', np.array(embs, dtype=np.float32))
chunks_out = [{'chunk_id': i, 'text': t} for i, t in enumerate(texts)]
Path('examples/longbench_hotpotqa/chunks.json').write_text(json.dumps(chunks_out))
print(f'HotpotQA embeddings: {len(embs)}')
" 2>&1
fi

log "=== ALL DATASETS PREPARED ==="
echo "  SQuAD:     examples/usecase3_squad/config.json"
echo "  TriviaQA:  examples/usecase5_triviaqa/config.json"
echo "  2WikiMQA:  examples/longbench_2wikimqa/config.json"
echo "  HotpotQA:  examples/longbench_hotpotqa/config.json"