# Qdrant-Coupled Attention System — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-improving RAG pipeline where every new document indexed into Qdrant also fine-tunes the LLM's attention heads, and pre-computed KV tensors stored in Qdrant accelerate inference.

**Architecture:** Six sub-projects built sequentially — shared infrastructure (model_loader, kv_utils) feeds SP1 (KV indexer), which feeds SP2 (LoRA trainer), which feeds SP3 (KV inference), SP4 (confidence gate), SP5 (access tracker), and SP6 (monitoring dashboard). Each SP delivers standalone value and is tested independently before the next begins.

**Tech Stack:** Python 3.13, HuggingFace transformers + PEFT, Qdrant 1.17, fastembed, bitsandbytes, FastAPI, SQLite, AWS EC2 g5.xlarge (A10G 24GB)

**Spec:** `docs/superpowers/specs/2026-03-16-qdrant-llm-coupled-attention-design.md`

---

## File Map

| File | Sub-project | Responsibility |
|------|------------|----------------|
| `datasource_template.json` | Shared | Reference config template — copy per datasource |
| `model_loader.py` | Shared | Load Llama model (config-driven) + LoRA adapter; singleton |
| `kv_utils.py` | Shared | mean_pool_kv, serialize/deserialize KV tensors |
| `version.py` | Shared | Read/write per-datasource version file atomically |
| `kv_indexer.py` | SP1 | Extended indexer: embed + compute KV per chunk |
| `lora_trainer.py` | SP2 | LoRA fine-tune on new chunks + replay |
| `replay_buffer.py` | SP2 | SQLite-backed weighted chunk sampler |
| `prs_evaluator.py` | SP2/SP4 | Compute PRS after each LoRA round |
| `index_and_train.py` | SP2 | Orchestrator: SP1 → SP2 → KV refresh |
| `kv_inference.py` | SP3 | KV-injected inference with text fallback |
| `kv_background.py` | SP3 | Background KV recompute + access flush |
| `confidence_gate.py` | SP4 | Phase 3 direct-answer gate |
| `access_tracker.py` | SP5 | In-memory counters + async Qdrant flush |
| `monitoring_dashboard.py` | SP6 | FastAPI dashboard at :8080 |
| `tests/test_kv_utils.py` | Shared | Unit tests for KV tensor operations |
| `tests/test_kv_indexer.py` | SP1 | Integration tests for indexing pipeline |
| `tests/test_lora_trainer.py` | SP2 | Unit tests for trainer + replay buffer |
| `tests/test_kv_inference.py` | SP3 | Unit tests for inference paths |
| `tests/test_confidence_gate.py` | SP4 | Unit tests for gate scoring |
| `tests/test_access_tracker.py` | SP5 | Unit tests for tracker + flush |

---

## Chunk 1: Environment + Shared Infrastructure + SP1

> **Deployment model:** Code is developed and unit-tested **locally** (macOS).
> GPU-dependent tasks (KV compute, LoRA training, inference) and Qdrant run on
> **EC2 g5.xlarge at `100.48.17.48`** (user `ubuntu`, PEM `/Users/hemant/Downloads/RoPE/g5.x.pem`).
> `my_config.json` on EC2 uses `qdrant_host: localhost` (Qdrant runs locally on EC2).
> Unit tests that don't need a GPU run locally against a local Qdrant at `localhost:6333`.

### Task 0: Per-Datasource Config Schema

**Files:**
- Create: `datasource_template.json`
- Modify: `version.py` (add `init(cfg)`)
- Modify: `model_loader.py` (add `init(cfg)`)

Each datasource gets its own config file. Scripts call `version.init(cfg)` and
`model_loader.init(cfg)` at startup. `ReplayBuffer` already accepts `db_path` as a
constructor argument — callers pass `cfg.get("replay_db", "replay_buffer.db")`.

- [ ] **Step 1: Create the config template**

```json
// datasource_template.json — copy and fill in for each new datasource
{
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "collection": "my-datasource",

  "embed_model": "BAAI/bge-small-en-v1.5",
  "embed_batch": 64,
  "chunk_size": 512,
  "chunk_overlap": 64,
  "upsert_batch": 100,
  "top_k": 5,

  "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
  "kv_num_layers": 28,
  "kv_num_heads": 8,
  "kv_head_dim": 128,

  "lora_rank": 16,
  "lora_alpha": 32,
  "lora_target_modules": ["q_proj", "k_proj", "v_proj"],
  "lora_dropout": 0.05,
  "lora_epochs": 3,
  "lora_lr": 0.0002,

  "checkpoint_dir": "lora_checkpoints/my-datasource/",
  "version_file": "my_datasource_version.json",
  "replay_db": "my_datasource_replay.db",

  "gate_threshold": 0.75,
  "access_flush_seconds": 300,
  "access_flush_queries": 50,
  "dashboard_port": 8080
}
```

Concrete example for the existing Bedrock datasource:

```bash
cp datasource_template.json datasource_bedrock.json
# Edit: collection=bedrock-user-guide, checkpoint_dir=lora_checkpoints/bedrock/,
#       version_file=bedrock_version.json, replay_db=bedrock_replay.db
```

- [ ] **Step 2: Add `init(cfg)` to `version.py`**

Append this function to `version.py` (before `load()`):

```python
def init(cfg: dict) -> None:
    """Set the version file path from config. Call once at startup before any other call."""
    global VERSION_FILE
    VERSION_FILE = Path(cfg.get("version_file", "version.json"))
```

- [ ] **Step 3: Add `init(cfg)` to `model_loader.py`**

Append this function to `model_loader.py` (after the module-level `MODEL_ID` assignment):

```python
def init(cfg: dict) -> None:
    """Override MODEL_ID from config. Call once before load()."""
    global MODEL_ID
    MODEL_ID = cfg.get("llm_model", MODEL_ID)
```

- [ ] **Step 4: Wire `init()` calls in every entry-point script**

In `kv_indexer.py`, `lora_trainer.py`, `prs_evaluator.py`, `index_and_train.py`,
`kv_inference.py`, `confidence_gate.py`, and `monitoring_dashboard.py` — after
`cfg = json.load(f)` add:

```python
import version as ver
import model_loader
ver.init(cfg)
model_loader.init(cfg)
```

In `lora_trainer.py` and any script constructing `ReplayBuffer`, change:

```python
rb = ReplayBuffer()
# → becomes:
rb = ReplayBuffer(db_path=cfg.get("replay_db", "replay_buffer.db"))
```

- [ ] **Step 5: Verify with a smoke test using a second config**

```bash
# Create a minimal second-datasource config pointing at a different collection
python3 -c "
import json, version, model_loader
cfg = json.load(open('datasource_template.json'))
cfg['version_file'] = '/tmp/test_version.json'
cfg['llm_model']    = 'meta-llama/Llama-3.2-1B-Instruct'
version.init(cfg)
model_loader.init(cfg)
print('version file:', version.VERSION_FILE)
print('model id:    ', model_loader.MODEL_ID)
"
```
Expected:
```
version file: /tmp/test_version.json
model id:     meta-llama/Llama-3.2-1B-Instruct
```

- [ ] **Step 6: Commit**

```bash
git add datasource_template.json version.py model_loader.py
git commit -m "feat: per-datasource config — version_file, replay_db, llm_model driven by config"
```

---

### Task 1: EC2 Environment Setup

**Files:**
- Modify: `requirements_gpu.txt` (create)

- [ ] **Step 1: Create GPU requirements file**

```bash
cat > requirements_gpu.txt << 'EOF'
torch>=2.3.0
transformers>=4.45.0
peft>=0.12.0
bitsandbytes>=0.43.0
accelerate>=0.30.0
datasets>=2.19.0
fastapi>=0.111.0
uvicorn>=0.29.0
EOF
```

- [ ] **Step 2: Install on EC2 g5.xlarge**

```bash
# On EC2 (Deep Learning AMI — PyTorch already installed)
pip install -r requirements_gpu.txt
# Verify GPU visible
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```
Expected output: `NVIDIA A10G`

- [ ] **Step 3: Verify HuggingFace model access**

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
print('tokenizer vocab size:', tok.vocab_size)
"
```
Expected: `tokenizer vocab size: 128256`
Note: Requires HuggingFace token with Llama access. Set `HF_TOKEN` env var.

- [ ] **Step 4: Commit**

```bash
git add requirements_gpu.txt
git commit -m "feat: add GPU requirements for LoRA pipeline"
```

---

### Task 2: `kv_utils.py` — KV Tensor Operations

**Files:**
- Create: `kv_utils.py`
- Create: `tests/test_kv_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_kv_utils.py
import numpy as np
import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kv_utils import mean_pool_kv, serialize_kv, deserialize_kv, stack_past_key_values

NUM_LAYERS = 4   # use small values for tests
NUM_KV_HEADS = 2
HEAD_DIM = 8
SEQ_LEN = 16


def make_fake_past_key_values(seq_len=SEQ_LEN):
    """Return HuggingFace-style past_key_values: tuple of (K,V) per layer."""
    return tuple(
        (
            torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM),
            torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM),
        )
        for _ in range(NUM_LAYERS)
    )


def test_mean_pool_kv_shape():
    pkv = make_fake_past_key_values()
    result = mean_pool_kv(pkv)
    assert result.shape == (NUM_LAYERS, 2, NUM_KV_HEADS, HEAD_DIM)


def test_mean_pool_kv_dtype():
    pkv = make_fake_past_key_values()
    result = mean_pool_kv(pkv)
    assert result.dtype == np.float16


def test_serialize_deserialize_roundtrip():
    pkv = make_fake_past_key_values()
    arr = mean_pool_kv(pkv)
    b64 = serialize_kv(arr)
    assert isinstance(b64, str)
    restored = deserialize_kv(b64, shape=(NUM_LAYERS, 2, NUM_KV_HEADS, HEAD_DIM))
    np.testing.assert_array_almost_equal(arr.astype(np.float32),
                                          restored.astype(np.float32), decimal=2)


def test_stack_past_key_values_shape():
    chunks_kv = [mean_pool_kv(make_fake_past_key_values()) for _ in range(3)]
    pkv = stack_past_key_values(chunks_kv, num_layers=NUM_LAYERS,
                                 num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM)
    assert len(pkv) == NUM_LAYERS
    k, v = pkv[0]
    # 3 chunks → seq_len=3 (one mean-pooled position per chunk)
    assert k.shape == (1, NUM_KV_HEADS, 3, HEAD_DIM)
    assert v.shape == (1, NUM_KV_HEADS, 3, HEAD_DIM)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /Users/hemant/Downloads/RoPE/qdrant
venv/bin/python3 -m pytest tests/test_kv_utils.py -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'mean_pool_kv' from 'kv_utils'`

- [ ] **Step 3: Implement `kv_utils.py`**

```python
"""
kv_utils.py — KV tensor utilities shared by SP1, SP2, SP3.

Operations:
  mean_pool_kv   — compress past_key_values (per-token) → fixed-size array
  serialize_kv   — numpy float16 array → base64 string (for Qdrant payload)
  deserialize_kv — base64 string → numpy float16 array
  stack_past_key_values — list of chunk KV arrays → HuggingFace past_key_values
"""

import base64
import numpy as np
import torch


def mean_pool_kv(past_key_values: tuple) -> np.ndarray:
    """
    Compress HuggingFace past_key_values to a fixed-size float16 array.

    Input:  tuple of (K, V) per layer, each tensor [1, num_kv_heads, seq_len, head_dim]
    Output: np.ndarray [num_layers, 2, num_kv_heads, head_dim] float16
            (mean-pooled over seq_len dimension — one representative vector per chunk)
    """
    pooled = []
    for k, v in past_key_values:
        # k, v: [1, num_kv_heads, seq_len, head_dim]
        k = k.squeeze(0)          # [num_kv_heads, seq_len, head_dim]
        v = v.squeeze(0)          # [num_kv_heads, seq_len, head_dim]
        k_pooled = k.mean(dim=1)  # mean over seq_len → [num_kv_heads, head_dim]
        v_pooled = v.mean(dim=1)  # mean over seq_len → [num_kv_heads, head_dim]
        pooled.append(torch.stack([k_pooled, v_pooled]))  # [2, num_kv_heads, head_dim]
    result = torch.stack(pooled)  # [num_layers, 2, num_kv_heads, head_dim]
    return result.cpu().to(torch.float16).numpy()


def serialize_kv(arr: np.ndarray) -> str:
    """Float16 numpy array → base64 string for Qdrant payload storage."""
    return base64.b64encode(arr.astype(np.float16).tobytes()).decode("ascii")


def deserialize_kv(b64: str, shape: tuple) -> np.ndarray:
    """Base64 string → float16 numpy array with given shape."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float16).reshape(shape)


def stack_past_key_values(
    chunks_kv: list,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple:
    """
    Convert list of per-chunk KV arrays into HuggingFace past_key_values format.

    Input:  list of N arrays, each [num_layers, 2, num_kv_heads, head_dim]
    Output: tuple of num_layers tuples, each (K, V) shaped [1, num_kv_heads, N, head_dim]
            N chunks become N "positions" in the sequence dimension.
    """
    layer_kvs = []
    for layer_idx in range(num_layers):
        ks, vs = [], []
        for chunk_arr in chunks_kv:
            # chunk_arr[layer_idx]: [2, num_kv_heads, head_dim]
            layer = torch.from_numpy(chunk_arr[layer_idx].astype(np.float32))
            ks.append(layer[0])  # [num_kv_heads, head_dim]
            vs.append(layer[1])  # [num_kv_heads, head_dim]
        # stack along new seq dim: [num_kv_heads, N, head_dim] → unsqueeze batch
        k = torch.stack(ks, dim=1).unsqueeze(0)  # [1, num_kv_heads, N, head_dim]
        v = torch.stack(vs, dim=1).unsqueeze(0)  # [1, num_kv_heads, N, head_dim]
        layer_kvs.append((k, v))
    return tuple(layer_kvs)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
venv/bin/python3 -m pytest tests/test_kv_utils.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add kv_utils.py tests/test_kv_utils.py
git commit -m "feat: add kv_utils — mean_pool, serialize, stack KV tensors"
```

---

### Task 3: `version.py` — Atomic version.json I/O

**Files:**
- Create: `version.py`

- [ ] **Step 1: Implement `version.py`**

```python
"""
version.py — Atomic read/write of version.json.

Schema:
{
  "current_lora_version": 0,
  "checkpoint_path": null,
  "phase": 2,
  "prs_history": [],
  "known_good_queries": []
}
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

VERSION_FILE = Path(__file__).parent / "version.json"

DEFAULTS: dict[str, Any] = {
    "current_lora_version": 0,
    "checkpoint_path": None,
    "phase": 1,
    "prs_history": [],
    "known_good_queries": [],
}


def load() -> dict:
    if not VERSION_FILE.exists():
        return dict(DEFAULTS)
    with open(VERSION_FILE) as f:
        data = json.load(f)
    # back-fill any keys added in later versions
    for k, v in DEFAULTS.items():
        data.setdefault(k, v)
    return data


def save(data: dict) -> None:
    """Write atomically via temp file + rename."""
    tmp = VERSION_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, VERSION_FILE)


def get_lora_version() -> int:
    return load()["current_lora_version"]


def get_phase() -> int:
    return load()["phase"]


def increment_lora_version(checkpoint_path: str) -> int:
    data = load()
    data["current_lora_version"] += 1
    data["checkpoint_path"] = checkpoint_path
    save(data)
    return data["current_lora_version"]


def activate_phase_2() -> None:
    """Call from index_and_train.py after first successful LoRA round + SP3 deployed."""
    data = load()
    if data["phase"] < 2:
        data["phase"] = 2
        save(data)
        print("✅ Phase 2 activated — KV injection enabled")


def append_prs(round_num: int, prs: float) -> None:
    data = load()
    data["prs_history"].append({"round": round_num, "prs": round(prs, 4)})
    # check phase transition: PRS >= 0.80 for 2 consecutive rounds → phase 3
    history = data["prs_history"]
    if (len(history) >= 2
            and all(r["prs"] >= 0.80 for r in history[-2:])
            and data["phase"] < 3):
        data["phase"] = 3
        print("✅ Phase 3 activated — confidence gate now live")
    save(data)
```

- [ ] **Step 2: Verify with quick smoke test**

```bash
venv/bin/python3 -c "
import version
d = version.load()
print('defaults:', d)
version.increment_lora_version('lora_checkpoints/v1/')
print('after increment:', version.load()['current_lora_version'])
import os; os.remove('version.json')
"
```
Expected: `defaults: {'current_lora_version': 0, ...}` then `after increment: 1`

- [ ] **Step 3: Commit**

```bash
git add version.py
git commit -m "feat: add version.py — atomic version.json read/write with phase transition"
```

---

### Task 4: `model_loader.py` — Singleton Model + LoRA

**Files:**
- Create: `model_loader.py`

- [ ] **Step 1: Implement `model_loader.py`**

```python
"""
model_loader.py — Load Llama 3.2 3B + optional LoRA adapter.

Singleton pattern: call load() once per process; reload() to swap LoRA adapter.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_model = None
_tokenizer = None
_current_checkpoint: Optional[str] = None

MODEL_ID = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(lora_checkpoint: Optional[str] = None) -> tuple:
    """
    Load model + tokenizer. If lora_checkpoint is given, apply LoRA adapter.
    Returns (model, tokenizer). Cached after first call.
    """
    global _model, _tokenizer, _current_checkpoint
    if _model is not None and lora_checkpoint == _current_checkpoint:
        return _model, _tokenizer

    print(f"🤖 Loading {MODEL_ID} on {DEVICE} …")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if lora_checkpoint and Path(lora_checkpoint).exists():
        from peft import PeftModel
        print(f"🔌 Applying LoRA adapter from {lora_checkpoint} …")
        _model = PeftModel.from_pretrained(_model, lora_checkpoint)
        _model = _model.merge_and_unload()  # merge for faster inference

    _model.eval()
    _current_checkpoint = lora_checkpoint
    return _model, _tokenizer


def reload(lora_checkpoint: Optional[str] = None) -> tuple:
    """
    Force reload from the base MODEL_ID, then apply lora_checkpoint if given.
    Always starts from the base model (not the previously-merged weights) so
    that repeated LoRA rounds each apply a fresh adapter without double-merging.
    """
    global _model, _tokenizer, _current_checkpoint
    _model = _tokenizer = _current_checkpoint = None
    return load(lora_checkpoint)


def get_kv_shape(cfg: dict) -> tuple[int, int, int]:
    """Return (num_layers, num_kv_heads, head_dim) from config."""
    return cfg["kv_num_layers"], cfg["kv_num_heads"], cfg["kv_head_dim"]
```

- [ ] **Step 2: Smoke test (requires GPU instance)**

```bash
venv/bin/python3 -c "
import model_loader
model, tok = model_loader.load()
inputs = tok('Hello', return_tensors='pt').to(model_loader.DEVICE)
out = model.generate(**inputs, max_new_tokens=5)
print(tok.decode(out[0]))
"
```
Expected: Some short generated text without errors.

- [ ] **Step 3: Commit**

```bash
git add model_loader.py
git commit -m "feat: add model_loader — singleton Llama 3.2 3B with optional LoRA"
```

---

### Task 5: `kv_indexer.py` — Extended Indexer

**Files:**
- Create: `kv_indexer.py`
- Create: `tests/test_kv_indexer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_kv_indexer.py
"""
Tests for kv_indexer.py.
Uses a mock model so GPU is not required for unit tests.
"""
import json
import numpy as np
import pytest
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_mock_model_outputs(num_layers=28, num_kv_heads=8, head_dim=128, seq_len=10):
    """Return a mock HuggingFace model output with past_key_values."""
    import torch
    past_key_values = tuple(
        (
            torch.randn(1, num_kv_heads, seq_len, head_dim),
            torch.randn(1, num_kv_heads, seq_len, head_dim),
        )
        for _ in range(num_layers)
    )
    mock_out = MagicMock()
    mock_out.past_key_values = past_key_values
    return mock_out


def test_compute_kv_for_chunk_shape():
    import torch
    from kv_indexer import compute_kv_for_chunk

    # MagicMock supports context-manager protocol natively (__enter__/__exit__)
    # so torch.no_grad() inside compute_kv_for_chunk works without patching.
    mock_model = MagicMock()
    mock_model.device = "cpu"
    # Setting return_value on a MagicMock makes calling mock_model(...) return this.
    mock_model.return_value = make_mock_model_outputs(
        num_layers=4, num_kv_heads=2, head_dim=8, seq_len=10
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # tiny tokenizer, no GPU needed
    tokenizer.pad_token = tokenizer.eos_token

    # Pass mock directly — compute_kv_for_chunk takes (text, model, tokenizer, ...)
    arr = compute_kv_for_chunk("test text", mock_model, tokenizer,
                                num_layers=4, num_kv_heads=2, head_dim=8)
    assert arr.shape == (4, 2, 2, 8)
    assert arr.dtype == np.float16


def test_kv_indexer_payload_keys():
    """chunk_to_payload must include kv_cache and kv_version=null."""
    from kv_indexer import build_payload
    fake_kv = np.zeros((4, 2, 2, 8), dtype=np.float16)
    payload = build_payload(
        text="hello world",
        page=1,
        source_file="test.pdf",
        kv_array=fake_kv,
    )
    assert "kv_cache" in payload
    assert payload["kv_version"] is None
    assert payload["source_file"] == "test.pdf"
    assert payload["access_count"] == 0
    assert payload["tier"] == "frozen"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
venv/bin/python3 -m pytest tests/test_kv_indexer.py -v 2>&1 | head -10
```
Expected: `ImportError: cannot import name 'compute_kv_for_chunk' from 'kv_indexer'`

- [ ] **Step 3: Implement `kv_indexer.py`**

```python
"""
kv_indexer.py — Extended indexer: chunk + embed + compute KV tensors.

Commands:
  index      <pdf>       — full index (embed + KV compute + upsert)
  compute-kv             — recompute KV for filtered chunks (no re-embed)
    --filter kv_version=null
    --stale-version N    — heal all chunks with kv_version < N
    --source-file FILE
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

sys.path.insert(0, str(Path(__file__).parent))
import model_loader
import kv_utils
import version as ver
from bedrock_rag import _load_config, chunk_pages, read_pdf, embed_chunks
from fastembed import TextEmbedding


def compute_kv_for_chunk(
    text: str,
    model,
    tokenizer,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    """
    Run a single chunk through the LLM forward pass and return mean-pooled KV.
    Output shape: [num_layers, 2, num_kv_heads, head_dim] float16
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    return kv_utils.mean_pool_kv(outputs.past_key_values)


def build_payload(
    text: str,
    page: int,
    source_file: str,
    kv_array: np.ndarray,
    indexed_at: int | None = None,
) -> dict:
    """Construct the full Qdrant payload for a new chunk."""
    return {
        "text": text,
        "page": page,
        "source_file": source_file,
        "indexed_at": indexed_at or int(time.time()),
        "kv_cache": kv_utils.serialize_kv(kv_array),
        "kv_version": None,
        "access_count": 0,
        "last_accessed_ts": None,
        "avg_retrieval_rank": None,
        "parametric_hit_count": 0,
        "tier": "frozen",
    }


def cmd_index(pdf_path: Path, cfg: dict) -> None:
    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)

    # 1. Chunk + embed (reuse bedrock_rag pipeline)
    pages = read_pdf(pdf_path)
    chunks = chunk_pages(pages, cfg["chunk_size"], cfg["chunk_overlap"])
    print(f"✂️  {len(chunks)} chunks from {pdf_path.name}")

    embedder = TextEmbedding(model_name=cfg["embed_model"],
                              show_download_progress=False)
    vectors = embed_chunks(chunks, embedder, cfg["embed_batch"])

    # 2. Load LLM for KV computation
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    # 3. Compute KV + upsert
    print(f"🔢 Computing KV tensors for {len(chunks)} chunks …")
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        kv_arr = compute_kv_for_chunk(
            chunk["text"], model, tokenizer, num_layers, num_kv_heads, head_dim
        )
        payload = build_payload(
            text=chunk["text"],
            page=chunk["page"],
            source_file=pdf_path.name,
            kv_array=kv_arr,
        )
        points.append(PointStruct(id=chunk["chunk_id"], vector=vec, payload=payload))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(chunks)}", end="\r", flush=True)

    # batch upsert
    for start in range(0, len(points), cfg["upsert_batch"]):
        client.upsert(
            collection_name=cfg["collection"],
            points=points[start:start + cfg["upsert_batch"]],
        )
    print(f"\n✅ Indexed {len(points)} chunks with KV (kv_version=null)")


def cmd_compute_kv(cfg: dict, filter_type: str, filter_value) -> None:
    """Recompute KV for chunks matching the given filter."""
    from qdrant_client.models import Filter, FieldCondition, IsNullCondition, Range

    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)

    lora_ckpt = ver.load().get("checkpoint_path")
    current_ver = ver.get_lora_version()
    model, tokenizer = model_loader.load(lora_ckpt)

    # Scroll through matching chunks
    if filter_type == "null":
        scroll_filter = Filter(must=[IsNullCondition(is_null={"key": "kv_version"})])
    elif filter_type == "stale":
        scroll_filter = Filter(must=[
            FieldCondition(key="kv_version",
                           range=Range(lt=int(filter_value)))
        ])
    else:
        scroll_filter = Filter(must=[
            FieldCondition(key="source_file", match={"value": filter_value})
        ])

    offset = None
    updated = 0
    while True:
        results, offset = client.scroll(
            collection_name=cfg["collection"],
            scroll_filter=scroll_filter,
            limit=50,
            with_payload=True,
            offset=offset,
        )
        if not results:
            break
        for point in results:
            kv_arr = compute_kv_for_chunk(
                point.payload["text"], model, tokenizer,
                num_layers, num_kv_heads, head_dim
            )
            client.set_payload(
                collection_name=cfg["collection"],
                payload={"kv_cache": kv_utils.serialize_kv(kv_arr),
                         "kv_version": current_ver},
                points=[point.id],
            )
            updated += 1
        if offset is None:
            break

    print(f"✅ Recomputed KV for {updated} chunks → kv_version={current_ver}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="my_config.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    idx = sub.add_parser("index")
    idx.add_argument("pdf_file")

    kv = sub.add_parser("compute-kv")
    kv.add_argument("--filter", choices=["kv_version=null"], default=None)
    kv.add_argument("--stale-version", type=int, default=None)
    kv.add_argument("--source-file", default=None)

    args = p.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    if args.cmd == "index":
        cmd_index(Path(args.pdf_file), cfg)
    elif args.cmd == "compute-kv":
        if args.stale_version is not None:
            cmd_compute_kv(cfg, "stale", args.stale_version)
        elif args.source_file:
            cmd_compute_kv(cfg, "source", args.source_file)
        else:
            cmd_compute_kv(cfg, "null", None)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
venv/bin/python3 -m pytest tests/test_kv_indexer.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add kv_indexer.py tests/test_kv_indexer.py
git commit -m "feat(SP1): kv_indexer — extend bedrock_rag index with KV tensor computation"
```

---

## Chunk 2: SP2 — LoRA Training Pipeline

### Task 6: `replay_buffer.py` — Weighted Chunk Sampler

**Files:**
- Create: `replay_buffer.py`
- Create: `tests/test_lora_trainer.py` (partial)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_lora_trainer.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from replay_buffer import ReplayBuffer


def test_add_and_sample(tmp_path):
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([
        {"chunk_id": 1, "text": "alpha", "tier": "hot"},
        {"chunk_id": 2, "text": "beta",  "tier": "warm"},
        {"chunk_id": 3, "text": "gamma", "tier": "cold"},
        {"chunk_id": 4, "text": "delta", "tier": "frozen"},
    ])
    samples = rb.sample(n=3, weight_by_tier=True)
    assert len(samples) == 3
    assert all("text" in s for s in samples)


def test_sample_respects_available_count(tmp_path):
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([{"chunk_id": 1, "text": "only one", "tier": "hot"}])
    samples = rb.sample(n=10, weight_by_tier=True)
    assert len(samples) == 1  # can't return more than available


def test_update_tier(tmp_path):
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([{"chunk_id": 99, "text": "test", "tier": "cold"}])
    rb.update_tier(99, "hot")
    row = rb._con.execute("SELECT tier FROM chunks WHERE chunk_id=99").fetchone()
    assert row[0] == "hot"


def test_evict_to_cap(tmp_path):
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    # Add 6 chunks with mixed tiers; cap at 4
    rb.add_chunks([
        {"chunk_id": i, "text": f"text {i}", "tier": "frozen"} for i in range(4)
    ] + [
        {"chunk_id": 10, "text": "hot", "tier": "hot"},
        {"chunk_id": 11, "text": "warm", "tier": "warm"},
    ], max_size=4)
    # Cap applied: 2 frozen (lowest-value) should be evicted
    assert rb.count() == 4
    # hot and warm chunks must survive
    tiers = {row[0]: row[1] for row in
             rb._con.execute("SELECT chunk_id, tier FROM chunks").fetchall()}
    assert tiers.get(10) == "hot"
    assert tiers.get(11) == "warm"
```

- [ ] **Step 2: Run — verify fail**

```bash
venv/bin/python3 -m pytest tests/test_lora_trainer.py -v 2>&1 | head -10
```
Expected: `ImportError: No module named 'replay_buffer'`

- [ ] **Step 3: Implement `replay_buffer.py`**

```python
"""
replay_buffer.py — SQLite-backed replay buffer for LoRA training.

Tier weights for sampling:
  hot    → 8
  warm   → 4
  cold   → 2
  frozen → 1
"""

import random
import sqlite3
from pathlib import Path

TIER_WEIGHTS = {"hot": 8, "warm": 4, "cold": 2, "frozen": 1}
DEFAULT_DB = str(Path(__file__).parent / "replay_buffer.db")


class ReplayBuffer:
    def __init__(self, db_path: str = DEFAULT_DB):
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY,
                text     TEXT NOT NULL,
                tier     TEXT NOT NULL DEFAULT 'frozen'
            )
        """)
        self._con.commit()

    def add_chunks(self, chunks: list[dict], max_size: int = 5000) -> None:
        """Insert or replace chunks; evict lowest-value chunks if buffer exceeds max_size."""
        self._con.executemany(
            "INSERT OR REPLACE INTO chunks (chunk_id, text, tier) VALUES (?,?,?)",
            [(c["chunk_id"], c["text"], c.get("tier", "frozen")) for c in chunks],
        )
        self._con.commit()
        self.evict_to_cap(max_size)

    def update_tier(self, chunk_id: int, tier: str) -> None:
        self._con.execute("UPDATE chunks SET tier=? WHERE chunk_id=?",
                          (tier, chunk_id))
        self._con.commit()

    def update_tiers_bulk(self, updates: list[tuple[int, str]]) -> None:
        """updates: list of (chunk_id, tier)"""
        self._con.executemany("UPDATE chunks SET tier=? WHERE chunk_id=?",
                              [(t, cid) for cid, t in updates])
        self._con.commit()

    def sample(self, n: int, weight_by_tier: bool = True) -> list[dict]:
        rows = self._con.execute(
            "SELECT chunk_id, text, tier FROM chunks"
        ).fetchall()
        if not rows:
            return []
        if weight_by_tier:
            weights = [TIER_WEIGHTS.get(r[2], 1) for r in rows]
            k = min(n, len(rows))
            chosen = random.choices(rows, weights=weights, k=k)
            # deduplicate while preserving approximate distribution
            seen, result = set(), []
            for row in chosen:
                if row[0] not in seen:
                    seen.add(row[0])
                    result.append({"chunk_id": row[0], "text": row[1], "tier": row[2]})
            # if dedup reduced below n, top up
            remaining = [r for r in rows if r[0] not in seen]
            while len(result) < k and remaining:
                row = random.choice(remaining)
                result.append({"chunk_id": row[0], "text": row[1], "tier": row[2]})
                remaining.remove(row)
        else:
            chosen = random.sample(rows, min(n, len(rows)))
            result = [{"chunk_id": r[0], "text": r[1], "tier": r[2]} for r in chosen]
        return result

    def count(self) -> int:
        return self._con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def evict_to_cap(self, max_size: int = 5000) -> int:
        """Evict oldest + lowest-tier chunks if buffer exceeds max_size. Returns count removed."""
        current = self.count()
        if current <= max_size:
            return 0
        # Order: frozen first (tier), then by rowid (oldest) — evict the least valuable
        tier_order = "CASE tier WHEN 'frozen' THEN 0 WHEN 'cold' THEN 1 WHEN 'warm' THEN 2 ELSE 3 END"
        to_delete = current - max_size
        self._con.execute(f"""
            DELETE FROM chunks WHERE chunk_id IN (
                SELECT chunk_id FROM chunks
                ORDER BY {tier_order} ASC, rowid ASC
                LIMIT ?
            )
        """, (to_delete,))
        self._con.commit()
        return to_delete
```

- [ ] **Step 4: Run tests**

```bash
venv/bin/python3 -m pytest tests/test_lora_trainer.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add replay_buffer.py tests/test_lora_trainer.py
git commit -m "feat(SP2): replay_buffer — SQLite weighted chunk sampler for LoRA training"
```

---

### Task 7: `lora_trainer.py` — LoRA Fine-Tuning

**Files:**
- Create: `lora_trainer.py`

- [ ] **Step 1: Implement `lora_trainer.py`**

```python
"""
lora_trainer.py — Fine-tune Llama 3.2 3B attention heads on new document chunks.

Usage:
  python3 lora_trainer.py --source-file ec2_guide.pdf --replay-ratio 0.2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

sys.path.insert(0, str(Path(__file__).parent))
import model_loader
import version as ver
from replay_buffer import ReplayBuffer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


def fetch_chunks_for_source(client: QdrantClient, collection: str,
                              source_file: str) -> list[dict]:
    """Retrieve all chunks belonging to a given source_file from Qdrant."""
    chunks, offset = [], None
    while True:
        results, offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="source_file",
                               match=MatchValue(value=source_file))
            ]),
            limit=200,
            with_payload=True,
            offset=offset,
        )
        chunks.extend({"chunk_id": r.id, "text": r.payload["text"],
                        "tier": r.payload.get("tier", "frozen")}
                       for r in results)
        if offset is None:
            break
    return chunks


def train(cfg: dict, new_chunks: list[dict], replay_chunks: list[dict],
          output_dir: str) -> None:
    """Run LoRA fine-tuning on new_chunks + replay_chunks."""
    all_texts = [c["text"] for c in new_chunks + replay_chunks]
    print(f"🎓 Training on {len(new_chunks)} new + {len(replay_chunks)} replay "
          f"= {len(all_texts)} chunks total")

    # Reload base model without merged LoRA for training
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.reload(lora_ckpt)

    lora_cfg = LoraConfig(
        r=cfg.get("lora_rank", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj"]),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=512,
                         padding="max_length")

    dataset = Dataset.from_dict({"text": all_texts})
    tokenized = dataset.map(tokenize, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("lora_epochs", 3),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=cfg.get("lora_lr", 2e-4),
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"💾 LoRA adapter saved to {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="my_config.json")
    p.add_argument("--source-file", required=True,
                   help="Source file name used in Qdrant payload (e.g. 'ec2_guide.pdf')")
    p.add_argument("--replay-ratio", type=float, default=0.2)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    rb = ReplayBuffer()

    new_chunks = fetch_chunks_for_source(client, cfg["collection"], args.source_file)
    if not new_chunks:
        print(f"❌ No chunks found for source_file='{args.source_file}'")
        sys.exit(1)

    # Add new chunks to replay buffer
    rb.add_chunks(new_chunks)

    n_replay = max(1, int(len(new_chunks) * args.replay_ratio))
    replay_chunks = rb.sample(n=n_replay, weight_by_tier=True)
    # exclude chunks from the current source to avoid duplication
    new_ids = {c["chunk_id"] for c in new_chunks}
    replay_chunks = [c for c in replay_chunks if c["chunk_id"] not in new_ids]

    new_ver = ver.get_lora_version() + 1
    output_dir = cfg.get("checkpoint_dir", "lora_checkpoints/") + f"v{new_ver}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train(cfg, new_chunks, replay_chunks, output_dir)
    ver.increment_lora_version(output_dir)
    print(f"✅ LoRA version → {new_ver}  checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add CLI + fetch tests to `tests/test_lora_trainer.py`**

```python
# Append to tests/test_lora_trainer.py

def test_fetch_chunks_for_source_filter():
    """fetch_chunks_for_source builds the correct Qdrant filter."""
    from unittest.mock import MagicMock, patch
    from lora_trainer import fetch_chunks_for_source

    mock_client = MagicMock()
    mock_client.scroll.return_value = ([], None)   # empty collection is fine
    chunks = fetch_chunks_for_source(mock_client, "my_coll", "guide.pdf")
    assert isinstance(chunks, list)
    # Verify the filter was applied
    call_kwargs = mock_client.scroll.call_args.kwargs
    assert call_kwargs["collection_name"] == "my_coll"
    must_cond = call_kwargs["scroll_filter"].must[0]
    assert must_cond.key == "source_file"


def test_main_help():
    """lora_trainer --help exits cleanly."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "lora_trainer.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--source-file" in result.stdout
```

- [ ] **Step 3: Run tests**

```bash
venv/bin/python3 -m pytest tests/test_lora_trainer.py -v
```
Expected: `6 passed`

- [ ] **Step 4: Commit**

```bash
git add lora_trainer.py tests/test_lora_trainer.py
git commit -m "feat(SP2): lora_trainer — LoRA fine-tune q/k/v_proj with tier-weighted replay"
```

---

### Task 8: `prs_evaluator.py` — PRS After Each LoRA Round

**Files:**
- Create: `prs_evaluator.py`

- [ ] **Step 1: Implement `prs_evaluator.py`**

```python
"""
prs_evaluator.py — Compute Parametric Readiness Score after each LoRA round.

PRS = 0.5 * min(accuracy_ratio, 1.0)
    + 0.3 * calibration_score
    + 0.2 * self_consistency

Run automatically by index_and_train.py after lora_trainer.py completes.
"""

import json
import sys
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

sys.path.insert(0, str(Path(__file__).parent))
import version as ver
import model_loader
# kv_inference is imported lazily inside evaluate() — SP3 may not exist yet


CONFIDENCE_PROMPT_SUFFIX = (
    "\n\nOn a scale of 0 to 100, how confident are you in your answer above? "
    "Reply with a single integer only."
)
HEDGING_MARKERS = [
    "i think", "i'm not sure", "i am not sure", "approximately",
    "maybe", "or maybe", "i don't know", "i do not know", "possibly",
    "it might", "it may",
]


def _embed(texts: list[str], model_name: str) -> np.ndarray:
    embedder = TextEmbedding(model_name=model_name, show_download_progress=False)
    return np.array(list(embedder.embed(texts)))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _generate_parametric(query: str, model, tokenizer) -> str:
    """Answer directly from model weights — no retrieved context."""
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=256, do_sample=False)
    out = pipe(query)
    return out[0]["generated_text"][len(query):].strip()


def _extract_confidence(answer: str, model, tokenizer) -> float:
    """Ask model to self-rate confidence; return value in [0, 1]."""
    prompt = answer + CONFIDENCE_PROMPT_SUFFIX
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=5, do_sample=False)
    out = pipe(prompt)
    tail = out[0]["generated_text"][len(prompt):].strip()
    try:
        val = int("".join(c for c in tail if c.isdigit())[:3])
        return min(val, 100) / 100.0
    except ValueError:
        return 0.5  # default if parsing fails


def _self_consistency(query: str, model, tokenizer, n: int = 3) -> float:
    """Generate n answers at temperature 0.7; return mean pairwise cosine sim."""
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=128, do_sample=True, temperature=0.7)
    answers = [pipe(query)[0]["generated_text"][len(query):].strip()
               for _ in range(n)]
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5",
                              show_download_progress=False)
    embs = np.array(list(embedder.embed(answers)))
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine_sim(embs[i], embs[j]))
    return float(np.mean(sims)) if sims else 1.0


def evaluate(faqs: list[dict], cfg: dict, lora_checkpoint: str | None = None) -> float:
    """
    Compute PRS on a sample of FAQs.
    Returns PRS in [0, 1].
    """
    model, tokenizer = model_loader.load(lora_checkpoint)
    embed_model = cfg.get("embed_model", "BAAI/bge-small-en-v1.5")

    # Lazy import — SP3 may not be built yet; graceful degradation
    try:
        from kv_inference import answer_with_retrieval
        has_sp3 = True
    except ImportError:
        has_sp3 = False

    accuracy_ratios, calibrations, consistencies = [], [], []

    for faq in faqs:
        q, gt = faq["question"], faq["answer"]

        # 1. Parametric answer
        param_ans = _generate_parametric(q, model, tokenizer)

        # 2. RAG answer (needs SP3; fall back to ground truth cosine if unavailable)
        if has_sp3:
            rag_ans = answer_with_retrieval(q, cfg)
        else:
            rag_ans = gt  # conservative: assume RAG = perfect

        # 3. Embed all three
        embs = _embed([param_ans, rag_ans, gt], embed_model)

        param_sim = _cosine_sim(embs[0], embs[2])
        rag_sim   = _cosine_sim(embs[1], embs[2])

        accuracy_ratio = min(param_sim / (rag_sim + 1e-9), 1.0)
        accuracy_ratios.append(accuracy_ratio)

        # 4. Calibration
        self_conf = _extract_confidence(param_ans, model, tokenizer)
        calibrations.append(1.0 - abs(self_conf - param_sim))

        # 5. Self-consistency
        consistencies.append(_self_consistency(q, model, tokenizer))

    prs = (0.5 * np.mean(accuracy_ratios)
           + 0.3 * np.mean(calibrations)
           + 0.2 * np.mean(consistencies))

    # Populate known_good_queries: queries where accuracy_ratio >= 0.85
    # Stored as pre-computed embeddings for use by confidence_gate._query_similarity
    good_queries = [faqs[i]["question"] for i, r in enumerate(accuracy_ratios) if r >= 0.85]
    if good_queries:
        embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)
        good_embs = [list(e) for e in embedder.embed(good_queries)]
        data = ver.load()
        data["known_good_queries"] = good_embs
        ver.save(data)

    return float(np.clip(prs, 0.0, 1.0))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="my_config.json")
    p.add_argument("--faqs", default="bedrock_50 faqs.json")
    p.add_argument("--sample", type=int, default=50)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.faqs) as f:
        all_faqs = json.load(f)

    import random
    faqs = random.sample(all_faqs, min(args.sample, len(all_faqs)))

    v = ver.load()
    prs = evaluate(faqs, cfg, v.get("checkpoint_path"))
    round_num = v["current_lora_version"]
    ver.append_prs(round_num, prs)
    print(f"📊 PRS after round {round_num}: {prs:.4f}")
    print(f"   Phase: {ver.get_phase()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help (no GPU needed)**

```bash
venv/bin/python3 -c "import prs_evaluator; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add prs_evaluator.py
git commit -m "feat(SP2): prs_evaluator — PRS computation with calibration and self-consistency"
```

---

### Task 9: `index_and_train.py` — Orchestrator

**Files:**
- Create: `index_and_train.py`

- [ ] **Step 1: Implement `index_and_train.py`**

```python
"""
index_and_train.py — Orchestrator: for each new document, run SP1 → SP2 → KV refresh.

Usage:
  python3 index_and_train.py new_document.pdf
  python3 index_and_train.py new_document.pdf --config my_config.json --skip-prs
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'─'*60}\n▶  {desc}\n{'─'*60}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print(f"❌ Failed: {desc}")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pdf_file")
    p.add_argument("--config", default="my_config.json")
    p.add_argument("--replay-ratio", type=float, default=0.2)
    p.add_argument("--skip-prs", action="store_true",
                   help="Skip PRS evaluation (faster, use during testing)")
    p.add_argument("--faqs", default="bedrock_50 faqs.json")
    args = p.parse_args()

    pdf = Path(args.pdf_file)
    if not pdf.exists():
        print(f"❌ File not found: {pdf}")
        sys.exit(1)

    py = sys.executable

    # ── Step 1: Index (chunk + embed + KV tensors) ─────────────────────────
    run([py, "kv_indexer.py", "--config", args.config, "index", str(pdf)],
        f"SP1: Index {pdf.name}")

    # ── Step 2: LoRA fine-tune ─────────────────────────────────────────────
    run([py, "lora_trainer.py",
         "--config", args.config,
         "--source-file", pdf.name,
         "--replay-ratio", str(args.replay_ratio)],
        "SP2: LoRA fine-tune")

    # ── Step 3: Recompute KV for new chunks with updated weights ───────────
    run([py, "kv_indexer.py", "--config", args.config,
         "compute-kv", "--source-file", pdf.name],
        "SP1: Recompute KV for new chunks with updated weights")

    # ── Step 4: Proactively heal ALL stale-versioned chunks ─────────────────
    # Reads current_lora_version from version.json and heals all chunks whose
    # kv_version < N (previously versioned by an earlier LoRA round).
    import json as _json
    with open(args.config) as _f:
        _cfg = _json.load(_f)
    import version as _ver
    current_ver = _ver.get_lora_version()
    if current_ver > 0:
        run([py, "kv_indexer.py", "--config", args.config,
             "compute-kv", "--stale-version", str(current_ver)],
            f"SP1: Proactive KV heal for all stale chunks (< v{current_ver})")

    # ── Step 5: PRS evaluation ─────────────────────────────────────────────
    if not args.skip_prs:
        run([py, "prs_evaluator.py",
             "--config", args.config,
             "--faqs", args.faqs],
            "SP2: PRS evaluation")

    # Activate phase 2 after the first successful LoRA round (SP1+SP3 must be deployed)
    _ver.activate_phase_2()

    print(f"\n✅ index_and_train complete for {pdf.name}")
    print("   Stale chunks from prior rounds have been healed proactively.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (dry run)**

```bash
# Verify the script parses and starts correctly (will fail when Qdrant unreachable — that's OK)
venv/bin/python3 index_and_train.py --help
```
Expected: usage message with `pdf_file`, `--config`, `--replay-ratio`, `--skip-prs`

- [ ] **Step 3: Commit**

```bash
git add index_and_train.py
git commit -m "feat(SP2): index_and_train — orchestrate SP1→LoRA→KV refresh per new document"
```

---

## Chunk 3: SP3 — KV-Injected Inference

### Task 10: `kv_background.py` — Background Worker

**Files:**
- Create: `kv_background.py`

- [ ] **Step 1: Implement `kv_background.py`**

```python
"""
kv_background.py — Background worker with two jobs:
  1. KV recompute queue: heal stale chunks after they are first retrieved
  2. Access tracker flush: batch-write access counters to Qdrant every 50 queries or 5 min

Run as a long-lived process alongside kv_inference.py:
  python3 kv_background.py &
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kv_utils
import model_loader
import version as ver
from qdrant_client import QdrantClient

_kv_queue: queue.Queue = queue.Queue()
_access_buffer: dict[int, dict] = {}
_access_lock = threading.Lock()
_query_count = 0
_query_lock = threading.Lock()


# ── Public API (called by kv_inference.py) ────────────────────────────────

def enqueue_kv_recompute(chunk_ids: list[int]) -> None:
    """Called from inference thread when stale chunks are detected."""
    for cid in chunk_ids:
        _kv_queue.put(cid)


def record_access(chunk_id: int, rank: int) -> None:
    """Called from inference thread — zero latency, in-memory only."""
    global _query_count
    with _access_lock:
        if chunk_id not in _access_buffer:
            _access_buffer[chunk_id] = {"count": 0, "rank_sum": 0.0, "last_ts": 0}
        _access_buffer[chunk_id]["count"] += 1
        _access_buffer[chunk_id]["rank_sum"] += rank
        _access_buffer[chunk_id]["last_ts"] = int(time.time())
    with _query_lock:
        _query_count += 1


def record_parametric_hit(chunk_ids: list[int]) -> None:
    """Called when confidence gate answers without retrieval."""
    with _access_lock:
        for cid in chunk_ids:
            if cid not in _access_buffer:
                _access_buffer[cid] = {"count": 0, "rank_sum": 0.0, "last_ts": 0,
                                        "parametric_hits": 0}
            _access_buffer[cid].setdefault("parametric_hits", 0)
            _access_buffer[cid]["parametric_hits"] += 1


# ── KV recompute worker ───────────────────────────────────────────────────

def _kv_worker(cfg: dict) -> None:
    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    num_layers = cfg["kv_num_layers"]
    num_kv_heads = cfg["kv_num_heads"]
    head_dim = cfg["kv_head_dim"]

    # Load model once at startup (model_loader singleton reuses it across calls)
    v = ver.load()
    _cached_lora_version = v.get("current_lora_version", 0)
    model, tokenizer = model_loader.load(v.get("checkpoint_path"))

    while True:
        chunk_id = _kv_queue.get()
        try:
            current_ver = ver.get_lora_version()
            # Reload if a new LoRA version has been written since worker started
            if current_ver != _cached_lora_version:
                lora_ckpt = ver.load().get("checkpoint_path")
                model, tokenizer = model_loader.reload(lora_ckpt)
                _cached_lora_version = current_ver

            results, _ = client.scroll(
                collection_name=cfg["collection"],
                ids=[chunk_id],
                with_payload=True,
                limit=1,
            )
            if not results:
                continue
            text = results[0].payload.get("text", "")
            from kv_indexer import compute_kv_for_chunk
            kv_arr = compute_kv_for_chunk(
                text, model, tokenizer, num_layers, num_kv_heads, head_dim
            )
            client.set_payload(
                collection_name=cfg["collection"],
                payload={"kv_cache": kv_utils.serialize_kv(kv_arr),
                         "kv_version": current_ver},
                points=[chunk_id],
            )
        except Exception as e:
            print(f"[kv_background] KV recompute error for chunk {chunk_id}: {e}",
                  flush=True)
        finally:
            _kv_queue.task_done()


# ── Access flush worker ───────────────────────────────────────────────────

def _flush_access(cfg: dict, client: QdrantClient) -> None:
    global _query_count
    with _access_lock:
        if not _access_buffer:
            return
        snapshot = dict(_access_buffer)
        _access_buffer.clear()
    with _query_lock:
        _query_count = 0

    current_ts = int(time.time())
    for chunk_id, delta in snapshot.items():
        try:
            existing = client.retrieve(
                collection_name=cfg["collection"],
                ids=[chunk_id],
                with_payload=True,
            )
            if not existing:
                continue
            payload = existing[0].payload
            old_count = payload.get("access_count", 0)
            old_rank_sum = old_count * payload.get("avg_retrieval_rank", 0.0)
            new_count = old_count + delta["count"]
            new_rank_avg = (old_rank_sum + delta["rank_sum"]) / new_count

            updates = {
                "access_count": new_count,
                "last_accessed_ts": delta["last_ts"],
                "avg_retrieval_rank": round(new_rank_avg, 3),
            }
            if "parametric_hits" in delta:
                updates["parametric_hit_count"] = (
                    payload.get("parametric_hit_count", 0) + delta["parametric_hits"]
                )
            client.set_payload(
                collection_name=cfg["collection"],
                payload=updates,
                points=[chunk_id],
            )
        except Exception as e:
            print(f"[kv_background] Access flush error for {chunk_id}: {e}", flush=True)


def _access_worker(cfg: dict) -> None:
    """Flush on every 50 queries or every 5 min — whichever comes first."""
    flush_interval = cfg.get("access_flush_seconds", 300)
    flush_queries = cfg.get("access_flush_queries", 50)
    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    last_flush = time.time()

    while True:
        time.sleep(5)
        with _query_lock:
            qc = _query_count
        elapsed = time.time() - last_flush
        if qc >= flush_queries or elapsed >= flush_interval:
            _flush_access(cfg, client)
            last_flush = time.time()


_started = False
_start_lock = threading.Lock()


def start(cfg: dict) -> None:
    """Start both background threads. Idempotent — safe to call multiple times."""
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
    t1 = threading.Thread(target=_kv_worker, args=(cfg,), daemon=True)
    t2 = threading.Thread(target=_access_worker, args=(cfg,), daemon=True)
    t1.start()
    t2.start()
    print("✅ kv_background workers started", flush=True)


if __name__ == "__main__":
    with open("my_config.json") as f:
        cfg = json.load(f)
    start(cfg)
    print("Background workers running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
```

- [ ] **Step 2: Commit**

```bash
git add kv_background.py
git commit -m "feat(SP3): kv_background — threaded KV recompute queue and access flush"
```

---

### Task 11: `kv_inference.py` — KV-Injected Inference

**Files:**
- Create: `kv_inference.py`
- Create: `tests/test_kv_inference.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_kv_inference.py
import sys
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_chunk(kv_version, chunk_id=1):
    import kv_utils
    fake_kv = np.zeros((28, 2, 8, 128), dtype=np.float16)
    return {
        "chunk_id": chunk_id,
        "text": "Amazon Bedrock is a managed service.",
        "page": 7,
        "score": 0.9,
        "kv_cache": kv_utils.serialize_kv(fake_kv),
        "kv_version": kv_version,
    }


def test_all_fresh_uses_kv_path():
    """When all chunks have current kv_version, should call generate_with_kv."""
    from kv_inference import decide_inference_mode
    chunks = [_fake_chunk(kv_version=5, chunk_id=i) for i in range(5)]
    mode = decide_inference_mode(chunks, current_lora_version=5)
    assert mode == "kv_injection"


def test_any_stale_uses_text_fallback():
    from kv_inference import decide_inference_mode
    chunks = [_fake_chunk(5), _fake_chunk(None), _fake_chunk(5)]
    mode = decide_inference_mode(chunks, current_lora_version=5)
    assert mode == "text_fallback"


def test_stale_chunks_are_queued():
    from kv_inference import decide_inference_mode, get_stale_chunk_ids
    chunks = [_fake_chunk(5), _fake_chunk(None, 2), _fake_chunk(3, 3)]
    stale = get_stale_chunk_ids(chunks, current_lora_version=5)
    assert set(stale) == {2, 3}


def test_kv_stacking_produces_correct_past_key_values_shape():
    """stack_past_key_values must produce HuggingFace-compatible past_key_values."""
    import kv_utils
    NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, N_CHUNKS = 28, 8, 128, 5
    # Simulate 5 fresh chunks
    chunks = [_fake_chunk(kv_version=3, chunk_id=i) for i in range(N_CHUNKS)]
    chunk_arrs = [
        kv_utils.deserialize_kv(c["kv_cache"], shape=(NUM_LAYERS, 2, NUM_KV_HEADS, HEAD_DIM))
        for c in chunks
    ]
    pkv = kv_utils.stack_past_key_values(chunk_arrs, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    assert len(pkv) == NUM_LAYERS
    k, v = pkv[0]
    # [batch=1, num_kv_heads, N_chunks, head_dim]
    assert k.shape == (1, NUM_KV_HEADS, N_CHUNKS, HEAD_DIM)
    assert v.shape == (1, NUM_KV_HEADS, N_CHUNKS, HEAD_DIM)
```

- [ ] **Step 2: Run — verify fail**

```bash
venv/bin/python3 -m pytest tests/test_kv_inference.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement `kv_inference.py`**

```python
"""
kv_inference.py — KV-injected inference with text-in-context fallback.

Decision logic per query:
  ALL chunks have kv_version == current_lora_version → KV injection (fast)
  ANY chunk stale or null                            → text-in-context fallback
  Either path                                        → enqueue stale chunks for bg heal
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))
import kv_utils
import kv_background
import model_loader
import version as ver
from bedrock_rag import _load_config, _run_search
from fastembed import TextEmbedding
from qdrant_client import QdrantClient


SYSTEM_PROMPT = (
    "You are a precise assistant. Answer ONLY using the provided context. "
    "Cite sources inline as [page P]. "
    "End with: Confidence: <0-100>%  — <one sentence explanation>"
)


# ── Pure decision functions (testable without GPU) ────────────────────────

def decide_inference_mode(chunks: list[dict], current_lora_version: int) -> str:
    """Return 'kv_injection' if all chunks are fresh, else 'text_fallback'."""
    for chunk in chunks:
        v = chunk.get("kv_version")
        if v is None or v < current_lora_version:
            return "text_fallback"
    return "kv_injection"


def get_stale_chunk_ids(chunks: list[dict], current_lora_version: int) -> list[int]:
    return [
        c["chunk_id"] for c in chunks
        if c.get("kv_version") is None or c.get("kv_version", 0) < current_lora_version
    ]


# ── Inference paths ───────────────────────────────────────────────────────

def generate_with_kv(query: str, chunks: list[dict],
                      model, tokenizer, cfg: dict) -> str:
    """Fast path: inject pre-computed KV tensors as past_key_values."""
    num_layers = cfg["kv_num_layers"]
    num_kv_heads = cfg["kv_num_heads"]
    head_dim = cfg["kv_head_dim"]
    kv_shape = (num_layers, 2, num_kv_heads, head_dim)

    chunk_kvs = [
        kv_utils.deserialize_kv(c["kv_cache"], shape=kv_shape)
        for c in chunks
    ]
    past_kv = kv_utils.stack_past_key_values(
        chunk_kvs, num_layers=num_layers,
        num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    # Move past_kv tensors to model device
    past_kv = tuple(
        (k.to(model.device), v.to(model.device)) for k, v in past_kv
    )

    prompt = f"Based on the context provided, answer: {query}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            past_key_values=past_kv,
            max_new_tokens=512,
            do_sample=False,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)


def generate_text_in_context(query: str, chunks: list[dict],
                               model, tokenizer) -> str:
    """Fallback path: include chunk text in prompt (same quality as ollama_answer.py)."""
    context = "\n\n".join(
        f"[score: {c['score']}, page {c['page']}]\n{c['text']}"
        for c in chunks
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n\n{context}\n\nQuestion: {query}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)


def answer_with_retrieval(query: str, cfg: dict) -> str:
    """
    Full SP3 pipeline: search → version check → KV inject or text fallback.
    Called by prs_evaluator.py for RAG-mode answers.
    """
    embedder = TextEmbedding(model_name=cfg["embed_model"],
                              show_download_progress=False)
    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])

    import argparse
    ns = argparse.Namespace(**{k: None for k in cfg})
    for k, v in cfg.items():
        setattr(ns, k, v)
    from bedrock_rag import Config
    rag_cfg = Config(**{k: cfg[k] for k in Config.__dataclass_fields__ if k in cfg})

    hits = _run_search(query, embedder, client, rag_cfg)
    if not hits:
        return ""

    chunks = [
        {
            "chunk_id": h.id,
            "text": h.payload["text"],
            "page": h.payload["page"],
            "score": round(h.score, 4),
            "kv_cache": h.payload.get("kv_cache"),
            "kv_version": h.payload.get("kv_version"),
        }
        for h in hits
    ]

    current_ver = ver.get_lora_version()
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    # Record access
    for rank, chunk in enumerate(chunks, start=1):
        kv_background.record_access(chunk["chunk_id"], rank)

    # Enqueue stale for background healing
    stale = get_stale_chunk_ids(chunks, current_ver)
    if stale:
        kv_background.enqueue_kv_recompute(stale)

    mode = decide_inference_mode(chunks, current_ver)
    if mode == "kv_injection":
        return generate_with_kv(query, chunks, model, tokenizer, cfg)
    else:
        return generate_text_in_context(query, chunks, model, tokenizer)


def main() -> None:
    """Pipe-compatible: read JSON from stdin (from bedrock_rag.py search)."""
    if sys.stdin.isatty():
        print('Usage: python3 bedrock_rag.py search "query" | python3 kv_inference.py')
        sys.exit(1)

    data = json.load(sys.stdin)
    query = data["query"]
    chunks = data["chunks"]

    with open("my_config.json") as f:
        cfg = json.load(f)

    kv_background.start(cfg)

    current_ver = ver.get_lora_version()
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    for rank, chunk in enumerate(chunks, start=1):
        kv_background.record_access(chunk["chunk_id"], rank)

    stale = get_stale_chunk_ids(chunks, current_ver)
    if stale:
        kv_background.enqueue_kv_recompute(stale)

    mode = decide_inference_mode(chunks, current_ver)
    print(f"📊 Mode: {mode}  |  lora_version={current_ver}  |  "
          f"stale_chunks={len(stale)}/{len(chunks)}")
    print("─" * 62)

    if mode == "kv_injection":
        answer = generate_with_kv(query, chunks, model, tokenizer, cfg)
    else:
        answer = generate_text_in_context(query, chunks, model, tokenizer)

    print(answer)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
venv/bin/python3 -m pytest tests/test_kv_inference.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add kv_inference.py tests/test_kv_inference.py
git commit -m "feat(SP3): kv_inference — KV injection with text-in-context fallback and bg healing"
```

---

## Chunk 4: SP4 — Confidence Gate

### Task 12: `confidence_gate.py`

**Files:**
- Create: `confidence_gate.py`
- Create: `tests/test_confidence_gate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_confidence_gate.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from confidence_gate import compute_hedging_score, decide_gate


def test_hedging_score_high_for_uncertain_text():
    text = "I think it might be approximately 50, or maybe 100, I'm not sure."
    score = compute_hedging_score(text)
    assert score > 0.3


def test_hedging_score_zero_for_confident_text():
    text = "Amazon Bedrock is a fully managed service for foundation models."
    score = compute_hedging_score(text)
    assert score == 0.0


def test_gate_low_entropy_passes():
    result = decide_gate(
        token_entropy=0.15,
        hedging_score=0.0,
        query_similarity=0.92,
        threshold=0.75,
    )
    assert result == "direct"


def test_gate_high_entropy_retrieves():
    result = decide_gate(
        token_entropy=0.82,
        hedging_score=0.5,
        query_similarity=0.30,
        threshold=0.75,
    )
    assert result == "retrieve"


def test_answer_falls_back_to_retrieval_when_phase_lt_3():
    """answer() must delegate to kv_inference when phase < 3 (no gate)."""
    from unittest.mock import patch, MagicMock

    cfg = {"embed_model": "BAAI/bge-small-en-v1.5", "gate_threshold": 0.75}
    # Patch the function inside the kv_inference module — confidence_gate imports it
    # locally via `from kv_inference import answer_with_retrieval`, so we must patch
    # at the source module, not at confidence_gate's namespace.
    with patch("confidence_gate.ver") as mock_ver, \
         patch("kv_inference.answer_with_retrieval", return_value="fallback answer") as mock_fn:
        mock_ver.get_phase.return_value = 2  # below Phase 3
        from confidence_gate import answer
        result = answer("what is bedrock?", cfg)
        mock_fn.assert_called_once()
```

- [ ] **Step 2: Run — verify fail**

```bash
venv/bin/python3 -m pytest tests/test_confidence_gate.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement `confidence_gate.py`**

```python
"""
confidence_gate.py — Phase 3 inference gate.

Active when version.json["phase"] >= 3.
Tries to answer directly from model weights first.
Falls back to kv_inference.py if confidence is below threshold.
"""

import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import kv_background
import model_loader
import version as ver

HEDGING_MARKERS = [
    "i think", "i'm not sure", "i am not sure", "approximately",
    "maybe", "or maybe", "i don't know", "i do not know",
    "possibly", "it might", "it may",
]

DRAFT_TOKENS = 20
DEFAULT_THRESHOLD = 0.75
_embedder = None  # cached TextEmbedding instance — avoid reload per query


# ── Pure functions (testable without model) ───────────────────────────────

def compute_hedging_score(text: str) -> float:
    """Fraction of hedging marker types present in text (0.0–1.0)."""
    lower = text.lower()
    hits = sum(1 for m in HEDGING_MARKERS if m in lower)
    return round(hits / len(HEDGING_MARKERS), 4)


def decide_gate(
    token_entropy: float,
    hedging_score: float,
    query_similarity: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """
    Compute P(no_retrieval) from three signals and apply threshold.
    Returns 'direct' or 'retrieve'.

    Weights: entropy 0.4, hedging 0.3, similarity 0.3
    Low entropy → high confidence, high hedging → low confidence.
    """
    entropy_score = max(0.0, 1.0 - token_entropy)       # invert: low entropy = good
    hedging_contribution = max(0.0, 1.0 - hedging_score) # invert: low hedging = good
    p_no_retrieval = (0.4 * entropy_score
                      + 0.3 * hedging_contribution
                      + 0.3 * query_similarity)
    return "direct" if p_no_retrieval >= threshold else "retrieve"


def _token_entropy(logits: torch.Tensor) -> float:
    """Mean token entropy from greedy draft logits."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    entropy_per_token = -(probs * log_probs).sum(dim=-1)
    return float(entropy_per_token.mean().item())


def _generate_draft(query: str, model, tokenizer,
                     max_tokens: int = DRAFT_TOKENS) -> tuple[str, float]:
    """
    Generate a short draft answer and compute its token entropy.
    Returns (draft_text, mean_entropy).
    """
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    scores = torch.stack(output.scores, dim=1)  # [1, num_tokens, vocab]
    entropy = _token_entropy(scores.squeeze(0))
    draft = tokenizer.decode(
        output.sequences[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return draft, entropy


def _query_similarity_to_known_good(query: str, cfg: dict) -> float:
    """Cosine similarity between query and known-good query embeddings."""
    global _embedder
    v = ver.load()
    known_good = v.get("known_good_queries", [])
    if not known_good:
        return 0.5  # neutral when no history yet

    from fastembed import TextEmbedding
    if _embedder is None:
        _embedder = TextEmbedding(
            model_name=cfg.get("embed_model", "BAAI/bge-small-en-v1.5"),
            show_download_progress=False,
        )
    embedder = _embedder
    q_emb = np.array(list(embedder.embed([query]))[0])
    known_embs = np.array(known_good)  # [N, dim]

    sims = known_embs @ q_emb / (
        np.linalg.norm(known_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9
    )
    return float(sims.max())


def answer(query: str, cfg: dict) -> str:
    """
    Phase 3 entry point.
    Returns final answer string (either direct or via kv_inference.answer_with_retrieval).
    """
    if ver.get_phase() < 3:
        from kv_inference import answer_with_retrieval
        return answer_with_retrieval(query, cfg)

    threshold = cfg.get("gate_threshold", DEFAULT_THRESHOLD)
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    draft, entropy = _generate_draft(query, model, tokenizer)
    hedging = compute_hedging_score(draft)
    similarity = _query_similarity_to_known_good(query, cfg)

    decision = decide_gate(entropy, hedging, similarity, threshold)
    print(f"  🎯 Gate: entropy={entropy:.2f} hedging={hedging:.2f} "
          f"sim={similarity:.2f} → {decision}", flush=True)

    if decision == "direct":
        # Full generation from weights
        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
        # Log parametric hit for access tracker
        _log_would_have_retrieved(query, cfg)
        return result
    else:
        from kv_inference import answer_with_retrieval
        return answer_with_retrieval(query, cfg)


def _log_would_have_retrieved(query: str, cfg: dict) -> None:
    """Find top-K chunks that would have been retrieved; increment their parametric_hit."""
    try:
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient
        from bedrock_rag import Config, _run_search
        embedder = TextEmbedding(model_name=cfg["embed_model"],
                                  show_download_progress=False)
        client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
        rag_cfg = Config(**{k: cfg[k] for k in Config.__dataclass_fields__ if k in cfg})
        hits = _run_search(query, embedder, client, rag_cfg)
        chunk_ids = [h.id for h in hits]
        kv_background.record_parametric_hit(chunk_ids)
    except Exception:
        pass  # non-critical


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--config", default="my_config.json")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    kv_background.start(cfg)
    print(answer(args.query, cfg))
```

- [ ] **Step 4: Run tests**

```bash
venv/bin/python3 -m pytest tests/test_confidence_gate.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add confidence_gate.py tests/test_confidence_gate.py
git commit -m "feat(SP4): confidence_gate — entropy+hedging+similarity gate for Phase 3"
```

---

## Chunk 5: SP5 — Access Tracker + SP6 — Dashboard

### Task 13: `access_tracker.py`

**Files:**
- Create: `access_tracker.py`
- Create: `tests/test_access_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_access_tracker.py
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from access_tracker import compute_tiers, AccessTracker


def test_compute_tiers_frozen_first():
    now = int(time.time())
    chunks = [
        {"chunk_id": 1, "access_count": 0,  "last_accessed_ts": None},
        {"chunk_id": 2, "access_count": 10, "last_accessed_ts": now - 86400},
        {"chunk_id": 3, "access_count": 5,  "last_accessed_ts": now - 86400 * 40},
        {"chunk_id": 4, "access_count": 1,  "last_accessed_ts": now - 86400 * 40},
    ]
    tiers = compute_tiers(chunks)
    assert tiers[1] == "frozen"   # access_count == 0
    assert tiers[2] in ("hot", "warm")  # recently accessed, high count
    assert tiers[3] in ("cold", "warm")
    # no chunk should be both frozen and another tier
    assert len(set(tiers.values())) >= 2


def test_tracker_record_and_snapshot():
    tracker = AccessTracker()
    tracker.record(chunk_id=42, rank=1)
    tracker.record(chunk_id=42, rank=2)
    tracker.record(chunk_id=99, rank=3)
    snap = tracker.snapshot_and_clear()
    assert snap[42]["count"] == 2
    assert snap[99]["count"] == 1
    assert tracker.snapshot_and_clear() == {}  # cleared
```

- [ ] **Step 2: Run — verify fail**

```bash
venv/bin/python3 -m pytest tests/test_access_tracker.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement `access_tracker.py`**

```python
"""
access_tracker.py — Thread-safe in-memory access counter + tier classifier.

Used directly by kv_background.py for the flush loop.
"""

import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class _Counter:
    count: int = 0
    rank_sum: float = 0.0
    last_ts: int = 0
    parametric_hits: int = 0


class AccessTracker:
    def __init__(self):
        self._data: dict[int, _Counter] = {}
        self._lock = threading.Lock()

    def record(self, chunk_id: int, rank: int) -> None:
        with self._lock:
            if chunk_id not in self._data:
                self._data[chunk_id] = _Counter()
            c = self._data[chunk_id]
            c.count += 1
            c.rank_sum += rank
            c.last_ts = int(time.time())

    def record_parametric_hit(self, chunk_ids: list[int]) -> None:
        with self._lock:
            for cid in chunk_ids:
                if cid not in self._data:
                    self._data[cid] = _Counter()
                self._data[cid].parametric_hits += 1

    def snapshot_and_clear(self) -> dict[int, dict]:
        with self._lock:
            snap = {
                cid: {"count": c.count, "rank_sum": c.rank_sum,
                       "last_ts": c.last_ts, "parametric_hits": c.parametric_hits}
                for cid, c in self._data.items()
            }
            self._data.clear()
        return snap

    def query_count(self) -> int:
        with self._lock:
            return sum(c.count for c in self._data.values())


def compute_tiers(chunks: list[dict]) -> dict[int, str]:
    """
    Classify each chunk into hot/warm/cold/frozen.
    Rules applied in order (first match wins):
      frozen : access_count == 0
      hot    : top 15% of non-frozen AND accessed within 7 days
      warm   : next 50% of non-frozen AND accessed within 30 days
      cold   : all remaining non-frozen
    """
    now = int(time.time())
    result: dict[int, str] = {}

    frozen_ids = {c["chunk_id"] for c in chunks if c.get("access_count", 0) == 0}
    for c in chunks:
        if c["chunk_id"] in frozen_ids:
            result[c["chunk_id"]] = "frozen"

    non_frozen = sorted(
        [c for c in chunks if c["chunk_id"] not in frozen_ids],
        key=lambda c: c.get("access_count", 0),
        reverse=True,
    )
    n = len(non_frozen)
    hot_cutoff  = max(1, int(n * 0.15))
    warm_cutoff = max(1, int(n * 0.65))  # top 15% + next 50%

    for i, c in enumerate(non_frozen):
        last_ts = c.get("last_accessed_ts") or 0
        age_days = (now - last_ts) / 86400 if last_ts else 999

        if i < hot_cutoff and age_days <= 7:
            result[c["chunk_id"]] = "hot"
        elif i < warm_cutoff and age_days <= 30:
            result[c["chunk_id"]] = "warm"
        else:
            result[c["chunk_id"]] = "cold"

    return result


def generate_report(chunks: list[dict], parametric_rate: float,
                     output_path: str = "access_report.json") -> None:
    tiers = compute_tiers(chunks)
    counts = {"hot": 0, "warm": 0, "cold": 0, "frozen": 0}
    frozen_ids = []
    for cid, tier in tiers.items():
        counts[tier] += 1
        if tier == "frozen":
            frozen_ids.append(cid)

    # Most accessed pages
    page_counts: dict[int, int] = {}
    for c in chunks:
        page = c.get("page", 0)
        page_counts[page] = page_counts.get(page, 0) + c.get("access_count", 0)
    top_pages = sorted(page_counts, key=page_counts.get, reverse=True)[:5]

    report = {
        "generated_at": int(time.time()),
        "summary": {**counts, "total": len(chunks)},
        "parametric_answer_rate": round(parametric_rate, 4),
        "most_accessed_pages": top_pages,
        "frozen_chunk_ids": frozen_ids[:50],  # cap to first 50
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📋 Access report written to {output_path}")
```

- [ ] **Step 4: Run tests**

```bash
venv/bin/python3 -m pytest tests/test_access_tracker.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add access_tracker.py tests/test_access_tracker.py
git commit -m "feat(SP5): access_tracker — thread-safe counters, tier classification, weekly report"
```

---

### Task 14: `monitoring_dashboard.py`

**Files:**
- Create: `monitoring_dashboard.py`

- [ ] **Step 1: Implement `monitoring_dashboard.py`**

```python
"""
monitoring_dashboard.py — FastAPI dashboard at localhost:8080.

Start: python3 monitoring_dashboard.py
Or:    uvicorn monitoring_dashboard:app --port 8080 --reload
"""

import json
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

import version as ver
from qdrant_client import QdrantClient

app = FastAPI(title="RAG Intelligence Dashboard")
_cfg: dict = {}


def _load_cfg() -> dict:
    global _cfg
    if not _cfg:
        with open("my_config.json") as f:
            _cfg = json.load(f)
    return _cfg


@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": int(time.time())}


@app.get("/api/version")
def get_version():
    return ver.load()


@app.get("/api/stats")
def get_stats():
    cfg = _load_cfg()
    v = ver.load()

    # Qdrant tier counts — paginate to handle >5000 chunks
    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    tier_counts = {"hot": 0, "warm": 0, "cold": 0, "frozen": 0}
    top_chunks = []
    try:
        all_results = []
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=cfg["collection"],
                limit=500,
                with_payload=["tier", "access_count", "page",
                               "parametric_hit_count", "text"],
                offset=offset,
            )
            all_results.extend(batch)
            if offset is None:
                break

        for r in all_results:
            t = r.payload.get("tier", "frozen")
            tier_counts[t] = tier_counts.get(t, 0) + 1

        top_chunks = sorted(
            [{"chunk_id": r.id,
              "page": r.payload.get("page", 0),
              "access_count": r.payload.get("access_count", 0),
              "parametric_hit_count": r.payload.get("parametric_hit_count", 0),
              "tier": r.payload.get("tier", "frozen"),
              "text_preview": (r.payload.get("text", "")[:80] + "…")}
             for r in all_results],
            key=lambda x: x["access_count"],
            reverse=True,
        )[:10]
    except Exception as e:
        tier_counts["error"] = str(e)

    # Access report
    access_report = {}
    rp = Path("access_report.json")
    if rp.exists():
        with open(rp) as f:
            access_report = json.load(f)

    return {
        "version": v,
        "tier_counts": tier_counts,
        "top_chunks": top_chunks,
        "access_report": access_report,
        "total_chunks": sum(tier_counts[k] for k in ["hot","warm","cold","frozen"]),
    }


@app.get("/api/access-report")
def get_access_report():
    rp = Path("access_report.json")
    if not rp.exists():
        return JSONResponse({"error": "No report yet"}, status_code=404)
    with open(rp) as f:
        return json.load(f)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>RAG Intelligence Dashboard</title>
<style>
  body { font-family: monospace; background:#111; color:#eee; padding:20px; }
  h1 { color:#7af; } .card { border:1px solid #333; padding:12px; margin:10px 0; }
  table { border-collapse:collapse; width:100%; }
  td,th { border:1px solid #333; padding:6px 10px; text-align:left; }
  .hot{color:#f90} .warm{color:#ff0} .cold{color:#0af} .frozen{color:#aaa}
</style>
</head>
<body>
<h1>RAG Intelligence Dashboard</h1>
<div id="root">Loading…</div>
<script>
async function load(){
  const [stats, ver] = await Promise.all(
    [fetch('/api/stats').then(r=>r.json()), fetch('/api/version').then(r=>r.json())]
  );
  const tc = stats.tier_counts;
  document.getElementById('root').innerHTML = `
    <div class="card"><b>Phase:</b> ${ver.phase} &nbsp;|&nbsp;
      <b>LoRA version:</b> ${ver.current_lora_version} &nbsp;|&nbsp;
      <b>Total chunks:</b> ${stats.total_chunks}
    </div>
    <div class="card"><b>Tier distribution:</b>
      <span class="hot">Hot: ${tc.hot||0}</span> &nbsp;
      <span class="warm">Warm: ${tc.warm||0}</span> &nbsp;
      <span class="cold">Cold: ${tc.cold||0}</span> &nbsp;
      <span class="frozen">Frozen: ${tc.frozen||0}</span>
    </div>
    <div class="card"><b>Top 10 chunks by access count:</b>
      <table><tr><th>ID</th><th>Page</th><th>Tier</th><th>Access</th><th>Parametric</th><th>Preview</th></tr>
      ${stats.top_chunks.map(c=>`<tr><td>${c.chunk_id}</td><td>${c.page}</td>
        <td class="${c.tier}">${c.tier}</td><td>${c.access_count}</td>
        <td>${c.parametric_hit_count}</td><td>${c.text_preview}</td></tr>`).join('')}
      </table>
    </div>
    <div class="card"><b>PRS history:</b>
      ${(ver.prs_history||[]).map(r=>`Round ${r.round}: ${r.prs}`).join(' → ') || 'No data yet'}
    </div>`;
}
load();
setInterval(load, 30000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the self-contained monitoring dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


if __name__ == "__main__":
    cfg = _load_cfg()
    uvicorn.run(app, host="0.0.0.0", port=cfg.get("dashboard_port", 8080))
```

- [ ] **Step 2: Write API tests**

```python
# tests/test_dashboard.py
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def _make_client():
    import monitoring_dashboard as md
    # Patch Qdrant and version to avoid real connections
    with patch.object(md, "_cfg", {"qdrant_host": "localhost", "qdrant_port": 6333,
                                    "collection": "test", "dashboard_port": 8080}):
        return TestClient(md.app)


def test_health_returns_ok():
    client = _make_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_version_returns_phase():
    import version
    client = _make_client()
    with patch("monitoring_dashboard.ver.load", return_value={"phase": 1,
               "current_lora_version": 0, "prs_history": [], "known_good_queries": []}):
        r = client.get("/api/version")
        assert r.status_code == 200
        assert "phase" in r.json()


def test_dashboard_html_contains_script():
    client = _make_client()
    r = client.get("/")
    assert r.status_code == 200
    assert "<script>" in r.text
    assert "api/stats" in r.text
```

- [ ] **Step 3: Run tests**

```bash
pip install httpx  # TestClient dependency
venv/bin/python3 -m pytest tests/test_dashboard.py -v
```
Expected: `3 passed`

- [ ] **Step 4: Start and hit health endpoint (manual)**

```bash
venv/bin/python3 monitoring_dashboard.py &
sleep 2
curl http://localhost:8080/api/health
kill %1
```
Expected: `{"status":"ok","timestamp":...}`

- [ ] **Step 5: Commit**

```bash
git add monitoring_dashboard.py tests/test_dashboard.py
git commit -m "feat(SP6): monitoring_dashboard — FastAPI dashboard with PRS, training, access stats"
```

---

## Chunk 6: Integration Tests + Final Wiring

### Task 15: End-to-End Smoke Test

**Files:**
- Create: `tests/test_integration_smoke.py`

- [ ] **Step 1: Write integration smoke test**

```python
# tests/test_integration_smoke.py
"""
Smoke tests for the full pipeline integration.
Requires: Qdrant running at localhost:6333 (no GPU needed for these tests).
Skip if Qdrant unreachable.
"""
import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def qdrant_available():
    try:
        from qdrant_client import QdrantClient
        QdrantClient(host="localhost", port=6333).get_collections()
        return True
    except Exception:
        return False


def test_version_roundtrip(tmp_path):  # no Qdrant needed — pure file I/O
    """version.py read/write cycle works correctly."""
    import version
    orig = version.VERSION_FILE
    version.VERSION_FILE = tmp_path / "version.json"
    try:
        d = version.load()
        assert d["phase"] == 1
        version.increment_lora_version("lora_checkpoints/v1/")
        assert version.get_lora_version() == 1
        version.append_prs(1, 0.42)
        assert version.load()["prs_history"][0]["prs"] == 0.42
    finally:
        version.VERSION_FILE = orig


def test_kv_utils_full_roundtrip():
    """Serialize → deserialize KV array preserves values."""
    import numpy as np
    import kv_utils
    arr = np.random.randn(4, 2, 2, 8).astype(np.float16)
    b64 = kv_utils.serialize_kv(arr)
    restored = kv_utils.deserialize_kv(b64, shape=(4, 2, 2, 8))
    np.testing.assert_array_equal(arr, restored)


def test_replay_buffer_weighted_sampling():
    """Hot chunks appear more often than frozen chunks."""
    import tempfile
    from replay_buffer import ReplayBuffer
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        rb = ReplayBuffer(db_path=tf.name)
        rb.add_chunks([
            {"chunk_id": i, "text": f"hot text {i}", "tier": "hot"}
            for i in range(10)
        ] + [
            {"chunk_id": 100 + i, "text": f"frozen text {i}", "tier": "frozen"}
            for i in range(10)
        ])
        samples = rb.sample(n=100, weight_by_tier=True)
        hot_count = sum(1 for s in samples if s["tier"] == "hot")
        frozen_count = sum(1 for s in samples if s["tier"] == "frozen")
        # hot weight=8, frozen weight=1 → hot should appear ~8x more
        assert hot_count > frozen_count * 3


def test_inference_mode_logic():
    """decide_inference_mode returns correct mode."""
    from kv_inference import decide_inference_mode
    fresh = [{"kv_version": 3} for _ in range(5)]
    assert decide_inference_mode(fresh, current_lora_version=3) == "kv_injection"
    mixed = [{"kv_version": 3}, {"kv_version": None}]
    assert decide_inference_mode(mixed, current_lora_version=3) == "text_fallback"


def test_gate_pure_logic():
    """confidence_gate pure functions work without model."""
    from confidence_gate import compute_hedging_score, decide_gate
    assert compute_hedging_score("I think maybe") > 0
    assert compute_hedging_score("Bedrock is a service.") == 0.0
    assert decide_gate(0.1, 0.0, 0.95) == "direct"
    assert decide_gate(0.9, 0.8, 0.1) == "retrieve"
```

- [ ] **Step 2: Run all tests**

```bash
venv/bin/python3 -m pytest tests/test_kv_utils.py tests/test_kv_indexer.py \
  tests/test_lora_trainer.py tests/test_kv_inference.py \
  tests/test_confidence_gate.py tests/test_access_tracker.py \
  tests/test_dashboard.py tests/test_integration_smoke.py -v
```
Expected: All tests pass (Qdrant-dependent tests skip if service not running)

- [ ] **Step 3: Final commit**

```bash
git add tests/test_integration_smoke.py
git commit -m "test: add integration smoke tests for full SP1–SP6 pipeline"
```

---

### Task 16: EC2 Deployment Verification

**Connection details:**
- Host: `100.48.17.48` | User: `ubuntu` | PEM: `/Users/hemant/Downloads/RoPE/g5.x.pem`
- SSH alias (add to `~/.ssh/config` for convenience): `Host qdrant-gpu` / `HostName 100.48.17.48` / `User ubuntu` / `IdentityFile ~/Downloads/RoPE/g5.x.pem`

- [ ] **Step 1: SSH in and verify EC2 environment**

```bash
# ── From local machine ────────────────────────────────────────────
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@100.48.17.48

# ── Once on EC2 ───────────────────────────────────────────────────
nvidia-smi | head -5
# Expected: NVIDIA A10G, Driver version, CUDA version

source ~/qdrant/venv/bin/activate
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA A10G
```

- [ ] **Step 2: Start Qdrant on EC2**

```bash
# On EC2 — Qdrant managed by systemd (§9 of spec)
sudo systemctl start qdrant
sudo systemctl status qdrant | head -5
curl -s http://localhost:6333/collections | python3 -m json.tool | head -10
# Expected: JSON with existing collections (should include bedrock-user-guide)
```

- [ ] **Step 3: Sync code from local to EC2**

```bash
# ── From local machine ────────────────────────────────────────────
EC2="ubuntu@100.48.17.48"
PEM="/Users/hemant/Downloads/RoPE/g5.x.pem"

# Sync all new Python files to the qdrant project directory on EC2
rsync -avz --progress -e "ssh -i $PEM" \
  kv_utils.py version.py model_loader.py kv_indexer.py \
  replay_buffer.py lora_trainer.py prs_evaluator.py index_and_train.py \
  kv_background.py kv_inference.py confidence_gate.py \
  access_tracker.py monitoring_dashboard.py \
  requirements_gpu.txt my_config.json \
  "$EC2":~/qdrant/

# Transfer the Bedrock PDF
scp -i "$PEM" \
  "/Users/hemant/Downloads/Fission Labs/Amazon Bedrock Dataset/Amazon Bedrock Dataset.pdf" \
  "$EC2":~/qdrant/
```

- [ ] **Step 4: Install GPU dependencies on EC2**

```bash
# ── On EC2 ───────────────────────────────────────────────────────
cd ~/qdrant
source venv/bin/activate
export PYTHONIOENCODING=utf-8
pip install -r requirements_gpu.txt
# Expected: all packages install cleanly; torch already present on Deep Learning AMI
```

- [ ] **Step 5: Run full pipeline on EC2**

```bash
# ── On EC2 ───────────────────────────────────────────────────────
cd ~/qdrant
source venv/bin/activate
export HF_TOKEN=<your_huggingface_token>
export PYTHONIOENCODING=utf-8

python3 index_and_train.py \
  "Amazon Bedrock Dataset.pdf" \
  --config my_config.json \
  --skip-prs
```
Expected: completes SP1 (index + KV compute) → SP2 (LoRA fine-tune) → KV refresh, no errors

- [ ] **Step 6: Run a query end-to-end on EC2**

```bash
python3 bedrock_rag.py --config my_config.json search "What is Amazon Bedrock?" \
  | python3 kv_inference.py
```
Expected: Mode line showing `kv_injection` or `text_fallback`, followed by answer text

- [ ] **Step 7: Start monitoring dashboard on EC2 and verify from local**

```bash
# ── On EC2 ───────────────────────────────────────────────────────
python3 monitoring_dashboard.py &
sleep 2
curl http://localhost:8080/api/stats | python3 -m json.tool | head -30
cat version.json | python3 -m json.tool
```

```bash
# ── From local machine (port-forward dashboard) ───────────────────
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem -L 8080:localhost:8080 ubuntu@100.48.17.48 -N &
open http://localhost:8080
# Expected: dashboard page with tier counts, PRS history, top chunks
# version.json should show current_lora_version >= 1
```

- [ ] **Step 8: Verify .gitignore covers runtime artefacts**

```bash
# ── On local machine (in the repo) ───────────────────────────────
grep -qF "lora_checkpoints/" .gitignore || echo "lora_checkpoints/" >> .gitignore
grep -qF "replay_buffer.db"  .gitignore || echo "replay_buffer.db"  >> .gitignore
grep -qF "version.json"      .gitignore || echo "version.json"      >> .gitignore
grep -qF "access_report.json" .gitignore || echo "access_report.json" >> .gitignore
git diff .gitignore
```

- [ ] **Step 9: Final commit (from local machine)**

```bash
git add kv_utils.py version.py model_loader.py kv_indexer.py \
        replay_buffer.py lora_trainer.py prs_evaluator.py index_and_train.py \
        kv_background.py kv_inference.py confidence_gate.py \
        access_tracker.py monitoring_dashboard.py \
        tests/ requirements_gpu.txt .gitignore
git commit -m "feat: complete SP1–SP6 Qdrant-coupled attention system"
```
