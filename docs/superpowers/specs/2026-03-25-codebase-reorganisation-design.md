# KVForge Codebase Reorganisation — Design Spec

**Date:** 2026-03-25
**Branch:** kvforge-main
**Approach:** Surgical Reorganization (Approach A)

---

## Goal

Reorganise the KVForge repository to achieve three outcomes:

1. **Organised codebase** — pipeline/orchestration scripts separated from user-facing CLIs and utilities; Qdrant-internal tests labelled and isolated; legacy files removed from root.
2. **Complete project documentation** — hand-written guides + API reference + pdoc auto-generation config.
3. **Documented shell scripts** — every runnable Python tool gets a corresponding `.sh` wrapper in `scripts/` with usage documentation.

All existing tests must pass after every structural change.

---

## Constraints

- Do not move `embeddings/`, `ingestion/`, `vectorstore/`, or `tools/` — their structure is clean.
- Do not refactor logic — only move files, update imports, and add docs/scripts.
- Keep `kvforge.py` and `ask.py` at root (they are the primary user entry points).
- All tests in `tests/test_*.py` must continue to pass after the migration.

---

## Section 1: Directory Structure

### Target Layout

```
kvforge/
├── kvforge.py              ← main CLI (stays at root)
├── ask.py                      ← query CLI (stays at root)
├── config.py                   ← Pydantic DatasourceConfig (stays at root)
├── kv_utils.py                 ← KV tensor ops (stays at root)
├── model_loader.py             ← LLM singleton (stays at root)
├── version.py                  ← version state (stays at root)
├── confidence_gate.py          ← phase 3 gate (stays at root)
├── replay_buffer.py            ← SQLite replay (stays at root)
├── access_tracker.py           ← tier tracking (stays at root)
│
├── pipeline/                   ← NEW PACKAGE (scripts moved from root)
│   ├── __init__.py             ← empty init
│   ├── bedrock_rag.py          ← MOVED from root (legacy; active importers remain)
│   ├── kv_indexer.py           ← moved from root
│   ├── kv_inference.py         ← moved from root
│   ├── kv_background.py        ← moved from root
│   ├── lora_trainer.py         ← moved from root
│   ├── prs_evaluator.py        ← moved from root
│   ├── monitoring_dashboard.py ← moved from root
│   └── index_and_train.py      ← moved from root
│
├── embeddings/                 ← unchanged
├── ingestion/                  ← unchanged
├── vectorstore/                ← unchanged
│
├── tools/                      ← one addition
│   ├── generate_faqs.py        ← unchanged
│   └── gen_viewer.py           ← MOVED from root
│
├── scripts/                    ← NEW DIRECTORY (all shell wrappers)
│   ├── README.md               ← script catalog
│   ├── ask.sh
│   ├── dashboard.sh
│   ├── index.sh
│   ├── compute_kv.sh
│   ├── train_lora.sh
│   ├── evaluate_prs.sh
│   ├── generate_faqs.sh
│   ├── generate_docs.sh
│   └── run_pipeline.sh         ← full Phase 1→2→3 orchestration
│
├── tests/
│   ├── test_*.py               ← KVForge-specific (unchanged except import paths)
│   └── qdrant_internal/        ← MOVED from tests/ root
│       ├── README.md           ← label: upstream Qdrant tests, not KVForge
│       ├── consensus_tests/
│       ├── e2e_tests/
│       └── openapi/
│
├── examples/                   ← run_pipeline.sh + README.md updated; setup.py unchanged
│   ├── usecase1_customer_support/
│   ├── usecase2_pubmedqa/
│   └── usecase3_squad/
│
├── docs/
│   ├── faq/                    ← unchanged (7 topic pages)
│   ├── api/                    ← NEW: hand-written API reference
│   │   ├── index.md
│   │   ├── config.md
│   │   ├── pipeline.md
│   │   ├── vectorstore.md
│   │   ├── embeddings.md
│   │   └── ingestion.md
│   ├── guides/                 ← NEW: developer + operations guides
│   │   ├── quickstart.md
│   │   ├── architecture.md
│   │   ├── adding-backends.md
│   │   └── troubleshooting.md
│   └── superpowers/            ← unchanged (planning docs)
│
├── pdoc.toml                   ← NEW: pdoc config
├── README.md                   ← updated
├── FAQ.md                      ← unchanged
├── requirements_gpu.txt        ← unchanged
├── datasource_template.json    ← unchanged
└── datasource_bedrock.json     ← unchanged
```

### Files Moved from Root

| File | New Location | Note |
|------|-------------|------|
| `bedrock_rag.py` | `pipeline/bedrock_rag.py` | Legacy; still imported by `ask.py`, `kv_indexer.py`, `kv_inference.py`, `confidence_gate.py`, and test files |
| `kv_indexer.py` | `pipeline/kv_indexer.py` | Pipeline orchestration |
| `kv_inference.py` | `pipeline/kv_inference.py` | Pipeline orchestration |
| `kv_background.py` | `pipeline/kv_background.py` | Pipeline daemon |
| `lora_trainer.py` | `pipeline/lora_trainer.py` | Pipeline trainer |
| `prs_evaluator.py` | `pipeline/prs_evaluator.py` | Pipeline evaluator |
| `monitoring_dashboard.py` | `pipeline/monitoring_dashboard.py` | FastAPI service |
| `index_and_train.py` | `pipeline/index_and_train.py` | Orchestrator entry point |
| `gen_viewer.py` | `tools/gen_viewer.py` | Tool, not a root entry point |

---

## Section 2: Import Updates

All import changes are mechanical — no logic changes. Every file listed here requires only path updates to module names.

### 2a. Imports of `bedrock_rag` (from-import and bare-import)

| File | Current line | Updated line |
|------|-------------|-------------|
| `ask.py` line 43 | `import kv_background` | `import pipeline.kv_background as kv_background` |
| `ask.py` line 46 | `from bedrock_rag import _run_search, Config` | `from pipeline.bedrock_rag import _run_search, Config` |
| `confidence_gate.py` line 31 | `import kv_background` | `import pipeline.kv_background as kv_background` |
| `confidence_gate.py` line 184 | `from kv_inference import answer_with_retrieval` | `from pipeline.kv_inference import answer_with_retrieval` |
| `confidence_gate.py` line 210 | `from kv_inference import answer_with_retrieval` | `from pipeline.kv_inference import answer_with_retrieval` |
| `confidence_gate.py` line 228 | `from bedrock_rag import Config, _run_search` | `from pipeline.bedrock_rag import Config, _run_search` |
| `pipeline/kv_inference.py` line 19 | `import kv_background` | `import pipeline.kv_background as kv_background` |
| `pipeline/kv_inference.py` line 22 | `from bedrock_rag import _run_search, Config` | `from pipeline.bedrock_rag import _run_search, Config` |

### 2b. Pipeline-internal cross-imports (files inside `pipeline/`)

| File | Current line | Updated line |
|------|-------------|-------------|
| `pipeline/kv_indexer.py` line 34 | `from bedrock_rag import chunk_pages, read_pdf, embed_chunks` | `from pipeline.bedrock_rag import chunk_pages, read_pdf, embed_chunks` |
| `pipeline/kv_background.py` | `from kv_indexer import ...` | `from pipeline.kv_indexer import ...` |
| `pipeline/kv_background.py` | `import kv_utils`, `import model_loader`, `import version` | unchanged (root modules stay at root) |
| `pipeline/kv_inference.py` | `from kv_utils import ...`, `from model_loader import ...` | unchanged |
| `pipeline/prs_evaluator.py` line 158 | `from kv_inference import answer_with_retrieval` | `from pipeline.kv_inference import answer_with_retrieval` |
| `pipeline/monitoring_dashboard.py` line 60 | `import kv_background as _kb` | `import pipeline.kv_background as _kb` |
| `pipeline/monitoring_dashboard.py` line 61 | `import kv_inference as _ki` | `import pipeline.kv_inference as _ki` |
| `pipeline/monitoring_dashboard.py` line 282 | `from bedrock_rag import _run_search, Config` | `from pipeline.bedrock_rag import _run_search, Config` |
| `pipeline/monitoring_dashboard.py` | `from version import ...`, `from config import ...` | unchanged |

### 2c. Subprocess script references in `pipeline/index_and_train.py`

`index_and_train.py` has **5** `run([py, "<script>.py", ...])` call-sites. All must change from positional filename to `-m` module form:

| Line | Current | Updated |
|------|---------|---------|
| 42 | `run([py, "kv_indexer.py", "--config", args.config, "index", str(pdf)])` | `run([py, "-m", "pipeline.kv_indexer", "--config", args.config, "index", str(pdf)])` |
| 46 | `run([py, "lora_trainer.py", ...])` | `run([py, "-m", "pipeline.lora_trainer", ...])` |
| 53 | `run([py, "kv_indexer.py", "--config", args.config, "compute-kv", ...])` | `run([py, "-m", "pipeline.kv_indexer", "--config", args.config, "compute-kv", ...])` |
| 68 | `run([py, "kv_indexer.py", "--config", args.config, "compute-kv", ...])` | `run([py, "-m", "pipeline.kv_indexer", "--config", args.config, "compute-kv", ...])` |
| 74 | `run([py, "prs_evaluator.py", ...])` | `run([py, "-m", "pipeline.prs_evaluator", ...])` |

### 2d. Example shell scripts (`examples/usecase*/run_pipeline.sh`) and READMEs

Each of the 3 use-case directories has a `run_pipeline.sh` **and** a `README.md` with hardcoded `python <script>.py` invocations. Both must be updated for `kv_indexer.py`, `lora_trainer.py`, `prs_evaluator.py`, and `monitoring_dashboard.py`:

| Reference | Old form | New form |
|-----------|----------|---------|
| `run_pipeline.sh` and `README.md` (×3 dirs) | `python kv_indexer.py` | `python -m pipeline.kv_indexer` |
| `run_pipeline.sh` and `README.md` (×3 dirs) | `python lora_trainer.py` | `python -m pipeline.lora_trainer` |
| `run_pipeline.sh` and `README.md` (×3 dirs) | `python prs_evaluator.py` | `python -m pipeline.prs_evaluator` |
| `README.md` (×3 dirs) | `python monitoring_dashboard.py` | `python -m pipeline.monitoring_dashboard` |

### 2e. Test files that import pipeline modules or `bedrock_rag`

| Test file | Current import | Updated import |
|-----------|---------------|----------------|
| `tests/test_kv_indexer.py` | `from kv_indexer import ...` | `from pipeline.kv_indexer import ...` |
| `tests/test_kv_inference.py` | `from kv_inference import ...` | `from pipeline.kv_inference import ...` |
| `tests/test_lora_trainer.py` | `from lora_trainer import ...` | `from pipeline.lora_trainer import ...` |
| `tests/test_prs_evaluator.py` | `from prs_evaluator import ...` | `from pipeline.prs_evaluator import ...` |
| `tests/test_dashboard.py` line 11 | `import monitoring_dashboard as md` | `import pipeline.monitoring_dashboard as md` |
| `tests/test_embeddings.py` (×2, lines 7 and 19) | `from bedrock_rag import validate_embed_dim` | `from pipeline.bedrock_rag import validate_embed_dim` |
| `tests/test_integration_smoke.py` line 19 | `from kv_inference import decide_inference_mode` | `from pipeline.kv_inference import decide_inference_mode` |
| `tests/test_ingestion.py` line 138 | `from bedrock_rag import Config, cmd_index` | `from pipeline.bedrock_rag import Config, cmd_index` |
| `tests/test_ingestion.py` line 122 | `patch("bedrock_rag.TextEmbedding")` | `patch("pipeline.bedrock_rag.TextEmbedding")` |
| `tests/test_ingestion.py` line 123 | `patch("vectorstore.qdrant_store.QdrantClient")` | unchanged |

Note: `tests/test_kv_background.py` does **not** exist in the repository — no action needed for that filename.

### 2f. `pipeline/__init__.py`

Empty file — the package exposes no public API at the package level.

---

## Section 3: Shell Scripts

### Common pattern

Every script in `scripts/` follows this template:

```bash
#!/usr/bin/env bash
# ============================================================
# <Script name>
# Description: <one-line description>
#
# Usage:
#   scripts/<script>.sh <arg1> [arg2]
#
# Required arguments:
#   CONFIG  Path to datasource JSON config file
#   ...
#
# Example:
#   ./scripts/<script>.sh datasource_my-corpus.json
# ============================================================
set -euo pipefail

# Validate Python
if ! command -v python &>/dev/null; then
  echo "Error: python not found in PATH" >&2; exit 1
fi

# Validate config file argument
CONFIG="${1:-}"
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "Usage: $0 <config.json> ..." >&2; exit 1
fi
```

### Script inventory

| Script | Wraps | Required args | Optional args |
|--------|-------|---------------|---------------|
| `ask.sh` | `ask.py` | `CONFIG` `QUERY` | — |
| `dashboard.sh` | `python -m pipeline.monitoring_dashboard` | `CONFIG` | `PORT` (default 8080) |
| `index.sh` | `kvforge.py index` | `CONFIG` `SOURCE` | — |
| `compute_kv.sh` | `python -m pipeline.kv_indexer compute-kv` | `CONFIG` | — |
| `train_lora.sh` | `python -m pipeline.lora_trainer` | `CONFIG` `FAQS` | — |
| `evaluate_prs.sh` | `python -m pipeline.prs_evaluator` | `CONFIG` `FAQS` | — |
| `generate_faqs.sh` | `tools/generate_faqs.py` | `CONFIG` `OUTPUT` | `COUNT` (default 50) |
| `generate_docs.sh` | `pdoc` | — | `OUTPUT_DIR` (default `docs/api/generated`) |
| `run_pipeline.sh` | all above in sequence | `CONFIG` `SOURCE` `FAQS` | — |

### `scripts/README.md`

A markdown table of every script: description, usage syntax, required args, optional args, example invocation, and which underlying Python module it calls.

---

## Section 4: Documentation

### Hand-written Markdown

#### `docs/guides/quickstart.md`
- Prerequisites (Python 3.10+, GPU optional, Qdrant Docker)
- Step-by-step: install deps → `kvforge.py init` → `scripts/index.sh` → `scripts/ask.sh`
- Points to `examples/` for full end-to-end pipelines

#### `docs/guides/architecture.md`
- 3-phase pipeline overview (Phase 1: retrieval, Phase 2: KV injection, Phase 3: parametric)
- Mermaid data-flow diagram
- Module responsibility map (root utils, `pipeline/`, `vectorstore/`, `embeddings/`, `ingestion/`)
- Tier system (hot/warm/cold/frozen) and how it drives KV healing and LoRA training weights
- PRS gate logic and phase transition conditions

#### `docs/guides/adding-backends.md`
- Step-by-step: implement `VectorStore` protocol → create `vectorstore/<name>_store.py` → add `elif` in `vectorstore/registry.py`
- Same pattern for `Embedder` and `DocumentLoader`
- Example skeleton code for each

#### `docs/guides/troubleshooting.md`
- GPU OOM: reduce `embed_batch`, use 8-bit quantization
- Qdrant unreachable: Docker not running, wrong host/port
- PRS not improving: check FAQ quality, lower `prs_threshold`, increase `lora_epochs`
- KV shape mismatch: model changed, rerun `scripts/compute_kv.sh`
- ChromaDB tenant errors: wrong `chroma_persist_dir`
- FAISS index not found: `.index`/`.meta.pkl` deleted, re-index

#### `docs/api/index.md`
Overview table of all modules with one-line descriptions and links to detail pages.

#### `docs/api/config.md`
Full table of every `DatasourceConfig` field: name, type, default, description, and which scripts use it.

#### `docs/api/pipeline.md`
For each module in `pipeline/`: CLI synopsis, all flags, programmatic import API, example.

#### `docs/api/vectorstore.md`
VectorStore protocol method signatures, `Point`/`ScoredPoint` dataclasses, backend comparison table (Qdrant vs ChromaDB vs FAISS).

#### `docs/api/embeddings.md`
Embedder protocol, per-backend install command and configuration fields.

#### `docs/api/ingestion.md`
DocumentLoader protocol, per-loader options, output schema (`{"text": ..., "metadata": {...}}`).

### Auto-generated (pdoc)

`pdoc.toml`:
```toml
[pdoc]
output-directory = "docs/api/generated"
docformat = "google"
```

`scripts/generate_docs.sh` runs (pdoc 14+ syntax, no `--html` flag):
```bash
pdoc -o docs/api/generated \
  config kv_utils model_loader version confidence_gate \
  replay_buffer access_tracker \
  pipeline embeddings ingestion vectorstore tools
```

`docs/api/generated/` is added to `.gitignore` — generated on demand, not committed.

---

## Section 5: README Update

The root `README.md` gains three new sections:

1. **Project Structure** — annotated directory tree showing `pipeline/`, `scripts/`, `docs/` layout
2. **Scripts** — table linking to `scripts/README.md`, with the three most common commands shown
3. **Documentation** — links to `docs/guides/quickstart.md`, `docs/api/index.md`, `FAQ.md`

Existing sections (Architecture, Getting Started, Config Reference, Module Table, Tests, EC2 Deployment) are updated to reflect `pipeline/` module paths.

---

## Section 6: Verification Plan

After **each** step, run:
```bash
pytest tests/test_*.py -x -q
```
All tests must pass before proceeding.

### Step 1 — Create `pipeline/` package and move all scripts

1a. Create `pipeline/__init__.py` (empty).
1b. `git mv` all 9 scripts from root to `pipeline/`.
1c. Update all imports per Section 2a (bedrock_rag + kv_background bare imports).
1d. Update all imports per Section 2b (pipeline-internal).
1e. Update all 5 subprocess call-sites in `pipeline/index_and_train.py` (Section 2c).
1f. Update all 3 `examples/*/run_pipeline.sh` files and all 3 `examples/*/README.md` files (Section 2d).
1g. Update all test files per Section 2e.
1h. Run `pytest tests/test_*.py -x -q` — must pass.

### Step 2 — Move `gen_viewer.py` to `tools/`

2a. `git mv gen_viewer.py tools/gen_viewer.py`
2b. Run `pytest tests/test_*.py -x -q` — must pass (no tests import `gen_viewer`).

### Step 3 — Move Qdrant-internal tests to `tests/qdrant_internal/`

3a. Create `tests/qdrant_internal/README.md` explaining these are upstream Qdrant tests, not KVForge.
3b. `git mv tests/consensus_tests tests/qdrant_internal/consensus_tests`
3c. `git mv tests/e2e_tests tests/qdrant_internal/e2e_tests`
3d. `git mv tests/openapi tests/qdrant_internal/openapi`
3e. Run `pytest tests/test_*.py -x -q` — must pass.

### Step 4 — Add `scripts/` shell wrappers

4a. Create `scripts/` directory with all 9 `.sh` scripts.
4b. `chmod +x scripts/*.sh`
4c. `bash -n scripts/*.sh` — all must exit 0 (syntax-only check).
4d. Run `pytest tests/test_*.py -x -q` — must pass.

### Step 5 — Add documentation

5a. Create `docs/guides/` with 4 markdown files.
5b. Create `docs/api/` with 6 markdown files.
5c. Add `pdoc.toml`.
5d. Update `README.md`.
5e. Update `.gitignore` to include `docs/api/generated/`.
5f. Run `pytest tests/test_*.py -x -q` — must pass.

### Step 6 — Final check

6a. Run full test suite: `pytest tests/test_*.py -v`
6b. Syntax-check all scripts: `bash -n scripts/*.sh`
6c. Verify `git status` is clean.
6d. Commit and push.

---

## Out of Scope

- No logic changes to any Python module
- No new runtime Python dependencies (`pdoc` is a dev-only tool, added to `requirements_gpu.txt` with a `# dev` comment)
- No changes to `embeddings/`, `ingestion/`, or `vectorstore/` packages
- No changes to `examples/*/setup.py`
- No conversion to a pip-installable package (`pyproject.toml`)
