# KVForge Consolidated Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all five KVForge feature tracks in dependency order: Dynamic PRS → Flywheel Analytics → VDB Expansion → ModelScout → Multimodal Image Vectors.

**Architecture:** Five tracks share `core/config.py` (combined in Task 0) and three existing pipeline files (`prs_evaluator.py`, `kv_inference.py`, `studio/routes.py`). All shared-file modifications are sequenced to avoid conflicts. All new-file creation is independent and parallelisable. Every task is TDD: failing test → minimal implementation → passing test → commit.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, transformers (HuggingFace), pdfplumber, Pillow, psycopg2, pgvector, pymilvus, weaviate-client, pinecone, SQLite (stdlib), FastAPI, Pydantic v2.

---

## Conflict Resolution Decisions

| Conflict | Resolution |
|---|---|
| `core/config.py` touched by all 5 tracks | **Task 0** adds all 35 fields in one commit |
| `prs_evaluator.py` touched by Dynamic PRS + Flywheel | Dynamic PRS hook runs first (per-cluster PRS compute), Flywheel `record_round()` runs second |
| `kv_inference.py` touched by Dynamic PRS + Multimodal | Multimodal adds `extra_context` param (Task P3-a); Dynamic PRS adds `route_query()` as a new function (Task P3-b) — no signature clash |
| `studio/routes.py` touched by Dynamic PRS + ModelScout + Flywheel | One task (Task P4) adds all three imports/routers together |
| `core/version.py` cluster state | Dynamic PRS Task 3 extends version schema to support per-cluster dict; Flywheel reads the same dict |
| `kv_background.py` | Multimodal Task 10 adds `_image_kv_queue` + `_image_kv_worker` as additive changes only |

---

## Dependency Graph

```
Task 0  (config.py — all fields)
  │
  ├── Phase 1: New files only (fully independent, parallelisable)
  │     ├── P1-A: VDB store implementations (4 files)
  │     ├── P1-B: core/analytics.py + studio/flywheel_routes.py
  │     ├── P1-C: core/difficulty_estimators.py + core/cluster_manager.py + core/prs_adapter.py
  │     ├── P1-D: core/model_registry.py + core/model_registry.json + pipeline/model_scout.py + pipeline/model_scout_cli.py
  │     └── P1-E: core/multimodal_loader.py + embeddings/clip_embedder.py + ingestion/image_extractor.py
  │                + pipeline/image_inference.py + pipeline/image_indexer.py + pipeline/multimodal_query.py
  │                + pipeline/query_logger.py + pipeline/chunk_confidence.py
  │
  ├── Phase 2: Registry updates (after Phase 1)
  │     ├── P2-A: vectorstore/registry.py (register_store + 4 new backends)
  │     └── P2-B: embeddings/registry.py (add "clip" backend)
  │
  ├── Phase 3: Pipeline modifications to existing files (sequential)
  │     ├── P3-a: pipeline/kv_inference.py — extra_context param + route_query()
  │     ├── P3-b: core/version.py — cluster state CRUD
  │     ├── P3-c: pipeline/kv_indexer.py — clustering step
  │     ├── P3-d: pipeline/prs_evaluator.py — Dynamic PRS three-signal + Flywheel record_round hook
  │     ├── P3-e: pipeline/kv_background.py — image KV worker
  │     └── P3-f: pipeline/monitoring_dashboard.py — Flywheel tab + record_query hook
  │
  └── Phase 4: Studio integration + Final test suite
        ├── P4-a: studio/routes.py — ModelScout SSE + Flywheel router + Dynamic PRS UC settings
        └── P4-b: Run full test suite, generate test report
```

---

## Task 0: Unified Config Expansion

**Files:** `core/config.py`  
**Tests:** `tests/test_config.py`

Add all 35 new fields across 5 tracks in one commit. Fields are grouped by track.

```python
# After existing `prs_weights` field — Dynamic PRS (13 fields)
deployment_mode: Literal["greenfield", "brownfield", "auto"] = "auto"
difficulty_estimator: str = "intra_cluster_distance"
cluster_k_range: list[int] = Field(default_factory=lambda: [3, 20])
min_cluster_samples_for_adaptation: int = 10
prs_stability_window: int = 3
prs_advancement_threshold: float = 0.72
prs_auto_weight: bool = True
prs_signal_weights: dict = Field(default_factory=lambda: {"faq": 0.4, "vdb": 0.4, "realtime": 0.2})
brownfield_routing_threshold: float = 0.85
brownfield_confidence_floor: float = 0.80
brownfield_coverage_target: float = 0.70
realtime_requery_window_minutes: int = 10
query_log_db: str = "query_log.db"

# Flywheel Analytics (3 fields)
analytics_db: str = ""
cost_per_1k_tokens: float = 5.0
tokens_per_ms_baseline: float = 0.8

# VDB Expansion — extend existing Literal AND add 8 backend fields
# NOTE: Change vector_store Literal from ["qdrant","chroma","faiss"] to:
# ["qdrant", "chroma", "faiss", "pinecone", "pgvector", "weaviate", "milvus"]
pinecone_api_key: str = ""
pinecone_cloud: str = "aws"
pinecone_region: str = "us-east-1"
pgvector_dsn: str = ""
pgvector_table: str = ""
weaviate_url: str = "http://localhost:8080"
weaviate_api_key: str = ""
milvus_uri: str = "http://localhost:19530"
milvus_token: str = ""

# ModelScout (10 fields)
model_registry_path: str = ""
model_scout_program: str = "model_scout_program.md"
model_scout_results: str = "model_scout_results.json"
scout_initial_corpus_chunks: int = 200
scout_initial_faq_count: int = 30
scout_initial_lora_steps: int = 50
scout_initial_lora_rank: int = 8
scout_max_lora_steps: int = 200
scout_max_corpus_chunks: int = 2000
scout_max_faq_count: int = 200

# Multimodal (5 fields)
image_collection_suffix: str = "_images"
image_store_dir: str = ""
multimodal_model: str = "llava-hf/llava-1.5-7b-hf"
clip_model: str = "openai/clip-vit-base-patch32"
image_kv_inference: bool = False
```

**Test (add to `tests/test_config.py`):**
```python
def test_all_new_fields_have_correct_defaults():
    from core.config import DatasourceConfig
    cfg = DatasourceConfig(
        collection="t", embed_model="m", vector_dim=384,
        llm_model="m", checkpoint_dir="/t", version_file="/t/v.json", replay_db="/t/r.db"
    )
    # Dynamic PRS
    assert cfg.deployment_mode == "auto"
    assert cfg.prs_advancement_threshold == 0.72
    assert cfg.prs_signal_weights == {"faq": 0.4, "vdb": 0.4, "realtime": 0.2}
    assert cfg.query_log_db == "query_log.db"
    # Flywheel
    assert cfg.analytics_db == ""
    assert cfg.cost_per_1k_tokens == 5.0
    # VDB
    assert cfg.vector_store == "qdrant"
    assert cfg.pinecone_api_key == ""
    assert cfg.milvus_uri == "http://localhost:19530"
    # ModelScout
    assert cfg.scout_initial_lora_rank == 8
    # Multimodal
    assert cfg.image_collection_suffix == "_images"
    assert cfg.image_kv_inference is False
```

**Commit message:** `feat: unified config expansion — 35 new fields across 5 tracks`

---

## Phase 1-A: VDB Store Implementations

See full task details in `docs/superpowers/plans/2026-04-22-vdb-expansion.md` Tasks 3–6.

**Files to create:**
- `vectorstore/pinecone_store.py`
- `vectorstore/pgvector_store.py`
- `vectorstore/weaviate_store.py`
- `vectorstore/milvus_store.py`
- Tests in `tests/test_vectorstore.py` (extend existing)

**Commit message:** `feat: add Pinecone, PGVector, Weaviate, Milvus vector store backends`

---

## Phase 1-B: Flywheel Analytics Core

See full task details in `docs/superpowers/plans/2026-04-22-moat-flywheel.md` Tasks 1–6.

**Files to create:**
- `core/analytics.py`
- `studio/flywheel_routes.py`
- `tests/test_analytics.py`
- `tests/test_flywheel_routes.py`

**Commit message:** `feat: add Flywheel Analytics core (analytics.py + flywheel_routes.py)`

---

## Phase 1-C: Dynamic PRS Core Libraries

See full task details in `docs/superpowers/plans/2026-04-22-dynamic-prs.md` Tasks 2–5.

**Files to create:**
- `core/difficulty_estimators.py`
- `core/cluster_manager.py`
- `core/prs_adapter.py`
- `pipeline/query_logger.py`
- `pipeline/chunk_confidence.py`
- `tests/test_difficulty_estimators.py`
- `tests/test_cluster_manager.py`
- `tests/test_prs_adapter.py`
- `tests/test_query_logger.py`
- `tests/test_chunk_confidence.py`

**Commit message:** `feat: Dynamic PRS core libraries (difficulty estimators, cluster manager, PRS adapter)`

---

## Phase 1-D: ModelScout Core

See full task details in `docs/superpowers/plans/2026-04-22-modelscout.md` Tasks 1–5.

**Files to create:**
- `core/model_registry.json`
- `core/model_registry.py`
- `model_scout_program.md`
- `pipeline/model_scout.py`
- `pipeline/model_scout_cli.py`
- `tests/test_model_registry.py`
- `tests/test_model_scout.py`

**Commit message:** `feat: ModelScout core (registry, agent loop, CLI)`

---

## Phase 1-E: Multimodal New Files

See full task details in `docs/superpowers/plans/2026-04-23-multimodal.md` Tasks 2–8 and Task 10.

**Files to create:**
- `core/multimodal_loader.py`
- `embeddings/clip_embedder.py`
- `ingestion/image_extractor.py`
- `pipeline/image_inference.py`
- `pipeline/image_indexer.py`
- `pipeline/multimodal_query.py`
- `tests/test_multimodal.py`

**Commit message:** `feat: Multimodal pipeline (CLIP, LLaVA, image indexing and query)`

---

## Phase 2-A: vectorstore/registry.py

See full task details in `docs/superpowers/plans/2026-04-22-vdb-expansion.md` Tasks 1–2 and 7.

**Adds:** `_custom_registry`, `register_store()`, dispatches for all 7 backends.

**Conflict note:** Must run AFTER Phase 1-A (all store files must exist before imports are added to registry).

**Commit message:** `feat: pluggable vectorstore registry with register_store() and 4 new backend dispatches`

---

## Phase 2-B: embeddings/registry.py

**Adds:** `"clip"` backend dispatch.

**Conflict note:** Must run AFTER Phase 1-E (`embeddings/clip_embedder.py` must exist).

```python
    if backend == "clip":
        from embeddings.clip_embedder import CLIPEmbedder
        return CLIPEmbedder(model_name=cfg.get("clip_model", "openai/clip-vit-base-patch32"))
```

**Commit message:** `feat: add 'clip' backend to embeddings registry`

---

## Phase 3-a: pipeline/kv_inference.py

**Two additive changes (no conflict):**

1. Add `extra_context: str = ""` param to `generate_with_kv` and `generate_text_in_context` (Multimodal plan Task 6).
2. Add `route_query(query, cfg) -> list[dict]` function for Dynamic PRS cluster routing (Dynamic PRS plan Task 6).

**Conflict note:** Both are additive. Apply in sequence: extra_context first, route_query second.

**Commit message:** `feat: kv_inference — extra_context param + Dynamic PRS route_query dispatcher`

---

## Phase 3-b: core/version.py

**Adds** per-cluster state dict support (Dynamic PRS plan Task 3):
- `load_cluster_state(cfg) -> dict`
- `save_cluster_state(cfg, state: dict) -> None`
- `append_cluster_prs(cfg, cluster_id: str, prs: float) -> None`

The `version.json` schema gains a `"clusters"` key:
```json
{
  "current_lora_version": 0,
  "current_phase": 1,
  "prs_history": [],
  "clusters": {}
}
```

**Commit message:** `feat: version.py — per-cluster state CRUD for Dynamic PRS`

---

## Phase 3-c: pipeline/kv_indexer.py

**Adds** clustering step after all upserts in `cmd_index` (Dynamic PRS plan Task 6):
```python
from core.cluster_manager import cluster_embeddings, save_centroids
cluster_assignments = cluster_embeddings(all_vectors, cfg)
for chunk_id, cluster_id in cluster_assignments.items():
    store.set_payload(cfg["collection"], chunk_id, {"cluster_id": cluster_id})
save_centroids(cfg, centroids)
```

**Commit message:** `feat: kv_indexer — assign cluster_id to chunks after indexing`

---

## Phase 3-d: pipeline/prs_evaluator.py

**Two hooks in sequence (Dynamic PRS plan Task 7 + Flywheel plan Task 7):**

```python
# At end of evaluate() — after FAQ accuracy computed:

# Hook 1: Dynamic PRS three-signal per-cluster update
from core.prs_adapter import compute_per_cluster_prs, update_cluster_weights
cluster_state = compute_per_cluster_prs(cfg, faq_results, lora_version)
update_cluster_weights(cfg, cluster_state)
ver.save_cluster_state(cfg, cluster_state)

# Hook 2: Flywheel record_round (reads cluster_state already computed above)
from core.analytics import record_round, init_db
init_db(cfg)
from core.access_tracker import get_tier_distribution
tier_dist = get_tier_distribution(cfg)
record_round(cfg, lora_version, cluster_state, tier_dist)
```

**Commit message:** `feat: prs_evaluator — Dynamic PRS three-signal hook + Flywheel record_round hook`

---

## Phase 3-e: pipeline/kv_background.py

**Adds** image KV recomputation worker (Multimodal plan Task 10):
- `_image_kv_queue: queue.Queue`
- `enqueue_image_kv_recompute(chunk_ids: list[int]) -> None`
- `_image_kv_worker(cfg: dict) -> None` — uses `LLaVALoader`, targets `<collection>_images`
- Update `start(cfg)` to launch `_image_kv_worker` as third daemon thread

**Commit message:** `feat: kv_background — image KV recomputation worker (LLaVA, separate queue)`

---

## Phase 3-f: pipeline/monitoring_dashboard.py

**Adds** (Flywheel plan Tasks 8–10):
1. Call `record_query(cfg, cluster_id, phase_used, latency_ms)` in `run_query()` after inference
2. `GET /api/flywheel` — returns JSON with metric panels and cluster cards
3. `PATCH /api/flywheel/cost-rate` — updates `cost_per_1k_tokens` in config
4. `GET /flywheel` — renders Flywheel HTML tab

**Commit message:** `feat: monitoring_dashboard — Flywheel tab + record_query hook`

---

## Phase 4-a: studio/routes.py

**Three additions (Dynamic PRS plan Task 8 + ModelScout plan Task 6 + Flywheel plan Task 11):**

```python
# ModelScout SSE
from pipeline.model_scout import ScoutEventGenerator
@router.get("/api/modelscout/{uc_name}/stream")
async def modelscout_stream(uc_name: str): ...

@router.post("/api/modelscout/{uc_name}/respond")
async def modelscout_respond(uc_name: str, body: dict): ...

# Flywheel cross-UC summary
from studio.flywheel_routes import router as flywheel_router
app.include_router(flywheel_router)

# Dynamic PRS UC settings panel
@router.get("/api/uc/{uc_name}/settings")
async def get_uc_settings(uc_name: str): ...

@router.patch("/api/uc/{uc_name}/settings")
async def patch_uc_settings(uc_name: str, body: dict): ...
```

**Commit message:** `feat: studio/routes — ModelScout SSE, Flywheel cross-UC panel, Dynamic PRS UC settings`

---

## Phase 4-b: Full Test Suite + Test Report

Run all tests:
```bash
python -m pytest tests/ -v --override-ini="addopts=" --tb=short 2>&1 | tee docs/test-reports/2026-04-23-full-suite.txt
```

Generate test documentation:
- `docs/test-reports/2026-04-23-full-suite.txt` — raw pytest output
- `docs/test-reports/2026-04-23-test-report.md` — summary with pass/fail counts per track

---

## File Map Summary

| Track | New Files | Modified Files |
|---|---|---|
| Dynamic PRS | difficulty_estimators.py, cluster_manager.py, prs_adapter.py, query_logger.py, chunk_confidence.py | config.py, version.py, kv_indexer.py, prs_evaluator.py, kv_inference.py, studio/routes.py |
| Flywheel | analytics.py, flywheel_routes.py | config.py, monitoring_dashboard.py, prs_evaluator.py, studio/routes.py |
| VDB Expansion | pinecone_store.py, pgvector_store.py, weaviate_store.py, milvus_store.py | config.py, vectorstore/registry.py |
| ModelScout | model_registry.json, model_registry.py, model_scout.py, model_scout_cli.py, model_scout_program.md | config.py, studio/routes.py |
| Multimodal | multimodal_loader.py, clip_embedder.py, image_extractor.py, image_inference.py, image_indexer.py, multimodal_query.py | config.py, embeddings/registry.py, kv_inference.py, kv_background.py |
