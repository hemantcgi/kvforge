# Dynamic PRS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single hardcoded PRS threshold with a cluster-aware, three-signal adaptive system that uses the VDB as teacher, supports both greenfield and brownfield deployments, and exposes key thresholds as dashboard-editable per-UC settings.

**Architecture:** At index time, document embeddings are clustered (K-means, auto-K) and each chunk tagged with a `cluster_id`. Per-cluster PRS is computed from three signals — FAQ accuracy, VDB sampling coverage, and real-time query coverage — with weights that adapt via logistic regression after each LoRA round. Phase advancement is per-cluster (not global) once both `prs ≥ threshold` and `slope ≥ 0` hold simultaneously. For brownfield UCs (existing VDB), per-chunk confidence scores drive routing until enough coverage is achieved for a greenfield migration.

**Tech Stack:** Python 3.11+, NumPy (logistic regression + K-means — no sklearn), Pydantic v2, SQLite (query log), FastAPI (dashboard routes), existing PEFT/HuggingFace stack unchanged.

---

## File Map

**New files:**
- `core/difficulty_estimators.py` — DifficultyEstimator Protocol + 4 built-in implementations
- `core/cluster_manager.py` — K-means, silhouette K-selection, centroid persistence, nearest-cluster lookup
- `core/prs_adapter.py` — per-cluster PRS computation, logistic regression weight learning, advancement logic
- `pipeline/query_logger.py` — SQLite-backed real-time query logging (retrieval + parametric)
- `pipeline/chunk_confidence.py` — brownfield per-chunk confidence scoring (background process)
- `tests/test_difficulty_estimators.py`
- `tests/test_cluster_manager.py`
- `tests/test_prs_adapter.py`
- `tests/test_query_logger.py`
- `tests/test_chunk_confidence.py`

**Modified files:**
- `core/config.py` — add 11 new DatasourceConfig fields
- `core/version.py` — add cluster state CRUD, update DEFAULTS, update `append_prs` to delegate to per-cluster logic
- `pipeline/kv_indexer.py` — add clustering step in `build_payload`; run `cluster_embeddings` after all upserts
- `pipeline/prs_evaluator.py` — add `sample_vdb_coverage()`, integrate three-signal per-cluster update
- `pipeline/kv_inference.py` — add `route_query()` dispatcher (greenfield centroid routing + brownfield confidence routing)
- `studio/routes.py` — add UC settings panel with dashboard-editable fields + migration eligibility display
- `tests/test_config.py` — extend for new fields
- `tests/test_prs_evaluator.py` — extend for three-signal evaluation
- `tests/test_kv_inference.py` — extend for new routing modes

---

## Task 1: Config Schema — Add Dynamic PRS Fields

**Files:**
- Modify: `core/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py — add to existing file

def test_dynamic_prs_defaults():
    cfg = DatasourceConfig(
        collection="test", embed_model="BAAI/bge-small-en-v1.5",
        vector_dim=384, llm_model="meta-llama/Llama-3.2-3B-Instruct",
        checkpoint_dir="/tmp/ckpt", version_file="/tmp/v.json",
        replay_db="/tmp/r.db",
    )
    assert cfg.deployment_mode == "auto"
    assert cfg.difficulty_estimator == "intra_cluster_distance"
    assert cfg.cluster_k_range == [3, 20]
    assert cfg.prs_stability_window == 3
    assert cfg.prs_advancement_threshold == 0.72
    assert cfg.prs_auto_weight is True
    assert cfg.prs_signal_weights == {"faq": 0.4, "vdb": 0.4, "realtime": 0.2}
    assert cfg.brownfield_routing_threshold == 0.85
    assert cfg.brownfield_confidence_floor == 0.80
    assert cfg.brownfield_coverage_target == 0.70
    assert cfg.realtime_requery_window_minutes == 10
    assert cfg.query_log_db == "query_log.db"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_config.py::test_dynamic_prs_defaults -v --override-ini="addopts="
```

Expected: `FAILED — AttributeError: 'DatasourceConfig' object has no attribute 'deployment_mode'`

- [ ] **Step 3: Add fields to DatasourceConfig**

In `core/config.py`, add the following block after the existing `# Phase gating` block (after `prs_weights`):

```python
    # Dynamic PRS — deployment mode
    deployment_mode: Literal["greenfield", "brownfield", "auto"] = "auto"
    difficulty_estimator: str = "intra_cluster_distance"
    cluster_k_range: list[int] = Field(default_factory=lambda: [3, 20])
    min_cluster_samples_for_adaptation: int = 10
    prs_stability_window: int = 3
    prs_advancement_threshold: float = 0.72
    prs_auto_weight: bool = True
    # Three-signal PRS weights (faq/vdb/realtime — distinct from prs_weights which is internal FAQ sub-scores)
    prs_signal_weights: dict = Field(
        default_factory=lambda: {"faq": 0.4, "vdb": 0.4, "realtime": 0.2}
    )

    # Brownfield settings (all dashboard-editable)
    brownfield_routing_threshold: float = 0.85
    brownfield_confidence_floor: float = 0.80
    brownfield_coverage_target: float = 0.70
    realtime_requery_window_minutes: int = 10

    # Query log (SQLite, real-time training signal)
    query_log_db: str = "query_log.db"
```

Also update the docstring `Attributes:` section to describe each new field (add after the existing `prs_weights` description):

```
        deployment_mode: Deployment mode for dynamic PRS. ``'greenfield'`` uses
            cluster-aware routing from scratch; ``'brownfield'`` uses per-chunk
            confidence scoring for gradual migration; ``'auto'`` detects based
            on whether existing chunks have ``cluster_id``.
        difficulty_estimator: Name of the pluggable difficulty estimator used to
            set initial per-cluster PRS thresholds. Built-ins:
            ``'intra_cluster_distance'``, ``'vocab_complexity'``,
            ``'entity_density'``, ``'length_variance'``.
        cluster_k_range: ``[min_k, max_k]`` range for silhouette-based K selection.
        min_cluster_samples_for_adaptation: Minimum labeled query outcomes in a
            cluster before logistic regression weight adaptation runs.
        prs_stability_window: Number of LoRA rounds used to compute PRS slope
            for advancement stability check.
        prs_advancement_threshold: Initial per-UC PRS floor for phase advancement.
            Adapted per-cluster by difficulty estimator and empirical calibration.
        prs_auto_weight: If ``True``, per-cluster logistic regression adapts
            ``prs_signal_weights`` automatically from labeled query outcomes.
        prs_signal_weights: Weights for the three top-level PRS signals:
            ``'faq'``, ``'vdb'``, ``'realtime'``.
        brownfield_routing_threshold: Minimum ``model_confidence`` score required
            for all retrieved chunks to route a query parametrically (brownfield).
        brownfield_confidence_floor: Chunk confidence floor for migration eligibility.
        brownfield_coverage_target: Fraction of VDB chunks above confidence floor
            required before greenfield migration is offered.
        realtime_requery_window_minutes: Window in which a follow-up identical
            query is counted as a re-query (dissatisfaction signal).
        query_log_db: Path to the SQLite database for real-time query logging.
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_config.py::test_dynamic_prs_defaults -v --override-ini="addopts="
```

Expected: `PASSED`

- [ ] **Step 5: Run full config tests**

```bash
python -m pytest tests/test_config.py -v --override-ini="addopts="
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add core/config.py tests/test_config.py
git commit -m "feat: add dynamic PRS config fields to DatasourceConfig"
```

---

## Task 2: DifficultyEstimator Protocol and Built-in Implementations

**Files:**
- Create: `core/difficulty_estimators.py`
- Create: `tests/test_difficulty_estimators.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_difficulty_estimators.py

import numpy as np
import pytest
from core.difficulty_estimators import (
    DifficultyEstimator, IntraClusterDistance, VocabComplexity,
    EntityDensity, LengthVariance, get_estimator, register_estimator,
)


def test_protocol_structural():
    class Custom:
        def score(self, chunks, embeddings=None):
            return 0.5
    assert isinstance(Custom(), DifficultyEstimator)


def test_intra_cluster_distance_single_chunk():
    est = IntraClusterDistance()
    emb = np.array([[1.0, 0.0]])
    assert est.score(["hello"], emb) == 0.5


def test_intra_cluster_distance_orthogonal():
    est = IntraClusterDistance()
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    score = est.score(["a", "b"], emb)
    assert 0.9 < score <= 1.0  # orthogonal → distance ~1.0


def test_intra_cluster_distance_identical():
    est = IntraClusterDistance()
    emb = np.array([[1.0, 0.0], [1.0, 0.0]])
    score = est.score(["a", "b"], emb)
    assert score < 0.1  # identical → distance ~0.0


def test_vocab_complexity_high():
    est = VocabComplexity()
    chunks = ["phosphorylation methylation ubiquitination proteomics transcriptomics"]
    score = est.score(chunks)
    assert score > 0.3


def test_vocab_complexity_low():
    est = VocabComplexity()
    chunks = ["the cat sat on the mat and the dog ran fast"]
    score = est.score(chunks)
    assert score < 0.3


def test_entity_density_returns_float_in_range():
    est = EntityDensity()
    chunks = ["Apple Inc reported that Tim Cook met with Google CEO Sundar Pichai."]
    score = est.score(chunks)
    assert 0.0 <= score <= 1.0


def test_length_variance_uniform():
    est = LengthVariance()
    chunks = ["one two three", "four five six", "seven eight nine"]
    score = est.score(chunks)
    assert score < 0.1


def test_length_variance_mixed():
    est = LengthVariance()
    chunks = ["one", "one two three four five six seven eight nine ten eleven twelve"]
    score = est.score(chunks)
    assert score > 0.5


def test_get_estimator_known():
    est = get_estimator("intra_cluster_distance")
    assert isinstance(est, DifficultyEstimator)


def test_get_estimator_unknown():
    with pytest.raises(ValueError, match="Unknown difficulty estimator"):
        get_estimator("nonexistent_estimator")


def test_register_custom_estimator():
    class MyEst:
        def score(self, chunks, embeddings=None):
            return 0.42
    register_estimator("my_est", MyEst())
    est = get_estimator("my_est")
    assert est.score([]) == 0.42


def test_register_invalid_estimator():
    class BadEst:
        pass  # missing score()
    with pytest.raises(TypeError):
        register_estimator("bad", BadEst())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_difficulty_estimators.py -v --override-ini="addopts="
```

Expected: `ERROR — ModuleNotFoundError: No module named 'core.difficulty_estimators'`

- [ ] **Step 3: Create the module**

```python
# core/difficulty_estimators.py

from __future__ import annotations
import re
from collections import Counter
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DifficultyEstimator(Protocol):
    def score(self, chunks: list[str], embeddings: np.ndarray | None = None) -> float:
        ...


class IntraClusterDistance:
    def score(self, chunks: list[str], embeddings: np.ndarray | None = None) -> float:
        if embeddings is None or len(embeddings) < 2:
            return 0.5
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normed = embeddings / norms
        sims = normed @ normed.T
        n = len(embeddings)
        distances = [1.0 - float(sims[i, j]) for i in range(n) for j in range(i + 1, n)]
        return float(np.mean(distances)) if distances else 0.5


class VocabComplexity:
    def __init__(self, common_vocab_size: int = 10_000):
        self._common_vocab_size = common_vocab_size

    def score(self, chunks: list[str], embeddings: np.ndarray | None = None) -> float:
        tokens = [t for chunk in chunks for t in re.findall(r"\b\w+\b", chunk.lower())]
        if not tokens:
            return 0.5
        top_words = {w for w, _ in Counter(tokens).most_common(self._common_vocab_size)}
        rare = sum(1 for t in tokens if t not in top_words)
        return rare / len(tokens)


class EntityDensity:
    def score(self, chunks: list[str], embeddings: np.ndarray | None = None) -> float:
        total = entity = 0
        for chunk in chunks:
            for sent in re.split(r"[.!?]", chunk):
                words = sent.split()
                entity += sum(1 for i, w in enumerate(words) if i > 0 and w and w[0].isupper())
                total += len(words)
        if total == 0:
            return 0.5
        return min(entity / total * 10, 1.0)


class LengthVariance:
    def score(self, chunks: list[str], embeddings: np.ndarray | None = None) -> float:
        lengths = [len(c.split()) for c in chunks]
        if len(lengths) < 2:
            return 0.5
        mean = np.mean(lengths)
        if mean == 0:
            return 0.0
        return min(float(np.std(lengths) / mean), 1.0)


REGISTRY: dict[str, DifficultyEstimator] = {
    "intra_cluster_distance": IntraClusterDistance(),
    "vocab_complexity": VocabComplexity(),
    "entity_density": EntityDensity(),
    "length_variance": LengthVariance(),
}


def get_estimator(name: str) -> DifficultyEstimator:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown difficulty estimator '{name}'. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


def register_estimator(name: str, estimator: DifficultyEstimator) -> None:
    if not isinstance(estimator, DifficultyEstimator):
        raise TypeError(f"{estimator!r} does not implement DifficultyEstimator protocol")
    REGISTRY[name] = estimator
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_difficulty_estimators.py -v --override-ini="addopts="
```

Expected: all 12 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add core/difficulty_estimators.py tests/test_difficulty_estimators.py
git commit -m "feat: add DifficultyEstimator protocol and four built-in implementations"
```

---

## Task 3: Version Schema — Per-Cluster State

**Files:**
- Modify: `core/version.py`
- Modify: `tests/test_kvforge.py` (add new version tests)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kvforge.py — add these test functions

def test_version_cluster_state_roundtrip(tmp_path):
    import core.version as ver
    ver.VERSION_FILE = tmp_path / "version.json"
    state = {
        "label": "auth", "phase": 1, "faq_coverage": 0.8,
        "vdb_coverage": 0.7, "realtime_coverage": 0.6,
        "prs": 0.72, "prs_history": [0.72], "threshold": 0.72,
        "learned_weights": None, "query_count": 5,
    }
    ver.save_cluster_state("0", state)
    loaded = ver.get_cluster_state("0")
    assert loaded["label"] == "auth"
    assert loaded["prs"] == 0.72


def test_version_global_phase_min_across_clusters(tmp_path):
    import core.version as ver
    ver.VERSION_FILE = tmp_path / "version.json"
    ver.save_cluster_state("0", {"phase": 3})
    ver.save_cluster_state("1", {"phase": 1})
    assert ver.get_global_phase() == 1


def test_version_global_phase_no_clusters(tmp_path):
    import core.version as ver
    ver.VERSION_FILE = tmp_path / "version.json"
    assert ver.get_global_phase() == 1  # default when no clusters


def test_version_defaults_include_clusters(tmp_path):
    import core.version as ver
    ver.VERSION_FILE = tmp_path / "version.json"
    data = ver.load()
    assert "clusters" in data
    assert data["clusters"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_kvforge.py::test_version_cluster_state_roundtrip tests/test_kvforge.py::test_version_global_phase_min_across_clusters tests/test_kvforge.py::test_version_global_phase_no_clusters tests/test_kvforge.py::test_version_defaults_include_clusters -v --override-ini="addopts="
```

Expected: `FAILED — AttributeError: module 'core.version' has no attribute 'save_cluster_state'`

- [ ] **Step 3: Extend core/version.py**

Add `"clusters": {}` to `DEFAULTS`:

```python
DEFAULTS: dict[str, Any] = {
    "current_lora_version": 0,
    "checkpoint_path": None,
    "phase": 1,
    "prs_history": [],
    "known_good_queries": [],
    "clusters": {},
}
```

Add three new public functions at the end of `core/version.py`:

```python
def get_cluster_state(cluster_id: str) -> dict:
    """Return per-cluster PRS state dict, or empty dict if cluster not yet tracked."""
    return load().get("clusters", {}).get(str(cluster_id), {})


def save_cluster_state(cluster_id: str, state: dict) -> None:
    """Atomically update a single cluster's state in version.json."""
    data = load()
    data.setdefault("clusters", {})[str(cluster_id)] = state
    save(data)


def get_global_phase() -> int:
    """Return minimum phase across all clusters (conservative).
    Falls back to the top-level 'phase' when no clusters exist.
    """
    data = load()
    clusters = data.get("clusters", {})
    if not clusters:
        return data.get("phase", 1)
    return min(c.get("phase", 1) for c in clusters.values())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_kvforge.py::test_version_cluster_state_roundtrip tests/test_kvforge.py::test_version_global_phase_min_across_clusters tests/test_kvforge.py::test_version_global_phase_no_clusters tests/test_kvforge.py::test_version_defaults_include_clusters -v --override-ini="addopts="
```

Expected: all 4 `PASSED`

- [ ] **Step 5: Run full version-related tests**

```bash
python -m pytest tests/test_kvforge.py -v --override-ini="addopts="
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add core/version.py tests/test_kvforge.py
git commit -m "feat: add per-cluster state CRUD and get_global_phase to version.py"
```

---

## Task 4: QueryLogger — Real-Time Query Logging

**Files:**
- Create: `pipeline/query_logger.py`
- Create: `tests/test_query_logger.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_query_logger.py

import time
import pytest
from pipeline.query_logger import init_db, log_query, mark_requeried, get_cluster_stats, get_training_pairs


def test_init_creates_table(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    import sqlite3
    with sqlite3.connect(db) as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ("query_log",) in tables


def test_log_and_retrieve_retrieval_query(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    row_id = log_query(db, "What is RAG?", "RAG stands for...", "retrieval", cluster_id="0")
    assert row_id == 1
    pairs = get_training_pairs(db)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "What is RAG?"
    assert pairs[0]["cluster_id"] == "0"


def test_parametric_queries_not_in_training_pairs(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    log_query(db, "Who is CEO?", "Tim Cook", "parametric", cluster_id="1")
    pairs = get_training_pairs(db)
    assert len(pairs) == 0


def test_get_cluster_stats_no_requery(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    log_query(db, "Q1", "A1", "parametric", cluster_id="0")
    log_query(db, "Q2", "A2", "parametric", cluster_id="0")
    stats = get_cluster_stats(db, "0", window_minutes=10)
    assert stats["realtime_coverage"] == 1.0
    assert stats["query_count"] == 2


def test_mark_requeried_lowers_coverage(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    log_query(db, "Q1", "A1", "parametric", cluster_id="0")
    mark_requeried(db, "Q1", window_minutes=10)
    stats = get_cluster_stats(db, "0", window_minutes=10)
    assert stats["realtime_coverage"] == 0.0


def test_get_cluster_stats_empty(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    stats = get_cluster_stats(db, "99", window_minutes=10)
    assert stats["realtime_coverage"] == 0.0
    assert stats["query_count"] == 0


def test_get_training_pairs_filtered_by_cluster(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    log_query(db, "Q1", "A1", "retrieval", cluster_id="0")
    log_query(db, "Q2", "A2", "retrieval", cluster_id="1")
    pairs = get_training_pairs(db, cluster_id="0")
    assert len(pairs) == 1
    assert pairs[0]["question"] == "Q1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_query_logger.py -v --override-ini="addopts="
```

Expected: `ERROR — ModuleNotFoundError: No module named 'pipeline.query_logger'`

- [ ] **Step 3: Create the module**

```python
# pipeline/query_logger.py

from __future__ import annotations
import json
import sqlite3
import threading
import time
from typing import Optional

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    query_text TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    cluster_id TEXT,
    chunk_id TEXT,
    routed_to TEXT NOT NULL,
    requeried INTEGER DEFAULT 0,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_cluster ON query_log(cluster_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON query_log(timestamp);
"""


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


def log_query(
    db_path: str,
    query_text: str,
    answer_text: str,
    routed_to: str,
    cluster_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    embedding: Optional[list[float]] = None,
) -> int:
    emb_blob = json.dumps(embedding).encode() if embedding else None
    with _lock, sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO query_log
               (timestamp, query_text, answer_text, cluster_id, chunk_id, routed_to, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), query_text, answer_text, cluster_id, chunk_id, routed_to, emb_blob),
        )
        conn.commit()
        return cur.lastrowid


def mark_requeried(db_path: str, original_query: str, window_minutes: int = 10) -> None:
    cutoff = time.time() - window_minutes * 60
    with _lock, sqlite3.connect(db_path) as conn:
        conn.execute(
            """UPDATE query_log SET requeried = 1
               WHERE routed_to = 'parametric' AND timestamp > ? AND query_text = ?""",
            (cutoff, original_query),
        )
        conn.commit()


def get_cluster_stats(db_path: str, cluster_id: str, window_minutes: int = 10) -> dict:
    cutoff = time.time() - window_minutes * 60
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN routed_to='parametric' AND requeried=0 THEN 1 ELSE 0 END)
               FROM query_log WHERE cluster_id = ? AND timestamp > ?""",
            (cluster_id, cutoff),
        ).fetchone()
    total, good = row[0], row[1] or 0
    if not total:
        return {"realtime_coverage": 0.0, "query_count": 0}
    return {"realtime_coverage": good / total, "query_count": total}


def get_training_pairs(
    db_path: str, cluster_id: Optional[str] = None, limit: int = 1000
) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        if cluster_id is not None:
            rows = conn.execute(
                """SELECT query_text, answer_text, cluster_id FROM query_log
                   WHERE routed_to = 'retrieval' AND cluster_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (cluster_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT query_text, answer_text, cluster_id FROM query_log
                   WHERE routed_to = 'retrieval' ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()
    return [{"question": r[0], "answer": r[1], "cluster_id": r[2]} for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_query_logger.py -v --override-ini="addopts="
```

Expected: all 7 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/query_logger.py tests/test_query_logger.py
git commit -m "feat: add SQLite-backed QueryLogger for real-time query training signal"
```

---

## Task 5: ClusterManager — K-Means, Silhouette Selection, Centroid Routing

**Files:**
- Create: `core/cluster_manager.py`
- Create: `tests/test_cluster_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cluster_manager.py

import numpy as np
import pytest
from core.cluster_manager import (
    select_k, cluster_embeddings, save_clusters, load_clusters, nearest_cluster,
)


def _make_blobs(n_per_cluster=20, n_clusters=3, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 5
    parts = [rng.standard_normal((n_per_cluster, dim)) + centers[i] for i in range(n_clusters)]
    return np.vstack(parts)


def test_select_k_finds_true_k():
    emb = _make_blobs(n_per_cluster=20, n_clusters=3)
    k = select_k(emb, k_range=(2, 6))
    assert k == 3


def test_cluster_embeddings_returns_correct_shapes():
    emb = _make_blobs(n_per_cluster=15, n_clusters=4)
    centroids, labels = cluster_embeddings(emb, k_range=(2, 6))
    assert centroids.shape[1] == emb.shape[1]
    assert labels.shape == (len(emb),)
    assert set(labels) == set(range(len(centroids)))


def test_cluster_embeddings_label_count_matches_centroids():
    emb = _make_blobs(n_per_cluster=10, n_clusters=3)
    centroids, labels = cluster_embeddings(emb, k_range=(2, 5))
    assert len(centroids) == len(set(labels))


def test_save_and_load_clusters(tmp_path):
    emb = _make_blobs()
    centroids, labels = cluster_embeddings(emb, k_range=(2, 5))
    path = str(tmp_path / "clusters.json")
    save_clusters(path, centroids, labels, lora_version=1)
    data = load_clusters(path)
    assert data["k"] == len(centroids)
    assert np.allclose(data["centroids"], centroids)
    assert data["lora_version"] == 1
    assert len(data["cluster_labels"]) == len(centroids)


def test_nearest_cluster_returns_correct_index():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query = np.array([0.9, 0.1])
    idx = nearest_cluster(query, centroids)
    assert idx == 0


def test_nearest_cluster_second():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query = np.array([0.1, 0.9])
    idx = nearest_cluster(query, centroids)
    assert idx == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cluster_manager.py -v --override-ini="addopts="
```

Expected: `ERROR — ModuleNotFoundError: No module named 'core.cluster_manager'`

- [ ] **Step 3: Create the module**

```python
# core/cluster_manager.py

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _kmeans(
    embeddings: np.ndarray, k: int, max_iter: int = 100, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(embeddings), k, replace=False)
    centroids = embeddings[idx].copy().astype(float)
    labels = np.zeros(len(embeddings), dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(embeddings[:, None] - centroids[None], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = embeddings[mask].mean(axis=0)
    return centroids, labels


def _silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    n = len(embeddings)
    if n < 4:
        return 0.0
    scores = []
    unique = np.unique(labels)
    for i in range(n):
        same_mask = labels == labels[i]
        same_mask[i] = False
        other_clusters = [c for c in unique if c != labels[i]]
        if not other_clusters or same_mask.sum() == 0:
            continue
        a = np.linalg.norm(embeddings[i] - embeddings[same_mask], axis=1).mean()
        b = min(
            np.linalg.norm(embeddings[i] - embeddings[labels == c], axis=1).mean()
            for c in other_clusters
        )
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def select_k(embeddings: np.ndarray, k_range: tuple[int, int] = (3, 20)) -> int:
    best_k, best_score = k_range[0], -1.0
    for k in range(k_range[0], min(k_range[1] + 1, len(embeddings))):
        _, labels = _kmeans(embeddings, k)
        score = _silhouette(embeddings, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def cluster_embeddings(
    embeddings: np.ndarray, k_range: tuple[int, int] = (3, 20)
) -> tuple[np.ndarray, np.ndarray]:
    k = select_k(embeddings, k_range)
    return _kmeans(embeddings, k)


def save_clusters(
    path: str,
    centroids: np.ndarray,
    labels: np.ndarray,
    cluster_labels: Optional[list[str]] = None,
    lora_version: int = 0,
) -> None:
    data = {
        "k": len(centroids),
        "centroids": centroids.tolist(),
        "labels": labels.tolist(),
        "cluster_labels": cluster_labels or [f"cluster_{i}" for i in range(len(centroids))],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lora_version": lora_version,
    }
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_clusters(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    data["centroids"] = np.array(data["centroids"])
    return data


def nearest_cluster(query_embedding: np.ndarray, centroids: np.ndarray) -> int:
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    normed = centroids / norms
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    return int((normed @ q_norm).argmax())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cluster_manager.py -v --override-ini="addopts="
```

Expected: all 6 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add core/cluster_manager.py tests/test_cluster_manager.py
git commit -m "feat: add ClusterManager with numpy K-means, silhouette K-selection, centroid routing"
```

---

## Task 6: PRS Adapter — Per-Cluster Signal Tracking and Weight Learning

**Files:**
- Create: `core/prs_adapter.py`
- Create: `tests/test_prs_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prs_adapter.py

import numpy as np
import pytest
from core.prs_adapter import (
    compute_cluster_prs, compute_slope, should_advance,
    adapt_weights, initial_threshold,
)


def test_compute_cluster_prs_weighted():
    weights = {"faq": 0.4, "vdb": 0.4, "realtime": 0.2}
    prs = compute_cluster_prs(0.8, 0.6, 0.5, weights)
    assert abs(prs - (0.4 * 0.8 + 0.4 * 0.6 + 0.2 * 0.5)) < 1e-6


def test_compute_cluster_prs_clipped():
    weights = {"faq": 0.5, "vdb": 0.5, "realtime": 0.5}
    prs = compute_cluster_prs(1.0, 1.0, 1.0, weights)
    assert prs == 1.0


def test_compute_slope_increasing():
    slope = compute_slope([0.5, 0.6, 0.7, 0.8])
    assert slope > 0


def test_compute_slope_decreasing():
    slope = compute_slope([0.8, 0.7, 0.6])
    assert slope < 0


def test_compute_slope_flat():
    slope = compute_slope([0.7, 0.7, 0.7])
    assert abs(slope) < 1e-6


def test_compute_slope_single():
    assert compute_slope([0.7]) == 0.0


def test_should_advance_true_when_prs_above_threshold_and_slope_positive():
    state = {"prs": 0.75, "threshold": 0.72, "prs_history": [0.65, 0.70, 0.75]}
    assert should_advance(state, cfg_threshold=0.72, stability_window=3) is True


def test_should_advance_false_when_prs_below_threshold():
    state = {"prs": 0.60, "threshold": 0.72, "prs_history": [0.60, 0.60, 0.60]}
    assert should_advance(state, cfg_threshold=0.72, stability_window=3) is False


def test_should_advance_false_when_slope_negative():
    state = {"prs": 0.75, "threshold": 0.72, "prs_history": [0.80, 0.77, 0.75]}
    assert should_advance(state, cfg_threshold=0.72, stability_window=3) is False


def test_adapt_weights_returns_none_when_too_few_samples():
    state = {"labeled_history": [{"faq": 0.8, "vdb": 0.7, "realtime": 0.6, "correct": 1}]}
    result = adapt_weights(state, min_samples=10)
    assert result is None


def test_adapt_weights_returns_normalized_weights():
    history = [
        {"faq": 0.9, "vdb": 0.2, "realtime": 0.5, "correct": 1},
        {"faq": 0.8, "vdb": 0.3, "realtime": 0.4, "correct": 1},
        {"faq": 0.2, "vdb": 0.8, "realtime": 0.1, "correct": 0},
        {"faq": 0.1, "vdb": 0.9, "realtime": 0.2, "correct": 0},
        {"faq": 0.85, "vdb": 0.25, "realtime": 0.6, "correct": 1},
        {"faq": 0.15, "vdb": 0.75, "realtime": 0.3, "correct": 0},
        {"faq": 0.9, "vdb": 0.1, "realtime": 0.7, "correct": 1},
        {"faq": 0.1, "vdb": 0.85, "realtime": 0.2, "correct": 0},
        {"faq": 0.88, "vdb": 0.2, "realtime": 0.5, "correct": 1},
        {"faq": 0.12, "vdb": 0.82, "realtime": 0.1, "correct": 0},
    ]
    state = {"labeled_history": history}
    weights = adapt_weights(state, min_samples=10)
    assert weights is not None
    assert abs(sum(weights.values()) - 1.0) < 1e-4
    for v in weights.values():
        assert 0.10 <= v <= 0.70


def test_initial_threshold_scales_with_difficulty():
    t_easy = initial_threshold(0.72, difficulty_score=0.3, global_mean_difficulty=0.5)
    t_hard = initial_threshold(0.72, difficulty_score=0.7, global_mean_difficulty=0.5)
    assert t_hard > t_easy


def test_initial_threshold_clamped():
    t = initial_threshold(0.72, difficulty_score=100.0, global_mean_difficulty=0.5)
    assert t <= 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_prs_adapter.py -v --override-ini="addopts="
```

Expected: `ERROR — ModuleNotFoundError: No module named 'core.prs_adapter'`

- [ ] **Step 3: Create the module**

```python
# core/prs_adapter.py

from __future__ import annotations
from typing import Optional

import numpy as np

import core.version as ver


def _logistic_regression_weights(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 300
) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        logits = np.clip(X @ w, -20, 20)
        preds = 1 / (1 + np.exp(-logits))
        w -= lr * (X.T @ (preds - y)) / len(y)
    w = np.clip(np.abs(w), 0.10, 0.70)
    total = w.sum()
    return w / total if total > 0 else np.full_like(w, 1 / len(w))


def compute_cluster_prs(
    faq_coverage: float, vdb_coverage: float, realtime_coverage: float, weights: dict
) -> float:
    return float(np.clip(
        weights.get("faq", 0.4) * faq_coverage
        + weights.get("vdb", 0.4) * vdb_coverage
        + weights.get("realtime", 0.2) * realtime_coverage,
        0.0, 1.0,
    ))


def compute_slope(prs_history: list[float]) -> float:
    n = len(prs_history)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(prs_history, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def should_advance(
    cluster_state: dict, cfg_threshold: float, stability_window: int = 3
) -> bool:
    prs = cluster_state.get("prs", 0.0)
    threshold = cluster_state.get("threshold", cfg_threshold)
    if prs < threshold:
        return False
    history = cluster_state.get("prs_history", [])
    window = history[-stability_window:] if len(history) >= stability_window else history
    return compute_slope(window) >= 0.0


def adapt_weights(cluster_state: dict, min_samples: int = 10) -> Optional[dict]:
    history = cluster_state.get("labeled_history", [])
    if len(history) < min_samples:
        return None
    X = np.array([[h["faq"], h["vdb"], h["realtime"]] for h in history])
    y = np.array([float(h["correct"]) for h in history])
    if y.std() == 0:
        return None
    w = _logistic_regression_weights(X, y)
    return {"faq": float(w[0]), "vdb": float(w[1]), "realtime": float(w[2])}


def initial_threshold(
    base_threshold: float, difficulty_score: float, global_mean_difficulty: float
) -> float:
    if global_mean_difficulty == 0:
        return base_threshold
    ratio = difficulty_score / global_mean_difficulty
    multiplier = float(np.clip(ratio, 0.85, 1.15))
    return float(np.clip(base_threshold * multiplier, 0.5, 0.95))


def update_cluster_after_round(
    cluster_id: str,
    faq_coverage: float,
    vdb_coverage: float,
    realtime_stats: dict,
    cfg: dict,
) -> dict:
    state = ver.get_cluster_state(cluster_id)
    realtime_coverage = realtime_stats.get("realtime_coverage", 0.0)
    weights = state.get("learned_weights") or cfg.get(
        "prs_signal_weights", {"faq": 0.4, "vdb": 0.4, "realtime": 0.2}
    )
    prs = compute_cluster_prs(faq_coverage, vdb_coverage, realtime_coverage, weights)

    history = state.get("prs_history", [])
    history.append(round(prs, 4))
    state.update({
        "faq_coverage": round(faq_coverage, 4),
        "vdb_coverage": round(vdb_coverage, 4),
        "realtime_coverage": round(realtime_coverage, 4),
        "prs": round(prs, 4),
        "prs_history": history,
        "query_count": realtime_stats.get("query_count", state.get("query_count", 0)),
    })

    if cfg.get("prs_auto_weight", True):
        new_weights = adapt_weights(state, cfg.get("min_cluster_samples_for_adaptation", 10))
        if new_weights:
            state["learned_weights"] = new_weights
            prs = compute_cluster_prs(faq_coverage, vdb_coverage, realtime_coverage, new_weights)
            state["prs"] = round(prs, 4)

    threshold = state.get("threshold", cfg.get("prs_advancement_threshold", 0.72))
    window = cfg.get("prs_stability_window", 3)
    current_phase = state.get("phase", 1)
    if current_phase < 3 and should_advance(state, threshold, window):
        state["phase"] = current_phase + 1
        print(f"✅ Cluster {cluster_id} advanced to Phase {state['phase']}")

    ver.save_cluster_state(cluster_id, state)
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_prs_adapter.py -v --override-ini="addopts="
```

Expected: all 13 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add core/prs_adapter.py tests/test_prs_adapter.py
git commit -m "feat: add PRS adapter with three-signal computation and logistic regression weight learning"
```

---

## Task 7: kv_indexer — Clustering Step at Index Time

**Files:**
- Modify: `pipeline/kv_indexer.py`
- Modify: `tests/test_kv_indexer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kv_indexer.py — add this test

def test_build_payload_includes_cluster_id():
    import numpy as np
    from pipeline.kv_indexer import build_payload
    kv = np.zeros((28, 2, 8, 64), dtype=np.float16)
    payload = build_payload("test text", page=1, source_file="doc.pdf",
                            kv_array=kv, cluster_id="3")
    assert payload["cluster_id"] == "3"


def test_build_payload_cluster_id_defaults_none():
    import numpy as np
    from pipeline.kv_indexer import build_payload
    kv = np.zeros((28, 2, 8, 64), dtype=np.float16)
    payload = build_payload("test text", page=1, source_file="doc.pdf", kv_array=kv)
    assert payload["cluster_id"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_kv_indexer.py::test_build_payload_includes_cluster_id tests/test_kv_indexer.py::test_build_payload_cluster_id_defaults_none -v --override-ini="addopts="
```

Expected: `FAILED — TypeError: build_payload() got an unexpected keyword argument 'cluster_id'`

- [ ] **Step 3: Update build_payload in pipeline/kv_indexer.py**

Change the `build_payload` signature and return dict:

```python
def build_payload(
    text: str,
    page: int,
    source_file: str,
    kv_array: np.ndarray,
    indexed_at: int | None = None,
    cluster_id: str | None = None,
) -> dict:
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
        "cluster_id": cluster_id,
        "model_confidence": None,
        "confidence_lora_version": None,
    }
```

- [ ] **Step 4: Add post-index clustering call**

At the bottom of the `main()` function in `pipeline/kv_indexer.py`, after the upsert loop completes, add:

```python
# After all points are upserted — run clustering over collected embeddings
if args.command == "index" and len(all_embeddings) >= 6:
    from core.cluster_manager import cluster_embeddings, save_clusters
    from core.difficulty_estimators import get_estimator
    import json as _json
    cfg_dict = _json.load(open(args.config))
    k_range = tuple(cfg_dict.get("cluster_k_range", [3, 20]))
    emb_matrix = np.vstack(all_embeddings)
    centroids, labels = cluster_embeddings(emb_matrix, k_range=k_range)
    clusters_path = str(Path(cfg_dict["checkpoint_dir"]) / "clusters.json")
    save_clusters(clusters_path, centroids, labels)

    # Compute per-cluster initial difficulty thresholds
    from core.prs_adapter import initial_threshold
    estimator = get_estimator(cfg_dict.get("difficulty_estimator", "intra_cluster_distance"))
    base_threshold = cfg_dict.get("prs_advancement_threshold", 0.72)
    cluster_texts = {i: [] for i in range(len(centroids))}
    cluster_embs = {i: [] for i in range(len(centroids))}
    for idx, (text, emb) in enumerate(zip(all_texts, all_embeddings)):
        cid = int(labels[idx])
        cluster_texts[cid].append(text)
        cluster_embs[cid].append(emb)

    difficulty_scores = {
        cid: estimator.score(cluster_texts[cid], np.array(cluster_embs[cid]))
        for cid in cluster_texts
    }
    global_mean = np.mean(list(difficulty_scores.values())) if difficulty_scores else 0.5

    import core.version as _ver
    _ver.init(cfg_dict)
    for cid, diff in difficulty_scores.items():
        thresh = initial_threshold(base_threshold, diff, global_mean)
        state = _ver.get_cluster_state(str(cid))
        state.setdefault("phase", 1)
        state["threshold"] = round(thresh, 4)
        state["label"] = f"cluster_{cid}"
        _ver.save_cluster_state(str(cid), state)

    print(f"✅ Clustered {len(emb_matrix)} chunks into {len(centroids)} topics → {clusters_path}")
```

This requires collecting `all_embeddings`, `all_texts` during the upsert loop. Add these two lists before the loop:

```python
all_embeddings: list[np.ndarray] = []
all_texts: list[str] = []
```

And inside the upsert loop, after computing each embedding, append:

```python
all_embeddings.append(embedding)
all_texts.append(chunk_text)
```

- [ ] **Step 5: Run the new tests and existing kv_indexer tests**

```bash
python -m pytest tests/test_kv_indexer.py -v --override-ini="addopts="
```

Expected: all tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add pipeline/kv_indexer.py tests/test_kv_indexer.py
git commit -m "feat: add cluster_id to kv_indexer payload and run clustering after indexing"
```

---

## Task 8: PRS Evaluator — Three-Signal Per-Cluster Evaluation

**Files:**
- Modify: `pipeline/prs_evaluator.py`
- Modify: `tests/test_prs_evaluator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prs_evaluator.py — add these

def test_sample_vdb_coverage_returns_float_in_range():
    from unittest.mock import MagicMock, patch
    import numpy as np
    from pipeline.prs_evaluator import sample_vdb_coverage

    mock_store = MagicMock()
    mock_store.scroll.return_value = [
        MagicMock(payload={"text": "What is RAG?", "cluster_id": "0"}),
        MagicMock(payload={"text": "How does FAISS work?", "cluster_id": "0"}),
    ]
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("pipeline.prs_evaluator._generate_parametric", return_value="Some answer"):
        with patch("pipeline.prs_evaluator._cosine_sim", return_value=0.8):
            score = sample_vdb_coverage(mock_store, "0", mock_model,
                                        mock_tokenizer, embed_model="BAAI/bge-small-en-v1.5",
                                        sample_size=2)
    assert 0.0 <= score <= 1.0


def test_evaluate_per_cluster_returns_dict(tmp_path):
    """evaluate() should return {cluster_id: prs_float} when clusters exist."""
    from unittest.mock import MagicMock, patch
    from pipeline.prs_evaluator import evaluate
    import core.version as ver

    ver.VERSION_FILE = tmp_path / "version.json"
    ver.save_cluster_state("0", {"phase": 1, "threshold": 0.72})

    faqs = [{"question": "Q1", "answer": "A1", "cluster_id": "0"}] * 5

    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "faq_question_key": "question",
        "faq_answer_key": "answer",
        "prs_signal_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
        "prs_auto_weight": False,
        "prs_stability_window": 3,
        "prs_advancement_threshold": 0.72,
        "min_cluster_samples_for_adaptation": 10,
        "query_log_db": str(tmp_path / "q.db"),
        "realtime_requery_window_minutes": 10,
    }
    with patch("pipeline.prs_evaluator._generate_parametric", return_value="answer"):
        with patch("pipeline.prs_evaluator._cosine_sim", return_value=0.9):
            with patch("pipeline.prs_evaluator.sample_vdb_coverage", return_value=0.8):
                with patch("pipeline.prs_evaluator._self_consistency", return_value=0.9):
                    with patch("pipeline.prs_evaluator._extract_confidence", return_value=0.85):
                        result = evaluate(faqs, cfg, lora_checkpoint=None, store=None)
    assert isinstance(result, dict)
    assert "0" in result
    assert 0.0 <= result["0"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_prs_evaluator.py::test_sample_vdb_coverage_returns_float_in_range tests/test_prs_evaluator.py::test_evaluate_per_cluster_returns_dict -v --override-ini="addopts="
```

Expected: `FAILED — ImportError` or `AttributeError`

- [ ] **Step 3: Add sample_vdb_coverage to pipeline/prs_evaluator.py**

Add this function after the existing `_self_consistency` function:

```python
def sample_vdb_coverage(
    store,
    cluster_id: str,
    model,
    tokenizer,
    embed_model: str,
    sample_size: int = 20,
    accuracy_threshold: float = 0.70,
) -> float:
    """Fraction of sampled VDB chunks answerable from model weights."""
    from transformers import pipeline as hf_pipeline
    if store is None:
        return 0.0
    try:
        points = store.scroll(filter={"cluster_id": cluster_id}, limit=sample_size)
    except Exception:
        return 0.0
    if not points:
        return 0.0
    pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=128, do_sample=False)
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)
    correct = 0
    for point in points:
        text = point.payload.get("text", "")
        if not text:
            continue
        question = f"Summarize the key information in: {text[:200]}"
        param_ans = _generate_parametric(question, pipe)
        embs = np.array(list(embedder.embed([param_ans, text])))
        sim = _cosine_sim(embs[0], embs[1])
        if sim >= accuracy_threshold:
            correct += 1
    return correct / len(points) if points else 0.0
```

- [ ] **Step 4: Update evaluate() signature and body**

Change the `evaluate` function signature to accept `store=None` and return `dict[str, float]` (per-cluster PRS) instead of a single float:

```python
def evaluate(
    faqs: list[dict], cfg: dict, lora_checkpoint: str | None = None, store=None
) -> dict[str, float]:
    """Compute per-cluster PRS. Returns {cluster_id: prs_score}."""
    model, tokenizer = model_loader.load(lora_checkpoint)
    embed_model = cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
    q_key = cfg.get("faq_question_key", "question")
    a_key = cfg.get("faq_answer_key", "answer")

    from transformers import pipeline as hf_pipeline
    pipe_gen = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                            max_new_tokens=256, do_sample=False)
    pipe_conf = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                             max_new_tokens=5, do_sample=False)
    pipe_sample = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                               max_new_tokens=128, do_sample=True, temperature=0.7)
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)

    # Group FAQs by cluster_id
    from collections import defaultdict
    cluster_faqs: dict[str, list] = defaultdict(list)
    for faq in faqs:
        cid = str(faq.get("cluster_id", "default"))
        cluster_faqs[cid].append(faq)

    results: dict[str, float] = {}
    from core.prs_adapter import update_cluster_after_round
    from pipeline.query_logger import init_db, get_cluster_stats
    init_db(cfg.get("query_log_db", "query_log.db"))

    for cid, cluster_faq_list in cluster_faqs.items():
        accuracy_ratios, calibrations, consistencies = [], [], []
        good_queries = []

        for faq in cluster_faq_list:
            q, gt = _extract_qa(faq, q_key=q_key, a_key=a_key)
            param_ans = _generate_parametric(q, pipe_gen)
            embs = np.array(list(embedder.embed([param_ans, gt])))
            param_sim = _cosine_sim(embs[0], embs[1])
            accuracy_ratios.append(min(param_sim, 1.0))
            self_conf = _extract_confidence(param_ans, pipe_conf)
            calibrations.append(1.0 - abs(self_conf - param_sim))
            consistencies.append(_self_consistency(q, pipe_sample, embedder))
            if param_sim >= 0.85:
                good_queries.append(q)

        weights_internal = cfg.get("prs_weights",
                                    {"accuracy": 0.5, "calibration": 0.3, "consistency": 0.2})
        faq_prs = _compute_prs(accuracy_ratios, calibrations, consistencies, weights_internal)

        vdb_coverage = sample_vdb_coverage(
            store, cid, model, tokenizer, embed_model,
            sample_size=cfg.get("vdb_sample_size", 20),
        )

        realtime_stats = get_cluster_stats(
            cfg.get("query_log_db", "query_log.db"), cid,
            window_minutes=cfg.get("realtime_requery_window_minutes", 10),
        )

        state = update_cluster_after_round(cid, faq_prs, vdb_coverage, realtime_stats, cfg)
        results[cid] = state["prs"]

        if good_queries:
            good_embs = [e.astype(float).tolist() for e in embedder.embed(good_queries)]
            data = ver.load()
            existing = data.get("known_good_queries", [])
            data["known_good_queries"] = existing + good_embs
            ver.save(data)

    return results
```

- [ ] **Step 5: Run updated tests**

```bash
python -m pytest tests/test_prs_evaluator.py -v --override-ini="addopts="
```

Expected: all tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add pipeline/prs_evaluator.py tests/test_prs_evaluator.py
git commit -m "feat: update prs_evaluator to return per-cluster PRS with three-signal computation"
```

---

## Task 9: Greenfield Routing in kv_inference

**Files:**
- Modify: `pipeline/kv_inference.py`
- Modify: `tests/test_kv_inference.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kv_inference.py — add these

def test_route_query_greenfield_phase3_returns_parametric():
    from unittest.mock import MagicMock, patch
    import numpy as np
    from pipeline.kv_inference import route_query

    cfg = {"deployment_mode": "greenfield", "checkpoint_dir": "/tmp/ckpt",
           "query_log_db": "/tmp/q.db", "realtime_requery_window_minutes": 10}

    mock_clusters = {
        "centroids": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "k": 2,
    }
    mock_cluster_state = {"phase": 3}

    with patch("pipeline.kv_inference.load_clusters", return_value=mock_clusters):
        with patch("pipeline.kv_inference.nearest_cluster", return_value=0):
            with patch("pipeline.kv_inference.ver.get_cluster_state",
                       return_value=mock_cluster_state):
                with patch("pipeline.kv_inference.answer_parametric",
                           return_value="parametric answer") as mock_param:
                    result = route_query("test query", np.array([1.0, 0.0]), cfg)
    mock_param.assert_called_once()
    assert result == "parametric answer"


def test_route_query_greenfield_phase1_returns_retrieval():
    from unittest.mock import MagicMock, patch
    import numpy as np
    from pipeline.kv_inference import route_query

    cfg = {"deployment_mode": "greenfield", "checkpoint_dir": "/tmp/ckpt",
           "query_log_db": "/tmp/q.db", "realtime_requery_window_minutes": 10}
    mock_clusters = {"centroids": np.array([[1.0, 0.0]]), "k": 1}
    mock_cluster_state = {"phase": 1}

    with patch("pipeline.kv_inference.load_clusters", return_value=mock_clusters):
        with patch("pipeline.kv_inference.nearest_cluster", return_value=0):
            with patch("pipeline.kv_inference.ver.get_cluster_state",
                       return_value=mock_cluster_state):
                with patch("pipeline.kv_inference.answer_with_retrieval",
                           return_value="retrieved answer") as mock_ret:
                    result = route_query("test query", np.array([1.0, 0.0]), cfg)
    mock_ret.assert_called_once()
    assert result == "retrieved answer"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_kv_inference.py::test_route_query_greenfield_phase3_returns_parametric tests/test_kv_inference.py::test_route_query_greenfield_phase1_returns_retrieval -v --override-ini="addopts="
```

Expected: `FAILED — ImportError: cannot import name 'route_query'`

- [ ] **Step 3: Add route_query and answer_parametric to pipeline/kv_inference.py**

Add these two functions at the end of `pipeline/kv_inference.py`:

```python
def answer_parametric(query: str, cfg: dict) -> str:
    """Answer query directly from LoRA model weights (no retrieval)."""
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=cfg.get("max_new_tokens", 256),
                                 do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def route_query(query: str, query_embedding, cfg: dict) -> str:
    """Dispatcher: greenfield per-cluster routing or brownfield confidence routing."""
    from pathlib import Path
    from core.cluster_manager import load_clusters, nearest_cluster
    from pipeline.query_logger import init_db, log_query, mark_requeried

    init_db(cfg.get("query_log_db", "query_log.db"))
    mark_requeried(cfg.get("query_log_db", "query_log.db"), query,
                   cfg.get("realtime_requery_window_minutes", 10))

    clusters_path = str(Path(cfg.get("checkpoint_dir", ".")) / "clusters.json")
    if not Path(clusters_path).exists():
        # No clusters yet — fall back to legacy retrieval
        return answer_with_retrieval(query, cfg)

    clusters_data = load_clusters(clusters_path)
    cluster_id = str(nearest_cluster(query_embedding, clusters_data["centroids"]))
    cluster_state = ver.get_cluster_state(cluster_id)
    phase = cluster_state.get("phase", 1)

    if phase >= 3:
        answer = answer_parametric(query, cfg)
        log_query(cfg.get("query_log_db", "query_log.db"), query, answer,
                  "parametric", cluster_id=cluster_id)
        return answer
    else:
        answer = answer_with_retrieval(query, cfg)
        log_query(cfg.get("query_log_db", "query_log.db"), query, answer,
                  "retrieval", cluster_id=cluster_id)
        return answer
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_kv_inference.py -v --override-ini="addopts="
```

Expected: all tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/kv_inference.py tests/test_kv_inference.py
git commit -m "feat: add per-cluster route_query dispatcher to kv_inference"
```

---

## Task 10: Brownfield Per-Chunk Confidence Scoring

**Files:**
- Create: `pipeline/chunk_confidence.py`
- Create: `tests/test_chunk_confidence.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunk_confidence.py

from unittest.mock import MagicMock, patch
import pytest
from pipeline.chunk_confidence import (
    score_chunk, get_eligible_chunks, brownfield_coverage_stats,
)


def test_score_chunk_high_similarity():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    with patch("pipeline.chunk_confidence._generate_parametric", return_value="good answer"):
        with patch("pipeline.chunk_confidence._cosine_sim", return_value=0.88):
            score = score_chunk("RAG retrieves context from a database.",
                                mock_model, mock_tokenizer,
                                embed_model="BAAI/bge-small-en-v1.5")
    assert 0.0 <= score <= 1.0


def test_get_eligible_chunks_filters_stale(tmp_path):
    points = [
        MagicMock(id="a", payload={"confidence_lora_version": 1}),
        MagicMock(id="b", payload={"confidence_lora_version": None}),
        MagicMock(id="c", payload={"confidence_lora_version": 2}),
    ]
    eligible = get_eligible_chunks(points, current_lora_version=2)
    ids = [p.id for p in eligible]
    assert "b" in ids  # None → stale → eligible
    assert "a" in ids  # version 1 < 2 → stale → eligible
    assert "c" not in ids  # version 2 == current → fresh → skip


def test_brownfield_coverage_stats_all_above_floor():
    points = [
        MagicMock(payload={"model_confidence": 0.9}),
        MagicMock(payload={"model_confidence": 0.85}),
    ]
    stats = brownfield_coverage_stats(points, confidence_floor=0.80)
    assert stats["coverage_pct"] == 1.0
    assert stats["total_chunks"] == 2


def test_brownfield_coverage_stats_partial():
    points = [
        MagicMock(payload={"model_confidence": 0.9}),
        MagicMock(payload={"model_confidence": 0.5}),
        MagicMock(payload={"model_confidence": None}),
    ]
    stats = brownfield_coverage_stats(points, confidence_floor=0.80)
    assert abs(stats["coverage_pct"] - 1/3) < 1e-6


def test_brownfield_coverage_stats_empty():
    stats = brownfield_coverage_stats([], confidence_floor=0.80)
    assert stats["coverage_pct"] == 0.0
    assert stats["total_chunks"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_chunk_confidence.py -v --override-ini="addopts="
```

Expected: `ERROR — ModuleNotFoundError: No module named 'pipeline.chunk_confidence'`

- [ ] **Step 3: Create the module**

```python
# pipeline/chunk_confidence.py

from __future__ import annotations
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from transformers import pipeline as hf_pipeline

from pipeline.prs_evaluator import _cosine_sim


def _generate_parametric(question: str, pipe) -> str:
    out = pipe(question)
    return out[0]["generated_text"][len(question):].strip()


def score_chunk(
    chunk_text: str,
    model,
    tokenizer,
    embed_model: str,
    accuracy_threshold: float = 0.0,
) -> float:
    """Ask the model to answer a question derived from chunk_text; return cosine similarity."""
    question = f"Summarize the key information in: {chunk_text[:300]}"
    pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=128, do_sample=False)
    param_ans = _generate_parametric(question, pipe)
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)
    embs = np.array(list(embedder.embed([param_ans, chunk_text])))
    return _cosine_sim(embs[0], embs[1])


def get_eligible_chunks(points: list[Any], current_lora_version: int) -> list[Any]:
    """Return chunks whose model_confidence is stale (version < current or None)."""
    return [
        p for p in points
        if p.payload.get("confidence_lora_version") is None
        or p.payload["confidence_lora_version"] < current_lora_version
    ]


def brownfield_coverage_stats(points: list[Any], confidence_floor: float) -> dict:
    """Return {coverage_pct, total_chunks, mastered_chunks}."""
    if not points:
        return {"coverage_pct": 0.0, "total_chunks": 0, "mastered_chunks": 0}
    mastered = sum(
        1 for p in points
        if p.payload.get("model_confidence") is not None
        and p.payload["model_confidence"] >= confidence_floor
    )
    return {
        "coverage_pct": mastered / len(points),
        "total_chunks": len(points),
        "mastered_chunks": mastered,
    }


def run_brownfield_scoring(store, model, tokenizer, embed_model: str,
                            current_lora_version: int, cfg: dict) -> dict:
    """Score all stale chunks; update their model_confidence in the VDB. Returns stats."""
    all_points = store.scroll(limit=10_000)
    eligible = get_eligible_chunks(all_points, current_lora_version)
    for point in eligible:
        text = point.payload.get("text", "")
        if not text:
            continue
        confidence = score_chunk(text, model, tokenizer, embed_model)
        store.update_payload(point.id, {
            "model_confidence": round(float(confidence), 4),
            "confidence_lora_version": current_lora_version,
        })
    return brownfield_coverage_stats(all_points, cfg.get("brownfield_confidence_floor", 0.80))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_chunk_confidence.py -v --override-ini="addopts="
```

Expected: all 5 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/chunk_confidence.py tests/test_chunk_confidence.py
git commit -m "feat: add brownfield per-chunk confidence scoring"
```

---

## Task 11: Brownfield Routing in kv_inference

**Files:**
- Modify: `pipeline/kv_inference.py`
- Modify: `tests/test_kv_inference.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kv_inference.py — add these

def test_route_query_brownfield_all_high_confidence_goes_parametric():
    from unittest.mock import MagicMock, patch
    import numpy as np
    from pipeline.kv_inference import route_query_brownfield

    cfg = {"brownfield_routing_threshold": 0.80, "query_log_db": "/tmp/q.db",
           "realtime_requery_window_minutes": 10, "max_new_tokens": 64}
    chunks = [
        {"model_confidence": 0.90, "chunk_id": "a"},
        {"model_confidence": 0.85, "chunk_id": "b"},
    ]
    with patch("pipeline.kv_inference.answer_parametric", return_value="param") as mp:
        result = route_query_brownfield("Q?", chunks, cfg)
    mp.assert_called_once()
    assert result == "param"


def test_route_query_brownfield_low_confidence_goes_retrieval():
    from unittest.mock import patch
    from pipeline.kv_inference import route_query_brownfield

    cfg = {"brownfield_routing_threshold": 0.80, "query_log_db": "/tmp/q.db",
           "realtime_requery_window_minutes": 10}
    chunks = [
        {"model_confidence": 0.90, "chunk_id": "a"},
        {"model_confidence": 0.50, "chunk_id": "b"},
    ]
    with patch("pipeline.kv_inference.answer_with_retrieval", return_value="retrieved") as mr:
        result = route_query_brownfield("Q?", chunks, cfg)
    mr.assert_called_once()
    assert result == "retrieved"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_kv_inference.py::test_route_query_brownfield_all_high_confidence_goes_parametric tests/test_kv_inference.py::test_route_query_brownfield_low_confidence_goes_retrieval -v --override-ini="addopts="
```

Expected: `FAILED — ImportError: cannot import name 'route_query_brownfield'`

- [ ] **Step 3: Add route_query_brownfield to pipeline/kv_inference.py**

```python
def route_query_brownfield(query: str, retrieved_chunks: list[dict], cfg: dict) -> str:
    """Brownfield routing: go parametric only if ALL retrieved chunks have high confidence."""
    from pipeline.query_logger import init_db, log_query

    init_db(cfg.get("query_log_db", "query_log.db"))
    threshold = cfg.get("brownfield_routing_threshold", 0.85)
    all_confident = all(
        c.get("model_confidence") is not None and c["model_confidence"] >= threshold
        for c in retrieved_chunks
    )

    if all_confident:
        answer = answer_parametric(query, cfg)
        log_query(cfg.get("query_log_db", "query_log.db"), query, answer, "parametric")
        return answer
    else:
        answer = answer_with_retrieval(query, cfg)
        log_query(cfg.get("query_log_db", "query_log.db"), query, answer, "retrieval")
        return answer
```

- [ ] **Step 4: Run all kv_inference tests**

```bash
python -m pytest tests/test_kv_inference.py -v --override-ini="addopts="
```

Expected: all tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/kv_inference.py tests/test_kv_inference.py
git commit -m "feat: add brownfield confidence-gated routing to kv_inference"
```

---

## Task 12: Dashboard — Editable UC Settings Panel

**Files:**
- Modify: `studio/routes.py`
- Modify: `tests/test_studio_routes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_routes.py — add these

def test_get_uc_settings_returns_editable_fields(client, tmp_config):
    resp = client.get(f"/api/uc/{tmp_config}/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "realtime_requery_window_minutes" in data
    assert "brownfield_confidence_floor" in data
    assert "brownfield_coverage_target" in data
    assert "brownfield_routing_threshold" in data
    assert "prs_advancement_threshold" in data


def test_patch_uc_settings_updates_and_persists(client, tmp_config):
    resp = client.patch(
        f"/api/uc/{tmp_config}/settings",
        json={"realtime_requery_window_minutes": 15, "brownfield_confidence_floor": 0.75},
    )
    assert resp.status_code == 200
    # Verify persisted
    resp2 = client.get(f"/api/uc/{tmp_config}/settings")
    data = resp2.json()
    assert data["realtime_requery_window_minutes"] == 15
    assert data["brownfield_confidence_floor"] == 0.75


def test_patch_uc_settings_rejects_unknown_fields(client, tmp_config):
    resp = client.patch(
        f"/api/uc/{tmp_config}/settings",
        json={"nonexistent_field": 999},
    )
    assert resp.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_studio_routes.py::test_get_uc_settings_returns_editable_fields tests/test_studio_routes.py::test_patch_uc_settings_updates_and_persists tests/test_studio_routes.py::test_patch_uc_settings_rejects_unknown_fields -v --override-ini="addopts="
```

Expected: `FAILED — 404 Not Found` (route does not exist)

- [ ] **Step 3: Add the settings endpoints to studio/routes.py**

Add this block after the existing UC management routes:

```python
_EDITABLE_SETTINGS = {
    "realtime_requery_window_minutes",
    "brownfield_confidence_floor",
    "brownfield_coverage_target",
    "brownfield_routing_threshold",
    "prs_advancement_threshold",
}


@router.get("/api/uc/{config_name}/settings")
async def get_uc_settings(config_name: str):
    cfg_path = _config_path(config_name)
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail="UC config not found")
    with open(cfg_path) as f:
        cfg = json.load(f)
    return {k: cfg.get(k) for k in _EDITABLE_SETTINGS}


@router.patch("/api/uc/{config_name}/settings")
async def patch_uc_settings(config_name: str, body: dict):
    unknown = set(body) - _EDITABLE_SETTINGS
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown settings: {sorted(unknown)}. Editable: {sorted(_EDITABLE_SETTINGS)}",
        )
    cfg_path = _config_path(config_name)
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail="UC config not found")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.update(body)
    tmp = cfg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    tmp.replace(cfg_path)
    return {"updated": list(body.keys()), "status": "ok"}
```

Note: `_config_path` is a helper that must already exist in `studio/routes.py` or be added as:
```python
def _config_path(config_name: str) -> Path:
    return Path(config_name) if config_name.endswith(".json") else Path(f"{config_name}.json")
```

- [ ] **Step 4: Run the new tests**

```bash
python -m pytest tests/test_studio_routes.py::test_get_uc_settings_returns_editable_fields tests/test_studio_routes.py::test_patch_uc_settings_updates_and_persists tests/test_studio_routes.py::test_patch_uc_settings_rejects_unknown_fields -v --override-ini="addopts="
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Run full studio routes tests**

```bash
python -m pytest tests/test_studio_routes.py -v --override-ini="addopts="
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add studio/routes.py tests/test_studio_routes.py
git commit -m "feat: add dashboard-editable UC settings GET/PATCH endpoints"
```

---

## Task 13: Migration CLI + Dashboard Button

**Files:**
- Modify: `kvforge.py` (add `migrate-to-greenfield` subcommand)
- Modify: `studio/routes.py` (add migration eligibility endpoint + trigger)
- Modify: `tests/test_studio_routes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_routes.py — add these

def test_get_migration_eligibility_returns_stats(client, tmp_config):
    with patch("studio.routes.brownfield_coverage_stats",
               return_value={"coverage_pct": 0.65, "total_chunks": 100, "mastered_chunks": 65}):
        resp = client.get(f"/api/uc/{tmp_config}/migration-eligibility")
    assert resp.status_code == 200
    data = resp.json()
    assert "coverage_pct" in data
    assert "eligible" in data
    assert data["eligible"] is False  # 0.65 < 0.70 default target


def test_trigger_migration_when_eligible(client, tmp_config):
    with patch("studio.routes.brownfield_coverage_stats",
               return_value={"coverage_pct": 0.80, "total_chunks": 100, "mastered_chunks": 80}):
        with patch("studio.routes._run_greenfield_migration", return_value=None):
            resp = client.post(f"/api/uc/{tmp_config}/migrate-to-greenfield")
    assert resp.status_code == 200


def test_trigger_migration_when_not_eligible_returns_409(client, tmp_config):
    with patch("studio.routes.brownfield_coverage_stats",
               return_value={"coverage_pct": 0.40, "total_chunks": 100, "mastered_chunks": 40}):
        resp = client.post(f"/api/uc/{tmp_config}/migrate-to-greenfield")
    assert resp.status_code == 409
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_studio_routes.py::test_get_migration_eligibility_returns_stats tests/test_studio_routes.py::test_trigger_migration_when_eligible tests/test_studio_routes.py::test_trigger_migration_when_not_eligible_returns_409 -v --override-ini="addopts="
```

Expected: `FAILED — 404 Not Found`

- [ ] **Step 3: Add migration endpoints to studio/routes.py**

```python
from pipeline.chunk_confidence import brownfield_coverage_stats


def _run_greenfield_migration(cfg_path: Path, cfg: dict) -> None:
    """Cluster existing embeddings and switch deployment_mode to greenfield."""
    from vectorstore.registry import get_store
    from core.cluster_manager import cluster_embeddings, save_clusters
    import numpy as np
    store = get_store(cfg)
    all_points = store.scroll(limit=10_000)
    embeddings = np.array([p.vector for p in all_points if hasattr(p, "vector")])
    if len(embeddings) < 6:
        return
    k_range = tuple(cfg.get("cluster_k_range", [3, 20]))
    centroids, labels = cluster_embeddings(embeddings, k_range=k_range)
    clusters_path = str(Path(cfg.get("checkpoint_dir", ".")) / "clusters.json")
    save_clusters(clusters_path, centroids, labels)
    cfg["deployment_mode"] = "greenfield"
    tmp = cfg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    tmp.replace(cfg_path)
    print(f"✅ Migrated {cfg_path.name} to greenfield mode with {len(centroids)} clusters")


@router.get("/api/uc/{config_name}/migration-eligibility")
async def get_migration_eligibility(config_name: str):
    cfg_path = _config_path(config_name)
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail="UC config not found")
    with open(cfg_path) as f:
        cfg = json.load(f)
    from vectorstore.registry import get_store
    store = get_store(cfg)
    all_points = store.scroll(limit=10_000)
    floor = cfg.get("brownfield_confidence_floor", 0.80)
    target = cfg.get("brownfield_coverage_target", 0.70)
    stats = brownfield_coverage_stats(all_points, floor)
    stats["eligible"] = stats["coverage_pct"] >= target
    stats["target"] = target
    stats["confidence_floor"] = floor
    return stats


@router.post("/api/uc/{config_name}/migrate-to-greenfield")
async def trigger_migration(config_name: str):
    cfg_path = _config_path(config_name)
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail="UC config not found")
    with open(cfg_path) as f:
        cfg = json.load(f)
    from vectorstore.registry import get_store
    store = get_store(cfg)
    all_points = store.scroll(limit=10_000)
    floor = cfg.get("brownfield_confidence_floor", 0.80)
    target = cfg.get("brownfield_coverage_target", 0.70)
    stats = brownfield_coverage_stats(all_points, floor)
    if stats["coverage_pct"] < target:
        raise HTTPException(
            status_code=409,
            detail=f"Not eligible: {stats['coverage_pct']:.1%} mastered, need {target:.1%}",
        )
    _run_greenfield_migration(cfg_path, cfg)
    return {"status": "migration_started", "clusters_path": str(
        Path(cfg.get("checkpoint_dir", ".")) / "clusters.json"
    )}
```

- [ ] **Step 4: Add migrate-to-greenfield CLI subcommand to kvforge.py**

Locate the `subparsers` block in `kvforge.py` and add:

```python
p_migrate = subparsers.add_parser(
    "migrate-to-greenfield",
    help="Cluster existing VDB embeddings and switch UC to greenfield mode",
)
p_migrate.add_argument("--config", required=True, help="Path to UC config JSON")
```

In the dispatch block, add:

```python
elif args.command == "migrate-to-greenfield":
    from core.config import load_config
    from pathlib import Path
    cfg = load_config(args.config)
    cfg_dict = cfg.model_dump()
    from studio.routes import _run_greenfield_migration
    _run_greenfield_migration(Path(args.config), cfg_dict)
```

- [ ] **Step 5: Run the migration tests**

```bash
python -m pytest tests/test_studio_routes.py::test_get_migration_eligibility_returns_stats tests/test_studio_routes.py::test_trigger_migration_when_eligible tests/test_studio_routes.py::test_trigger_migration_when_not_eligible_returns_409 -v --override-ini="addopts="
```

Expected: all 3 `PASSED`

- [ ] **Step 6: Commit**

```bash
git add studio/routes.py kvforge.py tests/test_studio_routes.py
git commit -m "feat: add migration eligibility endpoint, migrate-to-greenfield trigger, and CLI command"
```

---

## Task 14: Integration Smoke Test

**Files:**
- Modify: `tests/test_integration_smoke.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration_smoke.py — add this test

def test_dynamic_prs_greenfield_round_trip(tmp_path):
    """Verify cluster creation → PRS update → advancement state written to version.json."""
    import numpy as np
    from core.cluster_manager import cluster_embeddings, save_clusters
    from core.prs_adapter import update_cluster_after_round
    from pipeline.query_logger import init_db, log_query
    import core.version as ver

    ver.VERSION_FILE = tmp_path / "version.json"
    db_path = str(tmp_path / "q.db")
    clusters_path = str(tmp_path / "clusters.json")
    init_db(db_path)

    # Simulate 60 chunks in 3 tight clusters
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((3, 16)) * 5
    embs = np.vstack([rng.standard_normal((20, 16)) + centers[i] for i in range(3)])
    centroids, labels = cluster_embeddings(embs, k_range=(2, 5))
    save_clusters(clusters_path, centroids, labels)
    assert len(centroids) == 3

    # Log some retrieval queries for cluster "0"
    for i in range(5):
        log_query(db_path, f"question {i}", f"answer {i}", "parametric", cluster_id="0")

    cfg = {
        "prs_signal_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
        "prs_auto_weight": False,
        "prs_stability_window": 3,
        "prs_advancement_threshold": 0.72,
        "min_cluster_samples_for_adaptation": 10,
        "query_log_db": db_path,
        "realtime_requery_window_minutes": 10,
    }

    # Three LoRA rounds, improving coverage each time
    for round_num in range(3):
        faq_cov = 0.60 + round_num * 0.08
        vdb_cov = 0.55 + round_num * 0.10
        from pipeline.query_logger import get_cluster_stats
        realtime = get_cluster_stats(db_path, "0", window_minutes=10)
        state = update_cluster_after_round("0", faq_cov, vdb_cov, realtime, cfg)

    # After 3 rounds of improvement, cluster state should be persisted
    loaded = ver.get_cluster_state("0")
    assert len(loaded["prs_history"]) == 3
    assert loaded["prs"] > 0.0
    assert "phase" in loaded

    # Global phase reflects cluster states
    global_phase = ver.get_global_phase()
    assert global_phase >= 1
```

- [ ] **Step 2: Run test to verify it passes (no mocks — pure logic test)**

```bash
python -m pytest tests/test_integration_smoke.py::test_dynamic_prs_greenfield_round_trip -v --override-ini="addopts="
```

Expected: `PASSED`

- [ ] **Step 3: Run the full test suite**

```bash
python -m pytest tests/ -v --override-ini="addopts="
```

Expected: all tests pass. If any fail, investigate and fix before merging.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_smoke.py
git commit -m "test: add dynamic PRS greenfield round-trip integration smoke test"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Greenfield mode — Tasks 5, 6, 7, 8, 9
- ✅ Brownfield mode — Tasks 10, 11, 13
- ✅ Three-signal PRS (faq/vdb/realtime) — Tasks 6, 8
- ✅ Logistic regression weight adaptation — Task 6
- ✅ Per-cluster phase advancement (slope + threshold) — Task 6
- ✅ DifficultyEstimator pluggable framework — Task 2
- ✅ Dashboard-editable settings — Task 12
- ✅ Migration eligibility + trigger — Task 13
- ✅ Config schema — Task 1
- ✅ version.json cluster state — Task 3
- ✅ QueryLogger — Task 4

**Known scope boundary:** The `store.scroll(filter={"cluster_id": ...})` call in `sample_vdb_coverage` and `_run_greenfield_migration` requires that each vectorstore backend implements filtered scroll. Qdrant supports this natively. If using Chroma or FAISS backends, verify their scroll implementations support cluster_id filtering before running Task 8 and 13 on those backends.
