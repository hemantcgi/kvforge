import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fake_chunks(n=100, dim=64):
    """Return n chunks with deterministic embeddings."""
    chunks = []
    embeddings = []
    for i in range(n):
        chunks.append({"chunk_id": i, "text": f"Chunk {i} text."})
        emb = np.zeros(dim, dtype=np.float32)
        emb[i % dim] = 1.0
        embeddings.append(emb)
    return chunks, np.array(embeddings)


def test_stratified_subsample_returns_n_chunks():
    from tools.corpus_slicer import stratified_subsample
    chunks, embs = _fake_chunks(100)
    result = stratified_subsample(chunks, n=20, embeddings=embs, n_clusters=5)
    assert len(result) == 20
    assert all(isinstance(c, dict) for c in result)


def test_stratified_subsample_preserves_cluster_distribution():
    from tools.corpus_slicer import stratified_subsample
    chunks, embs = _fake_chunks(100)
    for i in range(100):
        embs[i] = np.zeros(64, dtype=np.float32)
        embs[i][i // 20] = 1.0
    result = stratified_subsample(chunks, n=25, embeddings=embs, n_clusters=5)
    result_ids = [c["chunk_id"] for c in result]
    for cluster_start in range(0, 100, 20):
        count = sum(1 for cid in result_ids if cluster_start <= cid < cluster_start + 20)
        assert count >= 3, f"Cluster {cluster_start//20} underrepresented: {count}/25"


def test_stratified_subsample_n_larger_than_corpus_returns_all():
    from tools.corpus_slicer import stratified_subsample
    chunks, embs = _fake_chunks(10)
    result = stratified_subsample(chunks, n=50, embeddings=embs, n_clusters=5)
    assert len(result) == 10


def test_filter_questions_by_chunks():
    from tools.corpus_slicer import filter_questions_by_chunks
    questions = [
        {"question": "Q1", "source_chunk_ids": [1, 2]},
        {"question": "Q2", "source_chunk_ids": [3, 4]},
        {"question": "Q3", "source_chunk_ids": [5]},
    ]
    result = filter_questions_by_chunks(questions, {1, 2, 3})
    assert len(result) == 1
    assert result[0]["question"] == "Q1"


def test_filter_questions_by_chunks_empty_ids():
    from tools.corpus_slicer import filter_questions_by_chunks
    questions = [{"question": "Q1", "source_chunk_ids": [1]}]
    result = filter_questions_by_chunks(questions, set())
    assert result == []


def test_create_size_tiers():
    from tools.corpus_slicer import create_size_tiers
    chunks, embs = _fake_chunks(100)
    tiers = create_size_tiers(chunks, embs, tiers=[10, 50, 100])
    assert len(tiers) == 3
    assert len(tiers[10]) == 10
    assert len(tiers[50]) == 50
    assert len(tiers[100]) == 100
    ids_10 = {c["chunk_id"] for c in tiers[10]}
    ids_50 = {c["chunk_id"] for c in tiers[50]}
    ids_100 = {c["chunk_id"] for c in tiers[100]}
    assert ids_10.issubset(ids_50), "Tier 10 not subset of tier 50"
    assert ids_10.issubset(ids_100), "Tier 10 not subset of tier 100"
    assert ids_50.issubset(ids_100), "Tier 50 not subset of tier 100"


def test_create_size_tiers_reverse_order():
    from tools.corpus_slicer import create_size_tiers
    chunks, embs = _fake_chunks(80)
    tiers = create_size_tiers(chunks, embs, tiers=[100, 50, 10])
    assert len(tiers[10]) == 10
    assert len(tiers[50]) == 50
    ids_10 = {c["chunk_id"] for c in tiers[10]}
    ids_50 = {c["chunk_id"] for c in tiers[50]}
    assert ids_10.issubset(ids_50)
