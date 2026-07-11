"""End-to-end V2 Corpus Intelligence lifecycle test.

Covers: index → sleep-time curation → enhanced promotion → three-path query → archive → reinstate
"""
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def tmp_workspace(tmp_path):
    cfg = {
        "collection": "e2e_test",
        "version_file": str(tmp_path / "version.json"),
        "per_token_kv_dir": str(tmp_path / "per_token_kv"),
        "kv_num_layers": 4, "kv_num_heads": 2, "kv_head_dim": 64,
        "chunk_size": 16,
    }
    (tmp_path / "version.json").write_text(
        json.dumps({"phase": 2, "lora_version": 0}))
    return cfg, tmp_path


def _make_chunks(n=5):
    rng = np.random.default_rng(0)
    chunks = []
    for i in range(n):
        emb = rng.standard_normal(128).astype(np.float32)
        emb /= np.linalg.norm(emb)
        chunks.append({
            "id": f"chunk_{i}",
            "vector": emb.tolist(),
            "payload": {
                "text": f"Chunk {i} text about topic {i % 3}.",
                "kv_cache": "AAAA",
                "kv_token_path": None,
                "status": "active",
                "hit_count": i * 10,
                "archive_retrieval_count": 0,
            }
        })
    return chunks


def test_cis_and_tier_actions(tmp_workspace):
    from addons.corpus_intelligence.cis import (
        compute_access_score, compute_uniqueness_score,
        compute_coverage_score, compute_cis,
    )
    from addons.corpus_intelligence.config import CorpusIntelligenceConfig
    from pipeline.corpus_curation import identify_tier_actions

    chunks = _make_chunks(5)
    cfg = CorpusIntelligenceConfig(
        enhanced_tier_threshold=0.6, archive_candidate_threshold=0.25,
        uniqueness_floor=0.1)

    hit_counts = {c["id"]: c["payload"]["hit_count"] for c in chunks}
    vecs = {c["id"]: np.array(c["vector"]) for c in chunks}
    vecs["chunk_1"] = vecs["chunk_0"].copy()  # make chunk_1 a duplicate

    access  = compute_access_score(hit_counts)
    unique  = compute_uniqueness_score(vecs)
    faq_res = {0: ["chunk_4", "chunk_3"], 1: ["chunk_4", "chunk_2"]}
    cov     = compute_coverage_score(faq_res)
    cis     = compute_cis(access, unique, cov)

    actions = identify_tier_actions(cis, unique, cfg)
    assert len(actions["promote_to_enhanced"]) >= 1
    assert "chunk_1" in actions["archive_candidates"]


def test_enhanced_path_save_load(tmp_workspace):
    cfg_dict, tmp_path = tmp_workspace
    from core.kv_utils import compute_per_token_kv, save_token_kv, load_token_kv

    pkv = tuple(
        (torch.randn(1, 2, 8, 64), torch.randn(1, 2, 8, 64))
        for _ in range(4)
    )
    arr = compute_per_token_kv(pkv)
    path = tmp_path / "per_token_kv" / "chunk_4.npz"
    save_token_kv(arr, path, tq_config=None)
    loaded = load_token_kv(path, tq_config=None)
    np.testing.assert_array_equal(arr, loaded)


def test_three_path_routing_selects_correct_path(tmp_workspace):
    cfg_dict, tmp_path = tmp_workspace
    from pipeline.kv_inference import route_chunk_injection
    from core.kv_utils import save_token_kv, serialize_kv

    kv_path = str(tmp_path / "per_token_kv" / "chunk4.npz")
    arr = np.zeros((4, 2, 2, 8, 64), dtype=np.float16)
    save_token_kv(arr, kv_path)

    # Valid base64 kv_cache for the active chunk path
    active_kv_arr = np.zeros((4, 2, 2, 64), dtype=np.float16)
    active_kv_b64 = serialize_kv(active_kv_arr)

    enhanced_chunk = {"id": "c4", "payload": {
        "text": "t", "kv_cache": "AAAA", "kv_token_path": kv_path, "status": "active"}}
    active_chunk = {"id": "c5", "payload": {
        "text": "t", "kv_cache": active_kv_b64, "kv_token_path": None, "status": "active"}}
    archive_chunk = {"id": "c6", "payload": {
        "text": "t", "kv_cache": "", "kv_token_path": None, "status": "archived",
        "archive_path": None, "archive_retrieval_count": 0}}

    r_enh = route_chunk_injection(enhanced_chunk, cfg=cfg_dict)
    r_act = route_chunk_injection(active_chunk, cfg={
        "kv_num_layers": 4, "kv_num_heads": 2, "kv_head_dim": 64})
    r_arc = route_chunk_injection(archive_chunk, cfg={})

    assert r_enh["path"] == "enhanced"
    assert r_act["path"] == "active"
    assert r_arc["path"] == "archive"


def test_archive_and_reinstate_lifecycle(tmp_workspace):
    cfg_dict, tmp_path = tmp_workspace
    from addons.corpus_intelligence.archive import LocalArchiveBackend
    from pipeline.archive_manager import archive_chunk, reinstate_chunk
    from pipeline.reinstatement_tracker import check_reinstatement_candidates

    backend = LocalArchiveBackend(str(tmp_path / "archive"))
    mock_vs = MagicMock()
    chunk = {"id": "chunk_x", "payload": {
        "text": "Important archived content.",
        "kv_cache": "AAAA", "kv_token_path": None, "status": "active"}}

    archive_chunk(chunk, backend=backend, vector_store=mock_vs, collection="col")
    assert backend.read("chunk_x") == "Important archived content."

    archived_record = {"id": "chunk_x", "payload": {
        "archive_retrieval_count": 6, "status": "archived", "text": "..."}}
    candidates = check_reinstatement_candidates([archived_record], threshold=5)
    assert len(candidates) == 1
    assert candidates[0]["id"] == "chunk_x"

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    fake_pkv = tuple((torch.randn(1, 2, 4, 64), torch.randn(1, 2, 4, 64)) for _ in range(4))
    mock_out = MagicMock()
    mock_out.past_key_values = fake_pkv
    mock_model = MagicMock()
    mock_model.return_value = mock_out

    reinstate_chunk(
        {"id": "chunk_x", "payload": {"archive_path": backend.get_pointer("chunk_x")}},
        backend=backend, vector_store=mock_vs,
        collection="col", model=mock_model, tokenizer=mock_tokenizer,
        cfg=cfg_dict,
    )
    assert backend.read("chunk_x") is None
    calls = [c[1]["payload"] for c in mock_vs.update_payload.call_args_list]
    reinstate_call = calls[-1]
    assert reinstate_call["status"] == "active"
    assert reinstate_call["archive_path"] is None


def test_curation_run_pass_persists_cis(tmp_workspace):
    cfg_dict, tmp_path = tmp_workspace
    from pipeline.corpus_curation import run_curation_pass
    from addons.corpus_intelligence.config import CorpusIntelligenceConfig

    chunks = _make_chunks(3)
    faqs = ["What is RAG?", "How does caching work?"]
    ci_cfg = CorpusIntelligenceConfig()

    mock_embedder = MagicMock()
    mock_embedder.embed.side_effect = [
        np.array([[1, 0] + [0] * 126], dtype=np.float32),
        np.array([[0, 1] + [0] * 126], dtype=np.float32),
    ]

    actions = run_curation_pass(
        faqs=faqs, chunks=chunks, embedder=mock_embedder,
        cfg=ci_cfg, version_file=cfg_dict["version_file"],
    )
    cis_path = Path(cfg_dict["version_file"]).with_suffix(".cis.json")
    assert cis_path.exists(), "CIS scores should be persisted to disk"
    cis_data = json.loads(cis_path.read_text())
    assert len(cis_data) == 3
    assert "promote_to_enhanced" in actions
    assert "archive_candidates" in actions
