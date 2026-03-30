# tests/test_studio_migration.py
import sys, json, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from studio.migration import migrate_existing_use_cases, load_registry


def _make_fake_examples(tmp: Path):
    """Create minimal fake examples/usecase3_squad/config.json."""
    uc = tmp / "examples" / "usecase3_squad"
    uc.mkdir(parents=True)
    cfg = {
        "collection": "squad-qa",
        "vector_store": "faiss",
        "vector_dim": 384,
        "chunk_size": 600,
        "chunk_overlap": 60,
        "embed_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "quantization": "4bit",
        "vllm_url": "http://localhost:8093",
        "loader": "jsonl",
        "dashboard_port": 8083,
    }
    (uc / "config.json").write_text(json.dumps(cfg))
    return tmp


def test_migrate_creates_registry(tmp_path):
    _make_fake_examples(tmp_path)
    migrate_existing_use_cases(root=tmp_path)
    registry_path = tmp_path / "kvforge_registry.json"
    assert registry_path.exists()
    data = json.loads(registry_path.read_text())
    assert any(uc["id"] == "usecase3_squad" for uc in data["use_cases"])


def test_migrate_creates_uc_config(tmp_path):
    _make_fake_examples(tmp_path)
    migrate_existing_use_cases(root=tmp_path)
    cfg_path = tmp_path / "examples" / "usecase3_squad" / "uc_config.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text())
    assert cfg["vectordb"]["store"] == "faiss"
    assert cfg["vectordb"]["dimensions"] == 384
    assert cfg["llm"]["vllm_url"] == "http://localhost:8093"


def test_migrate_idempotent(tmp_path):
    _make_fake_examples(tmp_path)
    migrate_existing_use_cases(root=tmp_path)
    migrate_existing_use_cases(root=tmp_path)  # Second run should not error
    data = json.loads((tmp_path / "kvforge_registry.json").read_text())
    ids = [uc["id"] for uc in data["use_cases"]]
    assert ids.count("usecase3_squad") == 1  # No duplicates


def test_load_registry_returns_list(tmp_path):
    _make_fake_examples(tmp_path)
    migrate_existing_use_cases(root=tmp_path)
    ucs = load_registry(root=tmp_path)
    assert isinstance(ucs, list)
    assert len(ucs) == 1
    assert ucs[0]["id"] == "usecase3_squad"


def test_uc_config_type_is_example(tmp_path):
    _make_fake_examples(tmp_path)
    migrate_existing_use_cases(root=tmp_path)
    cfg = json.loads((tmp_path / "examples" / "usecase3_squad" / "uc_config.json").read_text())
    assert cfg["type"] == "example"
