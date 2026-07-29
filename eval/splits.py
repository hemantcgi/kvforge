"""Held-out data splits and contamination guard for KVForge evaluation.

Every evaluation split is constructed so that FAQ questions used to train a
LoRA are disjoint from the questions used to evaluate it.  Where an official
train/dev split exists (SQuAD, PubMedQA) we use it.  For proprietary corpora
we use a fixed random hold-out fraction.  For Amazon Bedrock (UC4) there is no
native gold test set, so we expose a small hand-verified test set path.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


def _download_hf_dataset(name: str, config: str | None, split: str, cache_dir: Path | None = None):
    """Download a HuggingFace dataset split and return it as a Python object."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required for auto-download. "
            "Install it with: pip install datasets"
        ) from exc
    return load_dataset(name, config, split=split, cache_dir=str(cache_dir) if cache_dir else None)


def _save_squad_json(ds, path: Path):
    """Convert a HuggingFace squad_v2 split to the standard SQuAD JSON format."""
    data = {"data": [], "version": "2.0"}
    for row in ds:
        ans = row.get("answers", {"text": [], "answer_start": []})
        # HuggingFace gives answers as a dict of lists; the official SQuAD format
        # is a list of {text, answer_start} dicts.
        answers = [
            {"text": t, "answer_start": s}
            for t, s in zip(ans.get("text", []), ans.get("answer_start", []))
        ]
        article = {
            "title": row.get("title", ""),
            "paragraphs": [
                {
                    "context": row.get("context", ""),
                    "qas": [
                        {
                            "question": row.get("question", ""),
                            "id": row.get("id", ""),
                            "is_impossible": row.get("is_impossible", False),
                            "answers": answers,
                        }
                    ],
                }
            ],
        }
        data["data"].append(article)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _save_pubmedqa_json(ds, path: Path):
    """Convert a HuggingFace PubMedQA split to the official JSON dict format."""
    data = {}
    for row in ds:
        pid = str(row.get("pubid", len(data)))
        data[pid] = {
            "QUESTION": row.get("question", ""),
            "LONG_ANSWER": row.get("long_answer", ""),
            "final_decision": row.get("final_decision", ""),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_disjoint(train_faqs: list[dict], eval_set: list[dict], key: str = "question") -> None:
    """Fail loudly if any eval question was also used for training.

    Args:
        train_faqs: FAQ list used for LoRA training.
        eval_set: Evaluation QA list.
        key: Field name containing the question text.
    """
    train_hashes = {_hash_text(str(f.get(key, "")).strip().lower()) for f in train_faqs}
    overlap = []
    for item in eval_set:
        q = str(item.get(key, "")).strip().lower()
        if _hash_text(q) in train_hashes:
            overlap.append(q[:120])
    if overlap:
        raise ValueError(
            f"Contamination detected: {len(overlap)} eval questions appear in "
            f"the training FAQ set. Examples: {overlap[:5]}"
        )


def load_json_faq(path: str | Path, q_key: str = "question", a_key: str = "answer") -> list[dict]:
    """Load a JSON FAQ file and normalize keys to ``question``/``answer``."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("data", data.get("faqs", []))
    return [
        {
            "question": str(item.get(q_key, item.get("question", ""))).strip(),
            "answer": str(item.get(a_key, item.get("answer", ""))).strip(),
        }
        for item in data
    ]


def load_squad_split(
    train_path: str | Path | None = None,
    dev_path: str | Path | None = None,
    faqs_train_path: str | Path | None = None,
    sample_dev: int | None = None,
    auto_download: bool = True,
) -> dict[str, Any]:
    """Return a SQuAD 2.0 train/dev split as question-answer pairs.

    The official train passages are used for FAQ generation / LoRA training, and
    the official dev set provides held-out gold answers.  If ``faqs_train_path``
    is provided, a contamination guard is enforced.  When ``auto_download``
    is True and a split file is missing, it is downloaded from HuggingFace.
    """
    result = {"train": [], "dev": []}
    for split, path in [("train", train_path), ("dev", dev_path)]:
        path = Path(path) if path else None
        if path is None:
            continue
        if not path.exists() and auto_download:
            print(f"Downloading SQuAD 2.0 {split} split from HuggingFace …")
            hf_split = "validation" if split == "dev" else split
            ds = _download_hf_dataset("rajpurkar/squad_v2", None, hf_split)
            _save_squad_json(ds, path)
        if not path.exists():
            continue
        data = json.loads(Path(path).read_text())
        rows = []
        for article in data.get("data", []):
            for paragraph in article.get("paragraphs", []):
                context = paragraph.get("context", "")
                for qa in paragraph.get("qas", []):
                    q = qa.get("question", "").strip()
                    if not q:
                        continue
                    if qa.get("is_impossible", False):
                        continue
                    answers = qa.get("answers", [])
                    if answers:
                        a = answers[0].get("text", "").strip()
                    else:
                        a = ""
                    rows.append({"question": q, "answer": a, "context": context})
        if sample_dev and split == "dev":
            rows = rows[:sample_dev]
        result[split] = rows

    if faqs_train_path:
        train_faqs = load_json_faq(faqs_train_path)
        assert_disjoint(train_faqs, result["dev"])
    return result


def load_pubmedqa_split(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    faqs_train_path: str | Path | None = None,
    sample_test: int | None = None,
    auto_download: bool = True,
) -> dict[str, Any]:
    """Return PubMedQA train/test split as question-answer pairs.

    Uses the official train/test split and the ``final_decision`` / ``long_answer``
    fields as gold answers.  Labels are also returned for yes/no/maybe questions.
    When ``auto_download`` is True and a split file is missing, it is downloaded.
    """
    result = {"train": [], "test": []}
    for split, path in [("train", train_path), ("test", test_path)]:
        path = Path(path) if path else None
        if path is None:
            continue
        if not path.exists() and auto_download:
            print(f"Downloading PubMedQA {split} split from HuggingFace …")
            # The HF "pqa_labeled" config only provides a train split.  We
            # download train and, if the caller asked for test, hold out 15%.
            ds = _download_hf_dataset("qiaojin/PubMedQA", "pqa_labeled", "train")
            if split == "test":
                import random
                rows = list(ds)
                rng = random.Random(42)
                rng.shuffle(rows)
                n_test = max(1, int(len(rows) * 0.15))
                test_rows = rows[:n_test]
                train_rows = rows[n_test:]
                train_path_out = path.parent / "train_set.json"
                _save_pubmedqa_json(train_rows, train_path_out)
                _save_pubmedqa_json(test_rows, path)
            else:
                _save_pubmedqa_json(ds, path)
        if not path.exists():
            continue
        data = json.loads(Path(path).read_text())
        rows = []
        for pid, item in data.items():
            q = item.get("QUESTION", "").strip()
            a = item.get("LONG_ANSWER", "").strip()
            label = item.get("final_decision", "").strip().lower()
            if not q:
                continue
            rows.append({"question": q, "answer": a, "label": label})
        if sample_test and split == "test":
            rows = rows[:sample_test]
        result[split] = rows

    if faqs_train_path:
        train_faqs = load_json_faq(faqs_train_path)
        assert_disjoint(train_faqs, result["test"])
    return result


def load_bitext_split(
    faqs_path: str | Path,
    test_fraction: float = 0.15,
    seed: int = 42,
    auto_download: bool = True,
    max_faqs: int = 200,
) -> dict[str, Any]:
    """Hold out 15% of Bitext customer-support intent-response pairs.

    Args:
        faqs_path: Path to the full FAQ file (or intent-response JSONL).
        test_fraction: Fraction to reserve for evaluation.
        seed: Random seed for reproducible hold-out.
        auto_download: Download the Bitext dataset from HuggingFace if the FAQ
            file is missing.
        max_faqs: Maximum FAQ pairs to keep when auto-downloading.

    Returns:
        Dict with ``train`` and ``test`` FAQ lists.
    """
    import random

    faqs_path = Path(faqs_path)
    if not faqs_path.exists() and auto_download:
        print("Downloading Bitext customer-support dataset from HuggingFace …")
        ds = _download_hf_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None, "train")
        faqs = []
        seen = set()
        for row in ds:
            q = str(row.get("instruction", "")).strip()
            a = str(row.get("response", "")).strip()
            if not q or not a or len(q) < 10 or len(a) < 10 or q.lower() in seen:
                continue
            seen.add(q.lower())
            faqs.append({"question": q, "answer": a})
            if len(faqs) >= max_faqs:
                break
        faqs_path.parent.mkdir(parents=True, exist_ok=True)
        faqs_path.write_text(json.dumps(faqs, indent=2, ensure_ascii=False))
        print(f"  Wrote {faqs_path} ({len(faqs)} pairs)")

    faqs = load_json_faq(faqs_path)
    rng = random.Random(seed)
    rng.shuffle(faqs)
    n_test = int(len(faqs) * test_fraction)
    result = {"train": faqs[n_test:], "test": faqs[:n_test]}
    assert_disjoint(result["train"], result["test"])
    return result


def load_bedrock_hand_verified(
    path: str | Path | None = None,
    faqs_train_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the hand-verified Amazon Bedrock test set.

    If the file does not exist, returns an empty list so callers can fall back
    to a smaller automatic split with a clear warning.
    """
    path = Path(path) if path else Path("examples/usecase4_bedrock_userguide/hand_verified_test.json")
    if path.exists():
        test_set = load_json_faq(path)
    else:
        test_set = []
    if faqs_train_path and test_set:
        train_faqs = load_json_faq(faqs_train_path)
        assert_disjoint(train_faqs, test_set)
    return {"test": test_set}
