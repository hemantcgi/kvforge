"""TDD for tools/dump_rag_preds.py — the RAG-mode 4-arm prediction dumper.

GPU-free: the generator is injected (str->str), so every test runs on CPU.
Guards plan §5.5: top-5 context via the SHARED render_prompt (train/eval parity),
R3 closed-book = question only, 481-fact selection, and the per-row schema.
"""
import importlib

dp = importlib.import_module("tools.dump_rag_preds")
from tools.raft_data import render_prompt, SHORT_ANSWER_INSTRUCTION


# --- fixtures -------------------------------------------------------------
def _items():
    return [
        {"chunk_id": 10, "question": "What is Bedrock?", "answer": "A managed service"},
        {"chunk_id": 20, "question": "Which page lists the models?", "answer": "page 7"},
    ]


def _recall_records():
    return {"records": [
        {"question": "What is Bedrock?", "gold_chunk_id": 10,
         "retrieved_ids": [10, 5, 3, 9, 1], "hit": True},
        {"question": "Which page lists the models?", "gold_chunk_id": 20,
         "retrieved_ids": [5, 3, 9, 1, 2], "hit": False},
    ]}


def _id2text():
    return {i: f"chunk-text-{i}" for i in range(0, 30)}


# --- load_recall_map ------------------------------------------------------
def test_load_recall_map_keys_by_question():
    m = dp.load_recall_map(_recall_records())
    assert set(m) == {"What is Bedrock?", "Which page lists the models?"}
    assert m["What is Bedrock?"]["retrieved_ids"] == [10, 5, 3, 9, 1]
    assert m["What is Bedrock?"]["hit"] is True


# --- select_items: 481 logic (page-trivia drop + retrieval coverage) ------
def test_select_items_drops_page_trivia():
    m = dp.load_recall_map(_recall_records())
    kept = dp.select_items(_items(), m)
    assert [it["chunk_id"] for it in kept] == [10]  # the "page" item is dropped


def test_select_items_raises_when_item_missing_from_recall():
    m = dp.load_recall_map(_recall_records())
    bad = [{"chunk_id": 99, "question": "unseen question?", "answer": "x"}]
    try:
        dp.select_items(bad, m)
        assert False, "expected KeyError for item absent from recall table"
    except KeyError:
        pass


# --- RAG mode: shared render + top-5 context, in retrieval order ----------
def test_rag_row_uses_shared_render_with_top5_context():
    m = dp.load_recall_map(_recall_records())
    it = _items()[0]
    rows = dp.build_rows([it], m, _id2text(), arm="R0", mode="rag",
                         generate=lambda p: "PRED::" + p)
    assert len(rows) == 1
    r = rows[0]
    expected_prompt = render_prompt(
        SHORT_ANSWER_INSTRUCTION,
        [f"chunk-text-{i}" for i in [10, 5, 3, 9, 1]],
        "What is Bedrock?")
    assert r["prompt"] == expected_prompt
    assert r["prediction"] == "PRED::" + expected_prompt
    assert r["retrieved_ids"] == [10, 5, 3, 9, 1]
    assert r["hit"] is True
    assert r["arm"] == "R0"
    assert r["chunk_id"] == 10
    assert r["question"] == "What is Bedrock?"   # natural question, not the packed prompt
    assert r["gold"] == "A managed service"


# --- R3 closed-book: question only, no context ----------------------------
def test_closedbook_row_is_question_only():
    m = dp.load_recall_map(_recall_records())
    it = _items()[0]
    rows = dp.build_rows([it], m, _id2text(), arm="R3", mode="closedbook",
                         generate=lambda p: "CB::" + p)
    r = rows[0]
    assert r["prompt"] == "What is Bedrock?"
    assert r["prediction"] == "CB::What is Bedrock?"
    assert r["retrieved_ids"] == []
    assert r["hit"] is None


# --- answer_in_context flag (distinct from hit) ---------------------------
def test_answer_in_context_true_when_gold_text_in_chunks():
    m = dp.load_recall_map(_recall_records())
    it = {"chunk_id": 10, "question": "What is Bedrock?", "answer": "managed"}
    id2text = dict(_id2text()); id2text[10] = "Bedrock is a fully managed service."
    rows = dp.build_rows([it], m, id2text, arm="R0", mode="rag",
                         generate=lambda p: "x")
    assert rows[0]["answer_in_context"] is True


def test_answer_in_context_false_when_gold_absent():
    m = dp.load_recall_map(_recall_records())
    it = {"chunk_id": 10, "question": "What is Bedrock?", "answer": "zzz-absent-zzz"}
    rows = dp.build_rows([it], m, _id2text(), arm="R0", mode="rag",
                         generate=lambda p: "x")
    assert rows[0]["answer_in_context"] is False


def test_closedbook_answer_in_context_is_false():
    m = dp.load_recall_map(_recall_records())
    rows = dp.build_rows([_items()[0]], m, _id2text(), arm="R3", mode="closedbook",
                         generate=lambda p: "x")
    assert rows[0]["answer_in_context"] is False
