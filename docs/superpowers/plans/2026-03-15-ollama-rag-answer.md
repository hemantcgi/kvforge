# Ollama RAG Answer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pipe `bedrock_rag.py search` output as JSON into `ollama_answer.py`, which uses `qwen3.5:0.8b` via Ollama to stream a cited, confidence-scored answer.

**Architecture:** `bedrock_rag.py` detects when stdout is piped and emits a single JSON object `{query, chunks}` instead of pretty output. `ollama_answer.py` reads that JSON from stdin, computes a rank-weighted retrieval confidence score, builds a strict RAG prompt, and streams the LLM response with the confidence line deferred to after a final separator.

**Tech Stack:** Python 3.13, qdrant-client 1.17, fastembed, ollama 0.6, httpx, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `bedrock_rag.py` | Modify | Add `_run_search()` + `_emit_json()`, update `cmd_search()` with TTY guard |
| `ollama_answer.py` | Create | Parse stdin JSON, compute confidence, build prompt, stream answer |
| `tests/test_bedrock_rag.py` | Create | Unit tests for `_run_search`, `_emit_json`, and `cmd_search` pipe mode |
| `tests/test_ollama_answer.py` | Create | Unit tests for pure functions in `ollama_answer.py` |

---

## Chunk 1: bedrock_rag.py — JSON pipe output

### Task 1: Refactor `query()` — extract `_run_search()`

**Files:**
- Modify: `bedrock_rag.py:166-222`

The search logic and display logic are currently mixed inside `query()`. Extract the Qdrant call into a standalone function so `cmd_search()` can call it directly when piped.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bedrock_rag.py`:

```python
"""Tests for bedrock_rag._run_search and JSON pipe output."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import bedrock_rag


def make_mock_hit(page: int, score: float, text: str):
    hit = MagicMock()
    hit.score = score
    hit.payload = {"page": page, "text": text}
    return hit


def test_run_search_returns_list_of_hits():
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = iter([np.array([0.1] * 384)])

    mock_hit = make_mock_hit(page=42, score=0.85, text="some content")
    mock_response = MagicMock()
    mock_response.points = [mock_hit]

    mock_client = MagicMock()
    mock_client.query_points.return_value = mock_response

    results = bedrock_rag._run_search("test question", mock_embedder, mock_client)

    assert len(results) == 1
    assert results[0].score == 0.85
    assert results[0].payload["page"] == 42
```

- [ ] **Step 2: Run test — verify it fails**

```bash
python3 -m pytest tests/test_bedrock_rag.py::test_run_search_returns_list_of_hits -v
```

Expected: `AttributeError: module 'bedrock_rag' has no attribute '_run_search'`

- [ ] **Step 3: Add `_run_search()` to `bedrock_rag.py`**

Add this function just before the `query()` function (around line 164):

```python
def _run_search(
    question: str,
    embedder: TextEmbedding,
    client: QdrantClient,
) -> list:
    """Embed question and return raw Qdrant scored results."""
    q_vector = next(iter(embedder.embed([question]))).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vector,
        limit=TOP_K,
        with_payload=True,
    )
    return response.points
```

Then update `query()` to call `_run_search()` and remove the now-redundant `sep_line` variable (reuse `sep`):

```python
def query(question: str, embedder: TextEmbedding, client: QdrantClient) -> None:
    """Embed the question, search Qdrant, and print the top-k results."""
    sep = "─" * 70
    log(f"\n{sep}")
    log(f"❓ Query: {question}")
    log(sep)

    results = _run_search(question, embedder, client)

    if not results:
        log("⚠️  No results found.")
        return

    log(f"\n🔍 Top {TOP_K} most relevant passages:\n")
    for rank, hit in enumerate(results, start=1):
        wrapped = textwrap.fill(
            hit.payload["text"], width=80,
            initial_indent="   ", subsequent_indent="   ",
        )
        log(f"  [{rank}] Score: {hit.score:.4f}  |  Page: {hit.payload['page']}")
        log(wrapped)
        log("")

    log(sep)
    log("💡 Answer — key sentences from the most relevant passages:\n")

    combined = " ".join(hit.payload["text"] for hit in results[:3])
    sentences = [
        s.strip() for s in combined.replace("\n", " ").split(". ")
        if len(s.strip()) > 50
    ]

    q_words = {w.lower().strip("?.,!") for w in question.split() if len(w) > 3}
    relevant = [
        s for s in sentences
        if sum(1 for kw in q_words if kw in s.lower()) >= 2
    ]

    if relevant:
        for sent in relevant[:8]:
            log(f"   • {sent.strip()}.")
    else:
        log(textwrap.fill(
            results[0].payload["text"], width=80,
            initial_indent="   ", subsequent_indent="   ",
        ))
```

- [ ] **Step 4: Run test — verify it passes**

```bash
python3 -m pytest tests/test_bedrock_rag.py::test_run_search_returns_list_of_hits -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add bedrock_rag.py tests/test_bedrock_rag.py
git commit -m "refactor: extract _run_search() from query() in bedrock_rag"
```

---

### Task 2: Add `_emit_json()` to `bedrock_rag.py`

**Files:**
- Modify: `bedrock_rag.py`
- Test: `tests/test_bedrock_rag.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bedrock_rag.py`:

```python
def test_emit_json_structure(capsys):
    hits = [
        make_mock_hit(page=10, score=0.90, text="first chunk"),
        make_mock_hit(page=20, score=0.75, text="second chunk"),
    ]
    bedrock_rag._emit_json("my question", hits)
    data = json.loads(capsys.readouterr().out)
    assert data["query"] == "my question"
    assert len(data["chunks"]) == 2
    assert data["chunks"][0] == {"page": 10, "score": 0.9, "text": "first chunk"}
    assert data["chunks"][1] == {"page": 20, "score": 0.75, "text": "second chunk"}


def test_emit_json_scores_rounded_to_4dp(capsys):
    hits = [make_mock_hit(page=1, score=0.123456789, text="text")]
    bedrock_rag._emit_json("q", hits)
    data = json.loads(capsys.readouterr().out)
    assert data["chunks"][0]["score"] == 0.1235


def test_emit_json_empty_results(capsys):
    bedrock_rag._emit_json("q", [])
    data = json.loads(capsys.readouterr().out)
    assert data["chunks"] == []
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python3 -m pytest tests/test_bedrock_rag.py::test_emit_json_structure tests/test_bedrock_rag.py::test_emit_json_scores_rounded_to_4dp -v
```

Expected: `AttributeError: module 'bedrock_rag' has no attribute '_emit_json'`

- [ ] **Step 3: Add `_emit_json()` to `bedrock_rag.py`**

Add `import json` to the imports block at the top of `bedrock_rag.py`. Then add this function after `_run_search()`:

```python
def _emit_json(question: str, results: list) -> None:
    """Emit search results as a single JSON object to stdout (pipe mode)."""
    payload = {
        "query": question,
        "chunks": [
            {
                "page": hit.payload["page"],
                "score": round(hit.score, 4),
                "text": hit.payload["text"],
            }
            for hit in results
        ],
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python3 -m pytest tests/test_bedrock_rag.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add bedrock_rag.py tests/test_bedrock_rag.py
git commit -m "feat: add _emit_json() to bedrock_rag for pipe mode output"
```

---

### Task 3: Update `cmd_search()` with TTY guard

**Files:**
- Modify: `bedrock_rag.py:253-264`
- Test: `tests/test_bedrock_rag.py`

When `sys.stdout.isatty()` is `False` (piped), informational messages go to stderr and only JSON goes to stdout.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bedrock_rag.py`:

```python
def test_cmd_search_emits_json_when_piped(capsys, monkeypatch):
    """cmd_search() must emit JSON on stdout when stdout is not a TTY."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_response = MagicMock()
    mock_response.points = [make_mock_hit(page=1, score=0.9, text="bedrock info")]
    mock_client.query_points.return_value = mock_response

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = iter([np.array([0.1] * 384)])

    with patch("bedrock_rag.TextEmbedding", return_value=mock_embedder), \
         patch("bedrock_rag.QdrantClient", return_value=mock_client):
        bedrock_rag.cmd_search("what is bedrock?")

    data = json.loads(capsys.readouterr().out)
    assert data["query"] == "what is bedrock?"
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["page"] == 1
    assert data["chunks"][0]["score"] == 0.9
    assert data["chunks"][0]["text"] == "bedrock info"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
python3 -m pytest tests/test_bedrock_rag.py::test_cmd_search_emits_json_when_piped -v
```

Expected: `json.JSONDecodeError` — stdout contains pretty-printed text, not JSON.

- [ ] **Step 3: Update `cmd_search()` in `bedrock_rag.py`**

Replace the existing `cmd_search()` function:

```python
def cmd_search(question: str) -> None:
    piped = not sys.stdout.isatty()

    def info(msg: str) -> None:
        """Print informational message: stderr when piped, stdout otherwise."""
        if piped:
            print(msg, file=sys.stderr, flush=True)
        else:
            log(msg)

    info(f"🤖 Loading embedding model '{EMBED_MODEL}' …")
    embedder = TextEmbedding(model_name=EMBED_MODEL, show_download_progress=False)

    info(f"🔗 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(COLLECTION_NAME):
        log(f"❌ Collection '{COLLECTION_NAME}' not found. Run 'index' first.")
        sys.exit(1)

    if piped:
        results = _run_search(question, embedder, client)
        _emit_json(question, results)
    else:
        query(question, embedder, client)
```

- [ ] **Step 4: Run test — verify it passes**

```bash
python3 -m pytest tests/test_bedrock_rag.py::test_cmd_search_emits_json_when_piped -v
```

Expected: `PASSED`

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/test_bedrock_rag.py -v
```

Expected: `5 passed`

- [ ] **Step 6: Verify interactive mode still works**

```bash
python3 bedrock_rag.py search "what is bedrock?"
```

Expected: pretty output with scores, passages, extractive answer (unchanged).

- [ ] **Step 7: Commit**

```bash
git add bedrock_rag.py tests/test_bedrock_rag.py
git commit -m "feat: TTY guard in cmd_search — emit JSON when piped, pretty output in terminal"
```

---

## Chunk 2: ollama_answer.py — LLM answer with streaming

### Task 4: `_compute_retrieval_confidence()` with tests

**Files:**
- Create: `ollama_answer.py`
- Create: `tests/test_ollama_answer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ollama_answer.py`:

```python
"""Tests for ollama_answer pure functions."""
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import ollama_answer


def test_confidence_single_chunk():
    chunks = [{"page": 1, "score": 0.8, "text": "x"}]
    assert ollama_answer._compute_retrieval_confidence(chunks) == pytest.approx(0.8)


def test_confidence_two_chunks_weighted():
    # N=2: weight_1 = 2/3, weight_2 = 1/3
    chunks = [
        {"page": 1, "score": 1.0, "text": "a"},
        {"page": 2, "score": 0.0, "text": "b"},
    ]
    result = ollama_answer._compute_retrieval_confidence(chunks)
    assert result == pytest.approx(2 / 3)


def test_confidence_five_chunks_known_value():
    # weights: 5/15, 4/15, 3/15, 2/15, 1/15
    chunks = [
        {"page": 1, "score": 0.7685, "text": ""},
        {"page": 2, "score": 0.7599, "text": ""},
        {"page": 3, "score": 0.7572, "text": ""},
        {"page": 4, "score": 0.7522, "text": ""},
        {"page": 5, "score": 0.7495, "text": ""},
    ]
    result = ollama_answer._compute_retrieval_confidence(chunks)
    assert result == pytest.approx(0.7605, abs=1e-4)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python3 -m pytest tests/test_ollama_answer.py -v
```

Expected: `ModuleNotFoundError: No module named 'ollama_answer'`

- [ ] **Step 3: Create `ollama_answer.py` with `_compute_retrieval_confidence()`**

```python
"""
Ollama RAG answer: read search results from bedrock_rag.py via pipe,
generate a streamed, cited, confidence-scored answer using qwen3.5:0.8b.

Usage:
  python3 bedrock_rag.py search "your question" | python3 ollama_answer.py
"""

import json
import re
import sys

import httpx
import ollama

MODEL = "qwen3.5:0.8b"
SEP = "─" * 62

SYSTEM_PROMPT = """\
You are a precise assistant. Answer ONLY using the provided context chunks.
If the answer is not found in the chunks, say exactly:
"I don't know based on the provided context."
Do not use outside knowledge.

Each chunk has a relevance score (0–1). Higher scores mean the chunk is more \
likely to contain the correct answer — weight your answer accordingly and draw \
primarily from high-scoring chunks.

Always cite sources inline after each sentence using page number and any section \
heading or URL visible in the chunk text.
Citation format: [page P] or [page P, "Section Name"] or [page P, <url>]

At the end of your answer, on a new line, output exactly:
Confidence: <0–100>%  — <one sentence explaining the score>

Example:
Confidence: 72% — Chunks cover the topic partially but lack a direct comparison.\
"""


def _compute_retrieval_confidence(chunks: list[dict]) -> float:
    """Rank-weighted average of chunk scores. Rank 1 gets highest weight."""
    n = len(chunks)
    denom = n * (n + 1) / 2
    return sum(
        (n - rank) / denom * chunk["score"]
        for rank, chunk in enumerate(chunks)
    )
```

Note: `enumerate(chunks)` is 0-based, so rank runs 0..N-1. Weight expression
`(n - rank)` produces `N, N-1, ..., 1` — summing to `N(N+1)/2 = denom`. ✓

- [ ] **Step 4: Run tests — verify they pass**

```bash
python3 -m pytest tests/test_ollama_answer.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add ollama_answer.py tests/test_ollama_answer.py
git commit -m "feat: add ollama_answer.py skeleton with _compute_retrieval_confidence"
```

---

### Task 5: `_build_messages()` with tests

**Files:**
- Modify: `ollama_answer.py`
- Modify: `tests/test_ollama_answer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ollama_answer.py`:

```python
def test_build_messages_contains_query():
    chunks = [{"page": 5, "score": 0.9, "text": "Bedrock is a managed service."}]
    messages = ollama_answer._build_messages("what is bedrock?", chunks)
    user_content = messages[1]["content"]
    assert "what is bedrock?" in user_content


def test_build_messages_includes_score_and_page():
    chunks = [{"page": 5, "score": 0.9, "text": "Bedrock is a managed service."}]
    messages = ollama_answer._build_messages("q", chunks)
    user_content = messages[1]["content"]
    assert "score: 0.9" in user_content
    assert "page 5" in user_content


def test_build_messages_system_role():
    chunks = [{"page": 1, "score": 0.5, "text": "x"}]
    messages = ollama_answer._build_messages("q", chunks)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_messages_all_chunks_present():
    chunks = [
        {"page": 1, "score": 0.9, "text": "alpha"},
        {"page": 2, "score": 0.8, "text": "beta"},
    ]
    messages = ollama_answer._build_messages("q", chunks)
    user_content = messages[1]["content"]
    assert "alpha" in user_content
    assert "beta" in user_content
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python3 -m pytest tests/test_ollama_answer.py::test_build_messages_contains_query -v
```

Expected: `AttributeError: module 'ollama_answer' has no attribute '_build_messages'`

- [ ] **Step 3: Add `_build_messages()` to `ollama_answer.py`**

```python
def _build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """Build the system + user messages for the Ollama chat call."""
    context_lines = []
    for chunk in chunks:
        context_lines.append(f"[score: {chunk['score']}, page {chunk['page']}]")
        context_lines.append(chunk["text"])
        context_lines.append("")

    user_content = "Context:\n\n" + "\n".join(context_lines) + f"\nQuestion: {query}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python3 -m pytest tests/test_ollama_answer.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ollama_answer.py tests/test_ollama_answer.py
git commit -m "feat: add _build_messages() to ollama_answer"
```

---

### Task 6: `_parse_stdin()` with error handling tests

**Files:**
- Modify: `ollama_answer.py`
- Modify: `tests/test_ollama_answer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ollama_answer.py`:

```python
def test_parse_stdin_valid():
    data = {"query": "test q", "chunks": [{"page": 1, "score": 0.9, "text": "hi"}]}
    with patch("sys.stdin", io.StringIO(json.dumps(data))):
        with patch("sys.stdin.isatty", return_value=False):
            query, chunks = ollama_answer._parse_stdin()
    assert query == "test q"
    assert chunks[0]["text"] == "hi"


def test_parse_stdin_exits_if_tty(capsys):
    with patch("sys.stdin.isatty", return_value=True):
        with pytest.raises(SystemExit) as exc:
            ollama_answer._parse_stdin()
    assert exc.value.code == 1
    assert "No input" in capsys.readouterr().out


def test_parse_stdin_exits_on_bad_json(capsys):
    with patch("sys.stdin", io.StringIO("not json")):
        with patch("sys.stdin.isatty", return_value=False):
            with pytest.raises(SystemExit) as exc:
                ollama_answer._parse_stdin()
    assert exc.value.code == 1


def test_parse_stdin_exits_on_empty_chunks(capsys):
    data = {"query": "q", "chunks": []}
    with patch("sys.stdin", io.StringIO(json.dumps(data))):
        with patch("sys.stdin.isatty", return_value=False):
            with pytest.raises(SystemExit) as exc:
                ollama_answer._parse_stdin()
    assert exc.value.code == 1
    assert "No chunks" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python3 -m pytest tests/test_ollama_answer.py::test_parse_stdin_valid -v
```

Expected: `AttributeError: module 'ollama_answer' has no attribute '_parse_stdin'`

- [ ] **Step 3: Add `_parse_stdin()` to `ollama_answer.py`**

```python
def _parse_stdin() -> tuple[str, list[dict]]:
    """Read and validate the JSON payload from stdin."""
    if sys.stdin.isatty():
        print(
            '❌ No input. Run: python3 bedrock_rag.py search "query" | python3 ollama_answer.py'
        )
        sys.exit(1)

    try:
        data = json.load(sys.stdin)
        query = data["query"]
        chunks = data["chunks"]
    except (json.JSONDecodeError, KeyError):
        print('❌ Invalid input from pipe — expected JSON with "query" and "chunks"')
        sys.exit(1)

    if not chunks:
        print("❌ No chunks returned — nothing to answer from")
        sys.exit(1)

    return query, chunks
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python3 -m pytest tests/test_ollama_answer.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add ollama_answer.py tests/test_ollama_answer.py
git commit -m "feat: add _parse_stdin() with full error handling to ollama_answer"
```

---

### Task 7: `_stream_answer()` — streaming with confidence line deferral

**Files:**
- Modify: `ollama_answer.py`

This function has side effects (Ollama network call + stdout streaming), so we test it via a mock in Task 8. Here we implement and smoke-test it.

- [ ] **Step 1: Add `_stream_answer()` to `ollama_answer.py`**

```python
def _stream_answer(messages: list[dict], retrieval_confidence: float) -> None:
    """
    Call Ollama with streaming. Print answer lines immediately.
    Defer the 'Confidence: N%' line to after a trailing separator.
    Uses line-level buffering: accumulate chars until newline, then
    check with re.match before deciding to print or defer.
    """
    confidence_line: str | None = None
    line_buf: list[str] = []

    try:
        stream = ollama.chat(model=MODEL, messages=messages, stream=True)
    except httpx.ConnectError:
        print("❌ Ollama not reachable at localhost:11434. Is ollama running?")
        sys.exit(1)
    except ollama.ResponseError as e:
        if e.status_code == 404:
            print(f"❌ Model '{MODEL}' not found. Run: ollama pull {MODEL}")
        else:
            print(f"❌ Ollama error: {e}")
        sys.exit(1)

    for chunk in stream:
        token = chunk["message"]["content"]
        for char in token:
            line_buf.append(char)
            if char == "\n":
                completed = "".join(line_buf).rstrip("\n")
                line_buf = []
                if re.match(r"^Confidence: \d+%", completed):
                    confidence_line = completed
                else:
                    print(completed, flush=True)

    # flush any remaining buffer (model may omit trailing newline)
    if line_buf:
        remaining = "".join(line_buf).rstrip("\n")
        if remaining and re.match(r"^Confidence: \d+%", remaining):
            confidence_line = remaining
        elif remaining:
            print(remaining, flush=True)

    if confidence_line:
        print(f"\n{SEP}")
        print(confidence_line)
```

- [ ] **Step 2: Verify the file parses without errors**

```bash
python3 -c "import ollama_answer; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ollama_answer.py
git commit -m "feat: add _stream_answer() with line-buffered confidence deferral"
```

---

### Task 8: `main()` and end-to-end test

**Files:**
- Modify: `ollama_answer.py`
- Modify: `tests/test_ollama_answer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ollama_answer.py`:

```python
def test_main_prints_retrieval_confidence(capsys):
    """main() must print the retrieval confidence header before streaming."""
    data = {
        "query": "what is bedrock?",
        "chunks": [
            {"page": 1, "score": 0.85, "text": "Bedrock is a managed service."},
            {"page": 2, "score": 0.70, "text": "It provides API access to FMs."},
        ],
    }

    fake_stream = [{"message": {"content": "Bedrock is managed.\n"}}]

    with patch("sys.stdin", io.StringIO(json.dumps(data))):
        with patch("sys.stdin.isatty", return_value=False):
            with patch("ollama.chat", return_value=iter(fake_stream)):
                ollama_answer.main()

    out = capsys.readouterr().out
    assert "Retrieval confidence:" in out
    assert "Bedrock is managed." in out
```

- [ ] **Step 2: Run test — verify it fails**

```bash
python3 -m pytest tests/test_ollama_answer.py::test_main_prints_retrieval_confidence -v
```

Expected: `AttributeError: module 'ollama_answer' has no attribute 'main'`

- [ ] **Step 3: Add `main()` to `ollama_answer.py`**

```python
def main() -> None:
    query, chunks = _parse_stdin()

    conf = _compute_retrieval_confidence(chunks)

    if conf < 0.5:
        print("⚠️  Low confidence — answer may be unreliable")

    print(f"📊 Retrieval confidence: {conf:.2f} (top-{len(chunks)} chunks, weighted by rank)")
    print(SEP)

    messages = _build_messages(query, chunks)
    _stream_answer(messages, conf)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests — verify they pass**

```bash
python3 -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Manual end-to-end test**

Run the full pipeline:

```bash
python3 bedrock_rag.py search "what is the difference between sagemaker and bedrock?" \
  | python3 ollama_answer.py
```

Expected output layout:
```
📊 Retrieval confidence: 0.76 (top-5 chunks, weighted by rank)
──────────────────────────────────────────────────────────────
<streamed answer with [page N] citations>

──────────────────────────────────────────────────────────────
Confidence: NN% — <one sentence>
```

- [ ] **Step 6: Verify error cases manually**

```bash
# No pipe — should exit with usage message
python3 ollama_answer.py

# Bad JSON — should exit with error
echo "not json" | python3 ollama_answer.py
```

- [ ] **Step 7: Final commit**

```bash
git add ollama_answer.py tests/test_ollama_answer.py
git commit -m "feat: complete ollama_answer.py with main(), end-to-end pipeline working"
```
