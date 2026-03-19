# Design: Ollama RAG Answer — `ollama_answer.py`

**Date:** 2026-03-15
**Status:** Approved

---

## Overview

Add a second script `ollama_answer.py` that receives vector search results from
`bedrock_rag.py` via a Unix pipe and uses a local Ollama model (`qwen3.5:0.8b`)
to generate a streamed, cited, confidence-scored answer to the original query.

---

## Usage

```bash
python3 bedrock_rag.py search "what is the difference between sagemaker and bedrock?" \
  | python3 ollama_answer.py
```

The query is typed once. `bedrock_rag.py` embeds it in the JSON it pipes.
`ollama_answer.py` takes no CLI arguments — query and chunks both arrive on stdin.

---

## Changes to `bedrock_rag.py`

### TTY detection in `cmd_search()`

The TTY guard must wrap the **entire output path of `cmd_search()`**, not just
`query()`. The two `log()` calls before `query()` (embedding model load message,
Qdrant connection message) also write to stdout and must be suppressed when piped,
otherwise `ollama_answer.py`'s `json.load(sys.stdin)` will fail.

When stdout is connected to a terminal (`sys.stdout.isatty() == True`), behaviour
is unchanged: pretty output with progress bars, scores table, and extractive answer.

When stdout is **not** a TTY (being piped), `cmd_search()` suppresses all `log()`
calls and emits a single JSON object as the only stdout output:

```json
{
  "query": "what is the difference between sagemaker and bedrock?",
  "chunks": [
    {"page": 2018, "score": 0.7685, "text": "..."},
    {"page": 2020, "score": 0.7599, "text": "..."},
    {"page": 312,  "score": 0.7572, "text": "..."},
    {"page": 314,  "score": 0.7522, "text": "..."},
    {"page": 318,  "score": 0.7495, "text": "..."}
  ]
}
```

Chunks are ordered highest-score-first (guaranteed by Qdrant `query_points`).
The `rank` field is omitted — consumers derive rank from array index.

---

## `ollama_answer.py` — New File

### Input

Reads one JSON object from stdin (format above). No CLI arguments.

### Processing Pipeline

```
stdin (JSON)
  │
  ├─ guard: if sys.stdin.isatty() → print usage, exit 1
  │
  ├─ json.load(sys.stdin)
  │    → query: str
  │    → chunks: list[{page, score, text}]
  │
  ├─ compute retrieval_confidence (rank-weighted average of chunk scores)
  │
  ├─ print retrieval confidence header
  │
  ├─ build prompt (system + user)
  │
  ├─ ollama.chat(model, messages, stream=True)
  │    → line-level buffered streaming:
  │       accumulate tokens until newline
  │       if re.match(r'^Confidence: \d+%', line) → store full line, don't print
  │       otherwise → flush line to stdout immediately
  │
  └─ print separator + stored confidence line
```

### Retrieval Confidence

Computed before the LLM call using a rank-weighted average of chunk scores.
Rank 1 (highest Qdrant score) receives the most weight.

```
N = number of chunks
weight_i = (N - rank_i + 1) / (N*(N+1)/2)   # weights sum to 1.0
retrieval_confidence = Σ weight_i * score_i
```

For the example scores above (descending order): **≈ 0.76**

Displayed before the answer:
```
📊 Retrieval confidence: 0.76 (top-5 chunks, weighted by rank)
──────────────────────────────────────────────────────────────
```

If `retrieval_confidence < 0.5`, prepend a warning:
```
⚠️  Low confidence — answer may be unreliable
```

### Prompt Design

**System prompt:**
```
You are a precise assistant. Answer ONLY using the provided context chunks.
If the answer is not found in the chunks, say exactly:
"I don't know based on the provided context."
Do not use outside knowledge.

Each chunk has a relevance score (0–1). Higher scores mean the chunk is more
likely to contain the correct answer — weight your answer accordingly and draw
primarily from high-scoring chunks.

Always cite sources inline after each sentence using page number and any section
heading or URL visible in the chunk text.
Citation format: [page P] or [page P, "Section Name"] or [page P, <url>]

At the end of your answer, on a new line, output exactly:
Confidence: <0–100>%  — <one sentence explaining the score>

Example:
Confidence: 72% — Chunks cover the topic partially but lack a direct comparison.
```

**User message:**
```
Context:

[score: 0.7685, page 2018]
<chunk text>

[score: 0.7599, page 2020]
<chunk text>

...

Question: what is the difference between sagemaker and bedrock?
```

### Streaming with Confidence Line Deferral

A naive token-by-token stream would print the confidence line in the wrong
position. The solution is **line-level buffering**:

1. Accumulate streamed tokens in a line buffer until a `\n` is received.
2. Check the completed line using `re.match(r'^Confidence: \d+%', line)` (prefix match).
3. If it matches: store the **entire line** (including the em-dash clause) in a variable, do not print it.
4. Otherwise: flush the completed line to stdout immediately.
5. After the stream ends: print the separator, then the stored confidence line.

This preserves low-latency output for the answer body while ensuring the
confidence line always appears after the separator.

### Output Layout

```
📊 Retrieval confidence: 0.76 (top-5 chunks, weighted by rank)
──────────────────────────────────────────────────────────────
<streamed LLM answer with inline citations>

──────────────────────────────────────────────────────────────
Confidence: 72% — Chunks cover the topic partially but lack a direct comparison.
```

### Error Handling

| Condition | Behaviour |
|---|---|
| stdin is a TTY (no pipe) | `❌ No input. Run: python3 bedrock_rag.py search "query" \| python3 ollama_answer.py` → exit 1 |
| Invalid or empty JSON | `❌ Invalid input from pipe — expected JSON with "query" and "chunks"` → exit 1 |
| Empty chunks list | `❌ No chunks returned — nothing to answer from` → exit 1 |
| `retrieval_confidence < 0.5` | Print `⚠️  Low confidence — answer may be unreliable` before streaming |
| Ollama not reachable | Catch `httpx.ConnectError` → `❌ Ollama not reachable at localhost:11434. Is ollama running?` → exit 1 |
| Model not pulled | Catch `ollama.ResponseError` (HTTP 404) → `❌ Model 'qwen3.5:0.8b' not found. Run: ollama pull qwen3.5:0.8b` → exit 1 |
| LLM omits Confidence line | Skip separator and confidence block silently |

---

## Files

| File | Change |
|---|---|
| `bedrock_rag.py` | TTY guard wrapping all of `cmd_search()` output; emit JSON when piped |
| `ollama_answer.py` | New file, ~90 lines |

---

## Dependencies

| Package | Already installed | Purpose |
|---|---|---|
| `ollama` | No — `pip install ollama` | Ollama Python client (streaming chat) |
| `httpx` | No — pulled in transitively by `ollama` | Used directly to catch `httpx.ConnectError` |

---

## Non-Goals

- No conversation history / multi-turn chat
- No fallback to other Ollama models
- No caching of LLM responses
- No modification to the indexing pipeline
