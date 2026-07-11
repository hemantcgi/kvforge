"""Sleep-time FAQ generation for KVForge.

Uses a cloud LLM (Gemini, Claude, or OpenAI) to reason over indexed chunks
offline and generate high-quality FAQ pairs with pre-drawn inferences.
Generated FAQs are saved to faqs.json and embedded questions are stored in
version.json["known_good_queries"] to pre-seed the Phase 3 parametric shortcut.

Note: prs_evaluator.py also writes known_good_queries (with evaluated embeddings
of high-accuracy queries). That later overwrite is intentional — PRS-evaluated
embeddings are higher quality. This module's pre-seeding is useful before the
first PRS eval run.

Usage::

    python -m pipeline.sleep_faq_generator \\
        --config examples/usecase1_customer_support/config.json \\
        --output examples/usecase1_customer_support/faqs.json \\
        --count 50

Provider/model are read from uc_config.json (llm.sleep_faq_provider,
llm.sleep_faq_model) with env var fallbacks SLEEP_FAQ_PROVIDER / SLEEP_FAQ_MODEL.
API keys are read from env vars: GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


SLEEP_PROMPT_TEMPLATE = """You are pre-reasoning over a document passage to help a retrieval-augmented AI system answer future questions faster and more accurately.

Passage:
{chunk}

Generate {n} question-answer pairs that a user might ask about this passage.
For each pair, include one key inference or conclusion that can be pre-drawn from the passage — this pre-loaded reasoning lets the system answer without re-reading the document.

Use exactly this format for each pair (end each block with ---):
Q: <question>
A: <answer>
INFERENCE: <key pre-drawn inference relevant to answering this question>
---
"""


def _build_sleep_prompt(chunk: str, n_per_chunk: int = 3) -> str:
    """Build the sleep-time reasoning prompt for a chunk."""
    return SLEEP_PROMPT_TEMPLATE.format(chunk=chunk[:1500], n=n_per_chunk)


def _parse_sleep_blocks(text: str) -> list[dict]:
    """Parse one or more Q/A/INFERENCE blocks from LLM output.

    Each block is separated by '---'. Returns list of dicts with keys
    'question', 'answer', 'inference'. Blocks missing INFERENCE are
    included with inference=''.
    """
    results = []
    blocks = re.split(r"\n---+\n?", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q_m = re.search(r"(?:Q|Question):\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
        a_m = re.search(r"(?:A|Answer):\s*(.+?)(?=\nINFERENCE:|\n---|\Z)",
                        block, re.IGNORECASE | re.DOTALL)
        i_m = re.search(r"INFERENCE:\s*(.+?)(?:\n---|\Z)", block, re.IGNORECASE | re.DOTALL)
        if not q_m or not a_m:
            continue
        q = q_m.group(1).strip()
        a = a_m.group(1).strip()
        inference = i_m.group(1).strip() if i_m else ""
        if q and a:
            results.append({"question": q, "answer": a, "inference": inference})
    return results


def _deduplicate(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge new FAQs into existing, skipping duplicate questions."""
    seen = {item.get("question", "").strip().lower() for item in existing}
    merged = list(existing)
    for item in new:
        key = item.get("question", "").strip().lower()
        if key and key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _call_provider(provider: str, model: str, api_key: str, prompt: str) -> str:
    """Call the specified cloud LLM provider and return the raw text response."""
    if provider == "gemini":
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        resp = httpx.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1024}},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    elif provider == "claude":
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": model, "max_tokens": 1024, "temperature": 0.4,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(c.get("text", "") for c in data.get("content", [])
                       if c.get("type") == "text")

    elif provider == "openai":
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model, "max_tokens": 1024, "temperature": 0.4,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use gemini, claude, or openai.")


def _load_uc_config(config_path: Path) -> dict:
    """Load uc_config.json from the same directory as config.json."""
    uc_path = config_path.parent / "uc_config.json"
    if uc_path.exists():
        return json.loads(uc_path.read_text())
    return {}


def _resolve_provider_config(uc_config: dict) -> tuple[str, str, str]:
    """Return (provider, model, api_key) from uc_config + env vars + Studio settings."""
    llm = uc_config.get("llm", {})
    provider = (llm.get("sleep_faq_provider")
                or os.environ.get("SLEEP_FAQ_PROVIDER", "gemini"))
    model = (llm.get("sleep_faq_model")
             or os.environ.get("SLEEP_FAQ_MODEL", "gemini-2.5-flash"))
    key_env = {
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider, "GEMINI_API_KEY")
    # Priority: env var → Studio settings DB
    api_key = os.environ.get(key_env, "")
    if not api_key:
        try:
            from studio.settings_manager import get_setting
            setting_key = {
                "gemini": "gemini_api_key",
                "claude": "anthropic_api_key",
                "openai": "openai_api_key",
            }.get(provider, "gemini_api_key")
            api_key = get_setting(setting_key) or ""
        except Exception:
            pass
    return provider, model, api_key


def generate(cfg: dict, config_path: Path, count: int, output_path: Path,
             n_per_chunk: int = 3) -> None:
    """Run sleep-time FAQ generation and save to output_path."""
    uc_config = _load_uc_config(config_path)
    provider, model, api_key = _resolve_provider_config(uc_config)

    if not api_key:
        key_env = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY",
                   "openai": "OPENAI_API_KEY"}.get(provider, "GEMINI_API_KEY")
        print(
            f"ERROR: No API key found for provider '{provider}'.\n"
            f"  • Set the {key_env} environment variable, or\n"
            f"  • Add the key in Studio → Settings → API Keys ({provider} field).",
            flush=True,
        )
        sys.exit(1)

    print(f"[sleep-faq] provider={provider} model={model} target={count} FAQs")

    from vectorstore.registry import get_store
    store = get_store(cfg)
    results, _ = store.scroll(cfg["collection"], limit=500,
                              with_payload=True, with_vectors=False)
    chunks = [r.payload.get("text", "") for r in results if r.payload.get("text")]
    if not chunks:
        print("[sleep-faq] ERROR: No chunks found. Run indexing first.")
        sys.exit(1)

    print(f"[sleep-faq] {len(chunks)} chunks in '{cfg['collection']}'")

    existing: list[dict] = []
    q_key = cfg.get("faq_question_key", "question")
    a_key = cfg.get("faq_answer_key", "answer")
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            print(f"[sleep-faq] Merging with {len(existing)} existing FAQs")
        except Exception:
            pass

    new_faqs: list[dict] = []
    chunk_idx = 0

    while len(new_faqs) < count and chunk_idx < len(chunks):
        chunk = chunks[chunk_idx]
        chunk_idx += 1
        if not chunk.strip():
            continue
        try:
            prompt = _build_sleep_prompt(chunk, n_per_chunk=n_per_chunk)
            raw = _call_provider(provider, model, api_key, prompt)
            for b in _parse_sleep_blocks(raw):
                if len(new_faqs) >= count:
                    break
                new_faqs.append({q_key: b["question"], a_key: b["answer"]})
                print(f"  [{len(new_faqs)}/{count}] Q: {b['question'][:80]}")
        except Exception as e:
            print(f"  [chunk {chunk_idx}] error: {e}")

    merged = _deduplicate(existing, new_faqs)
    output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"\n[sleep-faq] Saved {len(merged)} FAQs ({len(new_faqs)} new, {len(existing)} existing)")

    # Pre-seed known_good_queries with embedded FAQ questions
    questions = [item[q_key] for item in merged if item.get(q_key)]
    if questions:
        print(f"[sleep-faq] Embedding {len(questions)} questions -> known_good_queries...")
        try:
            from fastembed import TextEmbedding
            embedder = TextEmbedding(
                model_name=cfg.get("embed_model", "BAAI/bge-small-en-v1.5"),
                show_download_progress=False,
            )
            embs = [e.astype(float).tolist() for e in embedder.embed(questions)]
            import core.version as ver
            ver.init(cfg)
            data = ver.load()
            data["known_good_queries"] = embs
            ver.save(data)
            print(f"[sleep-faq] {len(embs)} embeddings written to version.json")
        except Exception as e:
            print(f"[sleep-faq] WARNING: Could not seed known_good_queries: {e}")


def tag_faq_with_chunk_ids(faq: dict, chunk_ids: list[str]) -> dict:
    """Return a copy of faq with source_chunk_ids added."""
    return {**faq, "source_chunk_ids": chunk_ids}


def build_faq_prompt(chunk: dict) -> str:
    """Build the FAQ generation prompt with temporal grounding."""
    from datetime import datetime
    text = chunk.get("text", "")
    effective_from = chunk.get("metadata", {}).get("effective_from", "")
    date_str = ""
    if effective_from:
        try:
            dt = datetime.fromisoformat(effective_from)
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = effective_from[:10]

    context_header = f"Context (as of {date_str}):\n" if date_str else "Context:\n"
    return (
        f"{context_header}"
        f"{text}\n\n"
        f"Generate one FAQ question and answer based on the above context."
        + (f" The answer should reflect information current as of {date_str}." if date_str else "")
    )


def is_faq_stale(faq: dict, superseded_chunk_ids: set[str]) -> bool:
    """Return True if all source chunks for this FAQ have been superseded."""
    source_ids = faq.get("source_chunk_ids", [])
    if not source_ids:
        return False
    return all(cid in superseded_chunk_ids for cid in source_ids)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sleep-time FAQ generation using a cloud LLM."
    )
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--count", type=int, default=50,
                   help="Target FAQ pairs to generate (default: 50)")
    p.add_argument("--output", default=None,
                   help="Output faqs.json path (default: same dir as config)")
    p.add_argument("--n-per-chunk", type=int, default=3,
                   help="Q&A pairs to request per chunk (default: 3)")
    args = p.parse_args()

    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())
    output_path = Path(args.output) if args.output else config_path.parent / "faqs.json"
    generate(cfg, config_path, args.count, output_path, args.n_per_chunk)


if __name__ == "__main__":
    main()
