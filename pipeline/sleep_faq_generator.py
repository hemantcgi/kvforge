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


# Paraphrase augmentation: a controlled GPU experiment (before_after_eval on
# usecase4_bedrock_userguide, see docs/superpowers specs) found that training
# on diverse paraphrases of each question — not repeated/duplicated copies —
# is what lets the LoRA generalize to unseen phrasings of a trained fact
# (held-out judge accuracy 0.06 -> 0.40 with 10 diverse paraphrases per
# question; 10 identical duplicates gave no improvement at all). This
# generator produces the diverse paraphrases; lora_trainer.py's chat-SFT
# path (sft_format="chat") does the rest.
PARAPHRASE_PROMPT_TEMPLATE = """Rewrite this question in {n} different ways. Keep the exact same meaning but vary the wording, phrasing, and sentence structure — the way different users would naturally ask about the same thing.

Question: {question}

Output ONLY the {n} rewritten questions, one per line, no numbering or bullets."""


def _build_paraphrase_prompt(question: str, n: int = 5) -> str:
    """Build the paraphrase-generation prompt for a single question."""
    return PARAPHRASE_PROMPT_TEMPLATE.format(question=question, n=n)


def _parse_paraphrase_lines(text: str) -> list[str]:
    """Parse one paraphrase per line from LLM output.

    Strips leading numbering/bullets (e.g. "1. ", "- ") and drops lines that
    don't end in "?" — this rejects preamble the model prepends despite being
    told "output ONLY the questions" (e.g. "Here are 5 ways to ask that:"),
    which would otherwise be admitted as paraphrase #1 and displace a real one.
    """
    results = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[\d\-\.\)\*]+\s*", "", line).strip().strip('"')
        if line.endswith("?"):
            results.append(line)
    return results


def _augment_with_paraphrases(faqs: list[dict], provider: str, model: str, api_key: str,
                               n_per_faq: int, q_key: str = "question",
                               a_key: str = "answer") -> list[dict]:
    """Generate diverse paraphrases of each FAQ's question, pairing each with
    the original answer. Returns only the new paraphrase FAQs (does not
    include the originals) — callers merge with `_deduplicate`.
    """
    import time as _time

    augmented: list[dict] = []
    # Same inter-request backoff as the main generation loop (2s -> 60s on
    # repeated 429s) — this loop makes one call per FAQ with no other pacing.
    inter_delay = 2.0
    idx = 0
    while idx < len(faqs):
        faq = faqs[idx]
        question = faq.get(q_key, "")
        answer = faq.get(a_key, "")
        if not question or not answer:
            idx += 1
            continue
        try:
            prompt = _build_paraphrase_prompt(question, n_per_faq)
            raw = _call_provider(provider, model, api_key, prompt, base_url=base_url)
            inter_delay = max(2.0, inter_delay * 0.8)
        except Exception as e:
            if "429" in str(e):
                inter_delay = min(60.0, inter_delay * 2)
                print(f"  [paraphrase] rate-limited (429) — waiting {inter_delay:.0f}s before retry")
                _time.sleep(inter_delay)
                continue  # retry this faq, idx unchanged
            print(f"  [paraphrase] warning: could not paraphrase {question[:60]!r}: {e}")
            idx += 1
            continue
        for p in _parse_paraphrase_lines(raw)[:n_per_faq]:
            augmented.append({q_key: p, a_key: answer})
        idx += 1
        _time.sleep(inter_delay)
    return augmented


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


def _call_provider(provider: str, model: str, api_key: str, prompt: str,
                   base_url: str = "") -> str:
    """Call the specified cloud LLM provider and return the raw text response.

    Args:
        base_url: Custom base URL for OpenAI-compatible endpoints
            (Fireworks, vLLM, Together, etc.).  Leave empty for default.
    """
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
        endpoint = base_url.rstrip("/") + "/chat/completions" if base_url \
            else "https://api.openai.com/v1/chat/completions"
        resp = httpx.post(
            endpoint,
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


def _resolve_provider_config(uc_config: dict) -> tuple[str, str, str, str]:
    """Return (provider, model, api_key, base_url) from uc_config + env vars.

    ``base_url`` is used for OpenAI-compatible endpoints (Fireworks, vLLM, etc.).
    An empty string means use the default ``https://api.openai.com/v1``.
    """
    llm = uc_config.get("llm", {})
    provider = (llm.get("sleep_faq_provider")
                or os.environ.get("SLEEP_FAQ_PROVIDER", "gemini"))
    model = (llm.get("sleep_faq_model")
             or os.environ.get("SLEEP_FAQ_MODEL", "gemini-2.5-flash"))
    base_url = llm.get("sleep_faq_base_url", "")
    key_map = {
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    key_env = key_map.get(provider, "GEMINI_API_KEY")
    api_key = llm.get("sleep_faq_api_key", "") or os.environ.get(key_env, "")
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
    return provider, model, api_key, base_url


def generate(cfg: dict, config_path: Path, count: int, output_path: Path,
             n_per_chunk: int = 3, paraphrases_per_faq: int = 0) -> None:
    """Run sleep-time FAQ generation and save to output_path.

    Args:
        paraphrases_per_faq: If > 0, generate this many diverse paraphrases
            of each newly-generated question (same answer), and append them
            alongside the originals. See `_augment_with_paraphrases` — this
            is what makes the resulting faqs.json a good chat-SFT training
            set instead of a single-phrasing-per-fact one.
    """
    uc_config = _load_uc_config(config_path)
    provider, model, api_key, base_url = _resolve_provider_config(uc_config)

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
    import time as _time
    chunks_file = cfg.get("chunks_file", "")
    if chunks_file:
        # Read from chunks.json directly (bypasses Qdrant)
        import json as _json
        all_chunks = _json.load(open(chunks_file))
        new_records = [(c["chunk_id"], c.get("text", "")) for c in all_chunks]
        all_count = len(all_chunks)
        store = None
        print(f"[sleep-faq] {all_count} chunks from {chunks_file}")
    else:
        store = get_store(cfg)
        all_results, _ = store.scroll(cfg["collection"], limit=10000,
                                      with_payload=True, with_vectors=False)
        if not all_results:
            print("[sleep-faq] ERROR: No chunks found. Run indexing first.")
            sys.exit(1)
        new_records = [(r.id, r.payload.get("text", ""))
                       for r in all_results
                       if r.payload.get("text") and not r.payload.get("faq_generated_at")]
        all_count = sum(1 for r in all_results if r.payload.get("text"))
        print(f"[sleep-faq] {all_count} chunks total, {len(new_records)} without FAQs")

    if not new_records:
        print("[sleep-faq] All chunks already have FAQs — nothing to do.")
        sys.exit(0)

    existing: list[dict] = []
    q_key = cfg.get("faq_question_key", "question")
    a_key = cfg.get("faq_answer_key", "answer")
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            print(f"[sleep-faq] Appending to {len(existing)} existing FAQs")
        except Exception:
            pass

    new_faqs: list[dict] = []
    chunk_idx = 0
    # Inter-request delay: start at 8s, back off up to 120s on repeated errors.
    # Keeps a steady pace so the API provider doesn't see a burst.
    _inter_delay = 8.0
    _consecutive_429 = 0

    while len(new_faqs) < count and chunk_idx < len(new_records):
        point_id, chunk = new_records[chunk_idx]
        chunk_idx += 1
        if not chunk.strip():
            continue
        try:
            prompt = _build_sleep_prompt(chunk, n_per_chunk=n_per_chunk)
            raw = _call_provider(provider, model, api_key, prompt, base_url=base_url)
            _consecutive_429 = 0          # reset on success
            # Keep a steady pace — do not decrease delay
            _inter_delay = max(8.0, _inter_delay * 0.95) if _inter_delay > 8.0 else 8.0
            chunk_new = []
            for b in _parse_sleep_blocks(raw):
                if len(new_faqs) >= count:
                    break
                new_faqs.append(tag_faq_with_chunk_ids({q_key: b["question"], a_key: b["answer"]}, [point_id]))
                chunk_new.append(b["question"][:80])
            if chunk_new:
                print(f"  [chunk {chunk_idx}/{len(new_records)}] +{len(chunk_new)} FAQs")
            # Mark chunk as processed so re-runs skip it (skipped when reading from file)
            if store is not None:
                try:
                    store.set_payload(cfg["collection"], point_id,
                                      {"faq_generated_at": int(_time.time())})
                except Exception as e:
                    print(f"  [chunk {chunk_idx}] warning: could not mark faq_generated_at: {e}")
        except Exception as e:
            err_str = str(e)
            is_retryable = "429" in err_str or "401" in err_str or "503" in err_str or "502" in err_str
            if is_retryable:
                _consecutive_429 += 1
                _inter_delay = min(60.0, _inter_delay * 2)
                wait = _inter_delay
                label = "rate-limited" if "429" in err_str else "auth/transient" 
                print(f"  [chunk {chunk_idx}] {label} ({err_str[:3]}) — waiting {wait:.0f}s before retry "
                      f"(consecutive: {_consecutive_429})")
                if _consecutive_429 >= 5:
                    print(f"\n[sleep-faq] Persistent errors from '{provider}/{model}' after 5 retries. "
                          f"Consider a different model or API key.\n")
                    _time.sleep(wait)
                    chunk_idx -= 1
                    continue
                _time.sleep(wait)
                chunk_idx -= 1  # retry this chunk
                continue
            else:
                print(f"  [chunk {chunk_idx}] error: {e}")
        # Save incrementally every 50 chunks to avoid losing progress
        if len(new_faqs) > 0 and len(new_faqs) % 50 == 0:
            merged = _deduplicate(existing, new_faqs)
            output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
            print(f"  [checkpoint] Saved {len(merged)} FAQs ({len(new_faqs)} new)")
        _time.sleep(_inter_delay)

    if paraphrases_per_faq > 0 and new_faqs:
        print(f"[sleep-faq] Generating {paraphrases_per_faq} paraphrases per FAQ "
              f"for {len(new_faqs)} newly-generated questions...")
        paraphrased = _augment_with_paraphrases(
            new_faqs, provider, model, api_key, paraphrases_per_faq, q_key=q_key, a_key=a_key)
        print(f"[sleep-faq] +{len(paraphrased)} paraphrase FAQs")
        new_faqs = new_faqs + paraphrased

    merged = _deduplicate(existing, new_faqs)
    output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"\n[sleep-faq] Saved {len(merged)} FAQs ({len(new_faqs)} appended from {chunk_idx} new chunks, {len(existing)} pre-existing)")

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
    p.add_argument("--paraphrases-per-faq", type=int, default=0,
                   help="Diverse paraphrases to generate per newly-generated FAQ question "
                        "(default: 0, disabled). Recommended for corpora with many distinct "
                        "facts and few phrasings each — a controlled experiment found this is "
                        "what lets chat-SFT training generalize to unseen phrasings; simply "
                        "repeating the same question does not.")
    p.add_argument("--chunks-file", default=None,
                   help="Path to chunks.json (bypasses Qdrant scroll).")
    args = p.parse_args()

    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())
    if args.chunks_file:
        cfg["chunks_file"] = args.chunks_file
    output_path = Path(args.output) if args.output else config_path.parent / "faqs.json"
    generate(cfg, config_path, args.count, output_path, args.n_per_chunk, args.paraphrases_per_faq)


if __name__ == "__main__":
    main()
