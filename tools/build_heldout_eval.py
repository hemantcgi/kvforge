"""Build a frozen, versioned held-out evaluation set for a KVForge use case.

The eval set must never be used for training. It contains three parts:

1. **Paraphrase questions** — rewrites of a held-out subset of training FAQs.
2. **Novel questions** — new questions generated from corpus chunks that were
   not used to generate training FAQs.
3. **Negative / out-of-corpus probes** — questions not answerable from the
   corpus (optional, for routing calibration).

The script checks for overlap (normalized text) against the training FAQ set
and rejects duplicates.

Usage::

    python3 -m tools.build_heldout_eval \
        --config examples/usecase4_bedrock_userguide/config.json \
        --train-faqs examples/usecase4_bedrock_userguide/faqs.json \
        --chunks examples/usecase4_bedrock_userguide/data/amazon-bedrock-user-guide.chunks.json \
        --output examples/usecase4_bedrock_userguide/eval_heldout_v1.json \
        --n-paraphrase 40 --n-novel 40 --n-probe 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


def normalize_text(text: str) -> str:
    """SQuAD-style normalization for overlap detection."""
    text = text.lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-ish token overlap between two strings."""
    tokens_a = set(normalize_text(a).split())
    tokens_b = set(normalize_text(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def is_overlap(question: str, train_questions: list[str], threshold: float = 0.85) -> bool:
    """Return True if the question is too close to any training question."""
    return any(_token_overlap(question, tq) >= threshold for tq in train_questions)


def _call_provider(provider: str, model: str, api_key: str, prompt: str) -> str:
    """Call a cloud LLM provider. Reuses the same pattern as sleep_faq_generator."""
    if provider == "gemini":
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        resp = httpx.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.5, "maxOutputTokens": 1024},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    elif provider == "claude":
        # Use the Anthropic SDK so model-specific parameter handling (e.g.
        # temperature deprecation) is handled automatically.
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            c.text for c in message.content if c.type == "text"
        )

    elif provider == "openai":
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 1024,
                "temperature": 0.5,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    raise ValueError(f"Unknown provider: {provider!r}. Use gemini, claude, or openai.")


def _resolve_provider(
    config_path: Path | None,
    cli_provider: str | None = None,
    cli_model: str | None = None,
) -> tuple[str, str, str]:
    """Return (provider, model, api_key) from CLI args, uc_config, or env vars."""
    provider = cli_provider or os.environ.get("SLEEP_FAQ_PROVIDER", "gemini")
    model = cli_model or os.environ.get("SLEEP_FAQ_MODEL", "gemini-2.5-flash")
    if config_path:
        uc_path = config_path.parent / "uc_config.json"
        if uc_path.exists():
            uc = json.loads(uc_path.read_text())
            llm = uc.get("llm", {})
            provider = llm.get("sleep_faq_provider") or provider
            model = llm.get("sleep_faq_model") or model
    key_env = {
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider, "GEMINI_API_KEY")
    api_key = os.environ.get(key_env, "")
    if not api_key:
        raise RuntimeError(f"Set {key_env} environment variable.")
    return provider, model, api_key


def _parse_json_list(raw: str) -> list[dict[str, Any]]:
    """Extract a JSON list from an LLM response, tolerating markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("\n", 1)[0]
    raw = raw.strip()
    return json.loads(raw)


def generate_paraphrases(
    faqs: list[dict],
    train_questions: list[str],
    provider: str,
    model: str,
    api_key: str,
    n: int,
    seed: int,
) -> list[dict]:
    """Generate paraphrases of held-out FAQ questions."""
    rng = random.Random(seed)
    selected = rng.sample(faqs, min(n, len(faqs)))
    results = []
    for item in selected:
        prompt = (
            "Rewrite the following question in different words while keeping the exact same "
            "answer. Return ONLY a JSON object with keys 'question' and 'answer'.\n\n"
            f"Original question: {item['question']}\n"
            f"Answer: {item['answer'][:300]}"
        )
        try:
            raw = _call_provider(provider, model, api_key, prompt)
            parsed = _parse_json_list(raw)
            if isinstance(parsed, list):
                parsed = parsed[0]
            q = parsed.get("question", "").strip()
            a = parsed.get("answer", item["answer"]).strip()
            if not q:
                continue
            if is_overlap(q, train_questions):
                continue
            results.append(
                {
                    "question": q,
                    "answer": a,
                    "type": "paraphrase",
                    "source_faq": item.get("question", ""),
                }
            )
        except Exception as exc:
            print(f"⚠️ Paraphrase generation failed: {exc}")
    return results


def generate_novel_questions(
    chunks: list[str],
    train_questions: list[str],
    provider: str,
    model: str,
    api_key: str,
    n: int,
    seed: int,
) -> list[dict]:
    """Generate novel questions from random corpus chunks."""
    rng = random.Random(seed)
    selected = rng.sample(chunks, min(n * 3, len(chunks)))
    results = []
    for chunk in selected:
        if len(chunk.strip()) < 100:
            continue
        prompt = (
            "Read the passage below and write a single factual question that is answered "
            "directly by the passage. Return ONLY a JSON object with keys 'question' and "
            "'answer'. The answer should be a short span from the passage.\n\n"
            f"Passage:\n{chunk[:800]}"
        )
        try:
            raw = _call_provider(provider, model, api_key, prompt)
            parsed = _parse_json_list(raw)
            if isinstance(parsed, list):
                parsed = parsed[0]
            q = parsed.get("question", "").strip()
            a = parsed.get("answer", "").strip()
            if not q or not a:
                continue
            if is_overlap(q, train_questions):
                continue
            results.append(
                {
                    "question": q,
                    "answer": a,
                    "type": "novel",
                    "source_chunk": chunk[:200],
                }
            )
            if len(results) >= n:
                break
        except Exception as exc:
            print(f"⚠️ Novel question generation failed: {exc}")
    return results[:n]


def generate_out_of_corpus_probes(
    provider: str,
    model: str,
    api_key: str,
    n: int,
) -> list[dict]:
    """Generate questions that should not be answerable from the corpus."""
    results = []
    for _ in range(n):
        prompt = (
            "Write a question about AWS or generative AI that is NOT answered by the "
            "Amazon Bedrock User Guide. It should be a plausible user question but the "
            "answer is not in the Bedrock docs. Return ONLY a JSON object with keys "
            "'question' and 'answer'. The answer should be a general knowledge answer."
        )
        try:
            raw = _call_provider(provider, model, api_key, prompt)
            parsed = _parse_json_list(raw)
            if isinstance(parsed, list):
                parsed = parsed[0]
            q = parsed.get("question", "").strip()
            a = parsed.get("answer", "").strip()
            if q and a:
                results.append({"question": q, "answer": a, "type": "probe"})
        except Exception as exc:
            print(f"⚠️ Probe generation failed: {exc}")
    return results


def compute_version_hash(items: list[dict]) -> str:
    """Deterministic hash of the eval set content for versioning."""
    content = json.dumps(items, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to use-case config.json")
    p.add_argument("--train-faqs", required=True, help="Path to training FAQs JSON")
    p.add_argument("--chunks", required=True, help="Path to chunks JSON file")
    p.add_argument("--output", required=True, help="Path to write eval set")
    p.add_argument("--n-paraphrase", type=int, default=40)
    p.add_argument("--n-novel", type=int, default=40)
    p.add_argument("--n-probe", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--provider", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--overlap-threshold", type=float, default=0.85)
    args = p.parse_args()

    random.seed(args.seed)

    config_path = Path(args.config)
    provider, model, api_key = _resolve_provider(config_path, args.provider, args.model)

    with open(args.train_faqs) as f:
        train_faqs = json.load(f)
    train_questions = [item["question"] for item in train_faqs]

    with open(args.chunks) as f:
        chunk_data = json.load(f)
    chunks = chunk_data.get("chunks", [])

    print(f"📝 Generating {args.n_paraphrase} paraphrases...")
    paraphrases = generate_paraphrases(
        train_faqs, train_questions, provider, model, api_key, args.n_paraphrase, args.seed
    )
    print(f"   → {len(paraphrases)} paraphrases accepted")

    print(f"📝 Generating {args.n_novel} novel questions...")
    novel = generate_novel_questions(
        chunks, train_questions, provider, model, api_key, args.n_novel, args.seed
    )
    print(f"   → {len(novel)} novel questions accepted")

    probes = []
    if args.n_probe > 0:
        print(f"📝 Generating {args.n_probe} out-of-corpus probes...")
        probes = generate_out_of_corpus_probes(provider, model, api_key, args.n_probe)
        print(f"   → {len(probes)} probes accepted")

    eval_items = paraphrases + novel + probes
    version_hash = compute_version_hash(eval_items)

    output = {
        "version": f"v1-{version_hash}",
        "source_config": str(config_path),
        "source_faqs": args.train_faqs,
        "source_chunks": args.chunks,
        "n_paraphrase": len(paraphrases),
        "n_novel": len(novel),
        "n_probe": len(probes),
        "seed": args.seed,
        "overlap_threshold": args.overlap_threshold,
        "items": eval_items,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"💾 Wrote eval set ({len(eval_items)} items) to {args.output}")
    print(f"   Version: {output['version']}")


if __name__ == "__main__":
    main()

