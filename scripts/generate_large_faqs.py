#!/usr/bin/env python3
"""
Generate 1500-2000 Q&A pairs from a PDF, covering the entire document.

Reads the PDF directly (no Qdrant required), chunks at --chunk-size chars,
samples chunks evenly across the full document for complete coverage, then
calls an LLM concurrently to produce Q&A pairs per chunk.

Saves incrementally (every --save-every records) so progress survives
interrupts — re-running resumes from where it left off.

Usage:
    # Anthropic Claude (default)
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_large_faqs.py \\
        --pdf examples/usecase4_bedrock_userguide/data/amazon-bedrock-user-guide.pdf \\
        --output examples/usecase4_bedrock_userguide/faqs_train.json \\
        --count 2000

    # Google Gemini
    export GEMINI_API_KEY=...
    python scripts/generate_large_faqs.py --provider gemini --model gemini-2.5-flash ...
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """\
You are generating training Q&A pairs from a section of the Amazon Bedrock User Guide.

Document section:
{chunk}

Generate exactly {n} question-answer pairs that:
- Cover distinct facts, concepts, API names, steps, or limitations in this section
- Use clear, specific questions a developer or architect would actually ask
- Provide accurate, self-contained answers grounded only in the text above
- Vary in type: factual, procedural, conceptual, and comparison questions

Format (end every block with ---):
Q: <question>
A: <answer>
---
"""


# ---------------------------------------------------------------------------
# Qdrant chunk pull
# ---------------------------------------------------------------------------

def pull_qdrant_chunks(host: str, port: int, collection: str) -> list[str]:
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        sys.exit("qdrant-client required: pip install qdrant-client")

    print(f"Pulling chunks from Qdrant {host}:{port}/{collection} …", flush=True)
    client = QdrantClient(host=host, port=port)
    chunks, offset = [], None
    while True:
        results, offset = client.scroll(
            collection, limit=200, with_payload=True,
            with_vectors=False, offset=offset,
        )
        for r in results:
            text = r.payload.get("text", "").strip()
            if len(text) >= 80:
                chunks.append(text)
        if offset is None:
            break
    print(f"  {len(chunks)} chunks pulled from Qdrant", flush=True)
    return chunks


# ---------------------------------------------------------------------------
# PDF chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        c = text[start:end].strip()
        if len(c) >= 120:
            chunks.append(c)
        start += size - overlap
    return chunks


def extract_chunks(
    pdf_path: str,
    chunk_size: int = 600,
    overlap: int = 60,
    cache_path: Path | None = None,
) -> list[str]:
    # Return cached chunks if available
    if cache_path and cache_path.exists():
        print(f"Loading cached chunks from {cache_path}", flush=True)
        data = json.loads(cache_path.read_text())
        chunks = data["chunks"]
        print(f"  {len(chunks)} chunks loaded (instant)", flush=True)
        return chunks

    print(f"Extracting text from {pdf_path} …", flush=True)
    t0 = time.time()
    pages = []

    # pypdf is ~4x faster than pdfplumber for text-only extraction
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        total = len(reader.pages)
        for i, page in enumerate(reader.pages, 1):
            t = page.extract_text()
            if t:
                pages.append(t)
            if i % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (total - i) / i
                print(f"  {i}/{total} pages … ETA {eta:.0f}s", flush=True)
    except ImportError:
        try:
            import pdfplumber
        except ImportError:
            sys.exit("Install pypdf or pdfplumber: pip install pypdf")
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages, 1):
                t = page.extract_text()
                if t:
                    pages.append(t)
                if i % 500 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed * (total - i) / i
                    print(f"  {i}/{total} pages … ETA {eta:.0f}s", flush=True)

    full_text = "\n".join(pages)
    chunks = _chunk_text(full_text, chunk_size, overlap)
    print(f"  {len(chunks)} chunks from {total} pages in {time.time()-t0:.1f}s", flush=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"chunks": chunks, "pdf": str(pdf_path)}))
        print(f"  Cached to {cache_path} (skipped on next run)", flush=True)

    return chunks


# ---------------------------------------------------------------------------
# Q&A parsing
# ---------------------------------------------------------------------------

def parse_qa(text: str) -> list[dict]:
    results = []
    for block in re.split(r"\n---+\n?", text):
        block = block.strip()
        if not block:
            continue
        q_m = re.search(r"Q:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
        a_m = re.search(r"A:\s*(.+?)(?:\n---|\Z)", block, re.IGNORECASE | re.DOTALL)
        if not q_m or not a_m:
            continue
        q = q_m.group(1).strip()
        a = a_m.group(1).strip()
        if len(q) >= 10 and len(a) >= 10:
            results.append({"question": q, "answer": a})
    return results


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

async def _call_vllm(client: httpx.AsyncClient, base_url: str, model: str, prompt: str) -> str:
    """OpenAI-compatible chat completions — works with vLLM, Ollama, any local server."""
    r = await client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"content-type": "application/json"},
        json={"model": model, "max_tokens": 1200, "temperature": 0.7,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def _call_claude(client: httpx.AsyncClient, key: str, model: str, prompt: str) -> str:
    r = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 1200, "temperature": 0.5,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=90,
    )
    r.raise_for_status()
    return "".join(c.get("text", "") for c in r.json().get("content", [])
                   if c.get("type") == "text")


async def _call_gemini(client: httpx.AsyncClient, key: str, model: str, prompt: str) -> str:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    r = await client.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.5, "maxOutputTokens": 1200}},
        timeout=90,
    )
    r.raise_for_status()
    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


class RateLimiter:
    """Token bucket: allows at most `rate` requests per second on average."""

    def __init__(self, rate: float):
        self._rate = rate          # requests per second
        self._min_interval = 1.0 / rate
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last + self._min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


async def process_chunk(
    sem: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    client: httpx.AsyncClient,
    provider: str,
    key: str,           # api key (empty string for vllm)
    model: str,
    chunk: str,
    n: int,
    vllm_base: str = "",
) -> list[dict]:
    async with sem:
        await rate_limiter.acquire()
        prompt = PROMPT.format(chunk=chunk[:1500], n=n)
        for attempt in range(4):
            try:
                if provider == "vllm":
                    raw = await _call_vllm(client, vllm_base, model, prompt)
                elif provider == "claude":
                    raw = await _call_claude(client, key, model, prompt)
                else:
                    raw = await _call_gemini(client, key, model, prompt)
                return parse_qa(raw)
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code == 429:
                    wait = 30 * (2 ** attempt)
                    await asyncio.sleep(wait)
                elif code >= 500:
                    await asyncio.sleep(10 * (attempt + 1))
                else:
                    return []
            except Exception:
                await asyncio.sleep(5 * (attempt + 1))
        return []


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------

async def run(
    chunks: list[str],
    provider: str,
    key: str,
    model: str,
    target: int,
    n_per_chunk: int,
    output: Path,
    save_every: int,
    concurrency: int,
    vllm_base: str = "",
    rate: float = 100.0,
) -> list[dict]:
    # Load any prior progress
    existing: list[dict] = []
    if output.exists():
        try:
            existing = json.loads(output.read_text())
            print(f"Resuming — {len(existing)} records already saved", flush=True)
        except Exception:
            pass

    seen = {item["question"].strip().lower() for item in existing}
    faqs = list(existing)

    # Sample chunks uniformly across the full document
    need_chunks = max(1, (target - len(faqs) + n_per_chunk - 1) // n_per_chunk)
    # Add 30% buffer for parse failures and duplicates
    need_chunks = int(need_chunks * 1.3)
    need_chunks = min(need_chunks, len(chunks))

    if need_chunks < len(chunks):
        step = max(1, len(chunks) // need_chunks)
        sampled = [chunks[i] for i in range(0, len(chunks), step)][:need_chunks]
    else:
        sampled = chunks

    already_done = len(faqs)
    print(
        f"\n{len(chunks)} total chunks → sampling {len(sampled)} evenly spaced "
        f"(step={len(chunks)//max(1,len(sampled))})\n"
        f"Target: {target} Q&A  |  {n_per_chunk}/chunk  |  {concurrency} concurrent  |  {provider}/{model}\n",
        flush=True,
    )

    # vLLM local: no rate limit. Cloud APIs: respect tier limits.
    rl = RateLimiter(rate)
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    completed = 0
    last_save = len(faqs)

    async with httpx.AsyncClient() as client:
        tasks = [
            process_chunk(sem, rl, client, provider, key, model, chunk,
                          n_per_chunk, vllm_base=vllm_base)
            for chunk in sampled
        ]

        for coro in asyncio.as_completed(tasks):
            pairs = await coro
            completed += 1

            for p in pairs:
                k = p["question"].strip().lower()
                if k not in seen:
                    faqs.append(p)
                    seen.add(k)

            elapsed = time.time() - t0
            rate = max(0.01, (len(faqs) - already_done) / elapsed)
            remaining = max(0, target - len(faqs))
            eta_min = (remaining / rate) / 60

            print(
                f"\r  {len(faqs)}/{target} Q&A  |  "
                f"{completed}/{len(tasks)} chunks  |  "
                f"{rate:.1f} Q/s  |  ETA {eta_min:.1f} min   ",
                end="", flush=True,
            )

            # Incremental save
            if len(faqs) - last_save >= save_every:
                output.write_text(json.dumps(faqs, indent=2, ensure_ascii=False))
                last_save = len(faqs)

            if len(faqs) >= target:
                break

    # Final save
    output.write_text(json.dumps(faqs, indent=2, ensure_ascii=False))
    print(f"\n\nDone — {len(faqs)} Q&A pairs saved to {output}", flush=True)
    return faqs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate large Q&A dataset from a PDF using a cloud LLM"
    )
    p.add_argument("--pdf", default=None, help="Path to source PDF (omit when using --qdrant-source)")
    p.add_argument("--output", required=True, help="Output JSON path (faqs_train.json)")
    p.add_argument("--count", type=int, default=2000, help="Target Q&A pairs (default: 2000)")
    p.add_argument("--provider", default="claude",
                   choices=["claude", "gemini", "vllm"],
                   help="LLM provider: vllm (local), claude, or gemini (default: claude)")
    p.add_argument("--model", default=None,
                   help="Model name (default per provider: Qwen/Qwen3-4B | claude-haiku-4-5-20251001 | gemini-2.5-flash)")
    p.add_argument("--vllm-base", default="http://localhost:8090/v1",
                   help="Base URL for local vLLM/OpenAI-compatible server (default: http://localhost:8090/v1)")
    p.add_argument("--qdrant-source", action="store_true",
                   help="Pull chunks from Qdrant instead of a PDF (requires --qdrant-collection)")
    p.add_argument("--qdrant-host", default="localhost")
    p.add_argument("--qdrant-port", type=int, default=6333)
    p.add_argument("--qdrant-collection", default="bedrock-user-guide")
    p.add_argument("--n-per-chunk", type=int, default=4,
                   help="Q&A pairs to request per chunk (default: 4)")
    p.add_argument("--chunk-size", type=int, default=600,
                   help="Chars per text chunk (default: 600, matches KVForge indexing)")
    p.add_argument("--chunk-overlap", type=int, default=60,
                   help="Overlap between chunks (default: 60)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Concurrent LLM requests (default: 8)")
    p.add_argument("--save-every", type=int, default=100,
                   help="Incremental save interval in records (default: 100)")
    p.add_argument("--no-cache", action="store_true",
                   help="Skip chunk cache (re-extract PDF every run)")
    args = p.parse_args()

    # Resolve model, key, and rate limit per provider
    vllm_base = ""
    rate = 100.0   # requests/sec; effectively unlimited for local vLLM
    if args.provider == "vllm":
        model = args.model or "Qwen/Qwen3-4B"
        key = ""
        vllm_base = args.vllm_base
    elif args.provider == "claude":
        model = args.model or "claude-haiku-4-5-20251001"
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            sys.exit("Set ANTHROPIC_API_KEY to use the claude provider")
        rate = 0.7   # ~40 RPM for Haiku Tier 1
    else:
        model = args.model or "gemini-2.5-flash"
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            sys.exit("Set GEMINI_API_KEY to use the gemini provider")
        rate = 10.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Source: Qdrant collection or PDF
    if args.qdrant_source:
        chunks = pull_qdrant_chunks(args.qdrant_host, args.qdrant_port, args.qdrant_collection)
    else:
        if not args.pdf:
            sys.exit("Provide --pdf or use --qdrant-source")
        cache = None if args.no_cache else Path(args.pdf).with_suffix(".chunks.json")
        chunks = extract_chunks(args.pdf, args.chunk_size, args.chunk_overlap, cache_path=cache)

    asyncio.run(run(
        chunks=chunks,
        provider=args.provider,
        key=key,
        model=model,
        target=args.count,
        n_per_chunk=args.n_per_chunk,
        output=output,
        save_every=args.save_every,
        concurrency=args.concurrency,
        vllm_base=vllm_base,
        rate=rate,
    ))


if __name__ == "__main__":
    main()
