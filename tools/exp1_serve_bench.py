#!/usr/bin/env python3
"""Exp 1 serving-cost benchmark (vLLM): per-query GPU cost for one arm.

Measures the per-query serving cost that the break-even cost model consumes.
ONE arm per process (prefix-caching and LoRA are engine-init flags; separate
processes keep GPU memory clean).

Design note (why representative prompts, not full semantic retrieval):
  Exp 1 measures COST, which is driven by prompt/output *token counts* and
  throughput, not by which chunks are retrieved. Serving cost parity (§0.3,
  "matched prompt budget") is therefore satisfied by matching prompt LENGTH.
  - parametric arm: prompt = the question (short; answered from weights).
  - rag / rag_cache: prompt = top-K real chunks (K=5, ~600 chars each) + question,
    i.e. the real RAG prefill length. Chunks are pulled from the live Qdrant
    collection by scroll (no embedder needed; content is cost-irrelevant).
  Retrieval service time is measured separately (--measure-retrieval) so it can
  be reported apart from LLM serving, per §0.3.

Arms:
  parametric  : LLM(base, enable_lora=True), generate with LoRARequest; no context.
  rag         : LLM(base, enable_prefix_caching=False); context-stuffed prompts.
  rag_cache   : LLM(base, enable_prefix_caching=True); repeats hit the prefix cache.

Usage (run with venv_vllm):
  venv_vllm/bin/python tools/exp1_serve_bench.py \
      --arm rag --base-model meta-llama/Llama-3.2-3B-Instruct \
      --collection bedrock-userguide --n 200 --out results/exp1/uc4_rag.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

QDRANT = "http://localhost:6333"
SYSTEM = "You are a helpful assistant. Answer the question in 2-4 sentences."


def qdrant_scroll(collection: str, limit: int) -> list[dict]:
    """Pull up to `limit` points (payloads) from a collection via REST scroll."""
    body = json.dumps({"limit": limit, "with_payload": True, "with_vector": False}).encode()
    req = urllib.request.Request(
        f"{QDRANT}/collections/{collection}/points/scroll",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["result"]["points"]


def chunk_text(payload: dict) -> str:
    for k in ("text", "content", "chunk", "page_content", "document"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # fall back to the longest string field
    strs = [v for v in payload.values() if isinstance(v, str)]
    return max(strs, key=len) if strs else ""


def load_questions(path: str | None, n: int, fallback_chunks: list[str]) -> list[str]:
    if path and os.path.exists(path):
        data = json.load(open(path))
        qs = data["queries"] if isinstance(data, dict) else data
        qs = [q["question"] if isinstance(q, dict) else q for q in qs]
        if qs:
            return (qs * (n // len(qs) + 1))[:n]
    # fallback: synthesise short questions from chunk openings
    base = [f"What does the document say about: {c[:60]}?" for c in fallback_chunks[:n]]
    return (base * (n // max(len(base), 1) + 1))[:n] if base else [f"question {i}" for i in range(n)]


def build_prompts(arm: str, questions: list[str], chunks: list[str], top_k: int) -> list[str]:
    if arm == "parametric":
        return [f"{SYSTEM}\n\nQuestion: {q}\nAnswer:" for q in questions]
    prompts = []
    for i, q in enumerate(questions):
        ctx = "\n\n".join(chunks[(i + j) % len(chunks)] for j in range(top_k))  # K real chunks
        prompts.append(f"{SYSTEM}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:")
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["parametric", "rag", "rag_cache"])
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", help="LoRA adapter path (parametric arm)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--queries", help="JSON query pool; else synthesised from chunks")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    chunks = [chunk_text(p["payload"]) for p in qdrant_scroll(args.collection, max(args.top_k * 4, 50))]
    chunks = [c for c in chunks if c] or ["(empty chunk)"]
    questions = load_questions(args.queries, args.n, chunks)
    prompts = build_prompts(args.arm, questions, chunks, args.top_k)

    engine_kwargs = dict(
        model=args.base_model, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem, enforce_eager=args.enforce_eager, dtype="auto",
    )
    if args.arm == "parametric":
        engine_kwargs.update(enable_lora=True, max_lora_rank=args.lora_rank)
    else:
        engine_kwargs.update(enable_prefix_caching=(args.arm == "rag_cache"))

    t_init = time.time()
    llm = LLM(**engine_kwargs)
    init_s = time.time() - t_init

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    lora_req = None
    if args.arm == "parametric" and args.adapter:
        lora_req = LoRARequest("exp1", 1, args.adapter)

    # warmup (also populates prefix cache for rag_cache so steady-state is measured)
    llm.generate(prompts[: min(8, len(prompts))], sp, lora_request=lora_req, use_tqdm=False)

    t0 = time.time()
    outs = llm.generate(prompts, sp, lora_request=lora_req, use_tqdm=False)
    wall = time.time() - t0

    prompt_toks = [len(o.prompt_token_ids) for o in outs]
    out_toks = [len(o.outputs[0].token_ids) for o in outs]
    n = len(outs)
    result = {
        "arm": args.arm, "base_model": args.base_model, "collection": args.collection,
        "n": n, "top_k": args.top_k, "max_tokens": args.max_tokens,
        "engine_init_s": round(init_s, 2),
        "wall_s": round(wall, 3),
        "throughput_qps": round(n / wall, 3),
        "gpu_s_per_query": round(wall / n, 5),          # single dedicated A10G: wall ≈ GPU time
        "mean_prompt_tokens": round(sum(prompt_toks) / n, 1),
        "mean_output_tokens": round(sum(out_toks) / n, 1),
        "total_prompt_tokens": sum(prompt_toks),
        "total_output_tokens": sum(out_toks),
        "enforce_eager": args.enforce_eager,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
