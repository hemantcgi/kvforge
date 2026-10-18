#!/usr/bin/env python3
"""Exp 1 real workload replay: MEASURED realized per-query serving cost.

Replaces the proxy c_warm / repeat_rate in the cost model with measured numbers.
Replays a Zipf(alpha) query stream through a vLLM engine in sequential waves at a
fixed concurrency, so first occurrences pay cold prefill and later repeats hit (or
miss) the real, finite, *evicting* prefix cache. One (arm, alpha) per process with
a FRESH engine, so the cache starts empty — no cross-condition contamination.

Prompts: `pool_size` DISTINCT, fixed (context, question) pairs built from real
chunks, one per pool item. A repeat of pool item i reuses i's exact prompt, so a
cache hit is a true identical-prefix hit (not an artifact of prompt construction).

Arms:
  parametric  : base + LoRA, no context (alpha-independent; run once).
  rag         : enable_prefix_caching=False (alpha-independent control; run once).
  rag_cache   : enable_prefix_caching=True (run per alpha).

Output: realized gpu_s/query (wall at stated concurrency / volume), throughput,
stream repeat-rate, and vLLM's prefix-cache hit rate when exposed by this version.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from tools.exp1_workload import sample_stream, stream_stats
from tools.exp1_serve_bench import qdrant_scroll, chunk_text, SYSTEM


def build_pool_prompts(arm: str, chunks: list[str], pool_size: int, top_k: int) -> list[str]:
    """pool_size DISTINCT fixed prompts; prompt[i] fully determined by pool item i."""
    nq = len(chunks)
    prompts = []
    for i in range(pool_size):
        q = f"Question {i}: what are the key points of this section?"
        if arm == "parametric":
            prompts.append(f"{SYSTEM}\n\nQuestion: {q}\nAnswer:")
        else:
            # distinct leading context per item -> exact-repeat caching; adjacent
            # items share a leading chunk, which is realistic prefix overlap.
            ctx = "\n\n".join(chunks[(i + j) % nq] for j in range(top_k))
            prompts.append(f"{SYSTEM}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:")
    return prompts


def try_hit_rate(llm) -> float | None:
    """Best-effort prefix-cache hit rate across vLLM metric APIs."""
    try:
        metrics = llm.get_metrics()
        q = h = None
        for m in metrics:
            name = getattr(m, "name", "")
            if "prefix_cache_queries" in name:
                q = getattr(m, "value", None)
            elif "prefix_cache_hits" in name:
                h = getattr(m, "value", None)
        if q and h is not None and q > 0:
            return round(h / q, 4)
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["parametric", "rag", "rag_cache"])
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--volume", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out-len", type=int, default=48,
                    help="matched output length; all arms generate exactly this many "
                         "tokens (ignore_eos) so wave time isn't gated by one slow request")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true",
                    help="use torch.compile/CUDA graphs; BROKEN on this cu130 env "
                         "(ninja build fails), so default is enforce_eager=True")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    raw = qdrant_scroll(args.collection, max(args.pool_size + args.top_k, 100))
    chunks = [c for c in (chunk_text(p["payload"]) for p in raw) if c] or ["(empty)"]
    pool_size = min(args.pool_size, max(len(chunks) - args.top_k, 1)) if args.arm != "parametric" else args.pool_size
    prompts = build_pool_prompts(args.arm, chunks, pool_size, args.top_k)

    stream = sample_stream(pool_size, args.alpha, args.volume, seed=args.seed)
    st = stream_stats(stream)
    stream_prompts = [prompts[i] for i in stream]

    engine_kwargs = dict(model=args.base_model, max_model_len=args.max_model_len,
                         gpu_memory_utilization=args.gpu_mem, dtype="auto",
                         enforce_eager=not args.compile,  # compile backend broken on cu130
                         disable_log_stats=False)         # enable prefix-cache metrics
    # LoRA only when an adapter is given: parametric WITHOUT --adapter is the
    # merged-equivalent (a deployed single-corpus Phase-3 model merges its adapter;
    # runtime LoRA is a multi-tenant path measured separately, per advisor).
    use_lora = args.arm == "parametric" and bool(args.adapter)
    if use_lora:
        engine_kwargs.update(enable_lora=True, max_lora_rank=args.lora_rank)
    if args.arm != "parametric":
        engine_kwargs.update(enable_prefix_caching=(args.arm == "rag_cache"))

    t_init = time.time()
    llm = LLM(**engine_kwargs)
    init_s = time.time() - t_init

    # matched output length across arms (ignore_eos) -> every request = out_len decode steps
    sp = SamplingParams(max_tokens=args.out_len, min_tokens=args.out_len,
                        ignore_eos=True, temperature=0.0)
    lora_req = LoRARequest("exp1", 1, args.adapter) if use_lora else None

    # warmup with throwaway prompts (generic prefix; does not pre-populate context cache)
    llm.generate(["hello"] * 4, sp, lora_request=lora_req, use_tqdm=False)

    C = args.concurrency
    t0 = time.time()
    tot_prompt = tot_out = 0
    for w in range(0, len(stream_prompts), C):
        outs = llm.generate(stream_prompts[w:w + C], sp, lora_request=lora_req, use_tqdm=False)
        for o in outs:
            tot_prompt += len(o.prompt_token_ids)
            tot_out += len(o.outputs[0].token_ids)
    wall = time.time() - t0
    n = len(stream_prompts)

    result = {
        "arm": args.arm, "base_model": args.base_model, "collection": args.collection,
        "alpha": args.alpha, "volume": n, "pool_size": pool_size,
        "concurrency": C, "top_k": args.top_k, "out_len": args.out_len,
        "ignore_eos": True, "runtime_lora": use_lora,
        "repeat_rate": round(st["repeat_rate"], 4), "distinct": st["distinct"],
        "engine_init_s": round(init_s, 1),
        "wall_s": round(wall, 3),
        "throughput_qps": round(n / wall, 3),
        "realized_gpu_s_per_query": round(wall / n, 5),
        "prefix_cache_hit_rate": try_hit_rate(llm),
        "mean_prompt_tokens": round(tot_prompt / n, 1),
        "mean_output_tokens": round(tot_out / n, 1),
        "enforce_eager": not args.compile,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
