#!/usr/bin/env python3
"""
Sequential benchmark: FP16 merged model vs AWQ quantized model via vLLM.

Flow:
  1. Start FP16 vLLM on port 8091, run 10 random UC4 FAQ queries, stop server
  2. Start AWQ vLLM on port 8091, run same 10 queries, stop server
  3. Print side-by-side comparison; save full log to awq_comparison_results.json

Run merge_and_quantize.py first to produce ./merged_fp16 and ./merged_awq.
"""

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
FAQS_PATH   = "examples/usecase4_bedrock_userguide/faqs.json"
FP16_MODEL  = "./merged_fp16"
AWQ_MODEL   = "./merged_awq"
PORT        = 8091
LOG_FILE    = "awq_comparison_results.json"
N_QUESTIONS = 10
SEED        = 42
MAX_TOKENS  = 200
TEMPERATURE = 0.0
STARTUP_TIMEOUT = 150   # seconds; --enforce-eager skips CUDA graph compilation (~80s startup)
GPU_MEM_UTIL      = "0.80"   # dashboard holds ~2.9GB on GPU3; need headroom below 19.09GB free
MAX_MODEL_LEN     = "4096"
TENSOR_PARALLEL   = "4"     # 4x A10G available on this instance

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_questions():
    with open(FAQS_PATH) as f:
        faqs = json.load(f)
    random.seed(SEED)
    return random.sample(faqs, N_QUESTIONS)


def wait_for_vllm(port: int, timeout: int = STARTUP_TIMEOUT) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=5)
            if r.status_code == 200:
                print()
                return True
        except Exception:
            pass
        time.sleep(3)
        print(".", end="", flush=True)
        dots += 1
    print()
    return False


def llama_prompt(question: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def query_vllm(question: str, model_name: str, port: int) -> dict:
    payload = {
        "model": model_name,
        "prompt": llama_prompt(question),
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    t0 = time.time()
    r = httpx.post(
        f"http://localhost:{port}/v1/completions",
        json=payload,
        timeout=120,
    )
    elapsed_ms = (time.time() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    answer = data["choices"][0]["text"].strip()
    usage = data.get("usage", {})
    prompt_tokens     = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens") or math.ceil(len(answer.split()) * 1.33)
    tok_per_sec = completion_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
    return {
        "answer":            answer,
        "latency_ms":        round(elapsed_ms, 1),
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "tok_per_sec":       round(tok_per_sec, 1),
    }


def run_phase(label: str, model_path: str, extra_flags: list, questions: list) -> list:
    model_name = Path(model_path).name
    print(f"\n{'='*72}")
    print(f"  PHASE: {label}   model: {model_path}")
    print(f"{'='*72}\n")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",              model_path,
        "--served-model-name",  model_name,
        "--port",               str(PORT),
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--max-model-len",      MAX_MODEL_LEN,
        "--tensor-parallel-size", TENSOR_PARALLEL,
        "--dtype",              "float16",
        "--enforce-eager",      # skips CUDA graph compilation; ~80s startup vs 10min
    ] + extra_flags

    print(f"CMD: {' '.join(cmd)}\n")
    vllm_log = open(f"vllm_{label.lower()}.log", "w")
    proc = subprocess.Popen(cmd, stdout=vllm_log, stderr=vllm_log)

    print(f"Waiting for vLLM (up to {STARTUP_TIMEOUT}s)", end="", flush=True)
    if not wait_for_vllm(PORT):
        proc.terminate()
        print("ERROR: vLLM did not become ready in time — check GPU memory")
        sys.exit(1)
    print("vLLM ready.\n")

    results = []
    for i, faq in enumerate(questions, 1):
        q  = faq["question"]
        gt = faq["answer"]
        print(f"[{i:2d}/{N_QUESTIONS}] Q: {q[:90]}")
        try:
            res = query_vllm(q, model_name, PORT)
            res.update({"question": q, "ground_truth": gt, "model": label})
            results.append(res)
            print(f"         A: {res['answer'][:150].replace(chr(10), ' ')}")
            print(f"         latency={res['latency_ms']:.0f}ms  "
                  f"tokens={res['completion_tokens']}  "
                  f"tok/s={res['tok_per_sec']:.1f}\n")
        except Exception as e:
            print(f"         ERROR: {e}\n")
            results.append({
                "question": q, "ground_truth": gt, "model": label,
                "error": str(e), "answer": "",
                "latency_ms": 0, "completion_tokens": 0,
                "prompt_tokens": 0, "tok_per_sec": 0,
            })

    proc.terminate()
    proc.wait()
    vllm_log.close()
    print(f"vLLM stopped. Sleeping 10s to free GPU memory...")
    time.sleep(10)
    return results


def avg(lst, key):
    vals = [x[key] for x in lst if isinstance(x.get(key), (int, float)) and x[key] > 0]
    return sum(vals) / len(vals) if vals else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
if not Path(FP16_MODEL).exists():
    print(f"ERROR: {FP16_MODEL} not found — run merge_and_quantize.py first")
    sys.exit(1)
if not Path(AWQ_MODEL).exists():
    print(f"ERROR: {AWQ_MODEL} not found — run merge_and_quantize.py first")
    sys.exit(1)

questions = load_questions()
print(f"Selected {N_QUESTIONS} questions (seed={SEED}):\n")
for i, faq in enumerate(questions, 1):
    print(f"  {i:2d}. {faq['question']}")

fp16_results = run_phase("FP16", FP16_MODEL, [], questions)
awq_results  = run_phase("AWQ",  AWQ_MODEL,  ["--quantization", "awq"], questions)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("COMPARISON SUMMARY")
print(f"{'='*72}")
print(f"{'#':>2}  {'FP16 ms':>8}  {'AWQ ms':>8}  {'Speedup':>8}  "
      f"{'FP16 tok/s':>10}  {'AWQ tok/s':>10}")
print("-" * 60)

for i, (fp, aq) in enumerate(zip(fp16_results, awq_results), 1):
    sp = (f"{fp['latency_ms']/aq['latency_ms']:.2f}x"
          if fp['latency_ms'] > 0 and aq['latency_ms'] > 0 else "N/A")
    print(f"{i:2d}  {fp['latency_ms']:>8.0f}  {aq['latency_ms']:>8.0f}  "
          f"{sp:>8}  {fp['tok_per_sec']:>10.1f}  {aq['tok_per_sec']:>10.1f}")

fp_lat = avg(fp16_results, 'latency_ms')
aq_lat = avg(awq_results,  'latency_ms')
speedup_str = f"{fp_lat/aq_lat:.2f}x" if aq_lat > 0 else "N/A"
print("-" * 60)
print(f"{'AVG':>2}  {fp_lat:>8.0f}  {aq_lat:>8.0f}  {speedup_str:>8}  "
      f"{avg(fp16_results,'tok_per_sec'):>10.1f}  {avg(awq_results,'tok_per_sec'):>10.1f}")

# ── Per-question detailed output ───────────────────────────────────────────────
print(f"\n{'='*72}")
print("PER-QUESTION DETAIL")
print(f"{'='*72}")
SEP = "-" * 72

for i, (fp, aq) in enumerate(zip(fp16_results, awq_results), 1):
    print(f"\n[Q{i}] {fp['question']}")
    print(SEP)
    gt_text = fp['ground_truth'][:400].replace('\n', ' ')
    fp_text = fp['answer'][:400].replace('\n', ' ')
    aq_text = aq['answer'][:400].replace('\n', ' ')
    print(f"  GROUND TRUTH :\n    {gt_text}")
    print(f"\n  FP16 ({fp['latency_ms']:.0f}ms | {fp['tok_per_sec']:.1f} tok/s | {fp['completion_tokens']} tokens):\n    {fp_text}")
    print(f"\n  AWQ  ({aq['latency_ms']:.0f}ms | {aq['tok_per_sec']:.1f} tok/s | {aq['completion_tokens']} tokens):\n    {aq_text}")

# ── Save full JSON log ────────────────────────────────────────────────────────
log = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "config": {
        "n_questions":  N_QUESTIONS,
        "seed":         SEED,
        "max_tokens":   MAX_TOKENS,
        "temperature":  TEMPERATURE,
        "fp16_model":   FP16_MODEL,
        "awq_model":    AWQ_MODEL,
    },
    "summary": {
        "fp16_avg_latency_ms":  round(fp_lat, 1),
        "awq_avg_latency_ms":   round(aq_lat, 1),
        "speedup":              round(fp_lat / aq_lat, 2) if aq_lat > 0 else None,
        "fp16_avg_tok_per_sec": round(avg(fp16_results, 'tok_per_sec'), 1),
        "awq_avg_tok_per_sec":  round(avg(awq_results,  'tok_per_sec'), 1),
    },
    "fp16_results": fp16_results,
    "awq_results":  awq_results,
}
with open(LOG_FILE, "w") as f:
    json.dump(log, f, indent=2)

print(f"\n\nFull results saved → {LOG_FILE}")
print(f"AWQ speedup: {speedup_str}  |  "
      f"FP16 avg: {fp_lat:.0f}ms  |  AWQ avg: {aq_lat:.0f}ms")
