"""KVForge Compute Worker — FastAPI service for remote GPU machines.

Loads an LLM at startup and serves KV tensor computation over HTTP.
Run with:
    python -m addons.compute.worker --model <hf-model-id> --port 8091

Both VectorDB and GPU can be remote machines managed by KVForge Studio.
This worker is what runs on the GPU side.
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel

import core.kv_utils as kv_utils

app = FastAPI(title="KVForge Compute Worker", version="1.0.0")

# Model state — populated by load_model() at startup or via CLI
_model = None
_tokenizer = None
_model_id: str = ""
_api_key: str = ""


# ── Auth ─────────────────────────────────────────────────────────────────────

def _check_auth(x_api_key: Optional[str] = Header(None)) -> None:
    if _api_key and x_api_key != _api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ── Schemas ───────────────────────────────────────────────────────────────────

class ComputeKVRequest(BaseModel):
    texts: list[str]
    num_layers: int
    num_kv_heads: int
    head_dim: int


class ComputeKVResponse(BaseModel):
    tensors: list[str]       # base64-encoded float16 KV arrays
    shape: list[int]         # [num_layers, 2, num_kv_heads, head_dim]
    elapsed_ms: float
    count: int


class HealthResponse(BaseModel):
    status: str
    device: str
    model_id: str
    model_loaded: bool
    gpu_info: Optional[dict]


# ── GPU helpers ───────────────────────────────────────────────────────────────

def _gpu_info() -> Optional[dict]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return {
            "name": torch.cuda.get_device_name(0),
            "memory_free_mb": torch.cuda.mem_get_info(0)[0] // (1024 * 1024),
            "memory_total_mb": torch.cuda.mem_get_info(0)[1] // (1024 * 1024),
        }
    except Exception:
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="ok",
        device=device,
        model_id=_model_id,
        model_loaded=_model is not None,
        gpu_info=_gpu_info(),
    )


@app.post("/compute_kv", response_model=ComputeKVResponse, dependencies=[Depends(_check_auth)])
def compute_kv(req: ComputeKVRequest) -> ComputeKVResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — call load_model() first")
    if not req.texts:
        return ComputeKVResponse(tensors=[], shape=[req.num_layers, 2, req.num_kv_heads, req.head_dim], elapsed_ms=0.0, count=0)

    from core.compute import compute_kv_for_chunk

    t0 = time.perf_counter()
    tensors = []
    for text in req.texts:
        arr = compute_kv_for_chunk(
            text, _model, _tokenizer,
            req.num_layers, req.num_kv_heads, req.head_dim,
        )
        tensors.append(kv_utils.serialize_kv(arr))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ComputeKVResponse(
        tensors=tensors,
        shape=[req.num_layers, 2, req.num_kv_heads, req.head_dim],
        elapsed_ms=elapsed_ms,
        count=len(tensors),
    )


@app.post("/infer")
def infer() -> dict:
    # Phase 2: KV-injected remote inference (not yet implemented)
    raise HTTPException(status_code=501, detail="Remote inference (Phase 2) not yet implemented")


# ── Model management ──────────────────────────────────────────────────────────

def load_model(model_id: str, lora_ckpt: Optional[str] = None) -> None:
    """Load model into VRAM. Called at worker startup."""
    global _model, _tokenizer, _model_id
    import core.model_loader as model_loader
    _model, _tokenizer = model_loader.load(lora_ckpt)
    _model_id = model_id


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="KVForge Compute Worker")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--port", type=int, default=8091, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--lora", default=None, help="Path to LoRA checkpoint directory")
    parser.add_argument("--api-key", default="", dest="api_key", help="Optional API key for auth")
    args = parser.parse_args()

    global _api_key
    _api_key = args.api_key

    print(f"Loading model {args.model} …")
    load_model(args.model, args.lora)
    print(f"Model loaded. Starting worker on {args.host}:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
