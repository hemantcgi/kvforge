"""
model_loader.py — Load Llama 3.2 3B + optional LoRA adapter.

Singleton pattern: call load() once per process; reload() to swap LoRA adapter.
"""

from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Optional

# Import heavy GPU modules at module level so they're resolved once in the main
# thread.  Worker threads that call load() will then reuse the already-imported
# names without hitting the transformers lazy-import lock.
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_model = None
_tokenizer = None
_current_checkpoint: Optional[str] = None
_load_lock = threading.Lock()   # prevents concurrent loads from both racing

MODEL_ID = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _get_device()


def init(cfg: dict) -> None:
    """Override MODEL_ID from config. Call once before load()."""
    global MODEL_ID
    MODEL_ID = cfg.get("llm_model", MODEL_ID)


def load(lora_checkpoint: Optional[str] = None) -> tuple:
    """
    Load model + tokenizer. If lora_checkpoint is given, apply LoRA adapter.
    Returns (model, tokenizer). Cached after first call; thread-safe.
    """
    global _model, _tokenizer, _current_checkpoint
    # Fast path — no lock needed for reads once loaded
    if _model is not None and lora_checkpoint == _current_checkpoint:
        return _model, _tokenizer

    with _load_lock:
        # Re-check inside lock in case another thread loaded while we waited
        if _model is not None and lora_checkpoint == _current_checkpoint:
            return _model, _tokenizer

        if not _HAS_TORCH:
            raise ImportError("torch / transformers not available in this environment")

        print(f"🤖 Loading {MODEL_ID} on {DEVICE} …")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if lora_checkpoint and Path(lora_checkpoint).exists():
            from peft import PeftModel
            print(f"🔌 Applying LoRA adapter from {lora_checkpoint} …")
            _model = PeftModel.from_pretrained(_model, lora_checkpoint)
            _model = _model.merge_and_unload()  # merge for faster inference

        _model.eval()
        _current_checkpoint = lora_checkpoint
        return _model, _tokenizer


def reload(lora_checkpoint: Optional[str] = None) -> tuple:
    """
    Force reload from the base MODEL_ID, then apply lora_checkpoint if given.
    Always starts from the base model (not the previously-merged weights) so
    that repeated LoRA rounds each apply a fresh adapter without double-merging.
    """
    global _model, _tokenizer, _current_checkpoint
    _model = _tokenizer = _current_checkpoint = None
    return load(lora_checkpoint)


def get_kv_shape(cfg: dict) -> tuple[int, int, int]:
    """Return (num_layers, num_kv_heads, head_dim) from config."""
    return cfg["kv_num_layers"], cfg["kv_num_heads"], cfg["kv_head_dim"]
