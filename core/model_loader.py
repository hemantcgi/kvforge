"""Singleton loader for the Llama 3.2 3B language model with optional LoRA adapters.

Provides a thread-safe, process-level singleton for the HuggingFace model and
tokenizer so that GPU memory is allocated only once per process.

Typical usage::

    import model_loader
    model_loader.init(cfg)         # set model ID and HF token from config
    model, tokenizer = model_loader.load()           # base model
    model, tokenizer = model_loader.load(lora_ckpt)  # with LoRA adapter
    model, tokenizer = model_loader.reload(new_ckpt) # force fresh load

Public API: ``init``, ``load``, ``reload``, ``get_kv_shape``,
``detect_lora_targets``.
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
    """Return ``'cuda'`` if a CUDA GPU is available, otherwise ``'cpu'``."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _get_device()


def init(cfg: dict) -> None:
    """Override the module-level ``MODEL_ID`` from config.  Call once before ``load()``.

    Args:
        cfg: Datasource configuration dictionary.  Uses ``cfg['llm_model']``
            to set the model identifier and ``cfg['hf_token']`` to set the
            ``HF_TOKEN`` environment variable for gated models.
    """
    global MODEL_ID
    MODEL_ID = cfg.get("llm_model", MODEL_ID)
    token = cfg.get("hf_token")
    if token:
        os.environ["HF_TOKEN"] = token


def load(lora_checkpoint: Optional[str] = None) -> tuple:
    """Load (or return the cached) model and tokenizer.

    On the first call the base model is downloaded, moved to the correct
    device, and optionally merged with a LoRA adapter.  Subsequent calls with
    the same *lora_checkpoint* value return the cached pair without reloading.
    Thread-safe via an internal lock.

    Args:
        lora_checkpoint: Path to a PEFT LoRA adapter directory.  If ``None``
            or the path does not exist, the base model is returned.

    Returns:
        A ``(model, tokenizer)`` tuple where *model* is a HuggingFace
        ``AutoModelForCausalLM`` (in eval mode) and *tokenizer* is an
        ``AutoTokenizer``.

    Raises:
        ImportError: If ``torch`` or ``transformers`` are not installed.
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
    """Force a fresh load of the base model, then optionally apply a LoRA adapter.

    Clears the module-level singleton so the next ``load()`` call downloads
    and initialises the model from scratch.  Use this between LoRA training
    rounds to avoid double-merging adapters.

    Args:
        lora_checkpoint: Path to a PEFT LoRA adapter directory to apply after
            loading the base model.  ``None`` returns the bare base model.

    Returns:
        A fresh ``(model, tokenizer)`` tuple.
    """
    global _model, _tokenizer, _current_checkpoint
    _model = _tokenizer = _current_checkpoint = None
    return load(lora_checkpoint)


def _kv_shape_from_hf_config(hf_cfg) -> tuple[int, int, int]:
    """Extract (num_layers, num_kv_heads, head_dim) from a HuggingFace model config."""
    num_layers = hf_cfg.num_hidden_layers
    num_kv_heads = hf_cfg.num_key_value_heads
    if hasattr(hf_cfg, "head_dim") and hf_cfg.head_dim is not None:
        head_dim = hf_cfg.head_dim
    else:
        head_dim = hf_cfg.hidden_size // hf_cfg.num_attention_heads
    return num_layers, num_kv_heads, head_dim


def detect_lora_targets(model, configured_targets: list[str]) -> list[str]:
    """Verify that configured LoRA target module names exist in the model.

    Returns configured_targets if all are found. If none match, issues a warning
    listing the actual module names so the user can correct their config.
    """
    import warnings
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    matched = [t for t in configured_targets if t in module_names]
    if not matched:
        all_linear = sorted(
            {name.split(".")[-1] for name, mod in model.named_modules()
             if hasattr(mod, "weight") and len(getattr(getattr(mod, "weight", None), "shape", []) or []) == 2}
        )
        warnings.warn(
            f"None of the configured lora_target_modules {configured_targets} were found "
            f"in the model. Available linear layer names: {all_linear[:10]}. "
            f"Check 'lora_target_modules' in your datasource config.",
            UserWarning, stacklevel=2
        )
        return configured_targets  # return as-is; let peft raise a clear error
    return matched


def get_kv_shape(cfg: dict) -> tuple[int, int, int]:
    """Return ``(num_layers, num_kv_heads, head_dim)`` for KV cache allocation.

    Resolution order:

    1. ``cfg['model_library']`` registry entry keyed by the model ID (explicit
       override; useful for backwards compatibility and untested models).
    2. Auto-discovery from the loaded model's HuggingFace config object.
    3. Explicit legacy keys ``cfg['kv_num_layers']``, ``cfg['kv_num_heads']``,
       ``cfg['kv_head_dim']`` (raises ``KeyError`` if absent).

    Args:
        cfg: Datasource configuration dictionary.

    Returns:
        A ``(num_layers, num_kv_heads, head_dim)`` integer triple.

    Raises:
        KeyError: If the shape cannot be determined from the config or the
            loaded model and the legacy keys are also absent.
    """
    model_id = cfg.get("llm_model", MODEL_ID)
    entry = cfg.get("model_library", {}).get(model_id)
    if entry:
        return entry["kv_num_layers"], entry["kv_num_heads"], entry["kv_head_dim"]

    # Auto-discover from loaded model's config
    if _model is not None and hasattr(_model, "config"):
        try:
            return _kv_shape_from_hf_config(_model.config)
        except AttributeError:
            pass

    # Legacy explicit fields
    return cfg["kv_num_layers"], cfg["kv_num_heads"], cfg["kv_head_dim"]
