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
_current_attn_impl: Optional[str] = None
_load_lock = threading.Lock()   # prevents concurrent loads from both racing

MODEL_ID = os.getenv("LLM_MODEL", "google/gemma-4-E2B-it")
# Quantization mode: None | "4bit" | "8bit".  Set via cfg["quantization"].
# 4-bit (NF4) reduces model weight reads from ~6.4 GB to ~1.6 GB, giving
# 2-3× decode speedup for memory-bandwidth-bound single-request inference.
QUANTIZATION: Optional[str] = None
# Attention implementation: None | "sdpa" | "eager" | "flash_attention_2".
# Eager is required for output_attentions (used by eval_attention_divergence.py).
ATTN_IMPLEMENTATION: Optional[str] = None


def _get_device() -> str:
    """Return ``'cuda'`` if a CUDA GPU is available, otherwise ``'cpu'``."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _get_device()


def init(cfg: dict) -> None:
    """Override the module-level ``MODEL_ID`` and ``QUANTIZATION`` from config.

    Args:
        cfg: Datasource configuration dictionary.  Uses ``cfg['llm_model']``
            to set the model identifier, ``cfg['hf_token']`` to set the
            ``HF_TOKEN`` environment variable, and ``cfg['quantization']``
            (``"4bit"`` | ``"8bit"`` | omit for fp16) to enable BitsAndBytes
            quantization for faster memory-bandwidth-bound inference.
    """
    global MODEL_ID, QUANTIZATION, ATTN_IMPLEMENTATION
    # Support both flat configs and the nested addon_config.inference layout.
    inference_cfg = cfg.get("addon_config", {}).get("inference", {})
    effective_cfg = {**cfg, **inference_cfg}
    MODEL_ID = effective_cfg.get("llm_model", MODEL_ID)
    QUANTIZATION = effective_cfg.get("quantization", None)
    ATTN_IMPLEMENTATION = effective_cfg.get("attn_implementation", ATTN_IMPLEMENTATION)
    token = effective_cfg.get("hf_token")
    if token:
        os.environ["HF_TOKEN"] = token


def _unwrap_gemma4(model):
    """Delete vision/audio towers and unwrap ClippableLinear in text attention layers.

    Gemma4's attention projections (q/k/v/o) are ``Gemma4ClippableLinear``
    wrappers that PEFT cannot inject LoRA into.  Replacing them with their
    inner ``.linear`` attribute exposes a standard ``nn.Linear`` that PEFT
    can work with.  Vision/audio towers are deleted to save VRAM and to
    remove their incompatible modules (32-dim inv_freq, etc.).
    """
    import torch.nn as nn
    unwrapped = 0
    for name, mod in list(model.named_modules()):
        parent_path = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]
        # Delete vision/audio tower submodules (skip children of already-deleted parents)
        if "vision_tower" in name or "audio_tower" in name:
            try:
                parent = model.get_submodule(parent_path) if parent_path else model
                delattr(parent, attr_name)
            except (AttributeError, TypeError):
                pass
            continue
        # Unwrap ClippableLinear in text layers to nn.Linear
        if ("language_model" in name and "Gemma4ClippableLinear" in type(mod).__name__
                and hasattr(mod, "linear") and isinstance(mod.linear, nn.Linear)):
            try:
                parent = model.get_submodule(parent_path) if parent_path else model
                setattr(parent, attr_name, mod.linear)
                unwrapped += 1
            except (AttributeError, TypeError):
                pass
    print(f"[gemma4] Unwrapped {unwrapped} ClippableLinear modules in text layers, "
          f"vision/audio towers removed, GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def load(lora_checkpoint: Optional[str] = None, attn_implementation: Optional[str] = None) -> tuple:
    """Load (or return the cached) model and tokenizer.

    On the first call the base model is downloaded, moved to the correct
    device, and optionally merged with a LoRA adapter.  Subsequent calls with
    the same *lora_checkpoint* value return the cached pair without reloading.
    Thread-safe via an internal lock.

    Args:
        lora_checkpoint: Path to a PEFT LoRA adapter directory.  If ``None``
            or the path does not exist, the base model is returned.
        attn_implementation: Optional attention backend override. ``"eager"``
            is required for ``output_attentions=True`` in generation.

    Returns:
        A ``(model, tokenizer)`` tuple where *model* is a HuggingFace
        ``AutoModelForCausalLM`` (in eval mode) and *tokenizer* is an
        ``AutoTokenizer``.

    Raises:
        ImportError: If ``torch`` or ``transformers`` are not installed.
    """
    global _model, _tokenizer, _current_checkpoint, _current_attn_impl
    attn_implementation = attn_implementation or ATTN_IMPLEMENTATION
    # Fast path — no lock needed for reads once loaded
    if _model is not None and lora_checkpoint == _current_checkpoint and attn_implementation == _current_attn_impl:
        return _model, _tokenizer

    with _load_lock:
        # Re-check inside lock in case another thread loaded while we waited
        if _model is not None and lora_checkpoint == _current_checkpoint and attn_implementation == _current_attn_impl:
            return _model, _tokenizer

        if not _HAS_TORCH:
            raise ImportError("torch / transformers not available in this environment")

        quant_label = f" [{QUANTIZATION}]" if QUANTIZATION else " [fp16]"
        print(f"🤖 Loading {MODEL_ID} on {DEVICE}{quant_label} …")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        # Build quantization config if requested.
        # 4-bit NF4 reduces model weight reads from ~6.4 GB to ~1.6 GB,
        # giving ~2-3× decode speedup for batch_size=1 memory-BW-bound inference.
        # 8-bit halves memory reads for a ~1.5× speedup with higher quality.
        quant_cfg = None
        if QUANTIZATION in ("4bit", "8bit"):
            if DEVICE == "cpu":
                print("⚠️  Quantization requires CUDA — running on CPU in fp32 instead")
            else:
                try:
                    from transformers import BitsAndBytesConfig
                    if QUANTIZATION == "4bit":
                        quant_cfg = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                    else:  # 8bit
                        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
                except (ImportError, ValueError) as e:
                    print(f"⚠️  bitsandbytes unavailable ({e}) — falling back to fp16")

        load_kwargs = {
            "pretrained_model_name_or_path": MODEL_ID,
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "quantization_config": quant_cfg,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        _model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        # Gemma4 post-load: delete vision/audio towers (save VRAM, remove
        # incompatible modules), then unwrap Gemma4ClippableLinear in text
        # layers so PEFT can inject LoRA into q_proj/k_proj/v_proj/o_proj.
        if "gemma-4" in MODEL_ID.lower() or "gemma4" in MODEL_ID.lower():
            _unwrap_gemma4(_model)

        if lora_checkpoint and Path(lora_checkpoint).exists():
            from peft import PeftModel
            print(f"🔌 Applying LoRA adapter from {lora_checkpoint} …")
            _model = PeftModel.from_pretrained(_model, lora_checkpoint)
            if quant_cfg is None:
                # Merge LoRA into base weights for faster inference (only safe
                # for non-quantized models; quantized models run PEFT forward).
                _model = _model.merge_and_unload()

        _model.eval()
        _current_checkpoint = lora_checkpoint
        _current_attn_impl = attn_implementation
        return _model, _tokenizer


def reload(lora_checkpoint: Optional[str] = None, attn_implementation: Optional[str] = None) -> tuple:
    """Force a fresh load of the base model, then optionally apply a LoRA adapter.

    Clears the module-level singleton so the next ``load()`` call downloads
    and initialises the model from scratch.  Use this between LoRA training
    rounds to avoid double-merging adapters.

    Args:
        lora_checkpoint: Path to a PEFT LoRA adapter directory to apply after
            loading the base model.  ``None`` returns the bare base model.
        attn_implementation: Optional attention backend override. ``"eager"``
            is required for ``output_attentions=True`` in generation.

    Returns:
        A fresh ``(model, tokenizer)`` tuple.
    """
    global _model, _tokenizer, _current_checkpoint, _current_attn_impl
    _model = _tokenizer = _current_checkpoint = _current_attn_impl = None
    return load(lora_checkpoint, attn_implementation=attn_implementation)


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

    Only returns target names where ALL matching modules are compatible PEFT
    types (``torch.nn.Linear`` subclasses).  This avoids failures on wrapped
    layers like ``Gemma4ClippableLinear`` which PEFT cannot inject into.
    """
    import warnings
    import torch.nn as nn
    # Group modules by leaf name and check types
    name_groups: dict[str, list[type]] = {}
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        name_groups.setdefault(leaf, []).append(type(mod))

    matched = []
    for target in configured_targets:
        types = name_groups.get(target)
        if types is None:
            continue
        # Only accept target if ALL modules with this name are nn.Linear subclasses
        if all(issubclass(t, nn.Linear) for t in types):
            matched.append(target)

    if not matched:
        all_safe = sorted(
            leaf for leaf, types in name_groups.items()
            if all(issubclass(t, nn.Linear) for t in types)
        )
        warnings.warn(
            f"None of the configured lora_target_modules {configured_targets} are safe "
            f"(all instances are standard nn.Linear). Available safe names: {all_safe[:15]}. "
            f"Check 'lora_target_modules' in your datasource config.",
            UserWarning, stacklevel=2
        )
        return configured_targets
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
    # Support both flat configs and nested addon_config.indexing.
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    effective_cfg = {**cfg, **indexing_cfg}
    model_id = effective_cfg.get("llm_model", MODEL_ID)
    entry = effective_cfg.get("model_library", {}).get(model_id)
    if entry:
        return entry["kv_num_layers"], entry["kv_num_heads"], entry["kv_head_dim"]

    # Auto-discover from loaded model's config
    if _model is not None and hasattr(_model, "config"):
        try:
            return _kv_shape_from_hf_config(_model.config)
        except AttributeError:
            pass

    # Legacy explicit fields
    for key in ("kv_num_layers", "kv_num_heads", "kv_head_dim"):
        if key in effective_cfg:
            return (effective_cfg.get("kv_num_layers", 35),
                    effective_cfg.get("kv_num_heads", 1),
                    effective_cfg.get("kv_head_dim", 256))

    # Defaults for known models when config is sparse (e.g. compute_kv=false)
    if "gemma-4" in model_id.lower() or "gemma4" in model_id.lower():
        return 35, 1, 256

    raise KeyError(
        f"Cannot determine KV shape for model {model_id!r}. "
        "Set kv_num_layers / kv_num_heads / kv_head_dim in config, "
        "or add a model_library entry."
    )
