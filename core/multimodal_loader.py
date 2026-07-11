"""MultimodalLLM protocol and LLaVALoader implementation.

LLaVALoader is a singleton — call LLaVALoader(cfg) and the model loads once.
"""
import numpy as np
from typing import Protocol, runtime_checkable

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

import core.kv_utils as kv_utils

_model = None
_processor = None
_loaded_model_name = None


@runtime_checkable
class MultimodalLLM(Protocol):
    def encode_image_kv(self, image_path: str) -> np.ndarray: ...
    def caption(self, image_path: str) -> str: ...

    @property
    def kv_shape(self) -> tuple[int, int, int]: ...


class LLaVALoader:
    """Loads a LLaVA-style multimodal LLM and computes image KV tensors.

    Uses a module-level singleton so the model loads once per process.
    """

    def __init__(self, cfg: dict) -> None:
        global _model, _processor, _loaded_model_name
        model_name = cfg.get("multimodal_model", "llava-hf/llava-1.5-7b-hf")
        if _model is None or _loaded_model_name != model_name:
            _model = LlavaForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
            _model.eval()
            _processor = AutoProcessor.from_pretrained(model_name)
            _loaded_model_name = model_name
        self._model = _model
        self._processor = _processor

    @property
    def kv_shape(self) -> tuple[int, int, int]:
        lm_cfg = self._model.language_model.config
        num_layers = lm_cfg.num_hidden_layers
        num_kv_heads = getattr(lm_cfg, "num_key_value_heads", lm_cfg.num_attention_heads)
        head_dim = lm_cfg.hidden_size // lm_cfg.num_attention_heads
        return (num_layers, num_kv_heads, head_dim)

    def encode_image_kv(self, image_path: str) -> np.ndarray:
        """Run image through LLaVA and mean-pool the KV tensors.

        Returns float16 array of shape [num_layers, 2, num_kv_heads, head_dim].
        """
        with Image.open(image_path) as img:
            inputs = self._processor(
                text="<image>",
                images=img,
                return_tensors="pt",
            )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, use_cache=True)
        return kv_utils.mean_pool_kv(outputs.past_key_values)

    def caption(self, image_path: str) -> str:
        """Generate a text caption for an image."""
        with Image.open(image_path) as img:
            inputs = self._processor(
                text="<image>\nDescribe this image concisely.",
                images=img,
                return_tensors="pt",
            )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        return self._processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
