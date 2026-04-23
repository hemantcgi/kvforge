"""CLIP-based embedder for images and text queries.

Used for the separate image collection. Both encode_image and encode_text
return 512-dim vectors (CLIP ViT-B/32), enabling cosine similarity between
text queries and image embeddings in the same vector space.
"""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self._model = CLIPModel.from_pretrained(model_name)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained(model_name)
        # Infer dim from model config
        self._dim: int = self._model.config.projection_dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode_image(self, image_path: str) -> list[float]:
        with Image.open(image_path) as img:
            inputs = self._processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].tolist()

    def encode_text(self, text: str) -> list[float]:
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].tolist()
