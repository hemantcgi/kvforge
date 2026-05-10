from pydantic import BaseModel


class MultimodalConfig(BaseModel):
    """Configuration for the multimodal (image) indexing and inference addon."""

    image_collection_suffix: str = "_images"
    image_store_dir: str = ""
    multimodal_model: str = "llava-hf/llava-1.5-7b-hf"
    clip_model: str = "openai/clip-vit-base-patch32"
    image_kv_inference: bool = False
