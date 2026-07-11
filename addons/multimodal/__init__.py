from addons.registry import AddonRegistry, AddonManifest
from addons.multimodal.config import MultimodalConfig

AddonRegistry.register(AddonManifest(
    name="multimodal",
    display_name="Multimodal (Images)",
    description=(
        "Index and query images alongside text using CLIP embeddings "
        "and LLaVA for visual question answering."
    ),
    config_schema=MultimodalConfig,
    requires=["indexing"],
))
