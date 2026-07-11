from addons.registry import AddonRegistry, AddonManifest
from addons.background.config import BackgroundConfig

AddonRegistry.register(AddonManifest(
    name="background",
    display_name="Background KV Recompute",
    description=(
        "Daemon threads that recompute stale KV tensors after LoRA updates "
        "and periodically flush access counts to the vector store."
    ),
    config_schema=BackgroundConfig,
    requires=["indexing", "inference"],
))
