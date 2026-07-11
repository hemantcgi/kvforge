from addons.registry import AddonRegistry, AddonManifest
from addons.inference.config import InferenceConfig

AddonRegistry.register(AddonManifest(
    name="inference",
    display_name="KV Inference",
    description=(
        "Phase-aware query routing: text-in-context (Phase 1), "
        "KV tensor injection (Phase 2), or confidence-gated parametric (Phase 3)."
    ),
    config_schema=InferenceConfig,
    requires=["indexing"],
))
