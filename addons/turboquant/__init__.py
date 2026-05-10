from addons.registry import AddonRegistry, AddonManifest
from addons.turboquant.config import TurboQuantConfig

AddonRegistry.register(AddonManifest(
    name="turboquant",
    display_name="TurboQuant KV Compression",
    description=(
        "Compresses per-token KV sequences on disk using TurboQuantProd "
        "(3-bit keys, 4-bit values) from arXiv:2504.19874. "
        "Reduces disk storage ~4.4x with negligible quality loss."
    ),
    config_schema=TurboQuantConfig,
    requires=["corpus_intelligence"],
))
