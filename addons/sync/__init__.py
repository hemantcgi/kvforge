from addons.registry import AddonRegistry, AddonManifest
from addons.sync.config import SyncConfig

AddonRegistry.register(AddonManifest(
    name="sync",
    display_name="Scheduled Sync",
    description=(
        "Periodic connector polling with section-hash diffing, HITL governance, "
        "PII detection, and regression guard."
    ),
    config_schema=SyncConfig,
    requires=["indexing"],
))
