from addons.registry import AddonRegistry, AddonManifest
from addons.monitoring.config import MonitoringConfig

AddonRegistry.register(AddonManifest(
    name="monitoring",
    display_name="Monitoring Dashboard",
    description=(
        "FastAPI dashboard showing phase, PRS history, tier breakdown, "
        "KV coverage heatmap, and A/B query panel."
    ),
    config_schema=MonitoringConfig,
    requires=[],
))
