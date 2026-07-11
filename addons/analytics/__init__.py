from addons.registry import AddonRegistry, AddonManifest
from addons.analytics.config import AnalyticsConfig

AddonRegistry.register(AddonManifest(
    name="analytics",
    display_name="Flywheel Analytics",
    description=(
        "Tracks token costs, latency baselines, and query quality trends "
        "to drive the continuous improvement flywheel."
    ),
    config_schema=AnalyticsConfig,
    requires=[],
))
