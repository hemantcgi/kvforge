from addons.registry import AddonRegistry, AddonManifest
from addons.model_scout.config import ModelScoutConfig

AddonRegistry.register(AddonManifest(
    name="model_scout",
    display_name="ModelScout",
    description=(
        "Automated model evaluation pipeline that benchmarks candidate LLMs "
        "on a sample corpus and selects the best fit for the use case."
    ),
    config_schema=ModelScoutConfig,
    requires=[],
))
