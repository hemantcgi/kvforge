from addons.registry import AddonRegistry, AddonManifest
from addons.training.config import TrainingConfig

AddonRegistry.register(AddonManifest(
    name="training",
    display_name="LoRA Training + PRS",
    description=(
        "Fine-tune the LLM on FAQ pairs using tier-weighted LoRA. "
        "Evaluates Parametric Readiness Score (PRS) and gates phase advancement."
    ),
    config_schema=TrainingConfig,
    requires=["indexing", "inference"],
))
