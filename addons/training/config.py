from __future__ import annotations
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for the LoRA training + PRS evaluation addon."""

    # LoRA hyper-parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"]
    )
    lora_dropout: float = 0.05
    lora_epochs: int = 3
    lora_lr: float = 0.0002

    # Storage paths
    checkpoint_dir: str  # required
    replay_db: str       # required

    # PRS thresholds and weights
    prs_threshold: float = 0.50  # DEPRECATED: superseded by phase2/phase3_advance_threshold
    phase2_advance_threshold: float = 0.30
    phase3_advance_threshold: float = 0.55
    prs_weights: dict = Field(
        default_factory=lambda: {"accuracy": 0.7, "calibration": 0.15, "consistency": 0.15}
    )
    prs_advancement_threshold: float = 0.72
    prs_regression_threshold: float = 0.25
    prs_auto_weight: bool = True
    prs_signal_weights: dict = Field(
        default_factory=lambda: {"faq": 0.4, "vdb": 0.4, "realtime": 0.2}
    )
    prs_stability_window: int = 3
    known_good_accuracy_threshold: float = 0.5

    # FAQ schema keys
    faq_question_key: str = "question"
    faq_answer_key: str = "answer"
