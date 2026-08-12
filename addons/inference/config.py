from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    """Configuration for the KV inference pipeline addon."""

    top_k: int = 5
    llm_model: str
    quantization: Literal["none", "4bit", "8bit"] = "4bit"
    hf_token: str = ""
    vllm_url: str = ""
    vllm_model: str = ""
    max_new_tokens: int = 256
    gate_threshold: float = 0.75
    parametric_eligibility_threshold: float = 0.85  # Phase-2 hard gate: min similarity-to-known-good
    model_library: dict = Field(default_factory=dict)
    query_log_db: str = "query_log.db"

    # Dynamic PRS / brownfield routing
    deployment_mode: Literal["greenfield", "brownfield", "auto"] = "auto"
    difficulty_estimator: str = "intra_cluster_distance"
    min_cluster_samples_for_adaptation: int = 10
    prs_stability_window: int = 3
    brownfield_routing_threshold: float = 0.85
    brownfield_confidence_floor: float = 0.80
    brownfield_coverage_target: float = 0.70
    realtime_requery_window_minutes: int = 10
