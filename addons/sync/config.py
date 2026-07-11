from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class SyncConfig(BaseModel):
    """Configuration for the scheduled sync + connector management addon."""

    interval_minutes: int = 60

    # Connector selection and its own config dict
    connector: str = ""                         # e.g. "gdrive", "s3", "sharepoint"
    connector_config: dict = Field(default_factory=dict)  # connector-specific params

    # Enterprise governance
    tenant_id: str = "default"
    hitl_mode: Literal["blocking", "non-blocking", "auto"] = "auto"
    hitl_sensitivity: Literal["high", "normal"] = "normal"
    pii_detection_enabled: bool = True
    allowed_pii_categories: list[str] = Field(default_factory=list)
    pii_rejection_threshold: int = 3
    local_mirror_path: str = ""

    # Regression guard
    sync_regression_mode: str = "pct"
    sync_regression_pct_threshold: float = 0.10
    sync_regression_tier_threshold: float = 0.15
