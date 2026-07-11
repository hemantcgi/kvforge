from pydantic import BaseModel, Field


class CorpusIntelligenceConfig(BaseModel):
    alpha: float = Field(0.33, ge=0, le=1, description="Weight for access_score")
    beta:  float = Field(0.33, ge=0, le=1, description="Weight for uniqueness_score")
    gamma: float = Field(0.34, ge=0, le=1, description="Weight for coverage_score")
    enhanced_tier_threshold:     float = Field(0.70, ge=0, le=1)
    archive_candidate_threshold: float = Field(0.20, ge=0, le=1)
    uniqueness_floor:            float = Field(0.15, ge=0, le=1)
    reinstatement_threshold:     int   = Field(5, ge=1)
    archive_backend:             str   = "local"
    archive_dir:                 str   = "data/archive"
    per_token_kv_dir:            str   = "data/per_token_kv"
