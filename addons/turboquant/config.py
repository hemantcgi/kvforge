from pydantic import BaseModel, Field
from typing import Literal


class TurboQuantConfig(BaseModel):
    key_bits: Literal[2, 3] = 3
    value_bits: Literal[2, 4] = 4
    group_size: int = Field(32, ge=8)
    seed: int = 42
