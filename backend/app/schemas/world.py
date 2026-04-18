from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=3,
        max_length=1000,
    )
    seed: int | None = Field(default=None)


class TerrainParamsOut(BaseModel):
    elev_mean: float
    elev_std: float
    slope_mean: float
    bio1: float
    bio4: float
    bio12: float


class GenerateResponse(BaseModel):
    prompt: str
    params: TerrainParamsOut
    heightmap_url: str
    heightmap_min_m: float
    heightmap_max_m: float


class WorldOut(BaseModel):
    id: uuid.UUID
    prompt: str
    status: str
    params: dict[str, Any] | None
    created_at: datetime
