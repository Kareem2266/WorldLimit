"""
generate.py — synchronous prompt → heightmap PNG.

Phase 5 pipeline (no DB, no queue):
    predict_terrain(prompt)     # MLP inference, ~0.1s
    generate_heightmap(params)  # Perlin noise, ~1.5s  (CPU-bound → to_thread)
    save_heightmap_png(heights) # PIL write, ~0.05s

Returns the 6 predicted params plus a URL the browser can fetch the PNG from.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.ml.inference import predict_terrain
from app.ml.terrain.export import save_heightmap_png
from app.ml.terrain.generator import generate_heightmap
from app.schemas.world import GenerateRequest, GenerateResponse, TerrainParamsOut

router = APIRouter()

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
HEIGHTMAP_DIR = DATA_DIR / "output" / "heightmaps"
STATIC_URL_PREFIX = "/static/heightmaps"


def _slugify(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:60] or "world"


@router.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    try:
        params = predict_terrain(body.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    seed = body.seed if body.seed is not None else 42
    heights = await asyncio.to_thread(generate_heightmap, params, seed=seed)

    slug = _slugify(body.prompt)
    out_path = HEIGHTMAP_DIR / f"{slug}.png"
    await asyncio.to_thread(save_heightmap_png, heights, out_path)

    return GenerateResponse(
        prompt=body.prompt,
        params=TerrainParamsOut(**params),
        heightmap_url=f"{STATIC_URL_PREFIX}/{slug}.png",
        heightmap_min_m=float(heights.min()),
        heightmap_max_m=float(heights.max()),
    )
