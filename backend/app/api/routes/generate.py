"""
generate.py — synchronous prompt → heightmap PNG.

Pipeline:
    predict_terrain(prompt)     # MLP inference, ~0.1s (Redis-cached)
    generate_heightmap(params)  # Perlin noise, ~1.5s  (CPU-bound → to_thread)
    save_heightmap_png(heights) # PIL write, ~0.05s

The full response is Redis-cached by (prompt, seed) for an hour. A cache hit
skips the Perlin step entirely — the PNG already lives on disk from the first
time this (prompt, seed) was generated. If the disk file is missing (e.g.
/data got cleared), we fall through and regenerate.

Returns the 6 predicted params plus a URL the browser can fetch the PNG from.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.cache import cache_get_json, cache_set_json
from app.ml.inference import predict_terrain
from app.ml.terrain.export import save_heightmap_png
from app.ml.terrain.generator import generate_heightmap
from app.schemas.world import GenerateRequest, GenerateResponse, TerrainParamsOut

router = APIRouter()

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
HEIGHTMAP_DIR = DATA_DIR / "output" / "heightmaps"
STATIC_URL_PREFIX = "/static/heightmaps"

HEIGHTMAP_CACHE_TTL = 60 * 60  # 1 hour
CACHE_VERSION = "v1"


def _slugify(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:60] or "world"


def _cache_key(prompt: str, seed: int) -> str:
    digest = hashlib.sha1(prompt.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"heightmap:{CACHE_VERSION}:{digest}:{seed}"


@router.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    seed = body.seed if body.seed is not None else 42
    slug = _slugify(body.prompt)
    out_path = HEIGHTMAP_DIR / f"{slug}_{seed}.png"

    cache_key = _cache_key(body.prompt, seed)
    cached = cache_get_json(cache_key)
    if cached is not None and out_path.exists():
        return GenerateResponse(**cached)

    try:
        params = predict_terrain(body.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    heights = await asyncio.to_thread(generate_heightmap, params, seed=seed)
    await asyncio.to_thread(save_heightmap_png, heights, out_path)

    response = GenerateResponse(
        prompt=body.prompt,
        params=TerrainParamsOut(**params),
        heightmap_url=f"{STATIC_URL_PREFIX}/{slug}_{seed}.png",
        heightmap_min_m=float(heights.min()),
        heightmap_max_m=float(heights.max()),
    )
    cache_set_json(cache_key, response.model_dump(), ttl_seconds=HEIGHTMAP_CACHE_TTL)
    return response
