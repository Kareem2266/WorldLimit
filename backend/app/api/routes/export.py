"""
export.py — synchronous "download this world as a Godot/Unity bundle" endpoint.

Given a prompt (+ optional seed), re-runs the MLP + Perlin generator and
packages the result into a ZIP suitable for importing into either engine.

Because generation is ~2s of CPU, the heightmap synth and the bundling step
are both pushed to a worker thread so the event loop stays responsive.

Kept the old job-based route commented in git history; the /api/jobs pathway
is untouched for Stage 6.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.ml.inference import predict_terrain
from app.ml.terrain.export_bundle import build_world_bundle
from app.ml.terrain.generator import generate_heightmap
from app.schemas.world import GenerateRequest

router = APIRouter()


@router.post("/export")
async def export_world(body: GenerateRequest) -> Response:
    """Prompt → ZIP of heightmap + terrain OBJ + trees + importer scripts."""
    try:
        params = predict_terrain(body.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    seed = body.seed if body.seed is not None else 42
    heights = await asyncio.to_thread(generate_heightmap, params, seed=seed)
    bundle = await asyncio.to_thread(build_world_bundle, body.prompt, seed, heights, params)

    return Response(
        content=bundle.zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{bundle.filename}"',
            "X-WorldLimit-Slug": bundle.slug,
            "X-WorldLimit-Tree-Count": str(bundle.tree_count),
        },
    )
