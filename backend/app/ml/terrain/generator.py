"""
generator.py — synthesize a heightmap from terrain parameters.

Uses multi-octave Perlin noise to produce a `size × size` grid of elevations
in meters. The 6 params from Phase 3 shape the noise:

    elev_mean   → base altitude (added to every pixel)
    elev_std    → amplitude of variation (how tall bumps get)
    slope_mean  → persistence (how much fine detail vs smooth rolling)
    bio1/4/12   → unused here; reserved for Phase 5 biome coloring & erosion
"""
from __future__ import annotations

import numpy as np
from noise import pnoise2

from app.ml.inference import TerrainParams

DEFAULT_SIZE = 512
DEFAULT_OCTAVES = 6
DEFAULT_SCALE = 4.0
DEFAULT_LACUNARITY = 2.0


def _slope_to_persistence(slope_mean: float) -> float:
    """Map predicted slope (~0–25°) to Perlin persistence (~0.30–0.65).

    Higher persistence = fine octaves keep more amplitude = rougher terrain.
    """
    p = 0.30 + (max(slope_mean, 0.0) / 25.0) * 0.35
    return float(np.clip(p, 0.30, 0.70))


def generate_heightmap(
    params: TerrainParams,
    size: int = DEFAULT_SIZE,
    octaves: int = DEFAULT_OCTAVES,
    scale: float = DEFAULT_SCALE,
    seed: int = 42,
) -> np.ndarray:
    """Return a (size, size) float32 heightmap in meters."""
    elev_mean = float(params["elev_mean"])
    elev_std = max(float(params["elev_std"]), 0.0)
    persistence = _slope_to_persistence(float(params["slope_mean"]))

    amplitude = elev_std * 2.5

    heights = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        ny = (i / size) * scale
        for j in range(size):
            nx = (j / size) * scale
            raw = pnoise2(
                nx,
                ny,
                octaves=octaves,
                persistence=persistence,
                lacunarity=DEFAULT_LACUNARITY,
                repeatx=1024,
                repeaty=1024,
                base=seed,
            )
            heights[i, j] = elev_mean + raw * amplitude

    np.maximum(heights, 0.0, out=heights)
    return heights


if __name__ == "__main__":
    import os
    import re
    import sys
    import time
    from pathlib import Path

    from app.ml.inference import predict_terrain
    from app.ml.terrain.export import save_heightmap_png

    prompt = " ".join(sys.argv[1:]) or "snowy alpine valley"
    print(f"Prompt: {prompt!r}")

    params = predict_terrain(prompt)
    print("Predicted terrain params:")
    for k, v in params.items():
        print(f"  {k:<12} {v:>8.1f}")

    print(f"\nGenerating {DEFAULT_SIZE}×{DEFAULT_SIZE} heightmap...")
    t0 = time.time()
    heights = generate_heightmap(params)
    print(f"Done in {time.time() - t0:.1f}s")

    print("\nHeightmap stats:")
    print(f"  shape: {heights.shape}")
    print(f"  min:   {heights.min():>8.1f} m")
    print(f"  max:   {heights.max():>8.1f} m")
    print(f"  mean:  {heights.mean():>8.1f} m")
    print(f"  std:   {heights.std():>8.1f} m")

    data_dir = Path(os.getenv("DATA_DIR", "/data"))
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:60]
    out_path = data_dir / "output" / "heightmaps" / f"{slug}.png"
    save_heightmap_png(heights, out_path)
    print(f"\nSaved heightmap → {out_path}")
