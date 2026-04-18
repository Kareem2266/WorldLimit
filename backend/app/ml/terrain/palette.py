"""
palette.py — bake a per-vertex colour texture for the exported terrain.

Mirrors `frontend/terrain.js::_pickPalette` + `_colorForHeight` so the baked
PNG looks identical to the browser preview when sampled through the UVs the
OBJ already carries.

Output: an (N, N, 3) uint8 RGB array where pixel (row, col) is the colour of
terrain vertex (row, col).
"""
from __future__ import annotations

import numpy as np

RGB = tuple[float, float, float]


def _pick_palette(bio1: float, bio12: float) -> dict:
    if bio1 >= 20:
        if bio12 < 300:
            lowland: RGB = (0.83, 0.69, 0.45)   # desert
            midland: RGB = (0.72, 0.47, 0.36)
        elif bio12 < 1200:
            lowland = (0.66, 0.72, 0.34)         # savanna
            midland = (0.48, 0.41, 0.23)
        else:
            lowland = (0.17, 0.42, 0.23)         # jungle
            midland = (0.24, 0.35, 0.20)
    elif bio1 >= 8:
        if bio12 < 400:
            lowland = (0.62, 0.65, 0.30)         # dry grassland
            midland = (0.46, 0.45, 0.24)
        else:
            lowland = (0.30, 0.55, 0.28)         # temperate forest
            midland = (0.20, 0.36, 0.20)
    elif bio1 >= -2:
        lowland = (0.22, 0.40, 0.22)             # boreal
        midland = (0.26, 0.32, 0.24)
    else:
        lowland = (0.50, 0.48, 0.42)             # tundra
        midland = (0.58, 0.55, 0.50)

    if bio1 >= 20:
        snow_line = 1.3
    elif bio1 >= 10:
        snow_line = 0.92
    elif bio1 >= 0:
        snow_line = 0.72
    else:
        snow_line = 0.5

    return {
        "lowland": lowland,
        "midland": midland,
        "rock": (0.42, 0.40, 0.38),
        "snow": (0.94, 0.95, 0.97),
        "snow_line": snow_line,
    }


def _colour_for_height(h: float, pal: dict) -> RGB:
    bands = [
        (0.00, pal["lowland"]),
        (0.35, pal["midland"]),
        (max(pal["snow_line"] - 0.12, 0.5), pal["rock"]),
        (pal["snow_line"], pal["snow"]),
    ]
    for i in range(1, len(bands)):
        if h <= bands[i][0]:
            t0, c0 = bands[i - 1]
            t1, c1 = bands[i]
            span = max(t1 - t0, 1e-6)
            t = min(max((h - t0) / span, 0.0), 1.0)
            return (
                c0[0] + (c1[0] - c0[0]) * t,
                c0[1] + (c1[1] - c0[1]) * t,
                c0[2] + (c1[2] - c0[2]) * t,
            )
    return pal["snow"]


def bake_colour_texture(heights_norm: np.ndarray, params: dict) -> np.ndarray:
    """Return (N, N, 3) uint8 array matching the browser preview colours."""
    bio1 = float(params["bio1"])
    bio12 = float(params["bio12"])
    pal = _pick_palette(bio1, bio12)

    n = heights_norm.shape[0]
    low = np.array(pal["lowland"], dtype=np.float32)
    mid = np.array(pal["midland"], dtype=np.float32)
    rock = np.array(pal["rock"], dtype=np.float32)
    snow = np.array(pal["snow"], dtype=np.float32)

    # Vectorised band interpolation — same math as _colour_for_height but over
    # the whole (N, N) array at once.
    h = heights_norm.astype(np.float32)
    t_mid = 0.35
    t_rock = max(pal["snow_line"] - 0.12, 0.5)
    t_snow = pal["snow_line"]

    # band 1: [0, t_mid] lowland → midland
    t1 = np.clip(h / max(t_mid, 1e-6), 0.0, 1.0)
    c1 = low + (mid - low) * t1[..., None]
    # band 2: [t_mid, t_rock] midland → rock
    t2 = np.clip((h - t_mid) / max(t_rock - t_mid, 1e-6), 0.0, 1.0)
    c2 = mid + (rock - mid) * t2[..., None]
    # band 3: [t_rock, t_snow] rock → snow
    t3 = np.clip((h - t_rock) / max(t_snow - t_rock, 1e-6), 0.0, 1.0)
    c3 = rock + (snow - rock) * t3[..., None]

    out = np.empty((n, n, 3), dtype=np.float32)
    m1 = h <= t_mid
    m2 = (h > t_mid) & (h <= t_rock)
    m3 = (h > t_rock) & (h <= t_snow)
    m4 = h > t_snow
    out[m1] = c1[m1]
    out[m2] = c2[m2]
    out[m3] = c3[m3]
    out[m4] = snow

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
