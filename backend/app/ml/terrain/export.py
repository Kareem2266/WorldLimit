"""
export.py — persist heightmaps as 16-bit grayscale PNGs.

16 bits (vs 8) gives 65,536 distinct height levels instead of 256, which
prevents visible "terracing" when Phase 5 converts the image to a 3D mesh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_heightmap_png(heights: np.ndarray, out_path: Path | str) -> Path:
    """Normalize a float heightmap to 16-bit grayscale and save as PNG.

    The absolute elevation range is remapped so the lowest pixel becomes
    black (0) and the highest becomes white (65535). The true min/max
    should be stored alongside if you need to recover real meters later
    (Phase 5 will; for now the PNG is just for visual inspection).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if heights.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {heights.shape}")

    h_min = float(heights.min())
    h_max = float(heights.max())
    if h_max - h_min < 1e-6:
        normalized = np.zeros_like(heights, dtype=np.uint16)
    else:
        normalized = ((heights - h_min) / (h_max - h_min) * 65535.0).astype(np.uint16)

    Image.fromarray(normalized, mode="I;16").save(out_path, format="PNG")
    return out_path
