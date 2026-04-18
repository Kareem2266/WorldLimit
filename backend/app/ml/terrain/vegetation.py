"""
vegetation.py — deterministic tree placement for exports.

Mirrors `frontend/terrain.js::_vegetationPlan` + the placement loop in
`_buildVegetation`, but with a seeded RNG so the ZIP export matches the
browser preview (if same seed) and is reproducible across runs.

Output: a list of `TreeInstance` dicts giving position, rotation (Y), scale,
and tree type ("conifer" | "broadleaf"), in the same world-space the terrain
OBJ uses.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Literal, TypedDict

import numpy as np

TreeType = Literal["conifer", "broadleaf"]


class TreeInstance(TypedDict):
    type: TreeType
    x: float
    y: float
    z: float
    rotation_y: float
    scale: float


@dataclass(frozen=True)
class VegetationPlan:
    attempts: int
    conifer_ratio: float
    broadleaf_ratio: float
    scale_min: float
    scale_max: float


def plan_from_params(bio1: float, bio12: float) -> VegetationPlan:
    """Ecological "budget" — must stay in lockstep with frontend _vegetationPlan."""
    if bio1 >= 20 and bio12 < 300:
        return VegetationPlan(0, 0.0, 0.0, 1.0, 1.0)  # desert
    if bio1 >= 20 and bio12 < 1200:
        return VegetationPlan(180, 0.0, 1.0, 0.7, 1.2)  # savanna
    if bio1 >= 20 and bio12 < 2000:
        return VegetationPlan(2600, 0.0, 1.0, 1.0, 1.8)  # tropical forest
    if bio1 >= 20:
        return VegetationPlan(4500, 0.0, 1.0, 1.4, 2.4)  # rainforest
    if bio1 >= 8 and bio12 < 400:
        return VegetationPlan(150, 0.3, 0.7, 0.7, 1.2)  # temperate dry
    if bio1 >= 8:
        return VegetationPlan(1100, 0.5, 0.5, 0.8, 1.5)  # temperate forest
    if bio1 >= -2:
        return VegetationPlan(700, 1.0, 0.0, 0.8, 1.6)  # boreal
    return VegetationPlan(0, 0.0, 0.0, 1.0, 1.0)  # tundra


def _snow_line(bio1: float) -> float:
    if bio1 >= 20:
        return 1.3
    if bio1 >= 10:
        return 0.92
    if bio1 >= 0:
        return 0.72
    return 0.5


def _water_level_fraction(bio12: float, elev_std: float) -> float | None:
    if bio12 < 200:
        return None
    if bio12 > 1500 and elev_std < 150:
        return 0.42
    if bio12 > 1200 and elev_std < 200:
        return 0.28
    if bio12 > 1500:
        return 0.18
    if bio12 > 800:
        return 0.12
    return 0.06


def place_trees(
    heights_norm: np.ndarray,
    params: dict,
    plane_size: float,
    height_scale: float,
    seed: int = 42,
) -> list[TreeInstance]:
    """
    heights_norm : (N, N) float array in [0, 1] — same normalization the PNG uses.
    params       : dict with bio1, bio12, elev_std.
    plane_size   : horizontal span (meters/world-units) — matches terrain OBJ.
    height_scale : multiplier applied to heights_norm to get world Y (matches frontend).
    seed         : deterministic RNG seed.
    """
    assert heights_norm.ndim == 2 and heights_norm.shape[0] == heights_norm.shape[1]
    grid = heights_norm.shape[0]

    bio1 = float(params["bio1"])
    bio12 = float(params["bio12"])
    elev_std = float(params["elev_std"])

    plan = plan_from_params(bio1, bio12)
    if plan.attempts == 0:
        return []

    snow_y = _snow_line(bio1) * height_scale
    wl = _water_level_fraction(bio12, elev_std)
    water_y = -float("inf") if wl is None else wl * height_scale
    max_slope = 0.08

    rng = random.Random(seed)
    out: list[TreeInstance] = []

    for _ in range(plan.attempts):
        col = rng.randrange(1, grid - 1)
        row = rng.randrange(1, grid - 1)
        h = float(heights_norm[row, col])
        y = h * height_scale

        if y < water_y + 0.4:
            continue
        if y > snow_y - 1.5:
            continue

        dx = abs(float(heights_norm[row, col + 1]) - h)
        dz = abs(float(heights_norm[row + 1, col]) - h)
        if max(dx, dz) > max_slope:
            continue

        r = rng.random()
        pick_conifer = r < plan.conifer_ratio
        pick_broadleaf = (not pick_conifer) and r < plan.conifer_ratio + plan.broadleaf_ratio
        if not pick_conifer and not pick_broadleaf:
            continue

        scl = plan.scale_min + rng.random() * (plan.scale_max - plan.scale_min)
        rot_y = rng.random() * 2.0 * np.pi

        x = -plane_size / 2 + (col / (grid - 1)) * plane_size
        z = -plane_size / 2 + (row / (grid - 1)) * plane_size

        out.append(
            TreeInstance(
                type=("conifer" if pick_conifer else "broadleaf"),
                x=float(x),
                y=float(y - 0.1),
                z=float(z),
                rotation_y=float(rot_y),
                scale=float(scl),
            )
        )

    return out


def partition_by_type(trees: Iterable[TreeInstance]) -> dict[TreeType, list[TreeInstance]]:
    buckets: dict[TreeType, list[TreeInstance]] = {"conifer": [], "broadleaf": []}
    for t in trees:
        buckets[t["type"]].append(t)
    return buckets
