"""
mesh_obj.py — OBJ writers for the terrain surface + tree templates.

OBJ is the lowest-common-denominator 3D format: both Godot (4.x) and Unity
import .obj files out of the box. Materials come via a sidecar .mtl file.

Axis convention: we emit right-handed, Y-up meshes (matches Three.js and
Godot). Unity is left-handed — its OBJ importer mirrors the X axis on
import, so a mesh that looks correct in Godot will also look correct in
Unity (no manual flipping required).
"""
from __future__ import annotations

import math
from io import StringIO

import numpy as np


# ---------- terrain ----------

def terrain_to_obj(
    heights_norm: np.ndarray,
    plane_size: float,
    height_scale: float,
    mtl_name: str = "terrain.mtl",
    material: str = "terrain_mat",
) -> str:
    """Turn a normalized heightmap into an OBJ string (+UVs, +triangles).

    heights_norm : (N, N) float array in [0, 1] (same as the PNG R channel).
    plane_size   : horizontal span, world-units (X and Z).
    height_scale : multiplier applied to heights_norm to get Y.

    Vertices are indexed row-major: i = row*N + col (same as the frontend).
    """
    assert heights_norm.ndim == 2 and heights_norm.shape[0] == heights_norm.shape[1]
    n = heights_norm.shape[0]
    half = plane_size / 2.0
    step = plane_size / (n - 1)

    buf = StringIO()
    buf.write("# WorldLimit terrain export\n")
    buf.write(f"mtllib {mtl_name}\n")
    buf.write("o terrain\n")

    for row in range(n):
        z = -half + row * step
        for col in range(n):
            x = -half + col * step
            y = float(heights_norm[row, col]) * height_scale
            buf.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")

    for row in range(n):
        v = row / (n - 1)
        for col in range(n):
            u = col / (n - 1)
            buf.write(f"vt {u:.5f} {v:.5f}\n")

    # compute vertex normals via finite differences
    dy_dx = np.zeros_like(heights_norm, dtype=np.float32)
    dy_dz = np.zeros_like(heights_norm, dtype=np.float32)
    dy_dx[:, 1:-1] = (heights_norm[:, 2:] - heights_norm[:, :-2]) * 0.5
    dy_dx[:, 0] = heights_norm[:, 1] - heights_norm[:, 0]
    dy_dx[:, -1] = heights_norm[:, -1] - heights_norm[:, -2]
    dy_dz[1:-1, :] = (heights_norm[2:, :] - heights_norm[:-2, :]) * 0.5
    dy_dz[0, :] = heights_norm[1, :] - heights_norm[0, :]
    dy_dz[-1, :] = heights_norm[-1, :] - heights_norm[-2, :]

    dx_scale = height_scale / step
    nx = -dy_dx * dx_scale
    nz = -dy_dz * dx_scale
    ny = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= length
    ny /= length
    nz /= length

    for row in range(n):
        for col in range(n):
            buf.write(f"vn {nx[row, col]:.4f} {ny[row, col]:.4f} {nz[row, col]:.4f}\n")

    buf.write(f"usemtl {material}\n")
    buf.write("s 1\n")

    # OBJ is 1-indexed
    def idx(r: int, c: int) -> int:
        return r * n + c + 1

    for row in range(n - 1):
        for col in range(n - 1):
            a = idx(row, col)
            b = idx(row, col + 1)
            c = idx(row + 1, col)
            d = idx(row + 1, col + 1)
            buf.write(f"f {a}/{a}/{a} {c}/{c}/{c} {b}/{b}/{b}\n")
            buf.write(f"f {b}/{b}/{b} {c}/{c}/{c} {d}/{d}/{d}\n")

    return buf.getvalue()


def terrain_mtl(
    material: str = "terrain_mat",
    diffuse_texture: str | None = "terrain_color.png",
) -> str:
    # Godot 4 PBR ignores Ka (ambient) and prints a warning, so we omit it.
    # `map_Kd` points at the baked per-vertex colour texture so both Godot and
    # Unity reproduce the browser's biome palette automatically.
    lines = [
        f"newmtl {material}",
        "Kd 1.0 1.0 1.0",
        "Ks 0.05 0.05 0.05",
        "Ns 8.0",
        "d 1.0",
        "illum 2",
    ]
    if diffuse_texture:
        lines.append(f"map_Kd {diffuse_texture}")
    return "\n".join(lines) + "\n"


# ---------- trees ----------
#
# Simple stylised templates (must stay visually close to the Three.js shapes
# built in `terrain.js::_getConiferGeom` / `_getBroadleafGeom`).

def _cylinder(radius_top: float, radius_bottom: float, height: float, segs: int, y_base: float):
    verts = []
    tris = []
    # base ring
    for i in range(segs):
        a = (i / segs) * 2.0 * math.pi
        verts.append((radius_bottom * math.cos(a), y_base, radius_bottom * math.sin(a)))
    # top ring
    for i in range(segs):
        a = (i / segs) * 2.0 * math.pi
        verts.append((radius_top * math.cos(a), y_base + height, radius_top * math.sin(a)))
    bot_center = len(verts); verts.append((0.0, y_base, 0.0))
    top_center = len(verts); verts.append((0.0, y_base + height, 0.0))
    for i in range(segs):
        j = (i + 1) % segs
        tris.append((i, j, i + segs))
        tris.append((j, j + segs, i + segs))
    for i in range(segs):
        j = (i + 1) % segs
        tris.append((bot_center, j, i))
        tris.append((top_center, i + segs, j + segs))
    return verts, tris


def _cone(radius: float, height: float, segs: int, y_base: float):
    verts = []
    tris = []
    for i in range(segs):
        a = (i / segs) * 2.0 * math.pi
        verts.append((radius * math.cos(a), y_base, radius * math.sin(a)))
    apex = len(verts); verts.append((0.0, y_base + height, 0.0))
    center = len(verts); verts.append((0.0, y_base, 0.0))
    for i in range(segs):
        j = (i + 1) % segs
        tris.append((i, j, apex))
        tris.append((center, j, i))
    return verts, tris


def _sphere(radius: float, wseg: int, hseg: int, y_center: float):
    verts = []
    tris = []
    for i in range(hseg + 1):
        phi = math.pi * (i / hseg)
        for j in range(wseg + 1):
            theta = 2.0 * math.pi * (j / wseg)
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi) + y_center
            z = radius * math.sin(phi) * math.sin(theta)
            verts.append((x, y, z))
    w = wseg + 1
    for i in range(hseg):
        for j in range(wseg):
            a = i * w + j
            b = a + 1
            c = a + w
            d = c + 1
            tris.append((a, c, b))
            tris.append((b, c, d))
    return verts, tris


def _merge_groups(groups):
    """groups: list of (name, verts, tris, mtl_name). Returns combined OBJ text."""
    buf = StringIO()
    buf.write("# WorldLimit tree template\n")
    buf.write("mtllib tree.mtl\n")
    offset = 0
    for name, verts, tris, mtl in groups:
        buf.write(f"o {name}\n")
        for v in verts:
            buf.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        buf.write(f"usemtl {mtl}\n")
        buf.write("s 1\n")
        for t in tris:
            a, b, c = t[0] + offset + 1, t[1] + offset + 1, t[2] + offset + 1
            buf.write(f"f {a} {b} {c}\n")
        offset += len(verts)
    return buf.getvalue()


def conifer_obj() -> str:
    trunk_v, trunk_t = _cylinder(0.14, 0.22, 1.6, 6, y_base=0.0)
    foliage_v, foliage_t = _cone(1.1, 3.3, 8, y_base=1.1)
    return _merge_groups([
        ("trunk", trunk_v, trunk_t, "bark"),
        ("foliage", foliage_v, foliage_t, "needles"),
    ])


def broadleaf_obj() -> str:
    trunk_v, trunk_t = _cylinder(0.17, 0.25, 1.8, 6, y_base=0.0)
    foliage_v, foliage_t = _sphere(1.3, 8, 6, y_center=2.6)
    return _merge_groups([
        ("trunk", trunk_v, trunk_t, "bark"),
        ("foliage", foliage_v, foliage_t, "leaves"),
    ])


def tree_mtl() -> str:
    # No Ka — Godot 4 PBR ignores it and prints a warning.
    return (
        "newmtl bark\n"
        "Kd 0.36 0.22 0.12\n"
        "Ks 0.02 0.02 0.02\nNs 4.0\nd 1.0\nillum 2\n\n"
        "newmtl needles\n"
        "Kd 0.12 0.36 0.18\n"
        "Ks 0.02 0.02 0.02\nNs 4.0\nd 1.0\nillum 2\n\n"
        "newmtl leaves\n"
        "Kd 0.22 0.44 0.20\n"
        "Ks 0.02 0.02 0.02\nNs 4.0\nd 1.0\nillum 2\n"
    )
