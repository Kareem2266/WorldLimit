"""
export_bundle.py — package a generated world into a ZIP that imports cleanly
into Godot 4.x and Unity (tested with 2022.3+).

Layout of the ZIP:

    world_<slug>/
        heightmap.png          16-bit grayscale PNG (512x512)  — works in Godot & Unity
        heightmap.raw          16-bit little-endian raw (513x513) — Unity Terrain
        terrain.obj            full terrain mesh + UVs, right-handed Y-up
        terrain.mtl            sidecar material
        tree_conifer.obj       template conifer mesh
        tree_broadleaf.obj     template broadleaf mesh
        tree.mtl               shared tree material
        trees.json             deterministic per-instance transforms
        metadata.json          params, dimensions, biome info, coord conventions
        import_godot.gd        drop-in GDScript importer (Godot 4.x)
        import_unity.cs        drop-in C# importer (Unity)
        README.md              step-by-step for both engines
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.inference import TerrainParams
from app.ml.terrain.mesh_obj import (
    broadleaf_obj,
    conifer_obj,
    terrain_mtl,
    terrain_to_obj,
    tree_mtl,
)
from app.ml.terrain.palette import bake_colour_texture
from app.ml.terrain.vegetation import partition_by_type, place_trees


# Visual constants — kept in lock-step with frontend/terrain.js so the export
# matches what the user sees in the browser preview.
PLANE_SIZE = 200.0
EXPORT_GRID = 256                # same downsampling the frontend uses
HEIGHT_SCALE_REF = 60.0
REFERENCE_RANGE_M = 700.0
HEIGHT_SCALE_MIN = 20.0
HEIGHT_SCALE_MAX = 110.0


@dataclass(frozen=True)
class ExportResult:
    slug: str
    zip_bytes: bytes
    tree_count: int
    filename: str


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "world"


def _compute_height_scale(min_m: float, max_m: float) -> float:
    rng = max(max_m - min_m, 0.0)
    raw = HEIGHT_SCALE_REF * (rng / REFERENCE_RANGE_M)
    return float(np.clip(raw, HEIGHT_SCALE_MIN, HEIGHT_SCALE_MAX))


def _normalize(heights: np.ndarray) -> tuple[np.ndarray, float, float]:
    h_min = float(heights.min())
    h_max = float(heights.max())
    if h_max - h_min < 1e-6:
        return np.zeros_like(heights, dtype=np.float32), h_min, h_max
    norm = ((heights - h_min) / (h_max - h_min)).astype(np.float32)
    return norm, h_min, h_max


def _downsample(arr: np.ndarray, target: int) -> np.ndarray:
    """Block-mean downsample a (N,N) array to (target,target). N must be divisible by target."""
    n = arr.shape[0]
    if n == target:
        return arr
    if n % target != 0:
        pil = Image.fromarray(arr).resize((target, target), Image.BILINEAR)
        return np.array(pil, dtype=arr.dtype)
    factor = n // target
    return arr.reshape(target, factor, target, factor).mean(axis=(1, 3)).astype(arr.dtype)


def _heightmap_png_bytes(norm: np.ndarray) -> bytes:
    """16-bit grayscale PNG — works in Godot ImageTexture & Unity Texture2D."""
    u16 = (norm * 65535.0).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(u16, mode="I;16").save(buf, format="PNG")
    return buf.getvalue()


def _colour_png_bytes(norm: np.ndarray, params: TerrainParams) -> bytes:
    """RGB PNG of the baked biome palette, aligned to the OBJ's UVs.

    OBJ's UV convention has v=0 at the bottom, but image row 0 is the top, so
    we flip vertically before encoding.
    """
    rgb = bake_colour_texture(norm, dict(params))
    rgb = np.flipud(rgb)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _unity_raw_bytes(norm: np.ndarray) -> tuple[bytes, int]:
    """Unity Terrain wants an NxN 16-bit little-endian RAW with N = 2^k + 1.

    We resample to the next valid size >= current resolution (usually 513).
    Returns (raw_bytes, side_length).
    """
    n = norm.shape[0]
    # pick smallest 2^k+1 that is >= n
    k = 1
    while (1 << k) + 1 < n:
        k += 1
    side = (1 << k) + 1
    if side != n:
        # PIL's I;16 mode can't do BILINEAR, so resample in float ('F') space.
        pil = Image.fromarray(norm.astype(np.float32), mode="F")
        pil = pil.resize((side, side), Image.BILINEAR)
        resized = np.asarray(pil, dtype=np.float32)
        u16 = np.clip(resized * 65535.0, 0, 65535).astype(np.uint16)
    else:
        u16 = (norm * 65535.0).astype(np.uint16)
    # Unity reads row-major, little-endian
    return u16.astype("<u2").tobytes(), side


def _godot_importer(slug: str, height_scale: float) -> str:
    return f'''@tool
extends EditorScript

## WorldLimit importer for Godot 4.x
##
## 1. Drop this whole unzipped folder anywhere under `res://`.
## 2. Open this script in the script editor and press Ctrl+Shift+X
##    (File → Run) to build the world into the currently open scene.
##
## What it builds under a new "World_{slug}" root:
##   - Terrain           : MeshInstance3D (from terrain.obj)
##   - TerrainBody       : StaticBody3D + ConcavePolygonShape3D collider
##   - Trees/Conifers    : MultiMeshInstance3D (one draw call for N conifers)
##   - Trees/Broadleafs  : MultiMeshInstance3D (one draw call for N broadleafs)

const PLANE_SIZE: float = {PLANE_SIZE:.1f}
const HEIGHT_SCALE: float = {height_scale:.4f}

func _run() -> void:
	var base_dir: String = (get_script() as Script).resource_path.get_base_dir()

	var root: Node3D = Node3D.new()
	root.name = "World_{slug}"

	# --- terrain mesh + collision ---
	var terrain_res: Resource = load(base_dir + "/terrain.obj")
	var terrain_mesh: Mesh = terrain_res as Mesh
	if terrain_mesh != null:
		var mi: MeshInstance3D = MeshInstance3D.new()
		mi.name = "Terrain"
		mi.mesh = terrain_mesh
		root.add_child(mi)

		var body: StaticBody3D = StaticBody3D.new()
		body.name = "TerrainBody"
		mi.add_child(body)

		var shape: CollisionShape3D = CollisionShape3D.new()
		var col: ConcavePolygonShape3D = ConcavePolygonShape3D.new()
		col.set_faces(terrain_mesh.get_faces())
		shape.shape = col
		body.add_child(shape)
	else:
		push_warning("WorldLimit: terrain.obj not found at " + base_dir)

	# --- trees ---
	var trees_file: FileAccess = FileAccess.open(base_dir + "/trees.json", FileAccess.READ)
	if trees_file != null:
		var parsed: Variant = JSON.parse_string(trees_file.get_as_text())
		trees_file.close()
		if parsed is Dictionary:
			var instances: Array = (parsed as Dictionary).get("instances", [])
			var conifers: Array = []
			var broadleafs: Array = []
			for t in instances:
				if t.get("type", "") == "conifer":
					conifers.append(t)
				else:
					broadleafs.append(t)

			var trees_root: Node3D = Node3D.new()
			trees_root.name = "Trees"
			root.add_child(trees_root)

			var conifer_mesh: Mesh = load(base_dir + "/tree_conifer.obj") as Mesh
			var broadleaf_mesh: Mesh = load(base_dir + "/tree_broadleaf.obj") as Mesh

			_add_multimesh(trees_root, "Conifers", conifer_mesh, conifers)
			_add_multimesh(trees_root, "Broadleafs", broadleaf_mesh, broadleafs)
			print("WorldLimit: %d conifers, %d broadleafs" % [conifers.size(), broadleafs.size()])

	# --- attach to open scene ---
	var scene_root: Node = get_scene()
	if scene_root != null:
		scene_root.add_child(root)
		_own_recursive(root, scene_root)
	else:
		push_warning("Open a scene before running this importer.")

func _add_multimesh(parent: Node3D, name: String, mesh: Mesh, instances: Array) -> void:
	if mesh == null or instances.is_empty():
		return
	var mm: MultiMesh = MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = mesh
	mm.instance_count = instances.size()
	for i in instances.size():
		var t: Dictionary = instances[i]
		var basis: Basis = Basis(Vector3.UP, float(t.rotation_y)).scaled(Vector3.ONE * float(t.scale))
		var origin: Vector3 = Vector3(float(t.x), float(t.y), float(t.z))
		mm.set_instance_transform(i, Transform3D(basis, origin))
	var mmi: MultiMeshInstance3D = MultiMeshInstance3D.new()
	mmi.name = name
	mmi.multimesh = mm
	parent.add_child(mmi)

func _own_recursive(n: Node, owner_node: Node) -> void:
	n.owner = owner_node
	for c in n.get_children():
		_own_recursive(c, owner_node)
'''


def _unity_importer(slug: str, height_scale: float) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", slug) or "World"
    return f'''// WorldLimit importer for Unity (2022.3+).
//
// Put this file + the rest of the unzipped folder under Assets/WorldLimit_{safe}/.
// In Unity: right-click the folder → "WorldLimit → Import {safe}".
//
// What it does:
//   - loads terrain.obj as a MeshFilter + MeshCollider
//   - reads trees.json and instantiates tree_conifer.obj / tree_broadleaf.obj
//     as children (GPU-instanced if you enable it on the material)
//
// NOTE: Unity's OBJ importer auto-mirrors X on load, so positions in trees.json
//       are mirrored on X below to compensate.
#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

public static class WorldLimitImporter_{safe}
{{
    const float PLANE_SIZE = {PLANE_SIZE:.1f}f;
    const float HEIGHT_SCALE = {height_scale:.4f}f;

    [MenuItem("Assets/WorldLimit/Import {safe}")]
    public static void Import()
    {{
        string folder = GetFolderPath();
        if (folder == null) {{ Debug.LogError("Run this from inside the world folder."); return; }}

        var terrainMesh = AssetDatabase.LoadAssetAtPath<Mesh>(folder + "/terrain.obj");
        var coniferMesh = AssetDatabase.LoadAssetAtPath<Mesh>(folder + "/tree_conifer.obj");
        var broadleafMesh = AssetDatabase.LoadAssetAtPath<Mesh>(folder + "/tree_broadleaf.obj");
        if (terrainMesh == null) {{ Debug.LogError("terrain.obj not found"); return; }}

        var root = new GameObject("World_{safe}");

        var terrainGo = new GameObject("Terrain");
        terrainGo.transform.SetParent(root.transform, false);
        terrainGo.AddComponent<MeshFilter>().sharedMesh = terrainMesh;
        terrainGo.AddComponent<MeshRenderer>().sharedMaterial =
            new Material(Shader.Find("Standard")) {{ color = new Color(0.55f, 0.55f, 0.55f) }};
        terrainGo.AddComponent<MeshCollider>().sharedMesh = terrainMesh;

        var treesGo = new GameObject("Trees");
        treesGo.transform.SetParent(root.transform, false);

        string json = File.ReadAllText(folder + "/trees.json");
        var data = JsonUtility.FromJson<TreeData>(WrapInstances(json));
        var coniferMat = new Material(Shader.Find("Standard")) {{ color = new Color(0.12f, 0.36f, 0.18f) }};
        var broadleafMat = new Material(Shader.Find("Standard")) {{ color = new Color(0.22f, 0.44f, 0.20f) }};

        int count = 0;
        foreach (var t in data.instances)
        {{
            var mesh = (t.type == "conifer") ? coniferMesh : broadleafMesh;
            if (mesh == null) continue;
            var go = new GameObject(t.type);
            go.transform.SetParent(treesGo.transform, false);
            // Unity mirrors X when importing OBJ, so negate X here to match.
            go.transform.localPosition = new Vector3(-t.x, t.y, t.z);
            go.transform.localRotation = Quaternion.Euler(0f, -Mathf.Rad2Deg * t.rotation_y, 0f);
            go.transform.localScale = Vector3.one * t.scale;
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterial =
                (t.type == "conifer") ? coniferMat : broadleafMat;
            count++;
        }}

        Debug.Log($"WorldLimit: imported {{count}} trees");
        Selection.activeGameObject = root;
    }}

    static string GetFolderPath()
    {{
        foreach (var g in Selection.assetGUIDs)
        {{
            string p = AssetDatabase.GUIDToAssetPath(g);
            if (AssetDatabase.IsValidFolder(p)) return p;
            return Path.GetDirectoryName(p).Replace("\\\\", "/");
        }}
        return null;
    }}

    static string WrapInstances(string raw) => raw;

    [System.Serializable] class TreeData {{ public Tree[] instances; }}
    [System.Serializable] class Tree
    {{
        public string type;
        public float x, y, z, rotation_y, scale;
    }}
}}
#endif
'''


def _readme(slug: str, params: TerrainParams, tree_count: int, heightmap_side: int, unity_side: int) -> str:
    return f"""# WorldLimit export — `{slug}`

This folder contains a single procedurally-generated world. It imports into
both **Godot 4.x** and **Unity 2022.3+**. All files are plain-text / standard
formats — no proprietary blobs.

## What's inside

| File | Purpose |
|---|---|
| `heightmap.png` | 16-bit grayscale, {heightmap_side}×{heightmap_side}. Height 0–65535 maps to [{0:.0f} m, real max below]. Works in any engine. |
| `heightmap.raw` | 16-bit little-endian RAW, {unity_side}×{unity_side}. Drop into Unity's Terrain → Import Raw… |
| `terrain.obj` (+`terrain.mtl`) | Full terrain mesh with UVs, right-handed Y-up. |
| `tree_conifer.obj`, `tree_broadleaf.obj` (+`tree.mtl`) | Two template meshes. |
| `trees.json` | Per-instance transforms: `{{type, x, y, z, rotation_y, scale}}`. |
| `metadata.json` | Predicted biome params, world dimensions, coord conventions. |
| `import_godot.gd` | Drop-in Godot 4.x importer. |
| `import_unity.cs` | Drop-in Unity editor importer. |

## Predicted biome params

| Param | Value |
|---|---|
| `elev_mean` | {params['elev_mean']:.0f} m |
| `elev_std`  | {params['elev_std']:.0f} m |
| `slope_mean`| {params['slope_mean']:.1f} ° |
| `bio1` (mean temp) | {params['bio1']:.1f} °C |
| `bio4` (temp seasonality) | {params['bio4']:.0f} |
| `bio12` (annual rain) | {params['bio12']:.0f} mm |
| Trees placed | **{tree_count}** |

## Import into Godot 4.x

1. Copy this whole folder anywhere into your project's `res://`.
2. In the FileSystem dock, double-click `import_godot.gd`.
3. Press **Ctrl+Shift+X** (File → Run). Your current open scene gets a new
   `World_{slug}` root with terrain + trees + collision.

Alternatively: use just `terrain.obj` as a `MeshInstance3D` and
`heightmap.png` as input to a `HeightMapShape3D` for physics.

## Import into Unity 2022.3+

**Option A — OBJ mesh (matches the browser preview exactly):**

1. Copy the folder into `Assets/WorldLimit_{slug}/`.
2. Right-click the folder → **WorldLimit → Import {slug}**.
3. A `World_{slug}` GameObject is created with `Terrain` + `Trees` children.

**Option B — Unity Terrain system (better LOD, grass/details, texture painting):**

1. Create a new Terrain in your scene.
2. Terrain Settings → Heightmap Resolution: **{unity_side}**.
3. Terrain → **Import Raw…** → select `heightmap.raw`.
   - Depth: 16-bit
   - Byte Order: Windows (little-endian)
   - Width/Height: {unity_side}
4. To keep trees: run the Option A importer (or place them manually using
   `trees.json`).

## Coordinate conventions

- Y-up, right-handed (Three.js / Godot convention).
- Terrain spans `[-{PLANE_SIZE/2:.0f}, +{PLANE_SIZE/2:.0f}]` on X and Z.
- Y is `normalized_height × height_scale` — see `metadata.json` for the exact
  `height_scale` used and the real elevation range.
- Unity's OBJ importer mirrors X automatically; `import_unity.cs` already
  negates tree X to compensate.

## Regeneration

This export is deterministic: given the same prompt + seed, the heightmap and
tree placements are byte-identical. Seed is stored in `metadata.json`.
"""


def build_world_bundle(
    prompt: str,
    seed: int,
    heights_m: np.ndarray,
    params: TerrainParams,
) -> ExportResult:
    slug = _slugify(prompt)

    heights_small = _downsample(heights_m, EXPORT_GRID).astype(np.float32)
    norm, h_min, h_max = _normalize(heights_small)
    height_scale = _compute_height_scale(h_min, h_max)

    trees = place_trees(
        norm,
        params={"bio1": params["bio1"], "bio12": params["bio12"], "elev_std": params["elev_std"]},
        plane_size=PLANE_SIZE,
        height_scale=height_scale,
        seed=seed,
    )
    tree_buckets = partition_by_type(trees)

    png_bytes = _heightmap_png_bytes(norm)
    colour_png_bytes = _colour_png_bytes(norm, params)
    raw_bytes, unity_side = _unity_raw_bytes(norm)
    terrain_obj_text = terrain_to_obj(norm, PLANE_SIZE, height_scale)
    conifer_obj_text = conifer_obj()
    broadleaf_obj_text = broadleaf_obj()

    metadata = {
        "prompt": prompt,
        "slug": slug,
        "seed": seed,
        "params": dict(params),
        "world": {
            "plane_size": PLANE_SIZE,
            "height_scale": height_scale,
            "grid": EXPORT_GRID,
            "heightmap_min_m": h_min,
            "heightmap_max_m": h_max,
            "unity_raw_side": unity_side,
            "axis": "right-handed, Y-up",
        },
        "trees": {
            "count": len(trees),
            "by_type": {k: len(v) for k, v in tree_buckets.items()},
        },
    }

    trees_payload = {"instances": [dict(t) for t in trees]}

    readme = _readme(slug, params, len(trees), norm.shape[0], unity_side)
    godot_script = _godot_importer(slug, height_scale)
    unity_script = _unity_importer(slug, height_scale)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        prefix = f"world_{slug}/"
        zf.writestr(prefix + "heightmap.png", png_bytes)
        zf.writestr(prefix + "heightmap.raw", raw_bytes)
        zf.writestr(prefix + "terrain_color.png", colour_png_bytes)
        zf.writestr(prefix + "terrain.obj", terrain_obj_text)
        zf.writestr(prefix + "terrain.mtl", terrain_mtl())
        zf.writestr(prefix + "tree_conifer.obj", conifer_obj_text)
        zf.writestr(prefix + "tree_broadleaf.obj", broadleaf_obj_text)
        zf.writestr(prefix + "tree.mtl", tree_mtl())
        zf.writestr(prefix + "trees.json", json.dumps(trees_payload, indent=2))
        zf.writestr(prefix + "metadata.json", json.dumps(metadata, indent=2))
        zf.writestr(prefix + "import_godot.gd", godot_script)
        zf.writestr(prefix + "import_unity.cs", unity_script)
        zf.writestr(prefix + "README.md", readme)

    return ExportResult(
        slug=slug,
        zip_bytes=zip_buf.getvalue(),
        tree_count=len(trees),
        filename=f"world_{slug}.zip",
    )
