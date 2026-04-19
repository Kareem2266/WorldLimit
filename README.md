# WorldLimit

Turn a natural-language prompt like *"himalayan peaks above the clouds"* or *"flooded lowland jungle"* into an interactive 3D terrain in the browser — and export the same world as a game-ready ZIP for Godot or Unity.

```
prompt  →  MLP  →  6 terrain params  →  Perlin heightmap  →  Three.js 3D scene
                                                          →  Godot/Unity ZIP
```

## What it does

- **Prompt → parameters.** A frozen `sentence-transformers` embedder + a tiny PyTorch MLP maps any English phrase to 6 geophysical numbers (mean elevation, elevation variability, slope, temperature, seasonality, precipitation). Targets were discovered by k-means over real Copernicus DEM + WorldClim data for 12 curated regions.
- **Parameters → heightmap.** Multi-octave Perlin noise, with slope controlling fractal persistence (dunes vs. jagged ridges), produces a 512×512 16-bit PNG heightmap.
- **Heightmap → 3D scene.** A Three.js viewer displaces a mesh, colors it with a climate-aware palette, adds water where appropriate, and scatters up to 4,500 GPU-instanced trees.
- **Scene → game asset.** `/api/export` returns a deterministic ZIP containing OBJ meshes, PNG/RAW heightmaps, baked biome textures, per-instance tree transforms, and one-click Godot/Unity importer scripts.

## Stack

| Layer | Tech |
|---|---|
| Orchestration | Docker Compose |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 15 (raw SQL via `asyncpg`) + Alembic migrations |
| Cache / broker | Redis 7 (fail-open JSON cache, versioned keys) |
| ML — embeddings | `sentence-transformers/all-mpnet-base-v2` (frozen) |
| ML — regression | PyTorch MLP (~265k params) |
| ML — clustering | scikit-learn `KMeans` + `StandardScaler` |
| Geospatial | `rasterio` + GDAL `/vsicurl/` |
| Procedural generation | `noise` (Perlin) |
| Frontend | Three.js 0.160 (vanilla, no bundler) |
| Async worker | Celery (reserved for future long jobs) |

## Repo layout

```
backend/
  app/
    api/routes/          # /generate, /export, /jobs
    ml/
      data/              # download.py, preprocess.py, cluster.py
      terrain/           # generator.py, export.py,
                         # mesh_obj.py, vegetation.py, palette.py,
                         # export_bundle.py
      biome_prompts.py   # hand-written training prompts per biome
      inference.py       # runtime prompt → 6 params
      train_prompt_model.py
    schemas/             # pydantic request/response models
    cache.py             # Redis wrapper
    config.py, database.py, main.py
  alembic/               # DB migrations
data/
  raw/srtm/              # Copernicus DEM GeoTIFFs (12 regions)
  raw/worldclim/         # WorldClim bioclim zip
  processed/             # features.csv, kmeans.joblib, scaler.joblib, mlp.pt
  output/heightmaps/     # generated PNGs
frontend/
  index.html, main.js, terrain.js
docker-compose.yml
REPORT.md                # deep-dive on every design decision
```

## Quickstart

**Prereqs:** Docker + Docker Compose. First run downloads ~420 MB of embedder weights into a named volume (one-time).

```bash
cp .env.example .env          # fill in POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB
docker-compose up --build
```

Services:
- API — http://localhost:8000 (OpenAPI docs at `/docs`)
- Viewer — http://localhost:8000/app/
- Postgres — localhost:5432
- Redis — localhost:6379

### Train the model (one-time, only if you want to rebuild artifacts)

The trained artifacts (`mlp.pt`, `scaler.joblib`, `kmeans.joblib`) are committed under `data/processed/`. To rebuild from scratch:

```bash
docker-compose exec api python -m app.ml.data.download
docker-compose exec api python -m app.ml.data.preprocess
docker-compose exec api python -m app.ml.data.cluster
docker-compose exec api python -m app.ml.train_prompt_model
```

## API

### `POST /api/generate`

```json
{ "prompt": "himalayan peaks above the clouds", "seed": 42 }
```

Returns:

```json
{
  "prompt": "himalayan peaks above the clouds",
  "params": {
    "elev_mean": 4820, "elev_std": 160, "slope_mean": 9.4,
    "bio1": -1.1, "bio4": 660, "bio12": 300
  },
  "heightmap_url": "/static/heightmaps/himalayan_peaks_42.png",
  "heightmap_min_m": 4612,
  "heightmap_max_m": 5310
}
```

Latency: ~2 s cold, ~5 ms cached.

### `POST /api/export`

Same request body. Returns `application/zip` with heightmap + mesh + trees + importer scripts for Godot 4.x and Unity 2022.3+.

## How it works (short version)

1. **Real data foundation.** `download.py` pulls Copernicus GLO-30 DEM tiles for 12 regions spanning every biome, plus WorldClim bioclimate rasters. `preprocess.py` slides a 0.1° grid and computes six features per cell (elevation mean/std, slope, temperature, seasonality, precipitation).
2. **K-means → 8 biome centroids.** `cluster.py` z-scores the features and fits `KMeans(k=8)`. Centroids become the regression targets. `scaler.joblib` is persisted so inference can un-scale predictions back to meters/°C/mm.
3. **160 hand-written prompts.** `biome_prompts.py` has ~20 phrases per biome. Each prompt's target is its biome's centroid.
4. **Frozen embedder + tiny MLP.** `all-mpnet-base-v2` produces 768-dim vectors; a 3-layer MLP (~265k params) regresses them to the 6-dim scaled targets over 200 epochs with early stopping on val loss.
5. **Perlin synthesis.** `generator.py` uses predicted `elev_mean` as base altitude, `elev_std` as amplitude, and `slope_mean` as octave persistence (smooth dunes ↔ jagged ridges). Output is a 16-bit PNG (not 8-bit — prevents stair-step terracing on mesh displacement).
6. **Three.js viewer.** `terrain.js` reads the PNG into a `Float32Array`, displaces a `PlaneGeometry`, colors per-vertex via a climate-aware palette, drops a water plane where precipitation warrants, and renders trees with `InstancedMesh` (one draw call per tree type, regardless of count).
7. **Deterministic export.** `export_bundle.py` rebuilds the same world server-side (same seed → byte-identical output), bakes the vertex palette into a texture, and packages OBJ + PNG + JSON + importer scripts into a ZIP.

For the full design rationale — every "why" behind every choice — see [REPORT.md](./REPORT.md).

## Design principles worth naming

- **Frozen pretrained model + small head.** 160 labeled prompts are enough because the embedder already encodes English.
- **Cluster centroids as regression targets.** Converts an ill-defined continuous regression into a stable soft-classification.
- **Scaler persisted next to weights.** Forgetting to `inverse_transform` with the exact training scaler is a silent 2σ bug.
- **CPU work off the event loop.** `asyncio.to_thread(generate_heightmap, ...)` so the 1.5 s Perlin loop doesn't stall concurrent requests.
- **Fail-open cache.** Redis down ≠ app down. Versioned keys (`worldlimit:params:v1:...`) invalidate cleanly on retrain without `FLUSHDB`.
- **Seeded determinism.** `(prompt, seed)` → the same terrain, the same tree placement, byte-identical ZIP. Diffable, cacheable, reproducible.
- **Instanced rendering.** 4,500 trees as one `InstancedMesh` draw call, mirrored in Godot's `MultiMeshInstance3D` and Unity's `Graphics.DrawMeshInstanced` on export.
