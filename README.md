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
