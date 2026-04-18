"""
preprocess.py — turn raw rasters into a feature table for k-means clustering.

For each region, slide a coarse grid over the bbox. For every grid cell compute:
  * elev_mean, elev_std          (from Copernicus DEM)
  * slope_mean                    (gradient magnitude, in degrees)
  * bio1, bio4, bio12             (mean annual temp, temp seasonality,
                                   annual precipitation — from WorldClim)

Output:
    data/processed/features.csv
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")

import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

from app.ml.data.download import REGIONS  # noqa: E402

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DEM_DIR = RAW_DIR / "srtm"
WORLDCLIM_ZIP = RAW_DIR / "worldclim" / "wc2.1_10m_bio.zip"

GRID_SIZE_DEG = 0.1                    # ~11 km cells at the equator
BIO_VARS = [1, 4, 12]                  # annual-mean temp, temp seasonality, annual precip
EARTH_M_PER_DEG = 111_320.0            # rough meters per degree of latitude


def _worldclim_path(n: int) -> str:
    """GDAL virtual-filesystem path to the Nth bioclim raster inside the zip."""
    return f"/vsizip/{WORLDCLIM_ZIP}/wc2.1_10m_bio_{n}.tif"


def _cell_origins(bbox: tuple[float, float, float, float], step: float):
    """Yield (cell_min_lon, cell_min_lat) for every grid cell overlapping bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    nx = math.ceil((max_lon - min_lon) / step)
    ny = math.ceil((max_lat - min_lat) / step)
    for j in range(ny):
        for i in range(nx):
            yield (min_lon + i * step, min_lat + j * step)


def _mean_slope_degrees(elev: np.ndarray, lat_deg: float, pixel_deg: float) -> float:
    """Average slope in degrees, computed from elevation gradients."""
    if elev.size < 4:
        return float("nan")

    dy_m = pixel_deg * EARTH_M_PER_DEG
    dx_m = pixel_deg * EARTH_M_PER_DEG * math.cos(math.radians(lat_deg))

    dz_dy, dz_dx = np.gradient(elev.astype(np.float32), dy_m, dx_m)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    return float(np.degrees(slope_rad).mean())


def _sample_raster_at(dataset: rasterio.DatasetReader, lon: float, lat: float) -> float:
    """Read a single pixel value at (lon, lat). Returns NaN if off-raster or nodata."""
    try:
        row, col = dataset.index(lon, lat)
        if not (0 <= row < dataset.height and 0 <= col < dataset.width):
            return float("nan")
        val = dataset.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
    except Exception:
        return float("nan")

    if dataset.nodata is not None and val == dataset.nodata:
        return float("nan")
    return float(val)


def process_region(
    name: str,
    bbox: tuple[float, float, float, float],
    bio_rasters: dict[int, rasterio.DatasetReader],
) -> list[dict]:
    dem_path = DEM_DIR / f"{name}.tif"
    if not dem_path.exists():
        print(f"  [skip] {name}.tif missing")
        return []

    rows: list[dict] = []
    with rasterio.open(dem_path) as dem:
        pixel_deg = abs(dem.transform.a)

        for cell_lon, cell_lat in _cell_origins(bbox, GRID_SIZE_DEG):
            cell_bounds = (
                cell_lon, cell_lat,
                cell_lon + GRID_SIZE_DEG, cell_lat + GRID_SIZE_DEG,
            )
            try:
                window = from_bounds(*cell_bounds, transform=dem.transform)
                elev = dem.read(1, window=window, boundless=False)
            except (ValueError, rasterio.errors.WindowError):
                continue

            if elev.size == 0:
                continue
            if dem.nodata is not None:
                elev = elev[elev != dem.nodata]
            if elev.size < 4:
                continue

            center_lon = cell_lon + GRID_SIZE_DEG / 2
            center_lat = cell_lat + GRID_SIZE_DEG / 2

            row: dict = {
                "region": name,
                "cell_lat": round(center_lat, 4),
                "cell_lon": round(center_lon, 4),
                "elev_mean": float(elev.mean()),
                "elev_std": float(elev.std()),
                "slope_mean": _mean_slope_degrees(elev, center_lat, pixel_deg),
            }
            for n in BIO_VARS:
                row[f"bio{n}"] = _sample_raster_at(bio_rasters[n], center_lon, center_lat)
            rows.append(row)

    print(f"  {name}: {len(rows)} grid cells")
    return rows


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Opening WorldClim bio rasters from inside zip...")
    bio_rasters = {n: rasterio.open(_worldclim_path(n)) for n in BIO_VARS}

    print("=== Extracting features per region ===")
    all_rows: list[dict] = []
    try:
        for name, bbox in REGIONS.items():
            all_rows.extend(process_region(name, bbox, bio_rasters))
    finally:
        for r in bio_rasters.values():
            r.close()

    if not all_rows:
        print("No rows produced — did you run download.py?")
        return

    df = pd.DataFrame(all_rows)
    out_path = PROCESSED_DIR / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(df.describe())


if __name__ == "__main__":
    main()
