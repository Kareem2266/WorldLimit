"""
download.py — fetch raw geodata from the internet and save to data/raw/

Run this once before preprocessing:
    python -m app.ml.data.download

Two datasets:
  1. Copernicus GLO-30 DEM — global 30 m elevation, 1°x1° Cloud-Optimized
     GeoTIFF tiles served anonymously from AWS Open Data.
  2. WorldClim bioclimatic variables zip from UC Davis.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import httpx

# GDAL HTTP retry settings — must be set BEFORE importing rasterio so GDAL
# picks them up. Remote COG reads occasionally fail mid-transfer; without
# retries a single flaky tile kills the whole merge.
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

import rasterio  # noqa: E402
from rasterio.merge import merge  # noqa: E402

RAW_DIR = Path(os.getenv("DATA_DIR", "/data")) / "raw"
SRTM_DIR = RAW_DIR / "srtm"
WORLDCLIM_DIR = RAW_DIR / "worldclim"

WORLDCLIM_URL = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip"
)

COPERNICUS_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
COPERNICUS_RES = "30"  # arc-seconds in filename: "10" for GLO-30, "30" for GLO-90

REGIONS: dict[str, tuple[float, float, float, float]] = {
    # name: (min_lon, min_lat, max_lon, max_lat)
    "iceland":         (-25.0,  63.0, -13.0,  66.5),
    "amazon":          (-70.0,  -8.0, -50.0,   3.0),
    "sahara":          (  0.0,  20.0,  20.0,  30.0),
    "himalayas":       ( 80.0,  27.0,  90.0,  35.0),
    "hawaii":          (-161.0, 18.5, -154.5, 22.5),
    "alps":            (  5.0,  45.0,  15.0,  48.0),
    "patagonia":       (-75.0, -55.0, -65.0, -45.0),
    "borneo":          (108.0,  -5.0, 118.0,   5.0),
    "greenland":       (-50.0,  65.0, -30.0,  75.0),
    "andes":           (-75.0, -20.0, -65.0, -10.0),
    "norway_coast":    (  5.0,  60.0,  15.0,  65.0),
    "east_africa":     ( 30.0,  -5.0,  40.0,   5.0),
}


def _tile_url(lat_int: int, lon_int: int) -> str:
    """Build the Copernicus COG URL for the 1°x1° tile whose SW corner is (lat_int, lon_int)."""
    lat_letter = "N" if lat_int >= 0 else "S"
    lon_letter = "E" if lon_int >= 0 else "W"
    name = (
        f"Copernicus_DSM_COG_{COPERNICUS_RES}_"
        f"{lat_letter}{abs(lat_int):02d}_00_"
        f"{lon_letter}{abs(lon_int):03d}_00_DEM"
    )
    return f"{COPERNICUS_BASE}/{name}/{name}.tif"


def _tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """Return all (lat, lon) tile SW corners whose 1°x1° footprint overlaps bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_start = math.floor(min_lat)
    lat_end = math.floor(max_lat - 1e-9)
    lon_start = math.floor(min_lon)
    lon_end = math.floor(max_lon - 1e-9)
    return [
        (lat, lon)
        for lat in range(lat_start, lat_end + 1)
        for lon in range(lon_start, lon_end + 1)
    ]


def _tile_exists(url: str, client: httpx.Client) -> bool:
    """HEAD-check whether a tile exists. Ocean-only tiles return 403/404."""
    try:
        return client.head(url).status_code == 200
    except httpx.RequestError:
        return False


def download_copernicus_region(name: str, bbox: tuple[float, float, float, float]) -> None:
    out_path = SRTM_DIR / f"{name}.tif"
    if out_path.exists():
        print(f"  [skip] {name}.tif already exists")
        return

    tiles = _tiles_for_bbox(bbox)
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        urls = [_tile_url(lat, lon) for lat, lon in tiles]
        available = [u for u in urls if _tile_exists(u, client)]

    print(f"  {name}: {len(available)}/{len(tiles)} tile(s) available on S3")

    if not available:
        print(f"  [warn] no land tiles for {name}, skipping")
        return

    datasets = [rasterio.open(f"/vsicurl/{u}") for u in available]
    try:
        mosaic, transform = merge(datasets, bounds=bbox)
    finally:
        for d in datasets:
            d.close()

    profile = {
        "driver": "GTiff",
        "dtype": mosaic.dtype,
        "count": 1,
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    print(f"  Saved {name}.tif ({mosaic.shape[1]}x{mosaic.shape[2]} px)")


def download_worldclim() -> None:
    out_path = WORLDCLIM_DIR / "wc2.1_10m_bio.zip"
    if out_path.exists():
        print("  [skip] WorldClim zip already exists")
        return

    print("  Downloading WorldClim bioclim variables (~120 MB)...")
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        response = client.get(WORLDCLIM_URL)
        response.raise_for_status()

    out_path.write_bytes(response.content)
    print(f"  Saved wc2.1_10m_bio.zip ({len(response.content) // 1_000_000} MB)")


def main() -> None:
    SRTM_DIR.mkdir(parents=True, exist_ok=True)
    WORLDCLIM_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Downloading Copernicus GLO-30 DEM tiles ===")
    for name, bbox in REGIONS.items():
        download_copernicus_region(name, bbox)

    print("\n=== Downloading WorldClim climate data ===")
    download_worldclim()

    print("\nDone. Raw data saved to data/raw/")


if __name__ == "__main__":
    main()
