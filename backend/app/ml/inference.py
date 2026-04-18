"""
inference.py — prompt → terrain parameters.

Loads the trained MLP, the sentence-transformers embedder, and the feature
scaler once, then exposes `predict_terrain(prompt)` for use by Phase 4.

Artifacts expected in data/processed/:
    mlp.pt          — trained MLP weights
    scaler.joblib   — StandardScaler fit on raw features in Phase 2
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.ml.train_prompt_model import EMBEDDER_NAME, TerrainMLP

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
PROCESSED_DIR = DATA_DIR / "processed"

FEATURE_NAMES = ("elev_mean", "elev_std", "slope_mean", "bio1", "bio4", "bio12")


class TerrainParams(TypedDict):
    elev_mean: float
    elev_std: float
    slope_mean: float
    bio1: float
    bio4: float
    bio12: float


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[SentenceTransformer, TerrainMLP, object]:
    """Load embedder + MLP + scaler once, cached for process lifetime."""
    embedder = SentenceTransformer(EMBEDDER_NAME)

    model = TerrainMLP()
    model.load_state_dict(torch.load(PROCESSED_DIR / "mlp.pt", map_location="cpu"))
    model.eval()

    scaler = joblib.load(PROCESSED_DIR / "scaler.joblib")
    return embedder, model, scaler


def predict_terrain(prompt: str) -> TerrainParams:
    """Map a free-text prompt to 6 terrain parameters in original units."""
    embedder, model, scaler = _load_artifacts()

    embedding = embedder.encode([prompt], convert_to_numpy=True)
    with torch.no_grad():
        pred_scaled = model(torch.tensor(embedding, dtype=torch.float32)).numpy()
    pred_original = scaler.inverse_transform(pred_scaled)[0]

    return TerrainParams(
        elev_mean=float(pred_original[0]),
        elev_std=float(pred_original[1]),
        slope_mean=float(pred_original[2]),
        bio1=float(pred_original[3]),
        bio4=float(pred_original[4]),
        bio12=float(pred_original[5]),
    )


def predict_batch(prompts: list[str]) -> np.ndarray:
    """Batched variant — returns (N, 6) array in original units."""
    embedder, model, scaler = _load_artifacts()
    embeddings = embedder.encode(prompts, convert_to_numpy=True)
    with torch.no_grad():
        pred_scaled = model(torch.tensor(embeddings, dtype=torch.float32)).numpy()
    return scaler.inverse_transform(pred_scaled)


if __name__ == "__main__":
    import sys

    test_prompts = sys.argv[1:] or [
        "a scorching desert with massive sand dunes",
        "misty evergreen mountains on the coast",
        "flooded lowland jungle near the equator",
        "frozen arctic tundra",
        "grassy plains with scattered trees",
    ]

    print(f"{'prompt':<45} {'elev':>7} {'std':>6} {'slope':>6} {'bio1':>6} {'bio4':>7} {'bio12':>7}")
    for p in test_prompts:
        t = predict_terrain(p)
        print(
            f"{p:<45} "
            f"{t['elev_mean']:>7.0f} {t['elev_std']:>6.0f} {t['slope_mean']:>6.1f} "
            f"{t['bio1']:>6.1f} {t['bio4']:>7.0f} {t['bio12']:>7.0f}"
        )
