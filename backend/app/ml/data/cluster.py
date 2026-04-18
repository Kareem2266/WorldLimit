"""
cluster.py — run k-means on the features table to discover biome clusters.

Input:
    data/processed/features.csv

Outputs:
    data/processed/clusters.csv   — per-cell cluster assignments
    data/processed/kmeans.joblib  — fitted k-means model (for Phase 3)
    data/processed/scaler.joblib  — fitted StandardScaler (same features in → same scale)
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
PROCESSED_DIR = DATA_DIR / "processed"

FEATURE_COLS = ["elev_mean", "elev_std", "slope_mean", "bio1", "bio4", "bio12"]
K = 8
RANDOM_STATE = 42


def elbow_analysis(X: np.ndarray, k_range: range) -> None:
    print("\n=== Elbow analysis (lower inertia = tighter clusters) ===")
    print(f"{'k':>3}  {'inertia':>12}")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        print(f"{k:>3}  {km.inertia_:>12.1f}")


def main() -> None:
    features_path = PROCESSED_DIR / "features.csv"
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} rows from features.csv")

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"After dropping NaN rows: {len(df)} rows")

    X_raw = df[FEATURE_COLS].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    elbow_analysis(X, range(2, 13))

    print(f"\n=== Fitting k-means with k={K} ===")
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit(X)
    df["cluster_id"] = km.labels_

    centroids_raw = scaler.inverse_transform(km.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_raw, columns=FEATURE_COLS)
    counts = pd.Series(km.labels_).value_counts().sort_index()
    centroids_df["count"] = counts.values
    print("\n=== Cluster centroids (original units) ===")
    print(centroids_df.round(1).to_string())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROCESSED_DIR / "clusters.csv"
    df[["region", "cell_lat", "cell_lon", "cluster_id"]].to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} cluster assignments to {out_csv}")

    joblib.dump(km, PROCESSED_DIR / "kmeans.joblib")
    joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")
    print("Saved kmeans.joblib and scaler.joblib")


if __name__ == "__main__":
    main()
