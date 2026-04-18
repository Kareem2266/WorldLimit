"""
train_prompt_model.py — train an MLP to map prompt embeddings → terrain features.

Pipeline:
    prompt → SentenceTransformer (mpnet) → 768-dim vector → MLP → 6 features

The MLP's targets are the cluster centroids from Phase 2 (in z-score space).
At inference, MLP outputs are inverse_transformed with scaler.joblib to
recover original terrain units (meters, °C, mm).

Inputs:
    data/processed/kmeans.joblib  — provides centroids (targets)
    data/processed/scaler.joblib  — needed later for inference

Outputs:
    data/processed/mlp.pt         — best-val model weights
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from app.ml.biome_prompts import iter_training_pairs

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
PROCESSED_DIR = DATA_DIR / "processed"

EMBEDDER_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 768
FEATURE_DIM = 6

EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
VAL_FRAC = 0.2
SEED = 42


class TerrainMLP(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM, out_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    set_seed(SEED)

    print(f"Loading k-means centroids from {PROCESSED_DIR}/kmeans.joblib")
    kmeans = joblib.load(PROCESSED_DIR / "kmeans.joblib")
    centroids_scaled = kmeans.cluster_centers_

    pairs = iter_training_pairs()
    prompts = [p for p, _ in pairs]
    cluster_ids = np.array([c for _, c in pairs])
    targets = centroids_scaled[cluster_ids]
    print(f"Loaded {len(prompts)} prompts across {len(np.unique(cluster_ids))} clusters")

    print(f"\nLoading embedder: {EMBEDDER_NAME}")
    print("(first run downloads ~420 MB to the hf_cache volume)")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    embeddings = embedder.encode(prompts, convert_to_numpy=True, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings,
        targets,
        test_size=VAL_FRAC,
        stratify=cluster_ids,
        random_state=SEED,
    )
    print(f"\nTrain: {X_train.shape[0]} examples | Val: {X_val.shape[0]} examples")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = TerrainMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    print("\n=== Training ===")
    print(f"{'epoch':>6}  {'train_loss':>11}  {'val_loss':>10}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}")

    print(f"\nBest val loss: {best_val:.4f} at epoch {best_epoch}")

    assert best_state is not None
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, PROCESSED_DIR / "mlp.pt")
    print(f"Saved best model → {PROCESSED_DIR}/mlp.pt")

    scaler = joblib.load(PROCESSED_DIR / "scaler.joblib")
    model.load_state_dict(best_state)
    model.eval()
    sample_prompts = [
        "sahara desert dunes",
        "himalayan peaks above the clouds",
        "amazon rainforest lowland",
        "greenland ice sheet",
        "swiss alpine valley",
        "east african savanna",
    ]
    with torch.no_grad():
        sample_emb = embedder.encode(sample_prompts, convert_to_numpy=True)
        pred_scaled = model(torch.tensor(sample_emb, dtype=torch.float32)).numpy()
    pred_original = scaler.inverse_transform(pred_scaled)

    print("\n=== Sanity check — predictions in original units ===")
    header = f"{'prompt':<38} {'elev':>7} {'std':>6} {'slope':>6} {'bio1':>6} {'bio4':>7} {'bio12':>7}"
    print(header)
    for prompt, row in zip(sample_prompts, pred_original):
        print(
            f"{prompt:<38} "
            f"{row[0]:>7.0f} {row[1]:>6.0f} {row[2]:>6.1f} "
            f"{row[3]:>6.1f} {row[4]:>7.0f} {row[5]:>7.0f}"
        )


if __name__ == "__main__":
    main()
