# scripts/03b_pitch_geometry_audiomae_from_npz.py
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = l2_normalize(x)
    sim = x @ x.T
    return 1.0 - sim


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y)
    x = x - x.mean(); y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    return float((x @ y) / denom) if denom > 0 else float("nan")


def fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def curvature_angles(vectors: np.ndarray, cents: np.ndarray) -> dict:
    order = np.argsort(cents)
    X = vectors[order]
    c = cents[order]
    if len(c) < 3:
        return {"n": 0, "mean_deg": float("nan"), "max_deg": float("nan")}

    angles = []
    for i in range(1, len(c) - 1):
        a = X[i] - X[i - 1]
        b = X[i + 1] - X[i]
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            continue
        cosang = float(np.clip((a @ b) / (na * nb), -1.0, 1.0))
        angles.append(math.degrees(math.acos(cosang)))

    if not angles:
        return {"n": 0, "mean_deg": float("nan"), "max_deg": float("nan")}
    angles = np.asarray(angles)
    return {"n": int(len(angles)), "mean_deg": float(np.mean(angles)), "max_deg": float(np.max(angles))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="path to audiomae_layerwise_embeddings.npz")
    ap.add_argument("--out_dir", type=str, default=r"..\outputs\analysis\audiomae")
    ap.add_argument("--metric", type=str, default="cos", choices=["cos", "l2"])
    args = ap.parse_args()

    npz_path = Path(args.npz).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(npz_path, allow_pickle=True)
    X_all = z["embeddings"]              # (N, L, D)
    stim_id = z["stimulus_id"].astype(str)
    cond = z["condition"].astype(str)
    cents = z["cents"].astype(float)
    layer_names = z["layer_names"].astype(str)

    N, L, D = X_all.shape
    print("=== PITCH GEOMETRY (AudioMAE, NPZ-only) ===")
    print("NPZ:", npz_path)
    print(f"N={N}, L={L}, D={D}")
    print("Conditions:", sorted(set(cond.tolist())))
    print("Cents range:", float(np.min(cents)), "..", float(np.max(cents)))
    print("OUT:", out_dir)

    # sanity
    if np.isnan(X_all).any() or np.isinf(X_all).any():
        raise ValueError("Embedding contains NaN/Inf.")

    # Prepare pairwise Δcents indices once
    pairs_i, pairs_j = np.triu_indices(N, k=1)
    dcents = np.abs(cents[pairs_i] - cents[pairs_j])

    results = []
    for li in range(L):
        X = X_all[:, li, :]  # (N, D)

        if args.metric == "cos":
            dist_mat = cosine_distance_matrix(X)
            dist = dist_mat[pairs_i, pairs_j]
        else:
            # L2 distances
            diff = X[pairs_i] - X[pairs_j]
            dist = np.linalg.norm(diff, axis=1)

        slope, intercept = fit_slope(dcents, dist)
        r = pearsonr(dcents, dist)

        # curvature overall + by condition
        curv_all = curvature_angles(X, cents)

        row = {
            "layer": li,
            "layer_name": layer_names[li] if li < len(layer_names) else f"layer_{li:02d}",
            "pairs": int(len(dcents)),
            "slope": slope,
            "intercept": intercept,
            "pearson_r": r,
            "curv_n": curv_all["n"],
            "curv_mean_deg": curv_all["mean_deg"],
            "curv_max_deg": curv_all["max_deg"],
        }
        results.append(row)

    df = pd.DataFrame(results)
    out_csv = out_dir / "audiomae_layerwise_geometry.csv"
    df.to_csv(out_csv, index=False)
    print("\n[SAVED]", out_csv)

    # Plot: layerwise slope and r
    fig = plt.figure()
    plt.plot(df["layer"], df["slope"], marker="o")
    plt.xlabel("Layer")
    plt.ylabel("slope (dist ~ Δcents)")
    plt.title("AudioMAE: layerwise pitch sensitivity (slope)")
    p1 = out_dir / "layerwise_slope.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(df["layer"], df["pearson_r"], marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Pearson r")
    plt.title("AudioMAE: layerwise pitch sensitivity (correlation)")
    p2 = out_dir / "layerwise_pearson_r.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(df["layer"], df["curv_mean_deg"], marker="o")
    plt.xlabel("Layer")
    plt.ylabel("mean curvature angle (deg)")
    plt.title("AudioMAE: layerwise curvature proxy")
    p3 = out_dir / "layerwise_curvature_mean.png"
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("[SAVED]", p1)
    print("[SAVED]", p2)
    print("[SAVED]", p3)

    # Also save a “best layer” summary
    best_slope = df.iloc[df["slope"].abs().values.argmax()]
    best_r = df.iloc[df["pearson_r"].abs().values.argmax()]
    print("\n[BEST by |slope|]")
    print(best_slope.to_string())
    print("\n[BEST by |pearson_r|]")
    print(best_r.to_string())

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
