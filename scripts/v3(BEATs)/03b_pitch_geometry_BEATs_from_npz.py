import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, math
from pathlib import Path

def l2_normalize(x, eps=1e-8):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def cosine_distance_matrix(x):
    x = l2_normalize(x)
    sim = x @ x.T
    return 1.0 - sim

def pearsonr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    x = x - x.mean(); y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    return float((x @ y) / denom) if denom > 0 else float("nan")

def fit_slope(x, y):
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)

def curvature_angles(vectors, cents):
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

def compute_pair_metrics(X_layer, cents, idx_i, idx_j, metric="cos"):
    # idx_i, idx_j are arrays of indices into X_layer
    dc = np.abs(cents[idx_i] - cents[idx_j])
    if metric == "cos":
        dist_mat = cosine_distance_matrix(X_layer)
        dist = dist_mat[idx_i, idx_j]
    else:
        diff = X_layer[idx_i] - X_layer[idx_j]
        dist = np.linalg.norm(diff, axis=1)

    slope, intercept = fit_slope(dc, dist)
    r = pearsonr(dc, dist)
    return slope, intercept, r, int(len(dc))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=str)
    ap.add_argument("--out_dir", default=r"..\..\outputs\analysis\audiomae_split", type=str)
    ap.add_argument("--metric", default="cos", choices=["cos","l2"])
    args = ap.parse_args()

    z = np.load(Path(args.npz), allow_pickle=True)
    X_all = z["embeddings"]          # (N,L,D)
    cond  = z["condition"].astype(str)
    cents = z["cents"].astype(float)
    layer_names = z["layer_names"].astype(str)

    N, L, D = X_all.shape
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    idx_p = np.where(cond == "pure")[0]
    idx_h = np.where(cond == "harmonic")[0]

    # precompute all pairs once
    I_all, J_all = np.triu_indices(N, k=1)

    # masks for pair types
    is_pure = np.isin(I_all, idx_p) & np.isin(J_all, idx_p)
    is_harm = np.isin(I_all, idx_h) & np.isin(J_all, idx_h)
    is_cross = ~(is_pure | is_harm)

    results = []
    for li in range(L):
        X = X_all[:, li, :]

        # pairwise
        slope_all, b_all, r_all, n_all = compute_pair_metrics(X, cents, I_all, J_all, args.metric)
        slope_p, b_p, r_p, n_p = compute_pair_metrics(X, cents, I_all[is_pure], J_all[is_pure], args.metric)
        slope_h, b_h, r_h, n_h = compute_pair_metrics(X, cents, I_all[is_harm], J_all[is_harm], args.metric)
        slope_x, b_x, r_x, n_x = compute_pair_metrics(X, cents, I_all[is_cross], J_all[is_cross], args.metric)

        # curvature by condition (avoid Δc=0 cross contamination)
        curv_p = curvature_angles(X[idx_p], cents[idx_p])
        curv_h = curvature_angles(X[idx_h], cents[idx_h])

        results.append({
            "layer": li,
            "layer_name": layer_names[li] if li < len(layer_names) else f"layer_{li:02d}",
            "pairs_all": n_all, "slope_all": slope_all, "pearson_r_all": r_all,
            "pairs_pure": n_p, "slope_pure": slope_p, "pearson_r_pure": r_p,
            "pairs_harm": n_h, "slope_harm": slope_h, "pearson_r_harm": r_h,
            "pairs_cross": n_x, "slope_cross": slope_x, "pearson_r_cross": r_x,
            "curv_p_n": curv_p["n"], "curv_p_mean_deg": curv_p["mean_deg"],
            "curv_h_n": curv_h["n"], "curv_h_mean_deg": curv_h["mean_deg"],
        })

    df = pd.DataFrame(results)
    out_csv = out_dir / "BEATs_layerwise_geometry_split.csv"
    df.to_csv(out_csv, index=False)

    # quick figures (slope + r, pure vs harm vs cross)
    def plot_lines(ycols, title, ylabel, fname):
        fig = plt.figure()
        for c in ycols:
            plt.plot(df["layer"], df[c], marker="o", label=c)
        plt.xlabel("Layer"); plt.ylabel(ylabel); plt.title(title)
        plt.legend()
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    plot_lines(["slope_pure","slope_harm","slope_cross"], "BEATs: slope by pair type", "slope (dist ~ Δcents)", "layerwise_slope_split.png")
    plot_lines(["pearson_r_pure","pearson_r_harm","pearson_r_cross"], "BEATs: Pearson r by pair type", "Pearson r", "layerwise_r_split.png")
    plot_lines(["curv_p_mean_deg","curv_h_mean_deg"], "BEATs: curvature proxy (within-condition)", "mean angle (deg)", "layerwise_curv_split.png")

    print("[SAVED]", out_csv)
    print("[SAVED] figs in", out_dir)

if __name__ == "__main__":
    main()
