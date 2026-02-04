# scripts/03a_check_embeddings_audiomae.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_latest_npz(folder: Path) -> Path:
    npzs = sorted(folder.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not npzs:
        raise FileNotFoundError(f"No .npz files found in: {folder}")
    return npzs[0]


def maybe_find_stim_csv(root: Path) -> Path | None:
    # common candidates
    cands = [
        root / "outputs" / "stimuli" / "stimuli.csv",
        root / "outputs" / "stimuli.csv",
        root / "stimuli.csv",
        root / "outputs" / "stimuli" / "stimulus.csv",
        root / "outputs" / "stimulus.csv",
    ]
    for p in cands:
        if p.exists():
            return p
    # fallback: any csv under outputs/stimuli
    stim_dir = root / "outputs" / "stimuli"
    if stim_dir.exists():
        csvs = list(stim_dir.glob("*.csv"))
        if csvs:
            return csvs[0]
    return None


def summarize_array(name: str, arr: np.ndarray) -> dict:
    info = {
        "key": name,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "ndim": int(arr.ndim),
    }
    if np.issubdtype(arr.dtype, np.number):
        info["nan"] = int(np.isnan(arr).sum())
        info["inf"] = int(np.isinf(arr).sum())
        info["min"] = float(np.nanmin(arr)) if arr.size else None
        info["max"] = float(np.nanmax(arr)) if arr.size else None
        info["mean"] = float(np.nanmean(arr)) if arr.size else None
    else:
        info["nan"] = None
        info["inf"] = None
        info["min"] = None
        info["max"] = None
        info["mean"] = None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="")
    ap.add_argument("--npz_dir", type=str, default=r"outputs\embeddings\audiomae")
    ap.add_argument("--root", type=str, default="../../../", help="project root (default: .. from scripts/)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    npz_path = Path(args.npz).resolve() if args.npz else None
    npz_dir = (root / args.npz_dir).resolve()

    if npz_path is None:
        npz_path = find_latest_npz(npz_dir)

    print("=== EMBEDDINGS CHECK (AudioMAE) ===")
    print("ROOT   :", root)
    print("NPZ    :", npz_path)
    print("NPZ dir:", npz_dir)

    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    print("\n[NPZ KEYS]")
    for k in keys:
        v = data[k]
        if isinstance(v, np.ndarray):
            print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  - {k}: type={type(v)}")

    # Heuristic: find main embedding arrays (numeric arrays with ndim>=2)
    arrays = []
    for k in keys:
        v = data[k]
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number) and v.ndim >= 2:
            arrays.append((k, v))

    if not arrays:
        print("\n[WARN] No numeric arrays with ndim>=2 found. Check how you saved the NPZ.")
    else:
        print("\n[ARRAY SUMMARY]")
        rows = [summarize_array(k, v) for k, v in arrays]
        df = pd.DataFrame(rows).sort_values(["ndim", "key"], ascending=[True, True])
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
            print(df.to_string(index=False))

        # Try to infer N from arrays
        Ns = sorted(set(int(v.shape[0]) for _, v in arrays))
        print("\n[INFERRED N (first dim)]", Ns)

    stim_csv = maybe_find_stim_csv(root)
    if stim_csv is None:
        print("\n[STIM CSV] Not found (ok). If you want matching checks, pass or place stimuli.csv under outputs/stimuli.")
    else:
        print("\n[STIM CSV]", stim_csv)
        df = pd.read_csv(stim_csv)
        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        # show top few
        with pd.option_context("display.max_columns", 200):
            print(df.head(5).to_string(index=False))

        # Compare N if possible
        if arrays:
            n_emb = arrays[0][1].shape[0]
            if len(df) != n_emb:
                print(f"\n[WARN] stimuli rows ({len(df)}) != embedding N ({n_emb})")
                print("      You may need alignment via stim_id / wav filename in the geometry script.")
            else:
                print(f"\n[OK] stimuli rows == embedding N == {n_emb}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
