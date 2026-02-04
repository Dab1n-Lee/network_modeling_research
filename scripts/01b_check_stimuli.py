# 생성한 자극에 대해서 오차 없이 추출되었는지 확인하는 스크립트

# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Optional plots (OFF by default to avoid backend issues)
PLOT_EXAMPLES = False

FFT_N = 131072
MAX_FREQ = 2000.0


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def dominant_freq_fft(x: np.ndarray, sr: int) -> float:
    x = x.astype(np.float64)
    X = np.fft.rfft(x, n=FFT_N)
    freqs = np.fft.rfftfreq(FFT_N, 1.0 / sr)
    mag = np.abs(X)

    mask = freqs <= MAX_FREQ
    freqs = freqs[mask]
    mag = mag[mask]

    return float(freqs[int(np.argmax(mag))])


def fundamental_freq_autocorr(x: np.ndarray, sr: int) -> float:
    x = x.astype(np.float64)
    x -= np.mean(x)

    corr = np.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2 :]
    if len(corr) == 0:
        return float("nan")

    corr[0] = 0.0

    min_lag = int(sr / 1000)  # 1000 Hz
    max_lag = int(sr / 80)    # 80 Hz
    if max_lag <= min_lag + 1:
        return float("nan")

    lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
    return float(sr / lag) if lag > 0 else float("nan")


def cents_error(est_hz: float, target_hz: float) -> float:
    if not np.isfinite(est_hz) or est_hz <= 0 or target_hz <= 0:
        return float("nan")
    return 1200.0 * math.log2(est_hz / target_hz)


def main() -> None:
    # Ensure immediate printing
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    stim_root = root / "data" / "stimuli"
    meta_path = stim_root / "stimuli_meta.csv"

    print("=== 01b_check_stimuli.py START ===")
    print(f"Project root : {root}")
    print(f"Stimuli root : {stim_root}")
    print(f"Meta CSV     : {meta_path}")

    if not meta_path.exists():
        print("\n[ERROR] stimuli_meta.csv not found.")
        print("=> Did you run scripts/01_make_stimuli.py from the same project?")
        return

    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\nMeta rows: {len(rows)}")
    if len(rows) == 0:
        print("[ERROR] Meta CSV has 0 rows. Check CSV content.")
        return

    missing = 0
    results = []

    print("\nChecking files & estimating frequencies...\n")

    for row in tqdm(rows, total=len(rows)):
        stim_id = row["stimulus_id"]
        condition = row["condition"]
        target_hz = float(row["freq_hz"])

        wav_path = stim_root / condition / f"{stim_id}.wav"
        if not wav_path.exists():
            missing += 1
            continue

        x, sr = sf.read(wav_path)
        if x.ndim > 1:
            x = x.mean(axis=1)

        dom_fft = dominant_freq_fft(x, sr)
        f0_ac = fundamental_freq_autocorr(x, sr)

        err_fft = cents_error(dom_fft, target_hz)
        err_ac = cents_error(f0_ac, target_hz)

        peak = float(np.max(np.abs(x)))
        x_rms = rms(x)

        results.append((condition, err_fft, err_ac, x_rms, peak))

    print("\n=== FILE CHECK ===")
    print(f"Found wavs : {len(results)}")
    print(f"Missing    : {missing}")

    if len(results) == 0:
        print("\n[ERROR] No wav files were found for the rows in the CSV.")
        print("=> Check folder structure: data/stimuli/pure and data/stimuli/harmonic")
        return

    for cond in ["pure", "harmonic"]:
        sub = [r for r in results if r[0] == cond]
        if not sub:
            print(f"\n[{cond.upper()}] No files found.")
            continue

        fft_abs = [abs(r[1]) for r in sub if np.isfinite(r[1])]
        ac_abs  = [abs(r[2]) for r in sub if np.isfinite(r[2])]
        rms_vals = [r[3] for r in sub]
        peaks = [r[4] for r in sub]

        print(f"\n[{cond.upper()}]")
        print(f"  FFT abs cents error: mean={np.mean(fft_abs):.3f}, max={np.max(fft_abs):.3f}")
        print(f"  AC  abs cents error: mean={np.mean(ac_abs):.3f}, max={np.max(ac_abs):.3f}")
        print(f"  RMS mean={np.mean(rms_vals):.4f}, Peak max={np.max(peaks):.3f}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[CRASH] Exception occurred:")
        print(repr(e))
        raise
