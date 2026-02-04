# -*- coding: utf-8 -*-
"""
MVP1 Stimulus generator (48 kHz)
- Pure tones and harmonic complex tones around A4=440Hz
- Cents offsets: [-20,-10,-5,-3,-2,-1, +1,+2,+3,+5,+10,+20]
- Duration: 2.0 sec, with 20 ms fade in/out
- RMS normalization (target_rms)
Outputs:
  data/stimuli/pure/*.wav
  data/stimuli/harmonic/*.wav
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
SAMPLE_RATE = 48_000
DURATION_SEC = 2.0
BASE_FREQ_HZ = 440.0  # A4
CENTS_OFFSETS = [-20, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20]

FADE_MS = 20.0
TARGET_RMS = 0.05  # safe loudness, adjust later if needed
PEAK_LIMIT = 0.98  # prevent clipping

HARMONIC_K = 10          # number of harmonics
HARMONIC_AMP_POWER = 1.0 # 1.0 means amplitude ~ 1/k; 0.5 would be ~1/sqrt(k)


# -----------------------------
# Helpers
# -----------------------------
def cents_to_hz(base_hz: float, cents: float) -> float:
    """Convert cents offset relative to base frequency."""
    return base_hz * (2.0 ** (cents / 1200.0))


def make_time(sr: int, dur_sec: float) -> np.ndarray:
    n = int(round(sr * dur_sec))
    return np.arange(n, dtype=np.float32) / sr


def apply_fade(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    """Apply symmetric linear fade in/out to avoid clicks."""
    fade_n = int(round(sr * (fade_ms / 1000.0)))
    if fade_n <= 0:
        return x
    fade_n = min(fade_n, len(x) // 2)
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    x = x.copy()
    x[:fade_n] *= ramp
    x[-fade_n:] *= ramp[::-1]
    return x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def normalize_rms_peak(x: np.ndarray, target_rms: float, peak_limit: float) -> np.ndarray:
    """Normalize to target RMS, then ensure peak does not exceed peak_limit."""
    x = x.astype(np.float32, copy=False)
    cur_rms = rms(x)
    if cur_rms > 0:
        x = x * (target_rms / cur_rms)

    peak = float(np.max(np.abs(x)))
    if peak > peak_limit:
        x = x * (peak_limit / peak)
    return x


def make_pure_tone(freq_hz: float, sr: int, dur_sec: float) -> np.ndarray:
    t = make_time(sr, dur_sec)
    x = np.sin(2.0 * np.pi * freq_hz * t, dtype=np.float32)
    return x


def make_harmonic_complex(
    f0_hz: float,
    sr: int,
    dur_sec: float,
    k: int = 10,
    amp_power: float = 1.0,
) -> np.ndarray:
    """
    Harmonic complex: sum_{i=1..k} (1/i^amp_power) * sin(2*pi*(i*f0)*t)
    """
    t = make_time(sr, dur_sec)
    x = np.zeros_like(t, dtype=np.float32)

    # Keep only harmonics below Nyquist to avoid aliasing
    nyquist = sr / 2.0
    for i in range(1, k + 1):
        fi = i * f0_hz
        if fi >= nyquist:
            break
        amp = 1.0 / (i ** amp_power)
        x += (amp * np.sin(2.0 * np.pi * fi * t)).astype(np.float32)
    return x


def fmt_cents(c: int) -> str:
    return f"{c:+d}c".replace("+", "plus_").replace("-", "minus_")


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # subtype='PCM_16' for compatibility; you can switch to 'PCM_24'
    sf.write(str(path), x, sr, subtype="PCM_16")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    root = Path(__file__).resolve().parents[1]  # project root
    out_pure = root / "data" / "stimuli" / "pure"
    out_harm = root / "data" / "stimuli" / "harmonic"

    meta_lines = []
    meta_lines.append("stimulus_id,condition,base_hz,cents,freq_hz,sr,duration_sec\n")

    for cents in tqdm(CENTS_OFFSETS, desc="Generating stimuli"):
        freq = cents_to_hz(BASE_FREQ_HZ, cents)

        # Pure
        x_p = make_pure_tone(freq, SAMPLE_RATE, DURATION_SEC)
        x_p = apply_fade(x_p, SAMPLE_RATE, FADE_MS)
        x_p = normalize_rms_peak(x_p, TARGET_RMS, PEAK_LIMIT)
        fname_p = f"A4_{fmt_cents(cents)}.wav"
        write_wav(out_pure / fname_p, x_p, SAMPLE_RATE)

        # Harmonic
        x_h = make_harmonic_complex(freq, SAMPLE_RATE, DURATION_SEC, k=HARMONIC_K, amp_power=HARMONIC_AMP_POWER)
        x_h = apply_fade(x_h, SAMPLE_RATE, FADE_MS)
        x_h = normalize_rms_peak(x_h, TARGET_RMS, PEAK_LIMIT)
        fname_h = f"A4_{fmt_cents(cents)}.wav"
        write_wav(out_harm / fname_h, x_h, SAMPLE_RATE)

        meta_lines.append(f"A4_{fmt_cents(cents)},{'pure'},{BASE_FREQ_HZ},{cents},{freq},{SAMPLE_RATE},{DURATION_SEC}\n")
        meta_lines.append(f"A4_{fmt_cents(cents)},{'harmonic'},{BASE_FREQ_HZ},{cents},{freq},{SAMPLE_RATE},{DURATION_SEC}\n")

    # Also write the 0-cent reference (optional but convenient)

    meta_path = root / "data" / "stimuli" / "stimuli_meta.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("".join(meta_lines), encoding="utf-8")

    print("\nDone.")
    print(f"- Pure tones:     {out_pure}")
    print(f"- Harmonic tones: {out_harm}")
    print(f"- Meta CSV:       {meta_path}")


if __name__ == "__main__":
    main()
