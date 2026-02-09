# scripts/01_make_stimuli_v2_multif0.py
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
import math

# -------------------------
# Config
# -------------------------
SAMPLE_RATE = 48000
DURATION_SEC = 1.0
RMS_TARGET = 0.05

F0S = {
    "A3": 220.0,
    "A4": 440.0,
    "A5": 880.0,
}

CENTS = [-20, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20]
CONDITIONS = ["pure", "harmonic"]

OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "stimuli_v2"
WAV_ROOT = OUT_ROOT
META_CSV = OUT_ROOT / "stimuli_meta_v2.csv"

# -------------------------
# Utils
# -------------------------
def cents_to_ratio(c):
    return 2 ** (c / 1200.0)

def normalize_rms(x, target=RMS_TARGET):
    rms = np.sqrt(np.mean(x**2))
    if rms > 0:
        x = x * (target / rms)
    return x

def gen_pure_tone(freq, sr, dur):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = np.sin(2 * np.pi * freq * t)
    return x

def gen_harmonic_tone(freq, sr, dur, n_harm=8):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        x += (1.0 / k) * np.sin(2 * np.pi * freq * k * t)
    return x

# -------------------------
# Main
# -------------------------
def main():
    (WAV_ROOT / "pure").mkdir(parents=True, exist_ok=True)
    (WAV_ROOT / "harmonic").mkdir(parents=True, exist_ok=True)

    rows = []

    for f0_name, f0 in F0S.items():
        for c in CENTS:
            freq = f0 * cents_to_ratio(c)
            sign = "plus" if c > 0 else "minus" if c < 0 else "0"
            c_abs = abs(c)

            for cond in CONDITIONS:
                if cond == "pure":
                    x = gen_pure_tone(freq, SAMPLE_RATE, DURATION_SEC)
                else:
                    x = gen_harmonic_tone(freq, SAMPLE_RATE, DURATION_SEC)

                x = normalize_rms(x)

                if c == 0:
                    stim_id = f"{f0_name}_0c_{cond}"
                else:
                    stim_id = f"{f0_name}_{sign}_{c_abs}c_{cond}"

                wav_path = WAV_ROOT / cond / f"{stim_id}.wav"
                sf.write(wav_path, x.astype(np.float32), SAMPLE_RATE)

                rows.append({
                    "stimulus_id": stim_id,
                    "f0_name": f0_name,
                    "f0_hz": f0,
                    "cents": c,
                    "freq_hz": freq,
                    "condition": cond,
                    "wav_path": str(wav_path).replace("\\", "/"),
                    "sample_rate": SAMPLE_RATE,
                    "duration_sec": DURATION_SEC,
                })

    df = pd.DataFrame(rows)
    df.to_csv(META_CSV, index=False)

    # -------- sanity checks --------
    n_expected = len(F0S) * len(CENTS) * len(CONDITIONS)
    n_wavs = sum(1 for _ in WAV_ROOT.rglob("*.wav"))
    print("=== STIMULI V2 SUMMARY ===")
    print("Expected stimuli:", n_expected)
    print("Written wavs     :", n_wavs)
    print("CSV rows         :", len(df))
    assert n_wavs == n_expected == len(df), "Count mismatch!"
    print("OK.")

if __name__ == "__main__":
    main()
