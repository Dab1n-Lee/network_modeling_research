# scripts/03c_find_missing_stimulus_ids.py
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np


def stem_to_id(stem: str) -> str:
    # normalize common naming
    s = stem.strip()
    s = s.replace("PURE", "pure").replace("HARMONIC", "harmonic")
    s = s.replace("+0C", "+0c").replace("+0c", "+0c")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--wav_dir", type=str, default=r"..\outputs\stimuli\wavs")
    args = ap.parse_args()

    npz_path = Path(args.npz).resolve()
    wav_dir = Path(args.wav_dir).resolve()

    z = np.load(npz_path, allow_pickle=True)
    npz_ids = set([str(x) for x in z["stimulus_id"]])

    wavs = sorted(list(wav_dir.glob("*.wav")))
    wav_ids = set([stem_to_id(w.stem) for w in wavs])

    missing_in_npz = sorted(list(wav_ids - npz_ids))
    missing_wav = sorted(list(npz_ids - wav_ids))

    print("=== ID DIFF ===")
    print("NPZ N:", len(npz_ids))
    print("WAV N:", len(wav_ids))
    print("\n[WAV present but NOT in NPZ] (likely skipped during extraction):")
    for s in missing_in_npz:
        print(" ", s)
    print("\n[NPZ present but NOT in WAV] (id mismatch / naming mismatch):")
    for s in missing_wav:
        print(" ", s)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
