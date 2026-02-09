from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int = 16000) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr)


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    # wav: (T,) or (C,T)
    if wav.ndim == 1:
        return wav
    return wav.mean(dim=0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", type=str, required=True)
    ap.add_argument("--stimuli_dir", type=str, required=True)
    ap.add_argument("--beats_repo", type=str, required=True, help="path to BEATs repo root (e.g., .../unilm)")
    ap.add_argument("--ckpt", type=str, required=True, help="path to BEATs checkpoint .pt")
    ap.add_argument("--out_npz", type=str, default=r".\outputs\embeddings\beats\beats_layerwise_embeddings_v2.npz")
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    meta_csv = Path(args.meta_csv)
    stimuli_dir = Path(args.stimuli_dir)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_csv)
    assert ("wav_filename" in df.columns) or ("wav_path" in df.columns), \
        "meta csv must contain wav_filename or wav_path column"
    wav_col = "wav_path" if "wav_path" in df.columns else "wav_filename"

    # ---- import BEATs from repo ----
    beats_repo = Path(args.beats_repo).resolve()
    sys.path.insert(0, str(beats_repo))            # repo root
    sys.path.insert(0, str(beats_repo / "beats"))  # to satisfy local imports like backbone.py

    # unilm BEATs layout: beats/BEATs.py
    from BEATs import BEATs, BEATsConfig  # type: ignore

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().to(args.device)

    # Layer meta
    if not hasattr(model, "cfg") or not hasattr(model.cfg, "encoder_layers"):
        raise RuntimeError("BEATs model has no cfg.encoder_layers; cannot name layers reliably.")
    n_layers = int(model.cfg.encoder_layers)
    layer_names = np.array([f"block_{i:02d}" for i in range(n_layers)], dtype=object)

    embeddings = []
    stim_ids, conds, cents, freqs = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = Path(row[wav_col])
        if not wav_path.is_absolute():
            wav_path = stimuli_dir / wav_path

        wav_np, sr = sf.read(str(wav_path))
        wav = torch.tensor(wav_np, dtype=torch.float32)
        wav = to_mono(wav)
        wav = resample_if_needed(wav, sr, args.target_sr).to(args.device)

        # (B,T) raw waveform
        source = wav.unsqueeze(0)

        # padding_mask: keep None for now (safest)
        padding_mask = None

        # ---- BEATs frontend (mirrors BEATs.py extract_features internal steps) ----
        fbank = model.preprocess(source)  # (B, frames, mel)

        if padding_mask is not None:
            padding_mask = model.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)                 # (B,1,frames,mel)
        features = model.patch_embedding(fbank)    # conv2d output
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)        # (B,T,C)
        features = model.layer_norm(features)

        if padding_mask is not None:
            padding_mask = model.forward_padding_mask(features, padding_mask)

        if getattr(model, "post_extract_proj", None) is not None:
            features = model.post_extract_proj(features)

        x = model.dropout_input(features)          # (B,T,C) -> encoder input

        # ---- Transformer encoder (backbone.py) ----
        # IMPORTANT: layer_results is filled only when tgt_layer is not None.
        last_layer = n_layers - 1
        x_out, layer_results = model.encoder.extract_features(
            x,
            padding_mask=padding_mask,
            tgt_layer=last_layer
        )

        if layer_results is None or len(layer_results) < 2:
            raise RuntimeError(f"Unexpected empty/short layer_results: {0 if layer_results is None else len(layer_results)}")

        # layer_results[0] is the "input state"; drop it so we have exactly n_layers items
        layer_results = layer_results[1:]

        if len(layer_results) != n_layers:
            # Not fatal, but we want to know.
            print(f"[WARN] layer_results len={len(layer_results)} != encoder_layers={n_layers}. "
                  f"(Keeping the returned length.)")

        # ---- Pool each layer: (T,B,C) -> (C,) ----
        layer_vecs = []
        for (h, _) in layer_results:
            # h: (T,B,C)
            h = h.transpose(0, 1)                  # (B,T,C)
            v = h.mean(dim=1).squeeze(0)           # (C,)
            layer_vecs.append(v.detach().cpu().numpy())

        E_ld = np.stack(layer_vecs, axis=0)        # (L,C)
        embeddings.append(E_ld)

        # ---- metadata ----
        stim_ids.append(row["stimulus_id"] if "stimulus_id" in df.columns else wav_path.stem)
        conds.append(row["condition"] if "condition" in df.columns else "unknown")
        cents.append(float(row["cents"]) if "cents" in df.columns else np.nan)
        freqs.append(float(row["freq_hz"]) if "freq_hz" in df.columns else np.nan)

    E = np.stack(embeddings, axis=0)  # (N,L,D)

    # If layer count differs from cfg.encoder_layers, store the actual count in layer_names too.
    if E.shape[1] != len(layer_names):
        layer_names = np.array([f"block_{i:02d}" for i in range(E.shape[1])], dtype=object)

    np.savez(
        out_npz,
        embeddings=E,
        stimulus_id=np.array(stim_ids, dtype=object),
        condition=np.array(conds, dtype=object),
        cents=np.array(cents, dtype=np.float32),
        freq_hz=np.array(freqs, dtype=np.float32),
        layer_names=layer_names,
    )

    print("[SAVED]", out_npz)
    print("embeddings:", E.shape)
    uniq, cnt = np.unique(np.array(conds).astype(str), return_counts=True)
    print("condition counts:", dict(zip(uniq, cnt)))


if __name__ == "__main__":
    main()
