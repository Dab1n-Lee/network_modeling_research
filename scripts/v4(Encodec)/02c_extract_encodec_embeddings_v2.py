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


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    # wav: (T,) or (C,T)
    if wav.ndim == 1:
        return wav
    return wav.mean(dim=0)


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr)


def load_encodec_model(sample_rate_hint: int, device: str):
    """
    Loads EnCodec pretrained model.
    This is the official way: EncodecModel.encodec_model_24khz / 48khz
    (weights will auto-download to torch cache if needed)
    """
    from encodec import EncodecModel  # type: ignore

    if sample_rate_hint <= 24000:
        model = EncodecModel.encodec_model_24khz()
        target_sr = 24000
    else:
        model = EncodecModel.encodec_model_48khz()
        target_sr = 48000

    model.to(device).eval()
    return model, target_sr


def _normalize_encode_return(encoded):
    """
    EnCodec version differences:
    - Sometimes model.encode(x) returns: List[Tuple[codes, scale]]
    - Sometimes returns: List[Frame-like], or Tuple, etc.
    We normalize to:
        codes: Tensor (B, n_q, T) or (n_q, B, T)
        scale: optional Tensor or None
    """
    # Common: list of frames
    if isinstance(encoded, (list, tuple)):
        first = encoded[0]

        # Case: frame is (codes, scale) tuple/list
        if isinstance(first, (tuple, list)) and len(first) >= 1 and torch.is_tensor(first[0]):
            codes = first[0]
            scale = first[1] if (len(first) > 1 and torch.is_tensor(first[1])) else None
            return codes, scale

        # Case: frame-like object with .codes
        if hasattr(first, "codes"):
            codes = first.codes
            scale = getattr(first, "scale", None)
            return codes, scale if torch.is_tensor(scale) else None

        # Case: direct Tensor in list
        if torch.is_tensor(first):
            return first, None

        raise RuntimeError(f"Unsupported encode() element type: {type(first)}")

    # Rare: encode returns Tensor directly
    if torch.is_tensor(encoded):
        return encoded, None

    raise RuntimeError(f"Unsupported encode() return type: {type(encoded)}")


def _to_nqbt(codes: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Convert codes to (n_q, B, T) for convenient slicing by quantizer stage.
    Accepts:
      - (B, n_q, T)  -> transpose to (n_q, B, T)
      - (n_q, B, T)  -> keep
    """
    if codes.ndim != 3:
        raise RuntimeError(f"Unexpected codes ndim: {codes.ndim}, shape={tuple(codes.shape)}")

    # if first dim is B
    if codes.shape[0] == batch_size:
        return codes.transpose(0, 1)  # (n_q, B, T)

    # if second dim is B, already (n_q,B,T)
    if codes.shape[1] == batch_size:
        return codes

    # ambiguous
    raise RuntimeError(f"Cannot infer code layout with batch_size={batch_size}. shape={tuple(codes.shape)}")


def decode_latent_from_codes(quantizer, codes_nqbt: torch.Tensor, frame_rate=None) -> torch.Tensor:
    """
    Quantizer API differences:
      - Some versions: quantizer.decode(codes)  (NO frame_rate)
      - Some versions: quantizer.decode(codes, frame_rate) (positional)
      - Some versions: quantizer.decode(codes, frame_rate=...) (keyword)  [your version: NOT supported]
    We try in safe order.
    """
    # 1) no frame_rate
    try:
        return quantizer.decode(codes_nqbt)
    except TypeError:
        pass

    # 2) positional frame_rate
    if frame_rate is not None:
        return quantizer.decode(codes_nqbt, frame_rate)

    # 3) last resort: try keyword (some versions)
    try:
        return quantizer.decode(codes_nqbt, frame_rate=frame_rate)
    except Exception as e:
        raise RuntimeError(
            "quantizer.decode() API mismatch and frame_rate is unavailable/insufficient."
        ) from e


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", type=str, required=True)
    ap.add_argument("--stimuli_dir", type=str, required=True)
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--encodec_repo", type=str, required=True, help="path to repos/encodec (repo root)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=24000, help="hint: 24000 or 48000 (chooses official preset)")
    ap.add_argument("--max_q", type=int, default=12, help="max RVQ stages used as 'layers'")
    args = ap.parse_args()

    # ---- add encodec repo to sys.path (submodule style) ----
    encodec_repo = Path(args.encodec_repo).resolve()
    sys.path.insert(0, str(encodec_repo))

    meta_csv = Path(args.meta_csv)
    stimuli_dir = Path(args.stimuli_dir)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_csv)
    assert ("wav_filename" in df.columns) or ("wav_path" in df.columns), \
        "meta csv must contain wav_filename or wav_path"
    wav_col = "wav_path" if "wav_path" in df.columns else "wav_filename"

    device = args.device
    model, target_sr = load_encodec_model(args.sr, device)

    quant = getattr(model, "quantizer", None)
    if quant is None:
        raise RuntimeError("Encodec model has no .quantizer attribute.")

    # Some versions expose frame_rate at model level
    frame_rate = getattr(model, "frame_rate", None)

    embeddings = []
    stim_ids, conds, cents, freqs = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = Path(row[wav_col])
        if not wav_path.is_absolute():
            wav_path = stimuli_dir / wav_path

        wav_np, sr = sf.read(str(wav_path))
        wav = torch.tensor(wav_np, dtype=torch.float32)
        wav = to_mono(wav)
        wav = resample_if_needed(wav, sr, target_sr).to(device)

        # EnCodec expects (B,C,T)
        x = wav.unsqueeze(0).unsqueeze(0)

        # ✅ Use official API: model.encode (handles internal frame_rate requirements)
        encoded = model.encode(x)
        codes, scale = _normalize_encode_return(encoded)

        # normalize codes layout to (n_q, B, T)
        codes_nqbt = _to_nqbt(codes, batch_size=x.shape[0])
        n_q = int(codes_nqbt.shape[0])
        K = min(n_q, int(args.max_q))

        # stagewise decoded latents (RVQ stage = layer)
        layer_vecs = []
        for k in range(1, K + 1):
            z_qk = decode_latent_from_codes(quant, codes_nqbt[:k], frame_rate=frame_rate)  # (B,C,T)
            v = z_qk.mean(dim=-1).squeeze(0).squeeze(0)  # -> (C,)
            layer_vecs.append(v.detach().cpu().numpy())

        E_ld = np.stack(layer_vecs, axis=0)  # (K, C)
        embeddings.append(E_ld)

        stim_ids.append(row["stimulus_id"] if "stimulus_id" in df.columns else wav_path.stem)
        conds.append(row["condition"] if "condition" in df.columns else "unknown")
        cents.append(float(row["cents"]) if "cents" in df.columns else np.nan)
        freqs.append(float(row["freq_hz"]) if "freq_hz" in df.columns else np.nan)

    E = np.stack(embeddings, axis=0)  # (N, K, C)
    layer_names = np.array([f"rvq_{i:02d}" for i in range(E.shape[1])], dtype=object)

    np.savez(
        out_npz,
        embeddings=E,
        stimulus_id=np.array(stim_ids, dtype=object),
        condition=np.array(conds, dtype=object),
        cents=np.array(cents, dtype=np.float32),
        freq_hz=np.array(freqs, dtype=np.float32),
        layer_names=layer_names,
        model_name=np.array(["encodec"], dtype=object),
        target_sr=np.array([target_sr], dtype=np.int32),
    )

    print("[SAVED]", out_npz)
    print("embeddings:", E.shape)
    uniq, cnt = np.unique(np.array(conds).astype(str), return_counts=True)
    print("condition counts:", dict(zip(uniq, cnt)))
    print("target_sr:", target_sr)
    print("n_q:", E.shape[1], "latent_dim:", E.shape[2])


if __name__ == "__main__":
    main()
