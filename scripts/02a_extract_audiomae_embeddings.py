"""
AudioMAE 계열은 보통 log-mel spectrogram 크기가 (1024 time x 128 mel bins)인 형태를 많이 씀.
생성된 자극은 2초라서 그대로 mel을 만들면 time-bin이 부족함.

이에 2초 wav를 반복(loop)해서 10.24초로 만든 뒤 mel을 만들면 깔끔하게 1024 프레임(10ms hop)으로 맞출 수 있음.

target length: 10.24 sec
hop: 480 samples (= 48kHz x 0.01s)
frames ≈ 1024

pure/harmonic이 stationary라서 반복해도 의미가 유지되며, “모델 입력 규격”을 맞추는 데 가장 안정적인 방법.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


# -----------------------
# User config (EDIT THESE)
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_MAE_REPO = PROJECT_ROOT / "repos" / "AudioMAE"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "audiomae" / "pretrained.pth"  # <-- change

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Stimulus preprocessing (match 1024 x 128)
SR = 48_000
TARGET_SEC = 10.24                 # loop/pad to this length
TARGET_SAMPLES = int(SR * TARGET_SEC)
N_MELS = 128
N_FRAMES = 1024
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 480                   # 10 ms @ 48 kHz
FMIN = 0.0
FMAX = SR / 2.0

# Output
OUT_DIR = PROJECT_ROOT / "outputs" / "embeddings" / "audiomae"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# Utilities
# -----------------------
def loop_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Repeat/trim waveform to target length."""
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        return x[:target_len]
    reps = target_len // len(x)
    rem = target_len % len(x)
    y = np.concatenate([np.tile(x, reps), x[:rem]], axis=0)
    return y


def wav_to_logmel(wav: np.ndarray, sr: int) -> torch.Tensor:
    """
    Convert waveform to log-mel spectrogram tensor:
      output shape: (1, 1, 1024, 128)  -> (B, C, T, F)
    """
    if sr != SR:
        raise ValueError(f"Expected sr={SR}, got {sr}")

    wav = loop_to_length(wav, TARGET_SAMPLES).astype(np.float32)

    # (1, T)
    w = torch.from_numpy(wav).unsqueeze(0)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        f_min=FMIN,
        f_max=FMAX,
        n_mels=N_MELS,
        power=2.0,
        center=True,
        pad_mode="reflect",
    )(w)  # (1, n_mels, frames)

    # log-mel (dB)
    mel_db = torchaudio.transforms.AmplitudeToDB(stype="power")(mel)  # (1, n_mels, frames)

    # Ensure exact frames (1024)
    frames = mel_db.shape[-1]
    if frames < N_FRAMES:
        pad = N_FRAMES - frames
        mel_db = torch.nn.functional.pad(mel_db, (0, pad), mode="replicate")
    elif frames > N_FRAMES:
        mel_db = mel_db[..., :N_FRAMES]

    # Rearrange to (B, C, T, F)
    # currently: (1, n_mels, frames) => (1, 1, frames, n_mels)
    mel_db = mel_db.transpose(1, 2).unsqueeze(1).contiguous()
    return mel_db


def load_audiomae_model(repo_path: Path, ckpt_path: Path) -> torch.nn.Module:
    import os, sys, pathlib, torch

    # Windows ckpt pickle compatibility (PosixPath inside ckpt)
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath

    if not repo_path.exists():
        raise FileNotFoundError(f"AudioMAE repo not found: {repo_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    sys.path.insert(0, str(repo_path))
    import models_mae  # from the cloned repo

    # === MATCH main_pretrain.py (audio_exp path) ===
    # dataset='audioset' -> target_length=1024, mel_bins=128
    model = models_mae.mae_vit_base_patch16(
        norm_pix_loss=True,      # main_pretrain default sets norm_pix_loss True :contentReference[oaicite:6]{index=6}
        in_chans=1,
        audio_exp=True,
        img_size=(1024, 128),
        use_custom_patch=False,  # start with False (default training also False unless set) :contentReference[oaicite:7]{index=7}
        split_pos=False,
        pos_trainable=False,
        use_nce=False,
        decoder_mode=0,          # main_pretrain default decoder_mode=1 :contentReference[oaicite:8]{index=8}
        mask_2d=False,
        mask_t_prob=0.7,
        mask_f_prob=0.3,
        no_shift=False,
        alpha=0.0,
        mode=0,
    )
    model.to(DEVICE).eval()

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] loaded strict=False | missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) < 50: print("  missing(sample):", missing[:20])
    if len(unexpected) < 50: print("  unexpected(sample):", unexpected[:20])

    return model


@torch.no_grad()
def extract_layerwise_embeddings(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract per-layer embeddings from ViT blocks.
    Strategy:
      - hook each block output (B, tokens, dim)
      - take mean over patch tokens (exclude cls token if present)
      - return array shape (n_layers, dim)

    Returns:
      emb: (L, D)
      layer_names: list of block names
    """
    activations: Dict[str, torch.Tensor] = {}
    hooks = []

    # Find ViT blocks module list
    blocks = None
    if hasattr(model, "blocks"):
        blocks = model.blocks
    elif hasattr(model, "module") and hasattr(model.module, "blocks"):
        blocks = model.module.blocks

    if blocks is None:
        raise RuntimeError("Model has no attribute 'blocks'. Edit extractor for this model.")

    def make_hook(name: str):
        def hook(_m, _inp, out):
            activations[name] = out
        return hook

    layer_names = []
    for i, blk in enumerate(blocks):
        name = f"block_{i:02d}"
        layer_names.append(name)
        hooks.append(blk.register_forward_hook(make_hook(name)))

    # Forward
    x = x.to(DEVICE)
    _ = model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Convert activations to embeddings
    embs = []
    for name in layer_names:
        out = activations[name]
        # out: (B, tokens, dim) or sometimes (tokens, dim)
        if out.dim() == 2:
            out = out.unsqueeze(0)
        if out.dim() != 3:
            raise RuntimeError(f"Unexpected block output shape for {name}: {tuple(out.shape)}")

        # exclude cls token if tokens >= 2
        if out.shape[1] >= 2:
            patch_tokens = out[:, 1:, :]
        else:
            patch_tokens = out

        emb = patch_tokens.mean(dim=1)  # (B, D)
        embs.append(emb.squeeze(0).detach().cpu().numpy())

    emb_arr = np.stack(embs, axis=0)  # (L, D)
    return emb_arr, layer_names


def main() -> None:
    meta_path = PROJECT_ROOT / "data" / "stimuli" / "stimuli_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_path}")

    print("[INFO] Loading AudioMAE model...")
    model = load_audiomae_model(AUDIO_MAE_REPO, CKPT_PATH)

    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Keep order as in CSV (or filter as you want)
    embeddings = []
    ids = []
    conds = []
    cents = []
    freqs = []

    layer_names_ref = None
    dim_ref = None

    print("[INFO] Extracting embeddings...")
    for r in tqdm(rows):
        stim_id = r["stimulus_id"]
        cond = r["condition"]
        wav_path = PROJECT_ROOT / "data" / "stimuli" / cond / f"{stim_id}.wav"

        wav, sr = sf.read(str(wav_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        x = wav_to_logmel(wav, sr)  # (1,1,1024,128)
        emb, layer_names = extract_layerwise_embeddings(model, x)

        if layer_names_ref is None:
            layer_names_ref = layer_names
            dim_ref = emb.shape[1]
            print(f"[INFO] Layers={len(layer_names_ref)}, Dim={dim_ref}")

        embeddings.append(emb)
        ids.append(stim_id)
        conds.append(cond)
        cents.append(float(r["cents"]))
        freqs.append(float(r["freq_hz"]))

    E = np.stack(embeddings, axis=0)  # (N, L, D)

    out_path = OUT_DIR / "audiomae_layerwise_embeddings.npz"
    np.savez_compressed(
        out_path,
        embeddings=E,
        stimulus_id=np.array(ids),
        condition=np.array(conds),
        cents=np.array(cents, dtype=np.float32),
        freq_hz=np.array(freqs, dtype=np.float32),
        layer_names=np.array(layer_names_ref),
    )
    print(f"\n[OK] Saved: {out_path}")
    print(f"     embeddings shape = {E.shape} (N, L, D)")


if __name__ == "__main__":
    main()
