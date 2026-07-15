#!/usr/bin/env python3
"""Run Apollo audio restoration with an explicit local checkpoint.

The upstream Apollo inference script hardcodes ``JusperLee/Apollo`` as though it
were a local checkpoint path. This wrapper keeps our experiments repeatable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import look2hear.models


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "soft/ai_audio_tools/models/apollo/pytorch_model.bin"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-sr", type=int, default=44100)
    parser.add_argument("--chunk-seconds", type=float, default=0.0, help="Process in chunks to avoid GPU OOM. 0 means whole file.")
    parser.add_argument("--overlap-seconds", type=float, default=1.0, help="Crossfade overlap used with --chunk-seconds.")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing Apollo checkpoint: {args.checkpoint}")

    audio, sr = sf.read(args.input, always_2d=True, dtype="float32")
    if sr != args.expected_sr:
        raise SystemExit(f"Apollo input must be {args.expected_sr} Hz; got {sr}: {args.input}")

    # soundfile is [samples, channels], Apollo expects [batch, channels, samples].
    tensor = torch.from_numpy(audio.T[np.newaxis, :, :])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)

    model = look2hear.models.BaseModel.from_pretrain(
        str(args.checkpoint),
        sr=args.expected_sr,
        win=20,
        feature_dim=256,
        layer=6,
    ).to(device)
    model.eval()
    if args.chunk_seconds and args.chunk_seconds > 0:
        restored_np = restore_chunked(
            model,
            tensor,
            sample_count=audio.shape[0],
            sr=sr,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )
    else:
        with torch.no_grad():
            restored = model(tensor)
        restored_np = restored.squeeze(0).detach().cpu().numpy().T
    # Trim possible model padding back to source length.
    restored_np = restored_np[: audio.shape[0], :]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, restored_np, sr, subtype="PCM_24")
    print(f"wrote={args.output}")
    return 0


def restore_chunked(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    sample_count: int,
    sr: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> np.ndarray:
    chunk_samples = max(1, int(round(chunk_seconds * sr)))
    overlap_samples = max(0, int(round(overlap_seconds * sr)))
    if overlap_samples * 2 >= chunk_samples:
        raise SystemExit("--overlap-seconds must be less than half of --chunk-seconds")
    step = chunk_samples - overlap_samples
    channels = tensor.shape[1]
    device = tensor.device
    output = np.zeros((sample_count, channels), dtype=np.float64)
    weights = np.zeros((sample_count, 1), dtype=np.float64)
    starts = list(range(0, sample_count, step))

    for index, start in enumerate(starts, start=1):
        end = min(sample_count, start + chunk_samples)
        if end <= start:
            continue
        chunk = tensor[:, :, start:end]
        with torch.no_grad():
            restored = model(chunk)
        restored_np = restored.squeeze(0).detach().cpu().numpy().T[: end - start, :]
        weight = np.ones((end - start, 1), dtype=np.float64)
        if start > 0 and overlap_samples:
            fade_len = min(overlap_samples, end - start)
            weight[:fade_len, 0] = np.linspace(0.0, 1.0, fade_len, endpoint=False)
        if end < sample_count and overlap_samples:
            fade_len = min(overlap_samples, end - start)
            weight[-fade_len:, 0] *= np.linspace(1.0, 0.0, fade_len, endpoint=False)
        output[start:end, :] += restored_np.astype(np.float64) * weight
        weights[start:end, :] += weight
        print(f"chunk {index}/{len(starts)} samples={start}:{end}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    weights[weights == 0] = 1.0
    return (output / weights).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
