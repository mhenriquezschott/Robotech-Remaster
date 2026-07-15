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
    with torch.no_grad():
        restored = model(tensor)

    restored_np = restored.squeeze(0).detach().cpu().numpy().T
    # Trim possible model padding back to source length.
    restored_np = restored_np[: audio.shape[0], :]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, restored_np, sr, subtype="PCM_24")
    print(f"wrote={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
