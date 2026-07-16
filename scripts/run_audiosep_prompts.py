#!/usr/bin/env python3
"""Run AudioSep language-query separation for opening-credit SFX review."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIOSEP_ROOT = ROOT / "soft/ai_audio_tools/src/AudioSep"
DEFAULT_INPUT = ROOT / "work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav"
DEFAULT_OUT = ROOT / "work/review/opening_audio_audiosep_001"
DEFAULT_PROMPTS = [
    "motorcycle engine sound effect, no music, no speech",
    "motorcycle engine revving, no music, no narrator",
    "laser gun sound effects, no music, no speech",
    "spaceship and laser blast sound effects, no music, no narration",
    "Spanish narrator voice saying Robotech, no music",
]
WINDOWS = {
    "voice_024_027": (24.0, 27.0),
    "laser_mid_023_027": (23.0, 27.0),
    "effects_late_055_063": (55.0, 63.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audiosep-root", type=Path, default=DEFAULT_AUDIOSEP_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-chunk", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--model-id", default="nielsr/audiosep-demo")
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.audiosep_root.resolve()))

    import torch
    from models.audiosep import AudioSep
    from pipeline import separate_audio
    from utils import get_ss_model

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ss_model = get_ss_model(str(args.audiosep_root / "config/audiosep_base.yaml"))
    model = AudioSep.from_pretrained(args.model_id, ss_model=ss_model).eval().to(device)

    manifest = {
        "input": str(args.input),
        "audiosep_root": str(args.audiosep_root),
        "model_id": args.model_id,
        "device": str(device),
        "use_chunk": args.use_chunk,
        "outputs": [],
        "notes": [
            "AudioSep internally loads mono audio at 32 kHz.",
            "48 kHz review files are resampled and duplicated to stereo for project auditioning.",
            "Use these as extraction candidates, not final mix stems.",
        ],
    }

    for index, prompt in enumerate(prompts, start=1):
        label = f"{index:02d}_{slugify(prompt)}"
        raw_32k = args.out_dir / f"{label}_audiosep_32k_mono.wav"
        review_48k = args.out_dir / f"{label}_audiosep_48k_stereo.wav"
        norm_48k = args.out_dir / f"{label}_audiosep_48k_stereo_norm.wav"
        separate_audio(
            model=model,
            audio_file=str(args.input),
            text=prompt,
            output_file=str(raw_32k),
            device=device,
            use_chunk=args.use_chunk,
        )
        mono, sr = sf.read(raw_32k, always_2d=False, dtype="float32")
        if mono.ndim > 1:
            mono = np.mean(mono, axis=1)
        stereo = np.repeat(resample(mono, sr, args.sample_rate)[:, None], 2, axis=1)
        sf.write(review_48k, peak_guard(stereo), args.sample_rate, subtype="PCM_24")
        sf.write(norm_48k, peak_normalize(stereo, 0.85), args.sample_rate, subtype="PCM_24")

        windows_dir = args.out_dir / "windows"
        windows_dir.mkdir(exist_ok=True)
        for window_label, (start, end) in WINDOWS.items():
            clip = segment(stereo, args.sample_rate, start, end)
            sf.write(windows_dir / f"{label}_{window_label}.wav", peak_guard(clip), args.sample_rate, subtype="PCM_24")
            sf.write(
                windows_dir / f"{label}_{window_label}_norm.wav",
                peak_normalize(clip, 0.85),
                args.sample_rate,
                subtype="PCM_24",
            )
        manifest["outputs"].append(
            {
                "prompt": prompt,
                "label": label,
                "raw_32k_mono": str(raw_32k),
                "review_48k_stereo": str(review_48k),
                "normalized_48k_stereo": str(norm_48k),
            }
        )

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(args.out_dir, manifest)
    print(f"review_dir={args.out_dir}")
    print(f"windows={args.out_dir / 'windows'}")
    return 0


def slugify(value: str) -> str:
    value = value.lower().replace("spanish", "spa")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:72]


def resample(data: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return data.astype(np.float32)
    return signal.resample_poly(data, target_sr, source_sr).astype(np.float32)


def segment(data: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    return data[max(0, int(start * sr)) : min(len(data), int(end * sr))]


def peak_guard(data: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.99:
        return data.astype(np.float32)
    return (data / peak * 0.99).astype(np.float32)


def peak_normalize(data: np.ndarray, peak: float) -> np.ndarray:
    current = float(np.max(np.abs(data))) if data.size else 0.0
    if current < 1e-9:
        return data.astype(np.float32)
    return (data * (peak / current)).astype(np.float32)


def write_readme(out_dir: Path, manifest: dict) -> None:
    lines = [
        "# Opening AudioSep Prompt Tests",
        "",
        "These are text-query extraction candidates. They are mono 32 kHz model outputs resampled to 48 kHz stereo for review.",
        "",
        "Review order:",
        "",
        "- Full `*_audiosep_48k_stereo_norm.wav` files.",
        "- `windows/*laser_mid_023_027*` for voice/laser overlap.",
        "- `windows/*effects_late_055_063*` for the later SFX section.",
        "",
        "Prompts:",
        "",
    ]
    for item in manifest["outputs"]:
        lines.append(f"- `{item['label']}`: {item['prompt']}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
