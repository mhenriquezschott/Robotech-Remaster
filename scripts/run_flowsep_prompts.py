#!/usr/bin/env python3
"""Run FlowSep language-query separation on short opening-credit windows."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOWSEP_ROOT = ROOT / "soft/ai_audio_tools/src/FlowSep"
DEFAULT_INPUT = ROOT / "work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav"
DEFAULT_OUT = ROOT / "work/review/opening_audio_flowsep_001"
DEFAULT_PROMPTS = [
    "motorcycle engine sound effect, no music, no speech",
    "laser gun sound effects, no music, no speech",
    "spaceship sound effects and laser blasts, no music, no speech",
    "all non-music sound effects, no music, no speech",
    "spanish narrator voice saying robotech, no music",
    "drums and percussion only, no voice",
]
DEFAULT_WINDOWS = [
    (23.0, 33.24),
    (24.76, 35.0),
]
MODEL_SECONDS = 10.24
MODEL_RATE = 16000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowsep-root", type=Path, default=DEFAULT_FLOWSEP_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        action="append",
        metavar=("START", "END"),
        help="Source window in seconds. FlowSep consumes 10.24 seconds; longer windows are split/truncated intentionally.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--infer-step", type=int, default=20)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--normalize-peak", type=float, default=0.85)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--device", default="cuda", help="FlowSep upstream hardcodes CUDA; this is recorded for review.")
    args = parser.parse_args()

    flowsep_root = args.flowsep_root.resolve()
    if not (flowsep_root / "lass_inference.py").exists():
        raise SystemExit(f"FlowSep not found: {flowsep_root}")

    checkpoint = args.checkpoint or flowsep_root / "model_logs/pretrained/v2_100k.ckpt"
    vae_checkpoint = flowsep_root / "model_logs/pretrained/vae.ckpt"
    if not checkpoint.exists() or not vae_checkpoint.exists():
        raise SystemExit(
            "FlowSep checkpoints are missing. Run:\n"
            "  bash scripts/setup_audio_restoration_tools.sh download-flowsep-checkpoints"
        )

    prompts = [p.strip() for p in (args.prompt or DEFAULT_PROMPTS) if p.strip()]
    windows = args.window or DEFAULT_WINDOWS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.out_dir / "_flowsep_input_windows"
    temp_dir.mkdir(parents=True, exist_ok=True)

    source, source_sr = sf.read(args.input, always_2d=True, dtype="float32")
    source_mono = source.mean(axis=1)

    manifest = {
        "input": str(args.input),
        "flowsep_root": str(flowsep_root),
        "checkpoint": str(checkpoint),
        "vae_checkpoint": str(vae_checkpoint),
        "device": args.device,
        "model_sample_rate": MODEL_RATE,
        "model_seconds": MODEL_SECONDS,
        "review_sample_rate": args.sample_rate,
        "infer_step": args.infer_step,
        "windows": [{"start": start, "end": end} for start, end in windows],
        "outputs": [],
        "notes": [
            "FlowSep inference is fixed around 10.24 second mono windows at 16 kHz.",
            "Input windows are written as temporary 16 kHz files, then FlowSep output is copied to 48 kHz stereo review files.",
            "Normalized files are for listening only; raw-level files preserve model output level.",
            "Use this as candidate material for manual SFX assembly, not as a final stem without review.",
        ],
    }

    for window_index, (start, end) in enumerate(windows, start=1):
        clip = extract_model_window(source_mono, source_sr, start, end)
        window_label = f"w{window_index:02d}_{seconds_label(start)}_{seconds_label(min(end, start + MODEL_SECONDS))}"

        for prompt_index, prompt in enumerate(prompts, start=1):
            prompt_label = f"{prompt_index:02d}_{slugify(prompt)}"
            label = f"{window_label}_{prompt_label}"
            input_wav = temp_dir / f"{label}.wav"
            sf.write(input_wav, clip, MODEL_RATE, subtype="PCM_24")
            print(f"{label}: {start:.3f}-{min(end, start + MODEL_SECONDS):.3f}s | {prompt}")

            result_path = run_flowsep(
                flowsep_root=flowsep_root,
                input_wav=input_wav,
                prompt=prompt,
                checkpoint=checkpoint,
                infer_step=args.infer_step,
            )

            record = write_review_family(
                out_dir=args.out_dir,
                label=label,
                result_path=result_path,
                source_sr=MODEL_RATE,
                target_sr=args.sample_rate,
                normalize_peak=args.normalize_peak,
            )
            record.update(
                {
                    "prompt": prompt,
                    "window_start": start,
                    "window_end": min(end, start + MODEL_SECONDS),
                    "flowsep_raw_result": str(result_path),
                    "flowsep_input": str(input_wav),
                }
            )
            manifest["outputs"].append(record)

    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(args.out_dir, manifest)
    print(f"review_dir={args.out_dir}")
    return 0


def run_flowsep(
    *,
    flowsep_root: Path,
    input_wav: Path,
    prompt: str,
    checkpoint: Path,
    infer_step: int,
) -> Path:
    result_dir = flowsep_root / "lass_result"
    before = {p.resolve() for p in result_dir.glob("*.wav")} if result_dir.exists() else set()
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TRANSFORMERS_CACHE", str((ROOT / "soft/ai_audio_tools/models/flowsep/hf_cache").resolve()))
    env.setdefault("HF_HOME", str((ROOT / "soft/ai_audio_tools/models/flowsep/hf_home").resolve()))
    env["PYTHONPATH"] = str(flowsep_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "lass_inference.py",
        "--text",
        prompt,
        "--audio",
        str(input_wav.resolve()),
        "--load_checkpoint",
        str(checkpoint.resolve()),
        "--infer_step",
        str(infer_step),
    ]
    subprocess.run(command, cwd=flowsep_root, env=env, check=True)
    after = sorted(p for p in result_dir.glob("*.wav") if p.resolve() not in before)
    expected = result_dir / input_wav.name
    if expected.exists():
        return expected
    if after:
        return after[-1]
    raise RuntimeError(f"FlowSep finished but no result WAV was found in {result_dir}")


def extract_model_window(source_mono: np.ndarray, source_sr: int, start: float, end: float) -> np.ndarray:
    if end <= start:
        raise ValueError(f"Invalid window {start}-{end}")
    start_sample = max(0, int(round(start * source_sr)))
    end_sample = min(len(source_mono), int(round(min(end, start + MODEL_SECONDS) * source_sr)))
    clip = source_mono[start_sample:end_sample]
    clip = resample(clip, source_sr, MODEL_RATE)
    target_len = int(round(MODEL_SECONDS * MODEL_RATE))
    if len(clip) < target_len:
        clip = np.pad(clip, (0, target_len - len(clip)))
    return peak_guard(clip[:target_len])


def write_review_family(
    *,
    out_dir: Path,
    label: str,
    result_path: Path,
    source_sr: int,
    target_sr: int,
    normalize_peak: float,
) -> dict[str, str]:
    mono, sr = sf.read(result_path, always_2d=False, dtype="float32")
    if np.asarray(mono).ndim > 1:
        mono = np.asarray(mono).mean(axis=1)
    if sr != source_sr:
        source_sr = sr
    mono_48 = resample(np.asarray(mono, dtype=np.float32), source_sr, target_sr)
    stereo_48 = np.repeat(mono_48[:, None], 2, axis=1)

    raw_mono = out_dir / f"{label}_flowsep_{target_sr//1000}k_mono.wav"
    raw_stereo = out_dir / f"{label}_flowsep_{target_sr//1000}k_stereo.wav"
    norm_stereo = out_dir / f"{label}_flowsep_{target_sr//1000}k_stereo_norm.wav"
    sf.write(raw_mono, peak_guard(mono_48), target_sr, subtype="PCM_24")
    sf.write(raw_stereo, peak_guard(stereo_48), target_sr, subtype="PCM_24")
    sf.write(norm_stereo, peak_normalize(stereo_48, normalize_peak), target_sr, subtype="PCM_24")
    return {
        "mono": str(raw_mono),
        "stereo": str(raw_stereo),
        "normalized_stereo": str(norm_stereo),
    }


def resample(data: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return data.astype(np.float32)
    return signal.resample_poly(data, target_sr, source_sr).astype(np.float32)


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


def seconds_label(value: float) -> str:
    return f"{value:07.3f}".replace(".", "p")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value[:64]


def write_readme(out_dir: Path, manifest: dict) -> None:
    lines = [
        "# Opening FlowSep Prompt Tests",
        "",
        "FlowSep is a language-query separator based on rectified flow matching.",
        "These are review candidates for opening-credit SFX salvage.",
        "",
        "Review order:",
        "",
        "- `*_flowsep_48k_stereo_norm.wav`: normalized listening copy.",
        "- `*_flowsep_48k_stereo.wav`: raw-level stereo review copy.",
        "- `manifest.json`: prompt/window provenance.",
        "",
        "Prompts:",
        "",
    ]
    for item in manifest["outputs"]:
        lines.append(f"- `{Path(item['normalized_stereo']).name}`: {item['window_start']:.3f}-{item['window_end']:.3f}s, {item['prompt']}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
