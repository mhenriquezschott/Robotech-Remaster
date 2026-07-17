#!/usr/bin/env python3
"""Run SAM-Audio text/span prompt separation for opening-credit SFX review."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav"
DEFAULT_OUT = ROOT / "work/review/opening_audio_sam_audio_001"
DEFAULT_MODEL = "facebook/sam-audio-base-tv"
DEFAULT_PROMPTS = [
    "motorcycle engine sound effect",
    "motorcycle engine revving",
    "laser gun sound effects",
    "spaceship and laser blast sound effects",
    "all non-music sound effects",
    "spanish narrator voice saying robotech",
]
WINDOWS = {
    "voice_023p8_027p6": (23.8, 27.6),
    "laser_mid_023_027": (23.0, 27.0),
    "effects_late_055_063": (55.0, 63.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument(
        "--anchor",
        nargs=2,
        type=float,
        action="append",
        metavar=("START", "END"),
        help="Optional positive span prompt in seconds. Repeat for multiple spans.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--predict-spans", action="store_true")
    parser.add_argument("--reranking-candidates", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--normalize-peak", type=float, default=0.85)
    parser.add_argument("--process-start", type=float, default=None, help="Optional input segment start, in source seconds.")
    parser.add_argument("--process-end", type=float, default=None, help="Optional input segment end, in source seconds.")
    args = parser.parse_args()

    prompts = [p.lower().strip() for p in (args.prompt or DEFAULT_PROMPTS)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torchaudio
    from sam_audio import SAMAudio, SAMAudioProcessor

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading SAM-Audio model: {args.model}")
    # SAM-Audio's current BaseModel signature is slightly out of sync with
    # newer huggingface_hub.ModelHubMixin, which may drop `proxies` and
    # `resume_download` before dispatch. Calling the model's own loader directly
    # keeps the workaround local to this runner.
    model = SAMAudio._from_pretrained(
        model_id=args.model,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        token=None,
        visual_ranker=None,
        text_ranker=None,
    ).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(args.model)
    model_sr = int(processor.audio_sampling_rate)
    print(f"device={device} model_sample_rate={model_sr}")

    audio_input: str | torch.Tensor = str(args.input)
    process_offset = 0.0
    if args.process_start is not None or args.process_end is not None:
        source_wav, source_sr = torchaudio.load(str(args.input))
        if source_sr != model_sr:
            source_wav = torchaudio.functional.resample(source_wav, source_sr, model_sr)
        process_start = float(args.process_start or 0.0)
        process_end = float(args.process_end or (source_wav.shape[-1] / model_sr))
        if process_end <= process_start:
            raise ValueError("--process-end must be after --process-start")
        start_sample = max(0, int(round(process_start * model_sr)))
        end_sample = min(source_wav.shape[-1], int(round(process_end * model_sr)))
        audio_input = source_wav[:, start_sample:end_sample]
        process_offset = process_start
        print(f"processing segment={process_start:.3f}-{process_end:.3f}s duration={(end_sample - start_sample) / model_sr:.3f}s")

    anchors = None
    if args.anchor:
        shifted = []
        for start, end in args.anchor:
            local_start = max(0.0, float(start) - process_offset)
            local_end = max(0.0, float(end) - process_offset)
            if local_end > local_start:
                shifted.append(("+", local_start, local_end))
        anchors = [shifted]
        print("anchors=", anchors[0])

    manifest = {
        "input": str(args.input),
        "model": args.model,
        "device": str(device),
        "model_sample_rate": model_sr,
        "review_sample_rate": args.sample_rate,
        "predict_spans": args.predict_spans,
        "reranking_candidates": args.reranking_candidates,
        "anchors": anchors[0] if anchors else None,
        "process_start": args.process_start,
        "process_end": args.process_end,
        "outputs": [],
        "notes": [
            "SAM-Audio converts input to mono internally.",
            "Target is the prompted sound; residual is everything else according to the model.",
            "Stereo files duplicate the mono model output only for review/mixing convenience.",
            "Normalized files are for quick listening; compare raw-level files before choosing material.",
        ],
    }

    for index, prompt in enumerate(prompts, start=1):
        label = f"{index:02d}_{slugify(prompt)}"
        print(f"{label}: {prompt}")

        batch = processor(
            audios=[audio_input],
            descriptions=[prompt],
            anchors=anchors,
        ).to(device)

        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=args.predict_spans,
                reranking_candidates=args.reranking_candidates,
            )

        target = tensor_to_numpy(first_audio(result.target))
        residual = tensor_to_numpy(first_audio(result.residual))
        output_record = {
            "prompt": prompt,
            "label": label,
            "target": write_review_family(args.out_dir, label, "target", target, model_sr, args.sample_rate, args.normalize_peak),
            "residual": write_review_family(args.out_dir, label, "residual", residual, model_sr, args.sample_rate, args.normalize_peak),
        }
        write_windows(args.out_dir / "windows", label, "target", target, model_sr, args.sample_rate, args.normalize_peak, process_offset)
        write_windows(args.out_dir / "windows", label, "residual", residual, model_sr, args.sample_rate, args.normalize_peak, process_offset)
        manifest["outputs"].append(output_record)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(args.out_dir, manifest)
    print(f"review_dir={args.out_dir}")
    print(f"windows={args.out_dir / 'windows'}")
    return 0


def first_audio(value) -> object:
    if isinstance(value, (list, tuple)):
        return value[0]
    return value[0]


def tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    data = np.asarray(value, dtype=np.float32)
    if data.ndim > 1:
        data = np.mean(data, axis=0)
    return data.astype(np.float32)


def write_review_family(
    out_dir: Path,
    label: str,
    stem: str,
    mono: np.ndarray,
    source_sr: int,
    target_sr: int,
    normalize_peak: float,
) -> dict[str, str]:
    mono_48 = resample(mono, source_sr, target_sr)
    stereo_48 = np.repeat(mono_48[:, None], 2, axis=1)

    raw_mono = out_dir / f"{label}_sam_{stem}_{target_sr//1000}k_mono.wav"
    raw_stereo = out_dir / f"{label}_sam_{stem}_{target_sr//1000}k_stereo.wav"
    norm_stereo = out_dir / f"{label}_sam_{stem}_{target_sr//1000}k_stereo_norm.wav"

    sf.write(raw_mono, peak_guard(mono_48), target_sr, subtype="PCM_24")
    sf.write(raw_stereo, peak_guard(stereo_48), target_sr, subtype="PCM_24")
    sf.write(norm_stereo, peak_normalize(stereo_48, normalize_peak), target_sr, subtype="PCM_24")
    return {
        "mono": str(raw_mono),
        "stereo": str(raw_stereo),
        "normalized_stereo": str(norm_stereo),
    }


def write_windows(
    out_dir: Path,
    label: str,
    stem: str,
    mono: np.ndarray,
    source_sr: int,
    target_sr: int,
    normalize_peak: float,
    process_offset: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mono_48 = resample(mono, source_sr, target_sr)
    stereo_48 = np.repeat(mono_48[:, None], 2, axis=1)
    for window_label, (start, end) in WINDOWS.items():
        local_start = start - process_offset
        local_end = end - process_offset
        if local_end <= 0 or local_start >= (len(stereo_48) / target_sr):
            continue
        clip = segment(stereo_48, target_sr, max(0.0, local_start), max(0.0, local_end))
        raw = out_dir / f"{label}_{stem}_{window_label}.wav"
        norm = out_dir / f"{label}_{stem}_{window_label}_norm.wav"
        sf.write(raw, peak_guard(clip), target_sr, subtype="PCM_24")
        sf.write(norm, peak_normalize(clip, normalize_peak), target_sr, subtype="PCM_24")


def slugify(value: str) -> str:
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
    anchor_text = "none"
    if manifest["anchors"]:
        anchor_text = ", ".join(f"{token} {start:.3f}-{end:.3f}s" for token, start, end in manifest["anchors"])
    lines = [
        "# Opening SAM-Audio Prompt Tests",
        "",
        f"Model: `{manifest['model']}`",
        f"Anchors: `{anchor_text}`",
        "",
        "SAM-Audio is a promptable separator. These files are candidates for salvage, not final stems.",
        "",
        "Review order:",
        "",
        "- `*_sam_target_48k_stereo_norm.wav`: the prompted sound.",
        "- `*_sam_residual_48k_stereo_norm.wav`: everything else according to SAM-Audio.",
        "- `windows/*voice_023p8_027p6*`: narrator/laser overlap.",
        "- `windows/*laser_mid_023_027*`: early laser/voice overlap.",
        "- `windows/*effects_late_055_063*`: later SFX/motorcycle-heavy region.",
        "",
        "Prompts:",
        "",
    ]
    for item in manifest["outputs"]:
        lines.append(f"- `{item['label']}`: {item['prompt']}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
