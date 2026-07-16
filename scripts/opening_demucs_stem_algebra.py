#!/usr/bin/env python3
"""Create review candidates by recombining/subtracting Demucs opening stems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = ROOT / "work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav"
DEFAULT_STEM_DIR = ROOT / "work/review/opening_audio_demucs_001/htdemucs_6s/05_asset_track02_spa1_original_stereo"
DEFAULT_OUT = ROOT / "work/review/opening_audio_demucs_001/stem_algebra_001"
WINDOWS = {
    "voice_024_027": (24.0, 27.0),
    "laser_mid_023_027": (23.0, 27.0),
    "effects_late_055_063": (55.0, 63.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--stem-dir", type=Path, default=DEFAULT_STEM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-rate", type=int, default=48000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    original, sr = read_audio(args.original, args.sample_rate)
    stems = {path.stem: read_audio(path, args.sample_rate)[0] for path in sorted(args.stem_dir.glob("*.wav"))}
    length = min([len(original), *(len(stem) for stem in stems.values())])
    original = original[:length]
    stems = {name: stem[:length] for name, stem in stems.items()}

    combos: dict[str, np.ndarray] = {}
    combos["01_original_minus_demucs_other_gain100"] = original - stems["other"]
    combos["02_original_minus_demucs_other_gain125"] = original - stems["other"] * 1.25
    combos["03_original_minus_demucs_other_gain150"] = original - stems["other"] * 1.50
    combos["04_sum_all_except_other"] = sum_stems(stems, exclude={"other"})
    combos["05_vocals_plus_drums"] = stems["vocals"] + stems["drums"]
    combos["06_vocals_plus_drums_plus_guitar"] = stems["vocals"] + stems["drums"] + stems["guitar"]
    combos["07_vocals_plus_drums_plus_piano_guitar"] = stems["vocals"] + stems["drums"] + stems["piano"] + stems["guitar"]
    combos["08_all_except_other_bass"] = sum_stems(stems, exclude={"other", "bass"})
    combos["09_laser_candidate_drums_guitar_piano"] = stems["drums"] + stems["guitar"] + stems["piano"]
    combos["10_other_inverted_under_original_gain75"] = original + stems["other"] * -0.75

    manifest = {
        "original": str(args.original),
        "stem_dir": str(args.stem_dir),
        "sample_rate": args.sample_rate,
        "notes": [
            "These are algebraic recombinations of Demucs stems, not new model outputs.",
            "original_minus_demucs_other is a direct test of using Demucs 'other' as mostly-music to subtract.",
            "Higher other gain removes more of that stem but can create phase/music artifacts.",
        ],
        "outputs": [],
    }
    for name, data in combos.items():
        safe = peak_guard(data)
        path = args.out_dir / f"{name}.wav"
        sf.write(path, safe, args.sample_rate, subtype="PCM_24")
        norm_path = args.out_dir / f"{name}_norm.wav"
        sf.write(norm_path, peak_normalize(safe, 0.85), args.sample_rate, subtype="PCM_24")
        windows_dir = args.out_dir / "windows"
        windows_dir.mkdir(exist_ok=True)
        for label, (start, end) in WINDOWS.items():
            clip = segment(safe, args.sample_rate, start, end)
            sf.write(windows_dir / f"{name}_{label}.wav", clip, args.sample_rate, subtype="PCM_24")
            sf.write(windows_dir / f"{name}_{label}_norm.wav", peak_normalize(clip, 0.85), args.sample_rate, subtype="PCM_24")
        manifest["outputs"].append({"name": name, "file": str(path), "normalized_file": str(norm_path)})
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(args.out_dir, manifest)
    print(f"review_dir={args.out_dir}")
    print(f"windows={args.out_dir / 'windows'}")
    return 0


def read_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    data = data[:, :2]
    if sr != target_sr:
        data = signal.resample_poly(data, target_sr, sr, axis=0).astype(np.float32)
        sr = target_sr
    return data, sr


def sum_stems(stems: dict[str, np.ndarray], *, exclude: set[str]) -> np.ndarray:
    total = None
    for name, data in stems.items():
        if name in exclude:
            continue
        total = data.copy() if total is None else total + data
    if total is None:
        raise ValueError("No stems selected")
    return total


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
        "# Opening Demucs Stem Algebra",
        "",
        "These candidates test whether Demucs `other` can act as a mostly-music stem to subtract from the original.",
        "",
        "Best first files to audition:",
        "",
        "- `01_original_minus_demucs_other_gain100_norm.wav`",
        "- `02_original_minus_demucs_other_gain125_norm.wav`",
        "- `05_vocals_plus_drums_norm.wav`",
        "- `09_laser_candidate_drums_guitar_piano_norm.wav`",
        "- `windows/` short clips around 23-27s and 55-63s",
        "",
    ]
    for item in manifest["outputs"]:
        lines.append(f"- `{Path(item['normalized_file']).name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
