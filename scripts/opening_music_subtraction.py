#!/usr/bin/env python3
"""Align soundtrack music to opening audio and export cancellation residues.

The goal is not a mathematically perfect null. The opening contains narration,
SFX, compression, and probably a different master. This script creates review
files that make those non-music elements easier to hear.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "work/review/opening_audio_rebuild_001/music_subtraction_001"
DEFAULT_MUSIC = ROOT / "work/review/opening_audio_rebuild_001/sources/06_soundtrack_main_title_full.wav"
DEFAULT_TARGETS = [
    ROOT / "work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav",
    ROOT / "work/review/opening_audio_rebuild_001/sources/08_tv_copy_ep01_intro_stereo.wav",
    ROOT / "work/review/opening_audio_rebuild_001/sources/02_current_opening_stream2_stereo.wav",
]


@dataclass
class AlignmentResult:
    target: str
    speed_percent: float
    offset_seconds: float
    music_gain_left: float
    music_gain_right: float
    music_gain_db_left: float
    music_gain_db_right: float
    residual_rms_db: float
    target_rms_db: float
    output_prefix: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--music", type=Path, default=DEFAULT_MUSIC)
    parser.add_argument("--target", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--speeds", type=float, nargs="+", default=[99.0, 99.5, 100.0, 100.5, 101.0])
    parser.add_argument("--max-offset", type=float, default=4.0)
    parser.add_argument("--review-lowpass", type=float, default=12000.0)
    args = parser.parse_args()

    targets = args.target or DEFAULT_TARGETS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    music, sr = read_audio(args.music)
    results: list[AlignmentResult] = []
    for target_path in targets:
        target, target_sr = read_audio(target_path)
        if target_sr != sr:
            raise SystemExit(f"Sample-rate mismatch: {target_path} is {target_sr}, music is {sr}")
        result = process_target(
            target_path=target_path,
            target=target,
            music=music,
            sr=sr,
            speeds=args.speeds,
            max_offset=args.max_offset,
            out_dir=args.out_dir,
            review_lowpass=args.review_lowpass,
        )
        results.append(result)
        print(
            f"{target_path.name}: speed={result.speed_percent:.3f}% "
            f"offset={result.offset_seconds:+.4f}s "
            f"gain=({result.music_gain_db_left:+.2f},{result.music_gain_db_right:+.2f})dB "
            f"residual={result.residual_rms_db:.2f}dBFS"
        )

    manifest = {
        "music": str(args.music),
        "targets": [str(path) for path in targets],
        "sample_rate": sr,
        "speed_grid_percent": args.speeds,
        "max_offset_seconds": args.max_offset,
        "outputs": [asdict(result) for result in results],
        "notes": [
            "aligned_music is the soundtrack conformed to the target timing/gain.",
            "residue is target minus aligned_music; review it for voice/SFX.",
            "residue_presence isolates the likely high-frequency SFX/voice edge band.",
            "residue_voice_band is only a quick listening aid, not a clean voice stem.",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(args.out_dir, results)
    print(f"review_dir={args.out_dir}")
    return 0


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        # Simple stereo fold for matching/review. For 5.1 sources use prepared
        # stereo assets when possible.
        data = data[:, :2]
    return data, sr


def process_target(
    *,
    target_path: Path,
    target: np.ndarray,
    music: np.ndarray,
    sr: int,
    speeds: list[float],
    max_offset: float,
    out_dir: Path,
    review_lowpass: float,
) -> AlignmentResult:
    target = normalize_channels(target)
    music = normalize_channels(music)
    target_mono = target.mean(axis=1)

    best_score = float("inf")
    best_aligned: np.ndarray | None = None
    best_gains: np.ndarray | None = None
    best_speed = 100.0
    best_offset = 0

    for speed_percent in speeds:
        speed = speed_percent / 100.0
        sped_music = stretch_by_index(music, speed=speed, output_len=len(target) + int(abs(max_offset) * sr) * 2)
        offset = estimate_offset(target_mono, sped_music.mean(axis=1), sr=sr, max_offset=max_offset)
        aligned = align_to_target(sped_music, len(target), offset)
        gains = fit_channel_gains(target, aligned)
        residual = target - aligned * gains[np.newaxis, :]
        score = rms(residual)
        if score < best_score:
            best_score = score
            best_aligned = aligned * gains[np.newaxis, :]
            best_gains = gains
            best_speed = speed_percent
            best_offset = offset

    if best_aligned is None or best_gains is None:
        raise RuntimeError("No alignment result")

    residual = target - best_aligned
    prefix = target_path.stem.replace(" ", "_")
    aligned_path = out_dir / f"{prefix}_aligned_soundtrack.wav"
    residue_path = out_dir / f"{prefix}_residue_music_subtracted.wav"
    residue_voice_path = out_dir / f"{prefix}_residue_voice_band_180_4200.wav"
    residue_presence_path = out_dir / f"{prefix}_residue_presence_sfx_2500_12000.wav"
    residue_norm_path = out_dir / f"{prefix}_residue_music_subtracted_norm.wav"

    sf.write(aligned_path, best_aligned, sr, subtype="PCM_24")
    sf.write(residue_path, residual, sr, subtype="PCM_24")
    sf.write(residue_norm_path, peak_normalize(residual, peak=0.85), sr, subtype="PCM_24")
    sf.write(residue_voice_path, bandpass(residual, sr, 180.0, 4200.0), sr, subtype="PCM_24")
    sf.write(residue_presence_path, bandpass(residual, sr, 2500.0, review_lowpass), sr, subtype="PCM_24")

    return AlignmentResult(
        target=str(target_path),
        speed_percent=best_speed,
        offset_seconds=best_offset / sr,
        music_gain_left=float(best_gains[0]),
        music_gain_right=float(best_gains[1]),
        music_gain_db_left=float(linear_to_db(best_gains[0])),
        music_gain_db_right=float(linear_to_db(best_gains[1])),
        residual_rms_db=float(linear_to_db(rms(residual))),
        target_rms_db=float(linear_to_db(rms(target))),
        output_prefix=prefix,
    )


def normalize_channels(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2].astype(np.float32, copy=False)


def stretch_by_index(data: np.ndarray, *, speed: float, output_len: int) -> np.ndarray:
    positions = np.arange(output_len, dtype=np.float64) * speed
    positions = np.clip(positions, 0, len(data) - 1)
    left = np.interp(positions, np.arange(len(data)), data[:, 0])
    right = np.interp(positions, np.arange(len(data)), data[:, 1])
    return np.column_stack([left, right]).astype(np.float32)


def estimate_offset(target: np.ndarray, reference: np.ndarray, *, sr: int, max_offset: float) -> int:
    analysis_sr = 1000
    target_env = analysis_envelope(target, sr, analysis_sr)
    ref_env = analysis_envelope(reference, sr, analysis_sr)
    max_lag = int(max_offset * analysis_sr)
    corr = signal.correlate(target_env, ref_env, mode="full", method="fft")
    lags = signal.correlation_lags(len(target_env), len(ref_env), mode="full")
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lag = int(lags[mask][np.argmax(corr[mask])])
    # Positive lag means target starts later than reference, so reference needs
    # left padding.
    return int(round(lag * sr / analysis_sr))


def analysis_envelope(data: np.ndarray, sr: int, analysis_sr: int) -> np.ndarray:
    mono = data.astype(np.float64)
    mono = mono - np.mean(mono)
    sos = signal.butter(4, [120.0, 6000.0], btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfiltfilt(sos, mono)
    env = np.abs(filtered)
    decimated = signal.resample_poly(env, analysis_sr, sr)
    decimated = decimated - np.mean(decimated)
    std = np.std(decimated)
    if std > 1e-9:
        decimated = decimated / std
    return decimated


def align_to_target(reference: np.ndarray, target_len: int, offset: int) -> np.ndarray:
    out = np.zeros((target_len, reference.shape[1]), dtype=np.float32)
    if offset >= 0:
        ref_start = 0
        out_start = offset
    else:
        ref_start = -offset
        out_start = 0
    length = min(target_len - out_start, len(reference) - ref_start)
    if length > 0:
        out[out_start : out_start + length] = reference[ref_start : ref_start + length]
    return out


def fit_channel_gains(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    gains = []
    for channel in range(2):
        x = reference[:, channel].astype(np.float64)
        y = target[:, channel].astype(np.float64)
        denom = float(np.dot(x, x))
        gains.append(0.0 if denom < 1e-12 else float(np.dot(x, y) / denom))
    return np.asarray(gains, dtype=np.float32)


def bandpass(data: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, data, axis=0).astype(np.float32)


def peak_normalize(data: np.ndarray, peak: float) -> np.ndarray:
    current = float(np.max(np.abs(data)))
    if current < 1e-9:
        return data
    return (data * (peak / current)).astype(np.float32)


def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))


def linear_to_db(value: float) -> float:
    return 20.0 * np.log10(max(abs(float(value)), 1e-12))


def write_readme(out_dir: Path, results: list[AlignmentResult]) -> None:
    lines = [
        "# Opening Music Subtraction",
        "",
        "These files align the official soundtrack `Main Title` against opening-credit audio and subtract it.",
        "The residue files are the ones to review for Spanish narrator/SFX recovery.",
        "",
        "| Target | Speed | Offset | Music gain L/R | Files |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for result in results:
        prefix = result.output_prefix
        lines.append(
            f"| `{Path(result.target).name}` | {result.speed_percent:.3f}% | "
            f"{result.offset_seconds:+.4f}s | "
            f"{result.music_gain_db_left:+.2f}/{result.music_gain_db_right:+.2f} dB | "
            f"`{prefix}_residue_music_subtracted_norm.wav`, `{prefix}_residue_presence_sfx_2500_12000.wav` |"
        )
    lines.extend(
        [
            "",
            "Suggested review order:",
            "",
            "1. `*_residue_music_subtracted_norm.wav` for overall recovered non-music material.",
            "2. `*_residue_presence_sfx_2500_12000.wav` for high-frequency SFX/edge content.",
            "3. `*_residue_voice_band_180_4200.wav` for the Spanish “Robotech” narrator region.",
            "4. `*_aligned_soundtrack.wav` against the target to verify the subtraction did not drift.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
