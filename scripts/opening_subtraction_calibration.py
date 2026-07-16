#!/usr/bin/env python3
"""Calibrate opening music subtraction on known quiet/SFX windows.

This is a diagnostic pass. It tries to answer a simple question before we trust
any "residual": can the clean soundtrack cancel mostly-music regions of the
opening? If not, the residual is just a worse full mix, not usable SFX.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav"
DEFAULT_MUSIC = ROOT / "work/review/opening_audio_rebuild_001/sources/07_soundtrack_main_title_opening_length.wav"
DEFAULT_OUT = ROOT / "work/review/opening_audio_subtraction_calibration_001"

WINDOWS = {
    "quiet_intro_000_005": (0.0, 5.0),
    "effects_mid_023_025": (23.0, 25.0),
    "effects_late_055_063": (55.0, 63.0),
}


@dataclass(frozen=True)
class Variant:
    name: str
    filters: tuple[str, ...]


VARIANTS = [
    Variant("global_gain", ()),
    Variant("quiet_gain", ()),
    Variant("quiet_gain_delay", ()),
    Variant("quiet_gain_delay_eq8", ("eq8",)),
    Variant("quiet_gain_delay_eq16", ("eq16",)),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--music", type=Path, default=DEFAULT_MUSIC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet-window", default="0:5", help="Calibration window as start:end seconds.")
    parser.add_argument("--delay-search-ms", type=float, default=20.0)
    parser.add_argument("--write-window-clips", action="store_true", default=True)
    args = parser.parse_args()

    target, sr = read_audio(args.target)
    music, music_sr = read_audio(args.music)
    if music_sr != sr:
        raise SystemExit(f"Sample-rate mismatch: target={sr}, music={music_sr}")
    length = min(len(target), len(music))
    target = target[:length]
    music = music[:length]
    quiet_start, quiet_end = parse_window(args.quiet_window)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "target": str(args.target),
        "music": str(args.music),
        "sample_rate": sr,
        "quiet_calibration_window_seconds": [quiet_start, quiet_end],
        "notes": [
            "The quiet_intro window should cancel very strongly if the soundtrack master matches.",
            "If quiet_intro residue remains close to target level, subtraction is not isolating SFX.",
            "effects windows are useful only if they are clearly higher than quiet_intro after calibration.",
        ],
        "variants": [],
    }

    for variant in VARIANTS:
        aligned = music.copy()
        if "delay" in variant.name:
            delay = estimate_small_delay(target, aligned, sr, quiet_start, quiet_end, args.delay_search_ms)
            aligned = fractional_delay(aligned, delay)
        else:
            delay = 0.0

        if "quiet_gain" in variant.name:
            gains = fit_gains(segment(target, sr, quiet_start, quiet_end), segment(aligned, sr, quiet_start, quiet_end))
        else:
            gains = fit_gains(target, aligned)
        aligned = aligned * gains[np.newaxis, :]

        if "eq8" in variant.filters:
            aligned = multiband_match(target, aligned, sr, quiet_start, quiet_end, bands=8)
        elif "eq16" in variant.filters:
            aligned = multiband_match(target, aligned, sr, quiet_start, quiet_end, bands=16)

        residual = target - aligned
        variant_dir = args.out_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        sf.write(variant_dir / f"{variant.name}_aligned_music.wav", aligned, sr, subtype="PCM_24")
        sf.write(variant_dir / f"{variant.name}_residual_raw.wav", residual, sr, subtype="PCM_24")
        sf.write(variant_dir / f"{variant.name}_residual_norm.wav", peak_normalize(residual, 0.85), sr, subtype="PCM_24")

        window_rows = []
        for label, (start, end) in WINDOWS.items():
            target_seg = segment(target, sr, start, end)
            music_seg = segment(aligned, sr, start, end)
            residual_seg = segment(residual, sr, start, end)
            target_db = db(rms(target_seg))
            residual_db = db(rms(residual_seg))
            aligned_db = db(rms(music_seg))
            ratio_db = residual_db - target_db
            window_rows.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "target_rms_dbfs": round(target_db, 3),
                    "aligned_music_rms_dbfs": round(aligned_db, 3),
                    "residual_rms_dbfs": round(residual_db, 3),
                    "residual_vs_target_db": round(ratio_db, 3),
                }
            )
            if args.write_window_clips:
                sf.write(variant_dir / f"{label}_target.wav", target_seg, sr, subtype="PCM_24")
                sf.write(variant_dir / f"{label}_aligned_music.wav", music_seg, sr, subtype="PCM_24")
                sf.write(variant_dir / f"{label}_residual.wav", residual_seg, sr, subtype="PCM_24")
                sf.write(variant_dir / f"{label}_residual_norm.wav", peak_normalize(residual_seg, 0.85), sr, subtype="PCM_24")

        report["variants"].append(
            {
                "name": variant.name,
                "delay_samples": round(delay, 6),
                "delay_seconds": round(delay / sr, 9),
                "gain_left_db": round(db(abs(gains[0])), 3),
                "gain_right_db": round(db(abs(gains[1])), 3),
                "outputs": str(variant_dir),
                "windows": window_rows,
            }
        )

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.out_dir, report)
    print(f"review_dir={args.out_dir}")
    print(f"report={args.out_dir / 'REPORT.md'}")
    return 0


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2], sr


def parse_window(value: str) -> tuple[float, float]:
    start, end = value.split(":", 1)
    return float(start), float(end)


def segment(data: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    return data[max(0, int(round(start * sr))) : min(len(data), int(round(end * sr)))]


def fit_gains(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    gains = []
    for channel in range(2):
        x = reference[:, channel].astype(np.float64)
        y = target[:, channel].astype(np.float64)
        denom = np.dot(x, x)
        gains.append(0.0 if denom < 1e-12 else float(np.dot(x, y) / denom))
    return np.asarray(gains, dtype=np.float32)


def estimate_small_delay(target: np.ndarray, reference: np.ndarray, sr: int, start: float, end: float, max_ms: float) -> float:
    t = segment(target.mean(axis=1), sr, start, end)
    r = segment(reference.mean(axis=1), sr, start, end)
    max_lag = int(round(max_ms * sr / 1000.0))
    corr = signal.correlate(t - np.mean(t), r - np.mean(r), mode="full", method="fft")
    lags = signal.correlation_lags(len(t), len(r), mode="full")
    mask = (lags >= -max_lag) & (lags <= max_lag)
    if not np.any(mask):
        return 0.0
    return float(lags[mask][np.argmax(corr[mask])])


def fractional_delay(data: np.ndarray, delay_samples: float) -> np.ndarray:
    # Positive delay shifts reference later.
    positions = np.arange(len(data), dtype=np.float64) - delay_samples
    out = np.zeros_like(data)
    base = np.arange(len(data), dtype=np.float64)
    for channel in range(data.shape[1]):
        out[:, channel] = np.interp(positions, base, data[:, channel], left=0.0, right=0.0)
    return out


def multiband_match(target: np.ndarray, reference: np.ndarray, sr: int, start: float, end: float, *, bands: int) -> np.ndarray:
    edges = np.geomspace(60.0, min(20000.0, sr / 2 - 100.0), bands + 1)
    out = np.zeros_like(reference, dtype=np.float64)
    target_cal = segment(target, sr, start, end)
    ref_cal = segment(reference, sr, start, end)
    for low, high in zip(edges[:-1], edges[1:]):
        sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
        ref_band = signal.sosfiltfilt(sos, reference, axis=0)
        ref_band_cal = signal.sosfiltfilt(sos, ref_cal, axis=0)
        target_band_cal = signal.sosfiltfilt(sos, target_cal, axis=0)
        gains = fit_gains(target_band_cal, ref_band_cal)
        out += ref_band * gains[np.newaxis, :]
    return out.astype(np.float32)


def rms(data: np.ndarray) -> float:
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))


def db(value: float) -> float:
    return float(20.0 * np.log10(max(abs(value), 1e-12)))


def peak_normalize(data: np.ndarray, peak: float) -> np.ndarray:
    current = float(np.max(np.abs(data))) if data.size else 0.0
    if current < 1e-9:
        return data
    return (data * (peak / current)).astype(np.float32)


def write_markdown(out_dir: Path, report: dict) -> None:
    lines = [
        "# Opening Subtraction Calibration",
        "",
        "This test calibrates soundtrack subtraction on low-effect windows, then checks whether known SFX windows stand out.",
        "",
        f"- Target: `{report['target']}`",
        f"- Music: `{report['music']}`",
        "",
        "| Variant | Delay | Gain L/R | Quiet residual vs target | 23-25 residual vs target | 55-63 residual vs target |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for variant in report["variants"]:
        windows = {row["label"]: row for row in variant["windows"]}
        lines.append(
            f"| `{variant['name']}` | {variant['delay_samples']:.3f} samples | "
            f"{variant['gain_left_db']:+.2f}/{variant['gain_right_db']:+.2f} dB | "
            f"{windows['quiet_intro_000_005']['residual_vs_target_db']:+.2f} dB | "
            f"{windows['effects_mid_023_025']['residual_vs_target_db']:+.2f} dB | "
            f"{windows['effects_late_055_063']['residual_vs_target_db']:+.2f} dB |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- A useful subtraction should make `quiet_intro_000_005` much lower than the target.",
            "- If the SFX windows are not clearly higher than the quiet window, the residual is not reliable SFX.",
            "- Review each variant folder's `*_target.wav`, `*_aligned_music.wav`, and `*_residual_norm.wav` clips.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
