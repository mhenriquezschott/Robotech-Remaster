from __future__ import annotations

import math
import subprocess
from array import array
from pathlib import Path
from typing import Any


def stereo_correlation(path: Path, seconds: float | None = 180.0, sample_rate: int = 48_000) -> dict[str, Any]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    if seconds is not None:
        cmd[8:8] = ["-t", str(seconds)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    n = 0
    sum_l = 0.0
    sum_r = 0.0
    sum_l2 = 0.0
    sum_r2 = 0.0
    sum_lr = 0.0
    peak_l = 0
    peak_r = 0

    while True:
        chunk = proc.stdout.read(192_000)
        if not chunk:
            break
        samples = array("h")
        samples.frombytes(chunk)
        if len(samples) % 2:
            samples = samples[:-1]
        for i in range(0, len(samples), 2):
            left = samples[i]
            right = samples[i + 1]
            n += 1
            sum_l += left
            sum_r += right
            sum_l2 += left * left
            sum_r2 += right * right
            sum_lr += left * right
            peak_l = max(peak_l, abs(left))
            peak_r = max(peak_r, abs(right))

    _, stderr = proc.communicate()
    if proc.returncode:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))

    if n == 0:
        return {"samples": 0, "correlation": None, "classification": "empty"}

    mean_l = sum_l / n
    mean_r = sum_r / n
    var_l = max(sum_l2 / n - mean_l * mean_l, 0.0)
    var_r = max(sum_r2 / n - mean_r * mean_r, 0.0)
    cov = sum_lr / n - mean_l * mean_r
    denom = math.sqrt(var_l * var_r)
    corr = cov / denom if denom else 0.0
    rms_l = math.sqrt(sum_l2 / n) / 32768.0
    rms_r = math.sqrt(sum_r2 / n) / 32768.0
    balance_db = 20.0 * math.log10((rms_l + 1e-12) / (rms_r + 1e-12))

    if corr >= 0.995 and abs(balance_db) < 0.5:
        classification = "dual_mono_likely"
    elif corr >= 0.97:
        classification = "fake_or_narrow_stereo_likely"
    elif corr <= -0.6:
        classification = "phase_problem_possible"
    else:
        classification = "stereo_likely"

    return {
        "samples": n,
        "seconds_analyzed": n / sample_rate,
        "correlation": corr,
        "classification": classification,
        "rms_left": rms_l,
        "rms_right": rms_r,
        "balance_db_left_minus_right": balance_db,
        "peak_left": peak_l / 32768.0,
        "peak_right": peak_r / 32768.0,
    }


def energy_profile(path: Path, window_seconds: float = 1.0, sample_rate: int = 48_000) -> dict[str, Any]:
    """Measure short-window RMS/peak levels after decoding through FFmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    samples_per_window = max(int(sample_rate * window_seconds), 1)
    window_index = 0
    window_samples = 0
    sum_sq = 0.0
    peak = 0
    windows: list[dict[str, float | int]] = []

    def flush_window() -> None:
        nonlocal window_index, window_samples, sum_sq, peak
        if window_samples == 0:
            return
        rms = math.sqrt(sum_sq / window_samples) / 32768.0
        peak_value = peak / 32768.0
        windows.append(
            {
                "index": window_index,
                "start": window_index * window_seconds,
                "end": window_index * window_seconds + window_samples / sample_rate,
                "rms": rms,
                "rms_dbfs": 20.0 * math.log10(rms + 1e-12),
                "peak": peak_value,
                "peak_dbfs": 20.0 * math.log10(peak_value + 1e-12),
            }
        )
        window_index += 1
        window_samples = 0
        sum_sq = 0.0
        peak = 0

    while True:
        chunk = proc.stdout.read(192_000)
        if not chunk:
            break
        samples = array("h")
        samples.frombytes(chunk)
        for sample in samples:
            value = int(sample)
            sum_sq += value * value
            peak = max(peak, abs(value))
            window_samples += 1
            if window_samples >= samples_per_window:
                flush_window()

    flush_window()
    _, stderr = proc.communicate()
    if proc.returncode:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))

    return {
        "path": str(path),
        "window_seconds": window_seconds,
        "sample_rate": sample_rate,
        "windows": windows,
    }
