#!/usr/bin/env python3
"""Fill a short audio gap with profile-shaped synthetic texture.

This is a review tool, not a production patcher. It estimates the spectrum and
level envelope around a gap, generates filtered noise with that profile, and
mixes it into the silent gap. It is useful when copied source texture contains
unwanted dialogue but a plain synthetic tone/noise layer sounds too generic.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="WAV with the gap already silenced")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--texture-output", type=Path, help="Optional isolated generated texture")
    parser.add_argument("--gap-start", type=float, required=True, help="Gap start seconds in the input file")
    parser.add_argument("--gap-end", type=float, required=True, help="Gap end seconds in the input file")
    parser.add_argument("--context", type=float, default=1.25, help="Seconds of audio before/after the gap to profile")
    parser.add_argument("--highpass", type=float, default=2500.0)
    parser.add_argument("--lowpass", type=float, default=10000.0)
    parser.add_argument("--gain-db", type=float, default=-20.0, help="Gain applied to generated texture")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--fade", type=float, default=0.045, help="Fade in/out seconds on generated texture")
    parser.add_argument(
        "--mode",
        choices=("profile-noise", "profile-noise-warble", "profile-noise-granular"),
        default="profile-noise",
    )
    return parser.parse_args()


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio[:, :2]


def bandpass(audio: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    nyquist = sr / 2
    low = max(20.0, min(low, nyquist - 100.0))
    high = max(low + 100.0, min(high, nyquist - 50.0))
    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def shaped_noise(context: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    """Create noise shaped by the average magnitude spectrum of context."""
    out = np.zeros((samples, context.shape[1]), dtype=np.float64)
    for ch in range(context.shape[1]):
        ctx = context[:, ch]
        if len(ctx) < 256 or rms(ctx) < 1e-8:
            out[:, ch] = rng.normal(0.0, 1.0, samples)
            continue
        n_fft = 2 ** math.ceil(math.log2(max(samples, 2048)))
        ctx_fft = np.fft.rfft(ctx * signal.windows.hann(len(ctx)), n=n_fft)
        mag = np.abs(ctx_fft)
        mag = signal.medfilt(mag, kernel_size=9)
        mag = mag / max(np.max(mag), 1e-8)
        noise = rng.normal(0.0, 1.0, n_fft)
        noise_fft = np.fft.rfft(noise)
        shaped = np.fft.irfft(noise_fft * mag, n=n_fft)[:samples]
        out[:, ch] = shaped
    return out


def granular_profile(context: np.ndarray, samples: int, rng: np.random.Generator, sr: int) -> np.ndarray:
    """Build a filler from tiny high-passed context grains with randomized polarity."""
    if len(context) < int(0.08 * sr):
        return shaped_noise(context, samples, rng)
    grain = max(int(0.045 * sr), 256)
    hop = max(grain // 3, 1)
    window = signal.windows.hann(grain)[:, None]
    out = np.zeros((samples + grain, context.shape[1]), dtype=np.float64)
    weight = np.zeros((samples + grain, 1), dtype=np.float64)
    starts = np.arange(0, len(context) - grain, max(1, grain // 4))
    if len(starts) == 0:
        return shaped_noise(context, samples, rng)
    for pos in range(0, samples, hop):
        src = int(rng.choice(starts))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        piece = context[src : src + grain] * sign * window
        out[pos : pos + grain] += piece
        weight[pos : pos + grain] += window
    out = out[:samples] / np.maximum(weight[:samples], 1e-6)
    return out


def main() -> int:
    args = parse_args()
    audio, sr = sf.read(args.input, always_2d=True)
    audio = ensure_stereo(audio).astype(np.float64)
    start = max(0, min(int(round(args.gap_start * sr)), len(audio)))
    end = max(start, min(int(round(args.gap_end * sr)), len(audio)))
    gap_len = end - start
    if gap_len <= 0:
        raise SystemExit("Gap length is zero")

    ctx_len = int(round(args.context * sr))
    before = audio[max(0, start - ctx_len) : start]
    after = audio[end : min(len(audio), end + ctx_len)]
    context = np.concatenate([before, after], axis=0)
    context = bandpass(context, sr, args.highpass, args.lowpass)

    rng = np.random.default_rng(args.seed)
    if args.mode == "profile-noise-granular":
        texture = granular_profile(context, gap_len, rng, sr)
    else:
        texture = shaped_noise(context, gap_len, rng)
        if args.mode == "profile-noise-warble":
            t = np.arange(gap_len, dtype=np.float64) / sr
            texture *= (0.85 + 0.15 * np.sin(2 * np.pi * 7.0 * t))[:, None]

    texture = bandpass(texture, sr, args.highpass, args.lowpass)
    target = max(rms(context), 1e-8)
    texture = texture / max(rms(texture), 1e-8) * target
    texture *= 10 ** (args.gain_db / 20.0)

    fade_samples = min(int(round(args.fade * sr)), gap_len // 2)
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples)[:, None]
        fade_out = np.linspace(1.0, 0.0, fade_samples)[:, None]
        texture[:fade_samples] *= fade_in
        texture[-fade_samples:] *= fade_out

    output = audio.copy()
    output[start:end] += texture
    output = np.clip(output, -0.98, 0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output, sr, subtype="PCM_24")
    if args.texture_output:
        args.texture_output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.texture_output, texture, sr, subtype="PCM_24")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
