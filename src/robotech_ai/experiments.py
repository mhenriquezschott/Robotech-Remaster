from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CleanVariant:
    """An FFmpeg-only cleanup recipe for preparing spa1 dialogue extraction tests."""

    name: str
    filtergraph: str
    notes: str


@dataclass(frozen=True)
class EnhanceVariant:
    """A conservative post-extraction voice polish recipe."""

    name: str
    filtergraph: str
    notes: str


@dataclass(frozen=True)
class FullMixVariant:
    """A conservative full-mix restoration recipe for the original spa1 track."""

    name: str
    filtergraph: str
    notes: str


CLEAN_VARIANTS: dict[str, CleanVariant] = {
    "raw": CleanVariant(
        name="raw",
        filtergraph="anull",
        notes="No cleanup. Baseline for judging whether cleanup hurts the voice.",
    ),
    "light": CleanVariant(
        name="light",
        filtergraph="highpass=f=70,lowpass=f=12500,afftdn=nr=6:nf=-55",
        notes="Conservative rumble/hiss reduction before separation.",
    ),
    "medium": CleanVariant(
        name="medium",
        filtergraph="highpass=f=80,lowpass=f=11000,afftdn=nr=10:nf=-52",
        notes="Stronger VHS hiss reduction. Watch for watery consonants.",
    ),
    "clarity": CleanVariant(
        name="clarity",
        filtergraph="highpass=f=75,lowpass=f=12000,afftdn=nr=7:nf=-55,equalizer=f=300:t=q:w=1.2:g=-1.5,equalizer=f=3200:t=q:w=1.0:g=1.5",
        notes="Light cleanup plus gentle mud cut and presence lift.",
    ),
}


FULL_MIX_VARIANTS: dict[str, FullMixVariant] = {
    "dehiss_only": FullMixVariant(
        name="dehiss_only",
        filtergraph="highpass=f=45,lowpass=f=15000,afftdn=nr=5:nf=-58:ad=0.25:gs=6",
        notes="Minimal full-track cleanup. Best if the old mix should stay almost untouched.",
    ),
    "vhs_gentle": FullMixVariant(
        name="vhs_gentle",
        filtergraph=(
            "highpass=f=50,lowpass=f=14000,"
            "afftdn=nr=7:nf=-57:ad=0.30:gs=8,"
            "equalizer=f=180:t=q:w=1.0:g=0.8,"
            "equalizer=f=2800:t=q:w=1.0:g=0.8,"
            "acompressor=threshold=-22dB:ratio=1.25:attack=12:release=180:makeup=1.0,"
            "alimiter=limit=0.95"
        ),
        notes="Gentle VHS dehiss plus small body/presence lift without aggressive leveling.",
    ),
    "vhs_dialogue_forward": FullMixVariant(
        name="vhs_dialogue_forward",
        filtergraph=(
            "highpass=f=60,lowpass=f=12500,"
            "afftdn=nr=7:nf=-56:ad=0.28:gs=8,"
            "equalizer=f=250:t=q:w=1.0:g=0.8,"
            "equalizer=f=2200:t=q:w=1.0:g=1.0,"
            "equalizer=f=3600:t=q:w=1.0:g=0.8,"
            "acompressor=threshold=-24dB:ratio=1.45:attack=8:release=150:makeup=1.1,"
            "alimiter=limit=0.95"
        ),
        notes="Pushes dialogue range forward while keeping original music/effects in the old mix.",
    ),
    "vhs_broadcast_full": FullMixVariant(
        name="vhs_broadcast_full",
        filtergraph=(
            "highpass=f=55,lowpass=f=13500,"
            "afftdn=nr=6:nf=-58:ad=0.25:gs=8,"
            "equalizer=f=160:t=q:w=1.0:g=0.9,"
            "equalizer=f=3000:t=q:w=1.0:g=0.8,"
            "speechnorm=e=8:r=0.0001:l=1,"
            "acompressor=threshold=-21dB:ratio=1.2:attack=12:release=190:makeup=1.05,"
            "alimiter=limit=0.95"
        ),
        notes="Full-mix broadcast-style leveling. Watch for background pumping.",
    ),
    "tone_only": FullMixVariant(
        name="tone_only",
        filtergraph=(
            "highpass=f=45,lowpass=f=15000,"
            "equalizer=f=180:t=q:w=1.0:g=0.9,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "acompressor=threshold=-23dB:ratio=1.18:attack=14:release=200:makeup=1.0,"
            "alimiter=limit=0.95"
        ),
        notes="No denoise; only EQ/very light control. Useful if denoise damages old music texture.",
    ),
}


ENHANCE_VARIANTS: dict[str, EnhanceVariant] = {
    "dehiss_light": EnhanceVariant(
        name="dehiss_light",
        filtergraph="afftdn=nr=4:nf=-58",
        notes="Very light FFT denoise. Use to check if hiss can be lowered without changing voice character.",
    ),
    "polish_light": EnhanceVariant(
        name="polish_light",
        filtergraph=(
            "highpass=f=90,lowpass=f=10500,"
            "afftdn=nr=4:nf=-58,"
            "equalizer=f=220:t=q:w=1.0:g=0.8,"
            "equalizer=f=3200:t=q:w=1.0:g=0.8,"
            "acompressor=threshold=-20dB:ratio=1.5:attack=8:release=120:makeup=1"
        ),
        notes="Gentle cleanup, small body/presence lift, and light compression.",
    ),
    "polish_warm": EnhanceVariant(
        name="polish_warm",
        filtergraph=(
            "highpass=f=85,lowpass=f=9500,"
            "afftdn=nr=5:nf=-58,"
            "equalizer=f=180:t=q:w=1.0:g=1.2,"
            "equalizer=f=2800:t=q:w=1.0:g=0.6,"
            "acompressor=threshold=-21dB:ratio=1.7:attack=10:release=140:makeup=1.2"
        ),
        notes="Slightly warmer and more controlled. Watch for boxiness or dull consonants.",
    ),
    "nlm_smooth": EnhanceVariant(
        name="nlm_smooth",
        filtergraph="anlmdn=s=0.00003:p=0.002:r=0.006:m=11",
        notes="Light non-local-means denoise. Watch for watery or phasey voice texture.",
    ),
    "speech_level": EnhanceVariant(
        name="speech_level",
        filtergraph="speechnorm=e=12.5:r=0.0001:l=1,alimiter=limit=0.95",
        notes="Speech leveler only. Tests intelligibility without denoise/EQ.",
    ),
    "presence_soft": EnhanceVariant(
        name="presence_soft",
        filtergraph=(
            "highpass=f=85,"
            "equalizer=f=250:t=q:w=1.0:g=0.8,"
            "equalizer=f=2200:t=q:w=1.0:g=0.9,"
            "equalizer=f=4200:t=q:w=1.2:g=0.7,"
            "acompressor=threshold=-24dB:ratio=1.45:attack=6:release=130:makeup=1.1,"
            "alimiter=limit=0.95"
        ),
        notes="Small body and presence lift with light compression. First choice if SR sounds artificial.",
    ),
    "presence_clear": EnhanceVariant(
        name="presence_clear",
        filtergraph=(
            "highpass=f=95,"
            "equalizer=f=320:t=q:w=1.0:g=-0.8,"
            "equalizer=f=1800:t=q:w=1.0:g=1.0,"
            "equalizer=f=3400:t=q:w=1.0:g=1.2,"
            "acompressor=threshold=-25dB:ratio=1.6:attack=5:release=120:makeup=1.2,"
            "alimiter=limit=0.95"
        ),
        notes="Clearer midrange/presence without adding artificial harmonics.",
    ),
    "presence_edge": EnhanceVariant(
        name="presence_edge",
        filtergraph=(
            "highpass=f=100,"
            "equalizer=f=280:t=q:w=1.0:g=-0.6,"
            "equalizer=f=2600:t=q:w=0.9:g=1.4,"
            "equalizer=f=5200:t=q:w=1.0:g=1.0,"
            "acompressor=threshold=-26dB:ratio=1.8:attack=4:release=110:makeup=1.3,"
            "alimiter=limit=0.94"
        ),
        notes="More bite and consonant edge. Watch for harshness.",
    ),
    "air_tiny": EnhanceVariant(
        name="air_tiny",
        filtergraph=(
            "highpass=f=90,"
            "equalizer=f=2400:t=q:w=1.1:g=0.8,"
            "equalizer=f=6500:t=q:w=1.2:g=0.8,"
            "aexciter=level_in=1:level_out=1:amount=0.25:drive=1.1:blend=0.18,"
            "acompressor=threshold=-24dB:ratio=1.45:attack=6:release=130:makeup=1.0,"
            "alimiter=limit=0.94"
        ),
        notes="Very light harmonic air. Reject if it creates VHS sparkle or synthetic consonants.",
    ),
    "broadcast_tight": EnhanceVariant(
        name="broadcast_tight",
        filtergraph=(
            "highpass=f=100,"
            "equalizer=f=180:t=q:w=1.0:g=0.6,"
            "equalizer=f=500:t=q:w=1.0:g=-0.7,"
            "equalizer=f=3000:t=q:w=1.0:g=1.1,"
            "speechnorm=e=10:r=0.0001:l=1,"
            "acompressor=threshold=-23dB:ratio=1.7:attack=5:release=100:makeup=1.0,"
            "alimiter=limit=0.94"
        ),
        notes="Level plus compact voice tone. Tests final-mix readiness rather than restoration purity.",
    ),
    "broadcast_open": EnhanceVariant(
        name="broadcast_open",
        filtergraph=(
            "highpass=f=85,"
            "equalizer=f=170:t=q:w=1.0:g=1.0,"
            "equalizer=f=420:t=q:w=1.0:g=-0.35,"
            "equalizer=f=2600:t=q:w=1.0:g=0.9,"
            "speechnorm=e=11.5:r=0.0001:l=1,"
            "acompressor=threshold=-21dB:ratio=1.35:attack=8:release=150:makeup=1.05,"
            "alimiter=limit=0.95"
        ),
        notes="More open broadcast tone with less compression and more low-mid strength.",
    ),
    "broadcast_strong": EnhanceVariant(
        name="broadcast_strong",
        filtergraph=(
            "highpass=f=80,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Keeps speech_level strength while adding a small presence lift.",
    ),
    "broadcast_presence_only": EnhanceVariant(
        name="broadcast_presence_only",
        filtergraph=(
            "highpass=f=85,"
            "equalizer=f=180:t=q:w=1.0:g=0.8,"
            "equalizer=f=2800:t=q:w=1.0:g=0.9,"
            "equalizer=f=4200:t=q:w=1.2:g=0.45,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "alimiter=limit=0.95"
        ),
        notes="Speech leveler plus EQ only, avoiding compressor strength loss.",
    ),
    "broadcast_strong_gate_soft": EnhanceVariant(
        name="broadcast_strong_gate_soft",
        filtergraph=(
            "highpass=f=80,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "agate=threshold=0.004:ratio=1.5:attack=12:release=220:range=0.18:knee=3:detection=rms,"
            "alimiter=limit=0.95"
        ),
        notes="broadcast_strong plus a soft expander to lower tiny between-word residue without hard muting.",
    ),
    "broadcast_strong_gate_medium": EnhanceVariant(
        name="broadcast_strong_gate_medium",
        filtergraph=(
            "highpass=f=80,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "agate=threshold=0.006:ratio=2.0:attack=10:release=180:range=0.10:knee=2.8:detection=rms,"
            "alimiter=limit=0.95"
        ),
        notes="Stronger floor cleanup. Watch for clipped syllable starts or unnatural dropouts.",
    ),
    "broadcast_strong_denoise_gate": EnhanceVariant(
        name="broadcast_strong_denoise_gate",
        filtergraph=(
            "highpass=f=80,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "afftdn=nr=3:nf=-65:rf=-55:ad=0.35:gs=6,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "agate=threshold=0.0045:ratio=1.6:attack=12:release=220:range=0.15:knee=3:detection=rms,"
            "alimiter=limit=0.95"
        ),
        notes="Very light FFT cleanup plus soft gate. Reject if it dulls consonants or makes watery artifacts.",
    ),
    "broadcast_strong_wavelet_soft": EnhanceVariant(
        name="broadcast_strong_wavelet_soft",
        filtergraph=(
            "highpass=f=80,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "afwtdn=sigma=0.0012:percent=35:softness=5,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Light wavelet denoise without gating. Tests whether visible residue can be smoothed instead of gated.",
    ),
    "gate_pre_level_low": EnhanceVariant(
        name="gate_pre_level_low",
        filtergraph=(
            "highpass=f=80,"
            "agate=threshold=0.010:ratio=3.0:attack=8:release=170:range=0.08:knee=2:detection=rms,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Diagnostic gate before speech leveling. Low threshold in the stronger ladder.",
    ),
    "gate_pre_level_mid": EnhanceVariant(
        name="gate_pre_level_mid",
        filtergraph=(
            "highpass=f=80,"
            "agate=threshold=0.018:ratio=4.0:attack=8:release=150:range=0.04:knee=1.8:detection=rms,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Diagnostic gate before speech leveling. Should visibly lower residue if threshold reaches the floor.",
    ),
    "gate_pre_level_high": EnhanceVariant(
        name="gate_pre_level_high",
        filtergraph=(
            "highpass=f=80,"
            "agate=threshold=0.030:ratio=6.0:attack=6:release=130:range=0.025:knee=1.5:detection=rms,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Aggressive diagnostic gate. Expected to show the limit and may cut quiet syllables.",
    ),
    "gate_pre_level_extreme": EnhanceVariant(
        name="gate_pre_level_extreme",
        filtergraph=(
            "highpass=f=80,"
            "agate=threshold=0.050:ratio=8.0:attack=5:release=110:range=0.015:knee=1.3:detection=rms,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Extreme diagnostic only. If this does not move the residue, it is not behaving like low-level noise.",
    ),
    "gate_pre_level_high_slow": EnhanceVariant(
        name="gate_pre_level_high_slow",
        filtergraph=(
            "highpass=f=80,"
            "agate=threshold=0.030:ratio=6.0:attack=10:release=240:range=0.025:knee=1.8:detection=rms,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Same threshold as high, with slower open/close to preserve word tails.",
    ),
    "gate_pre_level_high_more_cut": EnhanceVariant(
        name="gate_pre_level_high_more_cut",
        filtergraph=(
            "highpass=f=80,"
            "agate=threshold=0.036:ratio=6.5:attack=6:release=150:range=0.018:knee=1.5:detection=rms,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Slightly stronger than high. Tests whether more residue can be removed before voice damage appears.",
    ),
    "gate_pre_level_high_denoise": EnhanceVariant(
        name="gate_pre_level_high_denoise",
        filtergraph=(
            "highpass=f=80,"
            "afftdn=nr=3:nf=-62:rf=-54:ad=0.25:gs=8,"
            "agate=threshold=0.030:ratio=6.0:attack=6:release=150:range=0.025:knee=1.5:detection=rms,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Light spectral cleanup before the high gate. Reject if it makes watery consonants.",
    ),
    "gate_pre_level_high_declick": EnhanceVariant(
        name="gate_pre_level_high_declick",
        filtergraph=(
            "highpass=f=80,"
            "adeclick,"
            "agate=threshold=0.030:ratio=6.0:attack=6:release=150:range=0.025:knee=1.5:detection=rms,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Declick plus high gate, useful if between-word residue is made of tiny ticks/dots.",
    ),
    "expander_floor_soft": EnhanceVariant(
        name="expander_floor_soft",
        filtergraph=(
            "highpass=f=80,"
            "compand=attacks=0.015:decays=0.20:points=-90/-90|-60/-72|-45/-54|-34/-38|-24/-24|0/0:soft-knee=4,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Compand downward expander. Soft low-level reduction before leveling.",
    ),
    "expander_floor_mid": EnhanceVariant(
        name="expander_floor_mid",
        filtergraph=(
            "highpass=f=80,"
            "compand=attacks=0.010:decays=0.16:points=-90/-90|-58/-78|-44/-60|-34/-43|-25/-25|0/0:soft-knee=3,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Compand downward expander. Stronger floor reduction before leveling.",
    ),
    "expander_floor_high": EnhanceVariant(
        name="expander_floor_high",
        filtergraph=(
            "highpass=f=80,"
            "compand=attacks=0.008:decays=0.13:points=-90/-90|-56/-84|-42/-68|-32/-48|-24/-24|0/0:soft-knee=2,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Aggressive expander diagnostic. May damage low syllables but should visibly reduce true low-level residue.",
    ),
    "expander_floor_high_no_level": EnhanceVariant(
        name="expander_floor_high_no_level",
        filtergraph=(
            "highpass=f=80,"
            "compand=attacks=0.008:decays=0.13:points=-90/-90|-56/-84|-42/-68|-32/-48|-24/-24|0/0:soft-knee=2,"
            "equalizer=f=160:t=q:w=1.0:g=1.2,"
            "equalizer=f=3000:t=q:w=1.0:g=0.9,"
            "acompressor=threshold=-20dB:ratio=1.25:attack=10:release=170:makeup=1.15,"
            "alimiter=limit=0.95"
        ),
        notes="Same expander without speechnorm, to check whether leveling is re-amplifying the residue.",
    ),
}


def clean_variant_names() -> list[str]:
    return sorted(CLEAN_VARIANTS)


def enhance_variant_names() -> list[str]:
    return sorted(ENHANCE_VARIANTS)


def full_mix_variant_names() -> list[str]:
    return sorted(FULL_MIX_VARIANTS)


def require_clean_variant(name: str) -> CleanVariant:
    try:
        return CLEAN_VARIANTS[name]
    except KeyError as exc:
        allowed = ", ".join(clean_variant_names())
        raise SystemExit(f"Unknown cleanup variant: {name}. Allowed: {allowed}") from exc


def require_enhance_variant(name: str) -> EnhanceVariant:
    try:
        return ENHANCE_VARIANTS[name]
    except KeyError as exc:
        allowed = ", ".join(enhance_variant_names())
        raise SystemExit(f"Unknown enhancement variant: {name}. Allowed: {allowed}") from exc


def require_full_mix_variant(name: str) -> FullMixVariant:
    try:
        return FULL_MIX_VARIANTS[name]
    except KeyError as exc:
        allowed = ", ".join(full_mix_variant_names())
        raise SystemExit(f"Unknown full-mix variant: {name}. Allowed: {allowed}") from exc


def tool_path(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    candidates = [
        Path(sys.executable).parent / name,
        Path(sys.prefix) / "bin" / name,
        Path.cwd() / ".venv-separation" / "bin" / name,
        Path.cwd() / ".venv-deepfilter" / "bin" / name,
        Path.cwd() / ".venv-clearvoice" / "bin" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def installed_tools() -> dict[str, str | None]:
    return {
        "ffmpeg": tool_path("ffmpeg"),
        "ffprobe": tool_path("ffprobe"),
        "demucs": tool_path("demucs"),
        "audio-separator": tool_path("audio-separator"),
        "deepFilter": tool_path("deepFilter"),
    }


def build_separation_command(
    engine: str,
    input_path: Path,
    out_dir: Path,
    model: str | None,
    extra_models: list[str] | None = None,
    shifts: int | None = None,
    overlap: float | None = None,
    segment: int | None = None,
    no_split: bool = False,
    single_stem: str | None = None,
    sample_rate: int | None = None,
    model_file_dir: Path | None = None,
    ensemble_algorithm: str | None = None,
    ensemble_weights: list[float] | None = None,
    mdxc_overlap: int | None = None,
    mdxc_segment_size: int | None = None,
    mdxc_batch_size: int | None = None,
) -> list[str]:
    """Build an external voice-separation command without assuming the tool is installed."""
    if engine == "demucs":
        cmd = [tool_path("demucs") or "demucs", "--two-stems", "vocals", "--int24", "--out", str(out_dir)]
        if model:
            cmd += ["-n", model]
        if shifts is not None:
            cmd += ["--shifts", str(shifts)]
        if overlap is not None:
            cmd += ["--overlap", str(overlap)]
        if no_split:
            cmd += ["--no-split"]
        elif segment is not None:
            cmd += ["--segment", str(segment)]
        cmd.append(str(input_path))
        return cmd
    if engine == "audio-separator":
        cmd = [
            tool_path("audio-separator") or "audio-separator",
            str(input_path),
            "--output_dir",
            str(out_dir),
            "--output_format",
            "WAV",
        ]
        if model:
            cmd += ["--model_filename", model]
        if extra_models:
            cmd += ["--extra_models", *extra_models]
        if model_file_dir:
            cmd += ["--model_file_dir", str(model_file_dir)]
        if ensemble_algorithm:
            cmd += ["--ensemble_algorithm", ensemble_algorithm]
        if ensemble_weights:
            cmd += ["--ensemble_weights", *(str(weight) for weight in ensemble_weights)]
        if single_stem:
            cmd += ["--single_stem", single_stem]
        if sample_rate:
            cmd += ["--sample_rate", str(sample_rate)]
        if mdxc_overlap is not None:
            cmd += ["--mdxc_overlap", str(mdxc_overlap)]
        if mdxc_segment_size is not None:
            cmd += ["--mdxc_segment_size", str(mdxc_segment_size)]
        if mdxc_batch_size is not None:
            cmd += ["--mdxc_batch_size", str(mdxc_batch_size)]
        return cmd
    raise SystemExit("Unknown separation engine. Allowed: demucs, audio-separator")


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
