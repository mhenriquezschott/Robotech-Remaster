from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import shlex
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape as xml_escape

from .audio import energy_profile, stereo_correlation
from .experiments import (
    build_separation_command,
    clean_variant_names,
    enhance_variant_names,
    full_mix_variant_names,
    installed_tools,
    require_clean_variant,
    require_enhance_variant,
    require_full_mix_variant,
    write_manifest,
)
from .media import EpisodeAssets, discover_episodes, duration_seconds, ffprobe_full, probe_episode


DEFAULT_INPUT = Path("Robotech/bluraytrimcrop/MacrossSaga")
DEFAULT_WORK = Path("work")
DEFAULT_COVER_ASSET = Path("Robotech/images/macross_thumb_asset.png")
DEFAULT_SUMMARY_CONFIG = Path("config/next_episode_summaries.json")
DEFAULT_REFERENCE_CHAPTERS = Path("config/reference_episode_chapters.json")
DEFAULT_SPANISH_ALLOWLIST = Path("config/subtitles/spanish_allowlist.txt")
DEFAULT_SPA2_TAIL_EXTENSION_DIR = Path("work/ready_spa2_tail_extensions")
DEFAULT_READY_EPISODE_SEGMENT_DIR = Path("work/ready_episode_segments")
DEFAULT_SPA51_PRESERVED_CHANNEL_GAIN_DB = 3.0
DEFAULT_SPA51_CENTER_BED_GAIN_DB = 3.0
DEFAULT_SPA51_DIALOGUE_GAIN_DB = 0.0
EPISODE_AUDIO_PATCH_VERSION = "audiofix_v006"
EPISODE_VIDEO_TAIL_TRIMS_SECONDS: dict[str, float] = {
    "S01E36": 30.0,
}
DEFAULT_CONFIG = Path("config/episodes")
EPISODE_TITLES = {
    "S01E01": "Boobytrap",
    "S01E02": "Countdown",
    "S01E03": "Space Fold",
    "S01E04": "The Long Wait",
    "S01E05": "Transformation",
    "S01E06": "Blitzkrieg",
    "S01E07": "Bye-bye Mars",
    "S01E08": "Sweet Sixteen",
    "S01E09": "Miss Macross",
    "S01E10": "Blind Game",
    "S01E11": "First Contact",
    "S01E12": "The Big Escape",
    "S01E13": "Blue Wind",
    "S01E14": "Gloval's Report",
    "S01E15": "Homecoming",
    "S01E16": "Battle Cry",
    "S01E17": "Phantasm",
    "S01E18": "Farewell, Big Brother",
    "S01E19": "Bursting Point",
    "S01E20": "Paradise Lost",
    "S01E21": "A New Dawn",
    "S01E22": "Battle Hymn",
    "S01E23": "Reckless",
    "S01E24": "Showdown",
    "S01E25": "Wedding Bells",
    "S01E26": "The Messenger",
    "S01E27": "Force of Arms",
    "S01E28": "Reconstruction Blues",
    "S01E29": "The Robotech Masters",
    "S01E30": "Viva Miriya",
    "S01E31": "Khyron's Revenge",
    "S01E32": "Broken Heart",
    "S01E33": "A Rainy Night",
    "S01E34": "Private Time",
    "S01E35": "Season's Greetings",
    "S01E36": "To the Stars",
}
SUBTITLE_NAME_SPECS = {
    "english_clean": ("eng", "English Subtitles", "eng"),
    "spanish_translated": ("spa", "Spanish Subtitles", "spa"),
    "french_translated": ("fre", "French Subtitles", "fre"),
    "portuguese_translated": ("por", "Portuguese Subtitles", "por"),
    "italian_translated": ("ita", "Italian Subtitles", "ita"),
    "german_translated": ("ger", "German Subtitles", "ger"),
    "japanese_translated": ("jpn", "Japanese Subtitles", "jpn"),
}


@dataclass(frozen=True)
class EpisodeAudioPatch:
    """A reversible, episode-specific audio patch applied after cached restoration stems."""

    patch_id: str
    start_seconds: float
    end_seconds: float
    floor: float
    targets: tuple[str, ...]
    description: str
    method: str = "volume_floor"
    bridge_pre_seconds: float = 0.0
    crossfade_seconds: float = 0.0
    rubberband_options: str = ""
    replacement_path: str = ""
    replacement_source_seconds: float = 0.0
    replacement_gain_db: float = 0.0
    texture_gain_db: float = 0.0
    texture_edge_seconds: float = 0.0
    texture_overlay_path: str = ""
    insert_fade_in_seconds: float = 0.0
    insert_fade_out_seconds: float = 0.0
    insert_fade_in_curve: str = "tri"
    supersedes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Spa2TailExtension:
    """Episode-specific tail audio for the newer Spanish dub before end credits."""

    episode_id: str
    extension_id: str
    audio_path: Path
    duration_seconds: float
    source_path: str
    source_audio_stream: int
    source_start_seconds: float
    old_spa1_tail_gain_db: float
    description: str


@dataclass(frozen=True)
class PreparedEpisodeSegment:
    """A pre-authored episode-specific video/audio segment inserted at concat time."""

    episode_id: str
    segment_id: str
    manifest_path: Path
    insert: str
    outputs: dict[str, Path]
    suppresses_ready_audio_patch_ids: tuple[str, ...]
    description: str
    subtitle_sources: dict[str, Path]


@dataclass(frozen=True)
class Spa2TailExtensionAdjustment:
    variant: str
    source: Path
    output: Path
    extension: Spa2TailExtension
    command: list[str]

    def as_manifest(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "source": str(self.source),
            "output": str(self.output),
            "extension_id": self.extension.extension_id,
            "audio_path": str(self.extension.audio_path),
            "duration_seconds": self.extension.duration_seconds,
            "old_spa1_tail_gain_db": self.extension.old_spa1_tail_gain_db,
            "source_path": self.extension.source_path,
            "source_audio_stream": self.extension.source_audio_stream,
            "source_start_seconds": self.extension.source_start_seconds,
            "description": self.extension.description,
            "command": self.command,
        }


@dataclass(frozen=True)
class ReferenceEpisodeChapter:
    title: str
    start_seconds: float


EPISODE_AUDIO_PATCHES: dict[str, tuple[EpisodeAudioPatch, ...]] = {
    "S01E01": (
        EpisodeAudioPatch(
            patch_id="s01e01_narrator_colision_bridge_pre70_xfade4",
            start_seconds=35.207,
            end_seconds=35.255,
            floor=1.0,
            targets=("dialogue",),
            description=(
                "Replace the stray 'a' in the narrator phrase with a 70 ms "
                "rubberband bridge and 4 ms crossfades so 'coalision' reads "
                "closer to 'colision'."
            ),
            method="rubberband_bridge",
            bridge_pre_seconds=0.070,
            crossfade_seconds=0.004,
        ),
    ),
    "S01E03": (
        EpisodeAudioPatch(
            patch_id="s01e03_title_space_fold_voice_m59_edge80_m24_ex15_stable_seed23",
            start_seconds=43.080,
            end_seconds=44.167,
            floor=1.0,
            targets=("dialogue",),
            description=(
                "Replace the incorrect title narrator words with the recovered "
                "Space Fold title, retimed to the selected 1.087 s slot, with "
                "low edge texture from the removed Spanish slot plus the approved "
                "Stable Audio seed 23 inpaint texture at generated level."
            ),
            method="external_replacement_with_edge_texture",
            replacement_path="generated_audio/titlenarrator/s01e03_fixedtitle.wav",
            replacement_source_seconds=0.878417,
            replacement_gain_db=-5.94,
            texture_gain_db=-24.0,
            texture_edge_seconds=0.080,
            texture_overlay_path="generated_audio/titlenarrator/s01e03_title_texture_stable_seed23_as_generated.wav",
            insert_fade_in_seconds=0.035,
            insert_fade_out_seconds=0.015,
            insert_fade_in_curve="losi",
        ),
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotech-ai")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source directory with Robotech files")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK, help="Working output directory")
    sub = parser.add_subparsers(dest="command", required=True)

    inventory_p = sub.add_parser("inventory", help="Discover and probe episode files")
    inventory_p.add_argument("--no-probe", action="store_true", help="Only discover file groups")

    probe_p = sub.add_parser("probe", help="Probe one episode or all episodes")
    probe_p.add_argument("episode", nargs="?", help="Episode id, for example S01E01")

    metadata_p = sub.add_parser("archive-metadata", help="Save detailed ffprobe metadata for mux/rebuild planning")
    metadata_p.add_argument("episode", nargs="?", help="Episode id, for example S01E01")

    config_p = sub.add_parser("init-configs", help="Create per-episode JSON config files")
    config_p.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    config_p.add_argument("--overwrite", action="store_true")

    analyze_p = sub.add_parser("analyze-stereo", help="Estimate stereo/dual-mono behavior for Spanish tracks")
    analyze_p.add_argument("episode", nargs="?", help="Episode id, for example S01E01")
    analyze_p.add_argument("--seconds", type=float, default=180.0, help="Seconds to analyze; use 0 for the full episode")
    analyze_p.add_argument("--include-spa2", action="store_true", help="Also analyze spa2 as reference-only metadata")
    analyze_p.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    analyze_p.add_argument("--update-config", action="store_true", help="Write spa1 classification into episode config")

    extract_p = sub.add_parser("extract", help="Extract working WAV files for one episode")
    extract_p.add_argument("episode")
    extract_p.add_argument("--start", help="Optional start timestamp for lab extraction, for example 00:05:00")
    extract_p.add_argument("--duration", help="Optional duration for lab extraction, for example 00:00:45")
    extract_p.add_argument("--include-spa2", action="store_true", help="Extract spa2 reference audio too; restoration never uses it")
    extract_p.add_argument("--run", action="store_true", help="Actually run ffmpeg; otherwise print commands")

    sample_p = sub.add_parser("make-samples", help="Extract short comparison clips for one episode")
    sample_p.add_argument("episode")
    sample_p.add_argument("--start", default="00:05:00")
    sample_p.add_argument("--duration", default="00:00:45")
    sample_p.add_argument("--include-spa2", action="store_true", help="Include spa2 reference clips; restoration never uses it")
    sample_p.add_argument("--run", action="store_true", help="Actually run ffmpeg; otherwise print commands")

    clean_p = sub.add_parser("clean-spa1", help="Create FFmpeg cleanup variants from old spa1")
    clean_p.add_argument("episode")
    clean_p.add_argument("--variant", choices=clean_variant_names(), default="light")
    clean_p.add_argument("--start", help="Optional start timestamp for lab extraction, for example 00:05:00")
    clean_p.add_argument("--duration", help="Optional duration for lab extraction, for example 00:00:45")
    clean_p.add_argument("--source-mode", choices=["stereo", "mono-sum", "left", "right"], default="stereo")
    clean_p.add_argument("--run", action="store_true", help="Actually run ffmpeg; otherwise print commands")

    separate_p = sub.add_parser("separate-voice", help="Run or print an external voice-separation command")
    separate_p.add_argument("input_wav", type=Path, help="Cleaned spa1 WAV to separate")
    separate_p.add_argument("--engine", choices=["demucs", "audio-separator"], default="demucs")
    separate_p.add_argument("--model", help="Optional model name or filename for the selected engine")
    separate_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "06_separate_spa1")
    separate_p.add_argument("--shifts", type=int, help="Demucs random shifts; higher is slower but can improve stability")
    separate_p.add_argument("--overlap", type=float, help="Demucs overlap between chunks")
    separate_p.add_argument("--segment", type=int, help="Demucs segment size")
    separate_p.add_argument("--no-split", action="store_true", help="Demucs: process without chunk splitting")
    separate_p.add_argument("--single-stem", help="audio-separator: output only one stem, e.g. Vocals")
    separate_p.add_argument("--sample-rate", type=int, help="audio-separator output sample rate")
    separate_p.add_argument("--mdxc-overlap", type=int, help="audio-separator MDXC overlap, higher can reduce bleed/artifacts")
    separate_p.add_argument("--mdxc-segment-size", type=int, help="audio-separator MDXC segment size")
    separate_p.add_argument("--mdxc-batch-size", type=int, help="audio-separator MDXC batch size")
    separate_p.add_argument(
        "--model-file-dir",
        type=Path,
        default=DEFAULT_WORK / "models" / "audio-separator",
        help="audio-separator model cache directory",
    )
    separate_p.add_argument("--run", action="store_true", help="Actually run the external tool; otherwise print command")

    energy_p = sub.add_parser("energy-profile", help="Write short-window RMS/peak timeline for an audio file")
    energy_p.add_argument("input_wav", type=Path)
    energy_p.add_argument("--window", type=float, default=1.0)
    energy_p.add_argument("--out", type=Path)

    post_p = sub.add_parser("normalize-wav", help="Convert a WAV to project working format")
    post_p.add_argument("input_wav", type=Path)
    post_p.add_argument("output_wav", type=Path)
    post_p.add_argument("--sample-rate", type=int, default=48000)
    post_p.add_argument("--run", action="store_true", help="Actually run ffmpeg; otherwise print command")

    review_p = sub.add_parser("prepare-review", help="Copy files into a flat human-review folder")
    review_p.add_argument("name", help="Review set name, for example S01E01_voice_gate_001")
    review_p.add_argument("files", nargs="+", type=Path, help="Audio files to copy in order")
    review_p.add_argument("--labels", nargs="+", help="Optional labels matching the files")
    review_p.add_argument("--review-root", type=Path, default=DEFAULT_WORK / "review")

    gate_p = sub.add_parser("voice-gate", help="Run the current best spa1 voice-extraction test for one window")
    gate_p.add_argument("episode")
    gate_p.add_argument("--start", required=True)
    gate_p.add_argument("--duration", default="00:00:20")
    gate_p.add_argument("--variant", choices=clean_variant_names(), default="light")
    gate_p.add_argument("--source-mode", choices=["stereo", "mono-sum", "left", "right"], default="stereo")
    gate_p.add_argument("--review-name", required=True)
    gate_p.add_argument("--engine", choices=["audio-separator"], default="audio-separator")
    gate_p.add_argument("--model", help="audio-separator model filename")
    gate_p.add_argument("--single-stem", default="Vocals")
    gate_p.add_argument("--sample-rate", type=int, default=48000)
    gate_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "06_separate_spa1" / "audio_separator")
    gate_p.add_argument("--run", action="store_true", help="Actually run processing; otherwise print commands")

    audition_p = sub.add_parser("audition-windows", help="Create numbered source-only spa1 clips for choosing test windows")
    audition_p.add_argument("episode")
    audition_p.add_argument("--starts", nargs="+", required=True, help="Start timestamps to audition, for example 00:04:00 00:07:00")
    audition_p.add_argument("--duration", default="00:00:20")
    audition_p.add_argument("--variant", choices=clean_variant_names(), default="light")
    audition_p.add_argument("--source-mode", choices=["stereo", "mono-sum", "left", "right"], default="stereo")
    audition_p.add_argument("--review-name", required=True)
    audition_p.add_argument("--run", action="store_true", help="Actually run ffmpeg; otherwise print commands")

    shootout_p = sub.add_parser("model-shootout", help="Run several spa1 vocal-separation models for one window")
    shootout_p.add_argument("episode")
    shootout_p.add_argument("--start", help="Single start timestamp for the shootout")
    shootout_p.add_argument("--starts", nargs="+", help="Multiple start timestamps; creates one review folder per start")
    shootout_p.add_argument("--duration", default="00:00:20")
    shootout_p.add_argument("--variant", choices=clean_variant_names(), default="light")
    shootout_p.add_argument("--source-mode", choices=["stereo", "mono-sum", "left", "right"], default="stereo")
    shootout_p.add_argument("--review-name", required=True)
    shootout_p.add_argument(
        "--models",
        nargs="+",
        default=["default", "UVR-MDX-NET-Voc_FT.onnx", "Kim_Vocal_2.onnx", "MDX23C-8KFFT-InstVoc_HQ.ckpt"],
        help="audio-separator models to test; use 'default' for the package default BS-RoFormer",
    )
    shootout_p.add_argument("--single-stem", default="Vocals")
    shootout_p.add_argument("--sample-rate", type=int, default=48000)
    shootout_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "06_separate_spa1" / "audio_separator")
    shootout_p.add_argument("--run", action="store_true", help="Actually run processing; otherwise print commands")

    ensemble_p = sub.add_parser("ensemble-shootout", help="Compare strong single models against small model ensembles")
    ensemble_p.add_argument("episode")
    ensemble_p.add_argument("--start", help="Single start timestamp for the shootout")
    ensemble_p.add_argument("--starts", nargs="+", help="Multiple start timestamps; creates one review folder per start")
    ensemble_p.add_argument("--duration", default="00:00:20")
    ensemble_p.add_argument("--variant", choices=clean_variant_names(), default="light")
    ensemble_p.add_argument("--source-mode", choices=["stereo", "mono-sum", "left", "right"], default="stereo")
    ensemble_p.add_argument("--review-name", required=True)
    ensemble_p.add_argument("--single-stem", default="Vocals")
    ensemble_p.add_argument("--sample-rate", type=int, default=48000)
    ensemble_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "06_separate_spa1" / "audio_separator")
    ensemble_p.add_argument("--run", action="store_true", help="Actually run processing; otherwise print commands")

    center_p = sub.add_parser("center-devoice-shootout", help="Test English center dialogue removal while preserving the bed")
    center_p.add_argument("episode")
    center_p.add_argument("--starts", nargs="+", required=True, help="Start timestamps to test, for example 00:04:00 00:13:00")
    center_p.add_argument("--duration", default="00:00:20")
    center_p.add_argument("--review-name", required=True)
    center_p.add_argument(
        "--models",
        nargs="+",
        default=[
            "melband_roformer_instvoc_duality_v1.ckpt",
            "melband_roformer_instvox_duality_v2.ckpt",
            "MDX23C-8KFFT-InstVoc_HQ.ckpt",
        ],
        help="audio-separator models to test on the English center channel",
    )
    center_p.add_argument("--sample-rate", type=int, default=48000)
    center_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "07_separate_eng_center" / "audio_separator")
    center_p.add_argument("--run", action="store_true", help="Actually run processing; otherwise print commands")

    mix_p = sub.add_parser("center-mix-test", help="Build a short restored-Spanish 5.1 test clip")
    mix_p.add_argument("episode")
    mix_p.add_argument("--start", required=True, help="Start timestamp matching the supplied test stems")
    mix_p.add_argument("--duration", default="00:00:20")
    mix_p.add_argument("--spanish-dialogue", type=Path, required=True, help="Processed spa1 dialogue WAV to place in center")
    mix_p.add_argument("--center-bed", type=Path, required=True, help="De-voiced English center bed WAV")
    mix_p.add_argument("--dialogue-gain-db", type=float, default=0.0)
    mix_p.add_argument("--center-bed-gain-db", type=float, default=0.0)
    mix_p.add_argument("--review-name", required=True)
    mix_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "10_rebuild_51")
    mix_p.add_argument("--run", action="store_true", help="Actually run processing; otherwise print commands")

    episode_review_p = sub.add_parser("episode-review-build", help="Build a full-episode restored Spanish review mux")
    episode_review_p.add_argument("episode")
    episode_review_p.add_argument("--review-name", required=True)
    episode_review_p.add_argument("--spa-model", default="melband_roformer_instvoc_duality_v1.ckpt")
    episode_review_p.add_argument("--center-model", default="melband_roformer_instvoc_duality_v1.ckpt")
    episode_review_p.add_argument("--clean-variant", choices=clean_variant_names(), default="light")
    episode_review_p.add_argument("--enhance-variant", choices=enhance_variant_names(), default="broadcast_strong")
    episode_review_p.add_argument("--source-mode", choices=["stereo", "mono-sum", "left", "right"], default="stereo")
    episode_review_p.add_argument("--sample-rate", type=int, default=48000)
    episode_review_p.add_argument("--spa51-dialogue-gain-db", dest="dialogue_gain_db", type=float, default=DEFAULT_SPA51_DIALOGUE_GAIN_DB, help=f"Gain for restored Spanish dialogue before mixing into the new Spanish 5.1 center (default: {DEFAULT_SPA51_DIALOGUE_GAIN_DB:g})")
    episode_review_p.add_argument("--dialogue-gain-db", dest="dialogue_gain_db", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    episode_review_p.add_argument("--spa51-center-bed-gain-db", dest="center_bed_gain_db", type=float, default=DEFAULT_SPA51_CENTER_BED_GAIN_DB, help=f"Gain for de-voiced center bed before mixing into the new Spanish 5.1 center (default: {DEFAULT_SPA51_CENTER_BED_GAIN_DB:g})")
    episode_review_p.add_argument("--center-bed-gain-db", dest="center_bed_gain_db", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    episode_review_p.add_argument("--spa51-preserved-channel-gain-db", dest="preserved_channel_gain_db", type=float, default=DEFAULT_SPA51_PRESERVED_CHANNEL_GAIN_DB, help=f"Gain for preserved FL/FR/LFE/SL/SR before joining the new Spanish 5.1 (default: {DEFAULT_SPA51_PRESERVED_CHANNEL_GAIN_DB:g})")
    episode_review_p.add_argument("--preserved-channel-gain-db", dest="preserved_channel_gain_db", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    episode_review_p.add_argument("--spa1-fullmix-variant", choices=full_mix_variant_names(), default="vhs_broadcast_full")
    episode_review_p.add_argument("--restored-ac3-bitrate", default="640k")
    episode_review_p.add_argument("--spa1-stereo-ac3-bitrate", default="224k")
    episode_review_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "full_episode_review")
    episode_review_p.add_argument("--run", action="store_true", help="Actually run processing; otherwise print commands")

    final_episode_p = sub.add_parser("episode-final-build", help="Build final per-episode muxes from restored video variants")
    final_episode_p.add_argument("episode")
    final_episode_p.add_argument("--review-name", default="final_mux_v1")
    final_episode_p.add_argument("--proc-video-root", type=Path, default=Path("Robotech/proc/MacrossSaga"))
    final_episode_p.add_argument("--opening-root", type=Path, default=Path("Robotech/oc-ec/oc/assets"))
    final_episode_p.add_argument("--generation", choices=["1", "2", "3"], default="1", help="Opening narration generation to use")
    final_episode_p.add_argument("--no-opening", action="store_true", help="Do not prepend the restored opening-credit segment")
    final_episode_p.add_argument("--end-credit-root", type=Path, default=Path("Robotech/oc-ec/ec"))
    final_episode_p.add_argument("--no-end-credit", action="store_true", help="Do not append the restored end-credit segment")
    final_episode_p.add_argument("--no-auto-end-fade", action="store_true", help="Do not generate a short fade-to-black adjustment before end credits")
    final_episode_p.add_argument("--no-episode-adjustments", action="store_true", help="Skip per-episode audio/video adjustment stages before final concatenation")
    final_episode_p.add_argument("--no-video-episode-adjustments", action="store_true", help="Skip per-episode video adjustment stages before final concatenation")
    final_episode_p.add_argument("--no-audio-episode-adjustments", action="store_true", help="Skip per-episode audio adjustment stages before final concatenation")
    final_episode_p.add_argument("--end-fade-frames", type=int, default=5, help="Frames used for generated end fade-to-black adjustment segments")
    final_episode_p.add_argument("--end-fade-black-avg", type=float, default=2.0, help="Average luma at or below this value is considered black")
    final_episode_p.add_argument("--end-fade-black-max", type=int, default=12, help="Maximum luma at or below this value is considered black")
    final_episode_p.add_argument("--spa-model", default="melband_roformer_instvoc_duality_v1.ckpt")
    final_episode_p.add_argument("--center-model", default="melband_roformer_instvoc_duality_v1.ckpt")
    final_episode_p.add_argument("--clean-variant", choices=clean_variant_names(), default="light")
    final_episode_p.add_argument("--enhance-variant", choices=enhance_variant_names(), default="broadcast_strong")
    final_episode_p.add_argument("--source-mode", choices=["stereo", "mono-sum", "left", "right"], default="stereo")
    final_episode_p.add_argument("--sample-rate", type=int, default=48000)
    final_episode_p.add_argument("--spa51-dialogue-gain-db", dest="dialogue_gain_db", type=float, default=DEFAULT_SPA51_DIALOGUE_GAIN_DB, help=f"Gain for restored Spanish dialogue before mixing into the new Spanish 5.1 center (default: {DEFAULT_SPA51_DIALOGUE_GAIN_DB:g})")
    final_episode_p.add_argument("--dialogue-gain-db", dest="dialogue_gain_db", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    final_episode_p.add_argument("--spa51-center-bed-gain-db", dest="center_bed_gain_db", type=float, default=DEFAULT_SPA51_CENTER_BED_GAIN_DB, help=f"Gain for de-voiced center bed before mixing into the new Spanish 5.1 center (default: {DEFAULT_SPA51_CENTER_BED_GAIN_DB:g})")
    final_episode_p.add_argument("--center-bed-gain-db", dest="center_bed_gain_db", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    final_episode_p.add_argument("--spa51-preserved-channel-gain-db", dest="preserved_channel_gain_db", type=float, default=DEFAULT_SPA51_PRESERVED_CHANNEL_GAIN_DB, help=f"Gain for preserved FL/FR/LFE/SL/SR before joining the new Spanish 5.1 (default: {DEFAULT_SPA51_PRESERVED_CHANNEL_GAIN_DB:g})")
    final_episode_p.add_argument("--preserved-channel-gain-db", dest="preserved_channel_gain_db", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    final_episode_p.add_argument("--spa1-fullmix-variant", choices=full_mix_variant_names(), default="vhs_broadcast_full")
    final_episode_p.add_argument("--silence-start-seconds", type=float, default=0.5)
    final_episode_p.add_argument("--restored-ac3-bitrate", default="640k")
    final_episode_p.add_argument("--eng-ac3-bitrate", default="448k")
    final_episode_p.add_argument("--stereo-ac3-bitrate", default="224k")
    final_episode_p.add_argument("--if-exists", choices=["ask", "skip", "overwrite"], default="ask")
    final_episode_p.add_argument("--rebuild-intermediates", action="store_true", help="Regenerate cached intermediate audio/separation files instead of reusing existing ones")
    final_episode_p.add_argument("--repair-recipe", action="append", type=Path, default=[], help="Apply a repair-tool recipe JSON to the restored Spanish dialogue before 5.1 mixing; can be used more than once")
    final_episode_p.add_argument("--repair-recipe-dir", type=Path, default=Path("work/repair_projects/final_build"), help="Optional folder containing per-episode repair recipe JSON files to apply during final build")
    final_episode_p.add_argument("--no-repair-recipe-dir", action="store_true", help="Do not auto-load repair recipes from --repair-recipe-dir")
    final_episode_p.add_argument("--ready-patch-dir", type=Path, default=Path("work/ready_audio_patches"), help="Folder with ready WAV+JSON audio patches discovered per episode")
    final_episode_p.add_argument("--no-ready-patches", action="store_true", help="Do not auto-load ready audio patch clips from --ready-patch-dir")
    final_episode_p.add_argument("--spa2-tail-extension-dir", type=Path, default=DEFAULT_SPA2_TAIL_EXTENSION_DIR, help="Folder with episode-specific Spanish redubbing tail extensions")
    final_episode_p.add_argument("--no-spa2-tail-extensions", action="store_true", help="Do not insert episode-specific Spanish redubbing tail extensions before end credits")
    final_episode_p.add_argument("--reference-chapter-file", type=Path, default=DEFAULT_REFERENCE_CHAPTERS, help="Local JSON with episode chapter marks harvested from reference MKVs")
    final_episode_p.add_argument("--no-reference-episode-chapters", action="store_true", help="Do not use harvested reference chapter marks")
    final_episode_p.add_argument("--subtitle-dir", type=Path, default=DEFAULT_WORK / "review" / "subtitles")
    final_episode_p.add_argument("--no-embed-subtitles", action="store_true", help="Do not embed available SRT subtitles into the review MKVs")
    final_episode_p.add_argument("--no-cover-art", action="store_true", help="Do not attach cover art to the review MKVs")
    final_episode_p.add_argument("--cover-asset", type=Path, default=DEFAULT_COVER_ASSET, help="Optional image to attach as cover art before falling back to first frame")
    final_episode_p.add_argument("--no-copy-sources", action="store_true", help="Do not copy episode source video/audio into the review folder")
    final_episode_p.add_argument("--keep-intermediate-segments", action="store_true", help="Keep large episode-only temporary segments after final concat")
    final_episode_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "final_episode_mux")
    final_episode_p.add_argument("--run", action="store_true", help="Actually run processing; otherwise print commands")

    spa2_tail_p = sub.add_parser("spa2-tail-extension", help="Prepare an episode-specific tail extension for the newer Spanish dub")
    spa2_tail_p.add_argument("episode", help="Episode id, for example S01E14")
    spa2_tail_p.add_argument("--reference-mkv", type=Path, required=True, help="Reference MKV containing the longer newer Spanish dub")
    spa2_tail_p.add_argument("--reference-episode-start", type=float, required=True, help="Seconds where the matching episode body starts in the reference MKV")
    spa2_tail_p.add_argument("--reference-audio-stream", type=int, default=3, help="Reference MKV audio stream index for the newer Spanish dub, usually 3")
    spa2_tail_p.add_argument("--tail-start", type=float, help="Override absolute seconds where the tail extraction starts in the reference MKV")
    spa2_tail_p.add_argument("--duration", type=float, default=2.2, help="Tail duration to extract, in seconds")
    spa2_tail_p.add_argument("--fade-in", type=float, default=0.025, help="Tiny fade-in at the start of the extracted tail to avoid a join click")
    spa2_tail_p.add_argument("--fade-out", type=float, default=0.15, help="Fade out at the end of the extracted tail, in seconds")
    spa2_tail_p.add_argument("--old-spa1-tail-gain-db", type=float, help="Manual gain for the same tail when inserted into the processed old Spanish stereo track")
    spa2_tail_p.add_argument("--old-spa1-match-audio", type=Path, help="Processed old Spanish stereo track used to auto-match the tail level")
    spa2_tail_p.add_argument("--old-spa1-match-window", type=float, default=1.0, help="Seconds from the destination track ending used for old Spanish stereo level matching")
    spa2_tail_p.add_argument("--old-spa1-max-gain-db", type=float, default=12.0, help="Clamp auto old Spanish stereo tail gain to +/- this many dB")
    spa2_tail_p.add_argument("--extension-id", help="Folder/id for this tail extension. Defaults to s01e##_spa2_tail_v001")
    spa2_tail_p.add_argument("--out-dir", type=Path, default=DEFAULT_SPA2_TAIL_EXTENSION_DIR)
    spa2_tail_p.add_argument("--review-dir", type=Path, default=DEFAULT_WORK / "review")
    spa2_tail_p.add_argument("--run", action="store_true")

    reference_chapters_p = sub.add_parser("reference-chapters", help="Collect reusable chapter marks from reference Robotech MKVs")
    reference_chapters_p.add_argument("--reference-root", type=Path, required=True, help="Folder containing reference MKVs named like Robotech-S01E14.mkv")
    reference_chapters_p.add_argument("--pattern", default="Robotech-S01E*.mkv", help="Glob pattern used under --reference-root")
    reference_chapters_p.add_argument("--out", type=Path, default=DEFAULT_REFERENCE_CHAPTERS, help="Local JSON chapter map used by episode-final-build")
    reference_chapters_p.add_argument("--run", action="store_true")

    episode_only_p = sub.add_parser("episode-only-from-review", help="Mux episode-only restored-audio videos from an existing final review folder")
    episode_only_p.add_argument("episode")
    episode_only_p.add_argument("--review-name", default="final_mux_oc_ec_v1")
    episode_only_p.add_argument("--proc-video-root", type=Path, default=Path("Robotech/proc/MacrossSaga"))
    episode_only_p.add_argument("--if-exists", choices=["ask", "skip", "overwrite"], default="ask")
    episode_only_p.add_argument("--run", action="store_true", help="Actually run muxing; otherwise print commands")

    adjust_p = sub.add_parser("episode-adjustments", help="Inspect/build per-episode adjustment segments such as end fade-to-black")
    adjust_p.add_argument("episode", nargs="?", default="all", help="Episode id, for example S01E01, or all")
    adjust_p.add_argument("--review-name", default="final_mux_oc_ec_v1")
    adjust_p.add_argument("--proc-video-root", type=Path, default=Path("Robotech/proc/MacrossSaga"))
    adjust_p.add_argument("--fade-frames", type=int, default=5)
    adjust_p.add_argument("--black-avg", type=float, default=2.0)
    adjust_p.add_argument("--black-max", type=int, default=12)
    adjust_p.add_argument("--eng-ac3-bitrate", default="448k")
    adjust_p.add_argument("--restored-ac3-bitrate", default="640k")
    adjust_p.add_argument("--stereo-ac3-bitrate", default="224k")
    adjust_p.add_argument("--run", action="store_true", help="Actually write needed adjustment segments; otherwise print the plan")

    subtitles_p = sub.add_parser("prepare-subtitles", help="Extract English PGS subtitles and clean Spanish SSA subtitles")
    subtitles_p.add_argument("episode", nargs="?", default="all", help="Episode id, for example S01E01, or all")
    subtitles_p.add_argument("--eng-root", type=Path, default=Path("Robotech/oc-ec-engsubs-mkv"))
    subtitles_p.add_argument("--spa-root", type=Path, default=Path("Robotech/subs/spa/ssa"))
    subtitles_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "review" / "subtitles")
    subtitles_p.add_argument("--run", action="store_true", help="Actually create subtitles; otherwise print commands")

    export_done_p = sub.add_parser("export-done", help="Copy final review videos to the Season 1 done folder with subtitles")
    export_done_p.add_argument("episode", nargs="?", default="all", help="Episode id, for example S01E01, or all")
    export_done_p.add_argument("--review-name", default="final_mux_oc_ec_v1")
    export_done_p.add_argument("--proc-video-root", type=Path, default=Path("Robotech/proc/MacrossSaga"))
    export_done_p.add_argument("--subtitle-dir", type=Path, default=DEFAULT_WORK / "review" / "subtitles")
    export_done_p.add_argument("--out-dir", type=Path, default=Path("Robotech/done/Robotech/Season 1"))
    export_done_p.add_argument("--if-exists", choices=["ask", "skip", "overwrite"], default="ask")
    export_done_p.add_argument("--no-cover-art", action="store_true", help=argparse.SUPPRESS)
    export_done_p.add_argument("--run", action="store_true", help="Actually create files; otherwise print commands")

    ocr_subs_p = sub.add_parser("ocr-english-subtitles", help="OCR embedded English PGS subtitles into text SRT")
    ocr_subs_p.add_argument("episode", nargs="?", default="all", help="Episode id, for example S01E01, or all")
    ocr_subs_p.add_argument("--eng-root", type=Path, default=Path("Robotech/oc-ec-engsubs-mkv"))
    ocr_subs_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "review" / "subtitles")
    ocr_subs_p.add_argument("--tesseract", default="tesseract")
    ocr_subs_p.add_argument("--crop-top-ratio", type=float, default=0.58, help="Deprecated; OCR now uses the full subtitle frame to catch top and bottom subtitles")
    ocr_subs_p.add_argument("--min-display-packet-size", type=int, default=1000)
    ocr_subs_p.add_argument("--keep-images", action="store_true", help="Keep OCR frame crops for review")
    ocr_subs_p.add_argument("--run", action="store_true", help="Actually OCR subtitles; otherwise print the plan")

    retime_spa_p = sub.add_parser("retime-spanish-subtitles", help="Apply English OCR subtitle timing to cleaned Spanish SRT text")
    retime_spa_p.add_argument("episode", nargs="?", default="all", help="Episode id, for example S01E01, or all")
    retime_spa_p.add_argument("--subtitle-dir", type=Path, default=DEFAULT_WORK / "review" / "subtitles")
    retime_spa_p.add_argument("--run", action="store_true", help="Actually write retimed Spanish SRT files")

    translate_spa_p = sub.add_parser("translate-spanish-subtitles", help="Translate English OCR SRT cues into Latin American Spanish")
    translate_spa_p.add_argument("episode", nargs="?", default="all", help="Episode id, for example S01E01, or all")
    translate_spa_p.add_argument("--provider", choices=["prompt-batch", "hf", "ollama"], default="prompt-batch")
    translate_spa_p.add_argument("--model", default="gemma_3_27b_it")
    translate_spa_p.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    translate_spa_p.add_argument("--model-config", type=Path, default=Path("config/llm_models/gemma_3_27b_it_subtitle_v001.json"))
    translate_spa_p.add_argument(
        "--model-cache",
        type=Path,
        default=Path("/mnt/088E1D428E1D29A8/Documents/Documentos Trabajos UACh/2026/research/AI-Math/code/local_models/cache"),
        help="Hugging Face cache directory for the local HF runner",
    )
    translate_spa_p.add_argument("--offline", action="store_true", help="Run the HF translator with local_files_only/HF offline mode")
    translate_spa_p.add_argument("--llm-python", default="python", help="Python executable from the LLM virtualenv")
    translate_spa_p.add_argument("--limit-chunks", type=int, help="HF smoke-test limit; omit for the full episode")
    translate_spa_p.add_argument("--subtitle-dir", type=Path, default=DEFAULT_WORK / "review" / "subtitles")
    translate_spa_p.add_argument("--glossary", type=Path, default=Path("config/subtitles/robotech_glossary.json"))
    translate_spa_p.add_argument("--chunk-size", type=int, default=20)
    translate_spa_p.add_argument("--temperature", type=float, default=0.1)
    translate_spa_p.add_argument("--retries", type=int, default=3, help="HF runner retries per failed subtitle chunk")
    translate_spa_p.add_argument("--overwrite", action="store_true", help="Rebuild translated SRTs that already exist")
    translate_spa_p.add_argument("--run", action="store_true", help="Write prompt batches or call the selected local model")

    import_spa_p = sub.add_parser("import-spanish-translation", help="Import LLM JSONL translation responses and write Spanish SRT")
    import_spa_p.add_argument("episode", help="Episode id, for example S01E01")
    import_spa_p.add_argument("--responses", type=Path, required=True, help="JSONL file with translations or raw model responses")
    import_spa_p.add_argument("--subtitle-dir", type=Path, default=DEFAULT_WORK / "review" / "subtitles")
    import_spa_p.add_argument("--run", action="store_true", help="Actually write the translated Spanish SRT")

    speech_map_p = sub.add_parser("speech-map", help="Create ASR word/phrase timestamp maps for an episode audio track")
    speech_map_p.add_argument("episode", help="Episode id, for example S01E01")
    speech_map_p.add_argument("--audio", type=Path, help="Explicit audio WAV to transcribe; overrides --source/--review-name")
    speech_map_p.add_argument("--source", choices=["dialogue", "spa1_fullmix"], default="dialogue", help="Default episode audio source to map")
    speech_map_p.add_argument("--review-name", default="final_mux_oc_ec_v1")
    speech_map_p.add_argument("--engine", choices=["whisperx", "faster-whisper"], default="whisperx")
    speech_map_p.add_argument("--model", default="large-v3")
    speech_map_p.add_argument("--language", default="es")
    speech_map_p.add_argument("--device", default="cuda")
    speech_map_p.add_argument("--compute-type", default="float16")
    speech_map_p.add_argument("--batch-size", type=int, default=8)
    speech_map_p.add_argument("--python", type=Path, default=Path(".venv-asr/bin/python"), help="Python interpreter with WhisperX/faster-whisper installed")
    speech_map_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "speech_maps")
    speech_map_p.add_argument("--overwrite", action="store_true", help="Rebuild speech-map outputs that already exist")
    speech_map_p.add_argument("--run", action="store_true", help="Actually run ASR; otherwise print the command")

    speech_find_p = sub.add_parser("speech-find", help="Find a phrase inside a generated speech-map word timeline")
    speech_find_p.add_argument("episode", help="Episode id, for example S01E01")
    speech_find_p.add_argument("phrase", help="Phrase to find, for example 'rumbo de colision'")
    speech_find_p.add_argument("--source", choices=["dialogue", "spa1_fullmix"], default="dialogue")
    speech_find_p.add_argument("--map", dest="map_path", type=Path, help="Explicit speech_map.json file")
    speech_find_p.add_argument("--speech-map-dir", type=Path, default=DEFAULT_WORK / "speech_maps")
    speech_find_p.add_argument("--around", help="Optional approximate time, for example 00:22:19 or 1339")
    speech_find_p.add_argument("--window", type=float, default=45.0, help="Seconds around --around to search")
    speech_find_p.add_argument("--limit", type=int, default=8)

    tts_plan_p = sub.add_parser("tts-summary-plan", help="Create a phrase plan for Qwen3-TTS next-episode narrator regeneration")
    tts_plan_p.add_argument("episode", help="Episode id, for example S01E04, or all")
    tts_plan_p.add_argument("--start", help="Summary start timestamp, for example 00:21:30. Optional when config has a start.")
    tts_plan_p.add_argument("--end", help="Summary end timestamp, for example 00:22:10. Defaults to the last ASR phrase after --start.")
    tts_plan_p.add_argument("--source", choices=["dialogue", "spa1_fullmix"], default="dialogue")
    tts_plan_p.add_argument("--speech-map-dir", type=Path, default=DEFAULT_WORK / "speech_maps")
    tts_plan_p.add_argument("--out-dir", type=Path, default=Path("generated_audio/next_episode_summary"))
    tts_plan_p.add_argument("--summary-id", default="summary_v001")
    tts_plan_p.add_argument("--config", type=Path, default=DEFAULT_SUMMARY_CONFIG)
    tts_plan_p.add_argument("--overwrite", action="store_true", help="Rewrite existing phrase plans")
    tts_plan_p.add_argument("--run", action="store_true", help="Actually write the phrase plan")

    spellcheck_p = sub.add_parser("subtitle-spellcheck", help="Report suspicious Spanish words in subtitles and ASR speech maps")
    spellcheck_p.add_argument("episode", help="Episode id, for example S01E01, or all")
    spellcheck_p.add_argument("--source", choices=["subtitles", "speech-map", "both"], default="both")
    spellcheck_p.add_argument("--subtitle-kind", default="spanish_translated", help="Subtitle stem kind, e.g. spanish_translated or spanish_clean")
    spellcheck_p.add_argument("--subtitle-dir", type=Path, default=DEFAULT_WORK / "review" / "subtitles")
    spellcheck_p.add_argument("--speech-map-dir", type=Path, default=DEFAULT_WORK / "speech_maps")
    spellcheck_p.add_argument("--hunspell-dic", type=Path, default=Path("/usr/share/hunspell/es_ES.dic"))
    spellcheck_p.add_argument("--glossary", type=Path, default=Path("config/subtitles/robotech_glossary.json"))
    spellcheck_p.add_argument("--allowlist", type=Path, default=DEFAULT_SPANISH_ALLOWLIST)
    spellcheck_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "review" / "subtitle_spellcheck")
    spellcheck_p.add_argument("--min-length", type=int, default=3)
    spellcheck_p.add_argument("--run", action="store_true", help="Write JSON/Markdown reports")

    review_workbook_p = sub.add_parser("subtitle-review-workbook", help="Create an XLSX workbook for reviewing and correcting Spanish subtitle issues")
    review_workbook_p.add_argument("episode", help="Episode id, for example S01E01, or all")
    review_workbook_p.add_argument("--source", choices=["subtitles", "speech-map", "both"], default="subtitles")
    review_workbook_p.add_argument("--subtitle-kind", default="spanish_translated")
    review_workbook_p.add_argument("--subtitle-dir", type=Path, default=DEFAULT_WORK / "review" / "subtitles")
    review_workbook_p.add_argument("--speech-map-dir", type=Path, default=DEFAULT_WORK / "speech_maps")
    review_workbook_p.add_argument("--hunspell-dic", type=Path, default=Path("/usr/share/hunspell/es_ES.dic"))
    review_workbook_p.add_argument("--glossary", type=Path, default=Path("config/subtitles/robotech_glossary.json"))
    review_workbook_p.add_argument("--allowlist", type=Path, default=DEFAULT_SPANISH_ALLOWLIST)
    review_workbook_p.add_argument("--out", type=Path, default=DEFAULT_WORK / "review" / "subtitle_language_review" / "spanish_language_review.xlsx")
    review_workbook_p.add_argument("--min-length", type=int, default=3)
    review_workbook_p.add_argument("--run", action="store_true", help="Write the XLSX workbook")

    tts_generate_p = sub.add_parser("tts-summary-generate", help="Generate Qwen3-TTS narrator takes from a summary phrase plan")
    tts_generate_p.add_argument("episode", help="Episode id, for example S01E04")
    tts_generate_p.add_argument("--plan", type=Path, help="Explicit phrase_plan.json path")
    tts_generate_p.add_argument("--start", help="Summary start timestamp; creates the phrase plan first when --plan is missing")
    tts_generate_p.add_argument("--end", help="Summary end timestamp; creates the phrase plan first when --plan is missing")
    tts_generate_p.add_argument("--source", choices=["dialogue", "spa1_fullmix"], default="dialogue")
    tts_generate_p.add_argument("--speech-map-dir", type=Path, default=DEFAULT_WORK / "speech_maps")
    tts_generate_p.add_argument("--summary-id", default="summary_v001")
    tts_generate_p.add_argument("--out-dir", type=Path, default=Path("generated_audio/next_episode_summary"))
    tts_generate_p.add_argument("--qwen-root", type=Path, default=Path("/home/mhenriquez/AI/qwen3-tts-gradio"))
    tts_generate_p.add_argument("--qwen-python", type=Path, default=Path("/home/mhenriquez/AI/qwen3-tts-gradio/venv/bin/python"))
    tts_generate_p.add_argument("--ref-audio", type=Path, default=Path("train/mainnarrator/mainnarrator01.wav"))
    tts_generate_p.add_argument("--ref-text", type=Path, default=Path("train/mainnarrator/mainnarrator01.txt"))
    tts_generate_p.add_argument("--takes", type=int, default=10)
    tts_generate_p.add_argument("--phrases", nargs="+", help="Only generate these phrase numbers, e.g. 03 or 01 04")
    tts_generate_p.add_argument("--replace-phrase", action="store_true", help="With --phrases, replace existing takes instead of appending")
    tts_generate_p.add_argument("--model-size", choices=["0.6B", "1.7B"], default="1.7B")
    tts_generate_p.add_argument("--language", default="Spanish", help="Qwen language label; the Gradio app uses 'Spanish'")
    tts_generate_p.add_argument("--chunk-size", type=int, default=200)
    tts_generate_p.add_argument("--chunk-gap", type=float, default=0.0)
    tts_generate_p.add_argument("--seed-base", type=int, default=-1)
    tts_generate_p.add_argument("--exact-seed", type=int, help="Use this exact seed for the first generated take; later takes increment by 1")
    tts_generate_p.add_argument("--temperature", type=float, help="Optional Qwen Base sampling temperature for new takes")
    tts_generate_p.add_argument("--top-p", type=float, help="Optional Qwen Base nucleus sampling value for new takes")
    tts_generate_p.add_argument("--top-k", type=int, help="Optional Qwen Base top-k sampling value for new takes")
    tts_generate_p.add_argument("--repetition-penalty", type=float, help="Optional Qwen Base repetition penalty for new takes")
    tts_generate_p.add_argument("--subtalker-temperature", type=float, help="Optional Qwen subtalker sampling temperature for new takes")
    tts_generate_p.add_argument("--subtalker-top-p", type=float, help="Optional Qwen subtalker nucleus sampling value for new takes")
    tts_generate_p.add_argument("--subtalker-top-k", type=int, help="Optional Qwen subtalker top-k sampling value for new takes")
    tts_generate_p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    tts_generate_p.add_argument("--x-vector-only", action="store_true")
    tts_generate_p.add_argument("--overwrite", action="store_true")
    tts_generate_p.add_argument("--assemble-only", action="store_true", help="Rebuild selected-takes preview without generating new audio")
    tts_generate_p.add_argument("--fit-to-slots", action="store_true", help="With --assemble-only, speed up selected takes that exceed their phrase slots")
    tts_generate_p.add_argument("--slot-margin", type=float, default=0.0, help="Seconds to leave free at the end of each phrase slot when fitting")
    tts_generate_p.add_argument("--speed-all-percent", type=float, default=100.0, help="With --assemble-only, apply one tempo change to all selected takes")
    tts_generate_p.add_argument("--avoid-overlap", action="store_true", help="With --assemble-only, shift phrase starts to prevent overlap without time-stretching")
    tts_generate_p.add_argument("--min-gap", type=float, default=0.10, help="Minimum gap between phrases when --avoid-overlap is used")
    tts_generate_p.add_argument("--balance-phrases", action="store_true", help="With --assemble-only, RMS-balance phrase gains for smoother joins")
    tts_generate_p.add_argument("--balance-max-gain-db", type=float, default=3.0, help="Maximum per-phrase gain change for --balance-phrases")
    tts_generate_p.add_argument("--summary-gain-db", type=float, default=0.0, help="With --assemble-only, apply final gain to the whole assembled summary after balancing")
    tts_generate_p.add_argument("--run", action="store_true", help="Actually run Qwen3-TTS; otherwise print the command")

    tts_promote_p = sub.add_parser("tts-summary-promote", help="Promote an approved TTS summary preview to a ready audio patch")
    tts_promote_p.add_argument("episode", help="Episode id, for example S01E15")
    tts_promote_p.add_argument("--summary-id", default="summary_v001")
    tts_promote_p.add_argument("--preview", type=Path, required=True, help="Approved assembled preview WAV")
    tts_promote_p.add_argument("--patch-id", help="Ready patch id/folder name. Defaults to s01e##_next_episode_summary_tts_v001")
    tts_promote_p.add_argument("--plan", type=Path, help="Explicit phrase_plan.json path")
    tts_promote_p.add_argument("--out-dir", type=Path, default=Path("generated_audio/next_episode_summary"))
    tts_promote_p.add_argument("--ready-patch-dir", type=Path, default=Path("work/ready_audio_patches"))
    tts_promote_p.add_argument("--source-work-track", type=Path, help="Restored dialogue track that this patch applies to")
    tts_promote_p.add_argument("--replacement-gain-db", type=float, default=0.0, help="Manual gain stored in patch.json and applied by final build")
    tts_promote_p.add_argument("--match-source-work-level", action="store_true", help="Auto-match replacement level to --source-work-track over the patch window")
    tts_promote_p.add_argument("--match-max-gain-db", type=float, default=6.0, help="Clamp --match-source-work-level gain to +/- this many dB")
    tts_promote_p.add_argument("--description", help="Optional human-readable patch description")
    tts_promote_p.add_argument("--overwrite", action="store_true", help="Overwrite an existing ready patch")
    tts_promote_p.add_argument("--run", action="store_true", help="Copy replacement.wav and write patch.json")

    enhance_p = sub.add_parser("enhance-voice", help="Create conservative post-extraction voice enhancement variants")
    enhance_p.add_argument("input_wavs", nargs="+", type=Path)
    enhance_p.add_argument("--variants", nargs="+", choices=enhance_variant_names(), default=["dehiss_light", "polish_light", "polish_warm", "nlm_smooth", "speech_level"])
    enhance_p.add_argument("--review-name", required=True)
    enhance_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "11_voice_enhance")
    enhance_p.add_argument("--run", action="store_true", help="Actually run ffmpeg; otherwise print commands")

    fullmix_p = sub.add_parser("spa1-fullmix-shootout", help="Test full old-spa1 restoration without voice extraction")
    fullmix_p.add_argument("episode")
    fullmix_p.add_argument("--starts", nargs="+", required=True)
    fullmix_p.add_argument("--duration", default="00:00:20")
    fullmix_p.add_argument("--variants", nargs="+", choices=full_mix_variant_names(), default=full_mix_variant_names())
    fullmix_p.add_argument("--review-name", required=True)
    fullmix_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "13_spa1_fullmix")
    fullmix_p.add_argument("--run", action="store_true", help="Actually run ffmpeg; otherwise print commands")

    clearvoice_p = sub.add_parser("clearvoice-enhance", help="Create AI voice enhancement variants with ClearVoice")
    clearvoice_p.add_argument("input_wavs", nargs="+", type=Path)
    clearvoice_p.add_argument(
        "--variants",
        nargs="+",
        choices=["se48k", "sr48k", "se48k_sr48k"],
        default=["se48k", "sr48k", "se48k_sr48k"],
        help="se48k=speech enhancement, sr48k=super-resolution, se48k_sr48k=enhance then super-resolve",
    )
    clearvoice_p.add_argument("--review-name", required=True)
    clearvoice_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "12_clearvoice")
    clearvoice_p.add_argument("--python", type=Path, default=Path(".venv-clearvoice/bin/python"), help="Python interpreter with clearvoice installed")
    clearvoice_p.add_argument("--run", action="store_true", help="Actually run ClearVoice; otherwise print commands")

    inpaint_p = sub.add_parser("ai-inpaint-stable", help="Run or print Stable Audio inpainting overlay candidates")
    inpaint_p.add_argument("--python", type=Path, default=Path(".venv-inpaint/bin/python"), help="Python executable from the dedicated inpainting environment")
    inpaint_p.add_argument("--context-audio", type=Path, default=Path("generated_audio/titlenarrator/s01e03_title_inpaint_context_gap_silenced.wav"))
    inpaint_p.add_argument("--overlay-base", type=Path, default=Path("generated_audio/titlenarrator/s01e03_title_approved_patch_context.wav"))
    inpaint_p.add_argument("--out-dir", type=Path, default=DEFAULT_WORK / "review" / "S01E03_title_narrator_ai_inpaint_stable_001")
    inpaint_p.add_argument("--stable-audio-root", type=Path, default=Path("soft/ai_audio_inpaint/stable-audio-tools"))
    inpaint_p.add_argument("--pretrained-name", default="stabilityai/stable-audio-3-medium")
    inpaint_p.add_argument("--gap-start", type=float, default=3.080)
    inpaint_p.add_argument("--gap-end", type=float, default=4.167)
    inpaint_p.add_argument("--prompt", default="old anime jet turbine high frequency background texture, continuous engine whistle, no speech, no dialogue")
    inpaint_p.add_argument("--negative-prompt", default="speech, dialogue, voice, talking, words, narration, singing, music change")
    inpaint_p.add_argument("--seeds", nargs="+", type=int, default=[11, 23])
    inpaint_p.add_argument("--steps", type=int, default=8)
    inpaint_p.add_argument("--cfg-scale", type=float, default=1.0)
    inpaint_p.add_argument("--sampler-type", default="pingpong")
    inpaint_p.add_argument("--texture-gain-db", type=float, default=-6.0)
    inpaint_p.add_argument("--texture-highpass", type=float, default=4500.0)
    inpaint_p.add_argument("--texture-lowpass", type=float, default=12000.0)
    inpaint_p.add_argument("--fade-in", type=float, default=0.14)
    inpaint_p.add_argument("--fade-out", type=float, default=0.06)
    inpaint_p.add_argument("--model-half", action="store_true")
    inpaint_p.add_argument("--device", default="cuda")
    inpaint_p.add_argument("--run", action="store_true")

    sub.add_parser("tools-status", help="Show installed restoration tools")

    args = parser.parse_args(argv)

    if args.command == "inventory":
        return cmd_inventory(args.input, args.work, probe=not args.no_probe)
    if args.command == "probe":
        return cmd_probe(args.input, args.work, args.episode)
    if args.command == "archive-metadata":
        return cmd_archive_metadata(args.input, args.work, args.episode)
    if args.command == "init-configs":
        return cmd_init_configs(args.input, args.config_dir, args.overwrite)
    if args.command == "analyze-stereo":
        return cmd_analyze_stereo(
            args.input,
            args.work,
            args.episode,
            args.seconds,
            include_spa2=args.include_spa2,
            config_dir=args.config_dir,
            update_config=args.update_config,
        )
    if args.command == "extract":
        return cmd_extract(
            args.input,
            args.work,
            args.episode,
            start=args.start,
            duration=args.duration,
            include_spa2=args.include_spa2,
            run=args.run,
        )
    if args.command == "make-samples":
        return cmd_make_samples(
            args.input,
            args.work,
            args.episode,
            args.start,
            args.duration,
            include_spa2=args.include_spa2,
            run=args.run,
        )
    if args.command == "clean-spa1":
        return cmd_clean_spa1(
            args.input,
            args.work,
            args.episode,
            variant_name=args.variant,
            source_mode=args.source_mode,
            start=args.start,
            duration=args.duration,
            run=args.run,
        )
    if args.command == "separate-voice":
        return cmd_separate_voice(
            args.input_wav,
            args.engine,
            args.model,
            args.out_dir,
            shifts=args.shifts,
            overlap=args.overlap,
            segment=args.segment,
            no_split=args.no_split,
            single_stem=args.single_stem,
            sample_rate=args.sample_rate,
            model_file_dir=args.model_file_dir,
            mdxc_overlap=args.mdxc_overlap,
            mdxc_segment_size=args.mdxc_segment_size,
            mdxc_batch_size=args.mdxc_batch_size,
            run=args.run,
        )
    if args.command == "energy-profile":
        return cmd_energy_profile(args.input_wav, args.window, args.out)
    if args.command == "normalize-wav":
        return cmd_normalize_wav(args.input_wav, args.output_wav, args.sample_rate, run=args.run)
    if args.command == "prepare-review":
        return cmd_prepare_review(args.name, args.files, args.labels, args.review_root)
    if args.command == "voice-gate":
        return cmd_voice_gate(
            args.input,
            args.work,
            args.episode,
            start=args.start,
            duration=args.duration,
            variant_name=args.variant,
            source_mode=args.source_mode,
            review_name=args.review_name,
            model=args.model,
            single_stem=args.single_stem,
            sample_rate=args.sample_rate,
            out_root=args.out_dir,
            run=args.run,
        )
    if args.command == "audition-windows":
        return cmd_audition_windows(
            args.input,
            args.work,
            args.episode,
            starts=args.starts,
            duration=args.duration,
            variant_name=args.variant,
            source_mode=args.source_mode,
            review_name=args.review_name,
            run=args.run,
        )
    if args.command == "model-shootout":
        shootout_starts = args.starts or ([args.start] if args.start else [])
        if not shootout_starts:
            raise SystemExit("model-shootout requires --start or --starts")
        return cmd_model_shootout(
            args.input,
            args.work,
            args.episode,
            starts=shootout_starts,
            duration=args.duration,
            variant_name=args.variant,
            source_mode=args.source_mode,
            review_name=args.review_name,
            models=args.models,
            single_stem=args.single_stem,
            sample_rate=args.sample_rate,
            out_root=args.out_dir,
            run=args.run,
        )
    if args.command == "ensemble-shootout":
        shootout_starts = args.starts or ([args.start] if args.start else [])
        if not shootout_starts:
            raise SystemExit("ensemble-shootout requires --start or --starts")
        return cmd_ensemble_shootout(
            args.input,
            args.work,
            args.episode,
            starts=shootout_starts,
            duration=args.duration,
            variant_name=args.variant,
            source_mode=args.source_mode,
            review_name=args.review_name,
            single_stem=args.single_stem,
            sample_rate=args.sample_rate,
            out_root=args.out_dir,
            run=args.run,
        )
    if args.command == "center-devoice-shootout":
        return cmd_center_devoice_shootout(
            args.input,
            args.work,
            args.episode,
            starts=args.starts,
            duration=args.duration,
            review_name=args.review_name,
            models=args.models,
            sample_rate=args.sample_rate,
            out_root=args.out_dir,
            run=args.run,
        )
    if args.command == "center-mix-test":
        return cmd_center_mix_test(
            args.input,
            args.work,
            args.episode,
            start=args.start,
            duration=args.duration,
            spanish_dialogue=args.spanish_dialogue,
            center_bed=args.center_bed,
            dialogue_gain_db=args.dialogue_gain_db,
            center_bed_gain_db=args.center_bed_gain_db,
            review_name=args.review_name,
            out_dir=args.out_dir,
            run=args.run,
        )
    if args.command == "episode-review-build":
        return cmd_episode_review_build(
            args.input,
            args.work,
            args.episode,
            review_name=args.review_name,
            spa_model=args.spa_model,
            center_model=args.center_model,
            clean_variant_name=args.clean_variant,
            enhance_variant_name=args.enhance_variant,
            source_mode=args.source_mode,
            sample_rate=args.sample_rate,
            dialogue_gain_db=args.dialogue_gain_db,
            center_bed_gain_db=args.center_bed_gain_db,
            preserved_channel_gain_db=args.preserved_channel_gain_db,
            spa1_fullmix_variant_name=args.spa1_fullmix_variant,
            restored_ac3_bitrate=args.restored_ac3_bitrate,
            spa1_stereo_ac3_bitrate=args.spa1_stereo_ac3_bitrate,
            out_dir=args.out_dir,
            run=args.run,
        )
    if args.command == "episode-final-build":
        if args.episode.lower() == "all":
            batch_started_at = time.monotonic()
            processed = 0
            for episode in discover_episodes(args.input):
                print(f"== {episode.episode_id} ==", flush=True)
                cmd_episode_final_build(
                    args.input,
                    args.work,
                    episode.episode_id,
                    review_name=args.review_name,
                    proc_video_root=args.proc_video_root,
                    opening_root=args.opening_root,
                    opening_generation=args.generation,
                    include_opening=not args.no_opening,
                    end_credit_root=args.end_credit_root,
                    include_end_credit=not args.no_end_credit,
                    spa_model=args.spa_model,
                    center_model=args.center_model,
                    clean_variant_name=args.clean_variant,
                    enhance_variant_name=args.enhance_variant,
                    source_mode=args.source_mode,
                    sample_rate=args.sample_rate,
                    dialogue_gain_db=args.dialogue_gain_db,
                    center_bed_gain_db=args.center_bed_gain_db,
                    preserved_channel_gain_db=args.preserved_channel_gain_db,
                    spa1_fullmix_variant_name=args.spa1_fullmix_variant,
                    silence_start_seconds=args.silence_start_seconds,
                    restored_ac3_bitrate=args.restored_ac3_bitrate,
                    eng_ac3_bitrate=args.eng_ac3_bitrate,
                    stereo_ac3_bitrate=args.stereo_ac3_bitrate,
                    apply_video_episode_adjustments=not args.no_episode_adjustments and not args.no_video_episode_adjustments,
                    apply_audio_episode_adjustments=not args.no_episode_adjustments and not args.no_audio_episode_adjustments,
                    auto_end_fade=not args.no_auto_end_fade,
                    end_fade_frames=args.end_fade_frames,
                    end_fade_black_avg=args.end_fade_black_avg,
                    end_fade_black_max=args.end_fade_black_max,
                    rebuild_intermediates=args.rebuild_intermediates,
                    repair_recipes=collect_repair_recipes(
                        episode.episode_id,
                        args.repair_recipe,
                        None if args.no_repair_recipe_dir else args.repair_recipe_dir,
                    ),
                    ready_patch_dir=None if args.no_ready_patches else args.ready_patch_dir,
                    spa2_tail_extension_dir=None if args.no_spa2_tail_extensions else args.spa2_tail_extension_dir,
                    reference_chapter_file=None if args.no_reference_episode_chapters else args.reference_chapter_file,
                    subtitle_dir=args.subtitle_dir,
                    embed_subtitles=not args.no_embed_subtitles,
                    attach_cover_art=not args.no_cover_art,
                    cover_asset=args.cover_asset,
                    if_exists=args.if_exists,
                    copy_sources=not args.no_copy_sources,
                    keep_intermediate_segments=args.keep_intermediate_segments,
                    out_dir=args.out_dir,
                    run=args.run,
                )
                processed += 1
            if args.run:
                print(f"Final batch complete: {processed} episode(s) in {format_elapsed(time.monotonic() - batch_started_at)}")
            return 0
        return cmd_episode_final_build(
            args.input,
            args.work,
            args.episode,
            review_name=args.review_name,
            proc_video_root=args.proc_video_root,
            opening_root=args.opening_root,
            opening_generation=args.generation,
            include_opening=not args.no_opening,
            end_credit_root=args.end_credit_root,
            include_end_credit=not args.no_end_credit,
            spa_model=args.spa_model,
            center_model=args.center_model,
            clean_variant_name=args.clean_variant,
            enhance_variant_name=args.enhance_variant,
            source_mode=args.source_mode,
            sample_rate=args.sample_rate,
            dialogue_gain_db=args.dialogue_gain_db,
            center_bed_gain_db=args.center_bed_gain_db,
            preserved_channel_gain_db=args.preserved_channel_gain_db,
            spa1_fullmix_variant_name=args.spa1_fullmix_variant,
            silence_start_seconds=args.silence_start_seconds,
            restored_ac3_bitrate=args.restored_ac3_bitrate,
            eng_ac3_bitrate=args.eng_ac3_bitrate,
            stereo_ac3_bitrate=args.stereo_ac3_bitrate,
            apply_video_episode_adjustments=not args.no_episode_adjustments and not args.no_video_episode_adjustments,
            apply_audio_episode_adjustments=not args.no_episode_adjustments and not args.no_audio_episode_adjustments,
            auto_end_fade=not args.no_auto_end_fade,
            end_fade_frames=args.end_fade_frames,
            end_fade_black_avg=args.end_fade_black_avg,
            end_fade_black_max=args.end_fade_black_max,
            rebuild_intermediates=args.rebuild_intermediates,
            repair_recipes=collect_repair_recipes(
                args.episode,
                args.repair_recipe,
                None if args.no_repair_recipe_dir else args.repair_recipe_dir,
            ),
            ready_patch_dir=None if args.no_ready_patches else args.ready_patch_dir,
            spa2_tail_extension_dir=None if args.no_spa2_tail_extensions else args.spa2_tail_extension_dir,
            reference_chapter_file=None if args.no_reference_episode_chapters else args.reference_chapter_file,
            subtitle_dir=args.subtitle_dir,
            embed_subtitles=not args.no_embed_subtitles,
            attach_cover_art=not args.no_cover_art,
            cover_asset=args.cover_asset,
            if_exists=args.if_exists,
            copy_sources=not args.no_copy_sources,
            keep_intermediate_segments=args.keep_intermediate_segments,
            out_dir=args.out_dir,
            run=args.run,
        )
    if args.command == "spa2-tail-extension":
        return cmd_spa2_tail_extension(
            input_dir=args.input,
            work_dir=args.work,
            episode_id=args.episode,
            reference_mkv=args.reference_mkv,
            reference_episode_start=args.reference_episode_start,
            reference_audio_stream=args.reference_audio_stream,
            tail_start=args.tail_start,
            duration=args.duration,
            fade_in=args.fade_in,
            fade_out=args.fade_out,
            old_spa1_tail_gain_db=args.old_spa1_tail_gain_db,
            old_spa1_match_audio=args.old_spa1_match_audio,
            old_spa1_match_window=args.old_spa1_match_window,
            old_spa1_max_gain_db=args.old_spa1_max_gain_db,
            extension_id=args.extension_id,
            out_dir=args.out_dir,
            review_dir=args.review_dir,
            run=args.run,
        )
    if args.command == "reference-chapters":
        return cmd_reference_chapters(
            reference_root=args.reference_root,
            pattern=args.pattern,
            out_path=args.out,
            run=args.run,
        )
    if args.command == "episode-only-from-review":
        if args.episode.lower() == "all":
            for episode in discover_episodes(args.input):
                print(f"== {episode.episode_id} ==", flush=True)
                cmd_episode_only_from_review(
                    args.work,
                    episode.episode_id,
                    review_name=args.review_name,
                    proc_video_root=args.proc_video_root,
                    if_exists=args.if_exists,
                    run=args.run,
                )
            return 0
        return cmd_episode_only_from_review(
            args.work,
            args.episode,
            review_name=args.review_name,
            proc_video_root=args.proc_video_root,
            if_exists=args.if_exists,
            run=args.run,
        )
    if args.command == "episode-adjustments":
        return cmd_episode_adjustments(
            args.episode,
            work_dir=args.work,
            review_name=args.review_name,
            proc_video_root=args.proc_video_root,
            fade_frames=args.fade_frames,
            black_avg=args.black_avg,
            black_max=args.black_max,
            eng_ac3_bitrate=args.eng_ac3_bitrate,
            restored_ac3_bitrate=args.restored_ac3_bitrate,
            stereo_ac3_bitrate=args.stereo_ac3_bitrate,
            run=args.run,
        )
    if args.command == "prepare-subtitles":
        return cmd_prepare_subtitles(
            args.episode,
            eng_root=args.eng_root,
            spa_root=args.spa_root,
            out_dir=args.out_dir,
            run=args.run,
        )
    if args.command == "export-done":
        return cmd_export_done(
            args.episode,
            work_dir=args.work,
            review_name=args.review_name,
            proc_video_root=args.proc_video_root,
            subtitle_dir=args.subtitle_dir,
            out_dir=args.out_dir,
            if_exists=args.if_exists,
            attach_cover_art=not args.no_cover_art,
            run=args.run,
        )
    if args.command == "ocr-english-subtitles":
        return cmd_ocr_english_subtitles(
            args.episode,
            eng_root=args.eng_root,
            out_dir=args.out_dir,
            tesseract=args.tesseract,
            crop_top_ratio=args.crop_top_ratio,
            min_display_packet_size=args.min_display_packet_size,
            keep_images=args.keep_images,
            run=args.run,
        )
    if args.command == "retime-spanish-subtitles":
        return cmd_retime_spanish_subtitles(
            args.episode,
            subtitle_dir=args.subtitle_dir,
            run=args.run,
        )
    if args.command == "translate-spanish-subtitles":
        return cmd_translate_spanish_subtitles(
            args.episode,
            provider=args.provider,
            model=args.model,
            ollama_url=args.ollama_url,
            model_config=args.model_config,
            model_cache=args.model_cache,
            offline=args.offline,
            llm_python=args.llm_python,
            limit_chunks=args.limit_chunks,
            subtitle_dir=args.subtitle_dir,
            glossary_path=args.glossary,
            chunk_size=args.chunk_size,
            temperature=args.temperature,
            retries=args.retries,
            overwrite=args.overwrite,
            run=args.run,
        )
    if args.command == "import-spanish-translation":
        return cmd_import_spanish_translation(
            args.episode,
            responses=args.responses,
            subtitle_dir=args.subtitle_dir,
            run=args.run,
        )
    if args.command == "speech-map":
        return cmd_speech_map(
            work_dir=args.work,
            episode_id=args.episode,
            audio=args.audio,
            source=args.source,
            review_name=args.review_name,
            engine=args.engine,
            model=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            python=args.python,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
            run=args.run,
        )
    if args.command == "speech-find":
        return cmd_speech_find(
            episode_id=args.episode,
            phrase=args.phrase,
            source=args.source,
            map_path=args.map_path,
            speech_map_dir=args.speech_map_dir,
            around=args.around,
            window=args.window,
            limit=args.limit,
        )
    if args.command == "tts-summary-plan":
        return cmd_tts_summary_plan(
            episode_id=args.episode,
            start=args.start,
            end=args.end,
            source=args.source,
            speech_map_dir=args.speech_map_dir,
            out_dir=args.out_dir,
            summary_id=args.summary_id,
            config=args.config,
            overwrite=args.overwrite,
            run=args.run,
        )
    if args.command == "subtitle-spellcheck":
        return cmd_subtitle_spellcheck(
            episode_id=args.episode,
            source=args.source,
            subtitle_kind=args.subtitle_kind,
            subtitle_dir=args.subtitle_dir,
            speech_map_dir=args.speech_map_dir,
            hunspell_dic=args.hunspell_dic,
            glossary=args.glossary,
            allowlist=args.allowlist,
            out_dir=args.out_dir,
            min_length=args.min_length,
            run=args.run,
        )
    if args.command == "subtitle-review-workbook":
        return cmd_subtitle_review_workbook(
            episode_id=args.episode,
            source=args.source,
            subtitle_kind=args.subtitle_kind,
            subtitle_dir=args.subtitle_dir,
            speech_map_dir=args.speech_map_dir,
            hunspell_dic=args.hunspell_dic,
            glossary=args.glossary,
            allowlist=args.allowlist,
            output=args.out,
            min_length=args.min_length,
            run=args.run,
        )
    if args.command == "tts-summary-generate":
        return cmd_tts_summary_generate(
            episode_id=args.episode,
            plan=args.plan,
            start=args.start,
            end=args.end,
            source=args.source,
            speech_map_dir=args.speech_map_dir,
            summary_id=args.summary_id,
            out_dir=args.out_dir,
            qwen_root=args.qwen_root,
            qwen_python=args.qwen_python,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            takes=args.takes,
            phrases=args.phrases,
            replace_phrase=args.replace_phrase,
            model_size=args.model_size,
            language=args.language,
            chunk_size=args.chunk_size,
            chunk_gap=args.chunk_gap,
            seed_base=args.seed_base,
            exact_seed=args.exact_seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            subtalker_temperature=args.subtalker_temperature,
            subtalker_top_p=args.subtalker_top_p,
            subtalker_top_k=args.subtalker_top_k,
            device=args.device,
            x_vector_only=args.x_vector_only,
            overwrite=args.overwrite,
            assemble_only=args.assemble_only,
            fit_to_slots=args.fit_to_slots,
            slot_margin=args.slot_margin,
            speed_all_percent=args.speed_all_percent,
            avoid_overlap=args.avoid_overlap,
            min_gap=args.min_gap,
            balance_phrases=args.balance_phrases,
            balance_max_gain_db=args.balance_max_gain_db,
            summary_gain_db=args.summary_gain_db,
            run=args.run,
        )
    if args.command == "tts-summary-promote":
        return cmd_tts_summary_promote(
            episode_id=args.episode,
            summary_id=args.summary_id,
            preview=args.preview,
            patch_id=args.patch_id,
            plan=args.plan,
            out_dir=args.out_dir,
            ready_patch_dir=args.ready_patch_dir,
            source_work_track=args.source_work_track,
            replacement_gain_db=args.replacement_gain_db,
            match_source_work_level=args.match_source_work_level,
            match_max_gain_db=args.match_max_gain_db,
            description=args.description,
            overwrite=args.overwrite,
            run=args.run,
        )
    if args.command == "enhance-voice":
        return cmd_enhance_voice(
            args.input_wavs,
            variants=args.variants,
            review_name=args.review_name,
            out_dir=args.out_dir,
            work_dir=args.work,
            run=args.run,
        )
    if args.command == "spa1-fullmix-shootout":
        return cmd_spa1_fullmix_shootout(
            args.input,
            args.work,
            args.episode,
            starts=args.starts,
            duration=args.duration,
            variants=args.variants,
            review_name=args.review_name,
            out_dir=args.out_dir,
            run=args.run,
        )
    if args.command == "clearvoice-enhance":
        return cmd_clearvoice_enhance(
            args.input_wavs,
            variants=args.variants,
            review_name=args.review_name,
            out_dir=args.out_dir,
            work_dir=args.work,
            python=args.python,
            run=args.run,
        )
    if args.command == "ai-inpaint-stable":
        return cmd_ai_inpaint_stable(
            python=args.python,
            context_audio=args.context_audio,
            overlay_base=args.overlay_base,
            out_dir=args.out_dir,
            stable_audio_root=args.stable_audio_root,
            pretrained_name=args.pretrained_name,
            gap_start=args.gap_start,
            gap_end=args.gap_end,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seeds=args.seeds,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            sampler_type=args.sampler_type,
            texture_gain_db=args.texture_gain_db,
            texture_highpass=args.texture_highpass,
            texture_lowpass=args.texture_lowpass,
            fade_in=args.fade_in,
            fade_out=args.fade_out,
            model_half=args.model_half,
            device=args.device,
            run=args.run,
        )
    if args.command == "tools-status":
        return cmd_tools_status()
    raise AssertionError(args.command)


def find_episode(input_dir: Path, episode_id: str) -> EpisodeAssets:
    for episode in discover_episodes(input_dir):
        if episode.episode_id == episode_id:
            return episode
    raise SystemExit(f"Episode not found: {episode_id}")


def cmd_inventory(input_dir: Path, work_dir: Path, probe: bool) -> int:
    episodes = discover_episodes(input_dir)
    out_dir = work_dir / "01_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    if probe:
        rows = [probe_episode(ep) for ep in episodes]
    else:
        rows = [{"episode_id": ep.episode_id, "missing": ep.missing_required(), "files": ep.as_dict(), "warnings": []} for ep in episodes]

    json_path = out_dir / "inventory.json"
    csv_path = out_dir / "inventory.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_inventory_csv(csv_path, rows)

    warning_count = sum(1 for row in rows if row.get("warnings"))
    print(f"episodes={len(episodes)}")
    print(f"warnings={warning_count}")
    print(f"wrote={json_path}")
    print(f"wrote={csv_path}")
    return 0


def write_inventory_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "episode_id",
        "role",
        "path",
        "duration",
        "codec",
        "channels",
        "channel_layout",
        "language",
        "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            warnings = ";".join(row.get("warnings", []))
            files = row.get("files", {})
            for role, item in files.items():
                if isinstance(item, str) or item is None:
                    writer.writerow({"episode_id": row["episode_id"], "role": role, "path": item, "warnings": warnings})
                else:
                    writer.writerow(
                        {
                            "episode_id": row["episode_id"],
                            "role": role,
                            "path": item.get("path"),
                            "duration": item.get("duration"),
                            "codec": item.get("codec"),
                            "channels": item.get("channels"),
                            "channel_layout": item.get("channel_layout"),
                            "language": item.get("language"),
                            "warnings": warnings,
                        }
                    )


def cmd_probe(input_dir: Path, work_dir: Path, episode_id: str | None) -> int:
    episodes = discover_episodes(input_dir)
    if episode_id:
        episodes = [find_episode(input_dir, episode_id)]
    out_dir = work_dir / "01_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        result = probe_episode(episode)
        out_path = out_dir / f"{episode.episode_id}.probe.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{episode.episode_id}: warnings={len(result['warnings'])} wrote={out_path}")
    return 0


def cmd_init_configs(input_dir: Path, config_dir: Path, overwrite: bool) -> int:
    config_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    updated = 0
    for episode in discover_episodes(input_dir):
        out_path = config_dir / f"{episode.episode_id}.json"
        if out_path.exists() and not overwrite:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            upgraded = merge_config_defaults(existing, default_episode_config(episode))
            if upgraded != existing:
                out_path.write_text(json.dumps(upgraded, indent=2), encoding="utf-8")
                updated += 1
            continue
        config = default_episode_config(episode)
        out_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        count += 1
    print(f"configs_written={count}")
    print(f"configs_updated={updated}")
    print(f"config_dir={config_dir}")
    return 0


def default_episode_config(episode: EpisodeAssets) -> dict:
    return {
        "episode_id": episode.episode_id,
        "sources": episode.as_dict(),
        "policy": {
            "target_spanish_track": "spa1",
            "spa2_role": "reference_only_do_not_restore",
            "final_mux_spa2": "include_original_unmodified",
            "final_mux_spa1": "include_restored_old_stereo_vhs_broadcast_full_for_preservation_review",
            "restore_strategy": "extract_spa1_dialogue_improve_voice_replace_eng_center_dialogue",
        },
        "spanish_source_type": "unknown_until_full_episode_analysis_or_manual_review",
        "sync": {"initial_offset_ms": 0, "drift_ratio": 1.0, "segments": []},
        "levels": {"spanish_dialogue_gain_db": 0.0, "center_bed_gain_db": -6.0, "preserved_channel_gain_db": 0.0},
        "audio_start_policy": "silence_first_0.5_seconds_all_final_audio_tracks_without_trimming",
        "models": {
            "spa1_dialogue_separator": None,
            "eng_center_devoice": None,
            "speech_enhancement": None,
            "spa1_fullmix_restoration": "vhs_broadcast_full",
        },
        "parameters": {
            "spa1_pre_clean": {
                "highpass_hz": 70,
                "lowpass_hz": 12000,
                "hum_filter_hz": None,
                "denoise_strength": None,
            },
            "dialogue_extraction": {
                "prefer_full_episode_source_classification": True,
                "treat_uncertain_spa1_as_stereo": True,
                "manual_review_required_before_downmix": True,
            },
            "dialogue_mix": {
                "dialogue_limiter": True,
                "max_peak_db": -1.0,
                "target_loudness_lufs": None,
            },
            "final_mux": {
                "include_video": True,
                "include_original_english_51": True,
                "include_restored_spa1_old_stereo": True,
                "include_original_spa1": False,
                "include_original_spa2": True,
                "include_restored_spa1_51": True,
                "restored_track_title": "Doblaje Original Latinoamericano Restaurado (5.1)",
                "restored_old_stereo_track_title": "Doblaje Original Latinoamericano Restaurado (Stereo VHS)",
                "preserve_source_metadata": True,
            },
        },
        "selected_chain": None,
        "qc_status": "pending",
        "notes": "",
    }


def merge_config_defaults(existing: dict, defaults: dict) -> dict:
    merged = dict(existing)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value
        elif isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = merge_config_defaults(merged[key], value)
    return merged


def cmd_archive_metadata(input_dir: Path, work_dir: Path, episode_id: str | None) -> int:
    episodes = discover_episodes(input_dir)
    if episode_id:
        episodes = [find_episode(input_dir, episode_id)]
    out_dir = work_dir / "01_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        payload = {"episode_id": episode.episode_id, "files": {}}
        for role, path in episode.as_dict().items():
            if role == "episode_id" or path is None:
                continue
            payload["files"][role] = {"path": path, "ffprobe": ffprobe_full(Path(path))}
        out_path = out_dir / f"{episode.episode_id}.metadata.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{episode.episode_id}: wrote={out_path}")
    return 0


def cmd_analyze_stereo(
    input_dir: Path,
    work_dir: Path,
    episode_id: str | None,
    seconds: float,
    include_spa2: bool,
    config_dir: Path,
    update_config: bool,
) -> int:
    episodes = discover_episodes(input_dir)
    if episode_id:
        episodes = [find_episode(input_dir, episode_id)]
    out_dir = work_dir / "03_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        result = {"episode_id": episode.episode_id, "tracks": {}}
        roles = [("spa1", episode.spa1)]
        if include_spa2:
            roles.append(("spa2", episode.spa2))
        analysis_seconds = None if seconds <= 0 else seconds
        for role, path in roles:
            if path:
                result["tracks"][role] = stereo_correlation(path, seconds=analysis_seconds)
                result["tracks"][role]["scope"] = "full_episode" if analysis_seconds is None else "segment"
        out_path = out_dir / f"{episode.episode_id}.stereo.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if update_config:
            update_config_from_stereo(config_dir / f"{episode.episode_id}.json", result)
        print(f"{episode.episode_id}: wrote={out_path}")
    return 0


def update_config_from_stereo(config_path: Path, result: dict) -> None:
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spa1 = result.get("tracks", {}).get("spa1", {})
    classification = spa1.get("classification")
    if classification:
        scope = spa1.get("scope")
        if scope == "full_episode":
            config["spanish_source_type"] = classification
        else:
            config["spanish_source_type"] = f"{classification}_segment_only_needs_full_episode_or_manual_review"
    config.setdefault("analysis", {})["stereo"] = {
        "scope": spa1.get("scope"),
        "seconds_analyzed": spa1.get("seconds_analyzed"),
        "spa1_correlation": spa1.get("correlation"),
        "spa1_balance_db_left_minus_right": spa1.get("balance_db_left_minus_right"),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def cmd_extract(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    start: str | None,
    duration: str | None,
    include_spa2: bool,
    run: bool,
) -> int:
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    suffix = "full"
    if start or duration:
        suffix = f"{start or 'start'}_{duration or 'to-end'}".replace(":", "-")
    out_dir = work_dir / "02_extract" / episode.episode_id / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = build_extract_commands(episode, out_dir, start=start, duration=duration, include_spa2=include_spa2)
    return run_or_print(commands, run)


def cmd_make_samples(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    start: str,
    duration: str,
    include_spa2: bool,
    run: bool,
) -> int:
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    out_dir = work_dir / "04_samples" / episode.episode_id / f"{start.replace(':', '-')}_{duration.replace(':', '-')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    roles = [("eng_51", episode.eng_51), ("spa1", episode.spa1)]
    if include_spa2:
        roles.append(("spa2", episode.spa2))
    for role, path in roles:
        if not path:
            continue
        commands.append(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-ss",
                start,
                "-i",
                str(path),
                "-t",
                duration,
                "-map",
                "0:a:0",
                "-c:a",
                "pcm_s24le",
                str(out_dir / f"{episode.episode_id}_{role}.wav"),
            ]
        )
    return run_or_print(commands, run)


def cmd_clean_spa1(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    variant_name: str,
    source_mode: str,
    start: str | None,
    duration: str | None,
    run: bool,
) -> int:
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    assert episode.spa1 is not None
    variant = require_clean_variant(variant_name)
    suffix = "full"
    if start or duration:
        suffix = f"{start or 'start'}_{duration or 'to-end'}".replace(":", "-")
    out_dir = work_dir / "05_clean_spa1" / episode.episode_id / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{episode.episode_id}_spa1_{source_mode}_{variant.name}.wav"
    audio_filter = source_mode_filter(source_mode, variant.filtergraph)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        *media_args(episode.spa1, start, duration),
        "-map",
        "0:a:0",
        "-af",
        audio_filter,
        "-c:a",
        "pcm_s24le",
        str(output),
    ]
    manifest = {
        "episode_id": episode.episode_id,
        "source": str(episode.spa1),
        "output": str(output),
        "source_mode": source_mode,
        "variant": variant.name,
        "filtergraph": audio_filter,
        "notes": variant.notes,
        "start": start,
        "duration": duration,
    }
    write_manifest(out_dir / f"{output.stem}.manifest.json", manifest)
    return run_or_print([cmd], run)


def source_mode_filter(source_mode: str, cleanup_filter: str) -> str:
    if source_mode == "stereo":
        return cleanup_filter
    if source_mode == "mono-sum":
        return f"pan=mono|c0=0.5*c0+0.5*c1,{cleanup_filter}"
    if source_mode == "left":
        return f"pan=mono|c0=c0,{cleanup_filter}"
    if source_mode == "right":
        return f"pan=mono|c0=c1,{cleanup_filter}"
    raise AssertionError(source_mode)


def cmd_separate_voice(
    input_wav: Path,
    engine: str,
    model: str | None,
    out_dir: Path,
    shifts: int | None,
    overlap: float | None,
    segment: int | None,
    no_split: bool,
    single_stem: str | None,
    sample_rate: int | None,
    model_file_dir: Path | None,
    mdxc_overlap: int | None,
    mdxc_segment_size: int | None,
    mdxc_batch_size: int | None,
    run: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if engine == "audio-separator" and model_file_dir:
        model_file_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_separation_command(
        engine,
        input_wav,
        out_dir,
        model,
        shifts=shifts,
        overlap=overlap,
        segment=segment,
        no_split=no_split,
        single_stem=single_stem,
        sample_rate=sample_rate,
        model_file_dir=model_file_dir,
        mdxc_overlap=mdxc_overlap,
        mdxc_segment_size=mdxc_segment_size,
        mdxc_batch_size=mdxc_batch_size,
    )
    manifest = {
        "input": str(input_wav),
        "engine": engine,
        "model": model,
        "out_dir": str(out_dir),
        "shifts": shifts,
        "overlap": overlap,
        "segment": segment,
        "no_split": no_split,
        "single_stem": single_stem,
        "sample_rate": sample_rate,
        "model_file_dir": str(model_file_dir) if model_file_dir else None,
        "mdxc_overlap": mdxc_overlap,
        "mdxc_segment_size": mdxc_segment_size,
        "mdxc_batch_size": mdxc_batch_size,
        "command": cmd,
    }
    write_manifest(out_dir / f"{input_wav.stem}.{engine}.manifest.json", manifest)
    if run and not installed_tools().get(engine):
        raise SystemExit(f"{engine} is not installed or not on PATH. Run tools-status or install it first.")
    return run_or_print([cmd], run)


def cmd_normalize_wav(input_wav: Path, output_wav: Path, sample_rate: int, run: bool) -> int:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_wav),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s24le",
        str(output_wav),
    ]
    write_manifest(
        output_wav.with_suffix(".manifest.json"),
        {
            "input": str(input_wav),
            "output": str(output_wav),
            "sample_rate": sample_rate,
            "codec": "pcm_s24le",
            "command": cmd,
        },
    )
    return run_or_print([cmd], run)


def cmd_energy_profile(input_wav: Path, window_seconds: float, out_path: Path | None) -> int:
    profile = energy_profile(input_wav, window_seconds=window_seconds)
    if out_path is None:
        out_path = input_wav.with_suffix(".energy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"wrote={out_path}")
    for window in profile["windows"]:
        print(
            f"{window['start']:06.2f}-{window['end']:06.2f}s "
            f"rms={window['rms_dbfs']:7.2f} dBFS peak={window['peak_dbfs']:7.2f} dBFS"
        )
    return 0


def cmd_prepare_review(name: str, files: list[Path], labels: list[str] | None, review_root: Path) -> int:
    if labels and len(labels) != len(files):
        raise SystemExit("--labels count must match file count")
    out_dir = review_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.iterdir():
        if stale.is_file() and stale.suffix.lower() in {".wav", ".json", ".csv", ".txt"}:
            stale.unlink()
    manifest = {"name": name, "files": []}
    for index, src in enumerate(files, start=1):
        if not src.exists():
            raise SystemExit(f"Review source does not exist: {src}")
        label = labels[index - 1] if labels else src.stem
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label).strip("_")
        dst = out_dir / f"{index:02d}_{safe_label}{src.suffix}"
        shutil.copy2(src, dst)
        manifest["files"].append({"index": index, "label": label, "source": str(src), "review_file": str(dst)})
        print(f"{index:02d}: {dst}")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest={manifest_path}")
    print(f"review_dir={out_dir}")
    return 0


def copy_source_file(source: Path, destination_dir: Path) -> Path:
    destination = destination_dir / source.name
    if destination.exists():
        return destination
    if free_bytes(destination_dir) >= 10 * 1024 * 1024 * 1024:
        shutil.copy2(source, destination)
        return destination
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def free_bytes(path: Path) -> int:
    stat = shutil.disk_usage(path)
    return stat.free


def cmd_voice_gate(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    start: str,
    duration: str,
    variant_name: str,
    source_mode: str,
    review_name: str,
    model: str | None,
    single_stem: str,
    sample_rate: int,
    out_root: Path,
    run: bool,
) -> int:
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    assert episode.spa1 is not None

    suffix = f"{start}_{duration}".replace(":", "-")
    clean_dir = work_dir / "05_clean_spa1" / episode.episode_id / suffix
    clean_dir.mkdir(parents=True, exist_ok=True)
    variant = require_clean_variant(variant_name)
    clean_wav = clean_dir / f"{episode.episode_id}_spa1_{source_mode}_{variant.name}.wav"
    clean_filter = source_mode_filter(source_mode, variant.filtergraph)
    clean_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        *media_args(episode.spa1, start, duration),
        "-map",
        "0:a:0",
        "-af",
        clean_filter,
        "-c:a",
        "pcm_s24le",
        str(clean_wav),
    ]
    write_manifest(
        clean_dir / f"{clean_wav.stem}.manifest.json",
        {
            "episode_id": episode.episode_id,
            "source": str(episode.spa1),
            "output": str(clean_wav),
            "source_mode": source_mode,
            "variant": variant.name,
            "filtergraph": clean_filter,
            "start": start,
            "duration": duration,
        },
    )

    sep_dir = out_root / f"{review_name}_bs_roformer"
    sep_dir.mkdir(parents=True, exist_ok=True)
    sep_cmd = build_separation_command(
        "audio-separator",
        clean_wav,
        sep_dir,
        model=model,
        single_stem=single_stem,
        sample_rate=sample_rate,
        model_file_dir=work_dir / "models" / "audio-separator",
    )
    write_manifest(
        sep_dir / f"{clean_wav.stem}.audio-separator.manifest.json",
        {
            "input": str(clean_wav),
            "engine": "audio-separator",
            "model": model,
            "out_dir": str(sep_dir),
            "single_stem": single_stem,
            "sample_rate": sample_rate,
            "command": sep_cmd,
        },
    )

    if run:
        for cmd in (clean_cmd, sep_cmd):
            print(shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        vocals = find_single_vocals_output(sep_dir)
        return cmd_prepare_review(
            review_name,
            [clean_wav, vocals],
            [f"input_spa1_{variant.name}_{start}", f"bs_roformer_vocals_{start}"],
            work_dir / "review",
        )
    print(shlex.join(clean_cmd))
    print(shlex.join(sep_cmd))
    print(f"review will be created after --run: {work_dir / 'review' / review_name}")
    return 0


def cmd_audition_windows(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    starts: list[str],
    duration: str,
    variant_name: str,
    source_mode: str,
    review_name: str,
    run: bool,
) -> int:
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    assert episode.spa1 is not None

    variant = require_clean_variant(variant_name)
    audition_dir = work_dir / "04_audition" / episode.episode_id / review_name
    audition_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    outputs: list[Path] = []
    labels: list[str] = []
    for start in starts:
        safe_start = start.replace(":", "m", 1).replace(":", "s")
        output = audition_dir / f"{episode.episode_id}_spa1_{source_mode}_{variant.name}_{start.replace(':', '-')}.wav"
        audio_filter = source_mode_filter(source_mode, variant.filtergraph)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            *media_args(episode.spa1, start, duration),
            "-map",
            "0:a:0",
            "-af",
            audio_filter,
            "-c:a",
            "pcm_s24le",
            str(output),
        ]
        commands.append(cmd)
        outputs.append(output)
        labels.append(f"source_{safe_start}_{variant.name}")

    manifest = {
        "episode_id": episode.episode_id,
        "source": str(episode.spa1),
        "review_name": review_name,
        "duration": duration,
        "source_mode": source_mode,
        "variant": variant.name,
        "filtergraph": source_mode_filter(source_mode, variant.filtergraph),
        "starts": starts,
        "outputs": [str(path) for path in outputs],
        "commands": commands,
    }
    write_manifest(audition_dir / "manifest.json", manifest)

    if run:
        for cmd in commands:
            print(shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        return cmd_prepare_review(review_name, outputs, labels, work_dir / "review")

    for cmd in commands:
        print(shlex.join(cmd))
    print(f"review will be created after --run: {work_dir / 'review' / review_name}")
    return 0


def cmd_model_shootout(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    starts: list[str],
    duration: str,
    variant_name: str,
    source_mode: str,
    review_name: str,
    models: list[str],
    single_stem: str,
    sample_rate: int,
    out_root: Path,
    run: bool,
) -> int:
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    assert episode.spa1 is not None

    out_root.mkdir(parents=True, exist_ok=True)
    variant = require_clean_variant(variant_name)
    if len(starts) > 1:
        for index, start in enumerate(starts, start=1):
            child_review_name = f"{review_name}_{index:02d}_{start.replace(':', '-')}"
            cmd_model_shootout(
                input_dir,
                work_dir,
                episode_id,
                starts=[start],
                duration=duration,
                variant_name=variant_name,
                source_mode=source_mode,
                review_name=child_review_name,
                models=models,
                single_stem=single_stem,
                sample_rate=sample_rate,
                out_root=out_root,
                run=run,
            )
        return 0

    start = starts[0]
    suffix = f"{start}_{duration}".replace(":", "-")
    clean_dir = work_dir / "05_clean_spa1" / episode.episode_id / suffix
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_wav = clean_dir / f"{episode.episode_id}_spa1_{source_mode}_{variant.name}.wav"
    clean_filter = source_mode_filter(source_mode, variant.filtergraph)
    clean_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        *media_args(episode.spa1, start, duration),
        "-map",
        "0:a:0",
        "-af",
        clean_filter,
        "-c:a",
        "pcm_s24le",
        str(clean_wav),
    ]

    model_jobs = []
    for model_token in models:
        model = None if model_token == "default" else model_token
        model_label = model_label_for_review(model_token)
        sep_dir = out_root / f"{review_name}_{model_label}"
        sep_cmd = build_separation_command(
            "audio-separator",
            clean_wav,
            sep_dir,
            model=model,
            single_stem=single_stem,
            sample_rate=sample_rate,
            model_file_dir=work_dir / "models" / "audio-separator",
        )
        model_jobs.append((model_token, model_label, sep_dir, sep_cmd))

    write_manifest(
        out_root / f"{review_name}.model-shootout.manifest.json",
        {
            "episode_id": episode.episode_id,
            "source": str(episode.spa1),
            "clean_wav": str(clean_wav),
            "start": start,
            "duration": duration,
            "source_mode": source_mode,
            "variant": variant.name,
            "filtergraph": clean_filter,
            "models": models,
            "single_stem": single_stem,
            "sample_rate": sample_rate,
            "clean_command": clean_cmd,
            "separation_commands": [job[3] for job in model_jobs],
        },
    )

    if run:
        print(shlex.join(clean_cmd), flush=True)
        subprocess.run(clean_cmd, check=True)
        review_files = [clean_wav]
        labels = [f"input_spa1_{variant.name}_{start}"]
        for _model_token, model_label, sep_dir, sep_cmd in model_jobs:
            sep_dir.mkdir(parents=True, exist_ok=True)
            print(shlex.join(sep_cmd), flush=True)
            subprocess.run(sep_cmd, check=True)
            review_files.append(find_single_separator_output(sep_dir, single_stem))
            labels.append(model_label)
        return cmd_prepare_review(review_name, review_files, labels, work_dir / "review")

    print(shlex.join(clean_cmd))
    for _model_token, _model_label, _sep_dir, sep_cmd in model_jobs:
        print(shlex.join(sep_cmd))
    print(f"review will be created after --run: {work_dir / 'review' / review_name}")
    return 0


def model_label_for_review(model_token: str) -> str:
    if model_token == "default":
        return "default_bs_roformer"
    stem = Path(model_token).stem.lower()
    return "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")


def cmd_enhance_voice(
    input_wavs: list[Path],
    variants: list[str],
    review_name: str,
    out_dir: Path,
    work_dir: Path,
    run: bool,
) -> int:
    out_dir = out_dir / review_name
    out_dir.mkdir(parents=True, exist_ok=True)
    review_files: list[Path] = []
    labels: list[str] = []
    commands: list[list[str]] = []

    for input_wav in input_wavs:
        if not input_wav.exists():
            raise SystemExit(f"Input WAV does not exist: {input_wav}")
        source_label = compact_source_label(input_wav)
        review_files.append(input_wav)
        labels.append(f"{source_label}_source")
        for variant_name in variants:
            variant = require_enhance_variant(variant_name)
            output = out_dir / f"{source_label}_{variant.name}.wav"
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                str(input_wav),
                "-af",
                variant.filtergraph,
                "-c:a",
                "pcm_s24le",
                str(output),
            ]
            commands.append(cmd)
            review_files.append(output)
            labels.append(f"{source_label}_{variant.name}")

    write_manifest(
        out_dir / "manifest.json",
        {
            "review_name": review_name,
            "inputs": [str(path) for path in input_wavs],
            "variants": [
                {
                    "name": require_enhance_variant(name).name,
                    "filtergraph": require_enhance_variant(name).filtergraph,
                    "notes": require_enhance_variant(name).notes,
                }
                for name in variants
            ],
            "commands": commands,
        },
    )

    if run:
        for cmd in commands:
            print(shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        return cmd_prepare_review(review_name, review_files, labels, work_dir / "review")

    for cmd in commands:
        print(shlex.join(cmd))
    print(f"review will be created after --run: {work_dir / 'review' / review_name}")
    return 0


def cmd_center_devoice_shootout(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    starts: list[str],
    duration: str,
    review_name: str,
    models: list[str],
    sample_rate: int,
    out_root: Path,
    run: bool,
) -> int:
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    assert episode.eng_51 is not None

    out_root.mkdir(parents=True, exist_ok=True)
    center_sources: list[dict] = []
    commands: list[list[str]] = []
    jobs: list[dict] = []

    for index, start in enumerate(starts, start=1):
        window_label = f"W{index:02d}_{compact_time_label(start)}"
        suffix = f"{start}_{duration}".replace(":", "-")
        center_dir = work_dir / "02_extract" / episode.episode_id / suffix
        center_dir.mkdir(parents=True, exist_ok=True)
        center_wav = center_dir / f"{episode.episode_id}_eng_center.wav"
        center_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            *media_args(episode.eng_51, start, duration),
            "-map",
            "0:a:0",
            "-af",
            "pan=mono|c0=c2",
            "-c:a",
            "pcm_s24le",
            str(center_wav),
        ]
        commands.append(center_cmd)
        center_sources.append(
            {
                "window": window_label,
                "start": start,
                "center_wav": center_wav,
                "label": f"{window_label}_eng_center_source",
            }
        )

        for model_token in models:
            model = None if model_token == "default" else model_token
            model_label = model_label_for_review(model_token)
            sep_dir = out_root / f"{review_name}_{window_label}_{model_label}"
            sep_cmd = build_separation_command(
                "audio-separator",
                center_wav,
                sep_dir,
                model=model,
                single_stem=None,
                sample_rate=sample_rate,
                model_file_dir=work_dir / "models" / "audio-separator",
            )
            commands.append(sep_cmd)
            jobs.append(
                {
                    "window": window_label,
                    "start": start,
                    "center_wav": str(center_wav),
                    "model": model_token,
                    "model_label": model_label,
                    "sep_dir": str(sep_dir),
                    "command": sep_cmd,
                }
            )

    write_manifest(
        out_root / f"{review_name}.center-devoice-shootout.manifest.json",
        {
            "episode_id": episode.episode_id,
            "source": str(episode.eng_51),
            "starts": starts,
            "duration": duration,
            "models": models,
            "sample_rate": sample_rate,
            "commands": commands,
            "jobs": jobs,
            "review_note": "Review folder contains only source center clips and Instrumental de-voiced center beds. Vocals diagnostics remain in each separator output folder.",
        },
    )

    if run:
        for cmd in commands:
            print(shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        review_files: list[Path] = []
        labels: list[str] = []
        jobs_by_window: dict[str, list[dict]] = {}
        for job in jobs:
            jobs_by_window.setdefault(job["window"], []).append(job)

        for source in center_sources:
            review_files.append(source["center_wav"])
            labels.append(source["label"])
            for job in jobs_by_window.get(source["window"], []):
                sep_dir = Path(job["sep_dir"])
                bed = find_single_stem_output(sep_dir, "Instrumental")
                review_files.append(bed)
                labels.append(f"{job['window']}_{job['model_label']}_bed_no_eng")
        return cmd_prepare_review(review_name, review_files, labels, work_dir / "review")

    for cmd in commands:
        print(shlex.join(cmd))
    print(f"review will be created after --run: {work_dir / 'review' / review_name}")
    return 0


def cmd_center_mix_test(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    start: str,
    duration: str,
    spanish_dialogue: Path,
    center_bed: Path,
    dialogue_gain_db: float,
    center_bed_gain_db: float,
    review_name: str,
    out_dir: Path,
    run: bool,
) -> int:
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    assert episode.eng_51 is not None
    if not spanish_dialogue.exists():
        raise SystemExit(f"Spanish dialogue WAV does not exist: {spanish_dialogue}")
    if not center_bed.exists():
        raise SystemExit(f"Center bed WAV does not exist: {center_bed}")

    suffix = compact_time_label(start)
    mix_dir = out_dir / episode.episode_id / review_name
    mix_dir.mkdir(parents=True, exist_ok=True)
    center_mix = mix_dir / f"{episode.episode_id}_{suffix}_restored_center.wav"
    restored_51 = mix_dir / f"{episode.episode_id}_{suffix}_restored_spa_51.wav"
    stereo_downmix = mix_dir / f"{episode.episode_id}_{suffix}_restored_spa_51_stereo_downmix.wav"
    original_eng_51 = mix_dir / f"{episode.episode_id}_{suffix}_original_eng_51.wav"
    original_eng_downmix = mix_dir / f"{episode.episode_id}_{suffix}_original_eng_51_stereo_downmix.wav"
    original_spa1 = mix_dir / f"{episode.episode_id}_{suffix}_original_spa1_raw.wav"
    original_spa2 = mix_dir / f"{episode.episode_id}_{suffix}_original_spa2_raw.wav"

    filter_complex = (
        "[0:a]channelsplit=channel_layout=5.1(side)[FL][FR][FC_ORIG][LFE][SL][SR];"
        "[FC_ORIG]anullsink;"
        "[1:a]pan=mono|c0=0.5*c0+0.5*c1,"
        f"volume={center_bed_gain_db}dB[bed];"
        "[2:a]pan=mono|c0=0.5*c0+0.5*c1,"
        f"volume={dialogue_gain_db}dB[dialogue];"
        "[bed][dialogue]amix=inputs=2:duration=first:normalize=0,"
        "alimiter=limit=0.95,asplit=2[FCNEW][CENTEROUT];"
        "[FL][FR][FCNEW][LFE][SL][SR]"
        "join=inputs=6:channel_layout=5.1(side):"
        "map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-SL|5.0-SR[out51]"
    )
    build_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        *media_args(episode.eng_51, start, duration),
        "-i",
        str(center_bed),
        "-i",
        str(spanish_dialogue),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out51]",
        "-c:a",
        "pcm_s24le",
        str(restored_51),
        "-map",
        "[CENTEROUT]",
        "-c:a",
        "pcm_s24le",
        str(center_mix),
    ]
    downmix_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(restored_51),
        "-af",
        "pan=stereo|FL=0.65*FL+0.46*FC+0.25*LFE+0.32*SL|FR=0.65*FR+0.46*FC+0.25*LFE+0.32*SR",
        "-c:a",
        "pcm_s24le",
        str(stereo_downmix),
    ]
    original_eng_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        *media_args(episode.eng_51, start, duration),
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_s24le",
        str(original_eng_51),
    ]
    original_eng_downmix_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(original_eng_51),
        "-af",
        "pan=stereo|FL=0.65*FL+0.46*FC+0.25*LFE+0.32*SL|FR=0.65*FR+0.46*FC+0.25*LFE+0.32*SR",
        "-c:a",
        "pcm_s24le",
        str(original_eng_downmix),
    ]
    original_spa_cmds: list[list[str]] = [
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            *media_args(episode.spa1, start, duration),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s24le",
            str(original_spa1),
        ]
    ]
    if episode.spa2:
        original_spa_cmds.append(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                *media_args(episode.spa2, start, duration),
                "-map",
                "0:a:0",
                "-c:a",
                "pcm_s24le",
                str(original_spa2),
            ]
        )
    write_manifest(
        mix_dir / f"{review_name}.center-mix-test.manifest.json",
        {
            "episode_id": episode.episode_id,
            "source_eng_51": str(episode.eng_51),
            "start": start,
            "duration": duration,
            "center_bed": str(center_bed),
            "spanish_dialogue": str(spanish_dialogue),
            "dialogue_gain_db": dialogue_gain_db,
            "center_bed_gain_db": center_bed_gain_db,
            "outputs": {
                "restored_51": str(restored_51),
                "center_mix": str(center_mix),
                "stereo_downmix": str(stereo_downmix),
                "original_eng_51": str(original_eng_51),
                "original_eng_51_stereo_downmix": str(original_eng_downmix),
                "original_spa1": str(original_spa1),
                "original_spa2": str(original_spa2) if episode.spa2 else None,
            },
            "commands": [build_cmd, downmix_cmd, original_eng_cmd, original_eng_downmix_cmd, *original_spa_cmds],
        },
    )

    if run:
        for cmd in [build_cmd, downmix_cmd, original_eng_cmd, original_eng_downmix_cmd, *original_spa_cmds]:
            print(shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        review_files = [restored_51, stereo_downmix, center_mix, original_eng_51, original_eng_downmix, original_spa1]
        labels = [
            f"{episode.episode_id}_{suffix}_restored_spa_51",
            f"{episode.episode_id}_{suffix}_restored_spa_51_stereo_downmix",
            f"{episode.episode_id}_{suffix}_restored_center_only",
            f"{episode.episode_id}_{suffix}_original_eng_51",
            f"{episode.episode_id}_{suffix}_original_eng_51_stereo_downmix",
            f"{episode.episode_id}_{suffix}_original_spa1_raw",
        ]
        if episode.spa2:
            review_files.append(original_spa2)
            labels.append(f"{episode.episode_id}_{suffix}_original_spa2_raw")
        return cmd_prepare_review(
            review_name,
            review_files,
            labels,
            work_dir / "review",
        )

    for cmd in [build_cmd, downmix_cmd, original_eng_cmd, original_eng_downmix_cmd, *original_spa_cmds]:
        print(shlex.join(cmd))
    print(f"review will be created after --run: {work_dir / 'review' / review_name}")
    return 0


def cmd_episode_review_build(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    review_name: str,
    spa_model: str,
    center_model: str,
    clean_variant_name: str,
    enhance_variant_name: str,
    source_mode: str,
    sample_rate: int,
    dialogue_gain_db: float,
    center_bed_gain_db: float,
    preserved_channel_gain_db: float,
    spa1_fullmix_variant_name: str,
    restored_ac3_bitrate: str,
    spa1_stereo_ac3_bitrate: str,
    out_dir: Path,
    run: bool,
) -> int:
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    if episode.spa2 is None:
        raise SystemExit(f"{episode.episode_id} missing spa2, required for this review mux")
    assert episode.eng_51 is not None and episode.spa1 is not None and episode.video is not None

    clean_variant = require_clean_variant(clean_variant_name)
    enhance_variant = require_enhance_variant(enhance_variant_name)
    spa1_fullmix_variant = require_full_mix_variant(spa1_fullmix_variant_name)
    build_dir = out_dir / episode.episode_id / review_name
    review_dir = work_dir / "review" / review_name
    if run:
        build_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

    clean_spa1 = build_dir / f"{episode.episode_id}_spa1_{source_mode}_{clean_variant.name}.wav"
    spa_sep_dir = build_dir / "spa1_dialogue_separator"
    spanish_dialogue = build_dir / f"{episode.episode_id}_spa1_dialogue_{enhance_variant.name}.wav"
    eng_center = build_dir / f"{episode.episode_id}_eng_center.wav"
    center_sep_dir = build_dir / "eng_center_devoice"
    restored_wav = build_dir / f"{episode.episode_id}_restored_spa_51_lossless.wav"
    spa1_fullmix_wav = build_dir / f"{episode.episode_id}_spa1_stereo_{spa1_fullmix_variant.name}.wav"

    review_mkv = review_dir / f"01_{episode.episode_id}_video_eng_restoredspa_spa1_spa2.mkv"
    restored_ac3 = review_dir / f"02_{episode.episode_id}_restored_spa_51.ac3"
    eng_ac3 = review_dir / f"03_{episode.episode_id}_english_original_51.ac3"
    spa1_ac3 = review_dir / f"04_{episode.episode_id}_spanish1_restored_old_stereo_{spa1_fullmix_variant.name}.ac3"
    spa2_ac3 = review_dir / f"05_{episode.episode_id}_spanish2_original_stereo.ac3"

    clean_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.spa1),
        "-map",
        "0:a:0",
        "-af",
        source_mode_filter(source_mode, clean_variant.filtergraph),
        "-c:a",
        "pcm_s24le",
        str(clean_spa1),
    ]
    spa_sep_cmd = build_separation_command(
        "audio-separator",
        clean_spa1,
        spa_sep_dir,
        model=spa_model,
        single_stem="Vocals",
        sample_rate=sample_rate,
        model_file_dir=work_dir / "models" / "audio-separator",
    )
    center_extract_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.eng_51),
        "-map",
        "0:a:0",
        "-af",
        "pan=mono|c0=c2",
        "-c:a",
        "pcm_s24le",
        str(eng_center),
    ]
    center_sep_cmd = build_separation_command(
        "audio-separator",
        eng_center,
        center_sep_dir,
        model=center_model,
        single_stem="Instrumental",
        sample_rate=sample_rate,
        model_file_dir=work_dir / "models" / "audio-separator",
    )
    spa1_fullmix_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.spa1),
        "-map",
        "0:a:0",
        "-af",
        spa1_fullmix_variant.filtergraph,
        "-c:a",
        "pcm_s24le",
        str(spa1_fullmix_wav),
    ]
    track_copy_cmds = [
        ["ffmpeg", "-hide_banner", "-y", "-i", str(episode.eng_51), "-map", "0:a:0", "-c:a", "copy", str(eng_ac3)],
        ["ffmpeg", "-hide_banner", "-y", "-i", str(episode.spa2), "-map", "0:a:0", "-c:a", "copy", str(spa2_ac3)],
    ]

    for cmd in (clean_cmd, spa_sep_cmd, center_extract_cmd, center_sep_cmd):
        print(shlex.join(cmd), flush=True)
        if run:
            subprocess.run(cmd, check=True)

    if not run:
        print("after separation: enhance Vocals, mix restored 5.1, encode restored AC-3, build restored spa1 stereo, copy source AC-3 tracks, mux MKV")
        print(f"review will be created after --run: {review_dir}")
        return 0

    spa_vocals = find_single_stem_output(spa_sep_dir, "Vocals")
    center_bed = find_single_stem_output(center_sep_dir, "Instrumental")
    devoiced_bed_51_filter = (
        "[0:a]channelsplit=channel_layout=5.1(side)[FL][FR][FC_ORIG][LFE][SL][SR];"
        "[FC_ORIG]anullsink;"
        "[1:a]pan=mono|c0=0.5*c0+0.5*c1[FCNEW];"
        "[FL][FR][FCNEW][LFE][SL][SR]"
        "join=inputs=6:channel_layout=5.1(side):"
        "map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-SL|5.0-SR[out51]"
    )
    devoiced_bed_51_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.eng_51),
        "-i",
        str(center_bed),
        "-filter_complex",
        devoiced_bed_51_filter,
        "-map",
        "[out51]",
        "-c:a",
        "pcm_s24le",
        str(devoiced_bed_51_wav),
    ]
    enhance_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(spa_vocals),
        "-af",
        enhance_variant.filtergraph,
        "-c:a",
        "pcm_s24le",
        str(spanish_dialogue),
    ]
    mix_filter = (
        "[0:a]channelsplit=channel_layout=5.1(side)[FL][FR][FC_ORIG][LFE][SL][SR];"
        "[FC_ORIG]anullsink;"
        f"[FL]volume={preserved_channel_gain_db}dB[FLP];"
        f"[FR]volume={preserved_channel_gain_db}dB[FRP];"
        f"[LFE]volume={preserved_channel_gain_db}dB[LFEP];"
        f"[SL]volume={preserved_channel_gain_db}dB[SLP];"
        f"[SR]volume={preserved_channel_gain_db}dB[SRP];"
        "[1:a]pan=mono|c0=0.5*c0+0.5*c1,"
        f"volume={center_bed_gain_db}dB[bed];"
        "[2:a]pan=mono|c0=0.5*c0+0.5*c1,"
        f"volume={dialogue_gain_db}dB[dialogue];"
        "[bed][dialogue]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[FCNEW];"
        "[FLP][FRP][FCNEW][LFEP][SLP][SRP]"
        "join=inputs=6:channel_layout=5.1(side):"
        "map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-SL|5.0-SR[out51]"
    )
    mix_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.eng_51),
        "-i",
        str(center_bed),
        "-i",
        str(spanish_dialogue),
        "-filter_complex",
        mix_filter,
        "-map",
        "[out51]",
        "-c:a",
        "pcm_s24le",
        str(restored_wav),
    ]
    restored_ac3_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(restored_wav),
        "-c:a",
        "ac3",
        "-b:a",
        restored_ac3_bitrate,
        str(restored_ac3),
    ]
    spa1_stereo_ac3_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(spa1_fullmix_wav),
        "-c:a",
        "ac3",
        "-b:a",
        spa1_stereo_ac3_bitrate,
        str(spa1_ac3),
    ]
    mux_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.video),
        "-i",
        str(eng_ac3),
        "-i",
        str(restored_ac3),
        "-i",
        str(spa1_ac3),
        "-i",
        str(spa2_ac3),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:a:0",
        "-map",
        "3:a:0",
        "-map",
        "4:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:0",
        "title=English Original 5.1",
        "-metadata:s:a:1",
        "language=spa",
        "-metadata:s:a:1",
        "title=Spanish Restored Test 5.1",
        "-metadata:s:a:2",
        "language=spa",
        "-metadata:s:a:2",
        f"title=Spanish 1 Restored Old Stereo ({spa1_fullmix_variant.name})",
        "-metadata:s:a:3",
        "language=spa",
        "-metadata:s:a:3",
        "title=Spanish Redubbing Original Stereo",
        "-disposition:a:0",
        "default",
        "-disposition:a:1",
        "0",
        "-disposition:a:2",
        "0",
        "-disposition:a:3",
        "0",
        str(review_mkv),
    ]

    for cmd in [enhance_cmd, mix_cmd, restored_ac3_cmd, spa1_fullmix_cmd, spa1_stereo_ac3_cmd, *track_copy_cmds, mux_cmd]:
        print(shlex.join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    write_manifest(
        review_dir / "manifest.json",
        {
            "episode_id": episode.episode_id,
            "review_name": review_name,
            "files": [
                {"index": 1, "label": "video_with_four_audio_tracks", "path": str(review_mkv)},
                {"index": 2, "label": "restored_spa_51_ac3", "path": str(restored_ac3)},
                {"index": 3, "label": "english_original_51_ac3", "path": str(eng_ac3)},
                {"index": 4, "label": f"spanish1_restored_old_stereo_{spa1_fullmix_variant.name}_ac3", "path": str(spa1_ac3)},
                {"index": 5, "label": "spanish2_original_stereo_ac3", "path": str(spa2_ac3)},
            ],
            "lossless_restored_wav": str(restored_wav),
            "lossless_spa1_fullmix_wav": str(spa1_fullmix_wav),
            "mix_levels": {
                "preserved_channel_gain_db": preserved_channel_gain_db,
                "center_bed_gain_db": center_bed_gain_db,
                "dialogue_gain_db": dialogue_gain_db,
            },
            "commands": [
                clean_cmd,
                spa_sep_cmd,
                center_extract_cmd,
                center_sep_cmd,
                enhance_cmd,
                mix_cmd,
                restored_ac3_cmd,
                spa1_fullmix_cmd,
                spa1_stereo_ac3_cmd,
                *track_copy_cmds,
                mux_cmd,
            ],
        },
    )
    print(f"review_dir={review_dir}")
    print(f"video={review_mkv}")
    return 0


def cmd_episode_only_from_review(
    work_dir: Path,
    episode_id: str,
    review_name: str,
    proc_video_root: Path,
    if_exists: str,
    run: bool,
) -> int:
    started_at = time.monotonic()
    review_dir = work_dir / "review" / "episodes" / episode_id / review_name
    audio_dir = review_dir / "audio"
    output_dir = review_dir / "video" / "episode_only"
    if not audio_dir.is_dir():
        raise SystemExit(f"Missing review audio folder: {audio_dir}")

    selected_videos = select_proc_videos(proc_video_root, episode_id)
    audio_tracks = review_audio_tracks(audio_dir, episode_id)
    planned_outputs = {
        variant: output_dir / f"{Path(final_video_name(episode_id, variant, video_path)).stem}_episode_only.mkv"
        for variant, video_path in selected_videos.items()
    }
    if run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(path.exists() for path in planned_outputs.values()):
            decision = decide_existing_outputs(episode_id, planned_outputs, if_exists)
            if decision == "skip":
                print(f"Skipping {episode_id}: episode-only output already exists")
                return 0

    commands: list[list[str]] = []
    for variant, video_path in selected_videos.items():
        output = planned_outputs[variant]
        cmd = final_mux_cmd(
            episode_id,
            variant,
            video_path,
            output,
            audio_tracks["restored_ac3"],
            audio_tracks["eng_ac3"],
            audio_tracks["spa1_ac3"],
            audio_tracks["spa2_ac3"],
        )
        print(shlex.join(cmd), flush=True)
        commands.append(cmd)
        if run:
            subprocess.run(cmd, check=True)

    if run:
        manifest = {
            "episode_id": episode_id,
            "review_name": review_name,
            "source": "episode-only-from-review",
            "selected_videos": {variant: str(path) for variant, path in selected_videos.items()},
            "audio_tracks": {key: str(path) for key, path in audio_tracks.items()},
            "episode_only_outputs": {variant: str(path) for variant, path in planned_outputs.items()},
            "commands": commands,
        }
        write_manifest(output_dir / "episode_only_manifest.json", manifest)
        write_episode_only_report(
            output_dir / "EPISODE_ONLY_REPORT.md",
            episode_id=episode_id,
            review_name=review_name,
            started_at=started_at,
            manifest=manifest,
            commands=commands,
        )
        print("")
        print(f"Episode-only mux complete: {episode_id}")
        print(f"Output folder: {output_dir}")
        print(f"Elapsed: {format_elapsed(time.monotonic() - started_at)}")
    else:
        print(f"episode-only outputs will be created after --run: {output_dir}")
    return 0


def cmd_episode_adjustments(
    episode: str,
    work_dir: Path,
    review_name: str,
    proc_video_root: Path,
    fade_frames: int,
    black_avg: float,
    black_max: int,
    eng_ac3_bitrate: str,
    restored_ac3_bitrate: str,
    stereo_ac3_bitrate: str,
    run: bool,
) -> int:
    started_at = time.monotonic()
    episode_ids = list(EPISODE_TITLES) if episode.lower() == "all" else [normalize_episode_id(episode)]
    all_manifests = []
    total_needed = 0
    total_created = 0
    total_missing = 0
    for episode_id in episode_ids:
        review_dir = work_dir / "review" / "episodes" / episode_id / review_name
        episode_only_dir = review_dir / "video" / "episode_only"
        adjustment_dir = review_dir / "segments" / "adjustments"
        selected_videos = select_proc_videos(proc_video_root, episode_id)
        plans: dict[str, EndFadeAdjustment] = {}
        tail_trim_plans: dict[str, TailTrimAdjustment] = {}
        missing_segments: dict[str, str] = {}
        print(f"== {episode_id} adjustments ==", flush=True)
        for variant, video_path in selected_videos.items():
            source_segment = resolve_episode_only_segment(episode_only_dir, episode_id, variant, video_path)
            if not source_segment.is_file():
                missing_segments[variant] = str(source_segment)
                total_missing += 1
                print(f"  {variant}: missing episode-only segment: {source_segment}")
                continue
            tail_trim_plan = plan_tail_trim_adjustment(
                episode_id,
                variant,
                source_segment,
                adjustment_dir,
            )
            fade_source_segment = source_segment
            if tail_trim_plan is not None:
                tail_trim_plans[variant] = tail_trim_plan
                if tail_trim_plan.needed:
                    print(
                        f"  {variant}: configured tail trim {tail_trim_plan.trim_seconds:g}s; "
                        f"{tail_trim_plan.source_duration_seconds:.3f}s -> "
                        f"{tail_trim_plan.output_duration_seconds:.3f}s"
                    )
                    print(f"    output: {tail_trim_plan.output}")
                    print(f"    command: {shlex.join(tail_trim_plan.command)}")
                    if run:
                        adjustment_dir.mkdir(parents=True, exist_ok=True)
                        subprocess.run(tail_trim_plan.command, check=True)
                        total_created += 1
                    fade_source_segment = tail_trim_plan.output if run else source_segment
            plan = plan_end_fade_adjustment(
                variant,
                fade_source_segment,
                adjustment_dir,
                fade_frames=fade_frames,
                black_avg=black_avg,
                black_max=black_max,
                restored_ac3_bitrate=restored_ac3_bitrate,
                eng_ac3_bitrate=eng_ac3_bitrate,
                stereo_ac3_bitrate=stereo_ac3_bitrate,
            )
            plans[variant] = plan
            stats = plan.stats
            state = "needs fade" if plan.needed else "already black"
            print(
                f"  {variant}: {state}; avg_luma={stats['avg_luma']} "
                f"max_luma={stats['max_luma']} source={source_segment}"
            )
            if plan.needed:
                total_needed += 1
                print(f"    output: {plan.output}")
                print(f"    command: {shlex.join(plan.command)}")
                if run:
                    adjustment_dir.mkdir(parents=True, exist_ok=True)
                    subprocess.run(plan.command, check=True)
                    total_created += 1

        manifest = {
            "episode_id": episode_id,
            "episode_title": episode_display_title(episode_id),
            "review_name": review_name,
            "source": "episode-adjustments",
            "fade_frames": fade_frames,
            "final_black_hold_frames": 1,
            "black_avg": black_avg,
            "black_max": black_max,
            "missing_episode_only_segments": missing_segments,
            "tail_trim_adjustments": {variant: plan.as_manifest() for variant, plan in tail_trim_plans.items()},
            "end_fade_adjustments": {variant: plan.as_manifest() for variant, plan in plans.items()},
        }
        all_manifests.append(manifest)
        if run:
            adjustment_dir.mkdir(parents=True, exist_ok=True)
            write_manifest(adjustment_dir / "adjustments_manifest.json", manifest)
            write_adjustments_report(adjustment_dir / "ADJUSTMENTS_REPORT.md", manifest)

    elapsed = format_elapsed(time.monotonic() - started_at)
    print("")
    print(
        "Adjustment summary: "
        f"episodes={len(episode_ids)} needed={total_needed} created={total_created} "
        f"missing_episode_only={total_missing} elapsed={elapsed}"
    )
    if run:
        root_manifest = {
            "review_name": review_name,
            "episodes": all_manifests,
            "summary": {
                "episodes": len(episode_ids),
                "needed": total_needed,
                "created": total_created,
                "missing_episode_only": total_missing,
                "elapsed": elapsed,
            },
        }
        write_manifest(work_dir / "review" / "episodes" / f"{review_name}_adjustments_manifest.json", root_manifest)
    return 0


def resolve_episode_only_segment(episode_only_dir: Path, episode_id: str, variant: str, video_path: Path) -> Path:
    episode_output = Path(final_video_name(episode_id, variant, video_path))
    expected = episode_only_dir / f"{episode_output.stem}_episode_only.mkv"
    if expected.is_file():
        return expected
    legacy_patterns = {
        "ai_remaster": [f"{episode_id}_AIRemaster*_episode_only.mkv"],
        "remaster": [f"{episode_id}_Remaster_restored_audio_episode_only.mkv", f"{episode_id}_RemasterRestoredAudio_episode_only.mkv"],
        "remaster_49fps": [f"{episode_id}_Remaster_49fps*_episode_only.mkv", f"{episode_id}_Remaster49fps*_episode_only.mkv"],
    }
    for pattern in legacy_patterns.get(variant, []):
        matches = sorted(episode_only_dir.glob(pattern))
        if matches:
            return matches[0]
    return expected


def cmd_prepare_subtitles(
    episode: str,
    eng_root: Path,
    spa_root: Path,
    out_dir: Path,
    run: bool,
) -> int:
    episode_ids = list(EPISODE_TITLES) if episode.lower() == "all" else [normalize_episode_id(episode)]
    manifests = []
    for episode_id in episode_ids:
        episode_dir = out_dir / episode_id
        english_source = find_english_subtitle_mkv(eng_root, episode_id)
        spanish_source = find_spanish_ssa(spa_root, episode_id)
        english_stream = subtitle_stream_index(english_source, language="eng")
        english_sup = episode_dir / f"{episode_id}_english_pgs.sup"
        spanish_srt = episode_dir / f"{episode_id}_spanish_clean.srt"
        extract_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(english_source),
            "-map",
            f"0:{english_stream}",
            "-c:s",
            "copy",
            str(english_sup),
        ]
        print(shlex.join(extract_cmd), flush=True)
        if run:
            episode_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(extract_cmd, check=True)
            spanish_count = write_clean_spanish_srt(spanish_source, spanish_srt)
            manifest = {
                "episode_id": episode_id,
                "episode_title": episode_display_title(episode_id),
                "english_source": str(english_source),
                "english_stream_index": english_stream,
                "english_output": str(english_sup),
                "english_format": "PGS/SUP image subtitles; OCR is required for SRT text.",
                "spanish_source": str(spanish_source),
                "spanish_output": str(spanish_srt),
                "spanish_format": "Clean UTF-8 SRT converted from SSA dialogue events.",
                "spanish_dialogue_count": spanish_count,
                "removed_spanish_lines": "SSA metadata/styles and translation-credit/URL promo dialogue lines.",
                "commands": [extract_cmd],
            }
            write_manifest(episode_dir / "manifest.json", manifest)
            manifests.append(manifest)
            print(f"{episode_id}: english={english_sup} spanish={spanish_srt} spanish_lines={spanish_count}")
        else:
            print(f"spanish SSA cleanup: {spanish_source} -> {spanish_srt}")
    if run:
        write_manifest(out_dir / "manifest.json", {"episodes": manifests})
        print(f"subtitle_dir={out_dir}")
    else:
        print(f"subtitle outputs will be created after --run: {out_dir}")
    return 0


def cmd_export_done(
    episode: str,
    work_dir: Path,
    review_name: str,
    proc_video_root: Path,
    subtitle_dir: Path,
    out_dir: Path,
    if_exists: str,
    attach_cover_art: bool,
    run: bool,
) -> int:
    started_at = time.monotonic()
    episode_ids = list(EPISODE_TITLES) if episode.lower() == "all" else [normalize_episode_id(episode)]
    if run:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_commands: list[list[str]] = []
    exported: dict[str, dict[str, str]] = {}
    skipped: dict[str, list[str]] = {}
    warnings: list[str] = []

    for episode_id in episode_ids:
        review_dir = work_dir / "review" / "episodes" / episode_id / review_name
        review_video_dir = review_dir / "video"
        selected_videos = select_proc_videos(proc_video_root, episode_id)
        subtitles = discover_srt_subtitles(subtitle_dir, episode_id)

        planned_outputs: dict[str, Path] = {}
        source_by_variant: dict[str, Path] = {}
        for variant, source_video in selected_videos.items():
            source = find_review_final_video(review_video_dir, episode_id, variant, source_video)
            source_by_variant[variant] = source
            planned_outputs[variant] = out_dir / final_video_name(episode_id, variant, source_video)

        if run and any(path.exists() for path in planned_outputs.values()):
            decision = decide_existing_outputs(episode_id, planned_outputs, if_exists)
            if decision == "skip":
                skipped[episode_id] = [str(path) for path in planned_outputs.values()]
                print(f"Skipping export for {episode_id}: done output already exists")
                continue

        exported[episode_id] = {}
        for variant, output in planned_outputs.items():
            source = source_by_variant[variant]
            copy_cmd = ["cp", str(source), str(output)]
            all_commands.append(copy_cmd)
            print(shlex.join(copy_cmd), flush=True)
            if run:
                shutil.copy2(source, output)
                exported[episode_id][variant] = str(output)
            for subtitle in subtitles:
                sidecar = output.with_suffix(f".{subtitle.sidecar_suffix}.srt")
                sidecar_cmd = ["cp", str(subtitle.path), str(sidecar)]
                all_commands.append(sidecar_cmd)
                print(shlex.join(sidecar_cmd), flush=True)
                if run:
                    shutil.copy2(subtitle.path, sidecar)

    if run:
        manifest = {
            "output_dir": str(out_dir),
            "review_name": review_name,
            "copy_only": True,
            "episodes": exported,
            "skipped": skipped,
            "warnings": warnings,
            "commands": all_commands,
            "notes": [
                "Export copies already-authored review MKVs without remuxing.",
                "Sidecar SRT files are copied next to each exported MKV for players that do not handle embedded subtitles well.",
                "Review MKVs should already contain embedded SRT subtitle streams and cover art when those assets were available during episode-final-build.",
            ],
        }
        write_manifest(out_dir / "export_manifest.json", manifest)
        write_done_export_report(out_dir / "EXPORT_REPORT.md", manifest, time.monotonic() - started_at)
        print_done_export_summary(out_dir, exported, skipped, warnings, time.monotonic() - started_at)
    else:
        print(f"done output will be created after --run: {out_dir}")
    return 0


def cmd_ocr_english_subtitles(
    episode: str,
    eng_root: Path,
    out_dir: Path,
    tesseract: str,
    crop_top_ratio: float,
    min_display_packet_size: int,
    keep_images: bool,
    run: bool,
) -> int:
    episode_ids = list(EPISODE_TITLES) if episode.lower() == "all" else [normalize_episode_id(episode)]
    if not 0.0 < crop_top_ratio < 0.95:
        raise SystemExit("--crop-top-ratio must be between 0 and 0.95")

    if run and shutil.which(tesseract) is None:
        raise SystemExit(
            f"Missing OCR tool: {tesseract}\n"
            "Install it with: sudo apt install -y tesseract-ocr tesseract-ocr-eng"
        )

    manifests = []
    for episode_id in episode_ids:
        source = find_english_subtitle_mkv(eng_root, episode_id)
        stream_index = subtitle_stream_index(source, language="eng")
        events = pgs_timing_events(source, min_display_packet_size=min_display_packet_size)
        episode_dir = out_dir / episode_id
        english_srt = episode_dir / f"{episode_id}_english_clean.srt"
        image_dir = episode_dir / "ocr_frames"
        print(f"{episode_id}: {len(events)} candidate PGS cues from {source}")
        if not run:
            print(f"would OCR stream {stream_index} -> {english_srt}")
            continue

        episode_dir.mkdir(parents=True, exist_ok=True)
        if keep_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        cues: list[SrtCue] = []
        with tempfile.TemporaryDirectory(prefix=f"{episode_id}_ocr_") as temp_dir:
            temp_path = Path(temp_dir)
            for index, event in enumerate(events, start=1):
                image_path = image_dir / f"{index:04d}.png" if keep_images else temp_path / f"{index:04d}.png"
                render_pgs_subtitle_frame(source, event.sample_time, image_path, crop_top_ratio)
                text = ocr_image(image_path, tesseract)
                if not text or is_ocr_noise(text, event.sample_time):
                    continue
                cues.append(SrtCue(index=len(cues) + 1, start=event.start, end=event.end, text=text))
                print(f"{episode_id} OCR {index}/{len(events)} -> cue {len(cues)}", flush=True)
        write_srt(english_srt, cues)
        manifest = {
            "episode_id": episode_id,
            "episode_title": episode_display_title(episode_id),
            "source": str(source),
            "subtitle_stream_index": stream_index,
            "candidate_pgs_cues": len(events),
            "ocr_cues_written": len(cues),
            "output": str(english_srt),
            "ocr_tool": tesseract,
            "crop_top_ratio": crop_top_ratio,
            "kept_images": keep_images,
            "notes": "OCR is review material; manually check names, punctuation, and sci-fi words.",
        }
        write_manifest(episode_dir / "english_ocr_manifest.json", manifest)
        manifests.append(manifest)
        print(f"{episode_id}: english_ocr_srt={english_srt} cues={len(cues)}")
    if run:
        write_manifest(out_dir / "english_ocr_manifest.json", {"episodes": manifests})
    return 0


def cmd_retime_spanish_subtitles(episode: str, subtitle_dir: Path, run: bool) -> int:
    episode_ids = list(EPISODE_TITLES) if episode.lower() == "all" else [normalize_episode_id(episode)]
    manifests = []
    for episode_id in episode_ids:
        episode_dir = subtitle_dir / episode_id
        english_srt = episode_dir / f"{episode_id}_english_clean.srt"
        spanish_srt = episode_dir / f"{episode_id}_spanish_clean.srt"
        output = episode_dir / f"{episode_id}_spanish_english_timed.srt"
        if not english_srt.is_file():
            raise SystemExit(f"Missing English OCR SRT: {english_srt}")
        if not spanish_srt.is_file():
            raise SystemExit(f"Missing Spanish clean SRT: {spanish_srt}")
        english_cues = read_srt(english_srt)
        spanish_cues = read_srt(spanish_srt)
        mapped_count = min(len(english_cues), len(spanish_cues))
        retimed: list[SrtCue] = []
        for index in range(mapped_count):
            retimed.append(
                SrtCue(
                    index=index + 1,
                    start=english_cues[index].start,
                    end=english_cues[index].end,
                    text=spanish_cues[index].text,
                )
            )
        warnings = []
        if len(english_cues) != len(spanish_cues):
            warnings.append(
                f"Cue count mismatch: English OCR has {len(english_cues)} cues; Spanish clean has {len(spanish_cues)} cues. "
                f"Only the first {mapped_count} cues were retimed by index."
            )
        print(f"{episode_id}: english={len(english_cues)} spanish={len(spanish_cues)} mapped={mapped_count} -> {output}")
        if run:
            write_srt(output, retimed)
            manifest = {
                "episode_id": episode_id,
                "episode_title": episode_display_title(episode_id),
                "english_timing_source": str(english_srt),
                "spanish_text_source": str(spanish_srt),
                "output": str(output),
                "english_cues": len(english_cues),
                "spanish_cues": len(spanish_cues),
                "mapped_cues": mapped_count,
                "warnings": warnings,
                "notes": "Sequential timing transfer only; review manually before final embedding.",
            }
            write_manifest(episode_dir / "spanish_retime_manifest.json", manifest)
            manifests.append(manifest)
            for warning in warnings:
                print(f"WARNING: {warning}")
    if run:
        write_manifest(subtitle_dir / "spanish_retime_manifest.json", {"episodes": manifests})
    return 0


def cmd_translate_spanish_subtitles(
    episode: str,
    provider: str,
    model: str,
    ollama_url: str,
    model_config: Path,
    model_cache: Path,
    offline: bool,
    llm_python: str,
    limit_chunks: int | None,
    subtitle_dir: Path,
    glossary_path: Path,
    chunk_size: int,
    temperature: float,
    retries: int,
    overwrite: bool,
    run: bool,
) -> int:
    if chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than zero")
    if retries < 0:
        raise SystemExit("--retries must be zero or greater")
    episode_ids = list(EPISODE_TITLES) if episode.lower() == "all" else [normalize_episode_id(episode)]
    batch_mode = len(episode_ids) > 1
    glossary = read_glossary(glossary_path)
    manifests = []
    started_all = time.perf_counter()
    summary = {"translated": 0, "skipped": 0, "planned": 0, "failed": 0}

    for episode_id in episode_ids:
        episode_started = time.perf_counter()
        episode_dir = subtitle_dir / episode_id
        english_srt = episode_dir / f"{episode_id}_english_clean.srt"
        spanish_reference = episode_dir / f"{episode_id}_spanish_clean.srt"
        output = episode_dir / f"{episode_id}_spanish_translated.srt"
        batch_dir = episode_dir / "translation_batches"
        prompt_jsonl = batch_dir / f"{episode_id}_{provider}_{model_safe_name(model)}_prompts.jsonl"
        response_jsonl = batch_dir / f"{episode_id}_{provider}_{model_safe_name(model)}_responses.jsonl"
        if not english_srt.is_file():
            message = f"Missing English OCR SRT: {english_srt}"
            if not batch_mode:
                raise SystemExit(message)
            summary["failed"] += 1
            elapsed = format_duration(time.perf_counter() - episode_started)
            manifest = {
                "episode_id": episode_id,
                "episode_title": episode_display_title(episode_id),
                "provider": provider,
                "model": model,
                "status": "failed_missing_english_srt",
                "error": message,
                "elapsed": elapsed,
            }
            manifests.append(manifest)
            print(f"{episode_id}: FAILED {message} elapsed={elapsed}")
            continue
        english_cues = read_srt(english_srt)
        reference_cues = read_srt(spanish_reference) if spanish_reference.is_file() else []
        chunks = list(chunk_cues(english_cues, chunk_size))
        actual_output = limited_hf_path(output, limit_chunks) if provider == "hf" else output
        print(f"{episode_id}: english_cues={len(english_cues)} chunks={len(chunks)} provider={provider} model={model}")
        if run and actual_output.is_file() and not overwrite:
            summary["skipped"] += 1
            elapsed = format_duration(time.perf_counter() - episode_started)
            manifest = {
                "episode_id": episode_id,
                "episode_title": episode_display_title(episode_id),
                "provider": provider,
                "model": model,
                "output": str(actual_output),
                "status": "skipped_existing_output",
                "elapsed": elapsed,
                "notes": "Use --overwrite to rebuild this translated subtitle file.",
            }
            manifests.append(manifest)
            print(f"{episode_id}: SKIP existing translated SRT: {actual_output} elapsed={elapsed}")
            continue
        if not run:
            summary["planned"] += 1
            print(f"would write prompt batch: {prompt_jsonl}")
            if provider == "hf":
                hf_output = actual_output
                hf_cmd = hf_translation_command(
                    llm_python=llm_python,
                    prompt_jsonl=prompt_jsonl,
                    english_srt=english_srt,
                    output_srt=output,
                    response_jsonl=response_jsonl,
                    model_config=model_config,
                    model_cache=model_cache,
                    offline=offline,
                    limit_chunks=limit_chunks,
                    retries=retries,
                    overwrite=overwrite,
                )
                print("would run local HF translator:")
                print("  " + shlex.join(hf_cmd))
                print(f"would write translated SRT: {hf_output}")
            elif provider == "ollama":
                print(f"would call Ollama endpoint: {ollama_url}")
                print(f"would write translated SRT: {output}")
            continue

        batch_dir.mkdir(parents=True, exist_ok=True)
        prompt_records = [
            build_translation_prompt_record(episode_id, chunk_index, cue_chunk, english_cues, reference_cues, glossary)
            for chunk_index, cue_chunk in enumerate(chunks, start=1)
        ]
        write_jsonl(prompt_jsonl, prompt_records)

        if provider == "prompt-batch":
            elapsed = format_duration(time.perf_counter() - episode_started)
            manifest = {
                "episode_id": episode_id,
                "episode_title": episode_display_title(episode_id),
                "provider": provider,
                "model": model,
                "english_source": str(english_srt),
                "spanish_reference": str(spanish_reference) if spanish_reference.is_file() else None,
                "prompt_batch": str(prompt_jsonl),
                "output_target": str(output),
                "chunks": len(chunks),
                "elapsed": elapsed,
                "status": "prompt_batch_written",
                "notes": "Prompt batch only. Run with an external/local LLM and import responses later.",
            }
            write_manifest(episode_dir / "spanish_translation_manifest.json", manifest)
            manifests.append(manifest)
            summary["translated"] += 1
            print(f"{episode_id}: prompt_batch={prompt_jsonl} elapsed={elapsed}")
            continue

        if provider == "hf":
            hf_cmd = hf_translation_command(
                llm_python=llm_python,
                prompt_jsonl=prompt_jsonl,
                english_srt=english_srt,
                output_srt=output,
                response_jsonl=response_jsonl,
                model_config=model_config,
                model_cache=model_cache,
                offline=offline,
                limit_chunks=limit_chunks,
                retries=retries,
                overwrite=overwrite,
            )
            print(shlex.join(hf_cmd))
            try:
                subprocess.run(hf_cmd, check=True)
            except subprocess.CalledProcessError as exc:
                summary["failed"] += 1
                elapsed = format_duration(time.perf_counter() - episode_started)
                manifest = {
                    "episode_id": episode_id,
                    "episode_title": episode_display_title(episode_id),
                    "provider": provider,
                    "model": model,
                    "status": "failed_hf_runner",
                    "error": str(exc),
                    "elapsed": elapsed,
                }
                manifests.append(manifest)
                print(f"{episode_id}: FAILED local HF runner exit={exc.returncode} elapsed={elapsed}")
                if batch_mode:
                    continue
                raise
            elapsed = format_duration(time.perf_counter() - episode_started)
            manifest = {
                "episode_id": episode_id,
                "episode_title": episode_display_title(episode_id),
                "provider": provider,
                "model": model,
                "model_config": str(model_config),
                "model_cache": str(model_cache),
                "offline": offline,
                "english_source": str(english_srt),
                "spanish_reference": str(spanish_reference) if spanish_reference.is_file() else None,
                "prompt_batch": str(prompt_jsonl),
                "response_batch": str(response_jsonl),
                "output": str(actual_output),
                "chunks": len(chunks),
                "elapsed": elapsed,
                "status": "translated",
                "notes": "Generated by the local Hugging Face runner; review before final mux.",
            }
            write_manifest(episode_dir / "spanish_translation_manifest.json", manifest)
            manifests.append(manifest)
            summary["translated"] += 1
            print(f"{episode_id}: DONE spanish_translated_srt={actual_output} elapsed={elapsed}")
            continue

        response_records = []
        translated: dict[int, str] = {}
        for record in prompt_records:
            raw_response = call_ollama(ollama_url, model, record["prompt"], temperature)
            translations = parse_translation_response(raw_response)
            response_record = {
                "chunk_index": record["chunk_index"],
                "cue_indexes": record["cue_indexes"],
                "raw_response": raw_response,
                "translations": translations,
            }
            response_records.append(response_record)
            for item in translations:
                translated[int(item["index"])] = str(item["text"]).strip()
            print(f"{episode_id}: translated chunk {record['chunk_index']}/{len(prompt_records)}", flush=True)
        write_jsonl(response_jsonl, response_records)
        translated_cues = []
        missing = []
        for cue in english_cues:
            text = translated.get(cue.index)
            if not text:
                missing.append(cue.index)
                text = cue.text
            translated_cues.append(SrtCue(index=cue.index, start=cue.start, end=cue.end, text=text))
        write_srt(output, translated_cues)
        elapsed = format_duration(time.perf_counter() - episode_started)
        manifest = {
            "episode_id": episode_id,
            "episode_title": episode_display_title(episode_id),
            "provider": provider,
            "model": model,
            "english_source": str(english_srt),
            "spanish_reference": str(spanish_reference) if spanish_reference.is_file() else None,
            "prompt_batch": str(prompt_jsonl),
            "response_batch": str(response_jsonl),
            "output": str(output),
            "chunks": len(chunks),
            "translated_cues": len(translated),
            "missing_cue_indexes": missing,
            "elapsed": elapsed,
            "status": "translated",
            "notes": "Review before final mux; local LLM output may need subtitle wording cleanup.",
        }
        write_manifest(episode_dir / "spanish_translation_manifest.json", manifest)
        manifests.append(manifest)
        summary["translated"] += 1
        print(f"{episode_id}: DONE spanish_translated_srt={output} translated={len(translated)} missing={len(missing)} elapsed={elapsed}")

    total_elapsed = format_duration(time.perf_counter() - started_all)
    print(
        "translation summary: "
        f"translated={summary['translated']} skipped={summary['skipped']} "
        f"planned={summary['planned']} failed={summary['failed']} elapsed={total_elapsed}"
    )
    if run:
        write_manifest(
            subtitle_dir / "spanish_translation_manifest.json",
            {"episodes": manifests, "summary": {**summary, "elapsed": total_elapsed}},
        )
    return 0


def cmd_import_spanish_translation(episode: str, responses: Path, subtitle_dir: Path, run: bool) -> int:
    episode_id = normalize_episode_id(episode)
    episode_dir = subtitle_dir / episode_id
    english_srt = episode_dir / f"{episode_id}_english_clean.srt"
    output = episode_dir / f"{episode_id}_spanish_translated.srt"
    if not english_srt.is_file():
        raise SystemExit(f"Missing English OCR SRT: {english_srt}")
    if not responses.is_file():
        raise SystemExit(f"Missing response JSONL: {responses}")
    english_cues = read_srt(english_srt)
    translated: dict[int, str] = {}
    for record in read_jsonl(responses):
        if "translations" in record:
            items = record["translations"]
        elif "raw_response" in record:
            items = parse_translation_response(str(record["raw_response"]))
        elif "response" in record:
            items = parse_translation_response(str(record["response"]))
        else:
            items = parse_translation_response(json.dumps(record, ensure_ascii=False))
        if not isinstance(items, list):
            raise SystemExit(f"Invalid response record in {responses}: {record}")
        for item in items:
            if not isinstance(item, dict):
                raise SystemExit(f"Invalid translation item in {responses}: {item}")
            translated[int(item["index"])] = str(item["text"]).strip()
    missing = []
    cues = []
    for cue in english_cues:
        text = translated.get(cue.index)
        if not text:
            missing.append(cue.index)
            text = cue.text
        cues.append(SrtCue(index=cue.index, start=cue.start, end=cue.end, text=text))
    print(f"{episode_id}: responses={responses} translated={len(translated)} missing={len(missing)} -> {output}")
    if not run:
        return 0
    write_srt(output, cues)
    manifest = {
        "episode_id": episode_id,
        "episode_title": episode_display_title(episode_id),
        "english_timing_source": str(english_srt),
        "responses": str(responses),
        "output": str(output),
        "translated_cues": len(translated),
        "missing_cue_indexes": missing,
    }
    write_manifest(episode_dir / "spanish_translation_import_manifest.json", manifest)
    return 0


def cmd_speech_map(
    work_dir: Path,
    episode_id: str,
    audio: Path | None,
    source: str,
    review_name: str,
    engine: str,
    model: str,
    language: str,
    device: str,
    compute_type: str,
    batch_size: int,
    python: Path,
    out_dir: Path,
    overwrite: bool,
    run: bool,
) -> int:
    if episode_id.lower() == "all":
        if audio is not None:
            raise SystemExit("--audio cannot be used with speech-map all")
        started_all = time.monotonic()
        summary = {"created": 0, "skipped": 0, "planned": 0, "failed": 0}
        for current_episode in EPISODE_TITLES:
            try:
                result = cmd_speech_map(
                    work_dir=work_dir,
                    episode_id=current_episode,
                    audio=None,
                    source=source,
                    review_name=review_name,
                    engine=engine,
                    model=model,
                    language=language,
                    device=device,
                    compute_type=compute_type,
                    batch_size=batch_size,
                    python=python,
                    out_dir=out_dir,
                    overwrite=overwrite,
                    run=run,
                )
                if result == 0:
                    expected = out_dir / current_episode / f"{current_episode}_{source}_speech_map.json"
                    if expected.is_file():
                        summary["created"] += 1
                    else:
                        summary["planned"] += 1
                elif result == 3:
                    summary["skipped"] += 1
                else:
                    summary["failed"] += 1
            except subprocess.CalledProcessError as exc:
                summary["failed"] += 1
                print(f"{current_episode}: FAILED speech-map exit={exc.returncode}")
                continue
        print(
            "speech-map summary: "
            f"created={summary['created']} skipped={summary['skipped']} "
            f"planned={summary['planned']} failed={summary['failed']} "
            f"elapsed={format_elapsed(time.monotonic() - started_all)}"
        )
        return 1 if summary["failed"] else 0

    episode_id = normalize_episode_id(episode_id)
    audio_path = audio or default_speech_map_audio(work_dir, episode_id, review_name, source)
    source_label = source if audio is None else model_safe_name(audio_path.stem)
    episode_out_dir = out_dir / episode_id
    expected_json = episode_out_dir / f"{episode_id}_{source_label}_speech_map.json"
    if expected_json.is_file() and not overwrite:
        print(f"{episode_id}: SKIP existing speech_map={expected_json}")
        print("Use --overwrite to rebuild it.")
        return 3
    command = [
        str(python),
        "scripts/run_speech_map_asr.py",
        "--audio",
        str(audio_path),
        "--out-dir",
        str(episode_out_dir),
        "--episode",
        episode_id,
        "--source-label",
        source_label,
        "--engine",
        engine,
        "--model",
        model,
        "--language",
        language,
        "--device",
        device,
        "--compute-type",
        compute_type,
        "--batch-size",
        str(batch_size),
    ]
    print(f"{episode_id}: speech-map source={source_label}")
    print(f"audio={audio_path}")
    print(f"output_dir={episode_out_dir}")
    print(shlex.join(command))
    if not run:
        return 0
    if not audio_path.is_file():
        raise SystemExit(f"Missing ASR source audio: {audio_path}")
    if not python.exists():
        raise SystemExit(f"Missing ASR Python env: {python}. Create .venv-asr and install requirements-asr.txt.")
    started = time.monotonic()
    subprocess.run(command, check=True)
    print(f"{episode_id}: speech map complete in {format_elapsed(time.monotonic() - started)}")
    return 0


def default_speech_map_audio(work_dir: Path, episode_id: str, review_name: str, source: str) -> Path:
    mux_dir = work_dir / "final_episode_mux" / episode_id / review_name
    if source == "dialogue":
        fixed = mux_dir / f"{episode_id}_spa1_dialogue_broadcast_strong_{EPISODE_AUDIO_PATCH_VERSION}.wav"
        if fixed.is_file():
            return fixed
        return mux_dir / f"{episode_id}_spa1_dialogue_broadcast_strong.wav"
    if source == "spa1_fullmix":
        fixed = mux_dir / f"{episode_id}_spa1_stereo_vhs_broadcast_full_silenced_{EPISODE_AUDIO_PATCH_VERSION}.wav"
        if fixed.is_file():
            return fixed
        return mux_dir / f"{episode_id}_spa1_stereo_vhs_broadcast_full_silenced.wav"
    raise SystemExit(f"Unsupported speech-map source: {source}")


def cmd_speech_find(
    episode_id: str,
    phrase: str,
    source: str,
    map_path: Path | None,
    speech_map_dir: Path,
    around: str | None,
    window: float,
    limit: int,
) -> int:
    episode_id = normalize_episode_id(episode_id)
    speech_map = map_path or speech_map_dir / episode_id / f"{episode_id}_{source}_speech_map.json"
    if not speech_map.is_file():
        raise SystemExit(f"Missing speech map: {speech_map}. Run speech-map first.")
    data = json.loads(speech_map.read_text(encoding="utf-8"))
    words = data.get("words", [])
    if not isinstance(words, list) or not words:
        raise SystemExit(f"Speech map has no words: {speech_map}")
    around_seconds = parse_flexible_seconds(around) if around else None
    candidates = find_phrase_candidates(words, phrase, around_seconds=around_seconds, window=window, limit=limit)
    print(f"speech_map={speech_map}")
    print(f"query={phrase!r}")
    if around_seconds is not None:
        print(f"around={seconds_to_clock(around_seconds)} window=+/-{window:.1f}s")
    if not candidates:
        print("No candidates found.")
        return 1
    for index, candidate in enumerate(candidates, start=1):
        distance = ""
        if around_seconds is not None:
            midpoint = (candidate["start"] + candidate["end"]) / 2.0
            distance = f" distance={midpoint - around_seconds:+.3f}s"
        print(
            f"{index:02d}. score={candidate['score']:.3f} "
            f"{seconds_to_clock(candidate['start'])} -> {seconds_to_clock(candidate['end'])}"
            f"{distance} words={candidate['first_word_index']}-{candidate['last_word_index']}"
        )
        print(f"    {candidate['text']}")
    return 0


def cmd_tts_summary_plan(
    episode_id: str,
    start: str | None,
    end: str | None,
    source: str,
    speech_map_dir: Path,
    out_dir: Path,
    summary_id: str,
    config: Path,
    overwrite: bool,
    run: bool,
) -> int:
    if episode_id.lower() == "all":
        return cmd_tts_summary_plan_all(
            source=source,
            speech_map_dir=speech_map_dir,
            out_dir=out_dir,
            summary_id=summary_id,
            config=config,
            overwrite=overwrite,
            run=run,
        )
    episode_id = normalize_episode_id(episode_id)
    configured = summary_config_entry(config, episode_id)
    if start is None:
        start = str(configured.get("start") or "") if configured else ""
    if not start:
        status = configured.get("status", "missing") if configured else "missing"
        print(f"{episode_id}: no configured summary start; status={status}")
        return 2
    start_seconds = parse_flexible_seconds(start)
    speech_map = speech_map_dir / episode_id / f"{episode_id}_{source}_speech_map.json"
    if not speech_map.is_file():
        speech_cmd = [
            ".venv-asr/bin/python",
            "-m",
            "robotech_ai.cli",
            "speech-map",
            episode_id,
            "--source",
            source,
            "--engine",
            "whisperx",
            "--model",
            "large-v3",
            "--language",
            "es",
            "--device",
            "cuda",
            "--compute-type",
            "float16",
            "--batch-size",
            "4",
            "--run",
        ]
        print(f"Missing speech map: {speech_map}")
        print("Create the full-episode Spanish speech map first:")
        print("PYTHONPATH=src " + shlex.join(speech_cmd))
        return 2

    data = json.loads(speech_map.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    end_seconds = parse_flexible_seconds(end) if end else infer_summary_end_seconds(segments, start_seconds)
    if end_seconds <= start_seconds:
        raise SystemExit(f"--end must be after --start: {start} -> {end or seconds_to_clock(end_seconds)}")
    phrases = []
    for segment in segments:
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        if segment_end < start_seconds or segment_start > end_seconds:
            continue
        phrase_start = max(segment_start, start_seconds)
        phrase_end = min(segment_end, end_seconds)
        if phrase_end <= phrase_start:
            continue
        phrases.append(
            {
                "number": len(phrases) + 1,
                "source_segment_index": segment.get("index"),
                "start": round(phrase_start, 3),
                "end": round(phrase_end, 3),
                "duration": round(phrase_end - phrase_start, 3),
                "text": text,
            }
        )

    if not phrases:
        raise SystemExit(
            f"No ASR phrases found in {speech_map} between {seconds_to_clock(start_seconds)} and {seconds_to_clock(end_seconds)}"
        )

    episode_out = out_dir / episode_id / summary_id
    plan_path = episode_out / "phrase_plan.json"
    if plan_path.exists() and not overwrite:
        print(f"{episode_id}: phrase plan already exists: {plan_path}")
        print("Use --overwrite to rewrite it.")
        return 0
    plan = {
        "kind": "robotech_qwen3_tts_summary_plan",
        "episode": episode_id,
        "episode_title": EPISODE_TITLES.get(episode_id, ""),
        "summary_id": summary_id,
        "source": source,
        "speech_map": str(speech_map),
        "start": round(start_seconds, 3),
        "end": round(end_seconds, 3),
        "duration": round(end_seconds - start_seconds, 3),
        "start_clock": seconds_to_clock(start_seconds),
        "end_clock": seconds_to_clock(end_seconds),
        "phrases": phrases,
    }
    print(f"{episode_id}: next-summary TTS phrase plan")
    print(f"speech_map={speech_map}")
    print(f"range={seconds_to_clock(start_seconds)} -> {seconds_to_clock(end_seconds)}")
    print(f"phrases={len(phrases)}")
    print(f"plan={plan_path}")
    for phrase in phrases:
        print(
            f"{phrase['number']:02d}. {seconds_to_clock(float(phrase['start']))} -> "
            f"{seconds_to_clock(float(phrase['end']))} {phrase['text']}"
        )
    if not run:
        print("Add --run to write the phrase plan.")
        return 0
    episode_out.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_manifest(episode_out / "manifest.json", {"phrase_plan": str(plan_path), "phrases": len(phrases)})
    print(f"wrote={plan_path}")
    return 0


def load_summary_config(config: Path) -> dict[str, object]:
    """Load configured next-episode summary start points."""

    if not config.is_file():
        return {"episodes": {}}
    return json.loads(config.read_text(encoding="utf-8"))


def summary_config_entry(config: Path, episode_id: str) -> dict[str, object]:
    payload = load_summary_config(config)
    episodes = payload.get("episodes", {})
    if not isinstance(episodes, dict):
        return {}
    entry = episodes.get(normalize_episode_id(episode_id), {})
    return entry if isinstance(entry, dict) else {}


def configured_summary_start_seconds(config: Path, episode_id: str) -> float | None:
    entry = summary_config_entry(config, episode_id)
    start = entry.get("start")
    status = str(entry.get("status", "ready"))
    if not start or status in {"pending", "none", "skip"}:
        return None
    return parse_flexible_seconds(str(start))


def infer_summary_end_seconds(segments: list[object], start_seconds: float) -> float:
    """Use the last ASR phrase after the summary start as the plan end."""

    ends: list[float] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            segment_start = float(segment.get("start", 0.0))
            segment_end = float(segment.get("end", segment_start))
        except (TypeError, ValueError):
            continue
        text = str(segment.get("text", "")).strip()
        if text and segment_end >= start_seconds:
            ends.append(segment_end)
    if not ends:
        raise SystemExit(f"Could not infer summary end after {seconds_to_clock(start_seconds)}")
    return max(ends)


def cmd_tts_summary_plan_all(
    source: str,
    speech_map_dir: Path,
    out_dir: Path,
    summary_id: str,
    config: Path,
    overwrite: bool,
    run: bool,
) -> int:
    payload = load_summary_config(config)
    episodes = payload.get("episodes", {})
    if not isinstance(episodes, dict) or not episodes:
        raise SystemExit(f"No episodes found in summary config: {config}")
    completed = 0
    skipped = 0
    failed: list[str] = []
    print(f"summary_config={config}")
    for episode_id in sorted(episodes):
        entry = episodes.get(episode_id, {})
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "ready"))
        start = entry.get("start")
        if not start or status in {"pending", "none", "skip"}:
            print(f"{episode_id}: skip status={status} start={start}")
            skipped += 1
            continue
        result = cmd_tts_summary_plan(
            episode_id=episode_id,
            start=str(start),
            end=None,
            source=source,
            speech_map_dir=speech_map_dir,
            out_dir=out_dir,
            summary_id=summary_id,
            config=config,
            overwrite=overwrite,
            run=run,
        )
        if result == 0:
            completed += 1
        else:
            failed.append(f"{episode_id}: exit={result}")
    print(f"tts-summary-plan all: completed={completed} skipped={skipped} failed={len(failed)}")
    for item in failed:
        print(f"FAILED {item}")
    return 1 if failed else 0


def cmd_tts_summary_generate(
    episode_id: str,
    plan: Path | None,
    start: str | None,
    end: str | None,
    source: str,
    speech_map_dir: Path,
    summary_id: str,
    out_dir: Path,
    qwen_root: Path,
    qwen_python: Path,
    ref_audio: Path,
    ref_text: Path,
    takes: int,
    phrases: list[str] | None,
    replace_phrase: bool,
    model_size: str,
    language: str,
    chunk_size: int,
    chunk_gap: float,
    seed_base: int,
    exact_seed: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None,
    subtalker_temperature: float | None,
    subtalker_top_p: float | None,
    subtalker_top_k: int | None,
    device: str,
    x_vector_only: bool,
    overwrite: bool,
    assemble_only: bool,
    fit_to_slots: bool,
    slot_margin: float,
    speed_all_percent: float,
    avoid_overlap: bool,
    min_gap: float,
    balance_phrases: bool,
    balance_max_gain_db: float,
    summary_gain_db: float,
    run: bool,
) -> int:
    episode_id = normalize_episode_id(episode_id)
    episode_out = out_dir / episode_id / summary_id
    plan_path = plan or episode_out / "phrase_plan.json"
    plan_will_be_created_on_run = False
    if not plan_path.is_file():
        if start and end and plan is None:
            plan_result = cmd_tts_summary_plan(
                episode_id=episode_id,
                start=start,
                end=end,
                source=source,
                speech_map_dir=speech_map_dir,
                out_dir=out_dir,
                summary_id=summary_id,
                config=DEFAULT_SUMMARY_CONFIG,
                overwrite=overwrite,
                run=run,
            )
            if plan_result != 0:
                return plan_result
            plan_will_be_created_on_run = not run
        else:
            print(f"Missing phrase plan: {plan_path}")
            print("Either create it first, or pass --start and --end to this command.")
            print("Create it first, for example:")
            print(
                shlex.join(
                    [
                        "robotech-ai",
                        "tts-summary-plan",
                        episode_id,
                        "--start",
                        "00:21:30",
                        "--end",
                        "00:22:10",
                        "--run",
                    ]
                )
            )
            print("Or generate in one step:")
            print(
                shlex.join(
                    [
                        "robotech-ai",
                        "tts-summary-generate",
                        episode_id,
                        "--start",
                        "00:21:30",
                        "--end",
                        "00:22:10",
                        "--run",
                    ]
                )
            )
            print("Use the actual start/end for the next-episode narrator summary.")
            return 2
    if not plan_path.is_file() and not plan_will_be_created_on_run:
        print(f"Missing phrase plan: {plan_path}")
        return 2
    if plan_path.is_file() and phrases:
        phrase_validation = validate_tts_phrase_selection(plan_path, phrases)
        if phrase_validation is not None:
            print(phrase_validation)
            return 2
    if not qwen_python.exists():
        raise SystemExit(f"Missing Qwen Python env: {qwen_python}")
    if not qwen_root.is_dir():
        raise SystemExit(f"Missing Qwen app folder: {qwen_root}")
    if not ref_audio.is_file():
        raise SystemExit(f"Missing narrator reference audio: {ref_audio}")
    if not ref_text.is_file():
        raise SystemExit(f"Missing narrator reference transcript: {ref_text}")
    if takes < 1:
        raise SystemExit("--takes must be at least 1")

    command = [
        str(qwen_python),
        "scripts/run_qwen3_tts_summary.py",
        "--qwen-root",
        str(qwen_root),
        "--plan",
        str(plan_path),
        "--out-dir",
        str(episode_out),
        "--ref-audio",
        str(ref_audio),
        "--ref-text",
        str(ref_text),
        "--takes",
        str(takes),
        "--model-size",
        model_size,
        "--language",
        language,
        "--chunk-size",
        str(chunk_size),
        "--chunk-gap",
        str(chunk_gap),
        "--seed-base",
        str(seed_base),
        "--device",
        device,
    ]
    if exact_seed is not None:
        command.extend(["--exact-seed", str(exact_seed)])
    optional_generation_args: list[tuple[str, object | None]] = [
        ("--temperature", temperature),
        ("--top-p", top_p),
        ("--top-k", top_k),
        ("--repetition-penalty", repetition_penalty),
        ("--subtalker-temperature", subtalker_temperature),
        ("--subtalker-top-p", subtalker_top_p),
        ("--subtalker-top-k", subtalker_top_k),
    ]
    for option, value in optional_generation_args:
        if value is not None:
            command.extend([option, str(value)])
    if phrases:
        command.append("--phrases")
        command.extend(phrases)
    if replace_phrase:
        command.append("--replace-phrase")
    if x_vector_only:
        command.append("--x-vector-only")
    if overwrite:
        command.append("--overwrite")
    if assemble_only:
        command.append("--assemble-only")
    if fit_to_slots:
        command.append("--fit-to-slots")
    if slot_margin:
        command.extend(["--slot-margin", str(slot_margin)])
    if speed_all_percent != 100.0:
        command.extend(["--speed-all-percent", str(speed_all_percent)])
    if avoid_overlap:
        command.append("--avoid-overlap")
    if min_gap != 0.10:
        command.extend(["--min-gap", str(min_gap)])
    if balance_phrases:
        command.append("--balance-phrases")
    if balance_max_gain_db != 3.0:
        command.extend(["--balance-max-gain-db", str(balance_max_gain_db)])
    if abs(summary_gain_db) > 0.0001:
        command.extend(["--summary-gain-db", str(summary_gain_db)])

    print(f"{episode_id}: Qwen3-TTS narrator summary generation")
    print(f"plan={plan_path}")
    print(f"output_dir={episode_out}")
    print(f"reference_audio={ref_audio}")
    print(f"reference_text={ref_text}")
    sampling_settings = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "subtalker_temperature": subtalker_temperature,
        "subtalker_top_p": subtalker_top_p,
        "subtalker_top_k": subtalker_top_k,
    }
    active_sampling = {key: value for key, value in sampling_settings.items() if value is not None}
    sampling_text = f" sampling={active_sampling}" if active_sampling else ""
    seed_text = f" exact_seed:{exact_seed}" if exact_seed is not None else f" seed_base:{seed_base}"
    print(
        f"settings=language:{language} model:{model_size} takes:{takes} "
        f"chunk_size:{chunk_size} chunk_gap:{chunk_gap}{seed_text}{sampling_text}"
    )
    print("NUMBA_CACHE_DIR=/tmp/robotech_numba_cache " + shlex.join(command))
    if not run:
        if assemble_only:
            print("Add --run to rebuild the selected-takes preview.")
        else:
            print("Add --run to generate the narrator takes.")
        return 0

    env = dict(os.environ)
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/robotech_numba_cache")
    started = time.monotonic()
    subprocess.run(command, check=True, env=env)
    print(f"{episode_id}: Qwen3-TTS summary generation complete in {format_elapsed(time.monotonic() - started)}")
    return 0


def validate_tts_phrase_selection(plan_path: Path, phrases: list[str]) -> str | None:
    """Return a friendly error if selected TTS phrase numbers are missing."""

    requested = parse_requested_phrase_numbers(phrases)
    if not requested:
        return None
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_phrases = plan_data.get("phrases", [])
    available = sorted(
        int(phrase["number"])
        for phrase in plan_phrases
        if isinstance(phrase, dict) and str(phrase.get("number", "")).isdigit()
    )
    missing = sorted(requested - set(available))
    if not missing:
        return None
    lines = [
        f"No matching phrase(s) in {plan_path}: {', '.join(f'{number:02d}' for number in missing)}",
        f"Available phrase numbers: {', '.join(f'{number:02d}' for number in available) or '(none)'}",
    ]
    for phrase in plan_phrases:
        if not isinstance(phrase, dict):
            continue
        try:
            number = int(phrase["number"])
        except (KeyError, TypeError, ValueError):
            continue
        lines.append(f"{number:02d}. {phrase.get('text', '')}")
    return "\n".join(lines)


def cmd_tts_summary_promote(
    episode_id: str,
    summary_id: str,
    preview: Path,
    patch_id: str | None,
    plan: Path | None,
    out_dir: Path,
    ready_patch_dir: Path,
    source_work_track: Path | None,
    replacement_gain_db: float,
    match_source_work_level: bool,
    match_max_gain_db: float,
    description: str | None,
    overwrite: bool,
    run: bool,
) -> int:
    episode_id = normalize_episode_id(episode_id)
    plan_path = plan or out_dir / episode_id / summary_id / "phrase_plan.json"
    if not plan_path.is_file():
        raise SystemExit(f"Missing phrase plan: {plan_path}")
    if not preview.is_file():
        raise SystemExit(f"Missing approved preview WAV: {preview}")

    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    start_seconds = float(plan_data["start"])
    replacement_duration = media_file_duration(preview)
    end_seconds = start_seconds + replacement_duration
    patch_id = patch_id or f"{episode_id.lower()}_next_episode_summary_tts_v001"
    patch_dir = ready_patch_dir / episode_id / patch_id
    replacement_path = patch_dir / "replacement.wav"
    manifest_path = patch_dir / "patch.json"
    selected_takes_path = out_dir / episode_id / summary_id / "selected_takes.json"
    if source_work_track is None:
        source_work_track = (
            DEFAULT_WORK
            / "final_episode_mux"
            / episode_id
            / "final_mux_oc_ec_v1"
            / f"{episode_id}_spa1_dialogue_broadcast_strong.wav"
        )
    applied_replacement_gain_db = float(replacement_gain_db)
    level_match: dict[str, object] = {"mode": "manual"}
    if match_source_work_level:
        if match_max_gain_db < 0:
            raise SystemExit(f"--match-max-gain-db cannot be negative, got {match_max_gain_db}")
        if not source_work_track.is_file():
            raise SystemExit(f"--match-source-work-level needs an existing --source-work-track: {source_work_track}")
        source_mean = ffmpeg_mean_volume_db(
            source_work_track,
            start_seconds=start_seconds,
            duration_seconds=replacement_duration,
        )
        preview_mean = ffmpeg_mean_volume_db(preview, start_seconds=0.0, duration_seconds=replacement_duration)
        if source_mean is None or preview_mean is None:
            raise SystemExit("Could not measure source/preview mean volume for --match-source-work-level")
        raw_gain = source_mean - preview_mean
        applied_replacement_gain_db = clamp(raw_gain, -match_max_gain_db, match_max_gain_db)
        level_match = {
            "mode": "auto_mean_volume",
            "source_work_track": str(source_work_track),
            "source_mean_volume_db": round(source_mean, 3),
            "preview_mean_volume_db": round(preview_mean, 3),
            "raw_gain_db": round(raw_gain, 3),
            "max_gain_db": match_max_gain_db,
            "applied_gain_db": round(applied_replacement_gain_db, 3),
        }
    patch = {
        "kind": "robotech_ready_audio_patch",
        "patch_id": patch_id,
        "episode": episode_id,
        "title": EPISODE_TITLES.get(episode_id, ""),
        "targets": ["dialogue"],
        "method": "ready_replacement_clip",
        "description": description
        or f"Qwen3-TTS regenerated next-episode narrator summary promoted from {preview.name}.",
        "start_seconds": round(start_seconds, 6),
        "end_seconds": round(end_seconds, 6),
        "replacement_path": "replacement.wav",
        "replacement_source_seconds": round(replacement_duration, 6),
        "replacement_gain_db": round(applied_replacement_gain_db, 3),
        "insert_fade_in_seconds": 0.0,
        "insert_fade_out_seconds": 0.0,
        "replacement_is_final": True,
        "source_preview": str(preview),
        "phrase_plan": str(plan_path),
        "selected_takes": str(selected_takes_path) if selected_takes_path.is_file() else "",
        "speech_map": str(plan_data.get("speech_map", "")),
        "source_work_track": str(source_work_track),
        "level_match": level_match,
        "assembly": {
            "mode": "promoted_preview",
            "summary_id": summary_id,
            "notes": "Patch replaces the full summary window with the already assembled TTS clip. Final build does not call Qwen3-TTS.",
        },
    }

    print(f"{episode_id}: promote TTS summary preview")
    print(f"preview={preview}")
    print(f"plan={plan_path}")
    print(f"patch_dir={patch_dir}")
    print(f"replacement_duration={replacement_duration:.3f}s")
    print(f"placement={seconds_to_clock(start_seconds)} -> {seconds_to_clock(end_seconds)}")
    print(f"replacement_gain_db={applied_replacement_gain_db:+.2f}")
    if match_source_work_level:
        print(f"level_match={level_match}")
    if patch_dir.exists() and any(patch_dir.iterdir()) and not overwrite:
        print(f"Ready patch already exists: {patch_dir}")
        print("Use --overwrite to replace replacement.wav and patch.json.")
        return 2
    if not run:
        print("Add --run to write replacement.wav and patch.json.")
        return 0

    patch_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(preview, replacement_path)
    manifest_path.write_text(json.dumps(patch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"replacement={replacement_path}")
    print(f"patch={manifest_path}")
    print("This ready patch will be picked up by episode-final-build unless ready patches are disabled.")
    return 0


def find_processed_old_spa1_stereo_track(work_dir: Path, episode_id: str, review_name: str = "final_mux_oc_ec_v1") -> Path | None:
    audio_dir = work_dir / "review" / "episodes" / episode_id / review_name / "audio"
    if not audio_dir.is_dir():
        return None
    matches = sorted(audio_dir.glob("03_*spanish1*stereo*.ac3")) + sorted(audio_dir.glob("03_*spanish1*stereo*.wav"))
    return matches[0] if matches else None


def ffmpeg_mean_volume_db(path: Path, *, start_seconds: float | None = None, duration_seconds: float | None = None) -> float | None:
    cmd = ["ffmpeg", "-hide_banner", "-nostats"]
    if start_seconds is not None:
        cmd.extend(["-ss", f"{max(start_seconds, 0.0):.6f}"])
    if duration_seconds is not None:
        cmd.extend(["-t", f"{max(duration_seconds, 0.001):.6f}"])
    cmd.extend(["-i", str(path), "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    output = result.stderr + "\n" + result.stdout
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", output)
    if not match:
        return None
    return float(match.group(1))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def cmd_spa2_tail_extension(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    reference_mkv: Path,
    reference_episode_start: float,
    reference_audio_stream: int,
    tail_start: float | None,
    duration: float,
    fade_in: float,
    fade_out: float,
    old_spa1_tail_gain_db: float | None,
    old_spa1_match_audio: Path | None,
    old_spa1_match_window: float,
    old_spa1_max_gain_db: float,
    extension_id: str | None,
    out_dir: Path,
    review_dir: Path,
    run: bool,
) -> int:
    episode_id = normalize_episode_id(episode_id)
    episode = find_episode(input_dir, episode_id)
    if episode.spa2 is None:
        raise SystemExit(f"{episode_id} has no local spa2 source track")
    if not reference_mkv.is_file():
        raise SystemExit(f"Reference MKV does not exist: {reference_mkv}")
    if duration <= 0:
        raise SystemExit(f"--duration must be positive, got {duration}")
    if fade_in < 0:
        raise SystemExit(f"--fade-in cannot be negative, got {fade_in}")
    if fade_out < 0:
        raise SystemExit(f"--fade-out cannot be negative, got {fade_out}")
    if old_spa1_match_window <= 0:
        raise SystemExit(f"--old-spa1-match-window must be positive, got {old_spa1_match_window}")
    if old_spa1_max_gain_db < 0:
        raise SystemExit(f"--old-spa1-max-gain-db cannot be negative, got {old_spa1_max_gain_db}")

    local_spa2_duration = media_file_duration(episode.spa2)
    source_start = float(tail_start) if tail_start is not None else reference_episode_start + local_spa2_duration
    extension_id = extension_id or f"{episode_id.lower()}_spa2_tail_v001"
    extension_dir = out_dir / episode_id / extension_id
    review_output_dir = review_dir / f"{episode_id}_spa2_tail_extension_001"
    tail_wav = extension_dir / "spa2_tail.wav"
    manifest_path = extension_dir / "tail.json"
    review_wav = review_output_dir / "candidate_spa2_tail.wav"
    fade_start = max(duration - fade_out, 0.0)
    audio_filters = []
    if fade_in > 0:
        audio_filters.append(f"afade=t=in:st=0:d={min(fade_in, duration):.6f}")
    if fade_out > 0:
        audio_filters.append(f"afade=t=out:st={fade_start:.6f}:d={fade_out:.6f}")
    audio_filter = ",".join(audio_filters) if audio_filters else "anull"
    extract_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        f"{source_start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(reference_mkv),
        "-map",
        f"0:{reference_audio_stream}",
        "-af",
        audio_filter,
        "-c:a",
        "pcm_s24le",
        str(tail_wav),
    ]
    review_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(tail_wav),
        "-c:a",
        "pcm_s24le",
        str(review_wav),
    ]
    old_spa1_tail_gain = float(old_spa1_tail_gain_db) if old_spa1_tail_gain_db is not None else 0.0
    old_spa1_level_match: dict[str, object] = {"mode": "manual" if old_spa1_tail_gain_db is not None else "none"}
    manifest = {
        "kind": "robotech_spa2_tail_extension",
        "episode_id": episode_id,
        "extension_id": extension_id,
        "audio_path": "spa2_tail.wav",
        "duration_seconds": duration,
        "source_path": str(reference_mkv),
        "source_audio_stream": reference_audio_stream,
        "source_start_seconds": round(source_start, 6),
        "reference_episode_start_seconds": round(reference_episode_start, 6),
        "local_spa2_duration_seconds": round(local_spa2_duration, 6),
        "fade_in_seconds": fade_in,
        "fade_out_seconds": fade_out,
        "old_spa1_tail_gain_db": round(old_spa1_tail_gain, 3),
        "old_spa1_level_match": old_spa1_level_match,
        "description": (
            "Extra Spanish redubbing tail inserted over black before restored end credits. "
            "English 5.1 and restored Spanish 5.1 remain silent in the hold segment; "
            "both stereo Spanish tracks receive the rescued tail audio."
        ),
    }
    print(f"{episode_id}: prepare Spanish redubbing tail extension")
    print(f"local_spa2_duration={local_spa2_duration:.3f}s")
    print(f"reference_tail_start={source_start:.3f}s duration={duration:.3f}s stream=0:{reference_audio_stream}")
    print(shlex.join(extract_cmd))
    if run:
        extension_dir.mkdir(parents=True, exist_ok=True)
        review_output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(extract_cmd, check=True)
        manifest["duration_seconds"] = round(media_file_duration(tail_wav), 6)
        if old_spa1_tail_gain_db is None:
            match_audio = old_spa1_match_audio or find_processed_old_spa1_stereo_track(work_dir, episode_id)
            if match_audio is not None and match_audio.is_file():
                destination_duration = media_file_duration(match_audio)
                match_duration = min(float(old_spa1_match_window), destination_duration, media_file_duration(tail_wav))
                destination_start = max(destination_duration - match_duration, 0.0)
                destination_mean = ffmpeg_mean_volume_db(
                    match_audio,
                    start_seconds=destination_start,
                    duration_seconds=match_duration,
                )
                tail_mean = ffmpeg_mean_volume_db(tail_wav, start_seconds=0.0, duration_seconds=match_duration)
                if destination_mean is not None and tail_mean is not None:
                    raw_gain = destination_mean - tail_mean
                    old_spa1_tail_gain = clamp(raw_gain, -old_spa1_max_gain_db, old_spa1_max_gain_db)
                    old_spa1_level_match = {
                        "mode": "auto_mean_volume",
                        "match_audio": str(match_audio),
                        "match_window_seconds": round(match_duration, 6),
                        "destination_mean_volume_db": round(destination_mean, 3),
                        "tail_mean_volume_db": round(tail_mean, 3),
                        "raw_gain_db": round(raw_gain, 3),
                        "max_gain_db": old_spa1_max_gain_db,
                        "applied_gain_db": round(old_spa1_tail_gain, 3),
                    }
                    print(
                        f"old spa1 tail auto gain: destination={destination_mean:.2f}dB "
                        f"tail={tail_mean:.2f}dB gain={old_spa1_tail_gain:+.2f}dB",
                        flush=True,
                    )
                else:
                    old_spa1_level_match = {
                        "mode": "auto_failed",
                        "match_audio": str(match_audio),
                        "reason": "volumedetect did not report mean_volume",
                    }
                    print(f"old spa1 tail auto gain failed for {match_audio}; using +0.00dB", flush=True)
            else:
                old_spa1_level_match = {
                    "mode": "auto_unavailable",
                    "reason": "No processed old Spanish stereo reference track found",
                }
                print("old spa1 tail auto gain unavailable; using +0.00dB", flush=True)
        manifest["old_spa1_tail_gain_db"] = round(old_spa1_tail_gain, 3)
        manifest["old_spa1_level_match"] = old_spa1_level_match
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        subprocess.run(review_cmd, check=True)
        print(f"tail={tail_wav}")
        print(f"manifest={manifest_path}")
        print(f"review={review_wav}")
    else:
        print(f"would write tail={tail_wav}")
        print(f"would write manifest={manifest_path}")
        print(f"would write review={review_wav}")
    return 0


def parse_requested_phrase_numbers(values: list[str]) -> set[int]:
    numbers: set[int] = set()
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                raise SystemExit(f"Invalid --phrases value {part!r}; use numbers like 03 or 01 04")
            numbers.add(int(part))
    return numbers


def cmd_subtitle_spellcheck(
    episode_id: str,
    source: str,
    subtitle_kind: str,
    subtitle_dir: Path,
    speech_map_dir: Path,
    hunspell_dic: Path,
    glossary: Path,
    allowlist: Path,
    out_dir: Path,
    min_length: int,
    run: bool,
) -> int:
    episode_ids = sorted(EPISODE_TITLES) if episode_id.lower() == "all" else [normalize_episode_id(episode_id)]
    dictionary = load_spanish_dictionary(hunspell_dic)
    allowed = load_spell_allowlist(allowlist, glossary)
    if not dictionary:
        raise SystemExit(f"Spanish dictionary is empty or missing: {hunspell_dic}")
    print(f"dictionary={hunspell_dic} words={len(dictionary)}")
    print(f"allowlist={allowlist} entries={len(allowed)}")
    all_reports: list[dict[str, object]] = []
    for current_episode in episode_ids:
        findings: list[dict[str, object]] = []
        if source in {"subtitles", "both"}:
            srt = subtitle_dir / current_episode / f"{current_episode}_{subtitle_kind}.srt"
            if srt.is_file():
                findings.extend(spellcheck_srt(srt, dictionary, allowed, min_length))
                findings.extend(subtitle_quality_findings(srt, current_episode, allowed))
            else:
                print(f"{current_episode}: missing subtitle file {srt}")
        if source in {"speech-map", "both"}:
            speech_map = speech_map_dir / current_episode / f"{current_episode}_dialogue_speech_map.json"
            if speech_map.is_file():
                findings.extend(spellcheck_speech_map(speech_map, dictionary, allowed, min_length))
            else:
                print(f"{current_episode}: missing speech map {speech_map}")
        grouped = group_spell_findings(findings)
        report = {
            "episode": current_episode,
            "source": source,
            "subtitle_kind": subtitle_kind,
            "findings_count": len(findings),
            "unique_suspicious_words": len(grouped),
            "words": grouped,
        }
        all_reports.append(report)
        print(f"{current_episode}: suspicious_words={len(grouped)} occurrences={len(findings)}")
        if run:
            episode_out = out_dir / current_episode
            episode_out.mkdir(parents=True, exist_ok=True)
            (episode_out / f"{current_episode}_spanish_spellcheck.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (episode_out / f"{current_episode}_spanish_spellcheck.md").write_text(
                spell_report_markdown(report),
                encoding="utf-8",
            )
    if run and episode_id.lower() == "all":
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "source": source,
            "subtitle_kind": subtitle_kind,
            "episodes": [
                {
                    "episode": item["episode"],
                    "unique_suspicious_words": item["unique_suspicious_words"],
                    "findings_count": item["findings_count"],
                }
                for item in all_reports
            ],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not run:
        print("Add --run to write JSON/Markdown spell-check reports.")
    return 0


def cmd_subtitle_review_workbook(
    episode_id: str,
    source: str,
    subtitle_kind: str,
    subtitle_dir: Path,
    speech_map_dir: Path,
    hunspell_dic: Path,
    glossary: Path,
    allowlist: Path,
    output: Path,
    min_length: int,
    run: bool,
) -> int:
    episode_ids = sorted(EPISODE_TITLES) if episode_id.lower() == "all" else [normalize_episode_id(episode_id)]
    dictionary = load_spanish_dictionary(hunspell_dic)
    allowed = load_spell_allowlist(allowlist, glossary)
    if not dictionary:
        raise SystemExit(f"Spanish dictionary is empty or missing: {hunspell_dic}")

    workbook: dict[str, list[list[object]]] = {}
    total_rows = 0
    for current_episode in episode_ids:
        findings: list[dict[str, object]] = []
        if source in {"subtitles", "both"}:
            srt = subtitle_dir / current_episode / f"{current_episode}_{subtitle_kind}.srt"
            if srt.is_file():
                findings.extend(spellcheck_srt(srt, dictionary, allowed, min_length))
            else:
                print(f"{current_episode}: missing subtitle file {srt}")
        if source in {"speech-map", "both"}:
            speech_map = speech_map_dir / current_episode / f"{current_episode}_dialogue_speech_map.json"
            if speech_map.is_file():
                findings.extend(spellcheck_speech_map(speech_map, dictionary, allowed, min_length))
            else:
                print(f"{current_episode}: missing speech map {speech_map}")
        rows = subtitle_review_rows(current_episode, findings)
        workbook[current_episode] = rows
        total_rows += max(0, len(rows) - 1)
        print(f"{current_episode}: review_rows={max(0, len(rows) - 1)}")

    if not run:
        print("Add --run to write the XLSX workbook.")
        print(f"Would write: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    write_simple_xlsx(output, workbook)
    print(f"workbook={output}")
    print(f"sheets={len(workbook)} rows={total_rows}")
    print("Edit action/replacement_word/replacement_context/notes. Leave episode/review_id/source/cue/time/path unchanged for later import.")
    return 0


def subtitle_review_rows(episode_id: str, findings: list[dict[str, object]]) -> list[list[object]]:
    headers = [
        "episode",
        "review_id",
        "action",
        "suspicious_word",
        "normalized",
        "replacement_word",
        "replacement_context",
        "source",
        "cue",
        "time",
        "context",
        "path",
        "notes",
    ]
    rows: list[list[object]] = [headers]
    sorted_findings = sorted(
        findings,
        key=lambda item: (
            str(item.get("source", "")),
            natural_sort_key(str(item.get("cue", ""))),
            str(item.get("time", "")),
            str(item.get("normalized", "")),
        ),
    )
    for index, finding in enumerate(sorted_findings, start=1):
        context = str(finding.get("context", ""))
        rows.append(
            [
                episode_id,
                f"{episode_id}-{index:04d}",
                "",
                str(finding.get("word", "")),
                str(finding.get("normalized", "")),
                "",
                context,
                str(finding.get("source", "")),
                str(finding.get("cue", "")),
                str(finding.get("time", "")),
                context,
                str(finding.get("path", "")),
                "",
            ]
        )
    return rows


def natural_sort_key(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (10**9, value)


def write_simple_xlsx(path: Path, sheets: dict[str, list[list[object]]]) -> None:
    """Write a minimal XLSX workbook using only the standard library."""

    safe_sheets = [(safe_excel_sheet_name(name), rows) for name, rows in sheets.items()]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
"""
            + "".join(
                f'  <Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>\n'
                for index, _sheet in enumerate(safe_sheets, start=1)
            )
            + "</Types>\n",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
"""
            + "".join(
                f'  <Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>\n'
                for index, _sheet in enumerate(safe_sheets, start=1)
            )
            + f'  <Relationship Id="rId{len(safe_sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>\n'
            + "</Relationships>\n",
        )
        archive.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
"""
            + "".join(
                f'    <sheet name="{xml_text(sheet_name)}" sheetId="{index}" r:id="rId{index}"/>\n'
                for index, (sheet_name, _rows) in enumerate(safe_sheets, start=1)
            )
            + """  </sheets>
</workbook>
""",
        )
        archive.writestr(
            "xl/styles.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2"><font/><font><b/></font></fonts>
  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>
""",
        )
        for index, (_sheet_name, rows) in enumerate(safe_sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet_xml(rows))


def worksheet_xml(rows: list[list[object]]) -> str:
    column_count = max((len(row) for row in rows), default=1)
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
        f"  <cols>{worksheet_column_widths(column_count)}</cols>",
        "  <sheetData>",
    ]
    for row_index, row in enumerate(rows, start=1):
        lines.append(f'    <row r="{row_index}">')
        for col_index, value in enumerate(row, start=1):
            cell_ref = f"{excel_column_name(col_index)}{row_index}"
            style = ' s="1"' if row_index == 1 else ""
            lines.append(f'      <c r="{cell_ref}" t="inlineStr"{style}><is><t>{xml_text(value)}</t></is></c>')
        lines.append("    </row>")
    lines.extend(
        [
            "  </sheetData>",
            f'  <autoFilter ref="A1:{excel_column_name(column_count)}{max(1, len(rows))}"/>',
            "</worksheet>",
            "",
        ]
    )
    return "\n".join(lines)


def worksheet_column_widths(column_count: int) -> str:
    widths = {
        1: 10,
        2: 14,
        3: 18,
        4: 18,
        5: 16,
        6: 20,
        7: 55,
        8: 12,
        9: 10,
        10: 27,
        11: 55,
        12: 45,
        13: 35,
    }
    return "".join(
        f'<col min="{index}" max="{index}" width="{widths.get(index, 15)}" customWidth="1"/>'
        for index in range(1, column_count + 1)
    )


def excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def safe_excel_sheet_name(value: str) -> str:
    cleaned = re.sub(r"[][\\/*?:]", "_", value.strip()) or "Sheet"
    return cleaned[:31]


def xml_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return xml_escape(text, {'"': "&quot;"})


def load_spanish_dictionary(path: Path) -> set[str]:
    words: set[str] = set()
    if not path.is_file():
        return words
    affix_rules = load_hunspell_affix_rules(path.with_suffix(".aff"))
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if line_number == 0 and line.strip().isdigit():
            continue
        raw = line.strip()
        if not raw:
            continue
        word, flags = parse_hunspell_dic_entry(raw)
        normalized = normalize_spell_word(word)
        if normalized:
            words.add(normalized)
        if flags:
            for expanded in expand_hunspell_word(word, flags, affix_rules):
                normalized_expanded = normalize_spell_word(expanded)
                if normalized_expanded:
                    words.add(normalized_expanded)
    return words


def parse_hunspell_dic_entry(raw: str) -> tuple[str, str]:
    first = re.split(r"[\t ]", raw, maxsplit=1)[0]
    if "/" not in first:
        return first, ""
    word, flags = first.split("/", 1)
    return word, flags


def load_hunspell_affix_rules(path: Path) -> dict[str, dict[str, list[tuple[str, str, str]]]]:
    rules: dict[str, dict[str, list[tuple[str, str, str]]]] = {"PFX": {}, "SFX": {}}
    if not path.is_file():
        return rules
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5 or parts[0] not in {"PFX", "SFX"}:
            continue
        kind, flag = parts[0], parts[1]
        if len(parts) == 4 and parts[3].isdigit():
            continue
        strip_value, add_value, condition = parts[2], parts[3], parts[4]
        if add_value == "0":
            add_value = ""
        if strip_value == "0":
            strip_value = ""
        rules.setdefault(kind, {}).setdefault(flag, []).append((strip_value, add_value, condition))
    return rules


def expand_hunspell_word(
    word: str,
    flags: str,
    affix_rules: dict[str, dict[str, list[tuple[str, str, str]]]],
) -> set[str]:
    expanded: set[str] = set()
    for flag in flags:
        for strip_value, add_value, condition in affix_rules.get("SFX", {}).get(flag, []):
            if strip_value and not word.endswith(strip_value):
                continue
            base = word[: -len(strip_value)] if strip_value else word
            candidate = base + add_value
            if hunspell_condition_matches(word, condition, suffix=True):
                expanded.add(candidate)
        for strip_value, add_value, condition in affix_rules.get("PFX", {}).get(flag, []):
            if strip_value and not word.startswith(strip_value):
                continue
            base = word[len(strip_value) :] if strip_value else word
            candidate = add_value + base
            if hunspell_condition_matches(word, condition, suffix=False):
                expanded.add(candidate)
    return expanded


def hunspell_condition_matches(word: str, condition: str, *, suffix: bool) -> bool:
    if condition == ".":
        return True
    pattern = condition
    if suffix:
        pattern = pattern + "$"
    else:
        pattern = "^" + pattern
    try:
        return re.search(pattern, word) is not None
    except re.error:
        return False


def load_spell_allowlist(allowlist: Path, glossary: Path) -> set[str]:
    words: set[str] = set()
    if allowlist.is_file():
        for line in allowlist.read_text(encoding="utf-8").splitlines():
            value = line.split("#", 1)[0].strip()
            if value:
                words.update(normalize_spell_word(part) for part in tokenize_spell_words(value))
    if glossary.is_file():
        payload = json.loads(glossary.read_text(encoding="utf-8"))
        for section in ("terms", "speaker_labels"):
            values = payload.get(section, {})
            if isinstance(values, dict):
                for key, value in values.items():
                    words.update(normalize_spell_word(part) for part in tokenize_spell_words(str(key)))
                    words.update(normalize_spell_word(part) for part in tokenize_spell_words(str(value)))
    return {word for word in words if word}


def spellcheck_srt(path: Path, dictionary: set[str], allowed: set[str], min_length: int) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    cue_index = ""
    cue_time = ""
    text_lines: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        stripped = line.strip()
        if not stripped:
            if text_lines:
                context = " ".join(text_lines)
                for word in tokenize_spell_words(context):
                    if is_suspicious_spanish_word(word, dictionary, allowed, min_length):
                        findings.append(
                            {
                                "word": word,
                                "normalized": normalize_spell_word(word),
                                "source": "subtitle",
                                "path": str(path),
                                "cue": cue_index,
                                "time": cue_time,
                                "context": context,
                            }
                        )
            cue_index = ""
            cue_time = ""
            text_lines = []
            continue
        if not cue_index and stripped.isdigit():
            cue_index = stripped
        elif "-->" in stripped:
            cue_time = stripped
        else:
            text_lines.append(stripped)
    return findings


def spellcheck_speech_map(path: Path, dictionary: set[str], allowed: set[str], min_length: int) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    findings: list[dict[str, object]] = []
    for segment in data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        time_label = f"{seconds_to_clock(float(segment.get('start', 0.0)))} --> {seconds_to_clock(float(segment.get('end', 0.0)))}"
        for word in tokenize_spell_words(text):
            if is_suspicious_spanish_word(word, dictionary, allowed, min_length):
                findings.append(
                    {
                        "word": word,
                        "normalized": normalize_spell_word(word),
                        "source": "speech-map",
                        "path": str(path),
                        "cue": segment.get("index", ""),
                        "time": time_label,
                        "context": text,
                    }
                )
    return findings


def subtitle_quality_findings(path: Path, episode_id: str, allowed: set[str]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    allowed_sdf = {"1"}
    if episode_id == "S01E36":
        allowed_sdf.update({"2", "3"})
    english_leftovers = {
        "about",
        "after",
        "against",
        "alien",
        "aliens",
        "and",
        "attack",
        "battle",
        "because",
        "before",
        "captain",
        "commander",
        "earth",
        "episode",
        "for",
        "from",
        "have",
        "next",
        "report",
        "space",
        "that",
        "their",
        "they",
        "this",
        "through",
        "under",
        "what",
        "will",
        "with",
        "you",
        "your",
    }
    cue_index = ""
    cue_time = ""
    text_lines: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        stripped = line.strip()
        if not stripped:
            if text_lines:
                context = " ".join(text_lines)
                for match in re.finditer(r"\bSDF[-–— ]*([A-Za-z0-9]+)\b", context, flags=re.IGNORECASE):
                    suffix = match.group(1).upper()
                    if suffix not in allowed_sdf:
                        findings.append(subtitle_quality_finding(path, cue_index, cue_time, context, match.group(0), "invalid_sdf_token"))
                if "\\" in context and not re.search(r"\{\\an\d+\}", context):
                    findings.append(subtitle_quality_finding(path, cue_index, cue_time, context, "\\", "literal_backslash"))
                for label in re.findall(r"\[([^\]]+)\]", context):
                    if suspicious_speaker_label(label, allowed):
                        findings.append(subtitle_quality_finding(path, cue_index, cue_time, context, f"[{label}]", "suspicious_speaker_label"))
                for word in tokenize_spell_words(context):
                    if word.lower() in english_leftovers and not word.isupper():
                        findings.append(subtitle_quality_finding(path, cue_index, cue_time, context, word, "possible_english_leftover"))
            cue_index = ""
            cue_time = ""
            text_lines = []
            continue
        if not cue_index and stripped.isdigit():
            cue_index = stripped
        elif "-->" in stripped:
            cue_time = stripped
        else:
            text_lines.append(stripped)
    return findings


def subtitle_quality_finding(path: Path, cue: str, time_label: str, context: str, word: str, source: str) -> dict[str, object]:
    return {
        "word": word,
        "normalized": normalize_spell_word(word) or word,
        "source": source,
        "path": str(path),
        "cue": cue,
        "time": time_label,
        "context": context,
    }


def suspicious_speaker_label(label: str, allowed: set[str]) -> bool:
    cleaned = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]", " ", label).strip()
    if not cleaned:
        return False
    parts = [normalize_spell_word(part) for part in cleaned.split() if part]
    if not parts:
        return False
    known_generic = {
        "alien",
        "aliens",
        "altavoz",
        "ambos",
        "anunciador",
        "aplausos",
        "capitan",
        "chicas",
        "comandante",
        "control",
        "director",
        "escuchan",
        "grupo",
        "hombre",
        "hombres",
        "locutor",
        "mujer",
        "narrador",
        "oficial",
        "radio",
        "risa",
        "risas",
        "risitas",
        "soldado",
        "soldados",
        "todos",
        "teniente",
        "voz",
        "zentraedi",
    }
    return not all(part in allowed or part in known_generic for part in parts)


def tokenize_spell_words(value: str) -> list[str]:
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:[-'][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)?", value)


def normalize_spell_word(value: str) -> str:
    value = value.strip().lower().replace("’", "'")
    value = "".join(
        char for char in unicodedata.normalize("NFD", value) if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-zñü'-]", "", value)


def is_suspicious_spanish_word(word: str, dictionary: set[str], allowed: set[str], min_length: int) -> bool:
    normalized = normalize_spell_word(word)
    if len(normalized) < min_length:
        return False
    if "-" in normalized:
        parts = [part for part in normalized.split("-") if part]
        if parts and all(
            part in dictionary or part in allowed or spanish_morphology_known(part, dictionary, allowed)
            for part in parts
        ):
            return False
    if normalized in dictionary or normalized in allowed or spanish_morphology_known(normalized, dictionary, allowed):
        return False
    if word.isupper() or normalized.isdigit():
        return False
    return True


def spanish_morphology_known(word: str, dictionary: set[str], allowed: set[str]) -> bool:
    """Cheap Spanish morphology fallback for raw Hunspell dictionaries.

    We parse the .dic file directly to avoid a runtime dependency. That means
    Hunspell flags are not expanded, so common forms like plurals and conjugated
    verbs need a conservative manual fallback.
    """

    candidates = set()
    if word.endswith("es") and len(word) > 4:
        candidates.add(word[:-2])
    if word.endswith("s") and len(word) > 3:
        candidates.add(word[:-1])
    if word.endswith("ces") and len(word) > 5:
        candidates.add(word[:-3] + "z")
    if word.endswith("a") and len(word) > 4:
        candidates.add(word[:-1] + "o")
    if word.endswith("as") and len(word) > 5:
        candidates.add(word[:-2] + "o")
        candidates.add(word[:-1])
    if word.endswith("os") and len(word) > 5:
        candidates.add(word[:-1])
    derivational_endings = [
        ("aciones", "ar"),
        ("acion", "ar"),
        ("iciones", "ir"),
        ("icion", "ir"),
        ("uciones", "uir"),
        ("ucion", "uir"),
        ("mente", ""),
    ]
    for ending, replacement in derivational_endings:
        if word.endswith(ending) and len(word) > len(ending) + 2:
            candidates.add(word[: -len(ending)] + replacement)
    enclitic_endings = ["melos", "mela", "melas", "selo", "sela", "selos", "selas", "nos", "los", "las", "les", "lo", "la", "le", "me", "te", "se"]
    for ending in enclitic_endings:
        if word.endswith(ending) and len(word) > len(ending) + 3:
            base = word[: -len(ending)]
            candidates.add(base)
            if base.endswith("yan") and len(base) > 4:
                candidates.add(base[:-3] + "ir")
                candidates.add(base[:-3] + "uir")
            if base.endswith("en") and len(base) > 4:
                candidates.add(base[:-2] + "ar")
                candidates.add(base[:-2] + "er")
                candidates.add(base[:-2] + "ir")
    verb_endings = [
        ("ando", "ar"),
        ("iendo", "er"),
        ("iendo", "ir"),
        ("ado", "ar"),
        ("ido", "er"),
        ("ido", "ir"),
        ("aron", "ar"),
        ("ieron", "er"),
        ("ieron", "ir"),
        ("aba", "ar"),
        ("aban", "ar"),
        ("are", "ar"),
        ("ere", "er"),
        ("ire", "ir"),
        ("o", "ar"),
        ("o", "er"),
        ("o", "ir"),
        ("as", "ar"),
        ("es", "er"),
        ("es", "ir"),
        ("a", "ar"),
        ("e", "er"),
        ("e", "ir"),
        ("amos", "ar"),
        ("emos", "er"),
        ("imos", "ir"),
        ("an", "ar"),
        ("en", "er"),
        ("en", "ir"),
    ]
    for ending, infinitive in verb_endings:
        if word.endswith(ending) and len(word) > len(ending) + 2:
            candidates.add(word[: -len(ending)] + infinitive)
    return any(candidate in dictionary or candidate in allowed for candidate in candidates)


def group_spell_findings(findings: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for finding in findings:
        key = str(finding["normalized"])
        item = grouped.setdefault(
            key,
            {
                "word": finding["word"],
                "normalized": key,
                "count": 0,
                "examples": [],
            },
        )
        item["count"] = int(item["count"]) + 1
        examples = item["examples"]
        if isinstance(examples, list) and len(examples) < 5:
            examples.append(
                {
                    "source": finding["source"],
                    "time": finding["time"],
                    "cue": finding["cue"],
                    "context": finding["context"],
                    "path": finding["path"],
                }
            )
    return sorted(grouped.values(), key=lambda item: (-int(item["count"]), str(item["normalized"])))


def spell_report_markdown(report: dict[str, object]) -> str:
    lines = [
        f"# {report['episode']} Spanish Spell Check",
        "",
        f"- source: `{report['source']}`",
        f"- subtitle kind: `{report['subtitle_kind']}`",
        f"- suspicious unique words: `{report['unique_suspicious_words']}`",
        f"- occurrences: `{report['findings_count']}`",
        "",
    ]
    for item in report.get("words", []):
        if not isinstance(item, dict):
            continue
        lines.append(f"## {item['word']} ({item['count']})")
        for example in item.get("examples", []):
            if not isinstance(example, dict):
                continue
            lines.append(f"- `{example.get('time', '')}` {example.get('source', '')}: {example.get('context', '')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def find_phrase_candidates(
    words: list[dict[str, object]],
    phrase: str,
    around_seconds: float | None,
    window: float,
    limit: int,
) -> list[dict[str, object]]:
    query_tokens = tokenize_speech_search(phrase)
    if not query_tokens:
        return []
    usable_words = []
    for word in words:
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if around_seconds is not None and (end < around_seconds - window or start > around_seconds + window):
            continue
        text = str(word.get("word", "")).strip()
        tokens = tokenize_speech_search(text)
        if not tokens:
            continue
        usable_words.append({**word, "start": start, "end": end, "search_token": " ".join(tokens), "word": text})

    candidates: list[dict[str, object]] = []
    query_norm = " ".join(query_tokens)
    query_compact = "".join(query_tokens)
    base_len = len(query_tokens)
    for start_index in range(len(usable_words)):
        for span_len in range(max(1, base_len - 2), base_len + 4):
            end_index = start_index + span_len
            if end_index > len(usable_words):
                continue
            span_words = usable_words[start_index:end_index]
            span_norm = " ".join(str(item["search_token"]) for item in span_words)
            span_compact = span_norm.replace(" ", "")
            score = max(
                SequenceMatcher(None, query_norm, span_norm).ratio(),
                SequenceMatcher(None, query_compact, span_compact).ratio(),
            )
            if query_compact == span_compact:
                score = max(score, 0.98)
            if query_norm in span_norm or span_norm in query_norm:
                score = max(score, 0.92)
            if query_compact and (query_compact in span_compact or span_compact in query_compact):
                score = max(score, 0.94)
            if around_seconds is not None:
                midpoint = (float(span_words[0]["start"]) + float(span_words[-1]["end"])) / 2.0
                distance_penalty = min(abs(midpoint - around_seconds) / max(window, 1.0), 1.0) * 0.08
                score -= distance_penalty
            if score < 0.58:
                continue
            candidates.append(
                {
                    "score": score,
                    "start": float(span_words[0]["start"]),
                    "end": float(span_words[-1]["end"]),
                    "first_word_index": int(span_words[0].get("index", start_index + 1)),
                    "last_word_index": int(span_words[-1].get("index", end_index)),
                    "text": " ".join(str(item["word"]) for item in span_words),
                }
            )
    candidates.sort(key=lambda item: (-float(item["score"]), float(item["start"])))
    deduped = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        key = (int(candidate["first_word_index"]), int(candidate["last_word_index"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= limit:
            break
    return deduped


def tokenize_speech_search(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFD", value.lower())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    normalized = re.sub(r"[^a-z0-9ñ]+", " ", normalized)
    return [token for token in normalized.split() if token]


def parse_flexible_seconds(value: str | None) -> float:
    if value is None:
        return 0.0
    stripped = value.strip().replace(",", ".")
    if not stripped:
        return 0.0
    parts = stripped.split(":")
    if len(parts) == 1:
        return float(parts[0])
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_clock(value: float) -> str:
    milliseconds_total = max(0, int(round(value * 1000)))
    seconds_total, milliseconds = divmod(milliseconds_total, 1000)
    minutes_total, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def cmd_episode_final_build(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    review_name: str,
    proc_video_root: Path,
    opening_root: Path,
    opening_generation: str,
    include_opening: bool,
    end_credit_root: Path,
    include_end_credit: bool,
    spa_model: str,
    center_model: str,
    clean_variant_name: str,
    enhance_variant_name: str,
    source_mode: str,
    sample_rate: int,
    dialogue_gain_db: float,
    center_bed_gain_db: float,
    preserved_channel_gain_db: float,
    spa1_fullmix_variant_name: str,
    silence_start_seconds: float,
    restored_ac3_bitrate: str,
    eng_ac3_bitrate: str,
    stereo_ac3_bitrate: str,
    apply_video_episode_adjustments: bool,
    apply_audio_episode_adjustments: bool,
    auto_end_fade: bool,
    end_fade_frames: int,
    end_fade_black_avg: float,
    end_fade_black_max: int,
    rebuild_intermediates: bool,
    repair_recipes: list[Path],
    ready_patch_dir: Path | None,
    spa2_tail_extension_dir: Path | None,
    reference_chapter_file: Path | None,
    subtitle_dir: Path,
    embed_subtitles: bool,
    attach_cover_art: bool,
    cover_asset: Path,
    if_exists: str,
    copy_sources: bool,
    keep_intermediate_segments: bool,
    out_dir: Path,
    run: bool,
) -> int:
    started_at = time.monotonic()
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode, require_video=False)
    if episode.spa2 is None:
        raise SystemExit(f"{episode.episode_id} missing spa2, required for final mux")
    assert episode.eng_51 is not None and episode.spa1 is not None

    clean_variant = require_clean_variant(clean_variant_name)
    enhance_variant = require_enhance_variant(enhance_variant_name)
    spa1_fullmix_variant = require_full_mix_variant(spa1_fullmix_variant_name)
    selected_videos = select_proc_videos(proc_video_root, episode.episode_id)
    summary_chapter_start = configured_summary_start_seconds(DEFAULT_SUMMARY_CONFIG, episode.episode_id)

    build_dir = out_dir / episode.episode_id / review_name
    review_root = work_dir / "review" / "episodes"
    review_dir = review_root / episode.episode_id / review_name
    audio_dir = review_dir / "audio"
    video_dir = review_dir / "video"
    episode_only_dir = video_dir / "episode_only"
    segment_dir = review_dir / "segments"
    adjustment_dir = segment_dir / "adjustments"
    cover_dir = review_dir / "cover_art"
    source_video_dir = review_dir / "sources" / "video"
    source_audio_dir = review_dir / "sources" / "audio"
    opening_dir = review_root / "_shared_openings" / review_name / f"generation_{opening_generation}"
    credit_dir = review_root / "_shared_end_credits" / review_name
    spa2_tail_extension = load_spa2_tail_extension(spa2_tail_extension_dir, episode.episode_id)
    prepared_episode_segments = load_prepared_episode_segments(DEFAULT_READY_EPISODE_SEGMENT_DIR, episode.episode_id)
    suppressed_ready_patch_ids = tuple(
        patch_id
        for segment in prepared_episode_segments
        for patch_id in segment.suppresses_ready_audio_patch_ids
    )
    reference_episode_chapters = load_reference_episode_chapters(reference_chapter_file, episode.episode_id)
    if run:
        build_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        episode_only_dir.mkdir(parents=True, exist_ok=True)
        segment_dir.mkdir(parents=True, exist_ok=True)
        adjustment_dir.mkdir(parents=True, exist_ok=True)
        if attach_cover_art:
            cover_dir.mkdir(parents=True, exist_ok=True)
        if copy_sources:
            source_video_dir.mkdir(parents=True, exist_ok=True)
            source_audio_dir.mkdir(parents=True, exist_ok=True)
        if include_opening:
            opening_dir.mkdir(parents=True, exist_ok=True)
        if include_end_credit:
            credit_dir.mkdir(parents=True, exist_ok=True)

    planned_outputs = {
        variant: video_dir / final_video_name(episode.episode_id, variant, video_path)
        for variant, video_path in selected_videos.items()
    }
    if run and any(path.exists() for path in planned_outputs.values()):
        decision = decide_existing_outputs(episode.episode_id, planned_outputs, if_exists)
        if decision == "skip":
            print(f"Skipping {episode.episode_id}: final output already exists")
            return 0

    clean_spa1 = build_dir / f"{episode.episode_id}_spa1_{source_mode}_{clean_variant.name}.wav"
    spa_sep_dir = build_dir / "spa1_dialogue_separator"
    spanish_dialogue = build_dir / f"{episode.episode_id}_spa1_dialogue_{enhance_variant.name}.wav"
    dialogue_patches = patches_for_target(
        episode.episode_id,
        "dialogue",
        apply_audio_episode_adjustments,
        ready_patch_dir,
        suppressed_ready_patch_ids,
    )
    patched_spanish_dialogue = (
        build_dir / f"{episode.episode_id}_spa1_dialogue_{enhance_variant.name}_{EPISODE_AUDIO_PATCH_VERSION}.wav"
    )
    recipe_spanish_dialogue = (
        build_dir / f"{episode.episode_id}_spa1_dialogue_{enhance_variant.name}_repair_recipes_v001.wav"
    )
    spanish_dialogue_before_recipes = patched_spanish_dialogue if dialogue_patches else spanish_dialogue
    spanish_dialogue_for_mix = recipe_spanish_dialogue if repair_recipes else spanish_dialogue_before_recipes
    eng_center = build_dir / f"{episode.episode_id}_eng_center.wav"
    center_sep_dir = build_dir / "eng_center_devoice"
    devoiced_bed_51_wav = build_dir / f"{episode.episode_id}_eng_devoiced_bed_51_lossless.wav"
    restored_wav = build_dir / f"{episode.episode_id}_restored_spa_51_lossless_silenced.wav"
    spa1_fullmix_wav = build_dir / f"{episode.episode_id}_spa1_stereo_{spa1_fullmix_variant.name}_silenced.wav"
    spa1_fullmix_patches = patches_for_target(
        episode.episode_id,
        "spa1_fullmix",
        apply_audio_episode_adjustments,
        ready_patch_dir,
        suppressed_ready_patch_ids,
    )
    dialogue_ready_patches_active = any(
        patch.method == "ready_replacement_clip" for patch in dialogue_patches
    )
    spa1_ready_patches_active = any(
        patch.method == "ready_replacement_clip" for patch in spa1_fullmix_patches
    )
    patched_spa1_fullmix_wav = (
        build_dir
        / f"{episode.episode_id}_spa1_stereo_{spa1_fullmix_variant.name}_silenced_{EPISODE_AUDIO_PATCH_VERSION}.wav"
    )
    spa1_fullmix_for_ac3 = patched_spa1_fullmix_wav if spa1_fullmix_patches else spa1_fullmix_wav

    eng_ac3 = audio_dir / f"01_{episode.episode_id}_english_original_51.ac3"
    restored_ac3 = audio_dir / f"02_{episode.episode_id}_spanish_restored_51.ac3"
    spa1_ac3 = audio_dir / f"03_{episode.episode_id}_spanish1_restored_old_stereo_{spa1_fullmix_variant.name}.ac3"
    spa2_ac3 = audio_dir / f"04_{episode.episode_id}_spanish2_original_stereo.ac3"

    silence_filter = silence_start_filter(silence_start_seconds)
    clean_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.spa1),
        "-map",
        "0:a:0",
        "-af",
        source_mode_filter(source_mode, clean_variant.filtergraph),
        "-c:a",
        "pcm_s24le",
        str(clean_spa1),
    ]
    spa_sep_cmd = build_separation_command(
        "audio-separator",
        clean_spa1,
        spa_sep_dir,
        model=spa_model,
        single_stem="Vocals",
        sample_rate=sample_rate,
        model_file_dir=work_dir / "models" / "audio-separator",
    )
    center_extract_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.eng_51),
        "-map",
        "0:a:0",
        "-af",
        "pan=mono|c0=c2",
        "-c:a",
        "pcm_s24le",
        str(eng_center),
    ]
    center_sep_cmd = build_separation_command(
        "audio-separator",
        eng_center,
        center_sep_dir,
        model=center_model,
        single_stem="Instrumental",
        sample_rate=sample_rate,
        model_file_dir=work_dir / "models" / "audio-separator",
    )

    cache_events: list[dict[str, object]] = []
    first_phase = [clean_cmd, spa_sep_cmd, center_extract_cmd, center_sep_cmd]
    run_cached_command(
        clean_cmd,
        clean_spa1,
        "cleaned old Spanish spa1 WAV",
        rebuild=rebuild_intermediates,
        run=run,
        cache_events=cache_events,
    )
    cached_spa_vocals = cached_stem_output(spa_sep_dir, "Vocals")
    if cached_spa_vocals is not None and not rebuild_intermediates:
        print(f"reuse Spanish dialogue separator Vocals stem: {cached_spa_vocals}", flush=True)
        cache_events.append(
            {
                "label": "Spanish dialogue separator Vocals stem",
                "output": str(cached_spa_vocals),
                "action": "reused",
            }
        )
    else:
        print(shlex.join(spa_sep_cmd), flush=True)
        cache_events.append(
            {
                "label": "Spanish dialogue separator Vocals stem",
                "output": str(spa_sep_dir),
                "action": "rebuilt" if cached_spa_vocals is not None and rebuild_intermediates else "created",
                "command": spa_sep_cmd,
            }
        )
        if run:
            subprocess.run(spa_sep_cmd, check=True)
    run_cached_command(
        center_extract_cmd,
        eng_center,
        "English center mono WAV",
        rebuild=rebuild_intermediates,
        run=run,
        cache_events=cache_events,
    )
    cached_center_bed = cached_stem_output(center_sep_dir, "Instrumental")
    if cached_center_bed is not None and not rebuild_intermediates:
        print(f"reuse English center Instrumental stem: {cached_center_bed}", flush=True)
        cache_events.append(
            {
                "label": "English center Instrumental stem",
                "output": str(cached_center_bed),
                "action": "reused",
            }
        )
    else:
        print(shlex.join(center_sep_cmd), flush=True)
        cache_events.append(
            {
                "label": "English center Instrumental stem",
                "output": str(center_sep_dir),
                "action": "rebuilt" if cached_center_bed is not None and rebuild_intermediates else "created",
                "command": center_sep_cmd,
            }
        )
        if run:
            subprocess.run(center_sep_cmd, check=True)

    if not run:
        print(f"selected videos for {episode.episode_id}:")
        for variant, path in selected_videos.items():
            print(f"  {variant}: {path}")
            if include_opening:
                opening_assets = resolve_opening_assets(opening_root, opening_generation, work_dir)
                opening_output = shared_opening_path(opening_dir, opening_assets.video, path)
                target_info = video_encode_info(path)
                opening_info = video_encode_info(opening_assets.video)
                print(
                    "    opening: "
                    f"{opening_assets.video} -> {target_info.width}x{target_info.height} "
                    f"{target_info.rate_text} {target_info.profile or 'profile-auto'}"
                    f" (source {opening_info.rate_text}, generation {opening_generation})"
                )
                print(f"    prepared once at: {opening_output}")
            if include_end_credit:
                credit = select_end_credit(end_credit_root, path)
                target_info = video_encode_info(path)
                credit_info = video_encode_info(credit)
                shared_credit = shared_end_credit_path(credit_dir, credit, path)
                print(
                    "    end credit: "
                    f"{credit} -> {target_info.width}x{target_info.height} "
                    f"{target_info.rate_text} {target_info.profile or 'profile-auto'}"
                    f" (source {credit_info.rate_text})"
                )
                print(f"    prepared once at: {shared_credit}")
            for prepared_segment in prepared_episode_segments:
                prepared_output = prepared_segment.outputs.get(variant)
                if prepared_output is None:
                    continue
                print(
                    "    prepared segment: "
                    f"{prepared_segment.segment_id} -> {prepared_output} "
                    f"({prepared_segment.insert})"
                )
        if suppressed_ready_patch_ids:
            print(f"    suppressed ready audio patch ids: {', '.join(suppressed_ready_patch_ids)}")
        if include_end_credit:
            print("after separation: enhance dialogue, rebuild silenced audio tracks, mux episode segments, apply needed adjustment segments, prepare matching opening/end-credit segments, concat with video copy")
        elif include_opening:
            print("after separation: enhance dialogue, rebuild silenced audio tracks, mux episode segments, prepare matching opening segments, concat with video copy")
        else:
            print("after separation: enhance dialogue, rebuild silenced audio tracks, mux three final video variants")
        if copy_sources:
            print(f"source copies after --run: {review_dir / 'sources'}")
        print(f"review will be created after --run: {review_dir}")
        return 0

    source_copies: dict[str, dict[str, str]] = {"video": {}, "audio": {}}
    if copy_sources:
        for variant, source_video in selected_videos.items():
            copied = copy_source_file(source_video, source_video_dir)
            source_copies["video"][variant] = str(copied)
        source_audio_inputs = {
            "english_original_51_source": episode.eng_51,
            "spanish1_old_source": episode.spa1,
            "spanish2_newer_source": episode.spa2,
        }
        for label, source_audio in source_audio_inputs.items():
            if source_audio is None:
                continue
            copied = copy_source_file(source_audio, source_audio_dir)
            source_copies["audio"][label] = str(copied)

    spa_vocals = find_single_stem_output(spa_sep_dir, "Vocals")
    center_bed = find_single_stem_output(center_sep_dir, "Instrumental")
    devoiced_bed_51_filter = (
        "[0:a]channelsplit=channel_layout=5.1(side)[FL][FR][FC_ORIG][LFE][SL][SR];"
        "[FC_ORIG]anullsink;"
        "[1:a]pan=mono|c0=0.5*c0+0.5*c1[FCNEW];"
        "[FL][FR][FCNEW][LFE][SL][SR]"
        "join=inputs=6:channel_layout=5.1(side):"
        "map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-SL|5.0-SR[out51]"
    )
    devoiced_bed_51_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.eng_51),
        "-i",
        str(center_bed),
        "-filter_complex",
        devoiced_bed_51_filter,
        "-map",
        "[out51]",
        "-c:a",
        "pcm_s24le",
        str(devoiced_bed_51_wav),
    ]
    enhance_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(spa_vocals),
        "-af",
        enhance_variant.filtergraph,
        "-c:a",
        "pcm_s24le",
        str(spanish_dialogue),
    ]
    mix_filter = (
        "[0:a]channelsplit=channel_layout=5.1(side)[FL][FR][FC_ORIG][LFE][SL][SR];"
        "[FC_ORIG]anullsink;"
        f"[FL]volume={preserved_channel_gain_db}dB[FLP];"
        f"[FR]volume={preserved_channel_gain_db}dB[FRP];"
        f"[LFE]volume={preserved_channel_gain_db}dB[LFEP];"
        f"[SL]volume={preserved_channel_gain_db}dB[SLP];"
        f"[SR]volume={preserved_channel_gain_db}dB[SRP];"
        "[1:a]pan=mono|c0=0.5*c0+0.5*c1,"
        f"volume={center_bed_gain_db}dB[bed];"
        "[2:a]pan=mono|c0=0.5*c0+0.5*c1,"
        f"volume={dialogue_gain_db}dB[dialogue];"
        "[bed][dialogue]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[FCNEW];"
        "[FLP][FRP][FCNEW][LFEP][SLP][SRP]"
        "join=inputs=6:channel_layout=5.1(side):"
        "map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-SL|5.0-SR[raw51];"
        f"[raw51]{silence_filter}[out51]"
    )
    mix_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.eng_51),
        "-i",
        str(center_bed),
        "-i",
        str(spanish_dialogue_for_mix),
        "-filter_complex",
        mix_filter,
        "-map",
        "[out51]",
        "-c:a",
        "pcm_s24le",
        str(restored_wav),
    ]
    restored_ac3_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(restored_wav),
        "-c:a",
        "ac3",
        "-b:a",
        restored_ac3_bitrate,
        str(restored_ac3),
    ]
    eng_ac3_cmd = silence_audio_cmd(episode.eng_51, eng_ac3, eng_ac3_bitrate, silence_start_seconds)
    spa1_fullmix_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(episode.spa1),
        "-map",
        "0:a:0",
        "-af",
        f"{spa1_fullmix_variant.filtergraph},{silence_filter}",
        "-c:a",
        "pcm_s24le",
        str(spa1_fullmix_wav),
    ]
    spa1_ac3_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(spa1_fullmix_for_ac3),
        "-c:a",
        "ac3",
        "-b:a",
        stereo_ac3_bitrate,
        str(spa1_ac3),
    ]
    spa2_ac3_cmd = silence_audio_cmd(episode.spa2, spa2_ac3, stereo_ac3_bitrate, silence_start_seconds)

    dialogue_patch_steps = patch_audio_steps(
        spanish_dialogue,
        patched_spanish_dialogue,
        dialogue_patches,
        build_dir,
        f"{episode.episode_id}_dialogue_patch",
    )
    spa1_fullmix_patch_steps = patch_audio_steps(
        spa1_fullmix_wav,
        patched_spa1_fullmix_wav,
        spa1_fullmix_patches,
        build_dir,
        f"{episode.episode_id}_spa1_fullmix_patch",
    )
    repair_recipe_events: list[dict[str, object]] = []
    audio_cmds = [
        enhance_cmd,
        *[step[0] for step in dialogue_patch_steps],
        devoiced_bed_51_cmd,
        mix_cmd,
        restored_ac3_cmd,
        eng_ac3_cmd,
        spa1_fullmix_cmd,
        *[step[0] for step in spa1_fullmix_patch_steps],
        spa1_ac3_cmd,
        spa2_ac3_cmd,
    ]
    mix_levels_changed = any(
        abs(value) > 1e-9 for value in (preserved_channel_gain_db, center_bed_gain_db, dialogue_gain_db)
    )
    rebuild_mix_outputs = rebuild_intermediates or mix_levels_changed or bool(dialogue_patches) or bool(repair_recipes)
    run_cached_command(
        enhance_cmd,
        spanish_dialogue,
        "enhanced Spanish dialogue WAV",
        rebuild=rebuild_intermediates,
        run=True,
        cache_events=cache_events,
    )
    if dialogue_patches:
        for patch_index, (patch_cmd, patch_output, patch_label) in enumerate(dialogue_patch_steps, start=1):
            run_cached_command(
                patch_cmd,
                patch_output,
                f"episode-patched Spanish dialogue WAV step {patch_index}: {patch_label}",
                rebuild=rebuild_intermediates or dialogue_ready_patches_active,
                run=True,
                cache_events=cache_events,
            )
    if repair_recipes:
        from robotech_ai_repair.recipe_apply import apply_recipe_to_audio

        recipe_input = spanish_dialogue_before_recipes
        recipe_outputs: list[dict[str, object]] = []
        if recipe_spanish_dialogue.exists() and not rebuild_intermediates:
            print(f"reuse repair-recipe patched Spanish dialogue WAV: {recipe_spanish_dialogue}", flush=True)
            repair_recipe_events.append(
                {
                    "label": "repair-tool recipes",
                    "output": str(recipe_spanish_dialogue),
                    "action": "reused",
                    "recipes": [str(path) for path in repair_recipes],
                }
            )
        else:
            temp_input = recipe_input
            for index, recipe in enumerate(repair_recipes, start=1):
                recipe_output = (
                    recipe_spanish_dialogue
                    if index == len(repair_recipes)
                    else build_dir / f"{episode.episode_id}_spa1_dialogue_recipe_{index:02d}.wav"
                )
                print(f"apply repair recipe {index}/{len(repair_recipes)}: {recipe} -> {recipe_output}", flush=True)
                event = apply_recipe_to_audio(recipe, temp_input, recipe_output)
                recipe_outputs.append(event)
                temp_input = recipe_output
            repair_recipe_events.append(
                {
                    "label": "repair-tool recipes",
                    "output": str(recipe_spanish_dialogue),
                    "action": "created",
                    "recipes": [str(path) for path in repair_recipes],
                    "events": recipe_outputs,
                }
            )
    run_cached_command(
        devoiced_bed_51_cmd,
        devoiced_bed_51_wav,
        "full de-voiced English 5.1 bed WAV",
        rebuild=rebuild_intermediates,
        run=True,
        cache_events=cache_events,
    )
    run_cached_command(
        mix_cmd,
        restored_wav,
        "restored Spanish 5.1 lossless WAV",
        rebuild=rebuild_mix_outputs,
        run=True,
        cache_events=cache_events,
    )
    run_cached_command(
        restored_ac3_cmd,
        restored_ac3,
        "restored Spanish 5.1 AC3",
        rebuild=rebuild_mix_outputs,
        run=True,
        cache_events=cache_events,
    )
    run_cached_command(
        eng_ac3_cmd,
        eng_ac3,
        "silenced English 5.1 AC3",
        rebuild=rebuild_intermediates,
        run=True,
        cache_events=cache_events,
    )
    run_cached_command(
        spa1_fullmix_cmd,
        spa1_fullmix_wav,
        "restored old Spanish stereo lossless WAV",
        rebuild=rebuild_intermediates,
        run=True,
        cache_events=cache_events,
    )
    if spa1_fullmix_patches:
        for patch_index, (patch_cmd, patch_output, patch_label) in enumerate(spa1_fullmix_patch_steps, start=1):
            run_cached_command(
                patch_cmd,
                patch_output,
                f"episode-patched restored old Spanish stereo lossless WAV step {patch_index}: {patch_label}",
                rebuild=rebuild_intermediates or spa1_ready_patches_active,
                run=True,
                cache_events=cache_events,
            )
    run_cached_command(
        spa1_ac3_cmd,
        spa1_ac3,
        "restored old Spanish stereo AC3",
        rebuild=rebuild_intermediates or bool(spa1_fullmix_patches) or apply_audio_episode_adjustments,
        run=True,
        cache_events=cache_events,
    )
    run_cached_command(
        spa2_ac3_cmd,
        spa2_ac3,
        "silenced Spanish 2 newer dub AC3",
        rebuild=rebuild_intermediates,
        run=True,
        cache_events=cache_events,
    )

    mux_outputs: dict[str, str] = {}
    mux_cmds: list[list[str]] = []
    authoring_cmds: list[list[str]] = []
    cover_cmds: list[list[str]] = []
    authored_subtitles = discover_srt_subtitles(subtitle_dir, episode.episode_id) if embed_subtitles else []
    cover_outputs: dict[str, str] = {}
    opening_outputs: dict[str, str] = {}
    opening_sources: dict[str, str] = {}
    opening_cmds: list[list[str]] = []
    reused_openings: dict[str, bool] = {}
    credit_outputs: dict[str, str] = {}
    credit_sources: dict[str, str] = {}
    credit_cmds: list[list[str]] = []
    concat_cmds: list[list[str]] = []
    chapter_files: dict[str, str] = {}
    chapter_plans: dict[str, list[dict[str, object]]] = {}
    adjustment_cmds: list[list[str]] = []
    end_fade_adjustments: dict[str, dict[str, object]] = {}
    spa2_tail_extension_adjustments: dict[str, dict[str, object]] = {}
    tail_trim_adjustments: dict[str, dict[str, object]] = {}
    prepared_segment_outputs: dict[str, list[dict[str, str]]] = {}
    episode_segments: dict[str, str] = {}
    concat_lists: dict[str, str] = {}
    reused_credits: dict[str, bool] = {}
    opening_assets = resolve_opening_assets(opening_root, opening_generation, work_dir) if include_opening else None
    for variant, video_path in selected_videos.items():
        output = video_dir / final_video_name(episode.episode_id, variant, video_path)
        segment_output = output
        if include_opening or include_end_credit:
            segment_output = episode_only_dir / f"{output.stem}_episode_only.mkv"
        episode_segments[variant] = str(segment_output)
        mux_cmd = final_mux_cmd(
            episode.episode_id,
            variant,
            video_path,
            segment_output,
            restored_ac3,
            eng_ac3,
            spa1_ac3,
            spa2_ac3,
        )
        print(shlex.join(mux_cmd), flush=True)
        subprocess.run(mux_cmd, check=True)
        mux_cmds.append(mux_cmd)
        concat_segments = []
        if include_opening and opening_assets is not None:
            opening_output = shared_opening_path(opening_dir, opening_assets.video, video_path)
            opening_cmd = opening_segment_cmd(
                opening_assets,
                video_path,
                opening_output,
                restored_ac3_bitrate=restored_ac3_bitrate,
                eng_ac3_bitrate=eng_ac3_bitrate,
                stereo_ac3_bitrate=stereo_ac3_bitrate,
            )
            if not opening_output.exists():
                if not opening_assets.generation_spanish_51.is_file():
                    raise SystemExit(
                        "Missing opening generation narration asset needed to create a new shared opening:\n"
                        f"{opening_assets.generation_spanish_51}\n"
                        f"Existing shared openings can still be reused from: {opening_dir}"
                    )
                print(shlex.join(opening_cmd), flush=True)
                subprocess.run(opening_cmd, check=True)
                opening_cmds.append(opening_cmd)
                reused_openings[variant] = False
            else:
                print(f"reuse opening: {opening_output}", flush=True)
                reused_openings[variant] = True
            opening_sources[variant] = str(opening_assets.video)
            opening_outputs[variant] = str(opening_output)
            concat_segments.append(opening_output)
        episode_segment_for_concat = segment_output
        if include_end_credit and apply_video_episode_adjustments:
            tail_trim_plan = plan_tail_trim_adjustment(
                episode.episode_id,
                variant,
                segment_output,
                adjustment_dir,
            )
            if tail_trim_plan is not None:
                tail_trim_adjustments[variant] = tail_trim_plan.as_manifest()
                if tail_trim_plan.needed:
                    print(
                        f"{variant}: trim {tail_trim_plan.trim_seconds:g}s tail before end credits; "
                        f"{tail_trim_plan.source_duration_seconds:.3f}s -> "
                        f"{tail_trim_plan.output_duration_seconds:.3f}s",
                        flush=True,
                    )
                    print(shlex.join(tail_trim_plan.command), flush=True)
                    subprocess.run(tail_trim_plan.command, check=True)
                    adjustment_cmds.append(tail_trim_plan.command)
                    episode_segment_for_concat = tail_trim_plan.output
                else:
                    print(f"{variant}: configured tail trim not needed/invalid for {segment_output}", flush=True)
        concat_segments.append(episode_segment_for_concat)
        prepared_summary_start_for_chapters = summary_chapter_start
        inserted_prepared_segment = False
        inserted_prepared_segment_ref: PreparedEpisodeSegment | None = None
        inserted_prepared_segment_start_seconds: float | None = None
        if include_end_credit and apply_video_episode_adjustments and prepared_episode_segments:
            inserted_duration_base = media_file_duration(episode_segment_for_concat)
            for prepared_segment in prepared_episode_segments:
                if prepared_segment.insert != "before_end_credits":
                    continue
                prepared_output = prepared_segment.outputs.get(variant)
                if prepared_output is None:
                    raise SystemExit(
                        f"Prepared segment {prepared_segment.segment_id} has no output for variant {variant}:\n"
                        f"{prepared_segment.manifest_path}"
                    )
                if not prepared_output.is_file():
                    raise SystemExit(
                        f"Prepared segment output missing for {episode.episode_id} {variant}:\n{prepared_output}"
                    )
                print(f"{variant}: insert prepared segment {prepared_segment.segment_id}: {prepared_output}", flush=True)
                current_segment_start = sum(media_file_duration(segment) for segment in concat_segments)
                prepared_segment_outputs.setdefault(variant, []).append(
                    {
                        "segment_id": prepared_segment.segment_id,
                        "path": str(prepared_output),
                        "manifest": str(prepared_segment.manifest_path),
                        "description": prepared_segment.description,
                        "start_seconds": f"{current_segment_start:.6f}",
                    }
                )
                concat_segments.append(prepared_output)
                inserted_prepared_segment = True
                inserted_prepared_segment_ref = prepared_segment
                inserted_prepared_segment_start_seconds = current_segment_start
            if summary_chapter_start is None:
                prepared_summary_start_for_chapters = inserted_duration_base
        if include_end_credit and apply_video_episode_adjustments and spa2_tail_extension is not None:
            tail_plan = plan_spa2_tail_extension_adjustment(
                variant,
                episode_segment_for_concat,
                adjustment_dir,
                spa2_tail_extension,
                fade_frames=end_fade_frames if auto_end_fade else 0,
                restored_ac3_bitrate=restored_ac3_bitrate,
                eng_ac3_bitrate=eng_ac3_bitrate,
                stereo_ac3_bitrate=stereo_ac3_bitrate,
            )
            spa2_tail_extension_adjustments[variant] = tail_plan.as_manifest()
            print(
                f"{variant}: insert spa2 tail extension {spa2_tail_extension.extension_id} "
                f"({spa2_tail_extension.duration_seconds:.3f}s) before end credits",
                flush=True,
            )
            print(shlex.join(tail_plan.command), flush=True)
            subprocess.run(tail_plan.command, check=True)
            adjustment_cmds.append(tail_plan.command)
            concat_segments.append(tail_plan.output)
        elif include_end_credit and apply_video_episode_adjustments and auto_end_fade and not inserted_prepared_segment:
            fade_plan = plan_end_fade_adjustment(
                variant,
                episode_segment_for_concat,
                adjustment_dir,
                fade_frames=end_fade_frames,
                black_avg=end_fade_black_avg,
                black_max=end_fade_black_max,
                restored_ac3_bitrate=restored_ac3_bitrate,
                eng_ac3_bitrate=eng_ac3_bitrate,
                stereo_ac3_bitrate=stereo_ac3_bitrate,
            )
            end_fade_adjustments[variant] = fade_plan.as_manifest()
            stats = fade_plan.stats
            state = "generate end fade" if fade_plan.needed else "end frame already black"
            print(
                f"{variant}: {state}; avg_luma={stats['avg_luma']} "
                f"max_luma={stats['max_luma']}",
                flush=True,
            )
            if fade_plan.needed:
                print(shlex.join(fade_plan.command), flush=True)
                subprocess.run(fade_plan.command, check=True)
                adjustment_cmds.append(fade_plan.command)
                concat_segments.append(fade_plan.output)
        if include_end_credit:
            credit_source = select_end_credit(end_credit_root, video_path)
            credit_output = shared_end_credit_path(credit_dir, credit_source, video_path)
            credit_cmd = end_credit_segment_cmd(
                credit_source,
                video_path,
                credit_output,
                restored_ac3_bitrate=restored_ac3_bitrate,
                eng_ac3_bitrate=eng_ac3_bitrate,
                stereo_ac3_bitrate=stereo_ac3_bitrate,
            )
            if not credit_output.exists():
                print(shlex.join(credit_cmd), flush=True)
                subprocess.run(credit_cmd, check=True)
                credit_cmds.append(credit_cmd)
                reused_credits[variant] = False
            else:
                print(f"reuse end credit: {credit_output}", flush=True)
                reused_credits[variant] = True
            credit_sources[variant] = str(credit_source)
            credit_outputs[variant] = str(credit_output)
            concat_segments.append(credit_output)
        if include_opening or include_end_credit:
            concat_list = segment_dir / f"{episode.episode_id}_{variant}_concat.txt"
            write_concat_list(concat_list, concat_segments)
            concat_lists[variant] = str(concat_list)
            chapter_file = segment_dir / f"{episode.episode_id}_{variant}_chapters.ffmetadata"
            chapter_plan = write_chapter_metadata_file(
                chapter_file,
                concat_segments,
                include_opening=include_opening,
                include_end_credit=include_end_credit,
                episode_summary_start_seconds=prepared_summary_start_for_chapters,
                reference_episode_chapters=reference_episode_chapters,
            )
            chapter_files[variant] = str(chapter_file)
            chapter_plans[variant] = chapter_plan
            concat_cmd = final_concat_cmd(episode.episode_id, variant, concat_list, output, chapter_file)
            print(shlex.join(concat_cmd), flush=True)
            subprocess.run(concat_cmd, check=True)
            concat_cmds.append(concat_cmd)
        if authored_subtitles or attach_cover_art:
            subtitles_for_variant = authored_subtitles_with_prepared_segment_cues(
                authored_subtitles,
                inserted_prepared_segment_ref,
                inserted_prepared_segment_start_seconds,
                subtitle_dir / episode.episode_id / "authored_segments",
                variant,
            )
            cover_path = cover_art_path(cover_dir, output.stem, cover_asset) if attach_cover_art else None
            if cover_path is not None:
                cover_cmd = cover_art_cmd(output, cover_path, cover_asset)
                print(shlex.join(cover_cmd), flush=True)
                subprocess.run(cover_cmd, check=True)
                cover_cmds.append(cover_cmd)
                cover_outputs[variant] = str(cover_path)
            authored_output = output.with_name(f"{output.stem}.authoring.tmp{output.suffix}")
            author_cmd = review_authoring_cmd(output, authored_output, subtitles_for_variant, cover_path)
            print(shlex.join(author_cmd), flush=True)
            subprocess.run(author_cmd, check=True)
            authored_output.replace(output)
            authoring_cmds.append(author_cmd)
        mux_outputs[variant] = str(output)

    manifest = {
        "episode_id": episode.episode_id,
        "review_name": review_name,
        "silenced_start_seconds": silence_start_seconds,
        "mix_levels": {
            "preserved_channel_gain_db": preserved_channel_gain_db,
            "center_bed_gain_db": center_bed_gain_db,
            "dialogue_gain_db": dialogue_gain_db,
        },
        "episode_audio_patches": {
            "enabled": apply_audio_episode_adjustments,
            "version": EPISODE_AUDIO_PATCH_VERSION,
            "suppressed_ready_patch_ids": list(suppressed_ready_patch_ids),
            "dialogue_input": str(spanish_dialogue),
            "dialogue_after_static_patches": str(spanish_dialogue_before_recipes),
            "dialogue_effective": str(spanish_dialogue_for_mix),
            "spa1_fullmix_input": str(spa1_fullmix_wav),
            "spa1_fullmix_effective": str(spa1_fullmix_for_ac3),
            "dialogue": [audio_patch_as_manifest(patch) for patch in dialogue_patches],
            "spa1_fullmix": [audio_patch_as_manifest(patch) for patch in spa1_fullmix_patches],
            "repair_recipes": {
                "enabled": bool(repair_recipes),
                "recipes": [str(path) for path in repair_recipes],
                "events": repair_recipe_events,
            },
        },
        "rebuild_intermediates": rebuild_intermediates,
        "intermediate_cache": cache_events,
        "source_copies_enabled": copy_sources,
        "source_copies": source_copies,
        "keep_intermediate_segments": keep_intermediate_segments,
        "selected_videos": {variant: str(path) for variant, path in selected_videos.items()},
        "audio_tracks": {
            "english_devoiced_51_bed_lossless": str(devoiced_bed_51_wav),
            "english_original_51_silenced": str(eng_ac3),
            "spanish_restored_51_default": str(restored_ac3),
            "spanish1_restored_old_stereo": str(spa1_ac3),
            "spanish2_original_stereo_silenced": str(spa2_ac3),
        },
        "episode_segments": episode_segments,
        "video_outputs": mux_outputs,
        "embedded_subtitles": [track.as_manifest() for track in authored_subtitles],
        "cover_art": {
            "enabled": attach_cover_art,
            "files": cover_outputs,
        },
        "openings": {
            "enabled": include_opening,
            "generation": opening_generation,
            "sources": opening_sources,
            "segments": opening_outputs,
            "reused": reused_openings,
        },
        "end_credits": {
            "enabled": include_end_credit,
            "sources": credit_sources,
            "segments": credit_outputs,
            "reused": reused_credits,
            "concat_lists": concat_lists,
        },
        "chapters": {
            "enabled": include_opening or include_end_credit,
            "files": chapter_files,
            "plans": chapter_plans,
        },
        "adjustments": {
            "auto_end_fade_enabled": auto_end_fade,
            "video_episode_adjustments_enabled": apply_video_episode_adjustments,
            "audio_episode_adjustments_enabled": apply_audio_episode_adjustments,
            "end_fade_frames": end_fade_frames,
            "black_avg": end_fade_black_avg,
            "black_max": end_fade_black_max,
            "tail_trim": tail_trim_adjustments,
            "prepared_episode_segments": prepared_segment_outputs,
            "spa2_tail_extension": spa2_tail_extension_adjustments,
            "end_fade": end_fade_adjustments,
        },
        "commands": [
            *first_phase,
            *audio_cmds,
            *mux_cmds,
            *opening_cmds,
            *adjustment_cmds,
            *credit_cmds,
            *concat_cmds,
            *cover_cmds,
            *authoring_cmds,
        ],
    }
    write_manifest(review_dir / "manifest.json", manifest)
    write_final_build_report(
        review_dir / "BUILD_REPORT.md",
        episode_id=episode.episode_id,
        review_name=review_name,
        started_at=started_at,
        manifest=manifest,
        audio_tracks={
            "Track 1 - English Original 5.1": eng_ac3,
            "Track 2 - Spanish Restored Original Dub 5.1 (default)": restored_ac3,
            "Track 3 - Spanish Original Dub Restored Stereo": spa1_ac3,
            "Track 4 - Spanish Redubbing Original Stereo": spa2_ac3,
        },
        mux_cmds=mux_cmds,
        opening_cmds=opening_cmds,
        adjustment_cmds=adjustment_cmds,
        credit_cmds=credit_cmds,
        concat_cmds=concat_cmds,
        cover_cmds=cover_cmds,
        authoring_cmds=authoring_cmds,
    )
    print_final_episode_summary(
        episode.episode_id,
        review_dir,
        audio_tracks={
            "Track 1 - English Original 5.1": eng_ac3,
            "Track 2 - Spanish Restored Original Dub 5.1 (default)": restored_ac3,
            "Track 3 - Spanish Original Dub Restored Stereo": spa1_ac3,
            "Track 4 - Spanish Redubbing Original Stereo": spa2_ac3,
        },
        video_outputs={variant: Path(path) for variant, path in mux_outputs.items()},
        elapsed_seconds=time.monotonic() - started_at,
    )
    return 0


def collect_repair_recipes(episode_id: str, explicit: list[Path], recipe_dir: Path | None) -> list[Path]:
    """Collect explicit and per-episode repair-tool recipes for final build."""

    recipes: list[Path] = []
    recipes.extend(explicit)
    if recipe_dir is not None and recipe_dir.exists():
        patterns = [
            f"{episode_id}*.recipe.json",
            f"{episode_id}*.json",
        ]
        seen = {path.resolve() for path in recipes if path.exists()}
        for pattern in patterns:
            for path in sorted(recipe_dir.glob(pattern)):
                resolved = path.resolve()
                if resolved not in seen:
                    recipes.append(path)
                    seen.add(resolved)
    missing = [path for path in recipes if not path.exists()]
    if missing:
        raise SystemExit("Missing repair recipe(s):\n" + "\n".join(str(path) for path in missing))
    return recipes


def silence_start_filter(seconds: float) -> str:
    if seconds <= 0:
        return "anull"
    return f"volume=volume=0:enable='lt(t,{seconds:g})'"


def audio_patch_filter(patches: Iterable[EpisodeAudioPatch]) -> str:
    filters = []
    for patch in patches:
        if patch.method != "volume_floor":
            raise ValueError(f"audio_patch_filter only supports volume_floor patches, got {patch.method}")
        start = min(patch.start_seconds, patch.end_seconds)
        end = max(patch.start_seconds, patch.end_seconds)
        floor = max(min(patch.floor, 1.0), 0.0)
        filters.append(
            "volume="
            f"volume='if(between(t\\,{start:g}\\,{end:g})\\,{floor:g}\\,1)'"
            ":eval=frame"
        )
    return ",".join(filters) if filters else "anull"


def patches_for_target(
    episode_id: str,
    target: str,
    enabled: bool,
    ready_patch_dir: Path | None = None,
    suppressed_patch_ids: Iterable[str] = (),
) -> tuple[EpisodeAudioPatch, ...]:
    if not enabled:
        return ()
    static_patches = tuple(
        patch
        for patch in EPISODE_AUDIO_PATCHES.get(episode_id, ())
        if target in patch.targets
    )
    suppressed = set(suppressed_patch_ids)
    ready_patches = tuple(
        patch
        for patch in ready_patches_for_target(episode_id, target, ready_patch_dir)
        if patch.patch_id not in suppressed
    )
    if not ready_patches:
        return static_patches
    ready_ids = {patch.patch_id for patch in ready_patches}
    superseded_ids = {patch_id for patch in ready_patches for patch_id in patch.supersedes}
    all_ready_blocked_ids = ready_ids | superseded_ids
    filtered_static = tuple(patch for patch in static_patches if patch.patch_id not in all_ready_blocked_ids)
    filtered_ready = tuple(patch for patch in ready_patches if patch.patch_id not in superseded_ids)
    return filtered_static + filtered_ready


def ready_patches_for_target(episode_id: str, target: str, ready_patch_dir: Path | None) -> tuple[EpisodeAudioPatch, ...]:
    """Load ready WAV+JSON patch manifests for an episode/target."""

    if ready_patch_dir is None:
        return ()
    episode_dir = ready_patch_dir / episode_id
    if not episode_dir.exists():
        return ()
    patches = []
    for manifest_path in sorted(episode_dir.glob("*/patch.json")) + sorted(episode_dir.glob("*.patch.json")):
        patch = ready_patch_from_manifest(manifest_path)
        if target in patch.targets:
            patches.append(patch)
    return tuple(patches)


def ready_patch_from_manifest(manifest_path: Path) -> EpisodeAudioPatch:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("kind") not in {None, "robotech_ready_audio_patch"}:
        raise ValueError(f"Unsupported ready patch manifest kind in {manifest_path}: {data.get('kind')!r}")

    def resolve_path(value: str) -> str:
        if not value:
            return ""
        path = Path(value)
        if not path.is_absolute():
            path = manifest_path.parent / path
        return str(path)

    replacement_path = resolve_path(str(data.get("replacement_path") or "replacement.wav"))
    texture_overlay_path = resolve_path(str(data.get("texture_overlay_path") or ""))
    patch_id = str(data.get("patch_id") or manifest_path.parent.name or manifest_path.stem)
    start = float(data["start_seconds"])
    end = float(data["end_seconds"])
    replacement_source_seconds = float(data.get("replacement_source_seconds") or abs(end - start))
    return EpisodeAudioPatch(
        patch_id=patch_id,
        start_seconds=start,
        end_seconds=end,
        floor=float(data.get("floor", 1.0)),
        targets=tuple(data.get("targets", ("dialogue",))),
        description=str(data.get("description", f"Ready audio patch from {manifest_path}")),
        method=str(data.get("method", "external_replacement_with_edge_texture")),
        replacement_path=replacement_path,
        replacement_source_seconds=replacement_source_seconds,
        replacement_gain_db=float(data.get("replacement_gain_db", 0.0)),
        texture_gain_db=float(data.get("texture_gain_db", -120.0)),
        texture_edge_seconds=float(data.get("texture_edge_seconds", 0.0)),
        texture_overlay_path=texture_overlay_path,
        insert_fade_in_seconds=float(data.get("insert_fade_in_seconds", 0.0)),
        insert_fade_out_seconds=float(data.get("insert_fade_out_seconds", 0.0)),
        insert_fade_in_curve=str(data.get("insert_fade_in_curve", "tri")),
        supersedes=tuple(str(value) for value in data.get("supersedes", ())),
    )


def audio_patch_as_manifest(patch: EpisodeAudioPatch) -> dict[str, object]:
    return {
        "patch_id": patch.patch_id,
        "start_seconds": patch.start_seconds,
        "end_seconds": patch.end_seconds,
        "floor": patch.floor,
        "targets": list(patch.targets),
        "description": patch.description,
        "method": patch.method,
        "bridge_pre_seconds": patch.bridge_pre_seconds,
        "crossfade_seconds": patch.crossfade_seconds,
        "rubberband_options": patch.rubberband_options,
        "replacement_path": patch.replacement_path,
        "replacement_source_seconds": patch.replacement_source_seconds,
        "replacement_gain_db": patch.replacement_gain_db,
        "texture_gain_db": patch.texture_gain_db,
        "texture_edge_seconds": patch.texture_edge_seconds,
        "texture_overlay_path": patch.texture_overlay_path,
        "insert_fade_in_seconds": patch.insert_fade_in_seconds,
        "insert_fade_out_seconds": patch.insert_fade_out_seconds,
        "insert_fade_in_curve": patch.insert_fade_in_curve,
        "supersedes": list(patch.supersedes),
    }


def spa2_tail_extension_manifest_path(root: Path, episode_id: str, extension_id: str | None = None) -> Path:
    episode_id = normalize_episode_id(episode_id)
    if extension_id:
        return root / episode_id / extension_id / "tail.json"
    episode_dir = root / episode_id
    candidates = sorted(episode_dir.glob("*/tail.json"))
    if not candidates:
        return episode_dir / f"{episode_id.lower()}_spa2_tail_v001" / "tail.json"
    return candidates[0]


def load_spa2_tail_extension(root: Path | None, episode_id: str) -> Spa2TailExtension | None:
    if root is None:
        return None
    episode_id = normalize_episode_id(episode_id)
    episode_dir = root / episode_id
    if not episode_dir.exists():
        return None
    manifests = sorted(episode_dir.glob("*/tail.json"))
    if not manifests:
        return None
    if len(manifests) > 1:
        raise SystemExit(
            f"Multiple spa2 tail extensions found for {episode_id}; keep one active or use --no-spa2-tail-extensions:\n"
            + "\n".join(str(path) for path in manifests)
        )
    data = json.loads(manifests[0].read_text(encoding="utf-8"))
    if data.get("kind") not in {None, "robotech_spa2_tail_extension"}:
        raise SystemExit(f"Unsupported spa2 tail manifest kind in {manifests[0]}: {data.get('kind')!r}")
    audio_value = str(data.get("audio_path") or "spa2_tail.wav")
    audio_path = Path(audio_value)
    if not audio_path.is_absolute():
        audio_path = manifests[0].parent / audio_path
    if not audio_path.is_file():
        raise SystemExit(f"Missing spa2 tail audio declared by {manifests[0]}: {audio_path}")
    return Spa2TailExtension(
        episode_id=episode_id,
        extension_id=str(data.get("extension_id") or manifests[0].parent.name),
        audio_path=audio_path,
        duration_seconds=float(data.get("duration_seconds") or media_file_duration(audio_path)),
        source_path=str(data.get("source_path") or ""),
        source_audio_stream=int(data.get("source_audio_stream", -1)),
        source_start_seconds=float(data.get("source_start_seconds", 0.0)),
        old_spa1_tail_gain_db=float(data.get("old_spa1_tail_gain_db", 0.0)),
        description=str(data.get("description") or f"Newer Spanish dub tail extension for {episode_id}"),
    )


def load_prepared_episode_segments(root: Path, episode_id: str) -> list[PreparedEpisodeSegment]:
    episode_id = normalize_episode_id(episode_id)
    episode_dir = root / episode_id
    if not episode_dir.exists():
        return []
    segments: list[PreparedEpisodeSegment] = []
    for manifest_path in sorted(episode_dir.glob("*/segment.json")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("kind") not in {None, "robotech_prepared_episode_segment"}:
            raise SystemExit(f"Unsupported prepared segment kind in {manifest_path}: {data.get('kind')!r}")
        outputs: dict[str, Path] = {}
        raw_outputs = data.get("outputs", {})
        if not isinstance(raw_outputs, dict):
            raise SystemExit(f"Prepared segment manifest outputs must be an object: {manifest_path}")
        for variant, record in raw_outputs.items():
            if isinstance(record, dict):
                value = str(record.get("segment") or "")
            else:
                value = str(record)
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = path if path.exists() else manifest_path.parent / path
            outputs[str(variant)] = path
        segments.append(
            PreparedEpisodeSegment(
                episode_id=episode_id,
                segment_id=str(data.get("segment_id") or manifest_path.parent.name),
                manifest_path=manifest_path,
                insert=str(data.get("insert") or "before_end_credits"),
                outputs=outputs,
                suppresses_ready_audio_patch_ids=tuple(str(value) for value in data.get("suppresses_ready_audio_patch_ids", ())),
                description=str(data.get("description") or manifest_path.parent.name),
                subtitle_sources=resolve_prepared_segment_subtitles(manifest_path, data.get("subtitle_sources")),
            )
        )
    return segments


def resolve_prepared_segment_subtitles(manifest_path: Path, raw_sources: object) -> dict[str, Path]:
    if not isinstance(raw_sources, dict):
        return {}
    sources: dict[str, Path] = {}
    for key, value in raw_sources.items():
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = path if path.exists() else manifest_path.parent / path
        sources[str(key)] = path
    return sources


def load_reference_episode_chapters(path: Path | None, episode_id: str) -> list[ReferenceEpisodeChapter]:
    if path is None or not path.is_file():
        return []
    episode_id = normalize_episode_id(episode_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    record = data.get("episodes", {}).get(episode_id)
    if not isinstance(record, dict):
        return []
    chapters = []
    for item in record.get("episode_chapters", []):
        if not isinstance(item, dict):
            continue
        try:
            start_seconds = float(item.get("start_seconds", 0.0))
        except (TypeError, ValueError):
            continue
        title = str(item.get("title") or "Episode")
        chapters.append(ReferenceEpisodeChapter(title=title, start_seconds=max(start_seconds, 0.0)))
    chapters.sort(key=lambda chapter: chapter.start_seconds)
    filtered: list[ReferenceEpisodeChapter] = []
    for chapter in chapters:
        if filtered and abs(filtered[-1].start_seconds - chapter.start_seconds) < 0.05:
            continue
        filtered.append(chapter)
    return filtered


def cmd_reference_chapters(reference_root: Path, pattern: str, out_path: Path, run: bool) -> int:
    candidates = sorted(reference_root.glob(pattern))
    if not candidates:
        raise SystemExit(f"No reference MKVs matched {pattern!r} under {reference_root}")
    records: dict[str, dict[str, object]] = {}
    for reference_mkv in candidates:
        match = re.search(r"S(\d{2})E(\d{2})", reference_mkv.name, flags=re.IGNORECASE)
        if not match:
            print(f"skip reference without SxxExx id: {reference_mkv}", flush=True)
            continue
        episode_id = f"S{match.group(1)}E{match.group(2)}"
        probe_cmd = [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_chapters",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(reference_mkv),
        ]
        result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
        probe = json.loads(result.stdout)
        raw_chapters = []
        for chapter in probe.get("chapters", []):
            start = float(chapter.get("start_time", 0.0))
            end = float(chapter.get("end_time", start))
            tags = chapter.get("tags", {}) if isinstance(chapter.get("tags"), dict) else {}
            raw_chapters.append(
                {
                    "start_seconds": round(start, 6),
                    "end_seconds": round(end, 6),
                    "title": str(tags.get("title") or ""),
                }
            )
        raw_chapters.sort(key=lambda item: item["start_seconds"])
        if len(raw_chapters) < 3:
            print(f"{episode_id}: found {len(raw_chapters)} chapter(s), not enough to infer episode middle marks")
            continue
        opening_end = float(raw_chapters[0]["end_seconds"])
        end_credit_start = float(raw_chapters[-1]["start_seconds"])
        episode_chapters: list[dict[str, object]] = []
        for index, chapter in enumerate(raw_chapters[1:-1], start=1):
            relative_start = float(chapter["start_seconds"]) - opening_end
            if relative_start < -0.05 or float(chapter["start_seconds"]) >= end_credit_start - 0.05:
                continue
            relative_start = max(relative_start, 0.0)
            title = "Episode" if not episode_chapters else f"Episode Part {len(episode_chapters) + 1}"
            episode_chapters.append({"title": title, "start_seconds": round(relative_start, 3)})
        if not episode_chapters or episode_chapters[0]["start_seconds"] > 0.05:
            episode_chapters.insert(0, {"title": "Episode", "start_seconds": 0.0})
        records[episode_id] = {
            "source_path": str(reference_mkv),
            "reference_opening_end_seconds": round(opening_end, 6),
            "reference_end_credit_start_seconds": round(end_credit_start, 6),
            "reference_episode_duration_seconds": round(end_credit_start - opening_end, 6),
            "episode_chapters": episode_chapters,
            "raw_chapters": raw_chapters,
        }
        starts = ", ".join(f"{item['title']}={item['start_seconds']}s" for item in episode_chapters)
        print(f"{episode_id}: {starts}", flush=True)
    output = {
        "kind": "robotech_reference_episode_chapters",
        "description": (
            "Episode-relative chapter starts harvested once from reference Robotech MKVs. "
            "Final builds use these local values and do not need the reference files mounted."
        ),
        "reference_root": str(reference_root),
        "pattern": pattern,
        "episodes": records,
    }
    if run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote={out_path}")
    else:
        print(f"would write={out_path}")
    return 0


def patch_audio_steps(
    input_path: Path,
    output_path: Path,
    patches: Iterable[EpisodeAudioPatch],
    work_dir: Path,
    label_prefix: str,
) -> list[tuple[list[str], Path, str]]:
    """Build sequential patch commands so replacement patches can stack safely."""

    patch_list = tuple(patches)
    if not patch_list:
        return []
    steps: list[tuple[list[str], Path, str]] = []
    current_input = input_path
    for index, patch in enumerate(patch_list, start=1):
        step_output = output_path if index == len(patch_list) else work_dir / f"{label_prefix}_{index:02d}_{patch.patch_id}.wav"
        cmd = patch_audio_cmd(current_input, step_output, (patch,))
        steps.append((cmd, step_output, patch.patch_id))
        current_input = step_output
    return steps


def patch_audio_cmd(input_path: Path, output_path: Path, patches: Iterable[EpisodeAudioPatch]) -> list[str]:
    patch_list = tuple(patches)
    if any(patch.method == "rubberband_bridge" for patch in patch_list):
        if len(patch_list) != 1:
            raise ValueError("rubberband_bridge audio patches must be rendered one at a time")
        return rubberband_bridge_patch_cmd(input_path, output_path, patch_list[0])
    if any(patch.method == "external_replacement_with_edge_texture" for patch in patch_list):
        if len(patch_list) != 1:
            raise ValueError("external replacement audio patches must be rendered one at a time")
        return external_replacement_with_edge_texture_patch_cmd(input_path, output_path, patch_list[0])
    if any(patch.method == "ready_replacement_clip" for patch in patch_list):
        if len(patch_list) != 1:
            raise ValueError("ready replacement clip audio patches must be rendered one at a time")
        return ready_replacement_clip_patch_cmd(input_path, output_path, patch_list[0])
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-af",
        audio_patch_filter(patch_list),
        "-c:a",
        "pcm_s24le",
        str(output_path),
    ]


def external_replacement_with_edge_texture_patch_cmd(input_path: Path, output_path: Path, patch: EpisodeAudioPatch) -> list[str]:
    start = min(patch.start_seconds, patch.end_seconds)
    end = max(patch.start_seconds, patch.end_seconds)
    replacement_path = Path(patch.replacement_path)
    replacement_duration = end - start
    source_duration = patch.replacement_source_seconds
    if replacement_duration <= 0 or source_duration <= 0:
        raise ValueError(f"Invalid external replacement patch timing for {patch.patch_id}")
    if not replacement_path.exists():
        raise FileNotFoundError(f"Replacement audio does not exist: {replacement_path}")
    texture_overlay_path = Path(patch.texture_overlay_path) if patch.texture_overlay_path else None
    if texture_overlay_path and not texture_overlay_path.exists():
        raise FileNotFoundError(f"Texture overlay audio does not exist: {texture_overlay_path}")
    tempo = source_duration / replacement_duration
    edge = max(min(patch.texture_edge_seconds, replacement_duration / 2), 0.0)
    middle_start = edge
    middle_end = replacement_duration - edge
    fade_in = max(patch.insert_fade_in_seconds, 0.0)
    fade_out = max(patch.insert_fade_out_seconds, 0.0)
    fade_out_start = max(replacement_duration - fade_out, 0.0)
    fade_curve = patch.insert_fade_in_curve or "tri"
    texture_filter = (
        f"volume={patch.texture_gain_db:g}dB,"
        f"volume=enable='between(t,{middle_start:.6f},{middle_end:.6f})':volume=0"
    )
    insert_filters = [
        f"volume={patch.replacement_gain_db:g}dB",
    ]
    if fade_in:
        insert_filters.append(f"afade=t=in:st=0:d={fade_in:.6f}:curve={fade_curve}")
    if fade_out:
        insert_filters.append(f"afade=t=out:st={fade_out_start:.6f}:d={fade_out:.6f}")
    insert_filter = ",".join(insert_filters)
    texture_overlay_input = ""
    texture_mix_inputs = 2
    if texture_overlay_path:
        texture_overlay_input = (
            f"[2:a]aformat=sample_fmts=fltp:sample_rates=48000,"
            f"pan=stereo|c0=c0|c1=c1,apad,atrim=0:{replacement_duration:.6f},"
            "asetpts=N/SR/TB[texture_overlay];"
        )
        texture_mix_inputs = 3
    insert_mix_inputs = "[voice][texture]"
    if texture_overlay_path:
        insert_mix_inputs += "[texture_overlay]"
    filter_complex = (
        f"[0:a]atrim=0:{start:.6f},asetpts=N/SR/TB[pre];"
        f"[0:a]atrim={start:.6f}:{end:.6f},asetpts=N/SR/TB,{texture_filter}[texture];"
        f"[1:a]aformat=sample_fmts=fltp:sample_rates=48000,"
        f"pan=stereo|c0=c0|c1=c0,rubberband=tempo={tempo:.8f},"
        f"apad,atrim=0:{replacement_duration:.6f},asetpts=N/SR/TB,{insert_filter}[voice];"
        f"{texture_overlay_input}"
        f"{insert_mix_inputs}amix=inputs={texture_mix_inputs}:duration=first:normalize=0,alimiter=limit=0.95[insert];"
        f"[0:a]atrim=start={end:.6f},asetpts=N/SR/TB[post];"
        "[pre][insert][post]concat=n=3:v=0:a=1[out]"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-i",
        str(replacement_path),
    ]
    if texture_overlay_path:
        cmd.extend(["-i", str(texture_overlay_path)])
    cmd.extend([
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:a",
        "pcm_s24le",
        str(output_path),
    ])
    return cmd


def ready_replacement_clip_patch_cmd(input_path: Path, output_path: Path, patch: EpisodeAudioPatch) -> list[str]:
    start = min(patch.start_seconds, patch.end_seconds)
    end = max(patch.start_seconds, patch.end_seconds)
    replacement_path = Path(patch.replacement_path)
    replacement_duration = end - start
    source_duration = patch.replacement_source_seconds
    if replacement_duration <= 0 or source_duration <= 0:
        raise ValueError(f"Invalid ready replacement patch timing for {patch.patch_id}")
    if not replacement_path.exists():
        raise FileNotFoundError(f"Replacement audio does not exist: {replacement_path}")
    tempo = source_duration / replacement_duration
    fade_in = max(patch.insert_fade_in_seconds, 0.0)
    fade_out = max(patch.insert_fade_out_seconds, 0.0)
    fade_out_start = max(replacement_duration - fade_out, 0.0)
    fade_curve = patch.insert_fade_in_curve or "tri"
    insert_filters = [
        "aformat=sample_fmts=fltp:sample_rates=48000",
        "pan=stereo|c0=c0|c1=c1",
    ]
    if abs(tempo - 1.0) > 0.000001:
        insert_filters.append(f"rubberband=tempo={tempo:.8f}")
    insert_filters.extend(
        [
            f"volume={patch.replacement_gain_db:g}dB",
            f"apad",
            f"atrim=0:{replacement_duration:.6f}",
            "asetpts=N/SR/TB",
        ]
    )
    if fade_in:
        insert_filters.append(f"afade=t=in:st=0:d={fade_in:.6f}:curve={fade_curve}")
    if fade_out:
        insert_filters.append(f"afade=t=out:st={fade_out_start:.6f}:d={fade_out:.6f}")
    insert_filter = ",".join(insert_filters)
    filter_complex = (
        f"[0:a]atrim=0:{start:.6f},asetpts=N/SR/TB[pre];"
        f"[1:a]{insert_filter}[insert];"
        f"[0:a]atrim=start={end:.6f},asetpts=N/SR/TB[post];"
        "[pre][insert][post]concat=n=3:v=0:a=1[out]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-i",
        str(replacement_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:a",
        "pcm_s24le",
        str(output_path),
    ]


def rubberband_bridge_patch_cmd(input_path: Path, output_path: Path, patch: EpisodeAudioPatch) -> list[str]:
    start = min(patch.start_seconds, patch.end_seconds)
    end = max(patch.start_seconds, patch.end_seconds)
    bridge_pre = max(patch.bridge_pre_seconds, 0.0)
    crossfade = max(patch.crossfade_seconds, 0.0)
    bridge_start = max(start - bridge_pre, 0.0)
    source_bridge_duration = start - bridge_start
    replacement_duration = end - bridge_start
    rendered_bridge_duration = replacement_duration + (2 * crossfade)
    if source_bridge_duration <= 0 or rendered_bridge_duration <= 0:
        raise ValueError(f"Invalid rubberband bridge patch timing for {patch.patch_id}")
    tempo = source_bridge_duration / rendered_bridge_duration
    rubberband = f"rubberband=tempo={tempo:.8f}"
    if patch.rubberband_options:
        rubberband += f":{patch.rubberband_options}"
    filter_complex = (
        f"[0:a]atrim=0:{bridge_start:.6f},asetpts=N/SR/TB[a];"
        f"[0:a]atrim={bridge_start:.6f}:{start:.6f},asetpts=N/SR/TB,{rubberband}[b];"
        f"[0:a]atrim=start={end:.6f},asetpts=N/SR/TB[c];"
        f"[a][b]acrossfade=d={crossfade:.6f}:c1=tri:c2=tri[ab];"
        f"[ab][c]acrossfade=d={crossfade:.6f}:c1=tri:c2=tri[out]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:a",
        "pcm_s24le",
        str(output_path),
    ]


def silence_audio_cmd(input_path: Path, output_path: Path, bitrate: str, seconds: float) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-af",
        silence_start_filter(seconds),
        "-c:a",
        "ac3",
        "-b:a",
        bitrate,
        str(output_path),
    ]


def run_cached_command(
    cmd: list[str],
    output_path: Path,
    label: str,
    rebuild: bool,
    run: bool,
    cache_events: list[dict[str, object]],
) -> bool:
    if output_path.exists() and not rebuild:
        print(f"reuse {label}: {output_path}", flush=True)
        cache_events.append({"label": label, "output": str(output_path), "action": "reused"})
        return False
    print(shlex.join(cmd), flush=True)
    cache_events.append(
        {
            "label": label,
            "output": str(output_path),
            "action": "rebuilt" if output_path.exists() and rebuild else "created",
            "command": cmd,
        }
    )
    if run:
        subprocess.run(cmd, check=True)
    return True


def cached_stem_output(stem_dir: Path, stem: str) -> Path | None:
    try:
        return find_single_stem_output(stem_dir, stem)
    except SystemExit:
        return None


def select_proc_videos(proc_video_root: Path, episode_id: str) -> dict[str, Path]:
    episode_number = int(episode_id[-2:])
    episode_dir = proc_video_root / f"ep{episode_number:02d}"
    if not episode_dir.is_dir():
        raise SystemExit(f"Missing processed video folder: {episode_dir}")

    candidates = [path for path in episode_dir.glob(f"Robotech-{episode_id}*.mp4") if "copia" not in path.name.lower()]
    selected = {
        "ai_remaster": choose_proc_video(candidates, include=["_AIRemaster"], exclude=[]),
        "remaster": choose_proc_video(candidates, include=["_Remaster"], exclude=["AIRemaster", "W2xEX", "VFI"]),
        "remaster_49fps": choose_proc_video(candidates, include=["_Remaster", "W2xEX", "VFI"], exclude=["AIRemaster"]),
    }
    missing = [name for name, path in selected.items() if path is None]
    if missing:
        found = "\n".join(str(path) for path in candidates)
        raise SystemExit(f"{episode_id} missing processed video variants: {', '.join(missing)}\nFound:\n{found}")
    return {name: path for name, path in selected.items() if path is not None}


def choose_proc_video(candidates: list[Path], include: list[str], exclude: list[str]) -> Path | None:
    matches = []
    for path in candidates:
        name = path.name
        if all(token in name for token in include) and not any(token in name for token in exclude):
            matches.append(path)
    if not matches:
        return None
    matches.sort(key=lambda path: (not is_version_b(path), len(path.name), path.name))
    return matches[0]


def is_version_b(path: Path) -> bool:
    return "-B" in path.stem or "_B" in path.stem or " version b" in path.stem.lower()


@dataclass(frozen=True)
class VideoEncodeInfo:
    width: int
    height: int
    rate: Fraction
    rate_text: str
    profile: str | None
    level: str | None


@dataclass(frozen=True)
class OpeningAssets:
    video: Path
    english_51: Path
    spanish_original_stereo: Path
    generation_spanish_51: Path


@dataclass(frozen=True)
class SubtitleTrack:
    path: Path
    language: str
    title: str
    sidecar_suffix: str

    def as_manifest(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "language": self.language,
            "title": self.title,
            "sidecar_suffix": self.sidecar_suffix,
        }


@dataclass(frozen=True)
class EndFadeAdjustment:
    variant: str
    source: Path
    output: Path
    needed: bool
    stats: dict[str, float | int | bool | str]
    command: list[str]

    def as_manifest(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "output": str(self.output),
            "needed": self.needed,
            "stats": self.stats,
            "command": self.command,
        }


@dataclass(frozen=True)
class TailTrimAdjustment:
    variant: str
    source: Path
    output: Path
    needed: bool
    trim_seconds: float
    source_duration_seconds: float
    output_duration_seconds: float
    command: list[str]
    description: str

    def as_manifest(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "output": str(self.output),
            "needed": self.needed,
            "trim_seconds": self.trim_seconds,
            "source_duration_seconds": self.source_duration_seconds,
            "output_duration_seconds": self.output_duration_seconds,
            "description": self.description,
            "command": self.command,
        }


def resolve_opening_assets(opening_root: Path, generation: str, work_dir: Path) -> OpeningAssets:
    video = opening_root / "intromasterAI48fps1440x1080cropallac3_v56.mkv"
    english_51 = opening_root / "track01eng51.ac3"
    spanish_original = opening_root / "track02spa1ori.ac3"
    generation_tracks = {
        "1": work_dir / "review" / "opening_credits" / "OC_spa_voice_into_eng51_separate_001" / "01_p1_voice_plus_eng51_center_10pct.ac3",
        "2": work_dir / "review" / "opening_credits" / "OC_spa_voice_into_eng51_separate_001" / "02_p2_voice_plus_eng51_center_10pct.ac3",
        "3": work_dir / "review" / "opening_credits" / "OC_spa_voice_into_eng51_separate_001" / "03_p3_voice_plus_eng51_center_10pct.ac3",
    }
    assets = OpeningAssets(
        video=video,
        english_51=english_51,
        spanish_original_stereo=spanish_original,
        generation_spanish_51=generation_tracks[generation],
    )
    missing = [path for path in (assets.video, assets.english_51, assets.spanish_original_stereo) if not path.is_file()]
    if missing:
        raise SystemExit("Missing opening asset(s):\n" + "\n".join(str(path) for path in missing))
    return assets


def select_end_credit(end_credit_root: Path, target_video: Path) -> Path:
    if not end_credit_root.is_dir():
        raise SystemExit(f"Missing end-credit folder: {end_credit_root}")
    target_info = video_encode_info(target_video)
    candidates = [path for path in end_credit_root.glob("*.mp4") if "copia" not in path.name.lower()]
    if not candidates:
        raise SystemExit(f"No end-credit mp4 files found in {end_credit_root}")

    def score(path: Path) -> tuple[float, int, str]:
        info = video_encode_info(path)
        fps_delta = abs(float(info.rate) - float(target_info.rate))
        size_delta = abs(info.width - target_info.width) + abs(info.height - target_info.height)
        return (fps_delta, size_delta, path.name)

    return sorted(candidates, key=score)[0]


def video_encode_info(path: Path) -> VideoEncodeInfo:
    probe = ffprobe_full(path)
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        rate_text = stream.get("r_frame_rate") or stream.get("avg_frame_rate") or "0/1"
        rate = parse_rate(rate_text)
        if rate <= 0:
            rate_text = stream.get("avg_frame_rate") or "0/1"
            rate = parse_rate(rate_text)
        return VideoEncodeInfo(
            width=int(stream.get("width") or 0),
            height=int(stream.get("height") or 0),
            rate=rate,
            rate_text=rate_text,
            profile=stream.get("profile"),
            level=format_h264_level(stream.get("level")),
        )
    raise SystemExit(f"No video stream found in {path}")


def parse_rate(value: str) -> Fraction:
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return Fraction(0, 1)


def format_h264_level(value: object) -> str | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return f"{number // 10}.{number % 10}"


def x264_profile_arg(profile: str | None) -> str:
    normalized = (profile or "").strip().lower()
    if normalized == "main":
        return "main"
    if normalized == "baseline":
        return "baseline"
    return "high"


def final_video_name(episode_id: str, variant: str, source_video: Path) -> str:
    suffix = {
        "ai_remaster": "AIRemaster",
        "remaster": "Remaster",
        "remaster_49fps": "Remaster49fps",
    }[variant]
    title = safe_filename_part(episode_display_title(episode_id))
    return f"Robotech - {episode_id} - {title}_{suffix}RestoredAudio.mkv"


def legacy_final_video_name(episode_id: str, variant: str, source_video: Path) -> str:
    suffix = {
        "ai_remaster": "AIRemaster",
        "remaster": "Remaster",
        "remaster_49fps": "Remaster_49fps",
    }[variant]
    return f"{episode_id}_{suffix}_restored_audio.mkv"


def safe_filename_part(value: str) -> str:
    return re.sub(r'[\\\\/:*?"<>|]', "", value).strip()


def episode_display_title(episode_id: str) -> str:
    return EPISODE_TITLES.get(episode_id, episode_id)


def normalize_episode_id(value: str) -> str:
    match = re.search(r"(\d{1,2})$", value.strip(), re.IGNORECASE)
    if value.upper().startswith("S01E") and match:
        return f"S01E{int(match.group(1)):02d}"
    if match:
        return f"S01E{int(match.group(1)):02d}"
    raise SystemExit(f"Could not parse episode id: {value}")


def find_english_subtitle_mkv(root: Path, episode_id: str) -> Path:
    episode_number = int(episode_id[-2:])
    matches = []
    for path in root.glob("*.mkv"):
        match = re.search(r"1x(\d{1,2})", path.name, re.IGNORECASE)
        if match and int(match.group(1)) == episode_number:
            matches.append(path)
    if len(matches) != 1:
        raise SystemExit(f"Expected one English subtitle MKV for {episode_id}, found {len(matches)} in {root}")
    return matches[0]


def find_spanish_ssa(root: Path, episode_id: str) -> Path:
    episode_number = int(episode_id[-2:])
    pattern = f"Robotech {episode_number:02d} - *.ssa"
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"Expected one Spanish SSA for {episode_id}, found {len(matches)} in {root}")
    return matches[0]


def subtitle_stream_index(path: Path, language: str) -> int:
    data = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-hide_banner",
                "-v",
                "error",
                "-show_entries",
                "stream=index,codec_type:stream_tags=language",
                "-of",
                "json",
                str(path),
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout
    )
    subtitle_streams = [
        stream
        for stream in data.get("streams", [])
        if stream.get("codec_type") == "subtitle" and stream.get("tags", {}).get("language") == language
    ]
    if not subtitle_streams:
        subtitle_streams = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "subtitle"]
    if len(subtitle_streams) != 1:
        raise SystemExit(f"Expected one subtitle stream in {path}, found {len(subtitle_streams)}")
    return int(subtitle_streams[0]["index"])


@dataclass(frozen=True)
class PgsEvent:
    start: str
    end: str
    sample_time: float


@dataclass(frozen=True)
class SrtCue:
    index: int
    start: str
    end: str
    text: str


def pgs_timing_events(path: Path, min_display_packet_size: int) -> list[PgsEvent]:
    data = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-hide_banner",
                "-v",
                "error",
                "-select_streams",
                "s:0",
                "-show_packets",
                "-show_entries",
                "packet=pts_time,size",
                "-of",
                "json",
                str(path),
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout
    )
    groups: dict[float, list[int]] = {}
    for packet in data.get("packets", []):
        if "pts_time" not in packet:
            continue
        timestamp = round(float(packet["pts_time"]), 3)
        groups.setdefault(timestamp, []).append(int(packet.get("size", 0)))

    events: list[PgsEvent] = []
    active_start: float | None = None
    for timestamp in sorted(groups):
        is_display = max(groups[timestamp]) >= min_display_packet_size
        if is_display:
            if active_start is not None and timestamp > active_start + 0.1:
                events.append(pgs_event(active_start, timestamp))
            active_start = timestamp
        elif active_start is not None and timestamp > active_start:
            events.append(pgs_event(active_start, timestamp))
            active_start = None
    return events


def pgs_event(start: float, end: float) -> PgsEvent:
    sample_time = start + min(0.35, max(0.05, (end - start) / 2))
    return PgsEvent(start=seconds_to_srt_time(start), end=seconds_to_srt_time(end), sample_time=sample_time)


def render_pgs_subtitle_frame(source: Path, sample_time: float, image_path: Path, crop_top_ratio: float) -> None:
    seek_lead = 0.75 if sample_time >= 0.75 else sample_time
    input_seek = max(0.0, sample_time - seek_lead)
    frame_seek = sample_time - input_seek
    crop_filter = (
        "color=c=black:s=1440x1080:r=24000/1001[bg];"
        "[bg][0:s:0]overlay,"
        "scale=iw*2:ih*2:flags=lanczos,format=gray,eq=contrast=1.6:brightness=0.03"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{input_seek:.3f}",
        "-i",
        str(source),
        "-ss",
        f"{frame_seek:.3f}",
        "-filter_complex",
        crop_filter,
        "-frames:v",
        "1",
        "-update",
        "1",
        str(image_path),
    ]
    subprocess.run(cmd, check=True)


def ocr_image(image_path: Path, tesseract: str) -> str:
    cmd = [tesseract, str(image_path), "stdout", "-l", "eng", "--psm", "6"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return clean_ocr_text(result.stdout)


def clean_ocr_text(value: str) -> str:
    lines = []
    for line in value.replace("\r", "").splitlines():
        line = " ".join(line.strip().split())
        if not line:
            continue
        line = line.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        line = line.replace("|", "I")
        lines.append(line)
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def is_ocr_noise(value: str, sample_time: float) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z']+", value)
    alpha_count = sum(ch.isalpha() for ch in value)
    if alpha_count < 3:
        return True
    if sample_time < 60 and len(words) < 3:
        return True
    return False


def seconds_to_srt_time(value: float) -> str:
    milliseconds_total = max(0, int(round(value * 1000)))
    seconds_total, milliseconds = divmod(milliseconds_total, 1000)
    minutes_total, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def format_duration(seconds_value: float) -> str:
    seconds_total = max(0, int(round(seconds_value)))
    hours, remainder = divmod(seconds_total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def read_srt(path: Path) -> list[SrtCue]:
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", text.strip())
    cues: list[SrtCue] = []
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) < 2:
            continue
        timing_index = 1 if lines[0].strip().isdigit() else 0
        if timing_index >= len(lines) or "-->" not in lines[timing_index]:
            continue
        start, end = [part.strip().split()[0] for part in lines[timing_index].split("-->", 1)]
        cue_text = "\n".join(lines[timing_index + 1 :]).strip()
        if cue_text:
            cues.append(SrtCue(index=len(cues) + 1, start=start, end=end, text=cue_text))
    return cues


def write_srt(path: Path, cues: list[SrtCue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.extend([str(index), f"{cue.start} --> {cue.end}", cue.text.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def read_glossary(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"style": {}, "terms": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_cues(cues: list[SrtCue], chunk_size: int) -> Iterable[list[SrtCue]]:
    for start in range(0, len(cues), chunk_size):
        yield cues[start : start + chunk_size]


def build_translation_prompt_record(
    episode_id: str,
    chunk_index: int,
    cue_chunk: list[SrtCue],
    all_english_cues: list[SrtCue],
    reference_cues: list[SrtCue],
    glossary: dict[str, object],
) -> dict[str, object]:
    glossary_terms = glossary.get("terms", {})
    speaker_labels = glossary.get("speaker_labels", {})
    unit_conversions = glossary.get("unit_conversions", {})
    style = glossary.get("style", {})
    reference_text = proportional_reference_text(cue_chunk, all_english_cues, reference_cues)
    cue_payload = [
        {
            "index": cue.index,
            "start": cue.start,
            "end": cue.end,
            "text": cue.text,
        }
        for cue in cue_chunk
    ]
    prompt = (
        "You are translating subtitles for the 1985 TV series Robotech into neutral Latin American Spanish.\n"
        "Return ONLY valid JSON. Do not include markdown, comments, or explanations.\n"
        "Output schema: {\"translations\":[{\"index\":1,\"text\":\"...\"}]}\n\n"
        "Rules:\n"
        "- Translate the English subtitle text into natural Latin American Spanish.\n"
        "- Preserve each input index exactly once.\n"
        "- Keep timing out of the output; only return translated text.\n"
        "- Return strict JSON only; do not use markdown, code fences, comments, or raw backslash characters in text values.\n"
        "- Do not escape square brackets; write labels as normal text like [NARRADOR].\n"
        "- Keep subtitles concise and readable.\n"
        "- Translate every bracketed speaker, role, source, or status label into Spanish while preserving bracket format.\n"
        "- Bracket examples: [NARRATOR] -> [NARRADOR], [COMMANDER] -> [COMANDANTE], [ON RADIO] -> [POR RADIO].\n"
        "- If a bracketed label is not in the provided examples, translate it naturally instead of leaving it in English.\n"
        "- Convert imperial and US customary measurements into natural metric expressions when possible.\n"
        "- Use rounded, viewer-friendly metric values, for example three quarters of a mile becomes casi 1.2 kilometros.\n"
        "- Avoid literal calques, repeated roots, and redundant near-synonyms caused by word-for-word translation.\n"
        "- Prefer idiomatic TV subtitle wording even when it changes the sentence structure.\n"
        "- Example of the style principle: \"phenomenal event\" should be natural Spanish like \"evento fenomenal\", not a repeated-root phrase.\n"
        "- Use the glossary for Robotech terminology and names.\n"
        "- The old Spanish reference is for tone/terminology only; do not copy it blindly.\n\n"
        f"Episode: {episode_id} - {episode_display_title(episode_id)}\n"
        f"Style: {json.dumps(style, ensure_ascii=False)}\n"
        f"Glossary: {json.dumps(glossary_terms, ensure_ascii=False)}\n"
        f"Speaker labels: {json.dumps(speaker_labels, ensure_ascii=False)}\n"
        f"Unit conversions: {json.dumps(unit_conversions, ensure_ascii=False)}\n"
        f"Old Spanish reference nearby: {reference_text}\n\n"
        f"Input cues JSON: {json.dumps(cue_payload, ensure_ascii=False)}"
    )
    return {
        "episode_id": episode_id,
        "episode_title": episode_display_title(episode_id),
        "chunk_index": chunk_index,
        "cue_indexes": [cue.index for cue in cue_chunk],
        "prompt": prompt,
    }


def proportional_reference_text(cue_chunk: list[SrtCue], all_english_cues: list[SrtCue], reference_cues: list[SrtCue]) -> str:
    if not cue_chunk or not all_english_cues or not reference_cues:
        return ""
    start_ratio = (cue_chunk[0].index - 1) / max(1, len(all_english_cues) - 1)
    end_ratio = (cue_chunk[-1].index - 1) / max(1, len(all_english_cues) - 1)
    ref_start = max(0, int(start_ratio * len(reference_cues)) - 2)
    ref_end = min(len(reference_cues), int(end_ratio * len(reference_cues)) + 3)
    return " / ".join(cue.text.replace("\n", " ") for cue in reference_cues[ref_start:ref_end])


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise SystemExit(f"Expected JSON object per line in {path}")
            records.append(record)
    return records


def call_ollama(ollama_url: str, model: str, prompt: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }
    request = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not call Ollama at {ollama_url}. Is `ollama serve` running?\n{exc}") from exc
    return str(data.get("response", "")).strip()


def hf_translation_command(
    *,
    llm_python: str,
    prompt_jsonl: Path,
    english_srt: Path,
    output_srt: Path,
    response_jsonl: Path,
    model_config: Path,
    model_cache: Path,
    offline: bool,
    limit_chunks: int | None,
    retries: int,
    overwrite: bool,
) -> list[str]:
    if limit_chunks is not None:
        output_srt = limited_hf_path(output_srt, limit_chunks)
        response_jsonl = response_jsonl.with_name(f"{response_jsonl.stem}_limit{limit_chunks}{response_jsonl.suffix}")
    cmd = [
        llm_python,
        "scripts/run_subtitle_translation_hf.py",
        "--prompt-jsonl",
        str(prompt_jsonl),
        "--english-srt",
        str(english_srt),
        "--output-srt",
        str(output_srt),
        "--response-jsonl",
        str(response_jsonl),
        "--model-config",
        str(model_config),
        "--model-cache",
        str(model_cache),
        "--retries",
        str(retries),
    ]
    if offline:
        cmd.append("--offline")
    if limit_chunks is not None:
        cmd.extend(["--limit-chunks", str(limit_chunks)])
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def limited_hf_path(path: Path, limit_chunks: int | None) -> Path:
    if limit_chunks is None:
        return path
    return path.with_name(f"{path.stem}_limit{limit_chunks}{path.suffix}")


def parse_translation_response(raw_response: str) -> list[dict[str, object]]:
    parsed = extract_json_object(raw_response)
    translations = parsed.get("translations")
    if not isinstance(translations, list):
        raise SystemExit(f"Model response did not contain translations list:\n{raw_response}")
    clean = []
    for item in translations:
        if not isinstance(item, dict) or "index" not in item or "text" not in item:
            raise SystemExit(f"Invalid translation item in model response:\n{raw_response}")
        clean.append({"index": int(item["index"]), "text": str(item["text"]).strip()})
    return clean


def extract_json_object(value: str) -> dict[str, object]:
    candidate = value.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise SystemExit(f"Model response was not JSON:\n{value}") from first_error
        candidate = candidate[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            repaired = repair_common_model_json(candidate)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as final_error:
                raise SystemExit(f"Model response JSON could not be parsed:\n{value}") from final_error
    if not isinstance(parsed, dict):
        raise SystemExit(f"Model response JSON was not an object:\n{value}")
    return parsed


def repair_common_model_json(value: str) -> str:
    # LLMs sometimes emit invalid JSON escapes such as \[NARRADOR\], or
    # malformed line breaks such as \uno when they meant \nuno.
    repaired = re.sub(r"\\u(?![0-9A-Fa-f]{4})", r"\\nu", value)
    # Another common subtitle-specific slip is closing a sound-effect label
    # after the JSON string quote: "--[aplausos"] instead of "--[aplausos]".
    repaired = re.sub(r"\[([^\]\"]{1,48})\"\]", r'[\1]"', repaired)
    # Gemma can also produce quoted phrases like \"caballos de Troya,\"}
    # where the comma lands before the escaped quote and the JSON string is
    # never closed. Move the comma after the escaped quote and close the value.
    repaired = re.sub(r',\\"(?=[}\]])', r'\\","', repaired)
    # JSON only allows escapes like \n, \t, \\, \", or \uXXXX. Double
    # any other backslash and let json.loads validate the repaired result.
    return re.sub(r'\\(?!["\\/bfnrt]|u[0-9A-Fa-f]{4})', r"\\\\", repaired)


def model_safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def write_clean_spanish_srt(source: Path, destination: Path) -> int:
    text = source.read_text(encoding="cp1252", errors="replace")
    fields: list[str] | None = None
    in_events = False
    entries: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        if line.strip() == "[Events]":
            in_events = True
            fields = None
            continue
        if in_events and line.startswith("["):
            in_events = False
        if in_events and line.startswith("Format:") and fields is None:
            fields = [item.strip() for item in line.split(":", 1)[1].split(",")]
            continue
        if not in_events or not line.startswith("Dialogue:") or fields is None:
            continue
        values = line.split(":", 1)[1].strip().split(",", len(fields) - 1)
        if len(values) != len(fields):
            continue
        event = dict(zip(fields, values))
        subtitle_text = clean_ssa_text(event.get("Text", ""))
        if not subtitle_text or is_unwanted_spanish_subtitle(subtitle_text):
            continue
        entries.append((ssa_time_to_srt(event["Start"]), ssa_time_to_srt(event["End"]), subtitle_text))

    destination.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, (start, end, subtitle_text) in enumerate(entries, start=1):
        lines.extend([str(index), f"{start} --> {end}", subtitle_text, ""])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return len(entries)


def clean_ssa_text(value: str) -> str:
    value = re.sub(r"\{[^}]*\}", "", value)
    value = value.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    value = value.replace("\r", "")
    value = "\n".join(" ".join(part.split()) for part in value.split("\n"))
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def is_unwanted_spanish_subtitle(value: str) -> bool:
    normalized = value.casefold()
    return any(
        marker in normalized
        for marker in (
            "traducción y subtitulado",
            "traduccion y subtitulado",
            "adrián bergonzi",
            "adrian bergonzi",
            "scriptmania",
            "http://",
            "https://",
        )
    )


def ssa_time_to_srt(value: str) -> str:
    hours_text, minutes_text, seconds_text = value.strip().split(":")
    seconds, centiseconds = seconds_text.split(".")
    milliseconds = int(centiseconds.ljust(3, "0")[:3])
    return f"{int(hours_text):02d}:{int(minutes_text):02d}:{int(seconds):02d},{milliseconds:03d}"


def decide_existing_outputs(episode_id: str, planned_outputs: dict[str, Path], if_exists: str) -> str:
    existing = {variant: path for variant, path in planned_outputs.items() if path.exists()}
    if not existing:
        return "overwrite"
    if if_exists == "overwrite":
        return "overwrite"
    if if_exists == "skip":
        return "skip"

    print(f"{episode_id} already has final output file(s):")
    for variant, path in existing.items():
        print(f"  - {variant}: {path}")
    while True:
        try:
            answer = input("Overwrite this episode, skip/continue to next, or abort? [o/s/a]: ").strip().lower()
        except EOFError:
            print("No interactive input available; skipping existing episode.")
            return "skip"
        if answer in {"o", "overwrite"}:
            return "overwrite"
        if answer in {"s", "skip", "c", "continue", "next"}:
            return "skip"
        if answer in {"a", "abort", "q", "quit"}:
            raise SystemExit("Aborted by user")


def shared_end_credit_path(credit_dir: Path, credit_source: Path, target_video: Path) -> Path:
    target = video_encode_info(target_video)
    rate = target.rate_text.replace("/", "_")
    profile = (target.profile or "auto").lower().replace(" ", "_")
    level = (target.level or "auto").replace(".", "_")
    name = f"{credit_source.stem}__{target.width}x{target.height}__{rate}__{profile}__{level}.mkv"
    return credit_dir / name


def shared_opening_path(opening_dir: Path, opening_source: Path, target_video: Path) -> Path:
    target = video_encode_info(target_video)
    rate = target.rate_text.replace("/", "_")
    profile = (target.profile or "auto").lower().replace(" ", "_")
    level = (target.level or "auto").replace(".", "_")
    name = f"{opening_source.stem}__{target.width}x{target.height}__{rate}__{profile}__{level}.mkv"
    return opening_dir / name


def opening_segment_cmd(
    assets: OpeningAssets,
    target_video: Path,
    output_path: Path,
    restored_ac3_bitrate: str,
    eng_ac3_bitrate: str,
    stereo_ac3_bitrate: str,
) -> list[str]:
    target = video_encode_info(target_video)
    video_filter = f"[0:v]fps={target.rate_text},scale={target.width}:{target.height},format=yuv420p[v]"
    audio_filter = (
        "[2:a]aresample=48000,asplit=2[spa1stereo][spa1for51];"
        "[spa1for51]pan=5.1(side)|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=0*c0|SL=0*c0|SR=0*c1[spa1_51];"
        "[3:a]pan=stereo|c0=0.707*c0+0.5*c2+0.5*c4|c1=0.707*c1+0.5*c2+0.5*c5[spa2_stereo]"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(assets.video),
        "-i",
        str(assets.english_51),
        "-i",
        str(assets.spanish_original_stereo),
        "-i",
        str(assets.generation_spanish_51),
        "-filter_complex",
        f"{video_filter};{audio_filter}",
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-map",
        "[spa1_51]",
        "-map",
        "[spa1stereo]",
        "-map",
        "[spa2_stereo]",
        "-map_chapters",
        "-1",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-profile:v",
        x264_profile_arg(target.profile),
    ]
    if target.level:
        cmd.extend(["-level:v", target.level])
    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-r",
            target.rate_text,
            "-c:a:0",
            "ac3",
            "-b:a:0",
            eng_ac3_bitrate,
            "-c:a:1",
            "ac3",
            "-b:a:1",
            restored_ac3_bitrate,
            "-c:a:2",
            "ac3",
            "-b:a:2",
            stereo_ac3_bitrate,
            "-c:a:3",
            "ac3",
            "-b:a:3",
            stereo_ac3_bitrate,
            "-metadata",
            "title=Robotech - Restored Unified Opening",
            "-metadata:s:v:0",
            "language=eng",
            "-metadata:s:v:0",
            "title=Restored Unified Opening",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:0",
            "title=English Original Opening 5.1",
            "-metadata:s:a:1",
            "language=spa",
            "-metadata:s:a:1",
            "title=Spanish Original Latin American Opening 5.1",
            "-metadata:s:a:2",
            "language=spa",
            "-metadata:s:a:2",
            "title=Spanish Original Latin American Opening Stereo",
            "-metadata:s:a:3",
            "language=spa",
            "-metadata:s:a:3",
            "title=Spanish Redubbing Opening Stereo",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
            "-disposition:a:2",
            "0",
            "-disposition:a:3",
            "0",
            str(output_path),
        ]
    )
    return cmd


def end_credit_segment_cmd(
    credit_source: Path,
    target_video: Path,
    output_path: Path,
    restored_ac3_bitrate: str,
    eng_ac3_bitrate: str,
    stereo_ac3_bitrate: str,
) -> list[str]:
    target = video_encode_info(target_video)
    video_filter = f"[0:v]fps={target.rate_text},scale={target.width}:{target.height},format=yuv420p[v]"
    audio_filter = (
        "[0:a]aresample=48000,asplit=4[a51engsrc][a51spasrc][aSpa1][aSpa2];"
        "[a51engsrc]pan=5.1(side)|FL=c0|FR=c1|FC=0*c0|LFE=0*c0|SL=0*c0|SR=0*c1[aEng51];"
        "[a51spasrc]pan=5.1(side)|FL=c0|FR=c1|FC=0*c0|LFE=0*c0|SL=0*c0|SR=0*c1[aSpa51]"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(credit_source),
        "-filter_complex",
        f"{video_filter};{audio_filter}",
        "-map",
        "[v]",
        "-map",
        "[aEng51]",
        "-map",
        "[aSpa51]",
        "-map",
        "[aSpa1]",
        "-map",
        "[aSpa2]",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-profile:v",
        x264_profile_arg(target.profile),
    ]
    if target.level:
        cmd.extend(["-level:v", target.level])
    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-r",
            target.rate_text,
            "-c:a:0",
            "ac3",
            "-b:a:0",
            eng_ac3_bitrate,
            "-c:a:1",
            "ac3",
            "-b:a:1",
            restored_ac3_bitrate,
            "-c:a:2",
            "ac3",
            "-b:a:2",
            stereo_ac3_bitrate,
            "-c:a:3",
            "ac3",
            "-b:a:3",
            stereo_ac3_bitrate,
            "-metadata",
            "title=Robotech - Restored End Credits",
            "-metadata:s:v:0",
            "language=eng",
            "-metadata:s:v:0",
            "title=Restored End Credits",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:0",
            "title=English Original 5.1",
            "-metadata:s:a:1",
            "language=spa",
            "-metadata:s:a:1",
            "title=Spanish Restored Original Dub 5.1",
            "-metadata:s:a:2",
            "language=spa",
            "-metadata:s:a:2",
            "title=Spanish Original Dub Restored Stereo",
            "-metadata:s:a:3",
            "language=spa",
            "-metadata:s:a:3",
            "title=Spanish Redubbing Original Stereo",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
            "-disposition:a:2",
            "0",
            "-disposition:a:3",
            "0",
            str(output_path),
        ]
    )
    return cmd


def last_frame_luma_stats(path: Path, black_avg: float, black_max: int) -> dict[str, float | int | bool | str]:
    info = video_encode_info(path)
    frame_size = info.width * info.height
    if frame_size <= 0:
        raise SystemExit(f"Invalid video size for luma analysis: {path}")
    stdout = b""
    for tail_seconds in ("0.5", "2", "5"):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-sseof",
            f"-{tail_seconds}",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            "format=gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
        stdout = result.stdout
        if len(stdout) >= frame_size:
            break
    if len(stdout) < frame_size:
        raise SystemExit(f"Could not decode a last video frame from {path}")
    frame = stdout[-frame_size:]
    avg = sum(frame) / frame_size
    max_value = max(frame)
    return {
        "width": info.width,
        "height": info.height,
        "rate": info.rate_text,
        "avg_luma": round(avg, 4),
        "max_luma": max_value,
        "black_avg_threshold": black_avg,
        "black_max_threshold": black_max,
        "is_black": avg <= black_avg and max_value <= black_max,
    }


def end_fade_segment_cmd(
    source_segment: Path,
    output_path: Path,
    fade_frames: int,
    restored_ac3_bitrate: str,
    eng_ac3_bitrate: str,
    stereo_ac3_bitrate: str,
) -> list[str]:
    if fade_frames <= 0:
        raise SystemExit("--fade-frames must be greater than zero")
    target = video_encode_info(source_segment)
    total_frames = fade_frames + 1
    internal_frames = total_frames + 1
    duration = float(Fraction(total_frames, 1) / target.rate)
    duration_text = f"{duration:.6f}"
    fps_value = float(target.rate)
    video_filter = (
        "[0:v]reverse,trim=end_frame=1,setpts=PTS-STARTPTS,"
        f"scale={target.width}:{target.height},format=yuv420p,"
        f"loop=loop={internal_frames - 1}:size=1:start=0,"
        f"setpts=N/{fps_value:.6f}/TB,"
        f"fade=t=out:st=0:d={duration_text}:color=black,"
        f"trim=start_frame=1:end_frame={internal_frames},"
        f"setpts=N/{fps_value:.6f}/TB[v]"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-sseof",
        "-5",
        "-i",
        str(source_segment),
        "-f",
        "lavfi",
        "-t",
        duration_text,
        "-i",
        "anullsrc=channel_layout=5.1(side):sample_rate=48000",
        "-f",
        "lavfi",
        "-t",
        duration_text,
        "-i",
        "anullsrc=channel_layout=5.1(side):sample_rate=48000",
        "-f",
        "lavfi",
        "-t",
        duration_text,
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-f",
        "lavfi",
        "-t",
        duration_text,
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter_complex",
        video_filter,
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-map",
        "2:a:0",
        "-map",
        "3:a:0",
        "-map",
        "4:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-profile:v",
        x264_profile_arg(target.profile),
    ]
    if target.level:
        cmd.extend(["-level:v", target.level])
    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-r",
            target.rate_text,
            "-c:a:0",
            "ac3",
            "-b:a:0",
            eng_ac3_bitrate,
            "-c:a:1",
            "ac3",
            "-b:a:1",
            restored_ac3_bitrate,
            "-c:a:2",
            "ac3",
            "-b:a:2",
            stereo_ac3_bitrate,
            "-c:a:3",
            "ac3",
            "-b:a:3",
            stereo_ac3_bitrate,
            "-metadata",
            "title=Robotech - Generated End Fade Adjustment",
            "-metadata:s:v:0",
            "language=eng",
            "-metadata:s:v:0",
            "title=Generated End Fade to Black",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:0",
            "title=English Original 5.1 Silence",
            "-metadata:s:a:1",
            "language=spa",
            "-metadata:s:a:1",
            "title=Spanish Restored Original Dub 5.1 Silence",
            "-metadata:s:a:2",
            "language=spa",
            "-metadata:s:a:2",
            "title=Spanish Original Dub Restored Stereo Silence",
            "-metadata:s:a:3",
            "language=spa",
            "-metadata:s:a:3",
            "title=Spanish Redubbing Original Stereo Silence",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
            "-disposition:a:2",
            "0",
            "-disposition:a:3",
            "0",
            str(output_path),
        ]
    )
    return cmd


def spa2_tail_extension_segment_cmd(
    source_segment: Path,
    tail_audio: Path,
    output_path: Path,
    duration_seconds: float,
    fade_frames: int,
    restored_ac3_bitrate: str,
    eng_ac3_bitrate: str,
    stereo_ac3_bitrate: str,
    old_spa1_tail_gain_db: float,
) -> list[str]:
    if duration_seconds <= 0:
        raise SystemExit(f"Invalid spa2 tail extension duration for {source_segment}: {duration_seconds}")
    target = video_encode_info(source_segment)
    fps_value = float(target.rate)
    frame_count = max(int(duration_seconds * fps_value + 0.999999) + 2, 2)
    fade_duration = 0.0
    if fade_frames > 0:
        fade_duration = float(Fraction(fade_frames, 1) / target.rate)
    video_filter = (
        "[0:v]reverse,trim=end_frame=1,setpts=PTS-STARTPTS,"
        f"scale={target.width}:{target.height},format=yuv420p,"
        f"loop=loop={frame_count - 1}:size=1:start=0,"
        f"setpts=N/{fps_value:.6f}/TB,"
        f"fade=t=out:st=0:d={fade_duration:.6f}:color=black,"
        f"trim=duration={duration_seconds:.6f},setpts=N/{fps_value:.6f}/TB[v];"
        f"[4:a]aresample=48000,apad,atrim=0:{duration_seconds:.6f},asetpts=N/SR/TB,"
        "asplit=2[spa_tail_old_raw][spa_tail_new];"
        f"[spa_tail_old_raw]volume={old_spa1_tail_gain_db:g}dB[spa_tail_old]"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-sseof",
        "-5",
        "-i",
        str(source_segment),
        "-f",
        "lavfi",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        "anullsrc=channel_layout=5.1(side):sample_rate=48000",
        "-f",
        "lavfi",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        "anullsrc=channel_layout=5.1(side):sample_rate=48000",
        "-f",
        "lavfi",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-i",
        str(tail_audio),
        "-filter_complex",
        video_filter,
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-map",
        "2:a:0",
        "-map",
        "[spa_tail_old]",
        "-map",
        "[spa_tail_new]",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-profile:v",
        x264_profile_arg(target.profile),
    ]
    if target.level:
        cmd.extend(["-level:v", target.level])
    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-r",
            target.rate_text,
            "-c:a:0",
            "ac3",
            "-b:a:0",
            eng_ac3_bitrate,
            "-c:a:1",
            "ac3",
            "-b:a:1",
            restored_ac3_bitrate,
            "-c:a:2",
            "ac3",
            "-b:a:2",
            stereo_ac3_bitrate,
            "-c:a:3",
            "ac3",
            "-b:a:3",
            stereo_ac3_bitrate,
            "-metadata",
            "title=Robotech - Spanish 2 Tail Hold",
            "-metadata:s:v:0",
            "language=eng",
            "-metadata:s:v:0",
            "title=Generated Black Hold for Spanish 2 Tail",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:0",
            "title=English Original 5.1 Silence",
            "-metadata:s:a:1",
            "language=spa",
            "-metadata:s:a:1",
            "title=Spanish Restored Original Dub 5.1 Silence",
            "-metadata:s:a:2",
            "language=spa",
            "-metadata:s:a:2",
            "title=Spanish Original Dub Restored Stereo Tail",
            "-metadata:s:a:3",
            "language=spa",
            "-metadata:s:a:3",
            "title=Spanish Redubbing Tail",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
            "-disposition:a:2",
            "0",
            "-disposition:a:3",
            "0",
            str(output_path),
        ]
    )
    return cmd


def plan_end_fade_adjustment(
    variant: str,
    source_segment: Path,
    output_dir: Path,
    fade_frames: int,
    black_avg: float,
    black_max: int,
    restored_ac3_bitrate: str,
    eng_ac3_bitrate: str,
    stereo_ac3_bitrate: str,
) -> EndFadeAdjustment:
    stats = last_frame_luma_stats(source_segment, black_avg, black_max)
    output = output_dir / f"{source_segment.stem}_end_fade_to_black_{fade_frames}f_plus_black.mkv"
    command = end_fade_segment_cmd(
        source_segment,
        output,
        fade_frames,
        restored_ac3_bitrate=restored_ac3_bitrate,
        eng_ac3_bitrate=eng_ac3_bitrate,
        stereo_ac3_bitrate=stereo_ac3_bitrate,
    )
    return EndFadeAdjustment(
        variant=variant,
        source=source_segment,
        output=output,
        needed=not bool(stats["is_black"]),
        stats=stats,
        command=command,
    )


def plan_spa2_tail_extension_adjustment(
    variant: str,
    source_segment: Path,
    output_dir: Path,
    extension: Spa2TailExtension,
    fade_frames: int,
    restored_ac3_bitrate: str,
    eng_ac3_bitrate: str,
    stereo_ac3_bitrate: str,
) -> Spa2TailExtensionAdjustment:
    output = output_dir / f"{source_segment.stem}_{extension.extension_id}_black_hold.mkv"
    command = spa2_tail_extension_segment_cmd(
        source_segment,
        extension.audio_path,
        output,
        extension.duration_seconds,
        fade_frames,
        restored_ac3_bitrate=restored_ac3_bitrate,
        eng_ac3_bitrate=eng_ac3_bitrate,
        stereo_ac3_bitrate=stereo_ac3_bitrate,
        old_spa1_tail_gain_db=extension.old_spa1_tail_gain_db,
    )
    return Spa2TailExtensionAdjustment(
        variant=variant,
        source=source_segment,
        output=output,
        extension=extension,
        command=command,
    )


def tail_trim_segment_cmd(source_segment: Path, output_path: Path, output_duration_seconds: float) -> list[str]:
    if output_duration_seconds <= 0:
        raise SystemExit(f"Invalid tail-trim duration for {source_segment}: {output_duration_seconds}")
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source_segment),
        "-map",
        "0",
        "-t",
        f"{output_duration_seconds:.6f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]


def plan_tail_trim_adjustment(
    episode_id: str,
    variant: str,
    source_segment: Path,
    output_dir: Path,
) -> TailTrimAdjustment | None:
    trim_seconds = EPISODE_VIDEO_TAIL_TRIMS_SECONDS.get(episode_id)
    if trim_seconds is None or trim_seconds <= 0:
        return None
    source_duration = media_file_duration(source_segment)
    output_duration = max(source_duration - trim_seconds, 0.0)
    output = output_dir / f"{source_segment.stem}_trim_tail_{trim_seconds:g}s.mkv"
    command = tail_trim_segment_cmd(source_segment, output, output_duration)
    return TailTrimAdjustment(
        variant=variant,
        source=source_segment,
        output=output,
        needed=output_duration > 0 and trim_seconds < source_duration,
        trim_seconds=trim_seconds,
        source_duration_seconds=round(source_duration, 6),
        output_duration_seconds=round(output_duration, 6),
        command=command,
        description=(
            f"Trim {trim_seconds:g}s black tail from {episode_id} episode-only mux "
            "before appending restored end credits."
        ),
    )


def write_concat_list(path: Path, segments: list[Path]) -> None:
    lines = []
    for segment in segments:
        escaped = str(segment.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chapter_metadata_file(
    path: Path,
    concat_segments: list[Path],
    include_opening: bool,
    include_end_credit: bool,
    episode_summary_start_seconds: float | None = None,
    reference_episode_chapters: list[ReferenceEpisodeChapter] | None = None,
) -> list[dict[str, object]]:
    durations = [media_file_duration(segment) for segment in concat_segments]
    chapter_ranges: list[tuple[str, int, int]] = []
    episode_start_index = 0
    episode_end_index = len(concat_segments)
    if include_opening and concat_segments:
        chapter_ranges.append(("Opening Credits", 0, 1))
        episode_start_index = 1
    if include_end_credit and len(concat_segments) > episode_start_index:
        episode_end_index = len(concat_segments) - 1
    if episode_end_index > episode_start_index:
        chapter_ranges.append(("Episode", episode_start_index, episode_end_index))
    if include_end_credit and concat_segments:
        chapter_ranges.append(("End Credits", len(concat_segments) - 1, len(concat_segments)))

    chapters: list[dict[str, object]] = []
    lines = [";FFMETADATA1"]
    for title, start_index, end_index in chapter_ranges:
        start_seconds = sum(durations[:start_index])
        end_seconds = sum(durations[:end_index])
        if title == "Episode":
            split_points: list[tuple[str, float]] = []
            for reference_chapter in reference_episode_chapters or []:
                absolute_start = start_seconds + reference_chapter.start_seconds
                if start_seconds <= absolute_start < end_seconds - 0.05:
                    split_points.append((reference_chapter.title, absolute_start))
            if not split_points or split_points[0][1] > start_seconds + 0.05:
                split_points.insert(0, ("Episode", start_seconds))
            if episode_summary_start_seconds is not None:
                summary_start = start_seconds + episode_summary_start_seconds
                if start_seconds < summary_start < end_seconds - 0.05:
                    split_points.append(("Next Episode Summary", summary_start))
            split_points.sort(key=lambda item: item[1])
            deduped: list[tuple[str, float]] = []
            for split_title, split_start in split_points:
                if deduped and abs(deduped[-1][1] - split_start) < 0.05:
                    if split_title == "Next Episode Summary":
                        deduped[-1] = (split_title, deduped[-1][1])
                    continue
                deduped.append((split_title, split_start))
            chapter_items = []
            for index, (split_title, split_start) in enumerate(deduped):
                split_end = deduped[index + 1][1] if index + 1 < len(deduped) else end_seconds
                if split_end - split_start >= 0.05:
                    chapter_items.append((split_title, split_start, split_end))
        else:
            chapter_items = [(title, start_seconds, end_seconds)]
        for chapter_title, chapter_start, chapter_end in chapter_items:
            start_ms = int(round(chapter_start * 1000))
            end_ms = int(round(chapter_end * 1000))
            lines.extend(
                [
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={start_ms}",
                    f"END={end_ms}",
                    f"title={chapter_title}",
                ]
            )
            chapters.append(
                {
                    "title": chapter_title,
                    "segments": [str(segment) for segment in concat_segments[start_index:end_index]],
                    "start_seconds": round(chapter_start, 3),
                    "end_seconds": round(chapter_end, 3),
                    "duration_seconds": round(chapter_end - chapter_start, 3),
                }
            )
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return chapters


def media_file_duration(path: Path) -> float:
    duration = duration_seconds(ffprobe_full(path))
    if duration is None:
        raise SystemExit(f"Could not read duration for chapter metadata: {path}")
    return float(duration)


def final_concat_cmd(episode_id: str, variant: str, concat_list: Path, output_path: Path, chapters: Path | None = None) -> list[str]:
    variant_title = {
        "ai_remaster": "AI Remaster",
        "remaster": "Remaster",
        "remaster_49fps": "Remaster 49fps",
    }[variant]
    episode_title = episode_display_title(episode_id)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
    ]
    if chapters is not None:
        cmd.extend(["-i", str(chapters)])
    cmd.extend(
        [
        "-map",
        "0",
        "-c",
        "copy",
        ]
    )
    if chapters is not None:
        cmd.extend(["-map_chapters", "1"])
    cmd.extend(
        [
        "-metadata",
        f"title=Robotech - {episode_id} - {episode_title} - {variant_title} - Restored Audio",
        "-metadata",
        f"episode_id={episode_id}",
        "-metadata",
        f"episode={episode_title}",
        "-metadata:s:v:0",
        "language=eng",
        "-metadata:s:v:0",
        f"title={variant_title}",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:0",
        "title=English Original 5.1",
        "-metadata:s:a:1",
        "language=spa",
        "-metadata:s:a:1",
        "title=Spanish Restored Original Dub 5.1",
        "-metadata:s:a:2",
        "language=spa",
        "-metadata:s:a:2",
        "title=Spanish Original Dub Restored Stereo",
        "-metadata:s:a:3",
        "language=spa",
        "-metadata:s:a:3",
        "title=Spanish Redubbing Original Stereo",
        "-disposition:a:0",
        "0",
        "-disposition:a:1",
        "default",
        "-disposition:a:2",
        "0",
        "-disposition:a:3",
        "0",
        str(output_path),
        ]
    )
    return cmd


def final_mux_cmd(
    episode_id: str,
    variant: str,
    video_path: Path,
    output_path: Path,
    restored_ac3: Path,
    eng_ac3: Path,
    spa1_ac3: Path,
    spa2_ac3: Path,
) -> list[str]:
    variant_title = {
        "ai_remaster": "AI Remaster",
        "remaster": "Remaster",
        "remaster_49fps": "Remaster 49fps",
    }[variant]
    episode_title = episode_display_title(episode_id)
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(eng_ac3),
        "-i",
        str(restored_ac3),
        "-i",
        str(spa1_ac3),
        "-i",
        str(spa2_ac3),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:a:0",
        "-map",
        "3:a:0",
        "-map",
        "4:a:0",
        "-map_chapters",
        "0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-metadata",
        f"title=Robotech - {episode_id} - {episode_title} - {variant_title} - Restored Audio",
        "-metadata",
        f"episode_id={episode_id}",
        "-metadata",
        f"episode={episode_title}",
        "-metadata:s:v:0",
        "language=eng",
        "-metadata:s:v:0",
        f"title={variant_title}",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:0",
        "title=English Original 5.1",
        "-metadata:s:a:1",
        "language=spa",
        "-metadata:s:a:1",
        "title=Spanish Restored Original Dub 5.1",
        "-metadata:s:a:2",
        "language=spa",
        "-metadata:s:a:2",
        "title=Spanish Original Dub Restored Stereo",
        "-metadata:s:a:3",
        "language=spa",
        "-metadata:s:a:3",
        "title=Spanish Redubbing Original Stereo",
        "-disposition:a:0",
        "0",
        "-disposition:a:1",
        "default",
        "-disposition:a:2",
        "0",
        "-disposition:a:3",
        "0",
        str(output_path),
    ]


def subtitle_outputs(subtitle_dir: Path, episode_id: str) -> dict[str, Path]:
    episode_dir = subtitle_dir / episode_id
    translated_spanish = episode_dir / f"{episode_id}_spanish_translated.srt"
    cleaned_spanish = episode_dir / f"{episode_id}_spanish_clean.srt"
    english_srt = episode_dir / f"{episode_id}_english_clean.srt"
    outputs = {
        "english_srt": english_srt,
        "spanish_srt": translated_spanish if translated_spanish.is_file() else cleaned_spanish,
    }
    return outputs


def discover_srt_subtitles(subtitle_dir: Path, episode_id: str) -> list[SubtitleTrack]:
    episode_dir = subtitle_dir / episode_id
    tracks = []
    for stem_suffix, (language, title, sidecar_suffix) in SUBTITLE_NAME_SPECS.items():
        path = episode_dir / f"{episode_id}_{stem_suffix}.srt"
        if path.is_file():
            tracks.append(
                SubtitleTrack(
                    path=path,
                    language=language,
                    title=title,
                    sidecar_suffix=sidecar_suffix,
                )
            )
    return tracks


def authored_subtitles_with_prepared_segment_cues(
    subtitles: list[SubtitleTrack],
    prepared_segment: PreparedEpisodeSegment | None,
    segment_start_seconds: float | None,
    output_dir: Path,
    variant: str,
) -> list[SubtitleTrack]:
    if not subtitles or prepared_segment is None or segment_start_seconds is None:
        return subtitles
    output_dir.mkdir(parents=True, exist_ok=True)
    authored: list[SubtitleTrack] = []
    for subtitle in subtitles:
        key = subtitle_key_for_track(subtitle)
        relative_srt = prepared_segment.subtitle_sources.get(key)
        if relative_srt is None or not relative_srt.is_file():
            authored.append(subtitle)
            continue
        output = output_dir / f"{variant}_{key}.srt"
        base_cues = read_srt(subtitle.path)
        relative_cues = read_srt(relative_srt)
        shifted = [
            SrtCue(
                index=0,
                start=seconds_to_srt_time(segment_start_seconds + srt_time_to_seconds(cue.start)),
                end=seconds_to_srt_time(segment_start_seconds + srt_time_to_seconds(cue.end)),
                text=cue.text,
            )
            for cue in relative_cues
        ]
        write_srt(output, sorted(base_cues + shifted, key=lambda cue: srt_time_to_seconds(cue.start)))
        authored.append(
            SubtitleTrack(
                path=output,
                language=subtitle.language,
                title=subtitle.title,
                sidecar_suffix=subtitle.sidecar_suffix,
            )
        )
    return authored


def subtitle_key_for_track(subtitle: SubtitleTrack) -> str:
    for stem_suffix, (language, title, _sidecar_suffix) in SUBTITLE_NAME_SPECS.items():
        if subtitle.language == language and subtitle.title == title:
            return stem_suffix
    stem = subtitle.path.stem
    for stem_suffix in SUBTITLE_NAME_SPECS:
        if stem.endswith(stem_suffix):
            return stem_suffix
    return subtitle.language


def srt_time_to_seconds(value: str) -> float:
    match = re.match(r"(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)[,.](?P<ms>\d+)", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT time: {value!r}")
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(match.group("ms")[:3].ljust(3, "0")) / 1000.0
    )


def find_review_final_video(review_video_dir: Path, episode_id: str, variant: str, source_video: Path) -> Path:
    candidates = [
        review_video_dir / final_video_name(episode_id, variant, source_video),
        review_video_dir / legacy_final_video_name(episode_id, variant, source_video),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = "\n".join(str(path) for path in sorted(review_video_dir.glob("*.mkv")))
    raise SystemExit(
        f"Could not find final review video for {episode_id} {variant} in {review_video_dir}\n"
        f"Expected one of:\n" + "\n".join(str(path) for path in candidates) + f"\nFound:\n{found}"
    )


def done_export_cmd(
    episode_id: str,
    variant: str,
    source_video: Path,
    output_path: Path,
    subtitles: dict[str, Path],
    cover_path: Path | None,
) -> list[str]:
    variant_title = {
        "ai_remaster": "AI Remaster",
        "remaster": "Remaster",
        "remaster_49fps": "Remaster 49fps",
    }[variant]
    episode_title = episode_display_title(episode_id)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source_video),
        "-i",
        str(subtitles["english_srt"]),
        "-i",
        str(subtitles["spanish_srt"]),
        "-map",
        "0:v:0",
        "-map",
        "0:a",
        "-map",
        "1:0",
        "-map",
        "2:0",
        "-map_chapters",
        "0",
        "-map_metadata",
        "0",
        "-c",
        "copy",
        "-metadata",
        f"title=Robotech - {episode_id} - {episode_title} - {variant_title} - Restored Audio",
        "-metadata",
        f"episode_id={episode_id}",
        "-metadata",
        f"episode={episode_title}",
        "-metadata",
        "subtitle_tracks=English Subtitles; Spanish Subtitles",
        "-metadata:s:v:0",
        "language=eng",
        "-metadata:s:v:0",
        f"title={variant_title}",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:0",
        "title=English Original 5.1",
        "-metadata:s:a:1",
        "language=spa",
        "-metadata:s:a:1",
        "title=Spanish Restored Original Dub 5.1",
        "-metadata:s:a:2",
        "language=spa",
        "-metadata:s:a:2",
        "title=Spanish Original Dub Restored Stereo",
        "-metadata:s:a:3",
        "language=spa",
        "-metadata:s:a:3",
        "title=Spanish Redubbing Original Stereo",
        "-metadata:s:s:0",
        "language=eng",
        "-metadata:s:s:0",
        "title=English Subtitles",
        "-metadata:s:s:1",
        "language=spa",
        "-metadata:s:s:1",
        "title=Spanish Subtitles",
        "-disposition:a:0",
        "0",
        "-disposition:a:1",
        "default",
        "-disposition:a:2",
        "0",
        "-disposition:a:3",
        "0",
        "-disposition:s:0",
        "0",
        "-disposition:s:1",
        "0",
    ]
    if cover_path is not None:
        cmd.extend(
            [
                "-attach",
                str(cover_path),
                "-metadata:s:t:0",
                "mimetype=image/jpeg",
                "-metadata:s:t:0",
                "filename=cover.jpg",
            ]
        )
    cmd.append(str(output_path))
    return cmd


def review_authoring_cmd(
    source_video: Path,
    output_path: Path,
    subtitles: list[SubtitleTrack],
    cover_path: Path | None,
) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(source_video)]
    for subtitle in subtitles:
        cmd.extend(["-i", str(subtitle.path)])
    cmd.extend(["-map", "0"])
    for index, _subtitle in enumerate(subtitles, start=1):
        cmd.extend(["-map", f"{index}:0"])
    cmd.extend(["-map_chapters", "0", "-map_metadata", "0", "-c", "copy"])
    if subtitles:
        cmd.extend(["-metadata", "subtitle_tracks=" + "; ".join(track.title for track in subtitles)])
    for subtitle_index, subtitle in enumerate(subtitles):
        cmd.extend(
            [
                f"-metadata:s:s:{subtitle_index}",
                f"language={subtitle.language}",
                f"-metadata:s:s:{subtitle_index}",
                f"title={subtitle.title}",
                f"-disposition:s:{subtitle_index}",
                "0",
            ]
        )
    if cover_path is not None:
        cover_filename, cover_mime = cover_attachment_metadata(cover_path)
        cmd.extend(
            [
                "-attach",
                str(cover_path),
                "-metadata:s:t:0",
                f"mimetype={cover_mime}",
                "-metadata:s:t:0",
                f"filename={cover_filename}",
            ]
        )
    cmd.append(str(output_path))
    return cmd


def cover_art_path(cover_dir: Path, output_stem: str, cover_asset: Path) -> Path:
    if cover_asset.is_file():
        suffix = cover_asset.suffix.lower() or ".png"
        return cover_dir / f"{output_stem}{suffix}"
    return cover_dir / f"{output_stem}.jpg"


def cover_art_cmd(source_video: Path, cover_path: Path, cover_asset: Path) -> list[str]:
    if cover_asset.is_file():
        return ["cp", str(cover_asset), str(cover_path)]
    return first_frame_cover_cmd(source_video, cover_path)


def cover_attachment_metadata(cover_path: Path) -> tuple[str, str]:
    suffix = cover_path.suffix.lower()
    if suffix == ".png":
        return "cover.png", "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "cover.jpg", "image/jpeg"
    return f"cover{suffix or '.bin'}", "application/octet-stream"


def first_frame_cover_cmd(source_video: Path, cover_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-update",
        "1",
        "-q:v",
        "2",
        str(cover_path),
    ]


def write_done_export_report(path: Path, manifest: dict[str, object], elapsed_seconds: float) -> None:
    lines = [
        "# Season 1 Done Export",
        "",
        f"- Output folder: `{manifest.get('output_dir')}`",
        f"- Review name: `{manifest.get('review_name')}`",
        f"- Elapsed: `{format_elapsed(elapsed_seconds)}`",
        "- Final MKVs are written flat in the Season 1 folder.",
        "- Review MKV files are copied as-is; export does not remux or change streams.",
        "- Sidecar SRT files are copied next to each MKV for player compatibility.",
        "- Embedded subtitle tracks in the review MKVs are off by default.",
        "",
        "## Output Videos",
        "",
    ]
    episodes = manifest.get("episodes", {})
    if isinstance(episodes, dict):
        for episode_id, variants in episodes.items():
            lines.append(f"### {episode_id} - {episode_display_title(episode_id)}")
            for variant, output in dict(variants).items():
                lines.append(f"- {variant}: `{output}`")
            lines.append("")
    warnings = list(manifest.get("warnings", []))
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.extend(["## Commands", ""])
    append_command_blocks(lines, list(manifest.get("commands", [])))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_done_export_summary(
    out_dir: Path,
    exported: dict[str, dict[str, str]],
    skipped: dict[str, list[str]],
    warnings: list[str],
    elapsed_seconds: float,
) -> None:
    count = sum(len(variants) for variants in exported.values())
    print("")
    print("Done export complete")
    print(f"Output folder: {out_dir}")
    print(f"Videos exported: {count}")
    if skipped:
        print(f"Episodes skipped: {', '.join(skipped)}")
    if warnings:
        print("Warnings:")
        for warning in warnings[:10]:
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  - ... {len(warnings) - 10} more")
    print(f"Elapsed: {format_elapsed(elapsed_seconds)}")
    print("")


def print_final_episode_summary(
    episode_id: str,
    review_dir: Path,
    audio_tracks: dict[str, Path],
    video_outputs: dict[str, Path],
    elapsed_seconds: float,
) -> None:
    print("")
    print(f"Final mux complete: {episode_id}")
    print(f"Output folder: {review_dir}")
    print(f"Elapsed: {format_elapsed(elapsed_seconds)}")
    print("")
    print("Audio tracks:")
    for label, path in audio_tracks.items():
        print(f"  - {label}: {path}")
    print("")
    print("Video outputs:")
    for variant, path in video_outputs.items():
        print(f"  - {variant}: {path}")
    print("")


def review_audio_tracks(audio_dir: Path, episode_id: str) -> dict[str, Path]:
    tracks = {
        "eng_ac3": audio_dir / f"01_{episode_id}_english_original_51.ac3",
        "restored_ac3": audio_dir / f"02_{episode_id}_spanish_restored_51.ac3",
        "spa2_ac3": audio_dir / f"04_{episode_id}_spanish2_original_stereo.ac3",
    }
    spa1_matches = sorted(audio_dir.glob(f"03_{episode_id}_spanish1_restored_old_stereo_*.ac3"))
    if len(spa1_matches) != 1:
        raise SystemExit(f"Expected exactly one restored spa1 stereo track in {audio_dir}, found {len(spa1_matches)}")
    tracks["spa1_ac3"] = spa1_matches[0]
    missing = [path for path in tracks.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing review audio track(s):\n" + "\n".join(str(path) for path in missing))
    return tracks


def write_episode_only_report(
    path: Path,
    episode_id: str,
    review_name: str,
    started_at: float,
    manifest: dict[str, object],
    commands: list[list[str]],
) -> None:
    elapsed = format_elapsed(time.monotonic() - started_at)
    lines = [
        f"# {episode_id} Episode-Only Restored-Audio Build",
        "",
        f"- Review name: `{review_name}`",
        f"- Output folder: `{path.parent}`",
        f"- Elapsed when report was written: `{elapsed}`",
        "- These files contain the restored episode video/audio only, without opening or end credits.",
        "- Video streams are copied; the episode video is not re-encoded.",
        "",
        "## Audio Tracks Used",
        "",
    ]
    for label, track_path in dict(manifest.get("audio_tracks", {})).items():
        lines.append(f"- {label}: `{track_path}`")
    lines.extend(["", "## Output Videos", ""])
    for variant, output in dict(manifest.get("episode_only_outputs", {})).items():
        lines.append(f"- {variant}: `{output}`")
    lines.extend(["", "## Commands", ""])
    append_command_blocks(lines, commands)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_adjustments_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        f"# {manifest.get('episode_id')} Episode Adjustments",
        "",
        f"- Episode title: `{manifest.get('episode_title')}`",
        f"- Review name: `{manifest.get('review_name')}`",
        f"- Fade frames: `{manifest.get('fade_frames')}`",
        f"- Final black hold frames: `{manifest.get('final_black_hold_frames', 1)}`",
        f"- Black threshold avg: `{manifest.get('black_avg')}`",
        f"- Black threshold max: `{manifest.get('black_max')}`",
        "",
    ]
    missing = dict(manifest.get("missing_episode_only_segments", {}))
    if missing:
        lines.extend(["## Missing Episode-Only Segments", ""])
        for variant, source in missing.items():
            lines.append(f"- {variant}: `{source}`")
        lines.append("")

    tail_trims = dict(manifest.get("tail_trim_adjustments", {}))
    if tail_trims:
        lines.extend(["## Tail Trims Before End Credits", ""])
        for variant, adjustment in tail_trims.items():
            if not isinstance(adjustment, dict):
                continue
            state = "needed" if adjustment.get("needed") else "not needed"
            lines.append(f"### {variant}")
            lines.append(f"- State: `{state}`")
            lines.append(f"- Source: `{adjustment.get('source')}`")
            lines.append(f"- Output: `{adjustment.get('output')}`")
            lines.append(f"- Trim seconds: `{adjustment.get('trim_seconds')}`")
            lines.append(f"- Source duration: `{adjustment.get('source_duration_seconds')}`")
            lines.append(f"- Output duration: `{adjustment.get('output_duration_seconds')}`")
            if adjustment.get("needed"):
                lines.extend(["", "```bash", shlex.join(list(adjustment.get("command", []))), "```"])
            lines.append("")

    lines.extend(["## End Fade-To-Black", ""])
    adjustments = dict(manifest.get("end_fade_adjustments", {}))
    if not adjustments:
        lines.append("_No episode-only segments were available to inspect._")
    for variant, adjustment in adjustments.items():
        if not isinstance(adjustment, dict):
            continue
        stats = dict(adjustment.get("stats", {}))
        state = "needed" if adjustment.get("needed") else "not needed"
        lines.append(f"### {variant}")
        lines.append(f"- State: `{state}`")
        lines.append(f"- Source: `{adjustment.get('source')}`")
        lines.append(f"- Output: `{adjustment.get('output')}`")
        lines.append(f"- Average luma: `{stats.get('avg_luma')}`")
        lines.append(f"- Maximum luma: `{stats.get('max_luma')}`")
        if adjustment.get("needed"):
            lines.extend(["", "```bash", shlex.join(list(adjustment.get("command", []))), "```"])
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_build_report(
    path: Path,
    episode_id: str,
    review_name: str,
    started_at: float,
    manifest: dict[str, object],
    audio_tracks: dict[str, Path],
    mux_cmds: list[list[str]],
    opening_cmds: list[list[str]],
    adjustment_cmds: list[list[str]],
    credit_cmds: list[list[str]],
    concat_cmds: list[list[str]],
    cover_cmds: list[list[str]],
    authoring_cmds: list[list[str]],
) -> None:
    elapsed = format_elapsed(time.monotonic() - started_at)
    lines = [
        f"# {episode_id} Final Build Report",
        "",
        f"- Review name: `{review_name}`",
        f"- Output folder: `{path.parent}`",
        f"- Elapsed when report was written: `{elapsed}`",
        f"- First `{manifest.get('silenced_start_seconds')}` seconds of every final audio track are silenced, not trimmed.",
        f"- Intermediate cache mode: `{'rebuild' if manifest.get('rebuild_intermediates') else 'reuse existing files when present'}`",
        "- Episode video streams are copied during muxing; only shared opening/end-credit segments are re-encoded to match each target format.",
        "",
        "## Audio Tracks",
        "",
    ]
    mix_levels = manifest.get("mix_levels")
    if isinstance(mix_levels, dict):
        lines.extend(
            [
                "### Restored Spanish 5.1 Mix Levels",
                "",
                f"- Preserved FL/FR/LFE/SL/SR gain: `{mix_levels.get('preserved_channel_gain_db')} dB`",
                f"- De-voiced center bed gain: `{mix_levels.get('center_bed_gain_db')} dB`",
                f"- Restored Spanish dialogue gain: `{mix_levels.get('dialogue_gain_db')} dB`",
                "",
            ]
        )
    episode_audio_patches = manifest.get("episode_audio_patches")
    if isinstance(episode_audio_patches, dict):
        lines.extend(["### Episode Audio Patches", ""])
        lines.append(f"- Enabled: `{episode_audio_patches.get('enabled')}`")
        lines.append(f"- Version: `{episode_audio_patches.get('version')}`")
        for target in ("dialogue", "spa1_fullmix"):
            patches = episode_audio_patches.get(target)
            if not isinstance(patches, list) or not patches:
                continue
            lines.append(f"- {target}:")
            for patch in patches:
                if not isinstance(patch, dict):
                    continue
                lines.append(
                    "  "
                    f"`{patch.get('patch_id')}` "
                    f"{patch.get('start_seconds')}s-{patch.get('end_seconds')}s "
                    f"floor=`{patch.get('floor')}`"
                )
                lines.append(f"  {patch.get('description')}")
        lines.append(f"- Effective dialogue WAV: `{episode_audio_patches.get('dialogue_effective')}`")
        lines.append(f"- Effective spa1 fullmix WAV: `{episode_audio_patches.get('spa1_fullmix_effective')}`")
        lines.append("")
    for label, track_path in audio_tracks.items():
        lines.append(f"- {label}: `{track_path}`")

    source_copies = manifest.get("source_copies")
    if isinstance(source_copies, dict) and manifest.get("source_copies_enabled"):
        lines.extend(["", "## Episode-Local Source Copies", ""])
        video_copies = dict(source_copies.get("video", {}))
        audio_copies = dict(source_copies.get("audio", {}))
        if video_copies:
            lines.append("### Video")
            for variant, copied_path in video_copies.items():
                lines.append(f"- {variant}: `{copied_path}`")
        if audio_copies:
            lines.extend(["", "### Audio"])
            for label, copied_path in audio_copies.items():
                lines.append(f"- {label}: `{copied_path}`")

    lines.extend(["", "## Selected Episode Videos", ""])
    for variant, source in dict(manifest.get("selected_videos", {})).items():
        lines.append(f"- {variant}: `{source}`")

    lines.extend(["", "## Output Videos", ""])
    for variant, output in dict(manifest.get("video_outputs", {})).items():
        lines.append(f"- {variant}: `{output}`")

    subtitles = list(manifest.get("embedded_subtitles", []))
    lines.extend(["", "## Embedded Subtitles", ""])
    if subtitles:
        for subtitle in subtitles:
            if isinstance(subtitle, dict):
                lines.append(
                    f"- {subtitle.get('language')}: {subtitle.get('title')} -> `{subtitle.get('path')}`"
                )
    else:
        lines.append("_No expected SRT subtitle files were available, so no subtitles were embedded._")

    cover_art = manifest.get("cover_art")
    if isinstance(cover_art, dict):
        lines.extend(["", "## Cover Art", ""])
        if cover_art.get("enabled") and cover_art.get("files"):
            for variant, cover in dict(cover_art.get("files", {})).items():
                lines.append(f"- {variant}: `{cover}`")
        elif cover_art.get("enabled"):
            lines.append("_Cover art was enabled, but no cover files were recorded._")
        else:
            lines.append("_Cover art attachment was disabled._")

    episode_segments = dict(manifest.get("episode_segments", {}))
    if episode_segments:
        lines.extend(["", "## Episode-Only Videos Without Credits", ""])
        for variant, output in episode_segments.items():
            lines.append(f"- {variant}: `{output}`")

    cache_events = list(manifest.get("intermediate_cache", []))
    if cache_events:
        lines.extend(["", "## Intermediate Cache", ""])
        for event in cache_events:
            if not isinstance(event, dict):
                continue
            lines.append(
                f"- {event.get('label')}: `{event.get('action')}` -> `{event.get('output')}`"
            )

    openings = manifest.get("openings")
    if isinstance(openings, dict) and openings.get("enabled"):
        lines.extend(["", "## Openings", ""])
        lines.append(f"- Generation: `{openings.get('generation')}`")
        sources = dict(openings.get("sources", {}))
        segments = dict(openings.get("segments", {}))
        reused = dict(openings.get("reused", {}))
        for variant, source in sources.items():
            state = "reused" if reused.get(variant) else "created"
            lines.append(f"- {variant}: source `{source}`")
            lines.append(f"  prepared segment ({state}): `{segments.get(variant)}`")

    adjustments = manifest.get("adjustments")
    if isinstance(adjustments, dict):
        lines.extend(["", "## Episode Adjustments", ""])
        lines.append(
            f"- Video adjustments enabled: `{adjustments.get('video_episode_adjustments_enabled')}`"
        )
        lines.append(
            f"- Audio adjustments enabled: `{adjustments.get('audio_episode_adjustments_enabled')}`"
        )
        tail_trims = dict(adjustments.get("tail_trim", {}))
        if tail_trims:
            lines.append("- Episode tail trims before end credits:")
            for variant, trim in tail_trims.items():
                if not isinstance(trim, dict):
                    continue
                state = "created/used" if trim.get("needed") else "not needed"
                lines.append(
                    f"  - {variant}: {state}; trim=`{trim.get('trim_seconds')}`s, "
                    f"duration `{trim.get('source_duration_seconds')}`s -> "
                    f"`{trim.get('output_duration_seconds')}`s"
                )
                if trim.get("needed"):
                    lines.append(f"    segment: `{trim.get('output')}`")
        spa2_tail_extensions = dict(adjustments.get("spa2_tail_extension", {}))
        if spa2_tail_extensions:
            lines.append("- Spanish redubbing tail extensions before end credits:")
            for variant, extension in spa2_tail_extensions.items():
                if not isinstance(extension, dict):
                    continue
                lines.append(
                    f"  - {variant}: `{extension.get('extension_id')}` "
                    f"duration=`{extension.get('duration_seconds')}`s"
                )
                lines.append(f"    audio: `{extension.get('audio_path')}`")
                lines.append(f"    segment: `{extension.get('output')}`")
    if (
        isinstance(adjustments, dict)
        and adjustments.get("video_episode_adjustments_enabled")
        and adjustments.get("auto_end_fade_enabled")
    ):
        lines.append(
            "- Auto end fade-to-black: "
            f"`{adjustments.get('end_fade_frames')}` frame(s), "
            f"black thresholds avg<=`{adjustments.get('black_avg')}`, "
            f"max<=`{adjustments.get('black_max')}`"
        )
        for variant, adjustment in dict(adjustments.get("end_fade", {})).items():
            if not isinstance(adjustment, dict):
                continue
            stats = dict(adjustment.get("stats", {}))
            state = "created/used" if adjustment.get("needed") else "not needed"
            lines.append(
                f"- {variant}: {state}; avg_luma=`{stats.get('avg_luma')}`, "
                f"max_luma=`{stats.get('max_luma')}`"
            )
            if adjustment.get("needed"):
                lines.append(f"  segment: `{adjustment.get('output')}`")

    end_credits = manifest.get("end_credits")
    if isinstance(end_credits, dict) and end_credits.get("enabled"):
        lines.extend(["", "## End Credits", ""])
        sources = dict(end_credits.get("sources", {}))
        segments = dict(end_credits.get("segments", {}))
        reused = dict(end_credits.get("reused", {}))
        concat_lists = dict(end_credits.get("concat_lists", {}))
        for variant, source in sources.items():
            state = "reused" if reused.get(variant) else "created"
            lines.append(f"- {variant}: source `{source}`")
            lines.append(f"  prepared segment ({state}): `{segments.get(variant)}`")
            lines.append(f"  concat list: `{concat_lists.get(variant)}`")

    lines.extend(["", "## Video Mux Commands", ""])
    append_command_blocks(lines, mux_cmds)

    if opening_cmds:
        lines.extend(["", "## Opening Preparation Commands", ""])
        append_command_blocks(lines, opening_cmds)
    elif isinstance(openings, dict) and openings.get("enabled"):
        lines.extend(["", "## Opening Preparation Commands", "", "All needed prepared opening segments already existed and were reused."])

    if adjustment_cmds:
        lines.extend(["", "## Episode Adjustment Commands", ""])
        append_command_blocks(lines, adjustment_cmds)
    elif (
        isinstance(adjustments, dict)
        and adjustments.get("video_episode_adjustments_enabled")
        and adjustments.get("auto_end_fade_enabled")
    ):
        lines.extend(["", "## Episode Adjustment Commands", "", "No adjustment segment commands were needed."])

    if credit_cmds:
        lines.extend(["", "## End-Credit Preparation Commands", ""])
        append_command_blocks(lines, credit_cmds)
    elif isinstance(end_credits, dict) and end_credits.get("enabled"):
        lines.extend(["", "## End-Credit Preparation Commands", "", "All needed prepared end-credit segments already existed and were reused."])

    lines.extend(["", "## Final Concat Commands", ""])
    append_command_blocks(lines, concat_cmds)

    if cover_cmds:
        lines.extend(["", "## Cover Art Commands", ""])
        append_command_blocks(lines, cover_cmds)

    if authoring_cmds:
        lines.extend(["", "## Review MKV Authoring Commands", ""])
        append_command_blocks(lines, authoring_cmds)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_command_blocks(lines: list[str], commands: list[list[str]]) -> None:
    if not commands:
        lines.append("_No commands in this section._")
        return
    for index, cmd in enumerate(commands, start=1):
        lines.extend([f"### Command {index}", "", "```bash", shlex.join(cmd), "```", ""])


def format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def cmd_clearvoice_enhance(
    input_wavs: list[Path],
    variants: list[str],
    review_name: str,
    out_dir: Path,
    work_dir: Path,
    python: Path,
    run: bool,
) -> int:
    out_dir = out_dir / review_name
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Path("scripts/run_clearvoice.py")
    if not runner.exists():
        raise SystemExit(f"ClearVoice runner is missing: {runner}")

    review_files: list[Path] = []
    labels: list[str] = []
    commands: list[list[str]] = []

    for input_wav in input_wavs:
        if not input_wav.exists():
            raise SystemExit(f"Input WAV does not exist: {input_wav}")
        source_label = compact_source_label(input_wav)
        review_files.append(input_wav)
        labels.append(f"{source_label}_source")
        for variant in variants:
            output = out_dir / f"{source_label}_{variant}.wav"
            cmd = [
                str(python),
                str(runner),
                "--input",
                str(input_wav),
                "--output",
                str(output),
                "--variant",
                variant,
            ]
            commands.append(cmd)
            review_files.append(output)
            labels.append(f"{source_label}_{variant}")

    write_manifest(
        out_dir / "manifest.json",
        {
            "review_name": review_name,
            "inputs": [str(path) for path in input_wavs],
            "variants": variants,
            "python": str(python),
            "commands": commands,
            "notes": {
                "se48k": "ClearVoice MossFormer2_SE_48K speech enhancement.",
                "sr48k": "ClearVoice MossFormer2_SR_48K speech super-resolution.",
                "se48k_sr48k": "Speech enhancement followed by speech super-resolution.",
            },
        },
    )

    if run:
        if not python.exists():
            raise SystemExit(f"ClearVoice Python does not exist: {python}. Create .venv-clearvoice first.")
        for cmd in commands:
            print(shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        return cmd_prepare_review(review_name, review_files, labels, work_dir / "review")

    for cmd in commands:
        print(shlex.join(cmd))
    print(f"review will be created after --run: {work_dir / 'review' / review_name}")
    return 0


def cmd_ai_inpaint_stable(
    python: Path,
    context_audio: Path,
    overlay_base: Path,
    out_dir: Path,
    stable_audio_root: Path,
    pretrained_name: str,
    gap_start: float,
    gap_end: float,
    prompt: str,
    negative_prompt: str,
    seeds: list[int],
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    texture_gain_db: float,
    texture_highpass: float,
    texture_lowpass: float,
    fade_in: float,
    fade_out: float,
    model_half: bool,
    device: str,
    run: bool,
) -> int:
    runner = Path("scripts/run_stable_audio_inpaint.py")
    if not runner.exists():
        raise SystemExit(f"Stable Audio inpaint runner is missing: {runner}")
    if run:
        if not python.exists():
            raise SystemExit(
                f"Inpainting Python does not exist: {python}. "
                "Create .venv-inpaint first; see requirements-inpaint.txt."
            )
        if not context_audio.exists():
            raise SystemExit(f"Context audio does not exist: {context_audio}")
        if not overlay_base.exists():
            raise SystemExit(f"Overlay base does not exist: {overlay_base}")
        if not stable_audio_root.exists():
            raise SystemExit(f"Stable Audio Tools root does not exist: {stable_audio_root}")

    commands: list[list[str]] = []
    for seed in seeds:
        label = f"stable_audio_seed{seed}_steps{steps}_cfg{cfg_scale:g}"
        cmd = [
            str(python),
            str(runner),
            "--stable-audio-root",
            str(stable_audio_root),
            "--pretrained-name",
            pretrained_name,
            "--context-audio",
            str(context_audio),
            "--overlay-base",
            str(overlay_base),
            "--out-dir",
            str(out_dir),
            "--label",
            label,
            "--prompt",
            prompt,
            "--negative-prompt",
            negative_prompt,
            "--gap-start",
            f"{gap_start:.6f}",
            "--gap-end",
            f"{gap_end:.6f}",
            "--steps",
            str(steps),
            "--cfg-scale",
            f"{cfg_scale:g}",
            "--sampler-type",
            sampler_type,
            "--seed",
            str(seed),
            "--texture-gain-db",
            f"{texture_gain_db:g}",
            "--texture-highpass",
            f"{texture_highpass:g}",
            "--texture-lowpass",
            f"{texture_lowpass:g}",
            "--fade-in",
            f"{fade_in:g}",
            "--fade-out",
            f"{fade_out:g}",
            "--device",
            device,
        ]
        if model_half:
            cmd.append("--model-half")
        commands.append(cmd)

    out_dir.mkdir(parents=True, exist_ok=True)
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Stable Audio Inpaint Candidates",
                "",
                "This folder contains neural texture candidates for the approved S01E03 title narrator patch.",
                "",
                "The important distinction is:",
                "",
                "- `*_raw_model_output.wav`: diagnostic full-context model reconstruction. Do not judge this as the final patch.",
                "- `*_texture_only.wav`: only the generated gap texture after filtering, gain, and edge fades.",
                "- `*_OVER_APPROVED_PATCH.wav`: the approved voice patch plus the generated texture. This is the main file to review.",
                "",
                f"The target gap is `{gap_start:.3f}s` to `{gap_end:.3f}s` inside the 7s context.",
                "",
                "If the overlay sounds too thin or synthetic, the next useful test is to relax the texture filter, for example",
                "`--texture-highpass 1200 --texture-lowpass 0`, so the model can contribute more body instead of only high frequencies.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_manifest(
        out_dir / "stable_audio_inpaint_plan.json",
        {
            "backend": "stable-audio-tools",
            "pretrained_name": pretrained_name,
            "context_audio": str(context_audio),
            "overlay_base": str(overlay_base),
            "gap_start": gap_start,
            "gap_end": gap_end,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seeds": seeds,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_type": sampler_type,
            "texture_gain_db": texture_gain_db,
            "texture_highpass": texture_highpass,
            "texture_lowpass": texture_lowpass,
            "fade_in": fade_in,
            "fade_out": fade_out,
            "commands": commands,
            "notes": (
                "The model generates the silenced gap from context. The runner "
                "extracts only the generated gap texture and overlays it under "
                "the approved replacement voice."
            ),
        },
    )

    started = time.monotonic()
    result = run_or_print(commands, run)
    if run:
        print(f"Stable Audio inpaint complete: {len(commands)} candidate(s) in {format_elapsed(time.monotonic() - started)}")
        print(f"review_dir={out_dir}")
        print(f"notes={readme}")
        print("Review priority:")
        print("  1. *_OVER_APPROVED_PATCH.wav  -> approved patch plus generated texture")
        print("  2. *_texture_only.wav         -> generated texture by itself")
        print("  3. *_raw_model_output.wav     -> diagnostic only; it can sound weird")
    else:
        print(f"review will be created after --run: {out_dir}")
        print(f"notes will be written to: {readme}")
    return result


def cmd_spa1_fullmix_shootout(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    starts: list[str],
    duration: str,
    variants: list[str],
    review_name: str,
    out_dir: Path,
    run: bool,
) -> int:
    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    assert episode.spa1 is not None

    out_dir = out_dir / episode.episode_id / review_name
    out_dir.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    review_files: list[Path] = []
    labels: list[str] = []

    for index, start in enumerate(starts, start=1):
        window_label = f"W{index:02d}_{compact_time_label(start)}"
        raw_output = out_dir / f"{window_label}_spa1_original_raw.wav"
        raw_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            *media_args(episode.spa1, start, duration),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s24le",
            str(raw_output),
        ]
        commands.append(raw_cmd)
        review_files.append(raw_output)
        labels.append(f"{window_label}_spa1_original_raw")

        for variant_name in variants:
            variant = require_full_mix_variant(variant_name)
            output = out_dir / f"{window_label}_{variant.name}.wav"
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                *media_args(episode.spa1, start, duration),
                "-map",
                "0:a:0",
                "-af",
                variant.filtergraph,
                "-c:a",
                "pcm_s24le",
                str(output),
            ]
            commands.append(cmd)
            review_files.append(output)
            labels.append(f"{window_label}_{variant.name}")

    write_manifest(
        out_dir / "manifest.json",
        {
            "episode_id": episode.episode_id,
            "review_name": review_name,
            "source": str(episode.spa1),
            "starts": starts,
            "duration": duration,
            "variants": [
                {
                    "name": require_full_mix_variant(name).name,
                    "filtergraph": require_full_mix_variant(name).filtergraph,
                    "notes": require_full_mix_variant(name).notes,
                }
                for name in variants
            ],
            "commands": commands,
            "review_note": "Full old-spa1 restoration tests. These do not extract dialogue or use the English bed.",
        },
    )

    if run:
        for cmd in commands:
            print(shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        return cmd_prepare_review(review_name, review_files, labels, work_dir / "review")

    for cmd in commands:
        print(shlex.join(cmd))
    print(f"review will be created after --run: {work_dir / 'review' / review_name}")
    return 0


def compact_source_label(path: Path) -> str:
    stem = path.stem
    if stem.startswith("W"):
        parts = stem.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem).strip("_")


def compact_time_label(value: str) -> str:
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        if hours == "00":
            return f"{int(minutes):02d}m{int(float(seconds)):02d}"
        return f"{int(hours):02d}h{int(minutes):02d}m{int(float(seconds)):02d}"
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def find_single_stem_output(folder: Path, stem: str) -> Path:
    matches = sorted(folder.glob(f"*{stem}*.wav"))
    if len(matches) != 1:
        raise SystemExit(f"Expected one {stem} WAV in {folder}, found {len(matches)}")
    return matches[0]



def cmd_ensemble_shootout(
    input_dir: Path,
    work_dir: Path,
    episode_id: str,
    starts: list[str],
    duration: str,
    variant_name: str,
    source_mode: str,
    review_name: str,
    single_stem: str,
    sample_rate: int,
    out_root: Path,
    run: bool,
) -> int:
    candidates = [
        {
            "label": "kim_vocal_2",
            "model": "Kim_Vocal_2.onnx",
            "extra_models": [],
            "ensemble_algorithm": None,
        },
        {
            "label": "mdx23c_8kfft_instvoc_hq",
            "model": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
            "extra_models": [],
            "ensemble_algorithm": None,
        },
        {
            "label": "kim_mdx23c_avg_wave",
            "model": "Kim_Vocal_2.onnx",
            "extra_models": ["MDX23C-8KFFT-InstVoc_HQ.ckpt"],
            "ensemble_algorithm": "avg_wave",
        },
        {
            "label": "kim_mdx23c_median_wave",
            "model": "Kim_Vocal_2.onnx",
            "extra_models": ["MDX23C-8KFFT-InstVoc_HQ.ckpt"],
            "ensemble_algorithm": "median_wave",
        },
        {
            "label": "kim_mdx23c_uvr_max_spec",
            "model": "Kim_Vocal_2.onnx",
            "extra_models": ["MDX23C-8KFFT-InstVoc_HQ.ckpt"],
            "ensemble_algorithm": "uvr_max_spec",
        },
    ]

    if len(starts) > 1:
        for index, start in enumerate(starts, start=1):
            child_review_name = f"{review_name}_{index:02d}_{start.replace(':', '-')}"
            cmd_ensemble_shootout(
                input_dir,
                work_dir,
                episode_id,
                starts=[start],
                duration=duration,
                variant_name=variant_name,
                source_mode=source_mode,
                review_name=child_review_name,
                single_stem=single_stem,
                sample_rate=sample_rate,
                out_root=out_root,
                run=run,
            )
        return 0

    episode = find_episode(input_dir, episode_id)
    ensure_required(episode)
    assert episode.spa1 is not None

    out_root.mkdir(parents=True, exist_ok=True)
    variant = require_clean_variant(variant_name)
    start = starts[0]
    suffix = f"{start}_{duration}".replace(":", "-")
    clean_dir = work_dir / "05_clean_spa1" / episode.episode_id / suffix
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_wav = clean_dir / f"{episode.episode_id}_spa1_{source_mode}_{variant.name}.wav"
    clean_filter = source_mode_filter(source_mode, variant.filtergraph)
    clean_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        *media_args(episode.spa1, start, duration),
        "-map",
        "0:a:0",
        "-af",
        clean_filter,
        "-c:a",
        "pcm_s24le",
        str(clean_wav),
    ]

    jobs = []
    for candidate in candidates:
        sep_dir = out_root / f"{review_name}_{candidate['label']}"
        sep_cmd = build_separation_command(
            "audio-separator",
            clean_wav,
            sep_dir,
            model=candidate["model"],
            extra_models=candidate["extra_models"],
            single_stem=single_stem,
            sample_rate=sample_rate,
            model_file_dir=work_dir / "models" / "audio-separator",
            ensemble_algorithm=candidate["ensemble_algorithm"],
        )
        jobs.append((candidate["label"], sep_dir, sep_cmd))

    write_manifest(
        out_root / f"{review_name}.ensemble-shootout.manifest.json",
        {
            "episode_id": episode.episode_id,
            "source": str(episode.spa1),
            "clean_wav": str(clean_wav),
            "start": start,
            "duration": duration,
            "source_mode": source_mode,
            "variant": variant.name,
            "filtergraph": clean_filter,
            "single_stem": single_stem,
            "sample_rate": sample_rate,
            "candidates": candidates,
            "clean_command": clean_cmd,
            "separation_commands": [job[2] for job in jobs],
        },
    )

    if run:
        print(shlex.join(clean_cmd), flush=True)
        subprocess.run(clean_cmd, check=True)
        review_files = [clean_wav]
        labels = [f"input_spa1_{variant.name}_{start}"]
        for label, sep_dir, sep_cmd in jobs:
            sep_dir.mkdir(parents=True, exist_ok=True)
            print(shlex.join(sep_cmd), flush=True)
            subprocess.run(sep_cmd, check=True)
            review_files.append(find_single_vocals_output(sep_dir))
            labels.append(label)
        return cmd_prepare_review(review_name, review_files, labels, work_dir / "review")

    print(shlex.join(clean_cmd))
    for _label, _sep_dir, sep_cmd in jobs:
        print(shlex.join(sep_cmd))
    print(f"review will be created after --run: {work_dir / 'review' / review_name}")
    return 0


def find_single_separator_output(folder: Path, stem: str) -> Path:
    matches = sorted(folder.glob(f"*{stem}*.wav"))
    if len(matches) != 1:
        raise SystemExit(f"Expected one {stem} WAV in {folder}, found {len(matches)}")
    return matches[0]


def find_single_vocals_output(folder: Path) -> Path:
    return find_single_separator_output(folder, "Vocals")


def cmd_tools_status() -> int:
    for name, path in installed_tools().items():
        print(f"{name}: {path or 'missing'}")
    return 0


def ensure_required(episode: EpisodeAssets, require_video: bool = True) -> None:
    missing = episode.missing_required()
    if not require_video:
        missing = [item for item in missing if item != "video"]
    if missing:
        raise SystemExit(f"{episode.episode_id} missing required sources: {', '.join(missing)}")


def media_args(path: Path, start: str | None, duration: str | None) -> list[str]:
    args = ["-i", str(path)]
    if start:
        args = ["-ss", start] + args
    if duration:
        args += ["-t", duration]
    return args


def build_extract_commands(
    episode: EpisodeAssets,
    out_dir: Path,
    start: str | None = None,
    duration: str | None = None,
    include_spa2: bool = False,
) -> list[list[str]]:
    assert episode.eng_51 and episode.spa1
    commands: list[list[str]] = [
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            *media_args(episode.eng_51, start, duration),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s24le",
            str(out_dir / f"{episode.episode_id}_eng_51.wav"),
        ],
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            *media_args(episode.eng_51, start, duration),
            "-filter_complex",
            "channelsplit=channel_layout=5.1(side)[FL][FR][FC][LFE][SL][SR]",
            "-map",
            "[FL]",
            "-c:a",
            "pcm_s24le",
            str(out_dir / "FL.wav"),
            "-map",
            "[FR]",
            "-c:a",
            "pcm_s24le",
            str(out_dir / "FR.wav"),
            "-map",
            "[FC]",
            "-c:a",
            "pcm_s24le",
            str(out_dir / "FC.wav"),
            "-map",
            "[LFE]",
            "-c:a",
            "pcm_s24le",
            str(out_dir / "LFE.wav"),
            "-map",
            "[SL]",
            "-c:a",
            "pcm_s24le",
            str(out_dir / "SL.wav"),
            "-map",
            "[SR]",
            "-c:a",
            "pcm_s24le",
            str(out_dir / "SR.wav"),
        ],
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            *media_args(episode.spa1, start, duration),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s24le",
            str(out_dir / f"{episode.episode_id}_spa1_stereo.wav"),
        ],
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            *media_args(episode.spa1, start, duration),
            "-map",
            "0:a:0",
            "-af",
            "pan=mono|c0=0.5*c0+0.5*c1",
            "-c:a",
            "pcm_s24le",
            str(out_dir / f"{episode.episode_id}_spa1_mono_sum.wav"),
        ],
    ]
    if include_spa2 and episode.spa2:
        commands.append(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                *media_args(episode.spa2, start, duration),
                "-map",
                "0:a:0",
                "-c:a",
                "pcm_s24le",
                str(out_dir / f"{episode.episode_id}_spa2_stereo.wav"),
            ]
        )
    return commands


def run_or_print(commands: Iterable[list[str]], run: bool) -> int:
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
        if run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
