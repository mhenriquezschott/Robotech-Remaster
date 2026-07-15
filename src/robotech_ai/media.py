from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPISODE_RE = re.compile(r"Robotech-(S\d{2}E\d{2})(?:\.(mp4)|\.track(?P<track>\d{2})\((?P<tag>[^)]+)\)\.ac3)$")


@dataclass(frozen=True)
class EpisodeAssets:
    episode_id: str
    video: Path | None = None
    eng_51: Path | None = None
    spa1: Path | None = None
    spa2: Path | None = None

    def missing_required(self) -> list[str]:
        missing: list[str] = []
        if self.video is None:
            missing.append("video")
        if self.eng_51 is None:
            missing.append("eng_51")
        if self.spa1 is None:
            missing.append("spa1")
        return missing

    def as_dict(self) -> dict[str, str | None]:
        return {
            "episode_id": self.episode_id,
            "video": str(self.video) if self.video else None,
            "eng_51": str(self.eng_51) if self.eng_51 else None,
            "spa1": str(self.spa1) if self.spa1 else None,
            "spa2": str(self.spa2) if self.spa2 else None,
        }


def discover_episodes(input_dir: Path) -> list[EpisodeAssets]:
    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(input_dir.glob("Robotech-S??E??*")):
        if not path.is_file():
            continue
        match = EPISODE_RE.match(path.name)
        if not match:
            continue
        episode_id = match.group(1)
        group = grouped.setdefault(episode_id, {})
        if match.group(2):
            group["video"] = path
            continue
        track = match.group("track")
        tag = match.group("tag")
        if track == "00" and tag == "eng":
            group["eng_51"] = path
        elif track == "01" and tag == "spa1":
            group["spa1"] = path
        elif track == "02" and tag == "spa2":
            group["spa2"] = path

    episodes: list[EpisodeAssets] = []
    for episode_id, group in sorted(grouped.items()):
        episodes.append(
            EpisodeAssets(
                episode_id=episode_id,
                video=group.get("video"),
                eng_51=group.get("eng_51"),
                spa1=group.get("spa1"),
                spa2=group.get("spa2"),
            )
        )
    return episodes


def ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,channels,channel_layout:stream_tags=language,title",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout)


def ffprobe_full(path: Path) -> dict[str, Any]:
    """Return stream, format, chapter, tag, and disposition metadata for mux planning."""
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout)


def first_audio_stream(probe: dict[str, Any]) -> dict[str, Any] | None:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    return None


def duration_seconds(probe: dict[str, Any]) -> float | None:
    value = probe.get("format", {}).get("duration")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize_probe(path: Path, role: str, probe: dict[str, Any]) -> dict[str, Any]:
    audio = first_audio_stream(probe)
    return {
        "role": role,
        "path": str(path),
        "duration": duration_seconds(probe),
        "codec": audio.get("codec_name") if audio else None,
        "channels": audio.get("channels") if audio else None,
        "channel_layout": audio.get("channel_layout") if audio else None,
        "language": (audio.get("tags") or {}).get("language") if audio else None,
    }


def probe_episode(assets: EpisodeAssets) -> dict[str, Any]:
    roles = {
        "video": assets.video,
        "eng_51": assets.eng_51,
        "spa1": assets.spa1,
        "spa2": assets.spa2,
    }
    probed: dict[str, Any] = {"episode_id": assets.episode_id, "missing": assets.missing_required(), "files": {}}
    for role, path in roles.items():
        if path is None:
            continue
        probe = ffprobe(path)
        probed["files"][role] = summarize_probe(path, role, probe)
    probed["warnings"] = episode_warnings(probed)
    return probed


def episode_warnings(probed: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    files = probed.get("files", {})
    if probed.get("missing"):
        warnings.append("missing_required:" + ",".join(probed["missing"]))

    eng = files.get("eng_51")
    spa1 = files.get("spa1")
    spa2 = files.get("spa2")
    video = files.get("video")

    if eng and eng.get("channels") != 6:
        warnings.append("eng_51_not_6_channels")
    if spa1 and spa1.get("channels") != 2:
        warnings.append("spa1_not_stereo")
    if spa2 and spa2.get("channels") != 2:
        warnings.append("spa2_not_stereo")

    reference_duration = video.get("duration") if video else None
    if reference_duration:
        for role in ("eng_51", "spa1", "spa2"):
            item = files.get(role)
            if item and item.get("duration") is not None:
                delta = abs(float(item["duration"]) - float(reference_duration))
                if delta > 0.25:
                    warnings.append(f"{role}_duration_delta_gt_250ms:{delta:.3f}")
    return warnings
