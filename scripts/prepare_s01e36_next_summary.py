#!/usr/bin/env python3
"""Prepare S01E36 next-episode summary reconstruction assets."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MASTERS_ROOT = Path(
    "/mnt/usb-Seagate_Expansion_HDD_00000000NT17VSPP-0:0-part2/Multimedia/Videos/Series/"
    "Robotech_Bluraywithoriginalspanishdubengandnewspanish/Bluray/Season02-Masters/eps"
)
DVD_SUMMARY = ROOT / "Robotech/proc/MacrossSagaDVD/ep36/robotech-s01e36_nextsummary01.mp4"
DVD_FULL = ROOT / "Robotech/proc/MacrossSagaDVD/ep36/Robotech - 1x36 - To the Stars.mkv"
OUT_ROOT = ROOT / "work/review/S01E36_next_summary_reconstruction_001"
FPS = Fraction(24000, 1001)


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    source_name: str
    start_minutes: int
    start_seconds: int
    start_frame: int
    end_minutes: int
    end_seconds: int
    end_frame: int
    repeat_first_frames: int = 0
    repeat_last_frames: int = 0

    @property
    def source(self) -> Path:
        return MASTERS_ROOT / self.source_name

    @property
    def start(self) -> float:
        return timecode_to_seconds(self.start_minutes, self.start_seconds, self.start_frame)

    @property
    def end(self) -> float:
        return timecode_to_seconds(self.end_minutes, self.end_seconds, self.end_frame)

    @property
    def duration(self) -> float:
        return self.end - self.start


SCENES = [
    SceneSpec("01", "Robotech-S02E01.mkv", 3, 51, 10, 3, 53, 13),
    SceneSpec("02", "Robotech-S02E11.mkv", 14, 9, 4, 14, 10, 16),
    SceneSpec("03", "Robotech-S02E01.mkv", 17, 58, 18, 18, 4, 13),
    SceneSpec("04", "Robotech-S02E11.mkv", 14, 0, 22, 14, 6, 17, repeat_first_frames=2, repeat_last_frames=2),
]


def timecode_to_seconds(minutes: int, seconds: int, frame: int) -> float:
    return float(minutes * 60 + seconds + Fraction(frame, 1) / FPS)


def run_cmd(cmd: list[str], *, run: bool) -> None:
    print(shlex.join(cmd), flush=True)
    if run:
        subprocess.run(cmd, check=True)


def scene_source_copy_cmd(scene: SceneSpec, output: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        f"{scene.start:.6f}",
        "-t",
        f"{scene.duration:.6f}",
        "-i",
        str(scene.source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        str(output),
    ]


def scene_crop_lossless_cmd(scene: SceneSpec, output: Path) -> list[str]:
    filters = ["crop=1440:1080:240:0", "setsar=1"]
    if scene.repeat_first_frames or scene.repeat_last_frames:
        start_duration = float(Fraction(scene.repeat_first_frames, 1) / FPS)
        stop_duration = float(Fraction(scene.repeat_last_frames, 1) / FPS)
        filters.append(f"tpad=start_mode=clone:start_duration={start_duration:.9f}:stop_mode=clone:stop_duration={stop_duration:.9f}")
    filters.append("format=yuv420p")
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        f"{scene.start:.6f}",
        "-t",
        f"{scene.duration:.6f}",
        "-i",
        str(scene.source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-preset",
        "veryslow",
        "-crf",
        "0",
        "-r",
        "24000/1001",
        str(output),
    ]


def audio_extract_cmds(audio_dir: Path) -> list[list[str]]:
    start = "00:23:36.000"
    duration = "00:00:17.000"
    eng51_ac3 = audio_dir / "S01E36_next_summary_dvd_eng51_original.ac3"
    eng51_wav = audio_dir / "S01E36_next_summary_dvd_eng51_original.wav"
    center_wav = audio_dir / "S01E36_next_summary_dvd_center.wav"
    stereo_wav = audio_dir / "S01E36_next_summary_dvd_stereo_reference.wav"
    return [
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            start,
            "-t",
            duration,
            "-i",
            str(DVD_FULL),
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            str(eng51_ac3),
        ],
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(eng51_ac3),
            "-c:a",
            "pcm_s24le",
            str(eng51_wav),
        ],
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(eng51_ac3),
            "-af",
            "pan=mono|c0=FC",
            "-c:a",
            "pcm_s24le",
            str(center_wav),
        ],
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(DVD_SUMMARY),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s24le",
            str(stereo_wav),
        ],
    ]


def write_manifest(output: Path) -> None:
    manifest = {
        "kind": "robotech_s01e36_next_summary_reconstruction_assets",
        "fps": "24000/1001",
        "masters_root": str(MASTERS_ROOT),
        "dvd_summary": str(DVD_SUMMARY),
        "dvd_full": str(DVD_FULL),
        "notes": [
            "Scene timecodes are interpreted as minutes:seconds:frame at 24000/1001 fps.",
            "source_copy_1920 clips are stream-copy reference cuts and may start on nearby keyframes.",
            "crop1440_lossless clips are frame-accurate working clips encoded losslessly with x264 CRF 0.",
            "Scene 04 working clip repeats first and last frame twice using tpad clone.",
        ],
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "source_name": scene.source_name,
                "timecode_start": f"{scene.start_minutes:02d}:{scene.start_seconds:02d}:{scene.start_frame:02d}",
                "timecode_end": f"{scene.end_minutes:02d}:{scene.end_seconds:02d}:{scene.end_frame:02d}",
                "source": str(scene.source),
                "start_seconds": round(scene.start, 6),
                "end_seconds": round(scene.end, 6),
                "duration_seconds": round(scene.duration, 6),
                "repeat_first_frames": scene.repeat_first_frames,
                "repeat_last_frames": scene.repeat_last_frames,
            }
            for scene in SCENES
        ],
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root
    source_dir = out_root / "source_copy_1920"
    crop_dir = out_root / "crop1440_lossless"
    audio_dir = out_root / "audio"
    manifest_path = out_root / "manifest.json"
    for directory in (source_dir, crop_dir, audio_dir):
        if args.run:
            directory.mkdir(parents=True, exist_ok=True)
    if args.run:
        out_root.mkdir(parents=True, exist_ok=True)

    for scene in SCENES:
        if not scene.source.is_file():
            raise SystemExit(f"Missing source: {scene.source}")
        run_cmd(scene_source_copy_cmd(scene, source_dir / f"{scene.scene_id}_{scene.source.stem}_sourcecopy_1920.mkv"), run=args.run)
        run_cmd(scene_crop_lossless_cmd(scene, crop_dir / f"{scene.scene_id}_{scene.source.stem}_crop1440_lossless.mkv"), run=args.run)

    if not DVD_FULL.is_file():
        raise SystemExit(f"Missing full DVD episode: {DVD_FULL}")
    if not DVD_SUMMARY.is_file():
        raise SystemExit(f"Missing DVD summary clip: {DVD_SUMMARY}")
    for cmd in audio_extract_cmds(audio_dir):
        run_cmd(cmd, run=args.run)

    if args.run:
        write_manifest(manifest_path)
        print(f"manifest={manifest_path}")
    else:
        print(f"would write manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
