#!/usr/bin/env python3
"""Build an alternate S01E36 next-summary review segment from Southern Cross sources.

This does not overwrite the approved ready segment. It creates a comparison
package under ``work/review/S01E36_next_summary_southern_cross_source_001``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "Robotech/The Super Dimension Cavalry Southern Cross (1984)/Season 1"
OUT_ROOT = ROOT / "work/review/S01E36_next_summary_southern_cross_source_001"
READY_AUDIO_MUX = ROOT / "work/ready_episode_segments/S01E36/next_summary_v001/remaster/S01E36_next_summary_remaster.mkv"
INPUT_TIMECODE_FPS = Fraction(30000, 1001)
TARGET_FPS = Fraction(24000, 1001)


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

    @property
    def source(self) -> Path:
        return SOURCE_ROOT / self.source_name

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
    SceneSpec("01", "Super Dimension Cavalry Southern Cross - 1x13 - Triple Mirror-30fps.mp4", 2, 26, 17, 2, 28, 20),
    SceneSpec("02", "Super Dimension Cavalry Southern Cross - 1x10 - Outsider-24fps.mp4", 15, 47, 4, 15, 48, 19),
    SceneSpec("03", "Super Dimension Cavalry Southern Cross - 1x01 - Prisoner.mkv", 24, 4, 21, 24, 8, 28),
    SceneSpec("04", "Super Dimension Cavalry Southern Cross - 1x12 - Lost Memory-24fps.mp4", 13, 17, 27, 13, 19, 11),
    SceneSpec("05", "Super Dimension Cavalry Southern Cross - 1x10 - Outsider-24fps.mp4", 15, 38, 27, 15, 41, 21),
]


def timecode_to_seconds(minutes: int, seconds: int, frame: int) -> float:
    return float(minutes * 60 + seconds + Fraction(frame, 1) / INPUT_TIMECODE_FPS)


def run_cmd(cmd: list[str], *, run: bool) -> None:
    print(shlex.join(cmd), flush=True)
    if run:
        subprocess.run(cmd, check=True)


def media_duration(path: Path) -> float:
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
        ).strip()
    )


def source_width(path: Path) -> int:
    return int(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
        ).strip()
    )


def cut_scene_cmd(scene: SceneSpec, output: Path) -> list[str]:
    width = source_width(scene.source)
    if width >= 3840:
        video_filter = "crop=2880:2160:480:0,scale=1440:1080:flags=lanczos,setsar=1,fps=24000/1001,format=yuv420p"
    else:
        video_filter = "scale=1440:1080:flags=lanczos,setsar=1,fps=24000/1001,format=yuv420p"
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
        video_filter,
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


def build_video_cmd(clips: list[Path], output: Path) -> list[str]:
    scene_05_duration = media_duration(clips[-1])
    head_black = frame_seconds(20)
    fade_in_duration = frame_seconds(10)
    fade_out_duration = frame_seconds(10)
    fade_out_start = max(0.0, scene_05_duration - frame_seconds(8))
    post_scene_fade_black = frame_seconds(2)
    tail_black = frame_seconds(20)
    inputs = sum((["-i", str(clip)] for clip in clips), [])
    labels = []
    filters = [
        f"color=c=black:s=1440x1080:r=24000/1001:d={head_black:.9f}[headblack]",
        f"color=c=black:s=1440x1080:r=24000/1001:d={post_scene_fade_black:.9f}[fadeblack]",
        f"color=c=black:s=1440x1080:r=24000/1001:d={tail_black:.9f}[tailblack]",
    ]
    for index in range(len(clips)):
        if index == 0:
            filters.append(
                f"[{index}:v]setpts=PTS-STARTPTS,format=yuv420p,"
                f"fade=t=in:st=0:d={fade_in_duration:.9f}[v{index}]"
            )
        elif index == len(clips) - 1:
            filters.append(
                f"[{index}:v]setpts=PTS-STARTPTS,format=yuv420p[v{index}raw]"
            )
            filters.append(
                f"[v{index}raw][fadeblack]concat=n=2:v=1:a=0,"
                f"fade=t=out:st={fade_out_start:.9f}:d={fade_out_duration:.9f}[v{index}]"
            )
        else:
            filters.append(f"[{index}:v]setpts=PTS-STARTPTS,format=yuv420p[v{index}]")
        labels.append(f"[v{index}]")
    filter_complex = ";".join(filters) + ";" + "[headblack]" + "".join(labels) + "[tailblack]" + f"concat=n={len(clips)+2}:v=1:a=0,setsar=1[v]"
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map_chapters",
        "-1",
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


def mux_review_cmd(video: Path, output: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video),
        "-i",
        str(READY_AUDIO_MUX),
        "-map",
        "0:v:0",
        "-map",
        "1:a",
        "-map_chapters",
        "-1",
        "-c",
        "copy",
        "-metadata",
        "title=Robotech S01E36 Southern Cross Source Next Summary Review",
        "-metadata:s:v:0",
        "language=eng",
        "-metadata:s:v:0",
        "title=Southern Cross Source Summary Video",
        str(output),
    ]


def pad_to_reference_duration_cmd(video: Path, output: Path) -> list[str]:
    reference_duration = media_duration(READY_AUDIO_MUX)
    video_duration = media_duration(video)
    pad_duration = max(0.0, reference_duration - video_duration)
    if pad_duration <= 0.001:
        return [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            str(output),
        ]
    filter_complex = (
        f"color=c=black:s=1440x1080:r=24000/1001:d={pad_duration:.9f}[black];"
        "[0:v]setpts=PTS-STARTPTS,format=yuv420p[v0];"
        "[v0][black]concat=n=2:v=1:a=0,setsar=1[v]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map_chapters",
        "-1",
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


def frame_seconds(frames: int) -> float:
    return float(Fraction(frames, 1) / TARGET_FPS)


def write_manifest(out_root: Path, clips: list[Path], video: Path, mux: Path, padded_video: Path, padded_mux: Path) -> None:
    reference_duration = media_duration(READY_AUDIO_MUX)
    video_duration = media_duration(video)
    payload = {
        "kind": "s01e36_next_summary_southern_cross_source_review",
        "input_timecode_fps": "30000/1001",
        "target_fps": "24000/1001",
        "duration_check": {
            "ready_audio_mux_duration_seconds": round(reference_duration, 6),
            "southern_cross_video_duration_seconds": round(video_duration, 6),
            "missing_visual_seconds_before_padding": round(max(0.0, reference_duration - video_duration), 6),
        },
        "notes": [
            "This is a comparison package only and does not overwrite the approved S01E36 ready segment.",
            "User-provided timestamps are interpreted as minutes:seconds:frame at 30000/1001 because some frame values exceed 24fps.",
            "3840x2160 clips are center-cropped to 2880x2160 before scaling to 1440x1080; 2880x2160 sources are scaled directly.",
            "Fade-in/out framing matches the previous reconstructed summary approach.",
            "The raw Southern Cross scene ranges are shorter than the current ready audio segment, so a padded review variant is also written for audio comparison.",
        ],
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "source_name": scene.source_name,
                "source": str(scene.source),
                "source_start_timecode": f"{scene.start_minutes:02d}:{scene.start_seconds:02d}:{scene.start_frame:02d}",
                "source_end_timecode": f"{scene.end_minutes:02d}:{scene.end_seconds:02d}:{scene.end_frame:02d}",
                "start_seconds": round(scene.start, 6),
                "end_seconds": round(scene.end, 6),
                "duration_seconds": round(scene.duration, 6),
                "clip": str(clips[index]),
            }
            for index, scene in enumerate(SCENES)
        ],
        "outputs": {
            "video_24fps_lossless": str(video),
            "review_mux_24fps": str(mux),
            "video_24fps_lossless_padded_to_ready_audio": str(padded_video),
            "review_mux_24fps_padded_to_ready_audio": str(padded_mux),
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    cut_dir = args.out_root / "cuts_1440_lossless"
    assembled_dir = args.out_root / "assembled"
    if args.run:
        cut_dir.mkdir(parents=True, exist_ok=True)
        assembled_dir.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        if not scene.source.is_file():
            raise SystemExit(f"Missing source: {scene.source}")
    if not READY_AUDIO_MUX.is_file():
        raise SystemExit(f"Missing audio source mux: {READY_AUDIO_MUX}")

    clips = []
    for scene in SCENES:
        clip = cut_dir / f"{scene.scene_id}_{scene.source.stem}_1440_lossless.mkv"
        clips.append(clip)
        run_cmd(cut_scene_cmd(scene, clip), run=args.run)

    video = assembled_dir / "S01E36_next_summary_southern_cross_source_1440x1080_24fps_lossless.mkv"
    mux = assembled_dir / "S01E36_next_summary_southern_cross_source_review_24fps.mkv"
    padded_video = assembled_dir / "S01E36_next_summary_southern_cross_source_1440x1080_24fps_padded_to_ready_audio_lossless.mkv"
    padded_mux = assembled_dir / "S01E36_next_summary_southern_cross_source_review_24fps_padded_to_ready_audio.mkv"
    run_cmd(build_video_cmd(clips, video), run=args.run)
    run_cmd(mux_review_cmd(video, mux), run=args.run)
    run_cmd(pad_to_reference_duration_cmd(video, padded_video), run=args.run)
    run_cmd(mux_review_cmd(padded_video, padded_mux), run=args.run)
    if args.run:
        write_manifest(args.out_root, clips, video, mux, padded_video, padded_mux)
        print(f"review_mux={mux}")
        print(f"padded_review_mux={padded_mux}")
        print(f"video={video}")
        print(f"padded_video={padded_video}")
        print(f"manifest={args.out_root / 'manifest.json'}")
    else:
        print(f"would write outputs under {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
