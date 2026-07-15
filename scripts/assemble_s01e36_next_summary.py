#!/usr/bin/env python3
"""Assemble the reconstructed S01E36 next-episode summary package.

This script starts from the already prepared 1440x1080 lossless scene clips and
creates the review-ready summary assets: faded video, English 5.1 AC3, de-voiced
5.1 bed AC3, English/Spanish relative SRTs, and a Spanish Qwen3-TTS phrase plan.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "work/review/S01E36_next_summary_reconstruction_001"
GENERATED_TTS_ROOT = ROOT / "generated_audio/next_episode_summary/S01E36/summary_v001"
FPS = Fraction(24000, 1001)


SPANISH_PHRASES = [
    {
        "number": 1,
        "text": "Una nueva generación de guerreros ha sido llamada a la batalla.",
    },
    {
        "number": 2,
        "text": (
            "Enfrentan la amenaza de los Maestros de la Robotecnia, "
            "que han viajado quince años hasta la Tierra para recuperar "
            "su Fábrica de Protocultura perdida."
        ),
    },
    {
        "number": 3,
        "text": "No se pierdan La historia de Dana, el próximo capítulo en la saga de Robotech.",
    },
]


@dataclass(frozen=True)
class AssemblyPaths:
    root: Path
    crop_dir: Path
    assembled_dir: Path
    audio_dir: Path
    subtitle_dir: Path
    tts_dir: Path


def run_cmd(cmd: list[str], *, run: bool) -> None:
    print(shlex.join(cmd), flush=True)
    if run:
        subprocess.run(cmd, check=True)


def frame_seconds(frames: int) -> float:
    return float(Fraction(frames, 1) / FPS)


def media_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    return float(subprocess.check_output(cmd, text=True).strip())


def srt_timestamp(seconds: float) -> str:
    milliseconds_total = int(round(max(seconds, 0.0) * 1000))
    seconds_total, milliseconds = divmod(milliseconds_total, 1000)
    minutes_total, seconds_part = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"


def read_srt_cue_times(path: Path) -> list[tuple[int, float, float, str]]:
    cues: list[tuple[int, float, float, str]] = []
    blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        index = int(lines[0].strip())
        start_text, end_text = [part.strip() for part in lines[1].split("-->")]
        text = "\n".join(lines[2:]).strip()
        cues.append((index, parse_srt_timestamp(start_text), parse_srt_timestamp(end_text), text))
    return cues


def parse_srt_timestamp(value: str) -> float:
    time_part, millis_part = value.replace(".", ",").split(",", 1)
    hours_text, minutes_text, seconds_text = time_part.split(":")
    return int(hours_text) * 3600 + int(minutes_text) * 60 + int(seconds_text) + int(millis_part[:3]) / 1000


def write_srt(path: Path, cues: list[tuple[int, float, float, str]]) -> None:
    blocks = []
    for index, start, end, text in cues:
        blocks.append(f"{index}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def build_video_cmd(paths: AssemblyPaths, output: Path) -> list[str]:
    clips = [
        paths.crop_dir / "01_Robotech-S02E01_crop1440_lossless.mkv",
        paths.crop_dir / "02_Robotech-S02E11_crop1440_lossless.mkv",
        paths.crop_dir / "03_Robotech-S02E01_crop1440_lossless.mkv",
        paths.crop_dir / "04_Robotech-S02E11_crop1440_lossless.mkv",
    ]
    for clip in clips:
        if not clip.is_file():
            raise SystemExit(f"Missing crop-lossless clip: {clip}")
    scene_04_duration = media_duration(clips[3])
    head_black = frame_seconds(20)
    fade_in_duration = frame_seconds(10)
    fade_out_duration = frame_seconds(10)
    fade_out_start = max(0.0, scene_04_duration - frame_seconds(8))
    post_scene_fade_black = frame_seconds(2)
    tail_black = frame_seconds(20)
    filter_complex = (
        f"color=c=black:s=1440x1080:r=24000/1001:d={head_black:.9f}[headblack];"
        f"color=c=black:s=1440x1080:r=24000/1001:d={post_scene_fade_black:.9f}[fadeblack];"
        f"color=c=black:s=1440x1080:r=24000/1001:d={tail_black:.9f}[tailblack];"
        f"[0:v]setpts=PTS-STARTPTS,format=yuv420p,"
        f"fade=t=in:st=0:d={fade_in_duration:.9f}[v0];"
        "[1:v]setpts=PTS-STARTPTS,format=yuv420p[v1];"
        "[2:v]setpts=PTS-STARTPTS,format=yuv420p[v2];"
        f"[3:v]setpts=PTS-STARTPTS,format=yuv420p[v3raw];"
        "[v3raw][fadeblack]concat=n=2:v=1:a=0,"
        f"fade=t=out:st={fade_out_start:.9f}:d={fade_out_duration:.9f}[v3];"
        "[headblack][v0][v1][v2][v3][tailblack]concat=n=6:v=1:a=0,setsar=1[v]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        *sum((["-i", str(clip)] for clip in clips), []),
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


def pad_audio_ac3_cmd(input_path: Path, output_path: Path, duration: float, bitrate: str = "448k") -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"atrim=0:{duration:.6f},apad=whole_dur={duration:.6f},atrim=0:{duration:.6f},asetpts=N/SR/TB",
        "-c:a",
        "ac3",
        "-b:a",
        bitrate,
        str(output_path),
    ]


def devoiced_bed_51_ac3_cmd(paths: AssemblyPaths, output_path: Path, duration: float) -> list[str]:
    original = paths.audio_dir / "S01E36_next_summary_dvd_eng51_original.wav"
    center_bed = paths.audio_dir / "S01E36_next_summary_dvd_center_devoiced_bed_v1.wav"
    if not original.is_file() or not center_bed.is_file():
        raise SystemExit(f"Missing audio bed inputs: {original} / {center_bed}")
    filter_complex = (
        "[0:a]pan=mono|c0=FL[FL];"
        "[0:a]pan=mono|c0=FR[FR];"
        "[0:a]pan=mono|c0=LFE[LFE];"
        "[0:a]pan=mono|c0=SL[SL];"
        "[0:a]pan=mono|c0=SR[SR];"
        "[1:a]pan=mono|c0=0.5*c0+0.5*c1[FCNEW];"
        "[FL][FR][FCNEW][LFE][SL][SR]"
        "join=inputs=6:channel_layout=5.1(side):"
        "map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-SL|5.0-SR,"
        f"atrim=0:{duration:.6f},apad=whole_dur={duration:.6f},atrim=0:{duration:.6f},asetpts=N/SR/TB[out51]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(original),
        "-i",
        str(center_bed),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out51]",
        "-c:a",
        "ac3",
        "-b:a",
        "448k",
        str(output_path),
    ]


def mux_review_cmd(video: Path, english_51: Path, devoiced_51: Path, output: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video),
        "-i",
        str(english_51),
        "-i",
        str(devoiced_51),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:a:0",
        "-map_chapters",
        "-1",
        "-c",
        "copy",
        "-metadata",
        "title=Robotech S01E36 Reconstructed Next Episode Summary",
        "-metadata:s:v:0",
        "language=eng",
        "-metadata:s:v:0",
        "title=Reconstructed Southern Cross Summary Video",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:0",
        "title=English DVD Summary 5.1",
        "-metadata:s:a:1",
        "language=eng",
        "-metadata:s:a:1",
        "title=English De-voiced 5.1 Bed",
        str(output),
    ]


def write_spanish_subtitle_and_plan(paths: AssemblyPaths) -> tuple[Path, Path]:
    english_srt = paths.subtitle_dir / "S01E36_next_summary_english_phrases_relative.srt"
    if not english_srt.is_file():
        raise SystemExit(f"Missing English relative SRT: {english_srt}")
    cues = read_srt_cue_times(english_srt)
    if len(cues) != len(SPANISH_PHRASES):
        raise SystemExit(f"Expected {len(SPANISH_PHRASES)} cues, got {len(cues)} from {english_srt}")
    spanish_cues = [
        (index, start, end, SPANISH_PHRASES[index - 1]["text"])
        for index, start, end, _text in cues
    ]
    spanish_srt = paths.subtitle_dir / "S01E36_next_summary_spanish_phrases_relative.srt"
    write_srt(spanish_srt, spanish_cues)

    english_seed = paths.root / "phrase_plan_english_seed.json"
    seed_data = json.loads(english_seed.read_text(encoding="utf-8"))
    phrase_plan = {
        **seed_data,
        "kind": "robotech_qwen3_tts_summary_plan",
        "source_language": "en",
        "target_language": "es",
        "phrases": [
            {
                **phrase,
                "text": SPANISH_PHRASES[int(phrase["number"]) - 1]["text"],
                "source_text": phrase["text"],
                "language": "es",
            }
            for phrase in seed_data["phrases"]
        ],
        "notes": [
            "Spanish target phrase plan for reconstructing S01E36 next-episode narrator with Qwen3-TTS.",
            "Times are relative to the reconstructed summary segment, not the full S01E36 episode timeline.",
            "The English seed remains available as phrase_plan_english_seed.json for reference only.",
        ],
    }
    phrase_plan_path = paths.root / "phrase_plan_spanish.json"
    phrase_plan_path.write_text(json.dumps(phrase_plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths.tts_dir.mkdir(parents=True, exist_ok=True)
    generated_phrase_plan = paths.tts_dir / "phrase_plan.json"
    generated_phrase_plan.write_text(json.dumps(phrase_plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return spanish_srt, generated_phrase_plan


def write_manifest(paths: AssemblyPaths, *, video: Path, mux: Path, english_51: Path, devoiced_51: Path, spanish_srt: Path, phrase_plan: Path) -> Path:
    manifest_path = paths.assembled_dir / "assembly_manifest.json"
    payload = {
        "kind": "robotech_s01e36_next_summary_assembled_v001",
        "fps": "24000/1001",
        "fade_in": {
            "black_frames_before_scene_01": 20,
            "fade_frames": 10,
        },
        "fade_out": {
            "starts_frames_before_scene_04_end": 8,
            "black_frames_after_scene_04_end_for_fade": 2,
            "tail_black_frames": 20,
            "implementation_note": (
                "Two explicit black frames are appended after scene 04 so the 10-frame "
                "fade reaches full black just after the scene image ends."
            ),
        },
        "outputs": {
            "video_lossless": str(video),
            "review_mux": str(mux),
            "english_51_ac3": str(english_51),
            "devoiced_bed_51_ac3": str(devoiced_51),
            "spanish_relative_srt": str(spanish_srt),
            "spanish_phrase_plan": str(phrase_plan),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    paths = AssemblyPaths(
        root=args.work_root,
        crop_dir=args.work_root / "crop1440_lossless",
        assembled_dir=args.work_root / "assembled_v001",
        audio_dir=args.work_root / "audio",
        subtitle_dir=args.work_root / "subtitles",
        tts_dir=GENERATED_TTS_ROOT,
    )
    for directory in (paths.assembled_dir, paths.subtitle_dir):
        if args.run:
            directory.mkdir(parents=True, exist_ok=True)

    video = paths.assembled_dir / "S01E36_next_summary_reconstructed_1440x1080_24fps_lossless.mkv"
    english_51 = paths.assembled_dir / "S01E36_next_summary_english_dvd_51.ac3"
    devoiced_51 = paths.assembled_dir / "S01E36_next_summary_english_devoiced_bed_51.ac3"
    mux = paths.assembled_dir / "S01E36_next_summary_reconstructed_review.mkv"

    run_cmd(build_video_cmd(paths, video), run=args.run)
    final_duration = media_duration(video) if args.run and video.is_file() else 17.227
    run_cmd(pad_audio_ac3_cmd(paths.audio_dir / "S01E36_next_summary_dvd_eng51_original.wav", english_51, final_duration), run=args.run)
    run_cmd(devoiced_bed_51_ac3_cmd(paths, devoiced_51, final_duration), run=args.run)
    run_cmd(mux_review_cmd(video, english_51, devoiced_51, mux), run=args.run)

    if args.run:
        spanish_srt, phrase_plan = write_spanish_subtitle_and_plan(paths)
        shutil.copy2(paths.root / "phrase_plan_english_seed.json", paths.tts_dir / "phrase_plan_english_seed.json")
        manifest = write_manifest(
            paths,
            video=video,
            mux=mux,
            english_51=english_51,
            devoiced_51=devoiced_51,
            spanish_srt=spanish_srt,
            phrase_plan=phrase_plan,
        )
        print(f"review_mux={mux}")
        print(f"video={video}")
        print(f"english_51={english_51}")
        print(f"devoiced_51={devoiced_51}")
        print(f"spanish_srt={spanish_srt}")
        print(f"spanish_phrase_plan={phrase_plan}")
        print(f"manifest={manifest}")
    else:
        print(f"would write outputs under {paths.assembled_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
