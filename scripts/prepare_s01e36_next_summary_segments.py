#!/usr/bin/env python3
"""Prepare final S01E36 next-summary segments for episode-final-build.

Unlike normal ready audio patches, S01E36's missing next-episode summary is a
complete video segment. This script prepares one concat-ready MKV per target
episode variant, with the same four audio tracks as final episode outputs.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "work/review/S01E36_next_summary_reconstruction_001"
READY_ROOT = ROOT / "work/ready_episode_segments/S01E36/next_summary_v001"
ORIGINAL_VIDEO = WORK_ROOT / "assembled_v001/S01E36_next_summary_reconstructed_review.mkv"
REALCUGAN_VIDEO = (
    WORK_ROOT
    / "video_ai_tests/realcugan_ncnn_nodenoise_s2_n0_001/realcugan_ncnn_nodenoise_s2_n0_001_with_audio.mkv"
)
ORIGINAL_RIFE_49FPS_VIDEO = (
    WORK_ROOT
    / "video_ai_tests/rife_original_49fps_001/rife_original_49fps_001.mkv"
)
REALCUGAN_RIFE_50FPS_VIDEO = (
    WORK_ROOT
    / "video_ai_tests/rife_realcugan_nodenoise_50fps_001/rife_realcugan_nodenoise_50fps_001.mkv"
)
ENGLISH_51 = WORK_ROOT / "assembled_v001/S01E36_next_summary_english_dvd_51.ac3"
DEVOICED_51 = WORK_ROOT / "assembled_v001/S01E36_next_summary_english_devoiced_bed_51.ac3"
OLD_SPANISH_TTS = (
    ROOT
    / "generated_audio/next_episode_summary/S01E36/summary_v001/"
    "S01E36_summary_v001_selected_speed110_nooverlap_gap0p01_balanced_gainplus4db_preview.wav"
)
NEW_NARRATOR_DIR = ROOT / "generated_audio/newnarrator/S01E36/nextepisodesummary"


VARIANTS = {
    "remaster": {
        "source": ORIGINAL_VIDEO,
        "rate": "24000/1001",
        "profile": "main",
        "crf": "8",
        "title": "S01E36 Next Summary Remaster 24fps",
    },
    "remaster_49fps": {
        "source": ORIGINAL_RIFE_49FPS_VIDEO,
        "rate": "48000/1001",
        "profile": "high",
        "crf": "8",
        "title": "S01E36 Next Summary Remaster 49fps",
    },
    "ai_remaster": {
        "source": REALCUGAN_RIFE_50FPS_VIDEO,
        "rate": "50/1",
        "profile": "main",
        "crf": "8",
        "title": "S01E36 Next Summary AI Remaster 50fps",
    },
}


def run_cmd(cmd: list[str], *, run: bool) -> None:
    print(shlex.join(cmd), flush=True)
    if run:
        subprocess.run(cmd, check=True)


def media_duration(path: Path) -> float:
    result = subprocess.check_output(
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
    )
    return float(result.strip())


def ensure_inputs(variant_sources: dict[str, Path]) -> None:
    missing = [path for path in (ORIGINAL_VIDEO, REALCUGAN_VIDEO, ENGLISH_51, DEVOICED_51, OLD_SPANISH_TTS) if not path.is_file()]
    missing.extend(path for path in variant_sources.values() if not path.is_file())
    missing.extend(path for path in new_narrator_phrase_files() if not path.is_file())
    if missing:
        raise SystemExit("Missing S01E36 next-summary input(s):\n" + "\n".join(str(path) for path in missing))


def new_narrator_phrase_files() -> list[Path]:
    return [
        NEW_NARRATOR_DIR / "01/01-S01E36_NextEpSummaryNewSpaDub_v01.wav",
        NEW_NARRATOR_DIR / "02/02-S01E36_NextEpSummaryNewSpaDub_v01.wav",
        NEW_NARRATOR_DIR / "03/03-S01E36_NextEpSummaryNewSpaDub_v01.wav",
    ]


def assemble_new_narrator_cmd(output: Path, *, speed_percent: float, gap_seconds: float, voice_gain_db: float) -> list[str]:
    phrases = new_narrator_phrase_files()
    speed = speed_percent / 100.0
    filters = []
    concat_inputs = []
    for index, _phrase in enumerate(phrases):
        filters.append(
            f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp,"
            "pan=stereo|c0=c0|c1=c1,"
            f"loudnorm=I=-18:TP=-1.5:LRA=11,volume={voice_gain_db:+.3f}dB"
            f"[p{index}]"
        )
        concat_inputs.append(f"[p{index}]")
        if index < len(phrases) - 1:
            gap_label = f"gap{index}"
            filters.append(f"anullsrc=channel_layout=stereo:sample_rate=48000:d={gap_seconds:.6f}[{gap_label}]")
            concat_inputs.append(f"[{gap_label}]")
    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(concat_inputs)}:v=0:a=1,atempo={speed:.8f},alimiter=limit=0.95[out]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        *sum((["-i", str(path)] for path in phrases), []),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-c:a",
        "pcm_s24le",
        str(output),
    ]


def video_variant_cmd(source: Path, output: Path, *, rate: str, profile: str, crf: str) -> list[str]:
    vf = f"fps={rate},scale=1440:1080:flags=lanczos,format=yuv420p"
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        crf,
        "-profile:v",
        profile,
        "-pix_fmt",
        "yuv420p",
        "-r",
        rate,
        "-an",
        str(output),
    ]


def spanish_51_cmd(output: Path, duration: float, *, bed_gain_db: float, voice_gain_db: float) -> list[str]:
    filter_complex = (
        f"[0:a]volume={bed_gain_db:+.3f}dB,channelsplit=channel_layout=5.1(side)[FL][FR][FCBED][LFE][SL][SR];"
        "[1:a]aresample=48000,aformat=sample_fmts=fltp,"
        f"pan=mono|c0=0.5*c0+0.5*c1,volume={voice_gain_db:+.3f}dB[VO];"
        "[FCBED][VO]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[FCNEW];"
        "[FL][FR][FCNEW][LFE][SL][SR]"
        "join=inputs=6:channel_layout=5.1(side):"
        "map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-SL|5.0-SR,"
        f"apad,atrim=0:{duration:.6f},asetpts=N/SR/TB[out51]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(DEVOICED_51),
        "-i",
        str(OLD_SPANISH_TTS),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out51]",
        "-c:a",
        "ac3",
        "-b:a",
        "640k",
        str(output),
    ]


def spanish_stereo_cmd(voice: Path, output: Path, duration: float, *, bed_gain_db: float, voice_gain_db: float) -> list[str]:
    filter_complex = (
        "[0:a]aresample=48000,aformat=sample_fmts=fltp,"
        "pan=stereo|c0=0.75*FL+0.5*FC+0.5*SL+0.25*LFE|c1=0.75*FR+0.5*FC+0.5*SR+0.25*LFE[bed];"
        f"[bed]volume={bed_gain_db:+.3f}dB[bedgain];"
        f"[1:a]aresample=48000,aformat=sample_fmts=fltp,pan=stereo|c0=c0|c1=c1,volume={voice_gain_db:+.3f}dB[vo];"
        "[bedgain][vo]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95,"
        f"apad,atrim=0:{duration:.6f},asetpts=N/SR/TB[out]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(DEVOICED_51),
        "-i",
        str(voice),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:a",
        "ac3",
        "-b:a",
        "224k",
        str(output),
    ]


def pad_english_cmd(output: Path, duration: float) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(ENGLISH_51),
        "-af",
        f"apad,atrim=0:{duration:.6f},asetpts=N/SR/TB",
        "-c:a",
        "ac3",
        "-b:a",
        "448k",
        str(output),
    ]


def mux_segment_cmd(video: Path, english: Path, spa51: Path, spa1: Path, spa2: Path, output: Path, title: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video),
        "-i",
        str(english),
        "-i",
        str(spa51),
        "-i",
        str(spa1),
        "-i",
        str(spa2),
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
        "-c",
        "copy",
        "-metadata",
        f"title={title}",
        "-metadata:s:v:0",
        "language=eng",
        "-metadata:s:v:0",
        "title=Reconstructed Next Episode Summary",
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
        str(output),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=READY_ROOT)
    parser.add_argument("--new-dub-speed-percent", type=float, default=110.0)
    parser.add_argument("--new-dub-gap-seconds", type=float, default=0.01)
    parser.add_argument("--spanish-bed-gain-db", type=float, default=2.0)
    parser.add_argument("--spanish-voice-gain-db", type=float, default=4.0)
    parser.add_argument("--remaster-49fps-source", type=Path, default=ORIGINAL_RIFE_49FPS_VIDEO)
    parser.add_argument("--ai-remaster-50fps-source", type=Path, default=REALCUGAN_RIFE_50FPS_VIDEO)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    variant_sources = {
        "remaster": ORIGINAL_VIDEO,
        "remaster_49fps": args.remaster_49fps_source,
        "ai_remaster": args.ai_remaster_50fps_source,
    }
    ensure_inputs(variant_sources)
    out_dir = args.out_dir
    audio_dir = out_dir / "audio"
    video_dir = out_dir / "video"
    if args.run:
        audio_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

    new_narrator = audio_dir / "S01E36_next_summary_new_spanish_dub_joined_speed110.wav"
    run_cmd(
        assemble_new_narrator_cmd(
            new_narrator,
            speed_percent=args.new_dub_speed_percent,
            gap_seconds=args.new_dub_gap_seconds,
            voice_gain_db=args.spanish_voice_gain_db,
        ),
        run=args.run,
    )

    outputs: dict[str, dict[str, str]] = {}
    for variant, spec in VARIANTS.items():
        variant_dir = out_dir / variant
        if args.run:
            variant_dir.mkdir(parents=True, exist_ok=True)
        video_only = video_dir / f"S01E36_next_summary_{variant}_video.mkv"
        run_cmd(
            video_variant_cmd(
                variant_sources[variant],
                video_only,
                rate=str(spec["rate"]),
                profile=str(spec["profile"]),
                crf=str(spec["crf"]),
            ),
            run=args.run,
        )
        duration = media_duration(video_only) if args.run else 17.248
        english = audio_dir / f"S01E36_next_summary_{variant}_english_51.ac3"
        spa51 = audio_dir / f"S01E36_next_summary_{variant}_spanish_restored_51.ac3"
        spa1 = audio_dir / f"S01E36_next_summary_{variant}_spanish1_old_stereo.ac3"
        spa2 = audio_dir / f"S01E36_next_summary_{variant}_spanish2_newer_stereo.ac3"
        run_cmd(pad_english_cmd(english, duration), run=args.run)
        run_cmd(spanish_51_cmd(spa51, duration, bed_gain_db=args.spanish_bed_gain_db, voice_gain_db=args.spanish_voice_gain_db), run=args.run)
        run_cmd(spanish_stereo_cmd(new_narrator, spa1, duration, bed_gain_db=args.spanish_bed_gain_db, voice_gain_db=args.spanish_voice_gain_db), run=args.run)
        run_cmd(spanish_stereo_cmd(new_narrator, spa2, duration, bed_gain_db=args.spanish_bed_gain_db, voice_gain_db=args.spanish_voice_gain_db), run=args.run)
        output = variant_dir / f"S01E36_next_summary_{variant}.mkv"
        run_cmd(mux_segment_cmd(video_only, english, spa51, spa1, spa2, output, str(spec["title"])), run=args.run)
        outputs[variant] = {
            "path": str(output),
            "segment": str(output),
            "video_source": str(variant_sources[variant]),
            "rate": str(spec["rate"]),
            "duration_seconds": round(duration, 6),
        }

    manifest = {
        "kind": "robotech_prepared_episode_segment",
        "episode": "S01E36",
        "episode_id": "S01E36",
        "segment_id": "next_summary_v001",
        "insert": "before_end_credits",
        "description": "Prepared reconstructed next-episode summary segment for S01E36.",
        "suppresses_ready_audio_patch_ids": ["s01e36_next_episode_summary_tts_v001"],
        "tracks": [
            "English Original 5.1",
            "Spanish Restored Original Dub 5.1",
            "Spanish Original Dub Restored Stereo",
            "Spanish Redubbing Original Stereo",
        ],
        "old_spanish_tts": str(OLD_SPANISH_TTS),
        "new_spanish_dub_joined": str(new_narrator),
        "spanish_bed_gain_db": args.spanish_bed_gain_db,
        "spanish_voice_gain_db": args.spanish_voice_gain_db,
        "subtitle_sources": {
            "english_clean": str(WORK_ROOT / "subtitles/S01E36_next_summary_english_phrases_relative.srt"),
            "spanish_translated": str(WORK_ROOT / "subtitles/S01E36_next_summary_spanish_phrases_relative.srt"),
        },
        "outputs": outputs,
    }
    if args.run:
        (out_dir / "segment.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"segment_manifest={out_dir / 'segment.json'}")
    for variant, data in outputs.items():
        print(f"{variant}: {data['segment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
