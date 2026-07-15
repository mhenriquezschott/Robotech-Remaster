#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "work/review/opening_audio_rebuild_001"
DEFAULT_OC = ROOT / "Robotech/oc-ec/oc/assets/intromasterAI48fps1440x1080cropallac3_v56.mkv"
DEFAULT_OC_ENG51 = ROOT / "Robotech/oc-ec/oc/assets/track01eng51.ac3"
DEFAULT_OC_SPA1 = ROOT / "Robotech/oc-ec/oc/assets/track02spa1ori.ac3"
DEFAULT_MAIN_TITLE = ROOT / "Robotech/Robotech Perfect Soundtrack/Robotech Perfect Collection 1.1/01 Main Title.mp3"
DEFAULT_TV_EP01 = Path(
    "/mnt/usb-Seagate_Expansion_HDD_00000000NT17VSPP-0:0-part2/Multimedia/Videos/Series/"
    "Robotech [Esp]_TvQuality/Season 1/Robotech - 1x01 - Boobytrap.avi"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare repeatable opening-credit audio rebuild evaluation sources and first-pass FFmpeg stems."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--opening-video", type=Path, default=DEFAULT_OC)
    parser.add_argument("--opening-eng51", type=Path, default=DEFAULT_OC_ENG51)
    parser.add_argument("--opening-spa1", type=Path, default=DEFAULT_OC_SPA1)
    parser.add_argument("--main-title", type=Path, default=DEFAULT_MAIN_TITLE)
    parser.add_argument("--tv-copy", type=Path, default=DEFAULT_TV_EP01)
    parser.add_argument("--duration", type=float, default=91.304, help="Opening duration to extract in seconds.")
    parser.add_argument("--tv-start", type=float, default=0.0, help="Start time of TV-copy intro.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--separator-commands", action="store_true", help="Write audio-separator commands for key stems.")
    args = parser.parse_args()

    out = args.out_dir
    sources = out / "sources"
    stems = out / "ffmpeg_stems"
    commands = out / "commands"
    sources.mkdir(parents=True, exist_ok=True)
    stems.mkdir(parents=True, exist_ok=True)
    commands.mkdir(parents=True, exist_ok=True)

    require_file(args.opening_video)
    require_file(args.opening_eng51)
    require_file(args.opening_spa1)
    require_file(args.main_title)
    require_file(args.tv_copy)

    cmds: list[list[str]] = []
    # Current restored opening: stream inventory is diagnostic; stream-specific
    # WAVs let us hear what each embedded source is really carrying.
    cmds.append(ffmpeg_audio(args.opening_video, sources / "01_current_opening_stream1_51.wav", ["-map", "0:a:0", "-t", f"{args.duration:.6f}", "-ar", "48000"]))
    cmds.append(ffmpeg_audio(args.opening_video, sources / "02_current_opening_stream2_stereo.wav", ["-map", "0:a:1", "-t", f"{args.duration:.6f}", "-ar", "48000"]))
    cmds.append(ffmpeg_audio(args.opening_video, sources / "03_current_opening_stream3_stereo.wav", ["-map", "0:a:2", "-t", f"{args.duration:.6f}", "-ar", "48000"]))
    cmds.append(ffmpeg_audio(args.opening_eng51, sources / "04_asset_track01_eng51.wav", ["-t", f"{args.duration:.6f}", "-ar", "48000"]))
    cmds.append(ffmpeg_audio(args.opening_spa1, sources / "05_asset_track02_spa1_original_stereo.wav", ["-t", f"{args.duration:.6f}", "-ar", "48000"]))
    cmds.append(ffmpeg_audio(args.main_title, sources / "06_soundtrack_main_title_full.wav", ["-map", "0:a:0", "-ar", "48000"]))
    cmds.append(ffmpeg_audio(args.main_title, sources / "07_soundtrack_main_title_opening_length.wav", ["-map", "0:a:0", "-t", f"{args.duration:.6f}", "-ar", "48000"]))
    cmds.append(ffmpeg_audio(args.tv_copy, sources / "08_tv_copy_ep01_intro_stereo.wav", ["-ss", f"{args.tv_start:.6f}", "-t", f"{args.duration:.6f}", "-map", "0:a:0", "-ar", "48000"]))

    spa1 = sources / "05_asset_track02_spa1_original_stereo.wav"
    tv = sources / "08_tv_copy_ep01_intro_stereo.wav"
    current_mix = sources / "02_current_opening_stream2_stereo.wav"
    for label, source in (("spa1_asset", spa1), ("tvcopy_ep01", tv), ("current_stream2", current_mix)):
        cmds.extend(first_pass_stems(source, stems, label))

    manifest = {
        "goal": "Opening-credit audio rebuild evaluation: isolate narrator/effects from current and TV-copy intros, compare with official soundtrack Main Title.",
        "sources": {
            "opening_video": str(args.opening_video),
            "opening_eng51": str(args.opening_eng51),
            "opening_spa1": str(args.opening_spa1),
            "main_title": str(args.main_title),
            "tv_copy": str(args.tv_copy),
            "duration": args.duration,
            "tv_start": args.tv_start,
        },
        "outputs": {
            "sources": str(sources),
            "ffmpeg_stems": str(stems),
            "commands": str(commands),
        },
        "notes": [
            "FFmpeg stems are only triage candidates; judge by listening and spectrogram.",
            "Use audio-separator commands for narrator/voice removal; music/SFX separation may need model shootout.",
            "Soundtrack alignment/cancellation is intentionally not automatic yet because offset/tempo differences must be measured first.",
        ],
        "commands": [cmd_to_text(cmd) for cmd in cmds],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.md").write_text(readme_text(out), encoding="utf-8")

    if args.separator_commands:
        write_separator_commands(commands, sources)

    if args.run:
        for cmd in cmds:
            run(cmd)
    else:
        for cmd in cmds:
            print(cmd_to_text(cmd))

    print(f"review_dir={out}")
    print(f"manifest={out / 'manifest.json'}")
    return 0


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")


def ffmpeg_audio(input_path: Path, output_path: Path, input_opts: list[str]) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        *input_opts,
        "-vn",
        "-c:a",
        "pcm_s24le",
        str(output_path),
    ]


def first_pass_stems(source: Path, out_dir: Path, label: str) -> list[list[str]]:
    return [
        filter_audio(source, out_dir / f"{label}_mid_mono.wav", "pan=mono|c0=0.5*c0+0.5*c1"),
        filter_audio(source, out_dir / f"{label}_side_mono.wav", "pan=mono|c0=0.5*c0-0.5*c1"),
        filter_audio(source, out_dir / f"{label}_voice_band_180_4200.wav", "highpass=f=180,lowpass=f=4200"),
        filter_audio(source, out_dir / f"{label}_presence_sfx_2500_9000.wav", "highpass=f=2500,lowpass=f=9000"),
        filter_audio(source, out_dir / f"{label}_dialoguenhance.wav", "dialoguenhance"),
        filter_audio(source, out_dir / f"{label}_afftdn_light.wav", "afftdn=nr=8:nf=-40"),
        filter_audio(source, out_dir / f"{label}_stereo_width_wide.wav", "stereotools=mlev=0.8:slev=1.6"),
    ]


def filter_audio(source: Path, output_path: Path, audio_filter: str) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        "-af",
        audio_filter,
        "-ar",
        "48000",
        "-c:a",
        "pcm_s24le",
        str(output_path),
    ]


def write_separator_commands(commands_dir: Path, sources_dir: Path) -> None:
    key_sources = [
        sources_dir / "05_asset_track02_spa1_original_stereo.wav",
        sources_dir / "08_tv_copy_ep01_intro_stereo.wav",
        sources_dir / "02_current_opening_stream2_stereo.wav",
    ]
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for source in key_sources:
        out_dir = commands_dir.parent / "separator" / source.stem
        lines.extend(
            [
                "scripts/robotech-ai separate-voice \\",
                f"  {shell_quote(source)} \\",
                "  --engine audio-separator \\",
                "  --model melband_roformer_instvoc_duality_v1.ckpt \\",
                f"  --out-dir {shell_quote(out_dir)} \\",
                "  --single-stem Vocals \\",
                "  --sample-rate 48000 \\",
                "  --run",
                "",
                "scripts/robotech-ai separate-voice \\",
                f"  {shell_quote(source)} \\",
                "  --engine audio-separator \\",
                "  --model melband_roformer_instvoc_duality_v1.ckpt \\",
                f"  --out-dir {shell_quote(out_dir / 'instrumental')} \\",
                "  --single-stem Instrumental \\",
                "  --sample-rate 48000 \\",
                "  --run",
                "",
            ]
        )
    path = commands_dir / "run_separator_opening_candidates.sh"
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def readme_text(out: Path) -> str:
    return f"""# Opening Audio Rebuild Evaluation

This folder is for testing whether we can rebuild the opening audio from better
music plus recovered narration/effects.

Start here:

- `sources/07_soundtrack_main_title_opening_length.wav`: CD soundtrack source trimmed to the current opening length.
- `sources/05_asset_track02_spa1_original_stereo.wav`: current restored original Latin American opening audio.
- `sources/08_tv_copy_ep01_intro_stereo.wav`: old TV-copy intro audio.
- `ffmpeg_stems/*_mid_mono.wav` and `*_side_mono.wav`: simple mid/side separation checks.
- `ffmpeg_stems/*_dialoguenhance.wav`: FFmpeg dialogue-emphasis baseline.
- `ffmpeg_stems/*_presence_sfx_2500_9000.wav`: crude high-frequency SFX/presence band.

Optional separator commands are written under `commands/` when
`--separator-commands` is used. Those are the next pass for isolating the
Spanish “Robotech” narrator and de-voiced music/effects beds.

Review manifest:

`{out / "manifest.json"}`
"""


def shell_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "'\"'\"'") + "'"


def run(cmd: list[str]) -> None:
    print(cmd_to_text(cmd), flush=True)
    subprocess.run(cmd, check=True)


def cmd_to_text(cmd: list[str]) -> str:
    return " ".join(shell_quote(Path(part)) if "/" in part and not part.startswith("-") else part for part in cmd)


if __name__ == "__main__":
    raise SystemExit(main())
