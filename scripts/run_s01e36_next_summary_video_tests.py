#!/usr/bin/env python3
"""Prepare and run video restoration/interpolation tests for S01E36 summary.

The reconstructed summary is intentionally kept clean and lossless-ish. This
script writes experiment outputs under the reconstruction review folder so AI
video tests can be compared without touching the approved source assembly.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "work/review/S01E36_next_summary_reconstruction_001"
DEFAULT_SOURCE = WORK_ROOT / "assembled_v001/S01E36_next_summary_reconstructed_1440x1080_24fps_lossless.mkv"
DEFAULT_REVIEW_AUDIO_SOURCE = WORK_ROOT / "assembled_v001/S01E36_next_summary_reconstructed_review.mkv"
DEFAULT_OUT_ROOT = WORK_ROOT / "video_ai_tests"
SOURCE_FPS = "24000/1001"
DOUBLE_FPS = "48000/1001"
APISR_DEFAULT_WEIGHT = ROOT / "soft/ai_video_tools/src/APISR/pretrained/2x_APISR_RRDB_GAN_generator.pth"
APISR_DEFAULT_PYTHON = ROOT / ".venv-video-apisr/bin/python"
ANIMESR_DEFAULT_PYTHON = ROOT / ".venv-video-animesr/bin/python"
RIFE_DEFAULT_PYTHON = ROOT / ".venv-video-rife/bin/python"
REALCUGAN_NCNN_DEFAULT_BIN = ROOT / "soft/ai_video_tools/bin/realcugan-ncnn-vulkan/realcugan-ncnn-vulkan"
RIFE_NCNN_DEFAULT_BIN = ROOT / "soft/ai_video_tools/bin/rife-ncnn-vulkan/rife-ncnn-vulkan"


@dataclass(frozen=True)
class TestPaths:
    root: Path
    frames_in: Path
    frames_out: Path
    frames_encode: Path
    output_video: Path
    output_review: Path
    manifest: Path


def run_cmd(cmd: list[str], *, run: bool, cwd: Path | None = None) -> None:
    print(shlex.join(cmd), flush=True)
    if run:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def shell_template_to_cmd(template: str, *, input_dir: Path, output_dir: Path, input_video: Path, output_video: Path) -> list[str]:
    rendered = template.format(
        root=str(ROOT),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        input_video=str(input_video),
        output_video=str(output_video),
    )
    return shlex.split(rendered)


def ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def paths_for(out_root: Path, label: str, suffix: str = ".mkv") -> TestPaths:
    root = out_root / label
    return TestPaths(
        root=root,
        frames_in=root / "frames_in",
        frames_out=root / "frames_out",
        frames_encode=root / "frames_encode",
        output_video=root / f"{label}{suffix}",
        output_review=root / f"{label}_with_audio{suffix}",
        manifest=root / "manifest.json",
    )


def extract_frames_cmd(source: Path, frames_in: Path, fps: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        str(frames_in / "%06d.png"),
    ]


def encode_frames_cmd(frames_out: Path, output: Path, fps: str, *, scale_to_source: bool, crf: int) -> list[str]:
    vf = "format=yuv420p"
    if scale_to_source:
        vf = "scale=1440:1080:flags=lanczos," + vf
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-framerate",
        fps,
        "-i",
        str(frames_out / "%06d.png"),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-r",
        fps,
        str(output),
    ]


def normalize_numbered_frames(source_dir: Path, encode_dir: Path, *, overwrite: bool) -> None:
    ensure_clean_dir(encode_dir, overwrite=overwrite)
    candidates = sorted(source_dir.glob("*.png"))
    if not candidates:
        raise SystemExit(f"No PNG frames found in backend output folder: {source_dir}")
    for index, source in enumerate(candidates, start=1):
        target = encode_dir / f"{index:06d}.png"
        if target.exists():
            target.unlink()
        target.symlink_to(source.resolve())


def mux_audio_cmd(video: Path, source: Path, output: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c",
        "copy",
        "-shortest",
        str(output),
    ]


def mux_review_audio(paths: TestPaths, args: argparse.Namespace) -> None:
    if not args.review_audio_source or not args.review_audio_source.is_file():
        return
    run_cmd(mux_audio_cmd(paths.output_video, args.review_audio_source, paths.output_review), run=args.run)


def find_binary(user_path: Path | None, names: list[str]) -> Path | None:
    if user_path:
        return user_path if user_path.exists() else None
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    candidates: list[Path] = []
    for name in names:
        candidates.extend((ROOT / "soft").rglob(name))
    linux_candidates = [path for path in candidates if path.is_file() and path.suffix.lower() != ".exe"]
    return linux_candidates[0] if linux_candidates else None


def default_tool_paths() -> dict[str, Path]:
    return {
        "realcugan_repo": ROOT / "soft/ai_video_tools/src/bilibili-ailab/Real-CUGAN",
        "apisr_repo": ROOT / "soft/ai_video_tools/src/APISR",
        "animesr_repo": ROOT / "soft/ai_video_tools/src/AnimeSR",
        "rife_repo": ROOT / "soft/ai_video_tools/src/ECCV2022-RIFE",
    }


def run_realcugan(args: argparse.Namespace) -> None:
    if args.realcugan_cmd:
        run_template_backend(args, "Real-CUGAN", args.realcugan_cmd)
        return
    binary = find_binary(args.realcugan_bin, ["realcugan-ncnn-vulkan"])
    if not binary and REALCUGAN_NCNN_DEFAULT_BIN.is_file():
        binary = REALCUGAN_NCNN_DEFAULT_BIN
    if not binary:
        raise SystemExit(
            f"Missing Real-CUGAN NCNN binary: {REALCUGAN_NCNN_DEFAULT_BIN}\n"
            "Run: bash scripts/setup_s01e36_video_ai_tools.sh install-realcugan-ncnn"
        )

    label = args.label or f"realcugan_ncnn_s{args.scale}_n{args.denoise}_a{args.alpha:g}"
    paths = paths_for(args.out_root, label)
    ensure_clean_dir(paths.root, overwrite=args.overwrite)
    ensure_clean_dir(paths.frames_in, overwrite=args.overwrite)
    ensure_clean_dir(paths.frames_out, overwrite=args.overwrite)
    run_cmd(extract_frames_cmd(args.source, paths.frames_in, SOURCE_FPS), run=args.run)

    model_path = Path(args.realcugan_model) if args.realcugan_model else binary.parent / "models-se"
    cmd = [
        str(binary),
        "-i",
        str(paths.frames_in),
        "-o",
        str(paths.frames_out),
        "-s",
        str(args.scale),
        "-n",
        str(args.denoise),
        "-f",
        "png",
        "-m",
        str(model_path),
    ]
    if args.gpu is not None:
        cmd.extend(["-g", str(args.gpu)])
    if args.alpha is not None:
        cmd.extend(["-a", str(args.alpha)])
    run_cmd(cmd, run=args.run)

    if args.run:
        normalize_numbered_frames(paths.frames_out, paths.frames_encode, overwrite=args.overwrite)
    run_cmd(encode_frames_cmd(paths.frames_encode, paths.output_video, SOURCE_FPS, scale_to_source=args.scale_to_source, crf=args.crf), run=args.run)
    if args.run and paths.output_video.is_file():
        mux_review_audio(paths, args)
    write_manifest(
        paths,
        args,
        backend="realcugan-ncnn-vulkan",
        commands_note="Real-CUGAN NCNN/Vulkan frame-folder run, then downscaled/encoded for review.",
    )


def run_template_backend(args: argparse.Namespace, backend: str, template: str | None) -> None:
    if not template:
        raise SystemExit(f"{backend} needs --{backend.lower()}-cmd with placeholders like {{input_dir}} and {{output_dir}}.")
    label = args.label or backend.lower()
    paths = paths_for(args.out_root, label)
    ensure_clean_dir(paths.root, overwrite=args.overwrite)
    ensure_clean_dir(paths.frames_in, overwrite=args.overwrite)
    ensure_clean_dir(paths.frames_out, overwrite=args.overwrite)
    if not args.template_produces_video:
        run_cmd(extract_frames_cmd(args.source, paths.frames_in, SOURCE_FPS), run=args.run)
    run_cmd(
        shell_template_to_cmd(template, input_dir=paths.frames_in, output_dir=paths.frames_out, input_video=args.source, output_video=paths.output_video),
        run=args.run,
        cwd=args.template_cwd,
    )
    if not args.template_produces_video:
        if args.run:
            normalize_numbered_frames(paths.frames_out, paths.frames_encode, overwrite=args.overwrite)
        run_cmd(encode_frames_cmd(paths.frames_encode, paths.output_video, SOURCE_FPS, scale_to_source=args.scale_to_source, crf=args.crf), run=args.run)
    if args.run and paths.output_video.is_file():
        mux_review_audio(paths, args)
    write_manifest(paths, args, backend=backend, commands_note=f"{backend} command template; template_produces_video={args.template_produces_video}.")


def run_apisr(args: argparse.Namespace) -> None:
    repo = default_tool_paths()["apisr_repo"]
    if args.apisr_cmd:
        run_template_backend(args, "APISR", args.apisr_cmd)
        return
    if not APISR_DEFAULT_PYTHON.is_file():
        raise SystemExit(
            f"Missing APISR venv python: {APISR_DEFAULT_PYTHON}\n"
            "Run: bash scripts/setup_s01e36_video_ai_tools.sh install-apisr"
        )
    if not APISR_DEFAULT_WEIGHT.is_file():
        raise SystemExit(
            f"Missing APISR default weight: {APISR_DEFAULT_WEIGHT}\n"
            "Run: bash scripts/setup_s01e36_video_ai_tools.sh weights-apisr"
        )
    template = (
        f"{APISR_DEFAULT_PYTHON} test_code/inference.py "
        "--input_dir {input_dir} "
        "--model RRDB --scale 2 "
        f"--weight_path {APISR_DEFAULT_WEIGHT} "
        "--downsample_threshold -1 "
        "--float16_inference True "
        "--store_dir {output_dir}"
    )
    args.template_cwd = repo
    run_template_backend(args, "APISR", template)


def run_animesr(args: argparse.Namespace) -> None:
    repo = default_tool_paths()["animesr_repo"]
    if args.animesr_cmd:
        run_template_backend(args, "AnimeSR", args.animesr_cmd)
        return
    weight = repo / "weights/AnimeSR_v2.pth"
    if not ANIMESR_DEFAULT_PYTHON.is_file():
        raise SystemExit(
            f"Missing AnimeSR venv python: {ANIMESR_DEFAULT_PYTHON}\n"
            "Run: bash scripts/setup_s01e36_video_ai_tools.sh install-animesr"
        )
    if not weight.is_file():
        raise SystemExit(
            f"Missing AnimeSR weight: {weight}\n"
            "Download AnimeSR_v2.pth from the upstream AnimeSR Google Drive link and place it there."
        )
    template = (
        f"{ANIMESR_DEFAULT_PYTHON} scripts/inference_animesr_video.py "
        "-i {input_video} -n AnimeSR_v2 -s 1 "
        "-o {output_dir} --expname animesr_v2_s01e36 "
        "--suffix x1 --num_process_per_gpu 1 --half"
    )
    args.template_cwd = repo
    args.template_produces_video = True
    run_template_backend(args, "AnimeSR", template)


def run_rife(args: argparse.Namespace) -> None:
    label = args.label or "rife_48fps"
    paths = paths_for(args.out_root, label)
    ensure_clean_dir(paths.root, overwrite=args.overwrite)
    intermediate = args.input_video or args.source
    if args.rife_cmd:
        cmd = shell_template_to_cmd(
            args.rife_cmd,
            input_dir=paths.frames_in,
            output_dir=paths.frames_out,
            input_video=intermediate,
            output_video=paths.output_video,
        )
        run_cmd(cmd, run=args.run, cwd=args.template_cwd)
        if args.run and paths.output_video.is_file():
            mux_review_audio(paths, args)
        write_manifest(paths, args, backend="rife", commands_note="RIFE command template.")
        return
    repo = default_tool_paths()["rife_repo"]
    inference = repo / "inference_video.py"
    if not inference.is_file():
        raise SystemExit(
            f"Missing RIFE repo/script: {inference}\n"
            "Run scripts/setup_s01e36_video_ai_tools.sh clone first, install its requirements, then retry."
        )
    python = str(RIFE_DEFAULT_PYTHON if RIFE_DEFAULT_PYTHON.is_file() else "python3")
    cmd = [python, str(inference), "--exp=1", f"--video={intermediate}", f"--output={paths.output_video}"]
    run_cmd(cmd, run=args.run)
    if args.run and paths.output_video.is_file():
        mux_review_audio(paths, args)
    write_manifest(paths, args, backend="rife", commands_note="RIFE Python inference_video.py exp=1.")


def run_rife_ncnn(args: argparse.Namespace) -> None:
    binary = find_binary(args.rife_ncnn_bin, ["rife-ncnn-vulkan"])
    if not binary and RIFE_NCNN_DEFAULT_BIN.is_file():
        binary = RIFE_NCNN_DEFAULT_BIN
    if not binary:
        raise SystemExit(
            f"Missing RIFE NCNN binary: {RIFE_NCNN_DEFAULT_BIN}\n"
            "Run: bash scripts/setup_s01e36_video_ai_tools.sh install-rife-ncnn"
        )

    label = args.label or "rife_ncnn_48fps"
    paths = paths_for(args.out_root, label)
    ensure_clean_dir(paths.root, overwrite=args.overwrite)
    ensure_clean_dir(paths.frames_in, overwrite=args.overwrite)
    ensure_clean_dir(paths.frames_out, overwrite=args.overwrite)
    source = args.input_video or args.source
    run_cmd(extract_frames_cmd(source, paths.frames_in, SOURCE_FPS), run=args.run)
    cmd = [
        str(binary),
        "-i",
        str(paths.frames_in),
        "-o",
        str(paths.frames_out),
        "-m",
        str(args.rife_ncnn_model or binary.parent / "rife-v4.6"),
        "-f",
        "%06d.png",
    ]
    if args.rife_ncnn_num_frames:
        cmd.extend(["-n", str(args.rife_ncnn_num_frames)])
    if args.gpu is not None:
        cmd.extend(["-g", str(args.gpu)])
    if args.rife_ncnn_tta:
        cmd.append("-x")
    run_cmd(cmd, run=args.run)
    if args.run:
        normalize_numbered_frames(paths.frames_out, paths.frames_encode, overwrite=args.overwrite)
    run_cmd(encode_frames_cmd(paths.frames_encode, paths.output_video, args.rife_ncnn_fps, scale_to_source=False, crf=args.crf), run=args.run)
    if args.run and paths.output_video.is_file():
        mux_review_audio(paths, args)
    write_manifest(
        paths,
        args,
        backend="rife-ncnn-vulkan",
        commands_note="RIFE NCNN/Vulkan interpolation. No ffmpeg motion interpolation.",
    )


def write_manifest(paths: TestPaths, args: argparse.Namespace, *, backend: str, commands_note: str) -> None:
    payload = {
        "backend": backend,
        "source": str(args.source),
        "output_video": str(paths.output_video),
        "output_review": str(paths.output_review) if paths.output_review.exists() else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": commands_note,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    if args.run:
        paths.manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest={paths.manifest}")
    print(f"output_video={paths.output_video}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["realcugan", "apisr", "animesr", "rife", "rife-ncnn", "check"])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--review-audio-source", type=Path, default=DEFAULT_REVIEW_AUDIO_SOURCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--label")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--crf", type=int, default=8)
    parser.add_argument("--scale-to-source", action="store_true", default=True)
    parser.add_argument("--realcugan-bin", type=Path)
    parser.add_argument("--realcugan-cmd")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--denoise", type=int, default=-1, help="Real-CUGAN NCNN noise level. Try -1 conservative, 0 no denoise, 1/2/3 stronger denoise.")
    parser.add_argument("--alpha", type=float, default=None, help="Real-CUGAN enhancement alpha when supported; higher is softer, lower is stronger.")
    parser.add_argument("--realcugan-model", help="Optional Real-CUGAN NCNN model folder/name passed through with -m.")
    parser.add_argument("--model")
    parser.add_argument("--apisr-cmd")
    parser.add_argument("--animesr-cmd")
    parser.add_argument("--template-cwd", type=Path)
    parser.add_argument("--template-produces-video", action="store_true", help="Template writes {output_video} directly; skip frame-folder re-encode.")
    parser.add_argument("--rife-bin", type=Path)
    parser.add_argument("--rife-cmd")
    parser.add_argument("--rife-model")
    parser.add_argument("--rife-ncnn-bin", type=Path)
    parser.add_argument("--rife-ncnn-model", type=Path)
    parser.add_argument("--rife-ncnn-fps", default=DOUBLE_FPS, help="Target FPS for rife-ncnn-vulkan, e.g. 48000/1001 or 50.")
    parser.add_argument("--rife-ncnn-num-frames", type=int, help="Optional target frame count passed to rife-ncnn-vulkan -n.")
    parser.add_argument("--rife-ncnn-tta", action="store_true", help="Enable TTA mode for rife-ncnn-vulkan.")
    parser.add_argument("--input-video", type=Path, help="For RIFE, interpolate this already-restored video instead of --source.")
    args = parser.parse_args()

    if args.backend == "check":
        paths = default_tool_paths()
        checks = {
            "Real-CUGAN repo": paths["realcugan_repo"],
            "APISR repo": paths["apisr_repo"],
            "AnimeSR repo": paths["animesr_repo"],
            "RIFE repo": paths["rife_repo"],
        }
        for label, path in checks.items():
            print(f"{label}: {'found' if path.exists() else 'missing'} {path}")
        print(f"APISR default python: {'found' if APISR_DEFAULT_PYTHON.is_file() else 'missing'} {APISR_DEFAULT_PYTHON}")
        print(f"APISR default weight: {'found' if APISR_DEFAULT_WEIGHT.is_file() else 'missing'} {APISR_DEFAULT_WEIGHT}")
        print(f"Real-CUGAN NCNN binary: {'found' if REALCUGAN_NCNN_DEFAULT_BIN.is_file() else 'missing'} {REALCUGAN_NCNN_DEFAULT_BIN}")
        print(f"RIFE NCNN binary: {'found' if RIFE_NCNN_DEFAULT_BIN.is_file() else 'missing'} {RIFE_NCNN_DEFAULT_BIN}")
        print(f"AnimeSR default python: {'found' if ANIMESR_DEFAULT_PYTHON.is_file() else 'missing'} {ANIMESR_DEFAULT_PYTHON}")
        print(f"RIFE default python: {'found' if RIFE_DEFAULT_PYTHON.is_file() else 'missing'} {RIFE_DEFAULT_PYTHON}")
        print("Real-CUGAN: default command works after install-realcugan-ncnn.")
        print("Template placeholders: {root}, {input_dir}, {output_dir}, {input_video}, {output_video}")
        print("APISR: default command works after install-apisr + weights-apisr.")
        print("AnimeSR: default command works after install-animesr + placing AnimeSR_v2.pth in weights/.")
        print("RIFE NCNN: default command works after install-rife-ncnn and avoids PyTorch/CUDA.")
        print("Python RIFE: default uses soft/ai_video_tools/src/ECCV2022-RIFE/inference_video.py when installed; --rife-cmd can override.")
        return 0
    if not args.source.is_file():
        raise SystemExit(f"Missing source video: {args.source}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.backend == "realcugan":
        run_realcugan(args)
    elif args.backend == "apisr":
        run_apisr(args)
    elif args.backend == "animesr":
        run_animesr(args)
    elif args.backend == "rife":
        run_rife(args)
    elif args.backend == "rife-ncnn":
        run_rife_ncnn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
