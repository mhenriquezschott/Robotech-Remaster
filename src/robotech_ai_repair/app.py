"""Command-line launcher for the Robotech audio repair GUI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .models import ClipLane, RepairProject, RepairRegion
from .project import load_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robotech-repair-tool")
    parser.add_argument("--project", type=Path, help="Open an existing .repair.json project")
    parser.add_argument("--episode", default="S01E03")
    parser.add_argument("--title", default="Space Fold")
    parser.add_argument("--main-track", type=Path, help="Main/full audio track")
    parser.add_argument("--bed-track", type=Path, help="De-voiced English 5.1 bed track")
    parser.add_argument("--video-track", type=Path, help="Episode video reference track")
    parser.add_argument("--reference", type=Path, action="append", default=[], help="Reference audio track")
    parser.add_argument("--marker", type=float, default=43.5)
    parser.add_argument("--window", type=float, default=7.0)
    parser.add_argument("--cut-start", type=float, default=43.080)
    parser.add_argument("--cut-end", type=float, default=44.167)
    parser.add_argument("--clip", type=Path, action="append", default=[], help="Replacement/texture clip lane")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from .main_window import run_app
    except ModuleNotFoundError as exc:
        if exc.name in {"PySide6", "pyqtgraph", "soundfile", "sounddevice", "numpy"}:
            print(
                "The repair GUI dependencies are not installed.\n\n"
                "Install them with:\n"
                "  python -m pip install -r requirements-repair-tool.txt",
                file=sys.stderr,
            )
            return 2
        raise
    if args.project:
        return run_app(load_project(args.project), project_path=args.project)
    lanes = [
        ClipLane(name=f"Clip {index + 1}", path=str(path), role="other", fit_to_cut=index == 0)
        for index, path in enumerate(args.clip[:3])
    ]
    region = RepairRegion(
        repair_id=f"{args.episode.lower()}_repair",
        marker_seconds=args.marker,
        work_window_seconds=args.window,
        cut_start_seconds=args.cut_start,
        cut_end_seconds=args.cut_end,
        lanes=lanes,
    )
    project = RepairProject(
        episode=args.episode,
        title=args.title,
        main_track=str(args.main_track or ""),
        bed_track=str(args.bed_track or ""),
        video_track=str(args.video_track or ""),
        reference_tracks=[str(path) for path in args.reference],
        active_repair=region,
    )
    project.ensure_default_lanes()
    return run_app(project)


if __name__ == "__main__":
    raise SystemExit(main())
