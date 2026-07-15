#!/usr/bin/env python3
"""Apply deterministic cleanup rules to translated Spanish SRT files.

This is a small safety pass for local LLM/OCR subtitle artifacts that are easy
to identify and too specific to justify another translation run.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SUBTITLE_DIR = Path("work/review/subtitles")


@dataclass(frozen=True)
class RepairRule:
    pattern: re.Pattern[str]
    replacement: str
    description: str


REPAIR_RULES = [
    RepairRule(
        pattern=re.compile(r"\bhad[ií]an\b", flags=re.IGNORECASE),
        replacement="habían",
        description="Fix invalid LLM typo hadían/hadian -> habían",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subtitle-dir",
        type=Path,
        default=DEFAULT_SUBTITLE_DIR,
        help="Root folder containing per-episode subtitle folders.",
    )
    parser.add_argument(
        "--glob",
        default="S01E*/S01E*_spanish_translated.srt",
        help="Glob relative to --subtitle-dir. Defaults to translated Spanish SRTs.",
    )
    parser.add_argument("--run", action="store_true", help="Write changes. Without this, only report matches.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path. Defaults to subtitle_dir/spanish_subtitle_repairs_manifest.json when --run is used.",
    )
    args = parser.parse_args()

    started_at = time.monotonic()
    files = sorted(args.subtitle_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.subtitle_dir / args.glob}")

    manifest: dict[str, object] = {
        "subtitle_dir": str(args.subtitle_dir),
        "glob": args.glob,
        "run": args.run,
        "rules": [
            {
                "pattern": rule.pattern.pattern,
                "replacement": rule.replacement,
                "description": rule.description,
            }
            for rule in REPAIR_RULES
        ],
        "files": [],
    }

    changed_files = 0
    total_replacements = 0
    for path in files:
        original = path.read_text(encoding="utf-8-sig", errors="replace")
        repaired = original
        file_changes: list[dict[str, object]] = []

        for rule in REPAIR_RULES:
            repaired, count = rule.pattern.subn(lambda match: match_case(rule.replacement, match.group(0)), repaired)
            if count:
                file_changes.append(
                    {
                        "description": rule.description,
                        "count": count,
                    }
                )
                total_replacements += count

        if file_changes:
            changed_files += 1
            print(f"{path}: {sum(int(change['count']) for change in file_changes)} replacement(s)")
            for change in file_changes:
                print(f"  - {change['description']}: {change['count']}")
            if args.run:
                path.write_text(repaired, encoding="utf-8")

        manifest["files"].append(
            {
                "path": str(path),
                "changed": bool(file_changes),
                "changes": file_changes,
            }
        )

    elapsed = time.monotonic() - started_at
    manifest["summary"] = {
        "scanned_files": len(files),
        "changed_files": changed_files,
        "total_replacements": total_replacements,
        "elapsed_seconds": round(elapsed, 3),
    }

    if args.run:
        manifest_path = args.manifest or args.subtitle_dir / "spanish_subtitle_repairs_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"manifest={manifest_path}")

    mode = "UPDATED" if args.run else "DRY-RUN"
    print(
        f"{mode}: scanned={len(files)} changed_files={changed_files} "
        f"replacements={total_replacements} elapsed={elapsed:.2f}s"
    )
    if not args.run and total_replacements:
        print("Add --run to write the repaired SRT files.")
    return 0


def match_case(replacement: str, source: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


if __name__ == "__main__":
    raise SystemExit(main())
