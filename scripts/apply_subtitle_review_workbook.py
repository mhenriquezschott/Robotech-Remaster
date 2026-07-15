#!/usr/bin/env python3
"""Apply reviewed subtitle workbook word fixes to Spanish SRT files.

The workbook is intentionally imported conservatively: timing/index lines are
never rewritten, and only subtitle text lines inside targeted cues are edited.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


SPREADSHEET_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
WORD_CHARS = "A-Za-zÁÉÍÓÚÜÑáéíóúüñ"


@dataclass(frozen=True)
class WorkbookEdit:
    episode: str
    cue: str
    time: str
    path: Path
    suspicious_word: str
    replacement_word: str
    replacement_context: str
    original_context: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path("work/review/subtitle_language_review/spanish_language_review.xlsx"))
    parser.add_argument("--manifest", type=Path, default=Path("work/review/subtitle_language_review/spanish_language_review_apply_manifest.json"))
    parser.add_argument("--backup-suffix", default=None)
    parser.add_argument("--run", action="store_true", help="Write SRT files. Without this, only report planned changes.")
    args = parser.parse_args()

    started = time.monotonic()
    edits = read_review_workbook(args.workbook)
    edits_by_path: dict[Path, list[WorkbookEdit]] = {}
    for edit in edits:
        edits_by_path.setdefault(edit.path, []).append(edit)

    label_fixes = {
        "[IKHYRON]": "[KHYRON]",
    }
    manifest: dict[str, object] = {
        "workbook": str(args.workbook),
        "run": args.run,
        "edits": len(edits),
        "files": [],
        "label_fixes": label_fixes,
    }
    changed_files = 0
    total_word_replacements = 0
    total_context_replacements = 0
    total_label_replacements = 0
    backup_suffix = args.backup_suffix or time.strftime(".bak-language-review-%Y%m%d-%H%M%S")

    for path in sorted(set(edits_by_path) | set(Path("work/review/subtitles").glob("S01E*/S01E*_spanish_translated.srt"))):
        if not path.is_file():
            manifest["files"].append({"path": str(path), "missing": True})
            print(f"MISSING {path}")
            continue

        original = path.read_text(encoding="utf-8-sig", errors="replace")
        repaired = original
        word_changes: list[dict[str, object]] = []
        label_changes: list[dict[str, object]] = []

        if path in edits_by_path:
            repaired, word_changes = apply_word_edits_to_srt(repaired, edits_by_path[path])
            total_word_replacements += sum(int(item["count"]) for item in word_changes)
            total_context_replacements += sum(1 for item in word_changes if item.get("mode") == "context" and int(item["count"]) > 0)

        for old, new in label_fixes.items():
            repaired, count = repaired.replace(old, new), repaired.count(old)
            if count:
                label_changes.append({"from": old, "to": new, "count": count})
                total_label_replacements += count

        changed = repaired != original
        if changed:
            changed_files += 1
            print(f"{path}: word_replacements={sum(int(item['count']) for item in word_changes)} label_replacements={sum(int(item['count']) for item in label_changes)}")
            if args.run:
                backup_path = path.with_name(path.name + backup_suffix)
                if not backup_path.exists():
                    shutil.copy2(path, backup_path)
                path.write_text(repaired, encoding="utf-8")

        manifest["files"].append(
            {
                "path": str(path),
                "changed": changed,
                "word_changes": word_changes,
                "label_changes": label_changes,
            }
        )

    manifest["summary"] = {
        "changed_files": changed_files,
        "word_replacements": total_word_replacements,
        "context_replacements": total_context_replacements,
        "label_replacements": total_label_replacements,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if args.run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"manifest={args.manifest}")
    else:
        print("Add --run to write SRT changes.")
    print(json.dumps(manifest["summary"], ensure_ascii=False))
    return 0


def read_review_workbook(path: Path) -> list[WorkbookEdit]:
    edits: list[WorkbookEdit] = []
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        for sheet in workbook.findall(".//m:sheet", SPREADSHEET_NS):
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = relmap[rel_id]
            sheet_path = "xl/" + target if not target.startswith("/") else target[1:]
            rows = read_sheet_rows(archive, sheet_path, shared_strings)
            if not rows:
                continue
            header = {value: index for index, value in enumerate(rows[0])}
            required = {"episode", "cue", "time", "path", "suspicious_word", "replacement_word"}
            if not required.issubset(header):
                continue
            for row in rows[1:]:
                replacement_word = cell(row, header["replacement_word"]).strip()
                replacement_context = cell(row, header.get("replacement_context", -1)).strip()
                original_context = cell(row, header.get("context", -1)).strip()
                context_changed = bool(replacement_context and replacement_context != original_context)
                if not replacement_word and not context_changed:
                    continue
                edits.append(
                    WorkbookEdit(
                        episode=cell(row, header["episode"]).strip(),
                        cue=cell(row, header["cue"]).strip(),
                        time=cell(row, header["time"]).strip(),
                        path=Path(cell(row, header["path"]).strip()),
                        suspicious_word=cell(row, header["suspicious_word"]).strip(),
                        replacement_word=replacement_word,
                        replacement_context=replacement_context,
                        original_context=original_context,
                    )
                )
    return edits


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(text.text or "" for text in item.findall(".//m:t", SPREADSHEET_NS)) for item in root.findall("m:si", SPREADSHEET_NS)]


def read_sheet_rows(archive: zipfile.ZipFile, path: str, shared_strings: list[str]) -> list[list[str]]:
    root = ET.fromstring(archive.read(path))
    rows: list[list[str]] = []
    for row in root.findall(".//m:row", SPREADSHEET_NS):
        values: dict[int, str] = {}
        for cell_node in row.findall("m:c", SPREADSHEET_NS):
            ref = cell_node.attrib.get("r", "A1")
            values[column_index(re.match(r"([A-Z]+)", ref).group(1))] = read_cell_value(cell_node, shared_strings)
        if values:
            max_index = max(values)
            rows.append([values.get(index, "") for index in range(max_index + 1)])
    return rows


def read_cell_value(cell_node: ET.Element, shared_strings: list[str]) -> str:
    if cell_node.attrib.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell_node.findall(".//m:t", SPREADSHEET_NS))
    value = cell_node.find("m:v", SPREADSHEET_NS)
    if value is None or value.text is None:
        return ""
    if cell_node.attrib.get("t") == "s":
        return shared_strings[int(value.text)]
    return value.text


def cell(row: list[str], index: int) -> str:
    if index < 0:
        return ""
    return row[index] if index < len(row) else ""


def column_index(name: str) -> int:
    index = 0
    for char in name:
        index = index * 26 + ord(char) - 64
    return index - 1


def apply_word_edits_to_srt(content: str, edits: list[WorkbookEdit]) -> tuple[str, list[dict[str, object]]]:
    blocks = re.split(r"(\r?\n\r?\n)", content)
    edits_by_cue: dict[str, list[WorkbookEdit]] = {}
    for edit in edits:
        edits_by_cue.setdefault(edit.cue, []).append(edit)
    changes: list[dict[str, object]] = []
    for index in range(0, len(blocks), 2):
        block = blocks[index]
        if not block.strip():
            continue
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        cue = lines[0].strip()
        if cue not in edits_by_cue:
            continue
        text = "\n".join(lines[2:])
        new_text = text
        for edit in edits_by_cue[cue]:
            if edit.replacement_context and edit.replacement_context != edit.original_context:
                new_text = edit.replacement_context
                count = 1 if new_text != text else 0
                mode = "context"
            else:
                new_text, count = replace_word(new_text, edit.suspicious_word, edit.replacement_word)
                mode = "word"
            changes.append(
                {
                    "mode": mode,
                    "cue": cue,
                    "time": edit.time,
                    "from": edit.suspicious_word,
                    "to": edit.replacement_context if mode == "context" else edit.replacement_word,
                    "count": count,
                }
            )
        if new_text != text:
            blocks[index] = "\n".join(lines[:2] + new_text.splitlines())
    return "".join(blocks), changes


def replace_word(text: str, old: str, new: str) -> tuple[str, int]:
    pattern = re.compile(rf"(?<![{WORD_CHARS}]){re.escape(old)}(?![{WORD_CHARS}])")
    repaired, count = pattern.subn(new, text)
    if count:
        return repaired, count
    fallback = re.compile(rf"(?<![{WORD_CHARS}]){re.escape(old)}(?![{WORD_CHARS}])", flags=re.IGNORECASE)
    return fallback.subn(new, text)


if __name__ == "__main__":
    raise SystemExit(main())
