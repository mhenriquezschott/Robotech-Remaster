#!/usr/bin/env python3
"""Create an XLSX review workbook for bracketed Spanish subtitle labels.

This is intentionally separate from the normal spelling workbook. Bracketed
text often contains speaker labels or closed-caption style sound descriptions,
so the useful question is not "is this a dictionary word?" but "is this still
English or an OCR-damaged label?".
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robotech_ai.cli import EPISODE_TITLES, write_simple_xlsx  # noqa: E402


ENGLISH_BRACKET_WORDS = {
    "alarm": "alarma",
    "alarms": "alarmas",
    "applauding": "aplausos",
    "beeping": "pitidos",
    "both": "ambos",
    "cheering": "vitoreos",
    "chuckles": "ríe suavemente",
    "continues": "continúa",
    "crowd": "multitud",
    "crying": "llorando",
    "explosion": "explosión",
    "gasps": "jadeos",
    "gasp": "jadeo",
    "groans": "gemidos",
    "groan": "gemido",
    "grunts": "gruñidos",
    "grunt": "gruñido",
    "laughing": "riendo",
    "laughs": "risas",
    "man": "hombre",
    "men": "hombres",
    "music": "música",
    "narrator": "narrador",
    "people": "personas",
    "screaming": "gritando",
    "sighs": "suspira",
    "speaking": "hablando",
    "trilling": "trino",
    "whirring": "zumbido",
    "woman": "mujer",
    "yells": "grita",
}

PHRASE_SUGGESTIONS = {
    "people cheering": "personas vitoreando",
    "monitor trilling": "trino de monitor",
    "beeping": "pitidos",
    "alarm blaring": "alarma sonando",
    "crowd cheering": "multitud vitoreando",
    "crowd murmuring": "multitud murmurando",
    "indistinct chatter": "conversaciones indistintas",
    "over radio": "por radio",
    "on radio": "por radio",
}

SUSPICIOUS_LABELS = {
    "COLONEL": "CORONEL",
    "CAPTAIN": "CAPITAN",
    "COMMANDER": "COMANDANTE",
    "ANNOUNCER": "ANUNCIADOR",
    "NARRATOR": "NARRADOR",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtitle-dir", type=Path, default=Path("work/review/subtitles"))
    parser.add_argument("--allowlist", type=Path, default=Path("config/subtitles/spanish_bracket_allowlist.txt"))
    parser.add_argument("--out", type=Path, default=Path("work/review/subtitle_language_review/spanish_bracket_review.xlsx"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    workbook: dict[str, list[list[object]]] = {}
    allowlist = load_bracket_allowlist(args.allowlist)
    total = 0
    for episode_id in sorted(EPISODE_TITLES):
        srt = args.subtitle_dir / episode_id / f"{episode_id}_spanish_translated.srt"
        rows = bracket_review_rows(episode_id, srt, allowlist)
        workbook[episode_id] = rows
        total += max(0, len(rows) - 1)
        print(f"{episode_id}: bracket_review_rows={max(0, len(rows) - 1)}")

    if not args.run:
        print("Add --run to write the XLSX workbook.")
        print(f"Would write: {args.out}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_simple_xlsx(args.out, workbook)
    print(f"workbook={args.out}")
    print(f"sheets={len(workbook)} rows={total}")
    return 0


def load_bracket_allowlist(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        normalize_label(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip("[]")).casefold()


def bracket_review_rows(episode_id: str, path: Path, allowlist: set[str]) -> list[list[object]]:
    headers = [
        "episode",
        "review_id",
        "action",
        "suspicious_word",
        "normalized",
        "replacement_word",
        "replacement_context",
        "source",
        "cue",
        "time",
        "context",
        "path",
        "notes",
        "bracket_text",
        "suggested_bracket",
    ]
    rows: list[list[object]] = [headers]
    if not path.is_file():
        return rows

    content = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\r?\n", content.strip())
    review_index = 1
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        cue = lines[0].strip()
        time_line = lines[1].strip()
        text = "\n".join(lines[2:])
        for match in re.finditer(r"\[([^\]\n]{1,100})\]", text):
            bracket = match.group(1).strip()
            if normalize_label(bracket) in allowlist:
                continue
            suggestion, reason = classify_bracket(bracket)
            if not reason:
                continue
            rows.append(
                [
                    episode_id,
                    f"{episode_id}-BRACKET-{review_index:04d}",
                    "",
                    f"[{bracket}]",
                    bracket.lower(),
                    "",
                    text,
                    "spanish_translated_brackets",
                    cue,
                    time_line,
                    text,
                    str(path),
                    reason,
                    bracket,
                    suggestion,
                ]
            )
            review_index += 1
    return rows


def classify_bracket(bracket: str) -> tuple[str, str]:
    compact = re.sub(r"\s+", " ", bracket).strip()
    lower = compact.lower()
    if compact in SUSPICIOUS_LABELS:
        return SUSPICIOUS_LABELS[compact], "English role label inside brackets"
    if compact.startswith("I") and compact[1:] in {"KHYRON"}:
        return compact[1:], "Possible OCR confusion: leading I before speaker label"
    if lower in PHRASE_SUGGESTIONS:
        return PHRASE_SUGGESTIONS[lower], "English action label inside brackets"
    words = re.findall(r"[A-Za-z]+", lower)
    english_hits = [word for word in words if word in ENGLISH_BRACKET_WORDS]
    if english_hits:
        suggestion = lower
        for word in sorted(set(english_hits), key=len, reverse=True):
            suggestion = re.sub(rf"\b{re.escape(word)}\b", ENGLISH_BRACKET_WORDS[word], suggestion)
        return suggestion, "Possible English words inside bracket label: " + ", ".join(sorted(set(english_hits)))
    if re.search(r"\b[A-Z]{2,}T\b", compact) and compact not in {"PITIDO"}:
        return "", "Possible OCR-damaged uppercase speaker label"
    return "", ""


if __name__ == "__main__":
    raise SystemExit(main())
