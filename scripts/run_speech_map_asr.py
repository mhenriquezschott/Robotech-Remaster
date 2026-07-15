#!/usr/bin/env python3
"""Create Robotech speech maps with phrase and word timestamps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_speech_map_asr.py")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--engine", choices=["whisperx", "faster-whisper"], default="whisperx")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="es")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.engine == "whisperx":
        result = run_whisperx(args)
    else:
        result = run_faster_whisper(args)

    stem = f"{args.episode}_{args.source_label}"
    json_path = args.out_dir / f"{stem}_speech_map.json"
    words_tsv = args.out_dir / f"{stem}_words.tsv"
    words_srt = args.out_dir / f"{stem}_words.srt"
    phrases_srt = args.out_dir / f"{stem}_phrases.srt"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_words_tsv(words_tsv, result["words"])
    write_words_srt(words_srt, result["words"])
    write_phrases_srt(phrases_srt, result["segments"])

    print(f"speech_map={json_path}")
    print(f"words_tsv={words_tsv}")
    print(f"words_srt={words_srt}")
    print(f"phrases_srt={phrases_srt}")
    print(f"segments={len(result['segments'])} words={len(result['words'])}")
    return 0


def run_whisperx(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import whisperx
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing WhisperX dependency: {exc.name}. Install requirements-asr.txt in .venv-asr."
        ) from exc

    audio = whisperx.load_audio(str(args.audio))
    model = whisperx.load_model(args.model, args.device, compute_type=args.compute_type, language=args.language)
    transcription = model.transcribe(audio, batch_size=args.batch_size, language=args.language)
    if args.device == "cuda":
        del model
        torch.cuda.empty_cache()

    align_model, metadata = whisperx.load_align_model(language_code=args.language, device=args.device)
    aligned = whisperx.align(
        transcription["segments"],
        align_model,
        metadata,
        audio,
        args.device,
        return_char_alignments=False,
    )
    if args.device == "cuda":
        del align_model
        torch.cuda.empty_cache()

    segments = normalize_whisperx_segments(aligned.get("segments", []))
    words = words_from_segments(segments)
    return {
        "kind": "robotech_speech_map",
        "engine": "whisperx",
        "model": args.model,
        "language": args.language,
        "audio": str(args.audio),
        "episode": args.episode,
        "source_label": args.source_label,
        "segments": segments,
        "words": words,
    }


def run_faster_whisper(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing faster-whisper dependency: {exc.name}. Install requirements-asr.txt in .venv-asr."
        ) from exc

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments_iter, info = model.transcribe(
        str(args.audio),
        language=args.language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )
    segments = []
    for index, segment in enumerate(segments_iter, start=1):
        words = []
        for word in segment.words or []:
            words.append(
                {
                    "word": clean_word_text(word.word),
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "score": None,
                }
            )
        segments.append(
            {
                "index": index,
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": segment.text.strip(),
                "words": words,
            }
        )
    words = words_from_segments(segments)
    return {
        "kind": "robotech_speech_map",
        "engine": "faster-whisper",
        "model": args.model,
        "language": getattr(info, "language", args.language),
        "audio": str(args.audio),
        "episode": args.episode,
        "source_label": args.source_label,
        "segments": segments,
        "words": words,
    }


def normalize_whisperx_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments = []
    for index, segment in enumerate(raw_segments, start=1):
        words = []
        for word in segment.get("words", []) or []:
            if "start" not in word or "end" not in word:
                continue
            words.append(
                {
                    "word": clean_word_text(str(word.get("word", ""))),
                    "start": round(float(word["start"]), 3),
                    "end": round(float(word["end"]), 3),
                    "score": word.get("score"),
                }
            )
        segments.append(
            {
                "index": index,
                "start": round(float(segment.get("start", words[0]["start"] if words else 0.0)), 3),
                "end": round(float(segment.get("end", words[-1]["end"] if words else 0.0)), 3),
                "text": str(segment.get("text", "")).strip(),
                "words": words,
            }
        )
    return segments


def words_from_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words = []
    counter = 1
    for segment in segments:
        for word in segment.get("words", []):
            words.append(
                {
                    "index": counter,
                    "segment_index": segment["index"],
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                    "score": word.get("score"),
                }
            )
            counter += 1
    return words


def clean_word_text(value: str) -> str:
    return value.strip().strip("\ufeff")


def write_words_tsv(path: Path, words: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "segment_index", "start", "end", "word", "score"], delimiter="\t")
        writer.writeheader()
        for word in words:
            writer.writerow(word)


def write_words_srt(path: Path, words: list[dict[str, Any]]) -> None:
    lines = []
    for index, word in enumerate(words, start=1):
        lines.extend(
            [
                str(index),
                f"{srt_time(word['start'])} --> {srt_time(word['end'])}",
                word["word"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_phrases_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    lines = []
    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                str(index),
                f"{srt_time(segment['start'])} --> {srt_time(segment['end'])}",
                segment["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def srt_time(value: float) -> str:
    milliseconds_total = max(0, int(round(value * 1000)))
    seconds_total, milliseconds = divmod(milliseconds_total, 1000)
    minutes_total, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


if __name__ == "__main__":
    raise SystemExit(main())
