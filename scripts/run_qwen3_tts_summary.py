#!/usr/bin/env python3
"""Generate Qwen3-TTS voice-clone takes for Robotech narrator summaries."""

from __future__ import annotations

import argparse
import json
import json.decoder
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", type=Path, required=True)
    parser.add_argument("--takes", type=int, default=10)
    parser.add_argument("--phrases", nargs="+", help="Optional phrase numbers to generate, e.g. 03 or 01 04")
    parser.add_argument("--replace-phrase", action="store_true", help="When used with --phrases, replace that phrase's existing takes instead of appending")
    parser.add_argument("--model-size", choices=["0.6B", "1.7B"], default="1.7B")
    parser.add_argument("--language", default="Spanish")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-gap", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=-1)
    parser.add_argument("--temperature", type=float, help="Optional Qwen Base sampling temperature")
    parser.add_argument("--top-p", type=float, help="Optional Qwen Base nucleus sampling value")
    parser.add_argument("--top-k", type=int, help="Optional Qwen Base top-k sampling value")
    parser.add_argument("--repetition-penalty", type=float, help="Optional Qwen Base repetition penalty")
    parser.add_argument("--subtalker-temperature", type=float, help="Optional Qwen subtalker sampling temperature")
    parser.add_argument("--subtalker-top-p", type=float, help="Optional Qwen subtalker nucleus sampling value")
    parser.add_argument("--subtalker-top-k", type=int, help="Optional Qwen subtalker top-k sampling value")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--x-vector-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--assemble-only", action="store_true", help="Rebuild selected-takes preview without loading Qwen")
    parser.add_argument("--fit-to-slots", action="store_true", help="During assembly, speed up selected takes that exceed their phrase slot")
    parser.add_argument("--slot-margin", type=float, default=0.0, help="Seconds to leave free at the end of each phrase slot when --fit-to-slots is used")
    parser.add_argument("--speed-all-percent", type=float, default=100.0, help="During assembly, apply one tempo change to all selected takes")
    parser.add_argument("--avoid-overlap", action="store_true", help="During assembly, shift phrase starts to prevent overlap without time-stretching")
    parser.add_argument("--min-gap", type=float, default=0.10, help="Minimum gap between assembled phrases when --avoid-overlap is used")
    parser.add_argument("--balance-phrases", action="store_true", help="During assembly, RMS-balance selected takes so phrase loudness is closer")
    parser.add_argument("--balance-max-gain-db", type=float, default=3.0, help="Maximum per-phrase gain adjustment for --balance-phrases")
    parser.add_argument("--summary-gain-db", type=float, default=0.0, help="Final gain applied to the whole assembled summary after phrase balancing")
    args = parser.parse_args()

    started = time.monotonic()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/robotech_numba_cache")
    sys.path.insert(0, str(args.qwen_root))

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    phrases = plan.get("phrases", [])
    if not phrases:
        raise SystemExit(f"Phrase plan has no phrases: {args.plan}")

    selected_phrase_numbers = parse_phrase_numbers(args.phrases)
    if selected_phrase_numbers:
        phrases = [phrase for phrase in phrases if int(phrase["number"]) in selected_phrase_numbers]
        if not phrases:
            raise SystemExit(f"No matching phrases found for --phrases {args.phrases}")

    if args.assemble_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output = assemble_selected_preview(
            plan,
            args.out_dir,
            fit_to_slots=args.fit_to_slots,
            slot_margin=args.slot_margin,
            speed_all_percent=args.speed_all_percent,
            avoid_overlap=args.avoid_overlap,
            min_gap=args.min_gap,
            balance_phrases=args.balance_phrases,
            balance_max_gain_db=args.balance_max_gain_db,
            summary_gain_db=args.summary_gain_db,
        )
        manifest = read_json(args.out_dir / "manifest.json")
        if manifest:
            manifest["assembled_selected_preview"] = str(output)
            write_review_md(args.out_dir / "REVIEW.md", plan, manifest)
        print(f"assembled_selected={output}")
        print(f"done in {format_elapsed(time.monotonic() - started)}")
        return 0

    if args.out_dir.exists() and not args.overwrite and not selected_phrase_numbers:
        generated_outputs = [
            path
            for path in args.out_dir.rglob("*")
            if path.is_file()
            and (
                path.suffix.lower() == ".wav"
                or path.name in {"selected_takes.json", "REVIEW.md"}
                or path.parent.name.isdigit()
            )
        ]
        if generated_outputs:
            sample = generated_outputs[0]
            raise SystemExit(
                f"Generated TTS outputs already exist in {args.out_dir}. "
                f"Example: {sample}. Use --overwrite to regenerate."
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref_text = args.ref_text.read_text(encoding="utf-8").strip()
    if not ref_text and not args.x_vector_only:
        raise SystemExit(f"Reference text is empty: {args.ref_text}")

    from qwen_tts import Qwen3TTSModel

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    repo_id = f"Qwen/Qwen3-TTS-12Hz-{args.model_size}-Base"
    model_path = snapshot_download(repo_id)
    print(f"Loading Qwen3-TTS Base {args.model_size}: {model_path}")
    print(f"device={device} dtype={dtype} language={args.language}")
    tts = Qwen3TTSModel.from_pretrained(model_path, device_map=device, dtype=dtype)
    generation_kwargs = qwen_generation_kwargs(args)
    if generation_kwargs:
        print(f"Qwen generation overrides: {generation_kwargs}")

    print("Building narrator voice-clone prompt once...")
    voice_prompt = tts.create_voice_clone_prompt(
        ref_audio=str(args.ref_audio),
        ref_text=ref_text if not args.x_vector_only else None,
        x_vector_only_mode=args.x_vector_only,
    )

    seed_base = args.seed_base
    if seed_base < 0:
        seed_base = random.randint(100_000_000, 2_000_000_000)

    manifest: dict[str, Any] = {
        "kind": "robotech_qwen3_tts_summary_generation",
        "episode": plan.get("episode"),
        "plan": str(args.plan),
        "out_dir": str(args.out_dir),
        "qwen_root": str(args.qwen_root),
        "repo_id": repo_id,
        "model_size": args.model_size,
        "language": args.language,
        "reference_audio": str(args.ref_audio),
        "reference_text": str(args.ref_text),
        "x_vector_only": args.x_vector_only,
        "takes_per_phrase": args.takes,
        "selected_phrase_numbers": sorted(selected_phrase_numbers) if selected_phrase_numbers else None,
        "replace_phrase": args.replace_phrase,
        "chunk_size": args.chunk_size,
        "chunk_gap": args.chunk_gap,
        "seed_base": seed_base,
        "generation_kwargs": generation_kwargs,
        "phrases": [],
    }
    selected = read_existing_selection(args.out_dir, plan)
    phrase_audio_for_v01: list[tuple[dict[str, Any], Path]] = []

    for phrase in phrases:
        phrase_number = int(phrase["number"])
        phrase_label = f"{phrase_number:02d}_{plan['episode']}"
        phrase_dir = args.out_dir / f"{phrase_number:02d}"
        if selected_phrase_numbers and args.replace_phrase and phrase_dir.exists():
            for old_wav in phrase_dir.glob("*.wav"):
                old_wav.unlink()
        phrase_dir.mkdir(parents=True, exist_ok=True)
        (phrase_dir / "text.txt").write_text(str(phrase["text"]).strip() + "\n", encoding="utf-8")
        existing_manifest = read_json(phrase_dir / "manifest.json")
        existing_takes = existing_manifest.get("takes", []) if isinstance(existing_manifest, dict) else []
        if selected_phrase_numbers and not args.replace_phrase:
            start_take_number = next_take_number(phrase_dir, phrase_number, plan["episode"])
        else:
            start_take_number = 1
        phrase_manifest: dict[str, Any] = {
            "number": phrase_number,
            "start": phrase["start"],
            "end": phrase["end"],
            "target_duration": phrase["duration"],
            "text": phrase["text"],
            "takes": [] if args.replace_phrase or start_take_number == 1 else list(existing_takes),
        }
        print(f"[{phrase_number:02d}/{len(phrases):02d}] {phrase['text']}")
        for take_number in range(start_take_number, start_take_number + args.takes):
            seed = seed_base + phrase_number * 100 + take_number
            output = phrase_dir / f"{phrase_label}_v{take_number:02d}.wav"
            set_seed(seed)
            wav, sr = synthesize_phrase(
                tts=tts,
                text=str(phrase["text"]).strip(),
                language=args.language,
                voice_prompt=voice_prompt,
                max_chunk_chars=args.chunk_size,
                chunk_gap=args.chunk_gap,
                generation_kwargs=generation_kwargs,
            )
            sf.write(output, wav, sr, subtype="PCM_24")
            take_record = {
                "take": take_number,
                "version": f"v{take_number:02d}",
                "seed": seed,
                "path": str(output),
                "sample_rate": sr,
                "duration": round(float(len(wav) / sr), 3),
            }
            phrase_manifest["takes"].append(take_record)
            print(f"  v{take_number:02d} seed={seed} duration={take_record['duration']:.3f}s")
            if take_number == 1 and not selected_phrase_numbers:
                phrase_audio_for_v01.append((phrase, output))
                selected[f"{phrase_number:02d}"] = "v01"
            elif selected_phrase_numbers and f"{phrase_number:02d}" not in selected:
                selected[f"{phrase_number:02d}"] = f"v{take_number:02d}"
        (phrase_dir / "manifest.json").write_text(
            json.dumps(phrase_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest["phrases"].append(phrase_manifest)

    assembled = args.out_dir / f"{plan['episode']}_{plan['summary_id']}_all_v01_preview.wav"
    if phrase_audio_for_v01:
        assemble_preview(plan, phrase_audio_for_v01, assembled)
    selection_path = args.out_dir / "selected_takes.json"
    selection_path.write_text(
        json.dumps(
            {
                "kind": "robotech_qwen3_tts_summary_selection",
                "episode": plan.get("episode"),
                "summary_id": plan.get("summary_id"),
                "selected": selected,
                "notes": "Edit selected values to v02, v03, etc., then rebuild the assembly.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest["assembled_all_v01_preview"] = str(assembled) if assembled.is_file() else None
    manifest["selection"] = str(selection_path)
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_review_md(args.out_dir / "REVIEW.md", plan, manifest)

    print(f"assembled_all_v01={assembled}")
    print(f"selection={selection_path}")
    print(f"manifest={manifest_path}")
    print(f"done in {format_elapsed(time.monotonic() - started)}")
    return 0


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but torch.cuda.is_available() is false in this venv; using CPU.")
        return "cpu"
    return requested


def parse_phrase_numbers(values: list[str] | None) -> set[int]:
    if not values:
        return set()
    numbers: set[int] = set()
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if not stripped:
                continue
            numbers.add(int(stripped))
    return numbers


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: line {exc.lineno} column {exc.colno}: {exc.msg}") from exc


def read_existing_selection_data(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "selected_takes.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in {path}: line {exc.lineno} column {exc.colno}: {exc.msg}") from exc
        if isinstance(data, dict):
            return data
    return {}


def read_existing_selection(out_dir: Path, plan: dict[str, Any]) -> dict[str, str]:
    data = read_existing_selection_data(out_dir)
    selected = data.get("selected", {})
    if isinstance(selected, dict):
        return {str(key).zfill(2): str(value) for key, value in selected.items()}
    return {f"{int(phrase['number']):02d}": "v01" for phrase in plan.get("phrases", [])}


def read_phrase_speed_percent(out_dir: Path) -> dict[str, float]:
    data = read_existing_selection_data(out_dir)
    speed_data = data.get("speed_percent_by_phrase")
    if not isinstance(speed_data, dict):
        speed_data = data.get("speed_percent", {})
    if not isinstance(speed_data, dict):
        return {}
    speeds: dict[str, float] = {}
    for key, value in speed_data.items():
        try:
            speed = float(value)
        except (TypeError, ValueError):
            raise SystemExit(f"Invalid speed percent for phrase {key!r}: {value!r}")
        if speed <= 0:
            raise SystemExit(f"Phrase speed percent must be positive for phrase {key!r}: {speed}")
        speeds[str(key).zfill(2)] = speed
    return speeds


def next_take_number(phrase_dir: Path, phrase_number: int, episode_id: str) -> int:
    pattern = f"{phrase_number:02d}_{episode_id}_v*.wav"
    maximum = 0
    for path in phrase_dir.glob(pattern):
        match = path.stem.rsplit("_v", 1)
        if len(match) != 2:
            continue
        try:
            maximum = max(maximum, int(match[1]))
        except ValueError:
            continue
    return maximum + 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def qwen_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Return only explicitly requested Qwen generation overrides."""

    values = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_temperature": args.subtalker_temperature,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_top_k": args.subtalker_top_k,
    }
    return {key: value for key, value in values.items() if value is not None}


def chunk_text(text: str, max_chars: int) -> list[str]:
    stripped = " ".join(text.split())
    if max_chars <= 0 or len(stripped) <= max_chars:
        return [stripped]
    chunks: list[str] = []
    current = ""
    for token in stripped.split(" "):
        candidate = token if not current else current + " " + token
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = token
    if current:
        chunks.append(current)
    return chunks


def synthesize_phrase(
    tts: Any,
    text: str,
    language: str,
    voice_prompt: Any,
    max_chunk_chars: int,
    chunk_gap: float,
    generation_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    wavs: list[np.ndarray] = []
    sample_rate = 0
    for chunk in chunk_text(text, max_chunk_chars):
        generated, sample_rate = tts.generate_voice_clone(
            text=chunk,
            language=language,
            voice_clone_prompt=voice_prompt,
            max_new_tokens=2048,
            **(generation_kwargs or {}),
        )
        wavs.append(np.asarray(generated[0], dtype=np.float32))
    if not wavs:
        raise ValueError("No audio was generated")
    if len(wavs) == 1:
        return wavs[0], int(sample_rate)
    gap = np.zeros(int(round(sample_rate * max(0.0, chunk_gap))), dtype=np.float32)
    joined: list[np.ndarray] = []
    for index, wav in enumerate(wavs):
        joined.append(wav)
        if gap.size and index < len(wavs) - 1:
            joined.append(gap)
    return np.concatenate(joined), int(sample_rate)


def assemble_preview(
    plan: dict[str, Any],
    phrase_audio: list[tuple[dict[str, Any], Path]],
    output: Path,
    *,
    fit_to_slots: bool = False,
    slot_margin: float = 0.0,
    speed_all_percent: float = 100.0,
    avoid_overlap: bool = False,
    min_gap: float = 0.10,
    balance_phrases: bool = False,
    balance_max_gain_db: float = 3.0,
    phrase_speed_percent: dict[str, float] | None = None,
    summary_gain_db: float = 0.0,
) -> None:
    if not phrase_audio:
        return
    loaded: list[tuple[dict[str, Any], np.ndarray, int]] = []
    sample_rate = 0
    speed_all_percent = float(speed_all_percent)
    if speed_all_percent <= 0:
        raise SystemExit(f"--speed-all-percent must be positive, got {speed_all_percent}")
    phrase_speed_percent = phrase_speed_percent or {}
    with tempfile.TemporaryDirectory(prefix="robotech_tts_fit_") as temp_dir:
        for phrase, path in phrase_audio:
            working_path = path
            phrase_key = f"{int(phrase['number']):02d}"
            original_duration = audio_duration(path)
            if abs(speed_all_percent - 100.0) > 0.0001:
                sped_path = Path(temp_dir) / f"{path.stem}_speed{safe_float_label(speed_all_percent)}.wav"
                apply_tempo(path, sped_path, speed_all_percent / 100.0)
                sped_duration = audio_duration(sped_path)
                print(
                    f"speed phrase {int(phrase['number']):02d}: "
                    f"{original_duration:.3f}s -> {sped_duration:.3f}s speed={speed_all_percent:.3f}%"
                )
                working_path = sped_path
                original_duration = sped_duration
            per_phrase_speed = phrase_speed_percent.get(phrase_key, 100.0)
            if abs(per_phrase_speed - 100.0) > 0.0001:
                sped_path = Path(temp_dir) / f"{path.stem}_phrase{phrase_key}_speed{safe_float_label(per_phrase_speed)}.wav"
                apply_tempo(working_path, sped_path, per_phrase_speed / 100.0)
                sped_duration = audio_duration(sped_path)
                print(
                    f"speed phrase {phrase_key} override: "
                    f"{original_duration:.3f}s -> {sped_duration:.3f}s speed={per_phrase_speed:.3f}%"
                )
                working_path = sped_path
                original_duration = sped_duration
            if fit_to_slots:
                target_duration = max(0.1, float(phrase["duration"]) - max(0.0, slot_margin))
                if original_duration > target_duration:
                    fitted_path = Path(temp_dir) / f"{path.stem}_fit.wav"
                    fit_audio_to_duration(working_path, fitted_path, target_duration)
                    fitted_duration = audio_duration(fitted_path)
                    print(
                        f"fit phrase {int(phrase['number']):02d}: "
                        f"{original_duration:.3f}s -> {fitted_duration:.3f}s target={target_duration:.3f}s"
                    )
                    working_path = fitted_path
            wav, sr = sf.read(working_path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sample_rate and sr != sample_rate:
                raise SystemExit(f"Cannot assemble mixed sample rates: {sr} vs {sample_rate}")
            sample_rate = int(sr)
            loaded.append((phrase, wav.astype(np.float32), int(sr)))
    if balance_phrases:
        loaded = balance_loaded_phrases(loaded, max_gain_db=balance_max_gain_db)

    start = float(plan["start"])
    scheduled_starts = schedule_phrase_starts(
        loaded,
        plan_start=start,
        plan_end=float(plan["end"]),
        min_gap=max(0.0, min_gap),
        avoid_overlap=avoid_overlap,
    )
    end = max(float(plan["end"]), max(scheduled_starts[index] + len(w) / sample_rate for index, (_p, w, _sr) in enumerate(loaded)))
    out = np.zeros(int(round((end - start) * sample_rate)), dtype=np.float32)
    for index, (phrase, wav, _sr) in enumerate(loaded):
        offset = int(round((scheduled_starts[index] - start) * sample_rate))
        if offset < 0:
            wav = wav[-offset:]
            offset = 0
        finish = min(out.size, offset + wav.size)
        if finish > offset:
            out[offset:finish] += wav[: finish - offset]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    summary_gain_db = float(summary_gain_db)
    if abs(summary_gain_db) > 0.0001:
        gain = float(10.0 ** (summary_gain_db / 20.0))
        out = (out * gain).astype(np.float32)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.98:
            limiter_gain = 0.98 / peak
            out = (out * limiter_gain).astype(np.float32)
            print(
                f"summary gain requested={summary_gain_db:+.2f}dB limited_by_peak="
                f"{20.0 * float(np.log10(limiter_gain)):+.2f}dB"
            )
        else:
            print(f"summary gain applied={summary_gain_db:+.2f}dB")
    sf.write(output, out, sample_rate, subtype="PCM_24")


def assemble_selected_preview(
    plan: dict[str, Any],
    out_dir: Path,
    *,
    fit_to_slots: bool = False,
    slot_margin: float = 0.0,
    speed_all_percent: float = 100.0,
    avoid_overlap: bool = False,
    min_gap: float = 0.10,
    balance_phrases: bool = False,
    balance_max_gain_db: float = 3.0,
    summary_gain_db: float = 0.0,
) -> Path:
    selected = read_existing_selection(out_dir, plan)
    phrase_speed_percent = read_phrase_speed_percent(out_dir)
    phrase_audio: list[tuple[dict[str, Any], Path]] = []
    episode_id = str(plan["episode"])
    for phrase in plan.get("phrases", []):
        number = int(phrase["number"])
        key = f"{number:02d}"
        version = selected.get(key, "v01").lower()
        if not version.startswith("v"):
            version = "v" + version
        path = out_dir / key / f"{key}_{episode_id}_{version}.wav"
        if not path.is_file():
            raise SystemExit(f"Selected take is missing for phrase {key}: {path}")
        phrase_audio.append((phrase, path))
    suffix_parts = ["selected"]
    if abs(float(speed_all_percent) - 100.0) > 0.0001:
        suffix_parts.append(f"speed{safe_float_label(float(speed_all_percent))}")
    if fit_to_slots:
        suffix_parts.append("fit")
    if avoid_overlap:
        suffix_parts.append("nooverlap")
        suffix_parts.append(f"gap{safe_float_label(min_gap)}")
    if balance_phrases:
        suffix_parts.append("balanced")
    if phrase_speed_percent:
        suffix_parts.append("phrasespeed")
    if abs(float(summary_gain_db)) > 0.0001:
        suffix_parts.append(f"gain{safe_signed_float_label(float(summary_gain_db))}db")
    suffix_parts.append("preview")
    suffix = "_".join(suffix_parts)
    output = out_dir / f"{episode_id}_{plan['summary_id']}_{suffix}.wav"
    assemble_preview(
        plan,
        phrase_audio,
        output,
        fit_to_slots=fit_to_slots,
        slot_margin=slot_margin,
        speed_all_percent=speed_all_percent,
        avoid_overlap=avoid_overlap,
        min_gap=min_gap,
        balance_phrases=balance_phrases,
        balance_max_gain_db=balance_max_gain_db,
        phrase_speed_percent=phrase_speed_percent,
        summary_gain_db=summary_gain_db,
    )
    return output


def safe_float_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def safe_signed_float_label(value: float) -> str:
    prefix = "plus" if value >= 0 else "minus"
    return prefix + safe_float_label(abs(value))


def audio_duration(path: Path) -> float:
    with sf.SoundFile(path) as handle:
        return len(handle) / float(handle.samplerate)


def schedule_phrase_starts(
    loaded: list[tuple[dict[str, Any], np.ndarray, int]],
    *,
    plan_start: float,
    plan_end: float,
    min_gap: float,
    avoid_overlap: bool,
) -> list[float]:
    original = [float(phrase["start"]) for phrase, _wav, _sr in loaded]
    if not avoid_overlap or len(loaded) < 2:
        return original
    durations = [len(wav) / float(sr) for _phrase, wav, sr in loaded]
    starts = list(original)

    # First try to keep the final phrase inside the plan and borrow unused gaps
    # from the end backwards. The first phrase remains anchored.
    starts[-1] = min(starts[-1], plan_end - durations[-1])
    for index in range(len(starts) - 2, 0, -1):
        latest_start = starts[index + 1] - min_gap - durations[index]
        starts[index] = min(starts[index], latest_start)
    starts[0] = original[0]

    # Forward pass guarantees no overlap after the anchored first phrase.
    for index in range(1, len(starts)):
        earliest_start = starts[index - 1] + durations[index - 1] + min_gap
        starts[index] = max(starts[index], earliest_start)

    print(f"avoid-overlap schedule min_gap={min_gap:.3f}s")
    for index, (phrase, _wav, _sr) in enumerate(loaded):
        number = int(phrase["number"])
        shift = starts[index] - original[index]
        end = starts[index] + durations[index]
        spill = max(0.0, end - plan_end) if index == len(starts) - 1 else 0.0
        spill_text = f" spill_past_plan={spill:.3f}s" if spill else ""
        print(
            f"  phrase {number:02d}: {original[index]:.3f}->{starts[index]:.3f} "
            f"shift={shift:+.3f}s end={end:.3f}{spill_text}"
        )
    return starts


def balance_loaded_phrases(
    loaded: list[tuple[dict[str, Any], np.ndarray, int]],
    *,
    max_gain_db: float,
) -> list[tuple[dict[str, Any], np.ndarray, int]]:
    if not loaded:
        return loaded
    rms_values = [active_rms(wav) for _phrase, wav, _sr in loaded]
    positive = [value for value in rms_values if value > 0]
    if not positive:
        return loaded
    target = float(np.median(positive))
    balanced: list[tuple[dict[str, Any], np.ndarray, int]] = []
    max_gain_db = max(0.0, float(max_gain_db))
    print(f"balance phrases target_rms={target:.6f} max_gain_db={max_gain_db:.2f}")
    for (phrase, wav, sr), rms in zip(loaded, rms_values):
        if rms <= 0 or target <= 0:
            gain_db = 0.0
        else:
            gain_db = 20.0 * float(np.log10(target / rms))
            gain_db = max(-max_gain_db, min(max_gain_db, gain_db))
        gain = float(10.0 ** (gain_db / 20.0))
        adjusted = (wav * gain).astype(np.float32)
        peak = float(np.max(np.abs(adjusted))) if adjusted.size else 0.0
        if peak > 0.98:
            peak_gain = 0.98 / peak
            adjusted = (adjusted * peak_gain).astype(np.float32)
            gain_db += 20.0 * float(np.log10(peak_gain))
        print(f"  phrase {int(phrase['number']):02d}: rms={rms:.6f} gain={gain_db:+.2f}dB")
        balanced.append((phrase, adjusted, sr))
    return balanced


def active_rms(wav: np.ndarray) -> float:
    if wav.size == 0:
        return 0.0
    abs_wav = np.abs(wav)
    peak = float(np.max(abs_wav))
    if peak <= 0:
        return 0.0
    active = wav[abs_wav >= peak * 0.03]
    if active.size == 0:
        active = wav
    return float(np.sqrt(np.mean(np.square(active.astype(np.float64)))))


def fit_audio_to_duration(source: Path, output: Path, target_duration: float) -> None:
    current_duration = audio_duration(source)
    if current_duration <= 0:
        raise SystemExit(f"Cannot fit zero-duration audio: {source}")
    tempo = current_duration / target_duration
    filters = ",".join(atempo_chain(tempo))
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-af",
        filters,
        "-ar",
        "48000",
        "-ac",
        "1",
        str(output),
    ]
    subprocess.run(command, check=True)


def apply_tempo(source: Path, output: Path, tempo: float) -> None:
    filters = ",".join(atempo_chain(tempo))
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-af",
        filters,
        "-ar",
        "48000",
        "-ac",
        "1",
        str(output),
    ]
    subprocess.run(command, check=True)


def atempo_chain(tempo: float) -> list[str]:
    if tempo <= 0:
        raise ValueError(f"Invalid tempo: {tempo}")
    filters: list[str] = []
    remaining = tempo
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.8f}")
    return filters


def write_review_md(path: Path, plan: dict[str, Any], manifest: dict[str, Any]) -> None:
    start_clock, end_clock = plan_time_range(plan, manifest)
    lines = [
        f"# {plan['episode']} Next-Episode Summary TTS",
        "",
        f"- Summary id: `{plan['summary_id']}`",
        f"- Time range: `{start_clock}` -> `{end_clock}`",
        f"- Language: `{manifest['language']}`",
        f"- Takes per phrase: `{manifest['takes_per_phrase']}`",
        f"- Reference audio: `{manifest['reference_audio']}`",
        f"- All-v01 preview: `{manifest['assembled_all_v01_preview']}`",
        "",
        "## Phrases",
        "",
    ]
    for phrase in manifest["phrases"]:
        lines.append(
            f"- `{phrase['number']:02d}` `{seconds_to_clock(float(phrase['start']))}` "
            f"({phrase['target_duration']:.3f}s): {phrase['text']}"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            "Edit `selected_takes.json` to switch any phrase from `v01` to another take.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plan_time_range(plan: dict[str, Any], manifest: dict[str, Any]) -> tuple[str, str]:
    start_clock = plan.get("start_clock")
    end_clock = plan.get("end_clock")
    if start_clock and end_clock:
        return str(start_clock), str(end_clock)

    phrases = plan.get("phrases") or manifest.get("phrases") or []
    if "start" in plan:
        start_seconds = float(plan["start"])
    elif phrases:
        start_seconds = float(phrases[0].get("start", 0.0))
    else:
        start_seconds = 0.0

    if "end" in plan:
        end_seconds = float(plan["end"])
    elif phrases:
        end_seconds = float(phrases[-1].get("end", start_seconds))
    else:
        end_seconds = start_seconds

    return seconds_to_clock(start_seconds), seconds_to_clock(end_seconds)


def seconds_to_clock(value: float) -> str:
    milliseconds_total = max(0, int(round(value * 1000)))
    seconds_total, milliseconds = divmod(milliseconds_total, 1000)
    minutes_total, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


if __name__ == "__main__":
    raise SystemExit(main())
