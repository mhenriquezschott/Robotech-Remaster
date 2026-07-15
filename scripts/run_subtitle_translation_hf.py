#!/usr/bin/env python3
"""Translate Robotech subtitle prompt batches with local Hugging Face models.

This runner is intentionally separate from the main `robotech-ai` CLI so the
heavy ML dependencies can live in a dedicated virtualenv. It consumes the JSONL
prompt batches created by `robotech-ai translate-spanish-subtitles --provider hf`
and writes both raw model responses and a validated SRT file.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robotech_ai.cli import (  # noqa: E402
    SrtCue,
    parse_translation_response,
    read_jsonl,
    read_srt,
    write_jsonl,
    write_srt,
)


@dataclass(frozen=True)
class LocalGeneration:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-jsonl", type=Path, required=True)
    parser.add_argument("--english-srt", type=Path, required=True)
    parser.add_argument("--output-srt", type=Path, required=True)
    parser.add_argument("--response-jsonl", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output SRT")
    args = parser.parse_args(argv)

    if args.limit_chunks is not None and args.limit_chunks <= 0:
        raise SystemExit("--limit-chunks must be greater than zero")
    if args.retries < 0:
        raise SystemExit("--retries must be zero or greater")
    if args.output_srt.is_file() and not args.overwrite:
        print(f"SKIP existing translated SRT: {args.output_srt}")
        print("Use --overwrite to rebuild it.")
        return 0

    started_all = time.perf_counter()
    configure_hf_environment(args.model_cache, offline=args.offline)
    model_config = read_model_config(args.model_config, offline=args.offline)
    model_config["known_speaker_labels"] = collect_speaker_labels(args.prompt_jsonl)
    require_python_packages(model_config)

    prompt_records = read_jsonl(args.prompt_jsonl)
    if args.limit_chunks is not None:
        prompt_records = prompt_records[: args.limit_chunks]
    if not prompt_records:
        raise SystemExit(f"No prompt records found in {args.prompt_jsonl}")

    english_cues = read_srt(args.english_srt)
    cue_by_index = {cue.index: cue for cue in english_cues}

    backend = HfSubtitleBackend(model_config=model_config, model_cache=args.model_cache)
    print(f"Loading {model_config['repo_id']} from {args.model_cache} offline={args.offline}", flush=True)
    try:
        backend.load()
    except Exception as exc:
        message = str(exc)
        if args.offline and "gpt-oss-triton-kernels" in message:
            raise SystemExit(
                "GPT-OSS model weights are cached, but its Triton kernel package is not cached yet: "
                "kernels-community/gpt-oss-triton-kernels.\n"
                "Run the same command once without --offline, or prefetch that kernel repo with "
                "`huggingface-cli download kernels-community/gpt-oss-triton-kernels --cache-dir "
                f"{args.model_cache}`. After that, --offline should work."
            ) from exc
        raise

    translated: dict[int, str] = {}
    response_records: list[dict[str, Any]] = []
    failed_response_files: list[str] = []
    try:
        for position, record in enumerate(prompt_records, start=1):
            cue_indexes = [int(value) for value in record["cue_indexes"]]
            response_record = translate_record(
                backend=backend,
                record=record,
                cue_indexes=cue_indexes,
                retries=args.retries,
                response_jsonl=args.response_jsonl,
            )
            response_records.append(response_record)
            failed_response_files.extend(response_record.get("failed_response_files", []))
            for item in response_record["translations"]:
                translated[int(item["index"])] = normalize_translated_text(
                    str(item["text"]).strip(),
                    model_config.get("known_speaker_labels", {}),
                )
            speed = response_record.get("tokens_per_second")
            speed_text = f" {speed:.2f} tok/s" if isinstance(speed, float) else ""
            print(
                f"chunk {position}/{len(prompt_records)} cues={cue_indexes[0]}-{cue_indexes[-1]}{speed_text}",
                flush=True,
            )
            write_jsonl(args.response_jsonl, response_records)
    finally:
        backend.unload()

    output_cues = build_output_cues(
        english_cues,
        translated,
        only_translated=args.limit_chunks is not None,
    )
    write_srt(args.output_srt, output_cues)
    total_elapsed = time.perf_counter() - started_all
    missing_cue_indexes = [cue.index for cue in output_cues if cue.index not in translated]
    speeds = [
        float(record["tokens_per_second"])
        for record in response_records
        if isinstance(record.get("tokens_per_second"), (int, float))
    ]
    average_speed = round(sum(speeds) / len(speeds), 3) if speeds else None
    manifest = {
        "prompt_jsonl": str(args.prompt_jsonl),
        "response_jsonl": str(args.response_jsonl),
        "english_srt": str(args.english_srt),
        "output_srt": str(args.output_srt),
        "model_config": str(args.model_config),
        "model_cache": str(args.model_cache),
        "offline": args.offline,
        "processed_chunks": len(prompt_records),
        "translated_cues": len(translated),
        "missing_cue_indexes": missing_cue_indexes,
        "failed_response_files": failed_response_files,
        "elapsed_seconds": round(total_elapsed, 3),
        "average_tokens_per_second": average_speed,
    }
    manifest_path = args.output_srt.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"translated_srt={args.output_srt}")
    print(f"responses={args.response_jsonl}")
    print(f"manifest={manifest_path}")
    print(
        "summary: "
        f"chunks={len(prompt_records)} translated_cues={len(translated)} "
        f"missing={len(missing_cue_indexes)} failed_attempts={len(failed_response_files)} "
        f"elapsed={format_duration(total_elapsed)}"
        + (f" avg_speed={average_speed} tok/s" if average_speed is not None else "")
    )
    if failed_response_files:
        print("notes: one or more attempts needed JSON repair/retry; inspect failed_response_files in the manifest if needed.")
    return 0


def configure_hf_environment(model_cache: Path, *, offline: bool) -> None:
    model_cache = model_cache.expanduser().resolve()
    os.environ.setdefault("HF_HOME", str(model_cache.parent))
    os.environ.setdefault("HF_HUB_CACHE", str(model_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(model_cache))
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"


def read_model_config(path: Path, *, offline: bool) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if offline:
        config["local_files_only"] = True
    return config


def collect_speaker_labels(prompt_jsonl: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for record in read_jsonl(prompt_jsonl):
        prompt = str(record.get("prompt", ""))
        match = re.search(r"Speaker labels:\s*(\{.*?\})\n", prompt, flags=re.DOTALL)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            labels.update({str(key).upper(): str(value).upper() for key, value in parsed.items()})
    return labels


def require_python_packages(model_config: dict[str, Any]) -> None:
    missing = []
    for package in model_config.get("required_python_packages", []):
        import_name = package.get("import_name", package["name"])
        if importlib.util.find_spec(import_name) is None:
            missing.append(package["name"])
    if missing:
        raise SystemExit(
            f"Model {model_config.get('model_id', model_config.get('repo_id'))} needs missing package(s): "
            + ", ".join(missing)
        )


class HfSubtitleBackend:
    def __init__(self, *, model_config: dict[str, Any], model_cache: Path) -> None:
        self.model_config = model_config
        self.model_cache = model_cache
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = str(self.model_config.get("device", "cuda:0"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(
                "This model config requests CUDA, but PyTorch cannot see a GPU in this session. "
                "Run the command from your normal GPU-enabled terminal, or use a CPU/offload model config."
            )

        repo_id = self.model_config["repo_id"]
        revision = self.model_config.get("revision")
        trust_remote_code = bool(self.model_config.get("trust_remote_code", False))
        local_files_only = bool(self.model_config.get("local_files_only", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=str(self.model_cache),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "cache_dir": str(self.model_cache),
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "device_map": self.model_config.get("device_map", {"": device}),
        }
        quantization_config = build_quantization_config(self.model_config)
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = resolve_torch_dtype(self.model_config.get("torch_dtype", "auto"))

        self.model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompt: str) -> LocalGeneration:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")

        inputs = tokenize_prompt(self.tokenizer, prompt, self.model_config)
        device = self.model_config.get("device", "cuda:0")
        inputs = inputs.to(device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        generation_config = sanitize_generation_config(self.model_config.get("generation", {}))
        started = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=resolve_eos_token_ids(self.tokenizer, self.model_config),
                **generation_config,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        total_tokens = int(output_ids.shape[-1])
        output_tokens = max(0, total_tokens - input_tokens)
        generated_ids = output_ids[0, input_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        text = normalize_model_output(text, self.model_config)
        text = truncate_at_stop_strings(text, self.model_config.get("generation", {}).get("stop_strings", []))
        return LocalGeneration(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=elapsed,
        )

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def build_quantization_config(model_config: dict[str, Any]) -> Any | None:
    quantization = model_config.get("quantization")
    if not quantization:
        return None
    if quantization.get("method") != "bitsandbytes":
        raise SystemExit(f"Unsupported quantization method: {quantization.get('method')}")
    from transformers import BitsAndBytesConfig

    if quantization.get("load_in_4bit"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=resolve_torch_dtype(quantization.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=quantization.get("bnb_4bit_use_double_quant", True),
        )
    if quantization.get("load_in_8bit"):
        return BitsAndBytesConfig(load_in_8bit=True)
    raise SystemExit("bitsandbytes quantization requires load_in_4bit or load_in_8bit")


def resolve_torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise SystemExit(f"Unsupported torch_dtype: {name}")
    return mapping[name]


def tokenize_prompt(tokenizer: Any, prompt: str, model_config: dict[str, Any]) -> Any:
    messages = [
        {
            "role": "system",
            "content": "Return only valid JSON matching the requested schema. No markdown.",
        },
        {"role": "user", "content": prompt},
    ]
    if model_config.get("prompt_format") == "chat_template":
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    rendered = "\n\n".join(message["content"] for message in messages)
    return tokenizer(rendered, return_tensors="pt")


def sanitize_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(config)
    cleaned.pop("stop_strings", None)
    cleaned.pop("stop_string_merge_policy", None)
    if cleaned.get("do_sample") is False:
        cleaned.pop("temperature", None)
        cleaned.pop("top_p", None)
        cleaned.pop("top_k", None)
    return cleaned


def resolve_eos_token_ids(tokenizer: Any, model_config: dict[str, Any]) -> int | list[int] | None:
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for token in model_config.get("extra_eos_tokens", []):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            continue
        if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
            continue
        ids.append(int(token_id))
    ids = list(dict.fromkeys(ids))
    if not ids:
        return None
    return ids[0] if len(ids) == 1 else ids


def normalize_model_output(text: str, model_config: dict[str, Any]) -> str:
    if model_config.get("output_format") != "harmony":
        return text
    final_markers = [
        "assistantfinal",
        "<|start|>assistant<|channel|>final<|message|>",
        "<|channel|>final<|message|>",
    ]
    lowered = text.lower()
    positions = [(lowered.rfind(marker.lower()), marker) for marker in final_markers if lowered.rfind(marker.lower()) >= 0]
    if positions:
        position, marker = max(positions, key=lambda item: item[0])
        return text[position + len(marker) :].strip()
    if lowered.startswith("analysis"):
        json_start = text.find("{")
        return text[json_start:].strip() if json_start >= 0 else ""
    return text


def truncate_at_stop_strings(text: str, stop_strings: list[str]) -> str:
    positions = [text.find(stop) for stop in stop_strings if stop and text.find(stop) >= 0]
    return text[: min(positions)].strip() if positions else text.strip()


def translate_record(
    *,
    backend: HfSubtitleBackend,
    record: dict[str, Any],
    cue_indexes: list[int],
    retries: int,
    response_jsonl: Path,
) -> dict[str, Any]:
    last_error = ""
    failed_files: list[str] = []
    for attempt in range(1, retries + 2):
        result = backend.generate(str(record["prompt"]))
        try:
            translations = parse_translation_response(result.text)
            translations = complete_missing_translations(
                backend=backend,
                record=record,
                translations=translations,
                expected_indexes=cue_indexes,
                response_jsonl=response_jsonl,
                failed_files=failed_files,
                repair_attempts=max(1, min(retries, 2)),
            )
            validate_translations(translations, cue_indexes)
            elapsed = round(result.elapsed_seconds, 3)
            speed = result.output_tokens / result.elapsed_seconds if result.elapsed_seconds > 0 else None
            return {
                "chunk_index": record["chunk_index"],
                "cue_indexes": cue_indexes,
                "raw_response": result.text,
                "translations": translations,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "elapsed_seconds": elapsed,
                "tokens_per_second": round(speed, 3) if speed else None,
                "attempt": attempt,
                "failed_response_files": failed_files,
            }
        except (SystemExit, json.JSONDecodeError) as exc:
            last_error = str(exc)
            failed_path = failed_response_path(response_jsonl, int(record["chunk_index"]), attempt)
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_path.write_text(result.text, encoding="utf-8")
            failed_files.append(str(failed_path))
            print(f"chunk {record['chunk_index']} attempt {attempt} failed JSON validation: {last_error}", flush=True)
            print(f"raw failed response saved: {failed_path}", flush=True)
    raise SystemExit(f"Could not translate chunk {record['chunk_index']} after retries: {last_error}")


def failed_response_path(response_jsonl: Path, chunk_index: int, attempt: int) -> Path:
    return response_jsonl.with_name(f"{response_jsonl.stem}_chunk{chunk_index:03d}_attempt{attempt}_failed.txt")


def repair_failed_response_path(response_jsonl: Path, chunk_index: int, attempt: int) -> Path:
    return response_jsonl.with_name(f"{response_jsonl.stem}_chunk{chunk_index:03d}_repair{attempt}_failed.txt")


def complete_missing_translations(
    *,
    backend: HfSubtitleBackend,
    record: dict[str, Any],
    translations: list[dict[str, object]],
    expected_indexes: list[int],
    response_jsonl: Path,
    failed_files: list[str],
    repair_attempts: int,
) -> list[dict[str, object]]:
    current = translations_by_index(translations)
    missing = [index for index in expected_indexes if index not in current]
    if not missing:
        return ordered_translations(current, expected_indexes)

    cue_payload = extract_input_cue_payload(record)
    missing_payload = [cue for cue in cue_payload if int(cue.get("index", -1)) in missing]
    if not missing_payload:
        raise SystemExit(f"Missing cue indexes {missing}, but the prompt cue payload could not be recovered.")

    print(
        f"chunk {record['chunk_index']} missing cue indexes {missing}; requesting targeted repair",
        flush=True,
    )
    repair_prompt = build_missing_cue_repair_prompt(record, missing_payload)
    last_error = ""
    for attempt in range(1, repair_attempts + 1):
        result = backend.generate(repair_prompt)
        try:
            repaired = parse_translation_response(result.text)
            repaired_map = translations_by_index(repaired)
            missing_after_repair = [index for index in missing if index not in repaired_map]
            if missing_after_repair:
                raise SystemExit(f"Repair still missing cue indexes {missing_after_repair}")
            current.update({index: repaired_map[index] for index in missing})
            return ordered_translations(current, expected_indexes)
        except (SystemExit, json.JSONDecodeError) as exc:
            last_error = str(exc)
            failed_path = repair_failed_response_path(response_jsonl, int(record["chunk_index"]), attempt)
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_path.write_text(result.text, encoding="utf-8")
            failed_files.append(str(failed_path))
            print(f"chunk {record['chunk_index']} repair attempt {attempt} failed: {last_error}", flush=True)
            print(f"raw failed repair response saved: {failed_path}", flush=True)
    raise SystemExit(f"Could not repair missing cue indexes {missing}: {last_error}")


def translations_by_index(translations: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    by_index: dict[int, dict[str, object]] = {}
    for item in translations:
        by_index[int(item["index"])] = {"index": int(item["index"]), "text": str(item["text"]).strip()}
    return by_index


def ordered_translations(
    translations: dict[int, dict[str, object]],
    expected_indexes: list[int],
) -> list[dict[str, object]]:
    return [translations[index] for index in expected_indexes if index in translations]


def extract_input_cue_payload(record: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = str(record.get("prompt", ""))
    marker = "Input cues JSON:"
    marker_position = prompt.rfind(marker)
    if marker_position < 0:
        return []
    payload_text = prompt[marker_position + len(marker) :].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def build_missing_cue_repair_prompt(record: dict[str, Any], missing_payload: list[dict[str, Any]]) -> str:
    return (
        "You are repairing a Robotech subtitle translation JSON response.\n"
        "Return ONLY strict valid JSON. No markdown, comments, or code fences.\n"
        "Output schema: {\"translations\":[{\"index\":371,\"text\":\"...\"}]}\n"
        "Translate ONLY the input cues below. Preserve each listed index exactly once.\n"
        "Use natural Latin American Spanish, preserve Robotech names, translate bracketed labels, "
        "and convert imperial units to rounded metric units when useful.\n\n"
        f"Episode: {record.get('episode_id')} - {record.get('episode_title')}\n"
        f"Input cues JSON: {json.dumps(missing_payload, ensure_ascii=False)}"
    )


def format_duration(seconds_value: float) -> str:
    seconds_total = max(0, int(round(seconds_value)))
    hours, remainder = divmod(seconds_total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def validate_translations(translations: list[dict[str, object]], expected_indexes: list[int]) -> None:
    found = [int(item["index"]) for item in translations]
    if sorted(found) != sorted(expected_indexes):
        raise SystemExit(f"Expected cue indexes {expected_indexes}, got {found}")
    for item in translations:
        text = str(item["text"]).strip()
        if not text:
            raise SystemExit(f"Empty translation for cue {item['index']}")
        if re.search(r"[{}\\[\\]]", text) and text.count("{") != text.count("}"):
            raise SystemExit(f"Suspicious malformed text for cue {item['index']}: {text}")


def normalize_translated_text(text: str, speaker_labels: dict[str, str]) -> str:
    text = text.replace("\\[", "[").replace("\\]", "]")
    for english_label, spanish_label in speaker_labels.items():
        text = re.sub(
            rf"\[{re.escape(english_label)}\]",
            f"[{spanish_label}]",
            text,
            flags=re.IGNORECASE,
        )
    return text.strip()


def build_output_cues(
    english_cues: list[SrtCue],
    translated: dict[int, str],
    *,
    only_translated: bool,
) -> list[SrtCue]:
    cues = []
    for cue in english_cues:
        if only_translated and cue.index not in translated:
            continue
        text = translated.get(cue.index, cue.text)
        cues.append(SrtCue(index=cue.index, start=cue.start, end=cue.end, text=text))
    return cues


if __name__ == "__main__":
    raise SystemExit(main())
