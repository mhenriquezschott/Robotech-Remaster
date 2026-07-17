#!/usr/bin/env python3
"""Run AudioSep-DP language-query separation for opening-credit SFX review."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TQSED_ROOT = ROOT / "soft/ai_audio_tools/src/TQ-SED"
DEFAULT_INPUT = ROOT / "work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav"
DEFAULT_OUT = ROOT / "work/review/opening_audio_audiosep_dp_001"
DEFAULT_PROMPTS = [
    "motorcycle engine sound effect, no music, no speech",
    "motorcycle engine revving and vehicle engine sound, no music, no speech",
    "laser gun sound effects, no music, no speech",
    "spaceship and laser blast sound effects, no music, no narration",
    "all non-music sound effects including motorcycle engine and laser guns, no music",
    "Spanish narrator voice saying Robotech, no music",
]
WINDOWS = {
    "voice_023p8_027p6": (23.8, 27.6),
    "laser_mid_023_027": (23.0, 27.0),
    "effects_late_055_063": (55.0, 63.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tqsed-root", type=Path, default=DEFAULT_TQSED_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--clap-checkpoint", type=Path, default=ROOT / "soft/ai_audio_tools/src/AudioSep/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt")
    parser.add_argument("--tokenizer", default="roberta-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--model-rate", type=int, choices=(16000, 32000), default=32000)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.25)
    parser.add_argument("--normalize-windows", action="store_true", default=True)
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    lass_root = args.tqsed_root / "LASS_codes"
    sys.path.insert(0, str(args.tqsed_root.resolve()))
    sys.path.insert(0, str(lass_root.resolve()))
    previous_cwd = Path.cwd()
    os.chdir(lass_root)
    try:
        model, model_info = load_model(args, device)
    finally:
        os.chdir(previous_cwd)

    source, source_sr = sf.read(args.input, always_2d=True, dtype="float32")
    mono = source.mean(axis=1)
    model_audio = resample(mono, source_sr, args.model_rate)

    manifest = {
        "input": str(args.input),
        "tqsed_root": str(args.tqsed_root),
        "checkpoint": str(model_info["checkpoint"]),
        "clap_checkpoint": str(args.clap_checkpoint),
        "tokenizer": args.tokenizer,
        "device": str(device),
        "model_rate": args.model_rate,
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "outputs": [],
        "notes": [
            "AudioSep-DP is the LASS separator released with TQ-SED.",
            "The TQ-SED event detector is not used here; this runner queries the separator directly.",
            "Review files are resampled to 48 kHz stereo for auditioning.",
        ],
    }

    for index, prompt in enumerate(prompts, start=1):
        label = f"{index:02d}_{slugify(prompt)}"
        print(f"[{index}/{len(prompts)}] {prompt}")
        with torch.no_grad():
            separated = separate_prompt(
                model=model,
                audio=model_audio,
                prompt=prompt,
                device=device,
                sample_rate=args.model_rate,
                chunk_seconds=args.chunk_seconds,
                overlap_seconds=args.overlap_seconds,
            )

        review_mono = resample(separated, args.model_rate, args.sample_rate)
        review_stereo = np.repeat(review_mono[:, None], 2, axis=1)
        raw_path = args.out_dir / f"{label}_audiosep_dp_{args.model_rate//1000}k_mono.wav"
        review_path = args.out_dir / f"{label}_audiosep_dp_48k_stereo.wav"
        norm_path = args.out_dir / f"{label}_audiosep_dp_48k_stereo_norm.wav"
        sf.write(raw_path, peak_guard(separated), args.model_rate, subtype="PCM_24")
        sf.write(review_path, peak_guard(review_stereo), args.sample_rate, subtype="PCM_24")
        sf.write(norm_path, peak_normalize(review_stereo, 0.85), args.sample_rate, subtype="PCM_24")

        windows_dir = args.out_dir / "windows"
        windows_dir.mkdir(exist_ok=True)
        for window_label, (start, end) in WINDOWS.items():
            clip = segment(review_stereo, args.sample_rate, start, end)
            sf.write(windows_dir / f"{label}_{window_label}.wav", peak_guard(clip), args.sample_rate, subtype="PCM_24")
            sf.write(
                windows_dir / f"{label}_{window_label}_norm.wav",
                peak_normalize(clip, 0.85),
                args.sample_rate,
                subtype="PCM_24",
            )

        manifest["outputs"].append(
            {
                "prompt": prompt,
                "label": label,
                "raw_model_rate_mono": str(raw_path),
                "review_48k_stereo": str(review_path),
                "normalized_48k_stereo": str(norm_path),
            }
        )

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(args.out_dir, manifest)
    print(f"review_dir={args.out_dir}")
    print(f"windows={args.out_dir / 'windows'}")
    return 0


def load_model(args: argparse.Namespace, device: torch.device):
    from transformers import RobertaModel, RobertaTokenizer

    patch_torch_load_for_legacy_checkpoints()
    patch_roberta_tokenizer(RobertaTokenizer, args.tokenizer)
    patch_roberta_tokenizer(RobertaModel, args.tokenizer)
    from LASS_codes.models.CLAP.open_clip import create_model

    if args.model_rate == 32000:
        config_yaml = args.tqsed_root / "LASS_codes/config/Fsd_Clo_Caps_Autotest_ResUNet_32k.yaml"
        checkpoint = args.checkpoint or find_checkpoint(args.tqsed_root, "resunet_with_dprnn_32k")
    else:
        config_yaml = args.tqsed_root / "LASS_codes/config/Fsd_Clo_Caps_Autotest_ResUNet_16k.yaml"
        checkpoint = args.checkpoint or find_checkpoint(args.tqsed_root, "resunet_with_dprnn_16k")

    configs = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    model_config = configs["model"]
    ss_model_class = get_ss_model_class(model_config["model_type"])
    ss_model = ss_model_class(
        input_channels=model_config["input_channels"],
        output_channels=model_config["output_channels"],
        condition_size=model_config["condition_size"],
        dprnn=model_config["dprnn"],
        dprnn_layers=model_config["dprnn_layers"],
        dprnn_hidden=model_config["dprnn_hidden"],
    )
    query_encoder = QueryEncoder(
        create_model=create_model,
        tokenizer=RobertaTokenizer.from_pretrained(args.tokenizer),
        clap_checkpoint=args.clap_checkpoint,
        sample_rate=args.model_rate,
        device=device,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    ss_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("ss_model."):
            ss_state_dict[key.removeprefix("ss_model.")] = value
        elif key.startswith("model.ss_model."):
            ss_state_dict[key.removeprefix("model.ss_model.")] = value
    if not ss_state_dict:
        raise RuntimeError(f"No ss_model weights found in checkpoint: {checkpoint}")
    missing, unexpected = ss_model.load_state_dict(ss_state_dict, strict=False)
    if missing:
        print(f"warning: missing ss_model keys: {len(missing)}")
    if unexpected:
        print(f"warning: unexpected ss_model keys: {len(unexpected)}")
    model = SeparatorBundle(ss_model=ss_model.to(device), query_encoder=query_encoder)
    model.eval()
    return model, {"checkpoint": checkpoint, "config": config_yaml}


def patch_torch_load_for_legacy_checkpoints() -> None:
    """Allow trusted legacy CLAP/AudioSep-DP checkpoints in this process."""

    original_load = torch.load

    def load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_legacy_default


def get_ss_model_class(model_type: str):
    if model_type == "ResUNet30_32k":
        from LASS_codes.models.resunet_32k import ResUNet30

        return ResUNet30
    if model_type == "ResUNet30":
        from LASS_codes.models.resunet import ResUNet30

        return ResUNet30
    raise NotImplementedError(f"Unsupported AudioSep-DP model type: {model_type}")


def patch_roberta_tokenizer(roberta_class, replacement: str) -> None:
    """Redirect TQ-SED's hardcoded author-local tokenizer path.

    Some upstream CLAP modules call `RobertaTokenizer.from_pretrained()` at
    import time with `/mnt/nfs2/.../robera-base`. We intercept that one path and
    replace it with our configured tokenizer, usually `roberta-base`.
    """

    original = roberta_class.from_pretrained

    def from_pretrained(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith("/mnt/nfs2/hanyin/LASS4SED/pretrained_models/"):
            path = replacement
        return original(path, *args, **kwargs)

    roberta_class.from_pretrained = from_pretrained


class SeparatorBundle(torch.nn.Module):
    def __init__(self, ss_model: torch.nn.Module, query_encoder: torch.nn.Module):
        super().__init__()
        self.ss_model = ss_model
        self.query_encoder = query_encoder


class QueryEncoder(torch.nn.Module):
    def __init__(self, create_model, tokenizer, clap_checkpoint: Path, sample_rate: int, device: torch.device):
        super().__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.tokenize = tokenizer
        self.encoder_type = "CLAP"
        self.model, self.model_cfg = create_model(
            "HTSAT-base",
            "roberta",
            str(clap_checkpoint),
            precision="fp32",
            device=str(device),
            enable_fusion=False,
            fusion_type="aff_2d",
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def get_query_embed(self, modality, audio=None, text=None, use_text_ratio=1.0, device=None):
        if modality != "text":
            raise NotImplementedError("This review runner uses text prompts only.")
        with torch.no_grad():
            text_data = self.tokenizer(text or [""])
            text_data = {key: value.to(self.device) for key, value in text_data.items()}
            embed = self.model.get_text_embedding(text_data)
        return embed.float()

    def tokenizer(self, text):
        return self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )


def find_checkpoint(tqsed_root: Path, family: str) -> Path:
    checkpoint_root = tqsed_root / "LASS_codes/checkpoints"
    matches = sorted(checkpoint_root.glob(f"**/{family}/**/*.ckpt")) + sorted(checkpoint_root.glob(f"**/{family}*/**/*.ckpt"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {family} checkpoint under {checkpoint_root}. "
            "Run: bash scripts/setup_audio_restoration_tools.sh download-audiosep-dp-checkpoints"
        )
    return matches[0]


def separate_prompt(
    *,
    model,
    audio: np.ndarray,
    prompt: str,
    device: torch.device,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> np.ndarray:
    condition = model.query_encoder.get_query_embed(modality="text", text=[prompt], device=device).to(device)
    chunk_samples = int(round(chunk_seconds * sample_rate))
    overlap_samples = int(round(overlap_seconds * sample_rate))
    hop_samples = max(1, chunk_samples - overlap_samples)
    output = np.zeros(audio.shape[0], dtype=np.float32)
    weights = np.zeros(audio.shape[0], dtype=np.float32)

    for start in range(0, len(audio), hop_samples):
        end = min(start + chunk_samples, len(audio))
        segment = audio[start:end]
        padded = np.zeros(chunk_samples, dtype=np.float32)
        padded[: len(segment)] = segment
        tensor = torch.from_numpy(padded[None, :]).to(device)
        result = model.ss_model({"mixture": tensor[:, None, :], "condition": condition})["waveform"]
        separated = result.squeeze().detach().float().cpu().numpy()[: len(segment)]
        fade = window_weights(len(segment), start == 0, end == len(audio), overlap_samples)
        output[start:end] += separated * fade
        weights[start:end] += fade
        if end == len(audio):
            break

    return (output / np.maximum(weights, 1e-8)).astype(np.float32)


def window_weights(length: int, is_first: bool, is_last: bool, overlap_samples: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    fade_len = min(overlap_samples, length // 2)
    if fade_len <= 0:
        return weights
    if not is_first:
        weights[:fade_len] *= np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)
    if not is_last:
        weights[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, endpoint=False, dtype=np.float32)
    return weights


def slugify(value: str) -> str:
    value = value.lower().replace("spanish", "spa")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:72]


def resample(data: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return data.astype(np.float32)
    return signal.resample_poly(data, target_sr, source_sr).astype(np.float32)


def segment(data: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    return data[max(0, int(start * sr)) : min(len(data), int(end * sr))]


def peak_guard(data: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.99:
        return data.astype(np.float32)
    return (data / peak * 0.99).astype(np.float32)


def peak_normalize(data: np.ndarray, peak: float) -> np.ndarray:
    current = float(np.max(np.abs(data))) if data.size else 0.0
    if current < 1e-9:
        return data.astype(np.float32)
    return (data * (peak / current)).astype(np.float32)


def write_readme(out_dir: Path, manifest: dict) -> None:
    lines = [
        "# Opening AudioSep-DP Prompt Tests",
        "",
        "AudioSep-DP is the LASS separator released with TQ-SED. These files are prompt-query extraction candidates.",
        "",
        "Review order:",
        "",
        "- Full `*_audiosep_dp_48k_stereo_norm.wav` files.",
        "- `windows/*voice_023p8_027p6*` for the Spanish narrator voice.",
        "- `windows/*laser_mid_023_027*` for voice/laser overlap.",
        "- `windows/*effects_late_055_063*` for later SFX.",
        "",
        "Prompts:",
        "",
    ]
    for item in manifest["outputs"]:
        lines.append(f"- `{item['label']}`: {item['prompt']}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
