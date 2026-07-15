#!/usr/bin/env python3
"""Run Stable Audio Tools inpainting for a prepared audio gap.

The production use here is conservative:

1. Give the model a context file with the target gap silenced.
2. Ask it to generate only the gap region.
3. Extract/filter/gain that generated gap as a texture.
4. Overlay the texture under an approved base mix that still contains the
   hand-approved replacement dialogue.

This avoids letting a generative model rewrite the already approved voice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio.functional as AF
import soundfile as sf
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-audio-root", type=Path, default=Path("soft/ai_audio_inpaint/stable-audio-tools"))
    parser.add_argument("--pretrained-name", default="stabilityai/stable-audio-open-1.0")
    parser.add_argument("--context-audio", type=Path, required=True, help="Audio with the target gap silenced")
    parser.add_argument("--overlay-base", type=Path, required=True, help="Approved audio that keeps the fixed voice")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="stable_audio_inpaint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="speech, dialogue, voice, talking, words, narration, music change")
    parser.add_argument("--gap-start", type=float, required=True, help="Gap start seconds in the context/overlay files")
    parser.add_argument("--gap-end", type=float, required=True, help="Gap end seconds in the context/overlay files")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sampler-type", default="dpmpp-3m-sde")
    parser.add_argument("--sigma-min", type=float, default=0.03)
    parser.add_argument("--sigma-max", type=float, default=1000.0)
    parser.add_argument("--init-noise-level", type=float, default=1.0)
    parser.add_argument("--model-half", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--texture-gain-db", type=float, default=-6.0)
    parser.add_argument("--texture-highpass", type=float, default=4500.0)
    parser.add_argument("--texture-lowpass", type=float, default=12000.0)
    parser.add_argument("--fade-in", type=float, default=0.14)
    parser.add_argument("--fade-out", type=float, default=0.06)
    return parser.parse_args()


def add_stable_audio_to_path(root: Path) -> None:
    if not root.exists():
        raise SystemExit(f"Stable Audio Tools root does not exist: {root}")
    sys.path.insert(0, str(root.resolve()))


def load_audio(path: Path) -> tuple[int, torch.Tensor]:
    audio_np, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = torch.from_numpy(audio_np).transpose(0, 1).contiguous()
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return sr, audio


def save_audio(path: Path, audio: torch.Tensor, sr: int) -> None:
    audio_np = audio.detach().cpu().transpose(0, 1).numpy()
    sf.write(path, audio_np, sr, subtype="PCM_24")


def match_length(audio: torch.Tensor, samples: int) -> torch.Tensor:
    if audio.shape[-1] > samples:
        return audio[..., :samples]
    if audio.shape[-1] < samples:
        return torch.nn.functional.pad(audio, (0, samples - audio.shape[-1]))
    return audio


def db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)


def apply_texture_filter(texture: torch.Tensor, sr: int, highpass: float, lowpass: float) -> torch.Tensor:
    if highpass > 0:
        texture = AF.highpass_biquad(texture, sr, highpass)
    if lowpass > 0 and lowpass < sr / 2:
        texture = AF.lowpass_biquad(texture, sr, lowpass)
    return texture


def apply_fades(texture: torch.Tensor, sr: int, fade_in: float, fade_out: float) -> torch.Tensor:
    samples = texture.shape[-1]
    fade_in_samples = min(int(round(fade_in * sr)), samples // 2)
    fade_out_samples = min(int(round(fade_out * sr)), samples // 2)
    if fade_in_samples > 0:
        ramp = torch.linspace(0.0, 1.0, fade_in_samples, dtype=texture.dtype, device=texture.device)
        texture[..., :fade_in_samples] *= ramp
    if fade_out_samples > 0:
        ramp = torch.linspace(1.0, 0.0, fade_out_samples, dtype=texture.dtype, device=texture.device)
        texture[..., -fade_out_samples:] *= ramp
    return texture


def main() -> int:
    args = parse_args()
    add_stable_audio_to_path(args.stable_audio_root)

    from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint
    from stable_audio_tools.models.pretrained import get_pretrained_model

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available to this Python process; falling back to CPU.")
        device = "cpu"
    print(f"Using device: {device}")

    context_sr, context_audio = load_audio(args.context_audio)
    base_sr, base_audio = load_audio(args.overlay_base)
    if context_sr != base_sr:
        base_audio = AF.resample(base_audio, base_sr, context_sr)
        base_sr = context_sr
    total_samples = context_audio.shape[-1]
    base_audio = match_length(base_audio, total_samples)
    seconds_total = total_samples / context_sr

    try:
        model_config_path = hf_hub_download(args.pretrained_name, filename="model_config.json", repo_type="model")
    except Exception as exc:
        text = str(exc)
        if "gated repo" in text.lower() or "403" in text or "restricted" in text.lower():
            raise SystemExit(
                "Cannot download the requested Stable Audio model because Hugging Face "
                "reports it as gated/restricted.\n\n"
                f"Model: {args.pretrained_name}\n\n"
                "Fix:\n"
                "  1. Visit the model page in a browser and request/accept access terms.\n"
                "  2. Log in with the new Hugging Face CLI:\n"
                "     .venv-inpaint/bin/hf auth login\n"
                "  3. Run the inpaint command again.\n\n"
                "If access is granted but the model is not an inpainting checkpoint, "
                "the next validation step will stop with a model_type message."
            ) from exc
        raise
    with open(model_config_path, encoding="utf-8") as fh:
        model_config = json.load(fh)
    model_type = model_config.get("model_type")
    is_stable_audio_3 = args.pretrained_name.startswith("stabilityai/stable-audio-3")
    supports_inpaint_runner = model_type == "diffusion_cond_inpaint" or is_stable_audio_3
    if not supports_inpaint_runner:
        raise SystemExit(
            f"Model {args.pretrained_name!r} reports model_type={model_type!r}, "
            "but this runner needs either a diffusion_cond_inpaint checkpoint "
            "or a Stable Audio 3 checkpoint supported by generate_diffusion_cond_inpaint. "
            "This check happens before loading model weights, so no GPU time was spent."
        )
    print(f"Loading Stable Audio inpainting model: {args.pretrained_name}")
    model, model_config = get_pretrained_model(args.pretrained_name)
    model = model.to(device)
    use_half = args.model_half or device == "cuda"
    if use_half:
        print("Using half precision for Stable Audio model.")
        model = model.half()
    model.eval()
    model_dtype = torch.float16 if use_half else torch.float32

    model_sample_rate = int(getattr(model, "sample_rate", context_sr))
    sample_size = int(model_config.get("sample_size", int(seconds_total * model_sample_rate)))
    sample_size = max(sample_size, int(seconds_total * model_sample_rate))

    if context_sr != model_sample_rate:
        inpaint_audio = AF.resample(context_audio, context_sr, model_sample_rate)
    else:
        inpaint_audio = context_audio
    inpaint_audio = inpaint_audio.to(device=device, dtype=model_dtype)

    conditioning = [
        {
            "prompt": args.prompt,
            "seconds_start": 0,
            "seconds_total": seconds_total,
        }
    ]
    negative_conditioning = [
        {
            "prompt": args.negative_prompt,
            "seconds_start": 0,
            "seconds_total": seconds_total,
        }
    ] if args.negative_prompt else None

    with torch.no_grad():
        generated = generate_diffusion_cond_inpaint(
            model=model,
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            batch_size=1,
            sample_size=sample_size,
            seed=args.seed,
            device=device,
            sampler_type=args.sampler_type,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            init_noise_level=args.init_noise_level,
            inpaint_audio=(model_sample_rate, inpaint_audio),
            inpaint_mask_start_seconds=args.gap_start,
            inpaint_mask_end_seconds=args.gap_end,
            adapt_duration_to_conditioning=True,
            duration_padding_sec=0.0,
        )

    generated = generated.detach().to(torch.float32).cpu()
    if generated.ndim == 3:
        generated = generated[0]
    if generated.shape[0] > 2:
        generated = generated[:2]
    if generated.shape[0] == 1:
        generated = generated.repeat(2, 1)
    if model_sample_rate != context_sr:
        generated = AF.resample(generated, model_sample_rate, context_sr)

    generated = match_length(generated, total_samples)
    start = int(round(args.gap_start * context_sr))
    end = int(round(args.gap_end * context_sr))
    start = max(0, min(start, total_samples))
    end = max(start, min(end, total_samples))

    texture = torch.zeros_like(base_audio)
    gap_texture = generated[:, start:end].clone()
    gap_texture = apply_texture_filter(gap_texture, context_sr, args.texture_highpass, args.texture_lowpass)
    gap_texture = apply_fades(gap_texture, context_sr, args.fade_in, args.fade_out)
    gap_texture = gap_texture * db_to_linear(args.texture_gain_db)
    texture[:, start:end] = gap_texture
    overlay = torch.clamp(base_audio + texture, -0.98, 0.98)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"{args.label}_raw_model_output.wav"
    texture_path = args.out_dir / f"{args.label}_texture_only.wav"
    overlay_path = args.out_dir / f"{args.label}_OVER_APPROVED_PATCH.wav"
    save_audio(raw_path, generated, context_sr)
    save_audio(texture_path, texture, context_sr)
    save_audio(overlay_path, overlay, context_sr)

    manifest = {
        "backend": "stable-audio-tools",
        "pretrained_name": args.pretrained_name,
        "context_audio": str(args.context_audio),
        "overlay_base": str(args.overlay_base),
        "gap_start": args.gap_start,
        "gap_end": args.gap_end,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "texture_gain_db": args.texture_gain_db,
        "texture_highpass": args.texture_highpass,
        "texture_lowpass": args.texture_lowpass,
        "fade_in": args.fade_in,
        "fade_out": args.fade_out,
        "raw_model_output": str(raw_path),
        "texture_only": str(texture_path),
        "overlay_output": str(overlay_path),
    }
    (args.out_dir / f"{args.label}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"raw={raw_path}")
    print(f"texture={texture_path}")
    print(f"overlay={overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
