from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one ClearVoice enhancement variant.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=["se48k", "sr48k", "se48k_sr48k"])
    args = parser.parse_args()

    from clearvoice import ClearVoice

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.variant == "se48k":
        enhancer = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
        enhanced = enhancer(input_path=str(args.input), online_write=False)
        enhancer.write(enhanced, output_path=str(args.output))
        return 0

    if args.variant == "sr48k":
        super_resolver = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
        resolved = super_resolver(input_path=str(args.input), online_write=False)
        super_resolver.write(resolved, output_path=str(args.output))
        return 0

    temp_output = args.output.with_name(args.output.stem + ".se48k.tmp.wav")
    enhancer = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    enhanced = enhancer(input_path=str(args.input), online_write=False)
    enhancer.write(enhanced, output_path=str(temp_output))

    super_resolver = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
    resolved = super_resolver(input_path=str(temp_output), online_write=False)
    super_resolver.write(resolved, output_path=str(args.output))
    temp_output.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
