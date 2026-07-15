# Audio Background Fill Notes

These are notes for making patched dialogue sections feel less empty after a word/phrase replacement.

## Current Problem

For S01E03, the selected narrator title repair is now:

```text
work/review/S01E03_ai_inpaint_vs_approved_002/03_seed23_ai_overlay_as_generated_full_7s.wav
```

It uses the recovered Space Fold title voice plus the Stable Audio seed 23 texture at the generated level.

Current high-frequency texture experiment:

```text
work/review/S01E03_title_narrator_highpass_001/
```

This folder tests whether the missing Veritech/turbine whistle can be recovered from the original wrong-title slot or nearby material using high-pass/band-pass filtering, before considering any audio inpainting.

Current synthetic fill experiment:

```text
work/review/S01E03_title_narrator_synthfill_002/
```

This folder tests audible generated turbine-ish beds after the high-pass extraction proved contaminated by old wrong-title voice. The earlier `synthfill_001` pass was rejected because its texture sources were too quiet to evaluate.

Current inpainting input/profile-fill experiment:

```text
work/review/S01E03_title_narrator_inpaint_001/
```

This folder contains clean gap-silenced inputs for neural inpainting tests plus a local profile-fill baseline created by `scripts/audio_gap_profile_fill.py`. The profile-fill outputs are not AI model results; they are controlled references before trying heavier neural tools.

Previous approved profile-fill texture asset:

```text
generated_audio/titlenarrator/s01e03_title_texture_exp180.wav
```

This is the older `05_exp180` texture layer from the smoothed profile-fill review set. It remains as provenance, but it is no longer the default S01E03 patch texture.

Current approved Stable Audio texture asset:

```text
generated_audio/titlenarrator/s01e03_title_texture_stable_seed23_as_generated.wav
```

This is the seed 23 inpaint texture extracted from the S01E03 gap at the generated level. The review comparison is under:

```text
work/review/S01E03_ai_inpaint_vs_approved_002/
```

The Stable Audio inpainting input snippets are also stored as stable assets:

```text
generated_audio/titlenarrator/s01e03_title_inpaint_context_gap_silenced.wav
generated_audio/titlenarrator/s01e03_title_approved_patch_context.wav
```

Stable Audio neural inpainting command:

```bash
scripts/robotech-ai ai-inpaint-stable \
  --seeds 11 23 \
  --steps 8 \
  --cfg-scale 1 \
  --sampler-type pingpong \
  --texture-gain-db -6 \
  --run
```

Outputs go to:

```text
work/review/S01E03_title_narrator_ai_inpaint_stable_001/
```

The runner preserves the approved fixed-voice patch and only overlays the generated texture from the masked gap. It requires the dedicated `.venv-inpaint` environment documented in `requirements-inpaint.txt`. Run this from a terminal that can see CUDA; the sandbox used by Codex may not expose the NVIDIA driver.

Important model caveat: the command is now aimed at `stabilityai/stable-audio-3-medium`, which the official model card documents with `generate_diffusion_cond_inpaint`. This model is gated separately from `stable-audio-open-1.0`; accept the Stable Audio 3 Medium terms before running.

Stable Audio Tools imports `pytorch_lightning` during model construction even for
inference, so `.venv-inpaint` includes `pytorch_lightning==2.5.5` and
`torchmetrics==0.11.4`.

RTX 5090 note: the upstream Stable Audio Tools lock pins Torch `2.7.1+cu126`,
which cannot execute on `sm_120`. The `.venv-inpaint` env intentionally
overrides that stack with:

```text
torch 2.12.1+cu130
torchaudio 2.11.0+cu130
torchvision 0.27.1+cu130
```

Keep that override unless Stable Audio Tools publishes a newer official lock
that supports RTX 5090.

If Hugging Face returns a gated-model or 403 error, request/accept access on the
model page and log in with:

```bash
.venv-inpaint/bin/hf auth login
```

`huggingface-cli login` is deprecated in the current Hub package.

## Candidate Techniques

1. **Nearby ambience bed loop**

   Build a short bed from nearby non-dialogue fragments before/after the cut, then loop/crossfade it under the replacement. This is simple and controllable, but it needs careful source selection so it does not introduce repeated speech consonants.

2. **Spectral texture copy**

   Extract only low-energy or frequency-limited texture from neighboring frames, for example high-passed hiss/air or low-passed room/body, and mix it under the inserted voice. This is safer than full old-slot texture because it avoids carrying recognizable wrong words.

3. **Mid/side or difference texture**

   If the wrong voice is mostly centered, test a side-channel-only texture from the original slot or nearby old Spanish fullmix. This may preserve VHS/background width while rejecting some centered speech.

4. **Short audio inpainting**

   For very small missing regions, interpolation/autoregressive/sparse methods can synthesize plausible continuity from surrounding audio. This is promising for edge clicks and tens-of-ms transitions, less reliable for a full ~1 second phrase replacement.

5. **Generative audio inpainting**

   Diffusion/GAN-style audio inpainting can target longer gaps, but it may hallucinate or create unstable music/texture. Treat as an experiment only, with strict listening review.

   The current implementation path uses Stable Audio Tools because its inference API includes a real conditional inpainting mode. Stable Audio 3 is the most relevant current research direction because it explicitly supports audio editing/inpainting, but the released Hugging Face checkpoint must still be verified locally before production use.

6. **Commercial ambience matching**

   Tools like RX/SpectraLayers-style ambience matching may be useful if available outside the pipeline. If used, export the generated ambience patch and keep it as an explicit episode-specific source so the code remains reproducible.

## Practical Rule

For this project, prefer explicit, reviewable beds:

- source segment path,
- start/end time,
- filter/gain,
- exact mix level,
- exact patch timing.

Avoid hidden AI ambience generation in the production pipeline until a specific result is reviewed and frozen.
