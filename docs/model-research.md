# Current Tool Notes

Checked on 2026-06-24.

## Demucs

The original `facebookresearch/demucs` repository is archived as of 2025-01-01. It remains useful as a baseline for `--two-stems=vocals`, but it should not be the only dependency for this project.

Source: https://github.com/facebookresearch/demucs

## audio-separator

`python-audio-separator` is a CLI/Python wrapper for stem separation using many pretrained models, primarily from the UVR ecosystem. This looks better for automation than a GUI-only UVR workflow.

Source: https://github.com/nomadkaraoke/python-audio-separator

## ClearerVoice-Studio

ClearerVoice-Studio is speech-focused and supports speech enhancement, separation, super-resolution, and target-speaker extraction. It is a strong candidate for cleaning old Spanish dialogue after basic FFmpeg cleanup.

Source: https://github.com/modelscope/ClearerVoice-Studio

## DeepFilterNet

DeepFilterNet is a local speech enhancement/noise suppression framework. It should be tested gently because aggressive denoise can damage VHS-era voice character.

Source: https://github.com/Rikorose/DeepFilterNet

## Practical Recommendation

Start with FFmpeg-only extraction and cleanup samples, then compare:

1. Demucs vocal stem baseline.
2. audio-separator UVR/MDX vocal models.
3. ClearerVoice enhancement on the extracted Spanish dialogue.
4. DeepFilterNet as a conservative denoise pass.

Do not pick a final model from one clean scene. Use at least one dialogue scene, one action scene, and one music-under-dialogue scene.

`spa2` is intentionally excluded from model tests. It can be used only for optional human reference listening, never as replacement dialogue.
