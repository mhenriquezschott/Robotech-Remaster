# Robotech Remaster Tooling

This repository contains local tooling, scripts, documentation, and review
workflows for a preservation-oriented Robotech remaster project. The media
sources themselves are intentionally kept outside Git.

## Project Goal

The project rebuilds review and final episode muxes from restored video sources
and multiple audio/subtitle sources while avoiding unnecessary re-encoding of
the episode video streams. Most work is done as reproducible preparation steps:
extract, repair, document, review, promote, and then mux.

## Video Workflow

The video side starts from existing restored/remastered episode sources and
preserves them whenever possible with stream-copy muxing. The build pipeline
adds prepared opening credits, episode material, optional reconstructed
next-episode summary segments, end credits, subtitles, chapters, metadata, and
audio tracks.

For special cases, prepared video segments are generated separately and reused by
the episode builder. For example, `S01E36` has a reconstructed Southern
Cross/Masters next-episode summary prepared as reusable ready segments instead
of being rebuilt every time the episode is muxed.

AI/upscale/interpolation experiments are kept as review assets. The goal is to
compare methods such as Real-CUGAN/APISR/AnimeSR/RIFE-style interpolation
without forcing them into the final pipeline until a candidate is actually
approved.

## Audio Workflow

The audio workflow preserves the original language options while creating an
improved Spanish 5.1 track. The current practical chain for the old Latin
American Spanish dialogue is:

1. extract the old Spanish dialogue with `melband_roformer_instvoc_duality_v1`;
2. enhance the extracted dialogue with the `broadcast_strong` filter chain;
3. remove English dialogue from the English 5.1 center channel;
4. rebuild the Spanish 5.1 center channel with the restored Spanish dialogue;
5. keep the surrounding English 5.1 music/effects bed where it gives the best
   final result.

Episode-specific audio repairs are handled as ready patches or repair-tool
recipes. This allows exact phrase/word replacements, generated narrator fixes,
and local audio texture/inpainting experiments to be reviewed and reused.

## Opening And End Credits

Opening and end credits are treated as shared assets. The project can prepare
matching variants for different episode video formats and mux them without
re-encoding the episode video streams.

Current research is focused on rebuilding the opening credits audio more
cleanly: using the official soundtrack music as the base, recovering the
Spanish narrator voice and sound effects from current/TV-copy openings, and
testing whether restoration models or DSP can improve those extracted elements.

## Subtitles And Chapters

The subtitle workflow extracts, OCRs, translates, reviews, and fixes subtitles
as editable `.srt` files. Final muxes embed subtitles with language/title
metadata while also allowing exported sidecar `.srt` files.

The mux pipeline also writes chapter metadata without re-encoding video. Base
chapters include opening credits, episode, optional next-episode summary, and
end credits.

## Local Tooling

The repository includes code for:

- episode final muxing;
- opening/end-credit preparation;
- audio separation and enhancement tests;
- subtitle OCR/translation/review;
- Qwen3-TTS narrator summary generation;
- a custom audio repair GUI;
- optional AI audio/video restoration test setups.

Main environments currently used:

- `.venv-separation` for the main `robotech-ai` CLI and audio-separator work;
- `.venv-asr` for speech maps;
- `.venv-llm` for subtitle translation;
- `.venv-inpaint` for Stable Audio experiments;
- `.venv-audio-audiosr` for AudioSR tests;
- `.venv-audio-apollo` for Apollo tests;
- `.venv-gui` for the repair tool.

Large generated folders such as `work/`, `generated_audio/`, `Robotech/`,
`soft/`, and local virtual environments are excluded from Git.

