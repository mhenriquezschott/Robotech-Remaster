# Robotech A.I. Restoration Architecture

This project should behave like a lab workflow first and a batch processor later.

## Non-Negotiable Restoration Policy

`track02(spa2)` is not part of the restoration target. It is a newer Spanish dub and should remain untouched. The pipeline may inventory it, but it must not use it to replace voices, build final dialogue, or influence the restored performance.

The target is:

1. Use `track01(spa1)` as the only Spanish performance source.
2. Extract old Latin American Spanish dialogue from `spa1`.
3. Improve that extracted voice as much as practical without changing the actor character.
4. Split the English 5.1 track.
5. Work on the English center channel because dialogue is expected there, mixed with some mono effects/music.
6. Reduce/remove English dialogue from the center channel.
7. Mix the restored Spanish dialogue into the rebuilt center channel.
8. Rebuild the 5.1 track from original English FL, FR, LFE, SL, SR plus the new Spanish center.
9. Also build a restored old `spa1` stereo track with `vhs_broadcast_full` for preservation/listening, while leaving `spa2` otherwise untouched.
10. Silence, not trim, the first 0.5 seconds of every final audio track before muxing.
11. Mux the prepared audio into each selected processed video variant from `Robotech/proc/MacrossSaga`, preferring `-B` video versions when present.
12. Apply episode-specific adjustment segments before the final credit concat when needed. Examples include a generated fade-to-black segment, a black-frame pad, or future small per-episode audio/video patches.
13. Append restored end credits by re-encoding only the credit segment to match the target video stream, then concatenate with stream copy. Do not re-encode the processed episode video unless there is no other viable path.

Episode adjustments must be represented as explicit generated segments or audio patches. The processed episode video stream itself remains copied, not re-encoded. Generated adjustment segments may be encoded because they are new material inserted between existing streams.

Final episode builds are cache-aware. Existing named intermediate audio/separation files are reused by default so manual swaps can be tested without re-running the whole audio chain. Use `--rebuild-intermediates` only when the cache should be deliberately regenerated.

If `spa1` source type is uncertain, treat it as stereo until a full-episode analysis or manual waveform/listening review proves that downmixing is safe.

The working rule is:

1. Probe and report the real files.
2. Extract lossless working audio.
3. Test short clips with multiple cleanup and separation chains.
4. Save the winning settings per episode.
5. Run two or three complete pilot episodes.
6. Batch only after the pilot results are approved.

## Source Layout

The current `MacrossSaga` source set is expected to contain 36 groups:

- `Robotech-S01E##.mp4`
- `Robotech-S01E##.track00(eng).ac3`
- `Robotech-S01E##.track01(spa1).ac3`
- `Robotech-S01E##.track02(spa2).ac3`

The MP4 also appears to contain the same audio streams, but the sidecar AC3 files are the cleaner automation target because their roles are explicit.

For final episode builds, the `Robotech/bluraytrimcrop/MacrossSaga/*.mp4`
files are not used as the picture source. `episode-final-build` takes video
from `Robotech/proc/MacrossSaga/ep##/` and uses `bluraytrimcrop` for the
sidecar AC3 tracks. Older sample/review commands may still expect the
`bluraytrimcrop` MP4s, but the final mux pipeline does not require them.

Processed video selection keeps three variants per episode:

- `_AIRemaster`, preferring `-B` when present,
- `_Remaster`, preferring `-B` when present,
- `_Remaster...W2xEX...VFI`, preferring `-B` when present.

Files such as `Macross-*_nooped*.mp4`, Premiere `.prproj` files, `copia` files,
and non-selected pre-`-B` duplicates are not used by the current final mux
pipeline.

## Stages

| Stage | Folder | Purpose |
|---|---|---|
| Probe | `work/01_probe/` | Inventory, ffprobe metadata, warnings |
| Extract | `work/02_extract/` | Lossless WAV working files and split 5.1 channels |
| Analysis | `work/03_analysis/` | Stereo correlation, loudness, phase, duration reports |
| Samples | `work/04_samples/` | Short clips for model comparisons |
| Clean spa1 | `work/05_clean_spa1/` | Conservative VHS cleanup variants |
| Separate spa1 | `work/06_separate_spa1/` | Spanish dialogue stems from old dub |
| Separate English center | `work/07_separate_eng_center/` | De-voiced center bed experiments |
| Sync | `work/08_sync/` | Offset and drift-corrected Spanish dialogue |
| Center mix | `work/09_center_mix/` | Rebuilt Spanish center |
| Rebuild 5.1 | `work/10_rebuild_51/` | Lossless 5.1 masters |
| Full spa1 restoration | `work/13_spa1_fullmix/` | Preservation-oriented tests that improve the old Spanish mix as a whole |
| Episode review builds | `work/review/episodes/S01E##/<review-name>/` | Audio tracks, episode-only muxes, adjustment segments, authored final MKVs with embedded SRT subtitles/cover art, reports |
| Episode adjustments | `work/review/episodes/S01E##/<review-name>/segments/adjustments/` | Per-episode/per-variant fixes such as generated end fade-to-black segments |
| Subtitles | `work/review/subtitles/S01E##/` | OCR English SRTs and local-LLM Spanish translations |
| Final | `final/` | Delivery encodes and muxed files |

## First CLI Commands

Run discovery and metadata probing:

```bash
scripts/robotech-ai inventory
```

Create per-episode configs:

```bash
scripts/robotech-ai init-configs
```

Analyze Spanish stereo behavior for one episode:

```bash
scripts/robotech-ai analyze-stereo S01E01 --seconds 180
```

Analyze the full `spa1` episode before deciding mono/stereo handling:

```bash
scripts/robotech-ai analyze-stereo S01E01 --seconds 0 --update-config
```

Promote the measured `spa1` source classification into the episode config:

```bash
scripts/robotech-ai analyze-stereo S01E01 --seconds 180 --update-config
```

Print extraction commands for the first lab episode:

```bash
scripts/robotech-ai extract S01E01 --start 00:05:00 --duration 00:00:45
```

Actually run short lab extraction:

```bash
scripts/robotech-ai extract S01E01 --start 00:05:00 --duration 00:00:45 --run
```

`spa2` is excluded by default. Only include it when intentionally making reference clips:

```bash
scripts/robotech-ai make-samples S01E01 --include-spa2 --run
```

## Manual Review Points

I will ask you to listen or check Audacity when:

- `spa1` analysis is not clearly dual mono across the full episode.
- A downmix might cancel ambience, effects, or phase-shifted material.
- A model output improves intelligibility but changes the voice character.
- Residual English dialogue remains in the center bed.
- Sync looks plausible numerically but feels wrong by ear.

For Audacity checks, useful views are:

- stereo waveform similarity,
- phase/correlation meter if available,
- spectrogram for hiss/hum bands,
- A/B listening between raw `spa1`, cleaned `spa1`, extracted dialogue, and final center mix.

Create short test clips:

```bash
scripts/robotech-ai make-samples S01E01 --start 00:05:00 --duration 00:00:45 --run
```

## Model Candidates To Test

Model choices must be treated as experiments. Keep the exact command, model name, version, and parameters in the episode config notes.

Initial candidates:

- `audio-separator`: automation-friendly wrapper around UVR-style separation models.
- Demucs / HTDemucs: useful baseline vocal/dialogue separation, but the original Meta repo is archived.
- ClearerVoice-Studio: speech-focused enhancement, separation, and super-resolution.
- DeepFilterNet: local speech denoise/enhancement candidate.
- FFmpeg filters: high-pass, low-pass, hum removal, loudness measurement, channel split, muxing.

The first model test should use short samples from:

- normal dialogue,
- action with effects,
- music under dialogue,
- visibly noisy or damaged speech.
