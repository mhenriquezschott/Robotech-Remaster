# Testing Workflow

The restoration should be developed in small, repeatable experiments.

## Episode Flow

1. Probe sources.
2. Analyze full `spa1` stereo behavior.
3. Extract short lab clips.
4. Try cleanup variants on `spa1`.
5. Try dialogue extraction models.
6. Try speech enhancement models on extracted voice.
7. De-voice the English center channel.
8. Build a short test center mix.
9. Ask for listening review.
10. Save winning settings in `config/episodes/S01E##.json`.

## Source Rules

- Restore only `spa1`.
- Do not use `spa2` as replacement dialogue.
- Do not assume `spa1` is mono from a short segment.
- If `spa1` is uncertain, process as stereo until full-episode analysis or manual review proves downmixing is safe.

## First Lab Episode

Start with `S01E01`, but do not trust it as the whole-season pattern. After it works, test at least:

- an early episode,
- a middle episode,
- a late episode,
- one episode where full `spa1` analysis looks stereo or imbalanced.

## Recommended Sample Windows

Create several 30-60 second windows:

- normal dialogue,
- action/effects under dialogue,
- music under dialogue,
- noisy or damaged voice,
- quiet voice.

Use:

```bash
scripts/robotech-ai extract S01E01 --start 00:05:00 --duration 00:00:45 --run
```

Generate old Spanish cleanup variants for the same window:

```bash
scripts/robotech-ai clean-spa1 S01E01 --variant raw --source-mode stereo --start 00:05:00 --duration 00:00:45 --run
scripts/robotech-ai clean-spa1 S01E01 --variant light --source-mode stereo --start 00:05:00 --duration 00:00:45 --run
scripts/robotech-ai clean-spa1 S01E01 --variant medium --source-mode stereo --start 00:05:00 --duration 00:00:45 --run
scripts/robotech-ai clean-spa1 S01E01 --variant clarity --source-mode stereo --start 00:05:00 --duration 00:00:45 --run
```

Prepare a separation command for a cleaned file:

```bash
scripts/robotech-ai separate-voice work/05_clean_spa1/S01E01/00-05-00_00-00-45/S01E01_spa1_stereo_light.wav --engine demucs --model htdemucs_ft
```

Run it only after the external tool is installed:

```bash
scripts/robotech-ai separate-voice work/05_clean_spa1/S01E01/00-05-00_00-00-45/S01E01_spa1_stereo_light.wav --engine demucs --model htdemucs_ft --run
```

Normalize separator outputs back to the project working format when needed:

```bash
scripts/robotech-ai normalize-wav work/06_separate_spa1/htdemucs_ft/S01E01_spa1_stereo_light/vocals.wav work/06_separate_spa1/htdemucs_ft/S01E01_spa1_stereo_light/vocals_48k_s24.wav --run
```

For the first Demucs gate, compare:

- cleaned input,
- extracted `vocals_48k_s24.wav`,
- extracted `no_vocals_48k_s24.wav`.

Pass criteria:

- `vocals` contains recognizable Spanish dialogue,
- voice artifacts are not worse than the original damage,
- music/effects bleed is low enough that a center replacement might be viable,
- `no_vocals` proves Demucs understood at least some voice/non-voice separation.

Current `S01E01` note:

- Demucs `htdemucs_ft` extracts the first seconds fairly well.
- It then collapses even though the cleaned source keeps strong audio level.
- Higher overlap extends the useful region only a little and adds noise.
- Treat this Demucs chain as promising but unstable for `spa1`.

Try audio-separator next, using the default BS-RoFormer model and a project-local model cache:

```bash
robotech-ai separate-voice work/05_clean_spa1/S01E01/00-05-00_00-00-20/S01E01_spa1_stereo_light.wav --engine audio-separator --single-stem Vocals --sample-rate 48000 --out-dir work/06_separate_spa1/audio_separator/default_bs_roformer_stereo_light --run
```

If it needs to download model metadata or model weights, let it do so once. The model cache should go under:

```text
work/models/audio-separator
```

## Review Folders

Before running more separators, pick useful source windows. This avoids wasting model tests on scenes that are mostly music, crowd noise, or effects:

```bash
robotech-ai audition-windows S01E01 \
  --starts 00:04:00 00:07:00 00:10:00 00:13:00 00:15:00 00:16:30 00:18:30 \
  --duration 00:00:20 \
  --variant light \
  --source-mode stereo \
  --review-name S01E01_source_window_pick_002 \
  --run
```

Listen only to the numbered files in the generated review folder and choose windows with clear dialogue, dialogue under music, and dialogue under action/noise. Then run `voice-gate` or a model shootout on those chosen starts.

For a first separator model shootout on one chosen window:

```bash
robotech-ai model-shootout S01E01 \
  --start 00:07:00 \
  --duration 00:00:20 \
  --variant light \
  --source-mode stereo \
  --review-name S01E01_model_shootout_001 \
  --run
```

The default shootout compares the package default BS-RoFormer against `UVR-MDX-NET-Voc_FT.onnx`, `Kim_Vocal_2.onnx`, and `MDX23C-8KFFT-InstVoc_HQ.ckpt`. The first run may download additional model files into `work/models/audio-separator`.

If the audition windows are all useful, run the same shootout across the full set:

```bash
robotech-ai model-shootout S01E01 \
  --starts 00:04:00 00:07:00 00:10:00 00:13:00 00:15:00 00:16:30 00:18:30 \
  --duration 00:00:20 \
  --variant light \
  --source-mode stereo \
  --review-name S01E01_model_shootout_all_001 \
  --run
```

This creates one review folder per start time, for example `work/review/S01E01_model_shootout_all_001_01_00-04-00`.

Earlier ensemble/cascade tests with `Kim_Vocal_2` and `MDX23C-8KFFT-InstVoc_HQ` were slow and did not clearly beat single models. Keep ensemble/cascade as fallback only, not the main path.

Current leading single-model family after `S01E01_more_singles_001_02_00-15-00`:

- `melband_roformer_instvoc_duality_v1.ckpt` gave the cleanest music-under-dialogue result.
- `melband_roformer_instvox_duality_v2.ckpt` was very close.
- `MDX23C-8KFFT-InstVoc_HQ.ckpt` remains a strong baseline because it preserves more voice body than some thinner models.

Current preferred restored 5.1 chain:

- `spa1` stereo light pre-clean,
- `melband_roformer_instvoc_duality_v1.ckpt` `Vocals` extraction,
- `broadcast_strong` dialogue enhancement,
- `melband_roformer_instvoc_duality_v1.ckpt` `Instrumental` extraction from English center,
- rebuilt English 5.1 with only the center channel replaced by de-voiced center bed plus restored Spanish dialogue.

This is the best practical chain so far, but it intentionally borrows the cleaner English music/effects bed. Keep the old `spa1` track in every review mux and final mux for preservation.

Validate only the real contenders across representative windows:

```bash
robotech-ai model-shootout S01E01 \
  --starts 00:04:00 00:07:00 00:10:00 00:13:00 00:15:00 00:16:30 00:18:30 \
  --duration 00:00:20 \
  --variant light \
  --source-mode stereo \
  --review-name S01E01_top_single_models_001 \
  --models \
    MDX23C-8KFFT-InstVoc_HQ.ckpt \
    melband_roformer_instvoc_duality_v1.ckpt \
    melband_roformer_instvox_duality_v2.ckpt \
  --run
```

If `Kim_Vocal_2` and `MDX23C-8KFFT-InstVoc_HQ` need to be revisited later, a small ensemble command remains available:

```bash
robotech-ai ensemble-shootout S01E01 \
  --starts 00:04:00 00:07:00 00:10:00 00:13:00 00:15:00 00:16:30 00:18:30 \
  --duration 00:00:20 \
  --variant light \
  --source-mode stereo \
  --review-name S01E01_ensemble_shootout_all_001 \
  --run
```

The default ensemble shootout compares `Kim_Vocal_2`, `MDX23C-8KFFT-InstVoc_HQ`, and three Kim+MDX23C ensembles: `avg_wave`, `median_wave`, and `uvr_max_spec`.

## Preservation-Oriented spa1 Tests

Two extra branches exist for checking whether useful original old-dub material can be preserved beyond the dialogue stem.

Full old `spa1` mix restoration keeps the original old Spanish dialogue, music, and effects together. It does not use the English bed and it does not extract dialogue:

```bash
scripts/robotech-ai spa1-fullmix-shootout S01E01 \
  --starts 00:04:00 00:07:00 00:10:00 00:13:00 00:15:00 00:16:30 00:18:30 \
  --duration 00:00:20 \
  --review-name S01E01_spa1_fullmix_restore_001 \
  --run
```

Review folder:

```text
work/review/S01E01_spa1_fullmix_restore_001
```

For each window, the numbered files are ordered as:

1. raw old `spa1`,
2. `dehiss_only`,
3. `tone_only`,
4. `vhs_broadcast_full`,
5. `vhs_dialogue_forward`,
6. `vhs_gentle`.

Old `spa1` music/effects extraction asks the same separator family for `Instrumental` instead of `Vocals`. Treat this as a scene-specific recovery probe, not a main-chain replacement. The output may contain old music/effects that are missing from the newer English/spa2 mixes, but it can also carry voice residue and VHS artifacts:

```bash
scripts/robotech-ai model-shootout S01E01 \
  --starts 00:13:00 00:15:00 \
  --duration 00:00:20 \
  --variant light \
  --source-mode stereo \
  --review-name S01E01_spa1_old_bed_music_probe_fast_001 \
  --models \
    melband_roformer_instvoc_duality_v1.ckpt \
    melband_roformer_instvox_duality_v2.ckpt \
  --single-stem Instrumental \
  --run
```

Review folders:

```text
work/review/S01E01_spa1_old_bed_music_probe_fast_001_01_00-13-00
work/review/S01E01_spa1_old_bed_music_probe_fast_001_02_00-15-00
```

For each old-bed probe folder:

1. cleaned `spa1` input,
2. `melband_roformer_instvoc_duality_v1.ckpt` instrumental/bed,
3. `melband_roformer_instvox_duality_v2.ckpt` instrumental/bed.

Only consider adding old-bed material scene by scene if the restored 5.1 mix clearly misses an original cue and the extracted old bed has low enough voice residue.

After each experiment group, copy only the files that need human listening into a flat review folder:

```bash
robotech-ai prepare-review S01E01_voice_gate_001 \
  work/05_clean_spa1/S01E01/00-05-00_00-00-20/S01E01_spa1_stereo_light.wav \
  work/06_separate_spa1/htdemucs_ft/S01E01_spa1_stereo_light/vocals_48k_s24.wav \
  work/06_separate_spa1/audio_separator/default_bs_roformer_stereo_light/S01E01_spa1_stereo_light_\(Vocals\)_model_bs_roformer_ep_317_sdr_12.wav \
  --labels input_spa1_light demucs_vocals audio_separator_vocals
```

Use only the numbered files inside `work/review/...` for listening notes. This avoids mixing up raw experiment outputs with the files that are actually under review.

Check installed tools:

```bash
scripts/robotech-ai tools-status
```

See [setup.md](setup.md) for the project virtual environment and AI tool installation.

Archive detailed source metadata for final mux planning:

```bash
scripts/robotech-ai archive-metadata S01E01
```

## Parameters To Track

For every test, record:

- source episode and time window,
- whether `spa1` was stereo or downmixed,
- cleanup filters and values,
- separation model name and version,
- enhancement model name and version,
- de-voice method for English center,
- dialogue gain,
- center bed gain,
- audio start silence duration,
- limiter/normalization settings,
- subjective notes from listening.

## Approval Standard

A chain is not selected because it is technically impressive. It is selected only if:

- the original Spanish actor character survives,
- dialogue intelligibility improves,
- artifacts are less distracting than the original noise,
- residual English dialogue is not noticeable,
- sync feels correct,
- the center channel blends naturally with the rest of the 5.1 mix.

## Final Media Intent

The final mux command produces one folder under `work/review/episodes/S01E##/`, containing the prepared audio tracks, episode-only segments, and one mux per processed video variant:

- AI Remaster video,
- Remaster video,
- Remaster 49fps video,
- restored unified opening credits prepended when enabled,
- original English 5.1 as the first audio track, with only the first 0.5 seconds silenced,
- new restored Spanish 5.1 as the second audio track and the default audio,
- restored old Spanish `spa1` stereo using `vhs_broadcast_full`,
- original newer Spanish `spa2`, with only the first 0.5 seconds silenced,
- English and Spanish SRT subtitle tracks embedded directly in the review MKVs when expected SRT files are available,
- cover art attached directly to the review MKVs unless disabled,
- no old-bed/music extraction unless a later scene-specific manual decision proves it is worth the risk.

All final episode audio tracks must silence the first 0.5 seconds without trimming or shifting audio.

Openings are prepended by re-encoding only the restored unified opening clip to match each target video format, then concatenating with stream copy. Reusable prepared opening segments live under `work/review/episodes/_shared_openings/<review-name>/generation_<N>/`. Generation `1` uses the Macross newer Spanish narration insert, generation `2` uses the Masters insert, and generation `3` uses the New Generation insert.

Before end credits are appended, the pipeline can apply explicit episode adjustment segments. The first automatic adjustment checks the last episode-only frame and creates a short matching fade-to-black segment only when the last frame is not black enough. The default builds five fade frames plus one final full-black landing frame; the duplicate source-bright frame is dropped, so the first generated frame is already darker and the segment visibly reaches black before the end credits. This fixes cut-looking endings without re-encoding the episode video stream.

Episode-specific tail trims are also handled in this adjustment stage. `S01E36` currently trims a `30s` black tail from each episode-only mux with stream copy before end-credit concatenation, so the restored end credits start immediately after the episode instead of after the source tail.

Some newer Spanish dub tracks run a little past the restored episode-only video
and need a tiny episode-specific tail before the end credits. This is not a
global correction. Prepare it only for episodes that have been manually
identified:

```bash
robotech-ai spa2-tail-extension S01E14 \
  --reference-mkv "/path/to/reference/Robotech-S01E14.mkv" \
  --reference-episode-start 92.5925 \
  --reference-audio-stream 3 \
  --tail-start 1352.2475 \
  --duration 1.0 \
  --fade-out 0.15 \
  --run
```

This creates:

```text
work/ready_spa2_tail_extensions/S01E14/s01e14_spa2_tail_v001/spa2_tail.wav
work/ready_spa2_tail_extensions/S01E14/s01e14_spa2_tail_v001/tail.json
```

When `episode-final-build` sees exactly one tail manifest for that episode, it
inserts a short matching black/fade hold segment before the end credits. Tracks
1-2 are silence in that hold segment; both stereo Spanish tracks get the
rescued tail audio. Use `--no-spa2-tail-extensions` to disable this
episode-specific behavior for a test build.

The extracted tail stays at source level for the newer Spanish dub track. The
processed old-Spanish stereo track can be louder because of the VHS voice
restoration chain, so `spa2-tail-extension` auto-matches that destination level
when it can find the current processed stereo track under the review audio
folder. The resulting gain is saved in `tail.json` as
`old_spa1_tail_gain_db`. Override or guide this behavior with:

```bash
--old-spa1-tail-gain-db 6.0
--old-spa1-match-audio path/to/03_episode_spanish1_restored_old_stereo.ac3
--old-spa1-match-window 1.0
--old-spa1-max-gain-db 12
```

For S01E14 the approved tail is the first second of the reference newer Spanish
dub tail after trimming the first 39 ms, with a tiny fade-in to avoid a join
click, so the ready patch is exactly 1 second long. Quick review files are kept
at:

```text
work/review/S01E14_spa2_tail_extension_001/05_spa2_end_plus_new_1s_tail_source_gain.wav
work/review/S01E14_spa2_tail_extension_001/06_old_spa1_end_plus_new_1s_tail_auto_gain.wav
```

End credits are appended by re-encoding only the credit clip to match each target video format, then concatenating the episode segment, any needed adjustment segment, and credit segment with stream copy. Reusable prepared credit segments live under `work/review/episodes/_shared_end_credits/<review-name>/` and are shared across episodes with matching video format. The selected processed episode videos must not be re-encoded during audio muxing or credit append.

When opening and/or end credits are concatenated, `episode-final-build` also writes Matroska chapter metadata without re-encoding video streams. The base chapters are `Opening Credits`, `Episode`, and `End Credits`, with timestamps calculated from the actual prepared segment durations. Episodes with a configured next-episode summary start in `config/next_episode_summaries.json` split the episode chapter into `Episode` and `Next Episode Summary`, so viewers can skip the end hook before the end credits. S01E35 is pending and S01E36 has no summary marker.

S01E36 is a special case: recent releases omit its next-episode summary, so the
build inserts a prepared reconstructed summary segment from
`work/ready_episode_segments/S01E36/next_summary_v001/segment.json` before the
end credits. That segment carries its own four audio tracks and relative English
and Spanish SRTs; the builder shifts those subtitle cues to the actual inserted
timeline position for each final video. Normal S01E36 episode rebuilds do not
reconstruct, upscale, interpolate, or remix this summary again; they only pick
up the ready segment variants already present under
`work/ready_episode_segments/S01E36/next_summary_v001/`.

Reference Blu-ray MKVs can be scanned once to rescue their internal episode
chapter marks. The final build reads the local JSON, not the reference files:

```bash
robotech-ai reference-chapters \
  --reference-root "/path/to/reference/eps" \
  --out config/reference_episode_chapters.json \
  --run
```

The collected starts are episode-relative, so they work after our own prepared
opening credit segment is prepended. Disable them for a test build with
`--no-reference-episode-chapters`.

Next-episode narrator summary starts are saved in
`config/next_episode_summaries.json`. After selecting Qwen3-TTS takes, rebuild
the summary preview with `tts-summary-generate --assemble-only`. Use
`--balance-phrases --balance-max-gain-db N` only to match phrase loudness inside
the summary. Use `--summary-gain-db N` to raise or lower the whole assembled
summary after balancing. Per-phrase speed overrides can be added to
`selected_takes.json` under `"speed_percent": {"03": 102}` before rebuilding.

Current Spanish subtitle QA workbook:

```text
work/review/subtitle_language_review/spanish_language_review_pass5.xlsx
```

Reviewed fixes from pass2, pass3, pass4, and the first bracket-label pass were
applied with timestamped backups under `work/review/subtitles/backups/`.
Project-specific Robotech terms such as `Microniano`, `Microniana`,
`Micronianos`, and `Micronianas` are allowlisted and should not be treated as
Spanish subtitle errors. `spanish_language_review_pass5.xlsx` is the current
clean spell/quality baseline and has zero review rows.

The active subtitle files have been checked for bad `SDF-17`, `SDF-7`, and
`SDF-T` OCR errors; none remain in the active English or Spanish SRTs.

Bracketed speaker/action labels are reviewed separately because they often
contain names or closed-caption descriptions rather than normal prose. Generate
the focused workbook with:

```bash
python3 scripts/create_subtitle_bracket_review_workbook.py \
  --out work/review/subtitle_language_review/spanish_bracket_review_pass2.xlsx \
  --run
```

`spanish_bracket_review_pass2.xlsx` is the current clean bracket-label baseline
and has zero review rows.

The review-folder MKVs are the authored masters for checking and export:

```text
work/review/episodes/S01E##/<review-name>/video/*.mkv
```

When subtitle files exist in `work/review/subtitles/S01E##/`, `episode-final-build` embeds them as selectable SRT streams in those review MKVs. It also attaches cover art. If `Robotech/images/macross_thumb_asset.png` exists, that PNG is used for every Macross episode build; otherwise the build falls back to extracting a first-frame JPEG from each output. Players that ignore Matroska cover attachments may still generate their own thumbnails, but the file itself carries the intended cover image.

Metadata from `archive-metadata` should be used to preserve track language/title/disposition decisions as much as possible.

## Subtitles

English OCR SRT files and local-LLM Spanish translated SRT files are stored per episode:

```text
work/review/subtitles/S01E##/S01E##_english_clean.srt
work/review/subtitles/S01E##/S01E##_spanish_translated.srt
```

Current status: all 36 Macross episodes have both files, and English/Spanish cue counts match.

Manual edits to translated Spanish SRTs, such as adding localized episode-title subtitles, should be backed up outside the live subtitle tree before rerunning translation. The current manual backup is:

```text
work/review/subtitle_manual_backups/spanish_translated_20260628_225617/
```

`episode-final-build` discovers expected SRT files by name and embeds all that exist. Missing SRTs do not stop the build; they simply are not embedded. The current expected names are:

- `S01E##_english_clean.srt` -> embedded track title `English Subtitles`, sidecar `.eng.srt`
- `S01E##_spanish_translated.srt` -> embedded track title `Spanish Subtitles`, sidecar `.spa.srt`

The same discovery pattern is ready for later translated languages, for example `S01E##_french_translated.srt`, `S01E##_portuguese_translated.srt`, `S01E##_italian_translated.srt`, `S01E##_german_translated.srt`, and `S01E##_japanese_translated.srt`.

Prepare or refresh OCR English subtitles:

```bash
scripts/robotech-ai ocr-english-subtitles all --run
```

Translate Spanish subtitles with the local Hugging Face/Gemma path. Existing translated episodes are skipped unless `--overwrite` is passed:

```bash
scripts/robotech-ai translate-spanish-subtitles all \
  --provider hf \
  --model gemma_3_27b_it \
  --model-config config/llm_models/gemma_3_27b_it_subtitle_v001.json \
  --llm-python .venv-llm/bin/python \
  --offline \
  --chunk-size 10 \
  --retries 4 \
  --run
```

Run deterministic cleanup rules over translated Spanish SRTs after translation. This catches known local-LLM/OCR artifacts without rerunning the model, for example `hadían`/`hadian` -> `habían`:

```bash
scripts/repair_spanish_subtitles.py
scripts/repair_spanish_subtitles.py --run
```

## Episode Adjustments

Inspect needed per-episode adjustment segments without writing files:

```bash
scripts/robotech-ai episode-adjustments S01E01 --review-name final_mux_oc_ec_v1
```

Create needed adjustment segments:

```bash
scripts/robotech-ai episode-adjustments S01E01 --review-name final_mux_oc_ec_v1 --run
```

For S01E01, the current detector finds that only the `remaster_49fps` episode-only file needs an end fade-to-black segment. The normal AI Remaster and Remaster variants already end on true black.

Adjustment outputs and reports live here:

```text
work/review/episodes/S01E##/<review-name>/segments/adjustments/
```

Single Macross episode build:

```bash
scripts/robotech-ai episode-final-build S01E01 --review-name final_mux_oc_ec_v1 --generation 1 --run
```

By default this build embeds available SRT subtitles and attaches cover art. Use `--no-embed-subtitles` or `--no-cover-art` only for debugging those authoring stages.

The restored Spanish 5.1 mix has three separate level controls. These affect only
the new restored Spanish 5.1 track:

- `--spa51-preserved-channel-gain-db`: gain for preserved English-source FL/FR/LFE/SL/SR before joining the final 5.1.
- `--spa51-center-bed-gain-db`: gain for the de-voiced English center bed before Spanish dialogue is mixed into FC.
- `--spa51-dialogue-gain-db`: gain for restored Spanish dialogue before it is mixed into FC.

Current default restored Spanish 5.1 balance:

```text
--spa51-preserved-channel-gain-db 3
--spa51-center-bed-gain-db 3
--spa51-dialogue-gain-db 0
```

This matches the `01_bg_plus3_bed_plus3_voice_same` review candidate. For fastest
balance experiments, reuse the same review name so existing separator stems and
dialogue enhancement files stay cached. Changing these mix gains automatically
rebuilds only the restored Spanish 5.1 WAV/AC3. This overwrites the review outputs
for that review name, so use it when you are intentionally testing a replacement
mix:

```bash
scripts/robotech-ai episode-final-build S01E01 \
  --review-name final_mux_oc_ec_v1 \
  --generation 1 \
  --spa51-preserved-channel-gain-db 3 \
  --spa51-center-bed-gain-db 3 \
  --spa51-dialogue-gain-db -1 \
  --if-exists overwrite \
  --run
```

Using a new review name keeps outputs separate, but it also uses a new intermediate folder and may rerun the separator steps.

Automatic end fade detection is enabled by default for final builds with end credits. Use `--no-auto-end-fade` only when you want to disable just this adjustment.

Per-episode adjustment stages can be disabled at three levels:

- `--no-video-episode-adjustments`: skip video-only episode fixes, such as generated fade-to-black segments.
- `--no-audio-episode-adjustments`: skip audio-only episode fixes, such as future narrator/voice patch replacement lists.
- `--no-episode-adjustments`: master switch that skips both video and audio episode-specific fixes.

Episode-specific audio patches are applied after cached restoration stems are
available and before the final audio tracks are encoded. The original cached
dialogue/fullmix WAVs are not overwritten; patched WAVs are written with an
`audiofix_v###` suffix and used only as the effective inputs for the final mix.
This lets a manually replaced or externally repaired track be tested by disabling
the patch layer with `--no-audio-episode-adjustments`.

Current S01E01 audio patch:

```text
s01e01_narrator_colision_bridge_pre70_xfade4
35.207s-35.255s
method: rubberband bridge from the previous 70 ms of audio, with 4 ms crossfades at both joins
targets: restored Spanish dialogue only
not targeted: old Spanish restored stereo fullmix preservation track, spa2 newer Spanish dub
```

Review snippets:

```text
work/review/S01E01_audio_patch_colision_001/
work/review/S01E01_audio_patch_colision_002/
work/review/S01E01_audio_patch_colision_003/
```

The selected repair is the `09_pre70_default_crossfade_4ms_each.wav` candidate
from the archived `S01E01_audio_patch_colision_003` review set. The patch
output version is currently `audiofix_v006`, so older patch files are not reused
by mistake.

Current S01E03 title narrator replacement patch:

```text
s01e03_title_space_fold_voice_m59_edge80_m24_ex15_stable_seed23
43.080s-44.167s
method: external recovered title replacement, retimed to 1.087s
replacement: generated_audio/titlenarrator/s01e03_fixedtitle.wav
replacement gain: -5.94 dB
texture: first/last 80 ms of the removed Spanish slot at -24 dB, plus Stable Audio seed 23 inpaint texture at generated level
texture asset: generated_audio/titlenarrator/s01e03_title_texture_stable_seed23_as_generated.wav
insert fades: 35 ms logistic-style fade-in, 15 ms fade-out
targets: restored Spanish dialogue only
not targeted: old Spanish restored stereo fullmix preservation track, spa2 newer Spanish dub
```

Selected review candidates and comparison:

```text
work/review/S01E03_ai_inpaint_vs_approved_002/03_seed23_ai_overlay_as_generated_full_7s.wav
work/review/S01E03_ai_inpaint_vs_approved_002/34_seed11_ai_texture_plus6db_over_official_NONORMALIZE_full_7s.wav
```

Archived texture experiments for that same patch:

```text
work/review/delete/20260701_audio_review_tests/S01E03_title_narrator_highpass_001/
work/review/delete/20260701_audio_review_tests/S01E03_title_narrator_synthfill_002/
work/review/delete/20260701_audio_review_tests/S01E03_title_narrator_inpaint_001/
work/review/delete/20260701_audio_review_tests/S01E03_title_narrator_inpaint_smooth_002/
```

The selected production recipe is now the `voice_m59_edge80_m24_ex15_stable_seed23` patch above. The older exp180 and seed11+6 textures remain retained for provenance.

Neural inpainting test command:

```bash
scripts/robotech-ai ai-inpaint-stable \
  --seeds 11 23 \
  --steps 8 \
  --cfg-scale 1 \
  --sampler-type pingpong \
  --texture-gain-db -6 \
  --run
```

The command writes a plan manifest before running and outputs review WAVs under:

```text
work/review/S01E03_title_narrator_ai_inpaint_stable_001/
```

If it fails with a gated-model message, accept access for `stabilityai/stable-audio-3-medium` and log in again with `.venv-inpaint/bin/hf auth login`. Stable Audio Open access is not enough for Stable Audio 3 Medium.

Useful retained review/provenance folders:

```text
work/review/S01E03_title_narrator_startcut_001/
work/review/S01E03_title_narrator_endcut_002/
work/review/S01E03_title_narrator_mix_002/
work/review/S01E03_title_narrator_patch_002/
work/review/S01E03_title_narrator_patch_003/
```

Dead-end S01E03 narrator tests were moved, not deleted, to:

```text
work/review/delete/S01E03_title_narrator_obsolete/
```

For a subtle restored Spanish 5.1 balance test close to "background/base 10% up and restored dialogue 10% down", use approximately:

```bash
scripts/robotech-ai episode-final-build S01E01 \
  --review-name final_mux_oc_ec_v1 \
  --generation 1 \
  --spa51-preserved-channel-gain-db 0.83 \
  --spa51-center-bed-gain-db 0.83 \
  --spa51-dialogue-gain-db -0.92 \
  --if-exists overwrite \
  --run
```

Final builds reuse existing intermediate files by default. This is intentional: if a cleaned WAV, separator stem, enhanced dialogue WAV, restored 5.1 WAV/AC3, or preserved old-Spanish stereo AC3 already exists in the expected path, the build uses it instead of regenerating it. This makes last-minute manual file swaps practical and keeps test rebuilds fast.

If you replace one intermediate manually and want later files rebuilt from it, delete the downstream files you want regenerated before running the build. To force a full regeneration of cached intermediate audio/separation files, add:

```bash
--rebuild-intermediates
```

Batch Macross build without overwriting completed episodes:

```bash
scripts/robotech-ai episode-final-build all --review-name final_mux_oc_ec_v1 --generation 1 --if-exists skip --run
```

Use `--if-exists ask` for interactive overwrite/skip/abort prompts, or `--if-exists overwrite` for forced rebuilds.

## Qwen3-TTS Narrator Summary Regeneration

For the damaged "next episode" narrator summaries, see
[qwen3-tts-summary.md](qwen3-tts-summary.md). That workflow uses the local
Qwen3-TTS voice-clone checkout to generate multiple narrator takes per ASR
phrase, then creates an all-`v01` preview for review.

## Done Export

Export is intentionally copy-only. It does not remux, embed subtitles, attach thumbnails, or otherwise modify the review MKVs. The MKVs should already be final-authored by `episode-final-build`.

Dry-run one episode:

```bash
scripts/robotech-ai export-done S01E01 --review-name final_mux_oc_ec_v1
```

Copy the authored MKVs and matching external SRT sidecars:

```bash
scripts/robotech-ai export-done S01E01 --review-name final_mux_oc_ec_v1 --if-exists overwrite --run
```

The export folder receives three MKVs per episode plus sidecar SRTs for player compatibility:

```text
Robotech/done/Robotech/Season 1/
```
