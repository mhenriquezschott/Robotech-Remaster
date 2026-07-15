# Audio Repair Tool Status

The repair tool is a purpose-built desktop workbench for repeated Robotech audio patch work. It is intentionally smaller than a DAW and is meant to export reproducible repair recipes for the batch pipeline.

## Install

Use a dedicated GUI environment:

```bash
python3 -m venv .venv-repair
source .venv-repair/bin/activate
python -m pip install -e .
python -m pip install -r requirements-repair-tool.txt
```

## Launch

Basic launch:

```bash
robotech-repair-tool
```

Open the included S01E03 MVP project:

```bash
robotech-repair-tool --project work/repair_projects/S01E03_title_space_fold_ready_patch_center.repair.json
```

Apply exported repair recipes during a final episode build:

```bash
robotech-ai episode-final-build S01E03 \
  --review-name final_mux_oc_ec_v1 \
  --repair-recipe work/repair_projects/S01E03_title_space_fold_mvp.recipe.json \
  --run
```

Batch builds can auto-load per-episode recipes from `work/repair_projects/final_build`:

```bash
robotech-ai episode-final-build all \
  --review-name final_mux_oc_ec_v1 \
  --repair-recipe-dir work/repair_projects/final_build \
  --run
```

Recipe filenames in that folder should start with the episode id, for example `S01E03_title_narrator.recipe.json`.

Hot-swappable ready clip patches live under:

```text
work/ready_audio_patches/<episode>/<patch_id>/
  replacement.wav
  patch.json
```

The repair tool's `Export Ready Patch` button writes that folder format. The final build pipeline discovers it automatically unless `--no-ready-patches` is used. A ready patch with the same `patch_id` as a built-in code patch replaces the built-in patch, which allows last-minute swaps without editing Python.
Ready patches are direct replacements: `replacement.wav` is already the finished audio for the target span. The pipeline does not rebuild the fix from UI clip lanes; it only replaces `start_seconds` -> `end_seconds` in the restored Spanish dialogue track with that WAV. The old Spanish stereo/fullmix preservation track is enhanced as a whole but does not receive these per-episode dialogue fixes. The recipe export is the separate reproducible path when we want to rebuild a fix from UI settings.
By default `Export Ready Patch` exports the full visible work window, not just the selected cut. This preserves texture tails and clips that extend before or after the cut selection. Enable `Ready patch cut only` only when the desired final patch is exactly the selected cut span.
`Export Work WAV` and `Export Ready Patch` use deterministic project paths; they do not ask for an output folder. The status line is selectable, so the exported path can be copied from the bottom of the UI. For S01E03 title repair, `Export Ready Patch` writes:

```text
work/ready_audio_patches/S01E03/s01e03_title_space_fold_voice_m59_edge80_m24_ex15_stable_seed23/
  replacement.wav
  patch.json
```

S01E01 narrator `coAlision` micro-fix project:

```bash
robotech-repair-tool --project work/repair_projects/S01E01_narrator_colision_ready_patch.repair.json
```

This opens the restored old Spanish dialogue around `35.207s-35.255s` with no clips loaded. Start with `Rubberband bridge selected area`, `Edge src = 0.070s`, and `Fade in/Fade out = 0.004s`; this matches the old approved built-in repair style. If needed, also compare `Blend pre/post edges across selected area` and `Interpolate tiny gap with ambience`. The project uses patch id `s01e01_narrator_colision_bridge_pre70_xfade4`, so the exported ready patch replaces the old built-in bridge patch.

S01E03 title narrator test launch:

```bash
robotech-repair-tool \
  --episode S01E03 \
  --title "Space Fold" \
  --main-track work/final_episode_mux/S01E03/final_mux_oc_ec_v1/S01E03_spa1_dialogue_broadcast_strong.wav \
  --bed-track 'work/final_episode_mux/S01E03/final_mux_oc_ec_v1/eng_center_devoice/S01E03_eng_center_(Instrumental)_melband_roformer_instvoc_duality_v1.wav' \
  --video-track 'work/review/episodes/S01E03/final_mux_oc_ec_v1/video/episode_only/Robotech - S01E03 - Space Fold_RemasterBRestoredAudio_episode_only.mkv' \
  --reference work/review/S01E03_official_fix_shortlist_001/01_previous_official_exp180_reference_full_7s.wav \
  --marker 43.5 \
  --window 10 \
  --cut-start 43.080 \
  --cut-end 44.167 \
  --clip generated_audio/titlenarrator/s01e03_fixedtitle.wav \
  --clip generated_audio/titlenarrator/s01e03_title_texture_stable_seed23_as_generated.wav
```

The command above uses the full S01E03 Spanish dialogue track as the main top timeline. The cut times are absolute episode times.
The ready-patch-center project now uses clip banks:

- `Clip Bank 01`: `generated_audio/titlenarrator/s01e03_fixedtitle.wav`, preserving the approved replacement voice settings.
- `Clip Bank 02`: `work/review/repair_tool/texture_variants/S01E03_seed23_phase_fill_002_obvious/04_stress_extra_filled_texture.wav`, preserving the approved texture settings.
- `Clip Bank 03`: muted by default, with restored narrator phrase alternates from `generated_audio/mainnarrator/mainnarrator_yrescataunalindachica.wav` and `generated_audio/mainnarrator/mainnarrator_suviejoamigoroyfocker.wav`.

The current S01E03 ready-patch project uses the available de-voiced English center stem as the bed preview. The earlier full 5.1 downmix preview was removed from the UI because it was slower and did not help this repair workflow. The final-build pipeline still creates a reusable lossless full de-voiced English 5.1 bed at `work/final_episode_mux/<episode>/<review>/<episode>_eng_devoiced_bed_51_lossless.wav` for diagnostics and future tooling.

S01E18 original Spanish rescue project:

```bash
robotech-repair-tool --project work/repair_projects/S01E18_original_spanish_rescue_001.repair.json
```

Clip Bank 01 now uses `generated_audio/episode_fixes/S01E18/original_spanish_rescue_context_001/S01E18_original_spanish_rescue_context_32s_36s_broadcast_strong.wav`.
This is cropped from an 80-second context pass of `Robotech/proc/MacrossSagaTV/ep18/Robotech - 1x18 - Original.wav` covering `00:02:00` -> `00:03:20`, separated with `melband_roformer_instvoc_duality_v1.ckpt`, then enhanced with `broadcast_strong`.
The full enhanced context file to audit is `work/repair_sources/S01E18/original_spanish_rescue_context_001/enhance/S01E18_original_spanish_rescue_context_001/S01E18_original_spa_rescue_context_02m00_03m20_source__Vocals__melband_roformer_instvoc_duality_v1_broadcast_strong.wav`.
The active project cut is now absolute episode `00:02:32` -> `00:02:36`, matching local `00:00:32` -> `00:00:36` inside the enhanced context. The older direct 5-second extraction remains at `generated_audio/episode_fixes/S01E18/original_spanish_rescue_001/S01E18_original_spanish_rescue_02m30_02m35_broadcast_strong.wav` only as a rejected/older reference and is not loaded by the active S01E18 project.

S01E15 Claudia repair project:

```bash
robotech-repair-tool --project work/repair_projects/S01E15_claudia_5min_001.repair.json
```

This opens the S01E15 restored old Spanish dialogue around the first Claudia fix with the standard de-voiced English center bed and episode-only video reference. Clip Bank 01 loads `generated_audio/claudia/claudia_ah01.wav`; Clip Banks 02 and 03 are empty.

The nearby second Claudia fix is split into a separate project because the tool does not yet support multiple independent repair regions in one project:

```bash
robotech-repair-tool --project work/repair_projects/S01E15_claudia_5min_uh_002.repair.json
```

This second project is shifted `+4s` from the first and loads `generated_audio/claudia/claudia_uh01.wav` in Clip Bank 01.

Important UI model:

- `Main Episode Timeline` must be the full episode Spanish working track, normally the extracted/improved old Spanish dialogue or fullmix track.
- Clicking the top timeline is only navigation. It moves the marker and refreshes the lower work clip.
- `Window` is the total work-clip duration centered on the marker. Example: marker `35s`, window `10s` gives a work clip from `30s` to `40s`.
- Changing `Window` immediately recrops the Work Clip Preview while preserving the current absolute episode cut points.
- `Lock top marker` prevents accidental top-timeline clicks from changing the work clip once the repair area is selected.
- The cut/replacement is evaluated in the lower `Work Clip Preview`.
- The visible cut controls are local to the work clip. The project still stores absolute episode seconds internally so pipeline export stays precise.
- `Play Main From Marker` plays the full loaded main audio starting at the top marker.
- `Play Work Clip` plays the visible work preview exactly as shown, with no pending action secretly applied.
- `Play selected area` toggles play/pause for only the selected work region.
- `Play Work Clip` starts from the gray work start marker.
- The green marker is only the live playback position.
- `Stop` resets the gray work start marker and green playback marker to the start of the work clip.
- `Play Work Mix` adds enabled clip lanes on top of the visible work preview.
- Clip lane timeline bars represent the whole work clip. The draggable block is the clip itself, so moving it left/right changes where that clip enters the work mix.
- Clip lane offsets can be negative when the clip is positioned before the selected area.
- Clip lane `Length` shows the current audible clip duration after the visible `Speed` setting.
- `Reset clip` restores that lane to the settings it had when the current project was loaded.
- `Undo` / `Redo` are available from buttons and via `Ctrl+Z`, `Ctrl+Y`, or `Ctrl+Shift+Z`. The first pass snapshots the project state plus the visible work preview before edit actions.
- `Reload files` resets marker, work selection, and lane settings to the project state as it was loaded, then rereads the main/reference audio from disk.
- Work clip selection is shown with distinct start/end boundary colors and a shaded selected region.
- Current click model:
  - Left-click on the work waveform sets the cut start.
  - Right-click on the work waveform sets the cut end.
  - `Arrow keys move START` / `Arrow keys move END` only choose which boundary left/right arrow keys nudge.
- `Focus main`, `Focus work`, and `Focus reference` change keyboard focus without moving any marker.
- Double-click and drag on the work waveform creates a start/end selection directly.
- Left/right arrow keys nudge the active boundary by the current snap amount.
- If a clip lane is focused, left/right arrow keys nudge that lane offset instead.
- Spacebar plays/pauses the focused surface: main, work, de-voiced bed, reference, or clip lane.
- The right panel has a de-voiced English 5.1 bed preview above Reference. It follows the same work-window and selected region as the main work clip, but it does not define cut start/end.
- Bed preview can be played alone or as `Bed + Work Mix`, which overlays the visible work mix and enabled clip lanes over the bed preview.
- Bed actions use the work-clip selected area and support the same local actions except external voice removal: keep/gain, silence, delete, fade out/in, stretch pre-cut edge, and repeat pre-cut edge.
- A muted video reference panel below Reference can load an episode video. When `Follow audio` is enabled, it seeks and plays along with main/work/bed/reference playback using the corresponding episode time.
- Drag/drop replaces the whole target path. It should not insert the dropped path into the middle of the existing path text.
- Main waveform drag/drop loads/replaces the full main episode audio.
- Clip lane drag/drop works on the lane body and lane timeline bar, and loads/replaces that lane file.
- Work Clip Preview intentionally does not accept dropped files because it is derived from the main episode audio and current marker/window.
- De-voiced bed and Reference waveforms also accept dropped files as replacements for their corresponding path fields.
- The bed preview uses the loaded file as-is. For the normal repair project this is the center instrumental stem.
- The bed status label reports the loaded source channel count.
- Reference playback has its own loop checkbox, enabled by default.
- `Action` currently supports:
  - `Silence selected area (keep timing)`: replace the selected work-clip region with silence, keeping the clip duration.
  - `Delete selected area (shorten clip)`: remove that region and pull the later audio earlier.
  - `Fade out/in selected area`: fade out from the selection start, silence the middle, and fade in at the selection end.
  - `Rubberband bridge selected area`: reproduce the old FFmpeg/Rubber Band bridge method, stretching the pre-cut edge across the selected gap with short crossfades.
  - `Stretch pre-cut edge across selected area`: copy the small region immediately before the cut start and time-stretch it to fill the selected area. This is meant for tiny letter/syllable fixes, not long gaps.
  - `Repeat pre-cut edge across selected area`: copy the small region immediately before the cut start and tile/repeat it until the selected area is filled.
  - `Blend pre/post edges across selected area`: stretch the audio immediately before and after the selected area and crossblend them across the gap. This is better for tiny letter removals where using only the left edge clicks or buzzes.
  - `Interpolate tiny gap with ambience`: draw a click-free bridge from the sample before the gap to the sample after the gap, with a very low amount of local ambience mixed in. This is a subtle option for 10-60 ms fixes when silence jumps and stretch/repeat artifacts.
  - `Keep selected area (gain only / no cut)`: keep the selected audio in place; useful for selected-area gain changes or auditioning overlay clips.
  - `Remove voice from selected area (audio-separator)`: exports the selected local work region for reference, runs `audio-separator` on the full visible work context with `melband_roformer_instvoc_duality_v1.ckpt` and `--single_stem Instrumental`, then crops the matching selected time range from the de-voiced context and replaces only that selected preview area.
- `Selected gain` is independent from `Action`. It can raise or lower the selected region by itself, or combine with fade/silence/keep operations.
- `Voice keep` is only for the voice-removal action. It blends that percentage of the original selected audio back into the de-voiced replacement. `0%` is strict voice removal; higher values preserve more original background/texture but may also preserve some unwanted voice.
- `Edge src` controls how much audio immediately before the cut start is used by the stretch/repeat pre-cut edge actions.
- For gain-only edits, leave `Action` on `Keep selected area (gain only / no cut)`, set `Selected gain`, then press `Apply to preview`.
- For combined edits, choose the operation first, for example `Fade out/in selected area`, set `Selected gain`, then press `Apply to preview`; gain is applied before the fade/silence operation renders.
- `Cut length` shows the current selected duration in local work-clip seconds.
- `Lock selection` is enabled by default. While enabled, work-waveform clicks do not move cut start/end; left-click moves the gray work start marker instead.
- `Play selected area` plays only the selected work region from the visible work preview. It does not apply a pending action by itself.
- `Apply to preview` redraws the work waveform with the selected action applied; it does not overwrite the source audio.
- `Play Work Clip` plays the full visible work preview exactly as shown. If no action has been applied, this is the original work clip.
- `Play Work Mix` adds enabled clip lanes on top of the visible work preview. It does not secretly apply a pending selected action.
- Clip lanes are now mute-only in the mix. The old `Solo` option has been removed because it duplicated the practical use of mute/unmute during this repair workflow.
- Each clip lane now has a clip-bank combobox. `Open` or drag/drop adds a new audio file to that lane's list instead of replacing the whole lane conceptually. Selecting another item restores that item’s own path, gain, speed, offset, fade, mute, and lock settings.
- Work mix playback and exports include every unmuted item in each clip bank. The selected combo item is only the item currently being edited.
- Clip bank panels use generic titles (`Clip Bank 01`, `Clip Bank 02`, `Clip Bank 03`); semantic names belong in the individual clip labels inside the combobox.
- Old single-clip projects are still compatible: a lane with only `path` and no `clip_items` is automatically backfilled as a one-item clip bank on load.
- Project save/load, recipe export, and ready-patch manifests serialize the full clip-bank state. The currently selected clip item is mirrored to the lane-level fields so the existing renderer and final-build importer still use the visible lane settings.
- Voice-removal action outputs are saved under `work/review/repair_tool/voice_extract/<episode>/...` with `selected_region_input.wav`, `separator_context_input.wav`, separator output WAVs, `audio_separator.log`, and a `manifest.json`.
- Voice-removal manifests include the selected-gain setting and `Voice keep` percentage used for that preview.
- The separator context is padded to at least 20 seconds when needed because the current RoFormer model can fail internally on short 1-10 second clips while still returning a successful process exit code.
- `Export Preview WAV` exports the visible work preview plus enabled clip lanes, so applied preview actions such as voice removal are included.
- `Export Ready Patch` exports the visible work preview plus enabled clip lanes to `work/ready_audio_patches/.../replacement.wav`. Lane gain, speed, fades, and timeline placement are baked into the WAV exactly like `Play Work Mix`.
- `Ready patch cut only` is off by default, so the full work window is exported and `patch.json` stores `export_scope: full_work_window`. When enabled, only the selected cut span is exported with `export_scope: cut_only`.
- The current focus label shows whether arrow keys affect the main marker or the work selection boundary.
- `Pause` toggles approximate resume from the current playhead. Exact audio-device cursor tracking is still planned.

A headless render smoke test has already produced:

```text
work/review/repair_tool/S01E03_title_space_fold_mvp_preview.wav
work/review/repair_tool/S01E03_title_space_fold_mvp_seed23_preview.wav
```

## Ready In First Iteration

- Launchable PySide6 desktop app.
- Main audio path loader with file picker and drag/drop field.
- De-voiced bed audio path loader with file picker and drag/drop field.
- Reference audio path loader.
- File path fields refresh automatically after browse or drag/drop.
- Main waveform, de-voiced bed waveform, reference waveform, and clip lanes accept file drops.
- Dropping a file on a path field or supported waveform/lane replaces the target file path cleanly.
- Top main waveform overview.
- Top main waveform is a navigation surface in minutes: clicking it moves the marker and refreshes the work clip unless `Lock top marker` is enabled.
- Work-window waveform preview around the current marker.
- Work waveform clicks set either local cut start or local cut end, depending on the active cut target button.
- Double-click/drag selection on the work waveform.
- Work selection is drawn directly on the work waveform.
- Marker snapping by configured milliseconds.
- Default snap is 10 ms.
- Top marker lock and loop checkboxes are enabled by default.
- Three replacement/texture lanes.
- Clip-bank comboboxes for each lane, including backwards-compatible one-item lanes for older repair projects.
- Per-lane file path, gain, speed, clip-length readout, offset, timeline bar, mute, lock, auto-align, fit-length, play/stop, and reset controls.
- Lane timeline bars represent the full work-window duration; the draggable block represents the clip duration after the visible speed setting.
- Work-mix playback now honors the same lane timeline placement that is visible on screen, including negative offsets before the selected area.
- `Auto align` aligns to the selected area and enables the lane lock when a real selection exists. If no useful selection exists, it aligns to the work-window start and leaves the lane unlocked.
- On project load, clips with files that are aligned to an existing selected area are locked by default; clips are left unlocked when there is no useful selected area.
- Per-lane `Fade in` and `Fade out` apply to clip playback and work-mix playback.
- De-voiced bed preview panel with bed-only playback, bed-plus-work-mix playback, shared work selection, local bed actions, loop control, and bed marker lock.
- Muted video reference panel with project `video_track`, file picker, and follow-audio seek/play behavior.
- `Fit length` is the auto-length control. It calculates the speed needed to match the selected area and writes that value into the visible `Speed` field.
- `Auto align` places a lane at the current selected area. If no useful selection exists, it aligns to the work-window start.
- Individual lane playback has play/pause and stop controls. Gain and speed apply immediately during lane audition and work-mix preview; there is no hidden stretch unless the visible speed value changes.
- Qt standard media icons on transport buttons.
- Main/work/reference playhead marker while playback is active.
- Active main/work/reference/lane surfaces receive a visual focus highlight.
- First click on an unfocused waveform only focuses it; the next click edits/moves its marker.
- Waveform focus now also takes real keyboard focus, so arrow keys move the active marker/selection instead of walking across buttons.
- Per-scope loop checkboxes started for main/work playback; full cursor-accurate loop/resume behavior still needs refinement.
- Preview mix playback through `sounddevice`.
- Reference playback.
- Cached full-track audio/waveform overview so moving the marker does not reload/recompute the whole episode every time.
- Export current work-window mix as 24-bit WAV.
- Save/load `.repair.json` project files.
- Save/load `.repair.json` project files, including main/bed/video/reference sources, work selection parameters, edge settings, clip lane parameters, and new action history entries.
- Export `.recipe.json` files with sources, full work-window metadata, selected actions, lanes, and declared render targets.
- `Save Project` overwrites the currently opened `.repair.json` file. `Save Project As` creates a copy under a new path. A `.repair.json` file is the editable GUI project; a `.recipe.json` file is the pipeline instruction export and is not the same thing.
- Saving and deterministic exports persist the selected clip-bank item state: path, gain, speed, offset, fades, mute, and lock. `Export Ready Patch` and `Export Recipe` also quietly save the open `.repair.json` project so the rendered patch and project state do not drift.
- `episode-final-build --repair-recipe` imports deterministic repair recipes and applies them to the enhanced Spanish dialogue before the restored Spanish 5.1 mix is created.
- `episode-final-build --repair-recipe-dir` can auto-load per-episode recipe files for batch use.
- The importer supports deterministic work actions and unmuted lanes. Interactive `remove_voice` actions are intentionally rejected for now unless replaced by exported audio/lane material.
- `episode-final-build` also discovers ready WAV+JSON patches from `work/ready_audio_patches/<episode>/...` by default. Use `--ready-patch-dir` to point elsewhere or `--no-ready-patches` to disable them.
- Ready patch render steps are refreshed whenever a ready patch is active, even without `--rebuild-intermediates`; expensive extraction/separation/enhancement intermediates are reused.
- Ready patch JSON uses absolute episode `start_seconds` / `end_seconds`; `replacement.wav` is the finished insert clip for that slot. This is the preferred hot-swap format when the fix was finalized in the GUI or an external editor.
- For ready patches exported by the GUI, `replacement_is_final` is `true` and no extra gain/fade/texture processing is written by default. Optional gain/fade fields are still accepted by the low-level importer for manual advanced patches, but they are not part of the normal ready-fix flow.
- Qwen3-TTS next-episode narrator summaries can also be promoted to the same ready-patch format. The approved S01E01 summary patch is `work/ready_audio_patches/S01E01/s01e01_next_episode_summary_tts_v001/` and targets only `dialogue`.
- Export visible work-window mix WAV.
- Export visible bed-plus-work-window mix WAV.
- Export a full-length patched main WAV by replacing the current work-window span with the visible work mix.

## On Test

- Timeline mouse positioning and snap behavior.
- Playback device behavior on Ubuntu/PipeWire/PulseAudio.
- Preview mix correctness for fit-to-cut inserts.
- Whether the top timeline should use total window duration or a before/after radius. Current behavior treats `Window` as total visible duration centered on the marker.
- Drag/drop reliability across desktop file managers.
- Usability of local 7-second context workflow versus full episode workflow.

## Work Selection Flow

There are three equivalent ways to make a work selection:

1. Use direct clicks:
   - Left-click once on the work waveform where the selected region should start.
   - Right-click once on the work waveform where the selected region should end.

2. Use direct drag selection:
   - Double-click on the work waveform where the selection starts.
   - Keep holding and drag to the end.
   - Release the mouse button.

3. Use keyboard nudging:
   - Click `Arrow keys move START` or `Arrow keys move END`.
   - Press left/right arrows to move that boundary by the current snap amount.

After selecting, use `Play selected area` to audition only that region. Choose an `Action` and press `Apply to preview` when you want the visible work clip to change. `Play Work Clip` then plays that visible result, and `Play Work Mix` adds enabled lanes on top of it.

## Planned Next

- Real cut-marker handles drawn directly on the work waveform.
- Keyboard shortcuts for nudging marker/cut/selected lane by snap amount.
- Lane waveform previews.
- True visual clip blocks drawn over the work clip, with drag handles instead of only offset sliders.
- A/B transport buttons for current mix versus reference.
- Candidate folder browser for quickly swapping generated tests.
- Export full-track patched WAV, not only preview window.
- Import/export recipe files directly into `episode-final-build`.
- CLI importer for repair recipes that applies recorded work/bed actions to full episode intermediates before final 5.1 mixing.
- Support multiple repair regions in one project.
- Unload/remove a clip from a clip bank directly from the GUI.
- Better resampling/stretch via FFmpeg/rubberband for final-quality exports.
- Optional AI/inpaint launcher integration once the workflow proves useful.
- 5.1-aware preview/mix panel for center/dialogue versus bed channels.
- Background worker/progress UI for voice extraction. The first implementation is blocking and intentionally simple.
- Optional helper to add the extracted `Vocals` diagnostic into a clip lane when useful.

## Current Limitation

The first iteration is for workflow validation. It can load, preview, mix, and export short contexts, but the official batch pipeline still uses the existing CLI patch definitions. Recipe import into the final build command is the next bridge to add once the GUI behavior feels right.

Undo/redo covers marker moves, selection changes, preview actions, bed actions, path changes, clip align/fit/reset, and visible work/bed preview state. Fine-grained undo for every spinbox/edit-field parameter tweak is still being hardened; save/export does preserve the current parameter values.
