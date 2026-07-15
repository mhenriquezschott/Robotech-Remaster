# Robotech Audio Repair Tool Proposal

This proposal describes a purpose-built desktop tool for repeated Robotech audio repair tasks: segment replacement, muting, gain tests, texture overlays, edge smoothing, and exporting reproducible patch recipes back into the episode build pipeline.

## Recommendation

Use **Python + PySide6/Qt** for the desktop application.

Reasons:

- Qt has stronger cross-platform audio/editor UI building blocks than GTK for this kind of timeline-heavy tool.
- PySide6 gives us native menus, drag/drop, keyboard shortcuts, dockable panels, and custom timeline widgets.
- We can keep the same Python/FFmpeg/audio stack already used by the restoration pipeline.
- The GUI can generate the same structured patch recipes consumed by `robotech-ai`, instead of becoming a separate one-off editor.

Recommended first-pass libraries:

- `PySide6`: GUI framework.
- `pyqtgraph`: fast waveform/overview/timeline rendering.
- `soundfile`: precise WAV read/write.
- `numpy`: audio buffers and gain/mix math.
- `scipy`: optional resampling/analysis helpers.
- `sounddevice`: playback from selected buffers.
- `ffmpeg`: final render/export, AC3 encode, format conversion, filters.
- Existing project helpers in `robotech_ai.media` and `robotech_ai.audio`.

## Product Shape

The app is not a full DAW. It is a repair workbench for short, repeatable fixes.

Main layout:

1. **Main Episode Timeline**
   - Full episode/audio track overview across the top.
   - Time ruler in seconds/minutes.
   - Marker cursor for the current repair location.
   - Optional marker lock.
   - Configurable work-window duration, for example 7s or 10s.

2. **Work Area**
   - Left side below the main timeline.
   - Shows the current work clip cut from the main track around the marker.
   - Has editable start/end cut markers.
   - Supports millisecond snap: 1ms, 5ms, 10ms, 50ms, 100ms.
   - Arrow keys move selected marker/clip by the snap amount.
   - Supports clip mute, solo, lock, gain, fade-in/out, fit-to-gap speed stretch, and edge smoothing.

3. **Replacement/Texture Lanes**
   - Up to three short clips above the base work clip.
   - Each lane can hold a voice insert, generated phrase, texture, music/noise bed, or recovered fragment.
   - Each lane has offset, gain, mute, solo, lock, and fit-to-gap controls.
   - Drag/drop or file picker can load/replace a lane.
   - Right-click can swap among generated test candidates from a folder.

4. **Reference Area**
   - Right side below the main timeline.
   - Holds original/reference/current-approved repair context.
   - Allows quick A/B playback against the current work mix.
   - Can hold more than one reference, for example original and last-approved.

## Workflow

1. Open project or launch from command line with episode/patch files.
2. Load main track and optional reference files.
3. Place marker on the top timeline.
4. App extracts a work window around the marker.
5. User adjusts cut start/end with snap.
6. User loads/replaces/moves up to three patch clips.
7. User previews:
   - base only,
   - selected inserts only,
   - repair section only,
   - full 7s/10s context,
   - reference A/B.
8. User exports:
   - patch fragment,
   - patched full track,
   - structured patch recipe for the pipeline.

## Project File

Use a human-readable JSON project file:

```json
{
  "schema_version": 1,
  "episode": "S01E03",
  "title": "Space Fold",
  "main_track": "work/.../S01E03_dialogue.wav",
  "sample_rate": 48000,
  "channels": 2,
  "repairs": [
    {
      "id": "s01e03_title_space_fold",
      "marker_seconds": 43.5,
      "work_window_seconds": 7.0,
      "cut_start_seconds": 43.080,
      "cut_end_seconds": 44.167,
      "snap_ms": 10,
      "base_action": "replace",
      "lanes": [
        {
          "role": "replacement_voice",
          "path": "generated_audio/titlenarrator/s01e03_fixedtitle.wav",
          "offset_seconds": 0.0,
          "gain_db": -5.94,
          "fit_to_cut": true,
          "locked": false,
          "muted": false
        },
        {
          "role": "texture",
          "path": "generated_audio/titlenarrator/s01e03_title_texture_stable_seed11_plus6.wav",
          "offset_seconds": 0.0,
          "gain_db": 0.0,
          "fit_to_cut": false,
          "locked": false,
          "muted": false
        }
      ],
      "edge": {
        "fade_in_seconds": 0.035,
        "fade_out_seconds": 0.015,
        "curve": "losi"
      }
    }
  ]
}
```

This project file becomes the bridge between GUI work and automated episode builds.

## Code Architecture

Proposed package:

```text
src/robotech_ai_repair/
  __init__.py
  app.py
  main_window.py
  models.py
  project.py
  audio_engine.py
  timeline.py
  waveform_cache.py
  playback.py
  render.py
  recipe_export.py
  widgets/
    timeline_overview.py
    clip_lane.py
    transport.py
    inspector.py
```

Responsibilities:

- `models.py`: dataclasses for project, repair, lane, marker, render settings.
- `project.py`: load/save project JSON, path normalization, schema migration.
- `audio_engine.py`: read audio, crop work windows, apply gain/fades/stretch previews.
- `waveform_cache.py`: generate lightweight overview data so long tracks draw fast.
- `playback.py`: sounddevice playback of selected mix/reference.
- `render.py`: deterministic render of fragments/full-track WAV using Python or FFmpeg.
- `recipe_export.py`: convert GUI repair objects into `EpisodeAudioPatch`-compatible JSON/YAML/CLI recipes.
- `timeline.py` and `widgets/`: Qt graphics/timeline interaction, markers, snap, drag/drop.

## Pipeline Integration

The GUI should export recipes, not just audio.

Initial export targets:

- `work/repair_projects/S01E03_title_space_fold.repair.json`
- `work/repair_projects/S01E03_title_space_fold.recipe.json`
- optional rendered preview files under `work/review/...`

Later, `robotech-ai episode-final-build` can accept:

```bash
scripts/robotech-ai episode-final-build S01E03 \
  --repair-recipe work/repair_projects/S01E03_title_space_fold.recipe.json \
  --run
```

For now, the GUI recipe can map directly into the existing hard-coded `EPISODE_AUDIO_PATCHES`. Once stable, we can move episode-specific patches out of Python and into recipe JSON files.

## MVP Scope

Build the first version around the S01E03 title narrator repair.

MVP features:

- Load a main WAV track and one reference WAV.
- Display top overview timeline with marker.
- Extract 7s/10s work window around marker.
- Edit cut start/end with snap and arrow keys.
- Load up to three replacement/texture lanes.
- Gain, mute, lock, fit-to-cut, fade in/out.
- Play work mix and reference.
- Export full 7s preview WAV.
- Save/load `.repair.json`.
- Export recipe JSON compatible with the pipeline.

Not in MVP:

- Full multi-track 5.1 editing UI.
- Spectrogram editing.
- Real-time AI model execution from the GUI.
- AC3 final muxing; keep that in CLI pipeline.

## Development Plan

1. Create minimal PySide6 app shell and project model.
2. Implement audio loading, waveform overview cache, and playback.
3. Implement main timeline marker and work-window extraction.
4. Add base clip with start/end cut markers and snap movement.
5. Add three replacement lanes with offset/gain/mute/lock/fit-to-gap.
6. Add render/export of preview WAV.
7. Add project save/load.
8. Add recipe export and connect it to the existing CLI patch pipeline.

This keeps the tool useful early while avoiding a giant DAW-style rewrite.
