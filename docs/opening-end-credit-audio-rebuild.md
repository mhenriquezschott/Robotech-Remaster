# Opening And End-Credit Audio Rebuild

This is the evaluation plan for rebuilding the opening and end-credit audio
from cleaner soundtrack sources while preserving the original narration and
effects when possible.

## Current Sources

Opening assets:

- `Robotech/oc-ec/oc/assets/intromasterAI48fps1440x1080cropallac3_v56.mkv`
- `Robotech/oc-ec/oc/assets/track01eng51.ac3`
- `Robotech/oc-ec/oc/assets/track02spa1ori.ac3`

End-credit assets:

- `Robotech/oc-ec/ec/outromasterAI24fps1440x1080cropwbanner_v1.mp4`
- `Robotech/oc-ec/ec/outromasterAI60fps1440x1080cropwbanner_v1.mp4`

Soundtrack candidates:

- `Robotech/Robotech Perfect Soundtrack/Robotech Perfect Collection 1.1/01 Main Title.mp3`
- `Robotech/Robotech Perfect Soundtrack/Robotech Perfect Collection 1.1/42 End Title.mp3`

Old TV-copy opening source:

- `/mnt/usb-Seagate_Expansion_HDD_00000000NT17VSPP-0:0-part2/Multimedia/Videos/Series/Robotech [Esp]_TvQuality/Season 1/*.avi`

## Existing 49fps Credit Handling

The current episode builder does not use RIFE or any neural interpolation for
opening/end-credit shared segments. It re-encodes only the opening/end-credit
segments to match each target episode video and then concatenates with stream
copy.

For opening/end-credit video preparation, the current code uses:

```text
fps=<target_rate>,scale=<target_width>:<target_height>,format=yuv420p
```

That means:

- Opening 49fps variants are frame-rate conformed from the existing 47.952fps
  `intromasterAI48fps...` source.
- AI-remaster 50fps opening variants are frame-rate conformed from the same
  47.952fps source.
- End-credit 49fps variants are frame-rate conformed from the existing 59.94fps
  `outromasterAI60fps...` source.
- End-credit 24fps variants use the 23.976fps end-credit source.

This is not FFmpeg motion interpolation such as `minterpolate`, but it is still
FFmpeg frame-rate conversion. If we rebuild the credits visually, we should make
new RIFE/vs-mlrt or RIFE-NCNN interpolated 49fps/50fps sources instead of asking
the episode builder to invent motion.

## First Evaluation Harness

Prepare and run the lightweight opening audio source/stem extraction:

```bash
scripts/prepare_opening_audio_rebuild_eval.py --separator-commands --run
```

Outputs:

```text
work/review/opening_audio_rebuild_001/
  sources/
  ffmpeg_stems/
  separator/
  commands/run_separator_opening_candidates.sh
  manifest.json
  README.md
```

The first pass creates:

- current opening embedded audio streams as WAV,
- separate opening `track01eng51` and `track02spa1ori` WAVs,
- soundtrack `01 Main Title` full and opening-length WAVs,
- old TV-copy episode 1 intro audio,
- FFmpeg mid/side, dialogue-enhance, light denoise, presence-band, and voice-band candidates.

The generated separator script queues melband v1 `Vocals` and `Instrumental`
passes for the key opening sources. This is intentionally separate because
`audio-separator` currently reports CPU mode in this environment.

## Practical Test Order

1. Listen to `sources/07_soundtrack_main_title_opening_length.wav` against the
   current opening music to confirm timing/arrangement differences.
2. Listen to the original Spanish “Robotech” voice in:
   - `sources/05_asset_track02_spa1_original_stereo.wav`
   - `sources/08_tv_copy_ep01_intro_stereo.wav`
   - separator `Vocals` output when available.
3. Check whether FFmpeg mid/side gives useful effects or only phase trash:
   - `ffmpeg_stems/*_mid_mono.wav`
   - `ffmpeg_stems/*_side_mono.wav`
4. Check whether `presence_sfx_2500_9000` contains recoverable transition SFX
   without too much vocal/music bleed.
5. If the narration voice is recoverable, process it separately using the same
   conservative chain that worked for episode dialogue:
   light cleanup, melband v1 extraction, broadcast-style voice strengthening,
   and manual/repair-tool alignment if needed.
6. Mix the recovered narration/effects over the official soundtrack Main Title.

## AI Restoration Candidates

Use AI restoration only on stems where it has a plausible job:

- Full stereo music from the CD soundtrack: no AI restoration by default; keep it
  as the clean reference unless there is a specific defect.
- Current/TV degraded full mix: Apollo is a good first model to test because it
  targets compressed music restoration.
- De-voiced center or isolated narration/effects: FlashSR/AudioSR can be tested
  lightly for bandwidth extension, but overuse can make voice/effects synthetic.
- Local dropouts or missing transitions: A2SB is conceptually the best match
  because it supports inpainting, but public code/checkpoints were not available
  from NVIDIA's project page at the time of this note.

Suggested first model order:

1. Apollo on degraded full-mix and de-voiced music/effects candidates.
2. AudioSR on short voice/effects candidates if Apollo is too music-specific.
3. FlashSR only if runnable code/checkpoints are available locally later.
4. A2SB when NVIDIA releases code/checkpoints, or if a usable implementation
   appears.

## FFmpeg Baseline Tools

The local FFmpeg build has these useful filters:

- `dialoguenhance`
- `surround`
- `stereotools`
- `afftdn`
- `anequalizer`
- `highpass`
- `lowpass`

These are useful as transparent baselines and diagnostics. They are not expected
to reconstruct missing codec data, but they help reveal whether there is
recoverable voice/effects information before running heavier AI tools.

