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
2. Run soundtrack alignment/subtraction. This is now the preferred first serious
   test for recovering SFX and narrator pieces:

   ```bash
   .venv-separation/bin/python scripts/opening_music_subtraction.py
   ```

   Outputs:

   ```text
   work/review/opening_audio_rebuild_001/music_subtraction_001/
   ```

   Review `*_residue_music_subtracted_norm.wav`,
   `*_residue_presence_sfx_2500_12000.wav`, and
   `*_residue_voice_band_180_4200.wav`. These are the files most likely to
   contain recoverable non-music effects and the Spanish “Robotech” narrator.
3. Listen to the original Spanish “Robotech” voice in:
   - `sources/05_asset_track02_spa1_original_stereo.wav`
   - `sources/08_tv_copy_ep01_intro_stereo.wav`
   - separator `Vocals` output when available.
4. Check whether FFmpeg mid/side gives useful effects or only phase trash:
   - `ffmpeg_stems/*_mid_mono.wav`
   - `ffmpeg_stems/*_side_mono.wav`
5. Check whether `presence_sfx_2500_9000` contains recoverable transition SFX
   without too much vocal/music bleed.
6. If the narration voice is recoverable, process it separately using the same
   conservative chain that worked for episode dialogue:
   light cleanup, melband v1 extraction, broadcast-style voice strengthening,
   and manual/repair-tool alignment if needed.
7. Mix the recovered narration/effects over the official soundtrack Main Title.

Current review pack:

```text
work/review/opening_audio_rebuild_review_002/
```

Suggested review order:

- `01_spa1_aligned_soundtrack.wav`: soundtrack conformed to the current Spanish opening.
- `02_spa1_music_subtracted_residue_norm.wav`: current opening minus soundtrack.
- `03_spa1_residue_presence_sfx.wav`: high-frequency SFX/presence residue.
- `04_spa1_residue_voice_band.wav`: quick voice-band residue.
- `05_spa1_melband_v1_vocals.wav`: known-good MelBand v1 vocals from current Spanish opening.
- `06_spa1_melband_v1_vocals_broadcast_strong.wav`: same voice extraction with the episode-dialogue enhancement chain.
- `07_tvcopy_music_subtracted_residue_norm.wav`: TV-copy opening minus soundtrack.
- `08_tvcopy_melband_v1_vocals.wav`: MelBand v1 vocals from old TV-copy opening.
- `09_tvcopy_melband_v1_vocals_broadcast_strong.wav`: TV-copy vocals with `broadcast_strong`.
- `10_*` through `12_*`: short Apollo tests on residues/presence candidates.
- `13_soundtrack_main_title_opening_length.wav`: clean soundtrack baseline.

Current listening conclusion:

- The soundtrack subtraction and SFX/presence residue files are diagnostics, not
  clean extractions. They still sound too much like the original mix or like
  frequency cuts, so they should not be treated as usable isolated effects yet.
- A dedicated calibration pass confirmed this numerically: even in the first
  `0-5s` low-effect window, the best EQ/delay subtraction only reduced the
  residual by about `0.9 dB` versus the target. A useful null should be much
  lower. This means the official soundtrack waveform is not close enough to the
  opening music bed for phase subtraction to isolate SFX.
- The useful outputs so far are the MelBand v1 voice extractions:
  `05_spa1_melband_v1_vocals.wav`,
  `06_spa1_melband_v1_vocals_broadcast_strong.wav`,
  `08_tvcopy_melband_v1_vocals.wav`, and
  `09_tvcopy_melband_v1_vocals_broadcast_strong.wav`.
- Short Apollo tests were not enough to assess whether effects survive in the
  important sections, so full-length chunked Apollo candidates were added under
  `work/review/opening_audio_rebuild_review_002/apollo_full/`.

Full-length Apollo review files:

- `apollo_full/02_apollo_full_spa1_residue_48k.wav`
- `apollo_full/03_apollo_full_spa1_presence_sfx_48k.wav`
- `apollo_full/07_apollo_full_tvcopy_residue_48k.wav`

Subtraction calibration command:

```bash
.venv-separation/bin/python scripts/opening_subtraction_calibration.py
```

Main report:

`work/review/opening_audio_subtraction_calibration_001/REPORT.md`

The best current numbers still show poor cancellation:

- `quiet_intro_000_005`: only about `-0.86 dB` residual reduction.
- `effects_mid_023_025`: about `+0.81 dB` relative to target.
- `effects_late_055_063`: about `+1.12 dB` relative to target.

That is not enough separation to trust as SFX recovery.

Demucs comparison:

```bash
.venv-separation/bin/demucs -n htdemucs_6s --int24 -d cuda \
  -o work/review/opening_audio_demucs_001 \
  work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav
```

Review files:

- `work/review/opening_audio_demucs_001/review_48k/01_demucs6_other_48k.wav`
- `work/review/opening_audio_demucs_001/review_48k/02_demucs6_vocals_48k.wav`
- `work/review/opening_audio_demucs_001/review_48k/windows/`

The `other` stem is the only plausible Demucs SFX candidate. Treat it as a
listening test, not a proven extraction.

User review notes from the first Demucs pass:

- `review_48k/01_demucs6_other_48k.wav` sounds mostly like the music bed with
  reduced effects, especially high-frequency music content and little bass. It
  is useful as a candidate to subtract from the original opening, not as the
  final SFX stem by itself.
- `review_48k/02_demucs6_vocals_48k.wav` and
  `review_48k/03_demucs2_vocals_48k.wav` contain the Spanish “Robotech” voice
  around `24s-27s`. Outside that voice section, they also retain some effects,
  including laser-gun sounds, so they may be useful for manual SFX remixing
  after carefully cutting or cleaning the voice region.

Demucs stem algebra / reverse tests:

```bash
.venv-separation/bin/python scripts/opening_demucs_stem_algebra.py
```

Outputs:

```text
work/review/opening_audio_demucs_001/stem_algebra_001/
  01_original_minus_demucs_other_gain100_norm.wav
  02_original_minus_demucs_other_gain125_norm.wav
  03_original_minus_demucs_other_gain150_norm.wav
  04_sum_all_except_other_norm.wav
  05_vocals_plus_drums_norm.wav
  09_laser_candidate_drums_guitar_piano_norm.wav
  windows/
```

Review first:

- `01_original_minus_demucs_other_gain100_norm.wav`: direct “reverse of
  Demucs other” test. If `other` is mostly music, this should leave more
  voice/SFX.
- `02_original_minus_demucs_other_gain125_norm.wav`: same, with stronger
  cancellation of the `other` stem.
- `04_sum_all_except_other_norm.wav`: Demucs reconstruction using everything
  except the music-ish `other` stem.
- `05_vocals_plus_drums_norm.wav`: keeps the likely voice/SFX carriers while
  avoiding the `other` music stem.
- `09_laser_candidate_drums_guitar_piano_norm.wav`: targeted laser/effect
  candidate from the non-vocal, non-other stems.
- `windows/*laser_mid_023_027*` and `windows/*effects_late_055_063*`: fast
  review clips for the two sections where effects are most obvious.

If these still contain too much music, the next realistic model family is
text/query-based audio separation, not more classic music stem separation.
Prompts to test should be specific, for example:

- `laser gun sound effects, no music, no speech`
- `spaceship and laser blast sound effects, no music, no narration`
- `Spanish narrator voice saying Robotech, no music`

Most promising text/query separation tools to evaluate next:

- AudioSep / AudioSep-DP style language-queried separation. This is the closest
  match to “extract the laser sounds” because the model is conditioned by text,
  not by broad music stems.
- CLAPSep-style text/multimodal target separation if an installable checkpoint
  is easier to run locally.
- PromptSep-style generative sound separation/removal if the public code and
  checkpoints are available and stable enough.
- MMAudioSep only later, if video-conditioned extraction becomes useful; it is
  more complex than needed for the opening-audio-only test.

Do not spend more time trying plain phase subtraction as the primary path unless
we find a soundtrack source that matches the exact opening mix. The calibration
pass already showed the current CD soundtrack and opening bed do not null well
enough for clean SFX recovery.

AudioSep first-pass setup and run:

```bash
bash scripts/setup_audio_restoration_tools.sh clone
bash scripts/setup_audio_restoration_tools.sh install-audiosep
bash scripts/setup_audio_restoration_tools.sh download-audiosep-checkpoints
```

Run the default prompt pack against the current Spanish opening:

```bash
.venv-audio-audiosep/bin/python scripts/run_audiosep_prompts.py \
  --input work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav \
  --out-dir work/review/opening_audio_audiosep_001 \
  --device cuda \
  --use-chunk
```

Default prompts include:

- `motorcycle engine sound effect, no music, no speech`
- `motorcycle engine revving, no music, no narrator`
- `laser gun sound effects, no music, no speech`
- `spaceship and laser blast sound effects, no music, no narration`
- `Spanish narrator voice saying Robotech, no music`

The runner writes full review files plus short window clips:

```text
work/review/opening_audio_audiosep_001/
  *_audiosep_48k_stereo.wav
  *_audiosep_48k_stereo_norm.wav
  windows/
```

AudioSep outputs are mono at 32 kHz internally. The script resamples to 48 kHz
and duplicates to stereo only for review/mixing convenience. Treat them as raw
salvage candidates, not finished stereo stems.

Current AudioSep review sets:

```text
work/review/opening_audio_audiosep_001/
  First focused prompt pass on current Spanish opening.

work/review/opening_audio_audiosep_002_prompt_grid_current/
  Broader prompt grid on current Spanish opening.

work/review/opening_audio_audiosep_003_prompt_grid_tvcopy/
  Same broader prompt grid on old TV-copy opening.
```

The second/third grids intentionally include prompts that allow speech:

- `all anime sound effects, laser guns and motorcycle engine, no music`
- `all sound effects from an old anime opening, no background music`
- `motorcycle engine and laser gun sound effects, no music`
- `laser guns and motorcycle engine with speech allowed, no music`
- `all non-music sounds, speech and sound effects, no music`

The working assumption after first review is that “no voice” prompts are less
important than “no music” prompts. If AudioSep keeps the narrator but removes
most of the music, that may still be useful: the voice region can be cut or
replaced with a cleaner voice extraction, while the remaining motorcycle/laser
fragments can feed a manually rebuilt SFX layer.

Old TV-copy “Robotech” voice review pack:

```text
work/review/opening_robotech_voice_tvcopy_001/
```

Files to compare:

- `01_tvcopy_source_23p8_27p6.wav`: raw TV-copy window.
- `02_tvcopy_melband_v1_vocals_23p8_27p6.wav`: MelBand v1 vocal extraction.
- `03_tvcopy_melband_v1_broadcast_strong_23p8_27p6.wav`: MelBand v1 plus the
  same broadcast-style strengthening used for old Spanish dialogue.
- `04_tvcopy_melband_v1_presence_clear_23p8_27p6.wav`: alternate clearer/presence
  voice enhancement.
- `05_tvcopy_audiosep_voice_plus_lasers_23p8_27p6.wav`: AudioSep voice prompt,
  useful for comparing voice plus recovered laser texture.
- `06_tvcopy_audiosep_non_music_voice_sfx_23p8_27p6.wav`: AudioSep non-music
  prompt; may retain SFX around the narrator.
- `07_tvcopy_broadcast_strong_light_denoise_23p8_27p6.wav`: light denoise on
  the broadcast-strong voice.
- `08_tvcopy_broadcast_strong_voice_focus_23p8_27p6.wav`: more focused voice
  band plus dynamic normalization.
- `09_tvcopy_broadcast_plus_audiosep_texture_23p8_27p6.wav`: broadcast-strong
  voice with a small amount of AudioSep texture underneath.

AudioSep-DP / TQ-SED is the next heavier text-query model to test. Its official
repo is `soft/ai_audio_tools/src/TQ-SED`; pretrained AudioSep-DP checkpoints are
published on Zenodo as `resunet_with_dprnn_16k.zip` and
`resunet_with_dprnn_32k.zip`, about `1.5-1.6 GB` each. This needs a dedicated
wrapper around the LASS separation code rather than the SED training scripts.

Relationship between the names:

- `AudioSep-DP`: the language-query source separator we want for extraction.
- `TQ-SED`: the sound-event detection pipeline built around AudioSep-DP
  separations. Useful later for locating events, but not required for the first
  SFX extraction pass.

AudioSep-DP setup:

```bash
bash scripts/setup_audio_restoration_tools.sh install-audiosep-dp
bash scripts/setup_audio_restoration_tools.sh download-audiosep-dp-checkpoints
```

The first command creates:

```text
.venv-audio-audiosepdp/
```

The second downloads the 32 kHz checkpoint from Zenodo into:

```text
soft/ai_audio_tools/src/TQ-SED/LASS_codes/checkpoints/
```

Run a first prompt grid:

```bash
.venv-audio-audiosepdp/bin/python scripts/run_audiosep_dp_prompts.py \
  --input work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav \
  --out-dir work/review/opening_audio_audiosep_dp_001 \
  --device cuda
```

Review:

```text
work/review/opening_audio_audiosep_dp_001/
work/review/opening_audio_audiosep_dp_001/windows/
```

This runner uses the AudioSep-DP separator directly with text prompts, resamples
review files to 48 kHz stereo, and cuts the same voice/laser/SFX windows as the
AudioSep grid.

First completed review grids:

```text
work/review/opening_audio_audiosep_dp_001/
work/review/opening_audio_audiosep_dp_tvcopy_001/
```

The first uses the current restored opening-credit Spanish original stereo
track. The second uses the old TV-copy intro source, so it is the one to compare
against the TV-copy `Robotech` narrator voice pack.

SAM-Audio is the next model family to test after AudioSep-DP. Unlike the music
stem models, it can be given both a text prompt and a time span, which is useful
for the opening because the obvious review regions are already known:

- narrator/laser overlap: about `23.8s-27.6s`
- early laser overlap: about `23s-27s`
- later effects/motorcycle region: about `55s-63s`

SAM-Audio setup:

```bash
bash scripts/setup_audio_restoration_tools.sh clone
bash scripts/setup_audio_restoration_tools.sh install-sam-audio
```

Requirements:

- Python `3.11`
- CUDA PyTorch installed in `.venv-audio-samaudio`
- Hugging Face access to the requested SAM-Audio checkpoint, with local auth
  already configured, for example by running `hf auth login`

First smoke test, anchored to the narrator/laser section. Use
`--process-start/--process-end`; a full opening-length run with
`sam-audio-base-tv` OOMed on the RTX 5090 because the audio codec tried to
encode the whole minute at once.

```bash
.venv-audio-samaudio/bin/python scripts/run_sam_audio_prompts.py \
  --input work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav \
  --out-dir work/review/opening_audio_sam_audio_smoke_001 \
  --model facebook/sam-audio-base-tv \
  --prompt "motorcycle engine sound effect" \
  --prompt "laser gun sound effects" \
  --prompt "spanish narrator voice saying robotech" \
  --process-start 20 \
  --process-end 30 \
  --anchor 23.8 27.6 \
  --device cuda
```

Current-opening later SFX grid:

```bash
.venv-audio-samaudio/bin/python scripts/run_sam_audio_prompts.py \
  --input work/review/opening_audio_rebuild_001/sources/05_asset_track02_spa1_original_stereo.wav \
  --out-dir work/review/opening_audio_sam_audio_late_sfx_001 \
  --model facebook/sam-audio-base-tv \
  --prompt "motorcycle engine sound effect" \
  --prompt "motorcycle engine revving" \
  --prompt "laser gun sound effects" \
  --prompt "spaceship and laser blast sound effects" \
  --prompt "all non-music sound effects" \
  --process-start 52 \
  --process-end 66 \
  --anchor 55 63 \
  --device cuda
```

Old TV-copy narrator/laser grid:

```bash
.venv-audio-samaudio/bin/python scripts/run_sam_audio_prompts.py \
  --input work/review/opening_audio_rebuild_001/sources/08_tv_copy_ep01_intro_stereo.wav \
  --out-dir work/review/opening_audio_sam_audio_tvcopy_voice_001 \
  --model facebook/sam-audio-base-tv \
  --prompt "spanish narrator voice saying robotech" \
  --prompt "laser gun sound effects" \
  --prompt "all non-music sound effects" \
  --process-start 20 \
  --process-end 30 \
  --anchor 23.8 27.6 \
  --device cuda
```

The runner writes both target and residual files:

```text
work/review/opening_audio_sam_audio_late_sfx_001/
  *_sam_target_48k_stereo.wav
  *_sam_target_48k_stereo_norm.wav
  *_sam_residual_48k_stereo.wav
  *_sam_residual_48k_stereo_norm.wav
  windows/
```

Use `target` for the requested prompt and `residual` as the model's “everything
else” estimate. The normalized files are only for fast listening; use raw-level
files when deciding whether something can be mixed back into the rebuilt
opening.

Completed SAM-Audio review packs:

```text
work/review/opening_audio_sam_audio_smoke_001/
work/review/opening_audio_sam_audio_late_sfx_001/
work/review/opening_audio_sam_audio_tvcopy_voice_001/
```

Run these serially. Two simultaneous `sam-audio-base-tv` runs exhausted the GPU
when both models were loaded.

Compatibility notes:

- AudioSep expects `checkpoint/audiosep_base_4M_steps.ckpt` and
  `checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt` inside the AudioSep
  repo. The download command above fetches the same files used by AudioSep's
  Colab notebook.
- New PyTorch versions default to `torch.load(..., weights_only=True)`, which
  breaks these older trusted checkpoints. `scripts/run_audiosep_prompts.py`
  forces the legacy checkpoint loading mode inside that runner only.

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

Apollo setup and wrapper:

```bash
bash scripts/setup_audio_restoration_tools.sh install-apollo
source .venv-audio-apollo/bin/activate
```

Apollo's upstream inference script expects a local checkpoint path despite using
`JusperLee/Apollo` in the README. Use the local wrapper instead:

```bash
.venv-audio-apollo/bin/python scripts/run_apollo_restore.py \
  --input work/review/opening_audio_rebuild_001/apollo_tests/input/spa1_residue_12s_44100.wav \
  --output work/review/opening_audio_rebuild_001/apollo_tests/spa1_residue_12s_apollo.wav \
  --device cuda
```

Apollo expects 44.1 kHz input; resample candidates before running it.

Full opening-length files can OOM if Apollo receives the whole waveform at
once. Use chunked inference with overlap/crossfade:

```bash
.venv-audio-apollo/bin/python scripts/run_apollo_restore.py \
  --input work/review/opening_audio_rebuild_review_002/apollo_full_44k/02_spa1_music_subtracted_residue_norm_44k.wav \
  --output work/review/opening_audio_rebuild_review_002/apollo_full/02_apollo_full_spa1_residue_44k.wav \
  --device cuda \
  --chunk-seconds 8 \
  --overlap-seconds 1
```

Convert the Apollo result back to 48 kHz for side-by-side review with the rest
of the project audio:

```bash
ffmpeg -hide_banner -y \
  -i work/review/opening_audio_rebuild_review_002/apollo_full/02_apollo_full_spa1_residue_44k.wav \
  -ar 48000 \
  -c:a pcm_s24le \
  work/review/opening_audio_rebuild_review_002/apollo_full/02_apollo_full_spa1_residue_48k.wav
```

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
