# S01E36 Next Summary Reconstruction

This package reconstructs the missing "next episode" summary for `S01E36` using the best available Southern Cross/Masters scenes plus the old English DVD summary audio.

## Working Folder

`work/review/S01E36_next_summary_reconstruction_001`

## Video Sources

Source folder:

`/mnt/usb-Seagate_Expansion_HDD_00000000NT17VSPP-0:0-part2/Multimedia/Videos/Series/Robotech_Bluraywithoriginalspanishdubengandnewspanish/Bluray/Season02-Masters/eps`

Scene cuts are interpreted as `minutes:seconds:frame` at `24000/1001`.

| Scene | Source | Timecode | Notes |
| --- | --- | --- | --- |
| 01 | `Robotech-S02E01.mkv` | `03:51:10` -> `03:53:13` | Southern Cross scene |
| 02 | `Robotech-S02E11.mkv` | `14:09:04` -> `14:10:16` | Southern Cross scene |
| 03 | `Robotech-S02E01.mkv` | `17:58:18` -> `18:04:13` | Southern Cross scene |
| 04 | `Robotech-S02E11.mkv` | `14:00:22` -> `14:06:17` | first and last frame cloned twice |

Generated video folders:

- `source_copy_1920/`: no-reencode stream-copy reference cuts at original 1920x1080. These may be keyframe-imprecise.
- `crop1440_lossless/`: frame-accurate 1440x1080 working clips, center-cropped from 1920x1080 using x264 CRF 0. Cropping requires re-encoding; CRF 0 is used to avoid added compression loss before AI processing.

Command:

```bash
python3 scripts/prepare_s01e36_next_summary.py --run
```

## DVD Audio Seed

DVD summary/full episode sources:

- `Robotech/proc/MacrossSagaDVD/ep36/robotech-s01e36_nextsummary01.mp4`
- `Robotech/proc/MacrossSagaDVD/ep36/Robotech - 1x36 - To the Stars.mkv`

Generated audio:

- `audio/S01E36_next_summary_dvd_eng51_original.ac3`: original DVD 5.1 summary audio.
- `audio/S01E36_next_summary_dvd_eng51_original.wav`: decoded 5.1 WAV.
- `audio/S01E36_next_summary_dvd_center.wav`: extracted center channel.
- `audio/S01E36_next_summary_dvd_stereo_reference.wav`: stereo reference from the short summary clip.
- `audio/S01E36_next_summary_dvd_center_english_voice_v1.wav`: English voice stem from center.
- `audio/S01E36_next_summary_dvd_center_devoiced_bed_v1.wav`: de-voiced center bed from center.

ASR/subtitle seed:

- `subtitles/S01E36_next_summary_english_phrases_relative.srt`
- `subtitles/S01E36_next_summary_english_words_relative.tsv`
- `phrase_plan_english_seed.json`
- `generated_audio/next_episode_summary/S01E36/summary_v001/phrase_plan_english_seed.json`

The SRT and phrase plan times are relative to the reconstructed summary segment, not the original episode timeline.

## Assembled Review Package

The first assembled review package lives here:

`work/review/S01E36_next_summary_reconstruction_001/assembled_v001`

Command:

```bash
python3 scripts/assemble_s01e36_next_summary.py --run
```

Generated outputs:

- `assembled_v001/S01E36_next_summary_reconstructed_1440x1080_24fps_lossless.mkv`: reconstructed video only, 1440x1080, `24000/1001`, x264 CRF 0.
- `assembled_v001/S01E36_next_summary_english_dvd_51.ac3`: English DVD summary audio, padded to the assembled summary duration, AC3 5.1.
- `assembled_v001/S01E36_next_summary_english_devoiced_bed_51.ac3`: English summary 5.1 bed with the center channel replaced by the de-voiced center bed.
- `assembled_v001/S01E36_next_summary_reconstructed_review.mkv`: review mux with the reconstructed video plus both 5.1 AC3 tracks.
- `assembled_v001/assembly_manifest.json`: exact fade/padding settings and output paths.

Verified shape:

- Video frame count: `413`
- Frame rate: `24000/1001`
- Duration: about `17.25s`
- Video starts with `20` black frames, then scene 01 fades in across `10` frames.
- Scene 04 fades out over `10` frames, starting `8` frames before the scene image ends, with `2` black frames completing the fade, then `20` additional black frames.

Spanish review files:

- `subtitles/S01E36_next_summary_spanish_phrases_relative.srt`
- `generated_audio/next_episode_summary/S01E36/summary_v001/phrase_plan.json`

The English phrase plan is retained only as an ASR/timing seed and English reference. The Spanish `phrase_plan.json` is the file intended for Qwen3-TTS narrator reconstruction.

Generate 10 Spanish narrator takes for every phrase:

```bash
robotech-ai tts-summary-generate S01E36 \
  --summary-id summary_v001 \
  --takes 10 \
  --model-size 1.7B \
  --language Spanish \
  --chunk-size 200 \
  --chunk-gap 0 \
  --device cuda \
  --overwrite \
  --run
```

This uses `generated_audio/next_episode_summary/S01E36/summary_v001/phrase_plan.json`.
Save any manual punctuation/pronunciation edits before running. `--overwrite` is
needed because the summary folder already contains seed files; it does not create
a new phrase plan.

After choosing takes, edit:

`generated_audio/next_episode_summary/S01E36/summary_v001/selected_takes.json`

Then assemble a preview, for example:

```bash
robotech-ai tts-summary-generate S01E36 \
  --summary-id summary_v001 \
  --assemble-only \
  --avoid-overlap \
  --min-gap 0.10 \
  --balance-phrases \
  --balance-max-gain-db 5 \
  --summary-gain-db 0 \
  --run
```

Adjust `--summary-gain-db` only after phrase-to-phrase balance sounds right.

For S01E36, note that `phrase_plan.json` currently has a content range ending
at `00:00:15.634`, while the reconstructed summary video/audio package is about
`17.25s` because of the added black/fade frames. If a selected Spanish summary
preview is around `16s`, it can still be valid for the reconstructed summary
package even if the phrase-plan range warning mentions about `15s`.

## AI Video Test Runner

The reconstructed video should remain the clean source. AI video tests go under:

`work/review/S01E36_next_summary_reconstruction_001/video_ai_tests`

Check available local tools:

```bash
python3 scripts/run_s01e36_next_summary_video_tests.py check
```

Clone the Linux/Python source repos:

```bash
bash scripts/setup_s01e36_video_ai_tools.sh clone
```

The `check` and `clone` commands above do not need an activated Python env. Run
them from the repo root in a normal shell. The separate venvs are only needed
for installing/running APISR, AnimeSR, and RIFE.

Install and test APISR:

```bash
bash scripts/setup_s01e36_video_ai_tools.sh install-apisr
bash scripts/setup_s01e36_video_ai_tools.sh weights-apisr

python3 scripts/run_s01e36_next_summary_video_tests.py apisr \
  --label apisr_rrdb2x_downscale_test_001 \
  --overwrite \
  --run
```

The APISR command must run where CUDA is visible. In Codex, that means escalated
execution; in your terminal, just run it from the normal repo shell.

Current APISR result:

- `video_ai_tests/apisr_rrdb2x_downscale_test_001/apisr_rrdb2x_downscale_test_001.mkv`: video-only APISR 2x RRDB result, downscaled back to 1440x1080.
- `video_ai_tests/apisr_rrdb2x_downscale_test_001/apisr_rrdb2x_downscale_test_001_with_audio.mkv`: same video plus the existing English DVD 5.1 and English de-voiced 5.1 bed tracks for review.

Install and test Real-CUGAN NCNN/Vulkan:

```bash
bash scripts/setup_s01e36_video_ai_tools.sh install-realcugan-ncnn

python3 scripts/run_s01e36_next_summary_video_tests.py realcugan \
  --label realcugan_ncnn_conservative_s2_n-1_001 \
  --overwrite \
  --run
```

The default Real-CUGAN command uses the Linux `realcugan-ncnn-vulkan` release
and bundled `models-se` weights. It extracts PNG frames from the approved
summary source, runs Real-CUGAN at 2x, downscales back to `1440x1080`, and muxes
the review audio. It uses `--denoise -1` by default, which is the conservative
mode and is the safest first pass after APISR looked too artificial.

Current Real-CUGAN review results:

- `video_ai_tests/realcugan_ncnn_conservative_s2_n-1_001/realcugan_ncnn_conservative_s2_n-1_001_with_audio.mkv`: conservative mode, safest/least synthetic.
- `video_ai_tests/realcugan_ncnn_nodenoise_s2_n0_001/realcugan_ncnn_nodenoise_s2_n0_001_with_audio.mkv`: no-denoise mode, useful if conservative is too soft.
- `video_ai_tests/realcugan_ncnn_mild_denoise_s2_n1_001/realcugan_ncnn_mild_denoise_s2_n1_001_with_audio.mkv`: mild denoise, useful if source trash is still too visible.

Approved current video source:

- `video_ai_tests/realcugan_ncnn_nodenoise_s2_n0_001/realcugan_ncnn_nodenoise_s2_n0_001_with_audio.mkv`

Install and run RIFE NCNN/Vulkan interpolation after choosing the base image
restoration. This is the approved interpolation path; do not use ffmpeg
`minterpolate` for this summary.

```bash
bash scripts/setup_s01e36_video_ai_tools.sh install-rife-ncnn

python3 scripts/run_s01e36_next_summary_video_tests.py rife-ncnn \
  --label rife_original_49fps_001 \
  --rife-ncnn-model soft/ai_video_tools/bin/rife-ncnn-vulkan/rife-v4.6 \
  --rife-ncnn-fps 48000/1001 \
  --overwrite \
  --run

python3 scripts/run_s01e36_next_summary_video_tests.py rife-ncnn \
  --label rife_realcugan_nodenoise_50fps_001 \
  --input-video work/review/S01E36_next_summary_reconstruction_001/video_ai_tests/realcugan_ncnn_nodenoise_s2_n0_001/realcugan_ncnn_nodenoise_s2_n0_001.mkv \
  --review-audio-source work/review/S01E36_next_summary_reconstruction_001/video_ai_tests/realcugan_ncnn_nodenoise_s2_n0_001/realcugan_ncnn_nodenoise_s2_n0_001_with_audio.mkv \
  --rife-ncnn-model soft/ai_video_tools/bin/rife-ncnn-vulkan/rife-v4.6 \
  --rife-ncnn-fps 50 \
  --rife-ncnn-num-frames 862 \
  --overwrite \
  --run
```

The 50 fps AI remaster source needs `--rife-ncnn-num-frames 862`; default 2x
RIFE produces 826 frames, which is correct for `48000/1001` but too short at
`50/1`.

## Alternate Southern Cross Source Comparison

A later comparison package uses the Southern Cross source files from:

`Robotech/The Super Dimension Cavalry Southern Cross (1984)/Season 1`

This package is for review only. It does not replace the approved S01E36 ready
segment and is not consumed by `episode-final-build`.

Working folder:

`work/review/S01E36_next_summary_southern_cross_source_001`

Scene cuts are interpreted as `minutes:seconds:frame` at `30000/1001`. This is
intentional because the supplied timecodes include frame values such as `28`,
and some source files were exported as 29.97 fps or adaptive-rate derivatives.
Where fixed-rate derivatives exist, use them:

| Scene | Source | Timecode | Notes |
| --- | --- | --- | --- |
| 01 | `Super Dimension Cavalry Southern Cross - 1x13 - Triple Mirror-30fps.mp4` | `02:26:17` -> `02:28:20` | fixed 29.97 fps derivative |
| 02 | `Super Dimension Cavalry Southern Cross - 1x10 - Outsider-24fps.mp4` | `15:47:04` -> `15:48:19` | fixed 23.976 fps derivative |
| 03 | `Super Dimension Cavalry Southern Cross - 1x01 - Prisoner.mkv` | `24:04:21` -> `24:08:28` | original fixed 23.976 fps source |
| 04 | `Super Dimension Cavalry Southern Cross - 1x12 - Lost Memory-24fps.mp4` | `13:17:27` -> `13:19:11` | fixed 23.976 fps derivative |
| 05 | `Super Dimension Cavalry Southern Cross - 1x10 - Outsider-24fps.mp4` | `15:38:27` -> `15:41:21` | fixed 23.976 fps derivative |

Build the comparison package:

```bash
python3 scripts/prepare_s01e36_next_summary_southern_cross_source.py --run
```

Outputs to review:

- `assembled/S01E36_next_summary_southern_cross_source_review_24fps.mkv`
- `assembled/S01E36_next_summary_southern_cross_source_review_24fps_padded_to_ready_audio.mkv`
- `video_ai_tests/rife_southern_cross_padded_49fps_001/rife_southern_cross_padded_49fps_001_with_audio.mkv`
- `video_ai_tests/rife_southern_cross_padded_50fps_001/rife_southern_cross_padded_50fps_001_with_audio.mkv`

Important duration note:

- The five raw Southern Cross scene ranges plus the same fade framing produce
  about `14.056s` of video.
- The approved current ready summary audio package is about `17.248s`.
- The `*_padded_to_ready_audio*` files add black video at the end so the audio
  package can be reviewed without cutting off. This padding is a diagnostic
  accommodation, not proof that the source timecodes fully cover the old
  summary.

The interpolated comparison files were generated with RIFE NCNN/Vulkan, not
ffmpeg motion interpolation. Codex needs escalated execution for this command
so Vulkan sees the RTX GPU; inside the sandbox it falls back to `llvmpipe`.

```bash
python3 scripts/run_s01e36_next_summary_video_tests.py rife-ncnn \
  --source work/review/S01E36_next_summary_southern_cross_source_001/assembled/S01E36_next_summary_southern_cross_source_1440x1080_24fps_padded_to_ready_audio_lossless.mkv \
  --input-video work/review/S01E36_next_summary_southern_cross_source_001/assembled/S01E36_next_summary_southern_cross_source_1440x1080_24fps_padded_to_ready_audio_lossless.mkv \
  --review-audio-source work/review/S01E36_next_summary_southern_cross_source_001/assembled/S01E36_next_summary_southern_cross_source_review_24fps_padded_to_ready_audio.mkv \
  --out-root work/review/S01E36_next_summary_southern_cross_source_001/video_ai_tests \
  --label rife_southern_cross_padded_49fps_001 \
  --rife-ncnn-model soft/ai_video_tools/bin/rife-ncnn-vulkan/rife-v4.6 \
  --rife-ncnn-fps 48000/1001 \
  --overwrite \
  --run

python3 scripts/run_s01e36_next_summary_video_tests.py rife-ncnn \
  --source work/review/S01E36_next_summary_southern_cross_source_001/assembled/S01E36_next_summary_southern_cross_source_1440x1080_24fps_padded_to_ready_audio_lossless.mkv \
  --input-video work/review/S01E36_next_summary_southern_cross_source_001/assembled/S01E36_next_summary_southern_cross_source_1440x1080_24fps_padded_to_ready_audio_lossless.mkv \
  --review-audio-source work/review/S01E36_next_summary_southern_cross_source_001/assembled/S01E36_next_summary_southern_cross_source_review_24fps_padded_to_ready_audio.mkv \
  --out-root work/review/S01E36_next_summary_southern_cross_source_001/video_ai_tests \
  --label rife_southern_cross_padded_50fps_001 \
  --rife-ncnn-model soft/ai_video_tools/bin/rife-ncnn-vulkan/rife-v4.6 \
  --rife-ncnn-fps 50 \
  --rife-ncnn-num-frames 862 \
  --overwrite \
  --run
```

## Ready Episode Segment

S01E36's next-episode summary is prepared as a full reusable video segment,
not as a normal dialogue-track patch. The episode builder inserts it before the
end credits and suppresses the older ready audio patch
`s01e36_next_episode_summary_tts_v001`.

This is intentionally a ready-to-use package. Once the segment variants below
exist, normal `episode-final-build S01E36` runs do not rebuild the summary
video, do not rerun Real-CUGAN/RIFE, and do not remix the summary audio. The
builder only reads `segment.json`, selects the variant that matches the target
episode video, and concatenates it before the shared end credits with stream
copy.

Prepare/rebuild the reusable segment package:

```bash
python3 scripts/prepare_s01e36_next_summary_segments.py --run
```

Outputs:

- `work/ready_episode_segments/S01E36/next_summary_v001/remaster/S01E36_next_summary_remaster.mkv`
- `work/ready_episode_segments/S01E36/next_summary_v001/remaster_49fps/S01E36_next_summary_remaster_49fps.mkv`
- `work/ready_episode_segments/S01E36/next_summary_v001/ai_remaster/S01E36_next_summary_ai_remaster.mkv`
- `work/ready_episode_segments/S01E36/next_summary_v001/segment.json`

Each segment has the same four audio tracks as the final episode videos:

1. English Original 5.1
2. Spanish Restored Original Dub 5.1
3. Spanish Original Dub Restored Stereo
4. Spanish Redubbing Original Stereo

The new Spanish restored 5.1 summary uses:

- `generated_audio/next_episode_summary/S01E36/summary_v001/S01E36_summary_v001_selected_speed110_nooverlap_gap0p01_balanced_gainplus4db_preview.wav`

During segment preparation, Spanish summary beds are raised by `+2 dB` and only
the Spanish summary voices are raised by `+4 dB`. This gain is local to the
reconstructed summary segment, not the rest of the episode.

The old/new stereo Spanish summary tracks use the joined newer narrator takes
from:

- `generated_audio/newnarrator/S01E36/nextepisodesummary/`

`segment.json` also points to the relative subtitle files:

- `work/review/S01E36_next_summary_reconstruction_001/subtitles/S01E36_next_summary_english_phrases_relative.srt`
- `work/review/S01E36_next_summary_reconstruction_001/subtitles/S01E36_next_summary_spanish_phrases_relative.srt`

`episode-final-build` shifts those relative cues to the actual inserted summary
start for each final S01E36 output.

Rebuild S01E36 with the prepared summary inserted:

```bash
robotech-ai episode-final-build S01E36 \
  --review-name final_mux_oc_ec_v1 \
  --if-exists overwrite \
  --run
```

The setup script clones:

- `soft/ai_video_tools/src/bilibili-ailab` for Real-CUGAN.
- `soft/ai_video_tools/src/APISR`.
- `soft/ai_video_tools/src/AnimeSR`.
- `soft/ai_video_tools/src/ECCV2022-RIFE`.

My sandbox cannot access GitHub, so run that clone command from your terminal.
The script prints the venv/dependency install commands for each tool.

APISR regular inference follows the upstream form:
`python test_code/inference.py --input_dir XXX --weight_path XXX --store_dir XXX`.
Once APISR is installed and a model weight is in `pretrained/`, run:

```bash
python3 scripts/run_s01e36_next_summary_video_tests.py apisr \
  --label apisr_x2_test_001 \
  --apisr-cmd '{root}/.venv-video-apisr/bin/python test_code/inference.py --input_dir {input_dir} --weight_path pretrained/<APISR_WEIGHT>.pth --store_dir {output_dir}' \
  --template-cwd soft/ai_video_tools/src/APISR \
  --overwrite \
  --run
```

AnimeSR upstream video inference uses `scripts/inference_animesr_video.py`.
Once AnimeSR is installed and `AnimeSR_v2.pth` is in `weights/`, run:

```bash
python3 scripts/run_s01e36_next_summary_video_tests.py animesr \
  --label animesr_v2_x1_test_001 \
  --animesr-cmd '{root}/.venv-video-animesr/bin/python scripts/inference_animesr_video.py -i {input_video} -n AnimeSR_v2 -s 1 -o {output_dir} --expname animesr_v2_s01e36 --suffix x1 --num_process_per_gpu 1 --half' \
  --template-cwd soft/ai_video_tools/src/AnimeSR \
  --template-produces-video \
  --overwrite \
  --run
```

RIFE upstream Python inference uses `inference_video.py`. Its requirements pin
`numpy<=1.23.5`, so the installer now uses Python 3.10/3.11 and moves an
accidentally-created Python 3.12 RIFE venv aside. Once RIFE is installed and its
pretrained files are in `train_log/`, run:

```bash
python3 scripts/run_s01e36_next_summary_video_tests.py rife \
  --label rife_48fps_from_source \
  --rife-cmd '{root}/.venv-video-rife/bin/python inference_video.py --exp=1 --video={input_video} --output={output_video}' \
  --template-cwd soft/ai_video_tools/src/ECCV2022-RIFE \
  --overwrite \
  --run
```

Real-CUGAN PyTorch usage remains config-driven in the upstream repo, so the
default pipeline uses `realcugan-ncnn-vulkan` instead. If we later need the
PyTorch/VapourSynth path, run it through `--realcugan-cmd`.

## AI Video Test Candidates

Recommended test order:

1. Real-CUGAN NCNN/Vulkan or VapourSynth: anime-specific, already known to work well in prior local tests. Try conservative denoise first.
2. APISR: newer anime super-resolution approach; worth testing on the short 1440x1080 clips.
3. AnimeSR: animation video SR with temporal intent; heavier, but worth comparing if APISR/Real-CUGAN flicker.
4. RIFE or vs-mlrt RIFE for 49fps interpolation after the chosen upscale/restoration pass.

Avoid re-encoding the final episode remaster videos. Re-encode only this reconstructed summary material and credits/segments that must be resized, restored, or interpolated.

Reference links:

- Real-CUGAN: <https://github.com/bilibili/ailab/tree/main/Real-CUGAN>
- APISR: <https://arxiv.org/abs/2403.01598>
- AnimeSR: <https://arxiv.org/abs/2206.07038>
- RIFE NCNN/Vulkan: <https://github.com/nihui/rife-ncnn-vulkan>
- vs-mlrt: <https://github.com/AmusementClub/vs-mlrt>
