# Qwen3-TTS Next-Episode Narrator Summaries

This workflow regenerates the short Spanish narrator hook near the end of an
episode when the preserved source audio is too damaged to restore cleanly.

The tool reuses the local Qwen3-TTS Gradio checkout at:

```text
/home/mhenriquez/AI/qwen3-tts-gradio
```

It does not drive the Gradio UI. Instead it calls the same Python model wrapper
directly, loads the Base voice-clone model once, builds the narrator voice prompt
once, and then generates all phrase takes in one process.

## Reference Voice

Default reference files:

```text
train/mainnarrator/mainnarrator01.wav
train/mainnarrator/mainnarrator01.txt
```

The reference text must be the exact transcript of the reference audio. The
default language label is `Spanish`, matching the Qwen3-TTS app dropdown.

## Expressiveness / Enthusiasm Notes

The current `tts-summary-generate` command uses the Qwen3-TTS Base voice-clone
model. In this mode, expressiveness is influenced mostly by:

- the reference narrator clip and transcript,
- punctuation and wording in `phrase_plan.json`,
- seed/take selection,
- generation sampling parameters.

The Base voice-clone path does not expose a direct `instruct` or emotion tag.
Qwen3-TTS does expose instruction-based tone/prosody control in the CustomVoice
and VoiceDesign model families, and the official model card describes a
VoiceDesign-then-Clone workflow. That is a candidate future experiment if we
need more excited "next episode" narration while keeping a reusable narrator
voice. A smaller near-term experiment is to expose Base generation knobs such as
`temperature`, `top_p`, `subtalker_temperature`, and `subtalker_top_p` for wider
take variation.

Example: generate a more varied batch for only the final phrase without touching
the previously approved takes:

```bash
robotech-ai tts-summary-generate S01E01 \
  --summary-id summary_v001 \
  --phrases 04 \
  --takes 10 \
  --temperature 1.05 \
  --top-p 0.95 \
  --subtalker-temperature 1.05 \
  --subtalker-top-p 0.95 \
  --device cuda \
  --run
```

More conservative variation:

```bash
robotech-ai tts-summary-generate S01E01 \
  --summary-id summary_v001 \
  --phrases 04 \
  --takes 10 \
  --temperature 0.95 \
  --top-p 0.90 \
  --subtalker-temperature 1.00 \
  --device cuda \
  --run
```

These parameters are not direct emotion controls. They increase or narrow the
sampling space so the generated takes can vary more in delivery. Review the
takes by ear and then edit `selected_takes.json` as usual.

## 1. Create Or Refresh The Speech Map

The summary planner uses the full-episode ASR speech map. If it is missing,
`tts-summary-plan` prints this command:

```bash
PYTHONPATH=src .venv-asr/bin/python -m robotech_ai.cli speech-map S01E04 \
  --source dialogue \
  --engine whisperx \
  --model large-v3 \
  --language es \
  --device cuda \
  --compute-type float16 \
  --batch-size 4 \
  --run
```

Outputs:

```text
work/speech_maps/S01E##/S01E##_dialogue_speech_map.json
work/speech_maps/S01E##/S01E##_dialogue_phrases.srt
work/speech_maps/S01E##/S01E##_dialogue_words.tsv
work/speech_maps/S01E##/S01E##_dialogue_words.srt
```

## 2. One-Step Plan And Generation

Normally, run generation with `--start` and `--end`. If the phrase plan does not
exist yet, the command creates it first, then runs Qwen3-TTS:

```bash
robotech-ai tts-summary-generate S01E04 \
  --start 00:21:30 \
  --end 00:22:10 \
  --summary-id summary_v001 \
  --takes 10 \
  --model-size 1.7B \
  --language Spanish \
  --chunk-size 200 \
  --chunk-gap 0 \
  --device cuda \
  --run
```

## Optional: Create A Phrase Plan Only

Configured summary start points live in:

```text
config/next_episode_summaries.json
```

If `--end` is omitted, the planner uses the final ASR phrase after the summary
start. This is useful for the normal next-episode summaries, where the narrator
continues to the end of the episode segment.

Create all missing configured plans at once, skipping existing plans unless
`--overwrite` is provided:

```bash
robotech-ai tts-summary-plan all --run
```

Use the real start/end of one next-episode narrator section when overriding the
config:

```bash
robotech-ai tts-summary-plan S01E04 \
  --start 00:21:30 \
  --end 00:22:10 \
  --summary-id summary_v001 \
  --run
```

Output:

```text
generated_audio/next_episode_summary/S01E04/summary_v001/phrase_plan.json
```

The plan records phrase number, source timing, duration, and ASR text.
S01E35 is currently marked `pending`; S01E36 is marked `none`.

## Optional: Generate From An Existing Plan

Run this when you want to inspect or edit `phrase_plan.json` before spending GPU
time. The command uses the Qwen venv directly:

```bash
robotech-ai tts-summary-generate S01E04 \
  --summary-id summary_v001 \
  --takes 10 \
  --model-size 1.7B \
  --language Spanish \
  --chunk-size 200 \
  --chunk-gap 0 \
  --device cuda \
  --run
```

If CUDA is not visible inside the Qwen venv, use `--device auto` or fix that
environment before running. The runner sets:

```text
NUMBA_CACHE_DIR=/tmp/robotech_numba_cache
```

to avoid the local `librosa/numba` cache crash.

## Outputs

```text
generated_audio/next_episode_summary/S01E04/summary_v001/
  phrase_plan.json
  selected_takes.json
  S01E04_summary_v001_all_v01_preview.wav
  manifest.json
  REVIEW.md
  01/
    01_S01E04_v01.wav
    01_S01E04_v02.wav
    ...
  02/
    02_S01E04_v01.wav
    02_S01E04_v02.wav
    ...
```

The preview WAV uses `v01` for every phrase. Edit `selected_takes.json` later to
switch phrase `01`, `02`, etc. to better generated versions.

## Review And Edit Loop

Edit phrase text here when ASR wording needs small corrections:

```text
generated_audio/next_episode_summary/S01E##/summary_v001/phrase_plan.json
```

The per-phrase `text.txt` files are convenience copies only; the generator reads
the canonical text from `phrase_plan.json`.

Generate another ten takes for only phrase `03`, appending after existing takes
when possible. For example, if `v01` through `v10` already exist, this creates
`v11` through `v20`:

```bash
robotech-ai tts-summary-generate S01E01 \
  --summary-id summary_v001 \
  --phrases 03 \
  --takes 10 \
  --device cuda \
  --run
```

Replace an existing phrase's takes instead of appending:

```bash
robotech-ai tts-summary-generate S01E01 \
  --summary-id summary_v001 \
  --phrases 03 \
  --takes 10 \
  --replace-phrase \
  --device cuda \
  --run
```

Choose takes by editing:

```text
generated_audio/next_episode_summary/S01E##/summary_v001/selected_takes.json
```

Then rebuild the selected preview without loading Qwen:

```bash
robotech-ai tts-summary-generate S01E01 \
  --summary-id summary_v001 \
  --assemble-only \
  --run
```

If selected takes overlap because generated phrases are longer than the original
ASR slots, rebuild without changing speed or pitch by shifting later phrases
inside available gaps:

```bash
robotech-ai tts-summary-generate S01E01 \
  --summary-id summary_v001 \
  --assemble-only \
  --avoid-overlap \
  --min-gap 0.05 \
  --run
```

Add phrase gain balancing when generated takes have slightly different levels:

```bash
robotech-ai tts-summary-generate S01E01 \
  --summary-id summary_v001 \
  --assemble-only \
  --avoid-overlap \
  --min-gap 0.05 \
  --balance-phrases \
  --balance-max-gain-db 3 \
  --run
```

This changes only phrase placement and gain. It does not time-stretch, pitch
shift, or regenerate TTS.

`--balance-max-gain-db` limits how much each individual phrase may move while
matching phrase loudness. Do not use it as the general volume control for the
whole summary. If the complete summary is balanced internally but too low or
too loud in the final mix, add a final group gain after balancing:

```bash
robotech-ai tts-summary-generate S01E02 \
  --summary-id summary_v001 \
  --assemble-only \
  --avoid-overlap \
  --min-gap 0.01 \
  --balance-phrases \
  --balance-max-gain-db 3 \
  --summary-gain-db 2 \
  --run
```

Negative values work too, for example `--summary-gain-db -1.5`.

If the chosen takes are still too long, apply one speed change to every selected
take. Keep this as low as the approved audio allows; S01E15 needed `110%` to
fit its original window cleanly:

```bash
robotech-ai tts-summary-generate S01E15 \
  --summary-id summary_v001 \
  --assemble-only \
  --speed-all-percent 110 \
  --avoid-overlap \
  --min-gap 0.10 \
  --balance-phrases \
  --balance-max-gain-db 3 \
  --run
```

Per-phrase speed overrides can be placed in `selected_takes.json` when only one
line needs a tiny adjustment. The global `--speed-all-percent` still applies to
every phrase first; these overrides apply after that:

```json
{
  "kind": "robotech_qwen3_tts_summary_selection",
  "episode": "S01E02",
  "summary_id": "summary_v001",
  "selected": {
    "01": "v03",
    "02": "v15",
    "03": "v10"
  },
  "speed_percent": {
    "03": 102
  }
}
```

Then rebuild normally with `--assemble-only`. Phrase speeds should stay subtle
unless the clip is only a rough timing test.

## Subtitle / ASR Text Review

The Spanish text checker flags words that are missing from the Spanish
dictionary and the Robotech allowlist. It is a review tool, not an automatic
fixer.

```bash
robotech-ai subtitle-spellcheck all --source both --run
```

Outputs:

```text
work/review/subtitle_spellcheck/S01E##/S01E##_spanish_spellcheck.md
work/review/subtitle_spellcheck/S01E##/S01E##_spanish_spellcheck.json
```

Project-specific accepted words are in:

```text
config/subtitles/spanish_allowlist.txt
config/subtitles/robotech_glossary.json
```

Add valid Robotech names, ship names, sound labels, or recurring Spanish words
to the allowlist when the report is too noisy.

For correction work, generate an Excel workbook with one sheet per episode and
one row per suspicious subtitle occurrence:

```bash
robotech-ai subtitle-review-workbook all --source subtitles --run
```

Default output:

```text
work/review/subtitle_language_review/spanish_language_review.xlsx
```

Editable columns:

- `action`: planned values are `ignore`, `allowlist`, `replace_word`, or `replace_context`.
- `replacement_word`: use when only the suspicious word should change.
- `replacement_context`: use when the whole subtitle cue text should be replaced.
- `notes`: free review notes.

Leave `episode`, `review_id`, `source`, `cue`, `time`, and `path` unchanged so
the workbook can later be imported back into the SRT files.

After reviewing and saving the workbook, apply word-level fixes back to the SRT
files with:

```bash
scripts/apply_subtitle_review_workbook.py --run
```

The importer only changes subtitle text lines in the targeted cues. Cue numbers
and timing lines are not rewritten. `replacement_word` performs a conservative
word-level replacement inside the cue; `replacement_context` replaces only the
cue text block while preserving the original cue number and timing line.

Bracketed speaker/action labels are reviewed separately because names and sound
descriptions behave differently from normal prose:

```bash
python3 scripts/create_subtitle_bracket_review_workbook.py \
  --out work/review/subtitle_language_review/spanish_bracket_review_pass2.xlsx \
  --run
```

## Promote An Approved Summary To The Episode Pipeline

After a selected preview is approved, promote it to a ready audio patch. The
command copies the approved preview to `replacement.wav` and creates `patch.json`
with the placement derived from `phrase_plan.json`.

Example:

```bash
robotech-ai tts-summary-promote S01E15 \
  --summary-id summary_v001 \
  --preview generated_audio/next_episode_summary/S01E15/summary_v001/timing_fit_options_001/S01E15_summary_v001_selected_speed110_gap0p1_balanced_preview.wav \
  --overwrite \
  --run
```

Default output:

```text
work/ready_audio_patches/S01E##/<patch_id>/
  replacement.wav
  patch.json
```

If `--patch-id` is omitted, the command uses:

```text
s01e##_next_episode_summary_tts_v001
```

Use `--patch-id` only when you intentionally want a different ready patch folder.
Use `--overwrite` when replacing an existing approved summary patch.

If the assembled preview is internally balanced but too low or too loud after
being mixed into the episode, either rebuild the preview with
`--summary-gain-db`, or store a final patch gain during promotion:

```bash
robotech-ai tts-summary-promote S01E15 \
  --summary-id summary_v001 \
  --preview generated_audio/next_episode_summary/S01E15/summary_v001/S01E15_summary_v001_selected_speed110_nooverlap_gap0p1_balanced_preview.wav \
  --replacement-gain-db 2 \
  --overwrite \
  --run
```

To estimate that gain from the current restored dialogue destination, use:

```bash
robotech-ai tts-summary-promote S01E15 \
  --summary-id summary_v001 \
  --preview generated_audio/next_episode_summary/S01E15/summary_v001/S01E15_summary_v001_selected_speed110_nooverlap_gap0p1_balanced_preview.wav \
  --match-source-work-level \
  --match-max-gain-db 6 \
  --overwrite \
  --run
```

`--balance-max-gain-db` is only the cap for phrase-to-phrase RMS matching. Once
all phrases are already close to the target median, raising it further may not
change much. Also, assembled previews are peak-limited near full scale, so very
large gain requests can be silently reduced to avoid clipping.

Ready summary patches target the restored Spanish dialogue workflow only. They
are applied before the restored Spanish 5.1 mix is created. They do not touch
the old Spanish stereo/fullmix preservation track unless explicitly requested.

For S01E01, the currently approved summary patch is:

```text
work/ready_audio_patches/S01E01/s01e01_next_episode_summary_tts_v001/
```

It uses selected takes `01=v07`, `02=v02`, `03=v20`, `04=v32`, with no-overlap
scheduling at `0.10s` minimum gap and phrase RMS balancing.
