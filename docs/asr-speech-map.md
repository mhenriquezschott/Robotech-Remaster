# ASR Speech Maps

Use this workflow to create word-level timing maps for the restored Spanish dialogue tracks. The maps are meant to help find phrases like a subtitle file, then place repair-tool cuts more precisely.

## Environment

Use a separate ASR environment so WhisperX/faster-whisper dependencies do not disturb the separation or LLM environments.

```bash
python3 -m venv .venv-asr
.venv-asr/bin/python -m pip install --upgrade pip setuptools wheel
.venv-asr/bin/python -m pip install -r requirements-asr.txt
.venv-asr/bin/python -m pip install -e . --no-build-isolation
```

If the install fails on Python 3.13, create `.venv-asr` with Python 3.12 instead.

## Create A Speech Map

Default source is the restored Spanish dialogue track used for repair work:

```bash
robotech-ai speech-map S01E03 \
  --source dialogue \
  --engine whisperx \
  --model large-v3 \
  --language es \
  --batch-size 4 \
  --run
```

Outputs go to:

```text
work/speech_maps/S01E03/
```

Expected files:

- `S01E03_dialogue_speech_map.json`: canonical machine-readable map.
- `S01E03_dialogue_words.tsv`: word table for spreadsheet/manual review.
- `S01E03_dialogue_words.srt`: one subtitle cue per recognized word.
- `S01E03_dialogue_phrases.srt`: ASR phrase segments.

## Create Missing Maps For All Episodes

Existing speech maps are skipped by default:

```bash
robotech-ai speech-map all \
  --source dialogue \
  --engine whisperx \
  --model large-v3 \
  --language es \
  --device cuda \
  --compute-type float16 \
  --batch-size 4 \
  --run
```

Add `--overwrite` only when you want to rebuild existing maps.

## Find A Phrase

```bash
robotech-ai speech-find S01E03 "rumbo de colision" --around 00:00:35
```

The finder is accent/punctuation-insensitive and fuzzy, so it can still find close ASR variants.

## Fallback Engine

If WhisperX alignment is unstable or too heavy, use faster-whisper:

```bash
robotech-ai speech-map S01E03 \
  --source dialogue \
  --engine faster-whisper \
  --model large-v3 \
  --language es \
  --compute-type float16 \
  --run
```

For GPU memory issues, lower `--batch-size` or use `--compute-type int8`.
