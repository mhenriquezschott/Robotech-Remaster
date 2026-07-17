#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/soft/ai_audio_tools/src"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_audio_restoration_tools.sh clone
  bash scripts/setup_audio_restoration_tools.sh install-apollo
  bash scripts/setup_audio_restoration_tools.sh install-audiosr
  bash scripts/setup_audio_restoration_tools.sh install-audiosep
  bash scripts/setup_audio_restoration_tools.sh download-audiosep-checkpoints
  bash scripts/setup_audio_restoration_tools.sh install-audiosep-dp
  bash scripts/setup_audio_restoration_tools.sh download-audiosep-dp-checkpoints
  bash scripts/setup_audio_restoration_tools.sh check

Purpose:
  Prepare optional audio restoration tools for OC/EC rebuild experiments.

Notes:
  - Apollo targets compressed/music restoration and is the first model to test on
    degraded full-mix/music-like opening stems.
  - AudioSR is versatile SR, but its own docs warn MP3-style cutoff holes can be
    difficult unless preprocessed; use it as a controlled test, not a blanket fix.
  - AudioSep is a language-query separator. It is the first model to test for
    opening-credit SFX such as "motorcycle engine" or "laser gun sound effects".
  - AudioSep-DP is the separator released with TQ-SED. TQ-SED is the event
    detection research pipeline; for our opening-credit SFX extraction tests we
    use the AudioSep-DP LASS separator directly.
  - A2SB is documented as promising for bandwidth extension and inpainting, but
    NVIDIA's project page says code/checkpoints are coming soon.
EOF
}

clone_or_update() {
  local url="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -d "$dest/.git" ]]; then
    git -C "$dest" pull --ff-only
  else
    git clone "$url" "$dest"
  fi
}

python310() {
  if command -v python3.10 >/dev/null 2>&1; then
    command -v python3.10
  elif [[ -x "$ROOT/.uv-python/cpython-3.10-linux-x86_64-gnu/bin/python" ]]; then
    echo "$ROOT/.uv-python/cpython-3.10-linux-x86_64-gnu/bin/python"
  else
    echo "ERROR: Python 3.10 is required for these older CUDA dependency stacks." >&2
    echo "Install one with uv or system packages, for example:" >&2
    echo "  uv python install 3.10" >&2
    exit 1
  fi
}

install_apollo() {
  local py
  py="$(python310)"
  "$py" -m venv "$ROOT/.venv-audio-apollo"
  "$ROOT/.venv-audio-apollo/bin/python" -m pip install --upgrade pip wheel setuptools
  "$ROOT/.venv-audio-apollo/bin/python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
  "$ROOT/.venv-audio-apollo/bin/python" -m pip install \
    numpy==1.26.3 scipy soundfile librosa==0.10.2.post1 hydra-core==1.3.2 \
    pytorch-lightning==2.2.1 torchmetrics==1.4.1 huggingface-hub==0.24.6 \
    rich tqdm pyyaml omegaconf einops fast-bss-eval torch-complex
  "$ROOT/.venv-audio-apollo/bin/python" - <<PY
from pathlib import Path
import site
repo = Path("$SRC_DIR/Apollo").resolve()
for sp in site.getsitepackages():
    path = Path(sp) / "apollo-local.pth"
    path.write_text(str(repo) + "\\n", encoding="utf-8")
print("Apollo PYTHONPATH shim installed for", repo)
PY
  echo "Apollo env ready: $ROOT/.venv-audio-apollo"
  echo "Run Apollo from repo root with:"
  echo "  source .venv-audio-apollo/bin/activate"
  echo "  python soft/ai_audio_tools/src/Apollo/inference.py --in_wav INPUT.wav --out_wav OUTPUT.wav"
}

install_audiosr() {
  local py
  local filtered_requirements
  local utils_py
  py="$(python310)"
  filtered_requirements="$ROOT/work/tmp/audiosr-requirements-no-torch.txt"
  utils_py="$SRC_DIR/AudioSR/audiosr/utils.py"
  mkdir -p "$(dirname "$filtered_requirements")"
  grep -Ev '^(--extra-index-url|torch==|torchvision==|torchaudio==)' "$SRC_DIR/AudioSR/requirements.txt" > "$filtered_requirements"
  "$py" -m venv "$ROOT/.venv-audio-audiosr"
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install --upgrade pip wheel setuptools
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install "setuptools<81"
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install torchcodec
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install -r "$filtered_requirements"
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install -e "$SRC_DIR/AudioSR" --no-deps
  "$ROOT/.venv-audio-audiosr/bin/python" - <<PY
from pathlib import Path

path = Path("$utils_py")
text = path.read_text(encoding="utf-8")
old = "        # Reshape waveform for soundfile\\n        data_to_save = waveform[i].T.cpu().numpy()\\n"
new = """        # Reshape waveform for soundfile. Newer torch/torchaudio stacks may
        # return NumPy here while older AudioSR expected a Torch tensor.
        data_to_save = waveform[i].T
        if hasattr(data_to_save, \"cpu\"):
            data_to_save = data_to_save.cpu().numpy()
        else:
            data_to_save = np.asarray(data_to_save)
"""
if old in text:
    path.write_text(text.replace(old, new), encoding="utf-8")
print("AudioSR save compatibility patch checked:", path)
PY
  echo "AudioSR env ready: $ROOT/.venv-audio-audiosr"
  echo "Run AudioSR from repo root with:"
  echo "  source .venv-audio-audiosr/bin/activate"
  echo "  audiosr -i INPUT.wav -s OUTPUT_DIR --model_name basic -d cuda --ddim_steps 50"
}

install_audiosep() {
  local py
  py="$(python310)"
  "$py" -m venv "$ROOT/.venv-audio-audiosep"
  "$ROOT/.venv-audio-audiosep/bin/python" -m pip install --upgrade pip wheel setuptools
  "$ROOT/.venv-audio-audiosep/bin/python" -m pip install "setuptools<81"
  "$ROOT/.venv-audio-audiosep/bin/python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
  "$ROOT/.venv-audio-audiosep/bin/python" -m pip install \
    "numpy<2" scipy soundfile librosa==0.10.0.post2 soxr pyyaml \
    huggingface-hub transformers==4.28.1 lightning==2.0.0 \
    torchlibrosa==0.1.0 panns-inference==0.1.0 h5py timm==0.3.2 \
    ftfy regex braceexpand webdataset pandas wget tqdm
  "$ROOT/.venv-audio-audiosep/bin/python" - <<PY
from pathlib import Path
import site
repo = Path("$SRC_DIR/AudioSep").resolve()
for sp in site.getsitepackages():
    path = Path(sp) / "audiosep-local.pth"
    path.write_text(str(repo) + "\\n", encoding="utf-8")
print("AudioSep PYTHONPATH shim installed for", repo)
PY
  echo "AudioSep env ready: $ROOT/.venv-audio-audiosep"
  echo "Run AudioSep prompt tests from repo root with:"
  echo "  .venv-audio-audiosep/bin/python scripts/run_audiosep_prompts.py --input INPUT.wav --out-dir OUT --prompt 'motorcycle engine sound, no music, no speech' --device cuda --use-chunk"
}

install_audiosep_dp() {
  local py
  py="$(python310)"
  "$py" -m venv "$ROOT/.venv-audio-audiosepdp"
  "$ROOT/.venv-audio-audiosepdp/bin/python" -m pip install --upgrade pip wheel setuptools
  "$ROOT/.venv-audio-audiosepdp/bin/python" -m pip install "setuptools<81"
  "$ROOT/.venv-audio-audiosepdp/bin/python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
  "$ROOT/.venv-audio-audiosepdp/bin/python" -m pip install \
    "numpy<2" scipy soundfile librosa==0.10.0.post2 soxr pyyaml \
    huggingface-hub transformers==4.44.2 pytorch-lightning==2.2.0 \
    torchmetrics==1.4.1 torchlibrosa==0.1.0 pandas h5py ftfy regex \
    braceexpand webdataset tqdm einops timm==1.0.9 laion-clap==1.1.6
  "$ROOT/.venv-audio-audiosepdp/bin/python" - <<PY
from pathlib import Path
import site
repo = Path("$SRC_DIR/TQ-SED").resolve()
lass = repo / "LASS_codes"
for sp in site.getsitepackages():
    path = Path(sp) / "audiosep-dp-local.pth"
    path.write_text(str(repo) + "\\n" + str(lass) + "\\n", encoding="utf-8")
print("AudioSep-DP PYTHONPATH shim installed for", repo)
PY
  echo "AudioSep-DP env ready: $ROOT/.venv-audio-audiosepdp"
  echo "Run AudioSep-DP prompt tests from repo root with:"
  echo "  .venv-audio-audiosepdp/bin/python scripts/run_audiosep_dp_prompts.py --input INPUT.wav --out-dir OUT --prompt 'motorcycle engine sound, no music, no speech' --device cuda"
}

download_audiosep_checkpoints() {
  local checkpoint_dir="$SRC_DIR/AudioSep/checkpoint"
  mkdir -p "$checkpoint_dir"
  "$ROOT/.venv-audio-audiosep/bin/python" - <<PY
from pathlib import Path
from urllib.request import urlretrieve

checkpoint_dir = Path("$checkpoint_dir")
models = [
    (
        "https://huggingface.co/spaces/badayvedat/AudioSep/resolve/main/checkpoint/audiosep_base_4M_steps.ckpt",
        checkpoint_dir / "audiosep_base_4M_steps.ckpt",
    ),
    (
        "https://huggingface.co/spaces/badayvedat/AudioSep/resolve/main/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt",
        checkpoint_dir / "music_speech_audioset_epoch_15_esc_89.98.pt",
    ),
]
for url, path in models:
    if path.exists() and path.stat().st_size > 0:
        print("exists:", path)
        continue
    print("downloading:", url)
    urlretrieve(url, path)
    print("wrote:", path)
PY
}

download_audiosep_dp_checkpoints() {
  local checkpoint_dir="$SRC_DIR/TQ-SED/LASS_codes/checkpoints"
  mkdir -p "$checkpoint_dir"
  "$ROOT/.venv-audio-audiosepdp/bin/python" - <<PY
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

checkpoint_dir = Path("$checkpoint_dir")
models = [
    (
        "https://zenodo.org/records/14208090/files/resunet_with_dprnn_32k.zip?download=1",
        checkpoint_dir / "resunet_with_dprnn_32k.zip",
    ),
]
for url, path in models:
    if path.exists() and path.stat().st_size > 100_000_000:
        print("exists:", path)
    else:
        print("downloading:", url)
        urlretrieve(url, path)
        print("wrote:", path)
    extract_dir = checkpoint_dir / path.stem
    if extract_dir.exists() and any(extract_dir.rglob("*.ckpt")):
        print("extracted:", extract_dir)
        continue
    extract_dir.mkdir(parents=True, exist_ok=True)
    print("extracting:", path)
    with ZipFile(path) as archive:
        archive.extractall(extract_dir)
    print("extracted:", extract_dir)
PY
}

check_tools() {
  local py310_status="missing"
  if command -v python3.10 >/dev/null 2>&1; then
    py310_status="$(command -v python3.10)"
  elif [[ -x "$ROOT/.uv-python/cpython-3.10-linux-x86_64-gnu/bin/python" ]]; then
    py310_status="$ROOT/.uv-python/cpython-3.10-linux-x86_64-gnu/bin/python"
  fi
  echo "Python 3.10:  $py310_status"
  echo "Apollo repo:  $([[ -d "$SRC_DIR/Apollo" ]] && echo found || echo missing) $SRC_DIR/Apollo"
  echo "AudioSR repo: $([[ -d "$SRC_DIR/AudioSR" ]] && echo found || echo missing) $SRC_DIR/AudioSR"
  echo "AudioSep repo:$([[ -d "$SRC_DIR/AudioSep" ]] && echo found || echo missing) $SRC_DIR/AudioSep"
  echo "TQ-SED repo:  $([[ -d "$SRC_DIR/TQ-SED" ]] && echo found || echo missing) $SRC_DIR/TQ-SED"
  echo "Apollo env:   $([[ -x "$ROOT/.venv-audio-apollo/bin/python" ]] && echo found || echo missing) $ROOT/.venv-audio-apollo"
  echo "AudioSR env:  $([[ -x "$ROOT/.venv-audio-audiosr/bin/python" ]] && echo found || echo missing) $ROOT/.venv-audio-audiosr"
  echo "AudioSep env: $([[ -x "$ROOT/.venv-audio-audiosep/bin/python" ]] && echo found || echo missing) $ROOT/.venv-audio-audiosep"
  echo "AudioSep-DP env: $([[ -x "$ROOT/.venv-audio-audiosepdp/bin/python" ]] && echo found || echo missing) $ROOT/.venv-audio-audiosepdp"
  echo "AudioSep ckpt:$([[ -s "$SRC_DIR/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt" && -s "$SRC_DIR/AudioSep/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt" ]] && echo found || echo missing) $SRC_DIR/AudioSep/checkpoint"
  echo "AudioSep-DP ckpt:$([[ -n "$(find "$SRC_DIR/TQ-SED/LASS_codes/checkpoints" -name '*.ckpt' -print -quit 2>/dev/null)" ]] && echo found || echo missing) $SRC_DIR/TQ-SED/LASS_codes/checkpoints"
  echo
  echo "Run from repo root:"
  echo "  bash scripts/setup_audio_restoration_tools.sh install-apollo"
  echo "  bash scripts/setup_audio_restoration_tools.sh install-audiosr"
  echo "  bash scripts/setup_audio_restoration_tools.sh install-audiosep"
  echo "  bash scripts/setup_audio_restoration_tools.sh download-audiosep-checkpoints"
  echo "  bash scripts/setup_audio_restoration_tools.sh install-audiosep-dp"
  echo "  bash scripts/setup_audio_restoration_tools.sh download-audiosep-dp-checkpoints"
  echo
  echo "After install, run tools from:"
  echo "  source .venv-audio-apollo/bin/activate"
  echo "  source .venv-audio-audiosr/bin/activate"
  echo "  source .venv-audio-audiosep/bin/activate"
  echo "  source .venv-audio-audiosepdp/bin/activate"
  echo
  echo "Note: this check is intentionally passive. AudioSR may contact Hugging Face"
  echo "when imported, so run model tests with the explicit commands in the docs."
}

case "${1:-}" in
  clone)
    clone_or_update "https://github.com/JusperLee/Apollo.git" "$SRC_DIR/Apollo"
    clone_or_update "https://github.com/haoheliu/versatile_audio_super_resolution.git" "$SRC_DIR/AudioSR"
    clone_or_update "https://github.com/Audio-AGI/AudioSep.git" "$SRC_DIR/AudioSep"
    ;;
  install-apollo)
    install_apollo
    ;;
  install-audiosr)
    install_audiosr
    ;;
  install-audiosep)
    install_audiosep
    ;;
  install-audiosep-dp)
    install_audiosep_dp
    ;;
  download-audiosep-checkpoints)
    download_audiosep_checkpoints
    ;;
  download-audiosep-dp-checkpoints)
    download_audiosep_dp_checkpoints
    ;;
  check)
    check_tools
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
