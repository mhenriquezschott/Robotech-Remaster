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
  bash scripts/setup_audio_restoration_tools.sh check

Purpose:
  Prepare optional audio restoration tools for OC/EC rebuild experiments.

Notes:
  - Apollo targets compressed/music restoration and is the first model to test on
    degraded full-mix/music-like opening stems.
  - AudioSR is versatile SR, but its own docs warn MP3-style cutoff holes can be
    difficult unless preprocessed; use it as a controlled test, not a blanket fix.
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
  py="$(python310)"
  filtered_requirements="$ROOT/work/tmp/audiosr-requirements-no-torch.txt"
  mkdir -p "$(dirname "$filtered_requirements")"
  grep -Ev '^(--extra-index-url|torch==|torchvision==|torchaudio==)' "$SRC_DIR/AudioSR/requirements.txt" > "$filtered_requirements"
  "$py" -m venv "$ROOT/.venv-audio-audiosr"
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install --upgrade pip wheel setuptools
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install "setuptools<81"
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install -r "$filtered_requirements"
  "$ROOT/.venv-audio-audiosr/bin/python" -m pip install -e "$SRC_DIR/AudioSR" --no-deps
  echo "AudioSR env ready: $ROOT/.venv-audio-audiosr"
  echo "Run AudioSR from repo root with:"
  echo "  source .venv-audio-audiosr/bin/activate"
  echo "  audiosr -i INPUT.wav -s OUTPUT_DIR --model_name basic -d cuda --ddim_steps 50"
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
  echo "Apollo env:   $([[ -x "$ROOT/.venv-audio-apollo/bin/python" ]] && echo found || echo missing) $ROOT/.venv-audio-apollo"
  echo "AudioSR env:  $([[ -x "$ROOT/.venv-audio-audiosr/bin/python" ]] && echo found || echo missing) $ROOT/.venv-audio-audiosr"
  echo
  echo "Run from repo root:"
  echo "  bash scripts/setup_audio_restoration_tools.sh install-apollo"
  echo "  bash scripts/setup_audio_restoration_tools.sh install-audiosr"
  echo
  echo "After install, run tools from:"
  echo "  source .venv-audio-apollo/bin/activate"
  echo "  source .venv-audio-audiosr/bin/activate"
  echo
  echo "Note: this check is intentionally passive. AudioSR may contact Hugging Face"
  echo "when imported, so run model tests with the explicit commands in the docs."
}

case "${1:-}" in
  clone)
    clone_or_update "https://github.com/JusperLee/Apollo.git" "$SRC_DIR/Apollo"
    clone_or_update "https://github.com/haoheliu/versatile_audio_super_resolution.git" "$SRC_DIR/AudioSR"
    ;;
  install-apollo)
    install_apollo
    ;;
  install-audiosr)
    install_audiosr
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
