#!/usr/bin/env bash
set -euo pipefail

# Clone the Linux/Python AI video tools we want to test for the S01E36
# reconstructed next-episode summary. Dependency installation is intentionally
# split out because these projects have different CUDA/PyTorch expectations.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/soft/ai_video_tools/src"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/setup_s01e36_video_ai_tools.sh clone
  bash scripts/setup_s01e36_video_ai_tools.sh install-realcugan-ncnn
  bash scripts/setup_s01e36_video_ai_tools.sh install-apisr
  bash scripts/setup_s01e36_video_ai_tools.sh weights-apisr
  bash scripts/setup_s01e36_video_ai_tools.sh install-animesr
  bash scripts/setup_s01e36_video_ai_tools.sh install-rife
  bash scripts/setup_s01e36_video_ai_tools.sh install-rife-ncnn

This script uses Linux/Python tooling only. It does not reuse old Windows
executable bundles.
USAGE
}

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "already cloned: $dir"
    git -C "$dir" pull --ff-only
  else
    echo "cloning: $url -> $dir"
    git clone "$url" "$dir"
  fi
}

video_python() {
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
  elif command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
  elif command -v python3.10 >/dev/null 2>&1; then
    command -v python3.10
  else
    echo "ERROR: video AI tools need Python 3.10-3.12; Python 3.13 cannot satisfy current APISR pins." >&2
    exit 1
  fi
}

legacy_video_python() {
  if command -v python3.10 >/dev/null 2>&1; then
    command -v python3.10
  elif command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
  elif command -v uv >/dev/null 2>&1 && uv python find 3.11 >/dev/null 2>&1; then
    uv python find 3.11
  elif command -v uv >/dev/null 2>&1 && uv python find 3.10 >/dev/null 2>&1; then
    uv python find 3.10
  else
    echo "ERROR: RIFE pins numpy<=1.23.5, so use Python 3.10 or 3.11. Python 3.12/3.13 will try to build old NumPy and fail." >&2
    echo "If you use uv, run: uv python install 3.11" >&2
    exit 1
  fi
}

assert_supported_venv_python() {
  local venv_python="$1"
  local version
  version="$("$venv_python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  case "$version" in
    3.10|3.11|3.12) ;;
    *)
      echo "ERROR: existing venv uses Python $version: $venv_python" >&2
      echo "Move/remove that venv and rerun; APISR/AnimeSR/RIFE setup expects Python 3.10-3.12." >&2
      exit 1
      ;;
  esac
}

assert_rife_venv_python() {
  local venv_python="$1"
  local version
  version="$("$venv_python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  case "$version" in
    3.10|3.11) ;;
    *)
      echo "ERROR: existing RIFE venv uses Python $version: $venv_python" >&2
      echo "Move/remove .venv-video-rife and rerun install-rife; RIFE's pinned NumPy needs Python 3.10/3.11." >&2
      exit 1
      ;;
  esac
}

python310() {
  # Kept for old notes/scripts that may refer to this helper name.
  video_python
}

create_venv() {
  local venv_dir="$1"
  local py
  py="$(video_python)"
  if [ ! -x "$venv_dir/bin/python" ]; then
    "$py" -m venv "$venv_dir"
  fi
  assert_supported_venv_python "$venv_dir/bin/python"
}

create_rife_venv() {
  local venv_dir="$1"
  local py
  py="$(legacy_video_python)"
  if [ -x "$venv_dir/bin/python" ]; then
    local version
    version="$("$venv_dir/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    case "$version" in
      3.10|3.11) ;;
      *)
        local backup_dir="$ROOT/work/delete/venv-video-rife-py${version//./}-broken-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$(dirname "$backup_dir")"
        echo "Moving incompatible RIFE venv aside: $venv_dir -> $backup_dir"
        mv "$venv_dir" "$backup_dir"
        ;;
    esac
  fi
  if [ ! -x "$venv_dir/bin/python" ]; then
    "$py" -m venv "$venv_dir"
  fi
  assert_rife_venv_python "$venv_dir/bin/python"
}

install_torch_cu130() {
  local venv_python="$1"
  "$venv_python" -m pip install --upgrade pip wheel setuptools
  "$venv_python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
}

install_realcugan_ncnn() {
  local install_dir="$ROOT/soft/ai_video_tools/bin/realcugan-ncnn-vulkan"
  local tmp_dir="$ROOT/work/tmp/realcugan-ncnn-vulkan-download"
  mkdir -p "$install_dir" "$tmp_dir"
  "$ROOT/.venv-separation/bin/python" - <<'PY'
import json
import os
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

root = Path(os.environ["ROOT"])
install_dir = root / "soft/ai_video_tools/bin/realcugan-ncnn-vulkan"
tmp_dir = root / "work/tmp/realcugan-ncnn-vulkan-download"
api = "https://api.github.com/repos/nihui/realcugan-ncnn-vulkan/releases/latest"
with urllib.request.urlopen(api, timeout=60) as response:
    release = json.load(response)
assets = release.get("assets", [])
asset = None
for candidate in assets:
    name = candidate.get("name", "").lower()
    if name.endswith(".zip") and ("ubuntu" in name or "linux" in name):
        asset = candidate
        break
if not asset:
    raise SystemExit("Could not find a Linux/Ubuntu realcugan-ncnn-vulkan release zip in the latest GitHub release.")
url = asset["browser_download_url"]
zip_path = tmp_dir / asset["name"]
print(f"Downloading {url}")
urllib.request.urlretrieve(url, zip_path)
extract_dir = tmp_dir / "extract"
if extract_dir.exists():
    shutil.rmtree(extract_dir)
extract_dir.mkdir(parents=True)
with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(extract_dir)
binary_candidates = list(extract_dir.rglob("realcugan-ncnn-vulkan"))
if not binary_candidates:
    raise SystemExit("Downloaded archive did not contain realcugan-ncnn-vulkan.")
payload_dir = binary_candidates[0].parent
if install_dir.exists():
    shutil.rmtree(install_dir)
shutil.copytree(payload_dir, install_dir)
binary = install_dir / "realcugan-ncnn-vulkan"
binary.chmod(binary.stat().st_mode | 0o111)
print(f"Real-CUGAN NCNN ready: {binary}")
PY
}

install_rife_ncnn() {
  local install_dir="$ROOT/soft/ai_video_tools/bin/rife-ncnn-vulkan"
  local tmp_dir="$ROOT/work/tmp/rife-ncnn-vulkan-download"
  mkdir -p "$install_dir" "$tmp_dir"
  "$ROOT/.venv-separation/bin/python" - <<'PY'
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

root = Path(os.environ["ROOT"])
install_dir = root / "soft/ai_video_tools/bin/rife-ncnn-vulkan"
tmp_dir = root / "work/tmp/rife-ncnn-vulkan-download"
api = "https://api.github.com/repos/nihui/rife-ncnn-vulkan/releases/latest"
with urllib.request.urlopen(api, timeout=60) as response:
    release = json.load(response)
asset = None
for candidate in release.get("assets", []):
    name = candidate.get("name", "").lower()
    if name.endswith(".zip") and ("ubuntu" in name or "linux" in name):
        asset = candidate
        break
if not asset:
    raise SystemExit("Could not find a Linux/Ubuntu rife-ncnn-vulkan release zip in the latest GitHub release.")
url = asset["browser_download_url"]
zip_path = tmp_dir / asset["name"]
print(f"Downloading {url}")
urllib.request.urlretrieve(url, zip_path)
extract_dir = tmp_dir / "extract"
if extract_dir.exists():
    shutil.rmtree(extract_dir)
extract_dir.mkdir(parents=True)
with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(extract_dir)
binary_candidates = list(extract_dir.rglob("rife-ncnn-vulkan"))
if not binary_candidates:
    raise SystemExit("Downloaded archive did not contain rife-ncnn-vulkan.")
payload_dir = binary_candidates[0].parent
if install_dir.exists():
    shutil.rmtree(install_dir)
shutil.copytree(payload_dir, install_dir)
binary = install_dir / "rife-ncnn-vulkan"
binary.chmod(binary.stat().st_mode | 0o111)
print(f"RIFE NCNN ready: {binary}")
PY
}


cmd="${1:-}"
case "$cmd" in
  clone)
    mkdir -p "$SRC_DIR"
    clone_or_update "https://github.com/bilibili/ailab.git" "$SRC_DIR/bilibili-ailab"
    clone_or_update "https://github.com/Kiteretsu77/APISR.git" "$SRC_DIR/APISR"
    clone_or_update "https://github.com/TencentARC/AnimeSR.git" "$SRC_DIR/AnimeSR"
    clone_or_update "https://github.com/hzwer/ECCV2022-RIFE.git" "$SRC_DIR/ECCV2022-RIFE"
    cat <<'NEXT'

Cloned/updated source repos.

Next concrete steps:

  bash scripts/setup_s01e36_video_ai_tools.sh install-apisr
  bash scripts/setup_s01e36_video_ai_tools.sh weights-apisr
  python3 scripts/run_s01e36_next_summary_video_tests.py apisr --label apisr_rrdb2x_downscale_test_001 --overwrite --run

Optional later:

  bash scripts/setup_s01e36_video_ai_tools.sh install-realcugan-ncnn
  bash scripts/setup_s01e36_video_ai_tools.sh install-animesr
  bash scripts/setup_s01e36_video_ai_tools.sh install-rife-ncnn

Notes:
  - install-realcugan-ncnn downloads the current Linux release of nihui/realcugan-ncnn-vulkan.
  - install-rife-ncnn downloads the current Linux release of nihui/rife-ncnn-vulkan.
  - APISR weights-apisr downloads the 2x RRDB model directly from GitHub releases.
  - AnimeSR still needs AnimeSR_v2.pth manually placed in soft/ai_video_tools/src/AnimeSR/weights/.
  - Python RIFE still needs pretrained model files manually placed under soft/ai_video_tools/src/ECCV2022-RIFE/train_log/.
  - Real-CUGAN official Python mode is config-driven; we will configure it after clone.
NEXT
    ;;
  install-realcugan-ncnn)
    ROOT="$ROOT" install_realcugan_ncnn
    ;;
  install-rife-ncnn)
    ROOT="$ROOT" install_rife_ncnn
    ;;
  install-apisr)
    create_venv "$ROOT/.venv-video-apisr"
    install_torch_cu130 "$ROOT/.venv-video-apisr/bin/python"
    "$ROOT/.venv-video-apisr/bin/python" -m pip install -r "$SRC_DIR/APISR/requirements.txt"
    "$ROOT/.venv-video-apisr/bin/python" -m pip install ffmpegcv
    echo "APISR env ready: $ROOT/.venv-video-apisr"
    ;;
  weights-apisr)
    mkdir -p "$SRC_DIR/APISR/pretrained"
    curl -L \
      -o "$SRC_DIR/APISR/pretrained/2x_APISR_RRDB_GAN_generator.pth" \
      "https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/2x_APISR_RRDB_GAN_generator.pth"
    echo "APISR 2x RRDB weight ready: $SRC_DIR/APISR/pretrained/2x_APISR_RRDB_GAN_generator.pth"
    ;;
  install-animesr)
    create_venv "$ROOT/.venv-video-animesr"
    install_torch_cu130 "$ROOT/.venv-video-animesr/bin/python"
    "$ROOT/.venv-video-animesr/bin/python" -m pip install -r "$SRC_DIR/AnimeSR/requirements.txt"
    "$ROOT/.venv-video-animesr/bin/python" -m pip install -e "$SRC_DIR/AnimeSR"
    mkdir -p "$SRC_DIR/AnimeSR/weights"
    echo "AnimeSR env ready: $ROOT/.venv-video-animesr"
    echo "Place AnimeSR_v2.pth in: $SRC_DIR/AnimeSR/weights/"
    ;;
  install-rife)
    create_rife_venv "$ROOT/.venv-video-rife"
    install_torch_cu130 "$ROOT/.venv-video-rife/bin/python"
    "$ROOT/.venv-video-rife/bin/python" -m pip install -r "$SRC_DIR/ECCV2022-RIFE/requirements.txt"
    mkdir -p "$SRC_DIR/ECCV2022-RIFE/train_log"
    echo "RIFE env ready: $ROOT/.venv-video-rife"
    echo "Place RIFE pretrained files in: $SRC_DIR/ECCV2022-RIFE/train_log/"
    ;;
  *)
    usage
    exit 2
    ;;
esac
