# Setup

System packages are installed separately with `apt`. Python dependencies should live in a project-owned virtual environment under this repository.

## Create Environment

From the project root:

```bash
cd /mnt/088E1D428E1D29A8/tmp/tvseries/Robotech.A.I

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

## Install Separation Tools

Use a dedicated environment for separation tools:

```bash
cd /mnt/088E1D428E1D29A8/tmp/tvseries/Robotech.A.I

python3 -m venv .venv-separation
source .venv-separation/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-separation.txt
python -m pip install -e .
```

This installs:

- `demucs`
- `audio-separator[cpu]`
- `torchcodec`

Do not install DeepFilterNet into this environment. Current `audio-separator`
releases require `numpy>=2`, while `deepfilternet` installs `numpy<2`.

`torchcodec` is required by recent `torchaudio` versions when Demucs saves WAV
files. If Demucs finishes separation and then crashes with
`TorchCodec is required for save_with_torchcodec`, install/update it in the
already active `.venv-separation`:

```bash
python -m pip install torchcodec
```

## Install DeepFilterNet Separately

DeepFilterNet is optional and should live in its own environment:

```bash
cd /mnt/088E1D428E1D29A8/tmp/tvseries/Robotech.A.I

python3 -m venv .venv-deepfilter
source .venv-deepfilter/bin/activate

python -m pip install -U pip setuptools
python -m pip install -r requirements-deepfilter.txt
```

Do not upgrade `wheel` in this environment unless we need it for a specific
build. On Python 3.13, current DeepFilterNet installs `packaging<24`, while
newer `wheel` releases require `packaging>=24`, which causes a pip resolver
warning. If this happens, it is usually enough to remove `wheel` from the
DeepFilterNet env:

```bash
source .venv-deepfilter/bin/activate
python -m pip uninstall -y wheel
python -m pip check
deepFilter --help
```

## Verify

```bash
source .venv-separation/bin/activate

robotech-ai tools-status
demucs --help
audio-separator --env_info
```

If you created `.venv-separation` before this instruction existed, activate it
and install the project command:

```bash
source .venv-separation/bin/activate
python -m pip install -e .
```

For DeepFilterNet:

```bash
source .venv-deepfilter/bin/activate

deepFilter --help
```

If `robotech-ai` is not found, use:

```bash
scripts/robotech-ai tools-status
```

## GPU Note

Start with CPU installs for short samples. After we know the workflow is worth pursuing, we can tune GPU installs. `audio-separator` supports a GPU extra, but CUDA/PyTorch/ONNX Runtime versions need to match the machine, so that should be a separate controlled step.

On the current Ubuntu 25 setup, `audio-separator --env_info` may report CUDA
available in Torch while ONNX Runtime only has the CPU provider. That means
Torch-backed models can use CUDA, but ONNX-backed UVR models may run on CPU
until we tune ONNX Runtime GPU separately.

## Recover From A Mixed Environment

If `deepfilternet` was installed into the same environment as `audio-separator`,
the environment is probably inconsistent because of conflicting `numpy`
requirements. The safest recovery is to remove that experiment environment and
create the split environments above:

```bash
deactivate 2>/dev/null || true
rm -rf .venv-audio

python3 -m venv .venv-separation
source .venv-separation/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-separation.txt
```

Then create `.venv-deepfilter` only if we decide to test DeepFilterNet.

## Install ClearVoice Separately

ClearVoice is an AI speech enhancement and speech super-resolution candidate.
Keep it in its own environment because it requires `numpy<2`, while the current
separation environment needs `numpy>=2`.

```bash
python3 -m venv .venv-clearvoice
.venv-clearvoice/bin/python -m pip install -U pip setuptools wheel
.venv-clearvoice/bin/python -m pip install -r requirements-clearvoice.txt
```

Run ClearVoice tests from any already active project terminal by pointing the
command at the ClearVoice interpreter:

```bash
robotech-ai clearvoice-enhance \
  work/review/S01E01_top_single_models_001_REVIEW/W04_13m00_melband_v1.wav \
  work/review/S01E01_top_single_models_001_REVIEW/W05_15m00_melband_v1.wav \
  --review-name S01E01_clearvoice_001 \
  --python .venv-clearvoice/bin/python \
  --run
```

Review only:

```text
work/review/S01E01_clearvoice_001
```
