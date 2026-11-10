#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

VENV="${VENV:-.venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
# Bound cuDNN workspaces and avoid autotuning spikes on the shared GPU.
export CUDNN_CONV_WSCAP_DBG=64
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REQ_HASH=$(sha256sum requirements.txt | cut -d ' ' -f 1)
MARKER="$VENV/.requirements-$REQ_HASH"
if [[ ! -f "$MARKER" ]]; then
    echo "Installing requirements $REQ_HASH into $VENV"
    mapfile -t TORCH_REQUIREMENTS < <(awk '/^(torch|torchvision)==/ {print}' requirements.txt)
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 "${TORCH_REQUIREMENTS[@]}"
    # -c prevents transitive requirements from replacing the CUDA builds.
    python -m pip install --no-cache-dir -r requirements.txt -c requirements.txt
    python -m pip check
    touch "$MARKER"
fi

# main prints Python, torch, built CUDA, CUDA availability, GPU, and both
# watermark package versions before loading models or reading any images.
python -u main.py --config "${1:-config.yaml}"
