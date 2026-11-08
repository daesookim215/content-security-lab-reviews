#!/bin/sh
set -eu
cd "$(dirname "$0")"
review_python="${PYTHON:-python3.12}"
"$review_python" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install trustmark==0.9.2 numpy==2.2.6 pillow==11.3.0
.venv/bin/python apply_bch_compat.py
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/.cache"
export XDG_CONFIG_HOME="$PWD/.cache/config"
export TORCH_HOME="$PWD/.cache/torch"
.venv/bin/python download_inputs.py
.venv/bin/python download_models.py
.venv/bin/python run_experiment.py reproduced/run1
.venv/bin/python run_experiment.py reproduced/run2
cmp reproduced/run1/results.json reproduced/run2/results.json
cmp reproduced/run1/measurements.csv reproduced/run2/measurements.csv
cmp reproduced/run1/quality.csv reproduced/run2/quality.csv
cmp reproduced/run1/unmarked.csv reproduced/run2/unmarked.csv
.venv/bin/python verify_reproduction.py
