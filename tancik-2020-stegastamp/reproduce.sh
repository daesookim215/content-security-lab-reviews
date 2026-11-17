#!/bin/sh
set -eu
trap 'review_status=$?; if [ "$review_status" -ne 0 ]; then printf "%s\n" "FAIL: reproduction stopped"; fi' 0
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=""
export XDG_CACHE_HOME="$PWD/.cache" XDG_CONFIG_HOME="$PWD/.cache/config"
export TMPDIR="$PWD/.cache/tmp" PIP_CACHE_DIR="$PWD/.cache/pip"
export MAMBA_ROOT_PREFIX="$PWD/.cache/mamba" CONDA_PKGS_DIRS="$PWD/.cache/conda-pkgs"
export PIP_CONFIG_FILE=/dev/null
mkdir -p .cache/tmp run-output
if [ ! -x env/bin/python ]; then
    if command -v micromamba >/dev/null 2>&1; then
        micromamba create -y -p "$PWD/env" -c conda-forge python=3.7 pip=24.0
    elif command -v conda >/dev/null 2>&1; then
        conda create -y -p "$PWD/env" -c conda-forge python=3.7 pip=24.0
    else
        printf '%s\n' 'Install micromamba or conda to create the Python 3.7 environment.'
        exit 1
    fi
fi
env/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 7) else "Python 3.7 is required")'
env/bin/python -m pip --disable-pip-version-check install -r results/pip_freeze.txt
env/bin/python code/download_source.py
env/bin/python code/download_inputs.py
env/bin/python code/download_weights.py
env/bin/python code/measure_networks.py
env/bin/python -u code/run_experiment.py run1
env/bin/python -u code/run_experiment.py run2
env/bin/python code/compare_runs.py
env/bin/python code/report.py
printf '%s\n' 'PASS: downloads, two CPU runs, and recorded references'
