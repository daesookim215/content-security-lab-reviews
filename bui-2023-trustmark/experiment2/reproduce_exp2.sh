#!/bin/sh
set -eu
cd "$(dirname "$0")"

if [ ! -x ../.venv/bin/python ] || [ ! -f ../images/manifest.json ] || [ ! -f ../results/run1/results.json ]; then
    printf '%s\n' 'Run sh reproduce.sh in the parent folder first.' >&2
    exit 1
fi
if [ -e reproduced/run1 ] || [ -e reproduced/run2 ]; then
    printf '%s\n' 'Choose a fresh reproduced directory; run1 or run2 already exists.' >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8 BLIS_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/../.cache"
export XDG_CONFIG_HOME="$PWD/../.cache/config"
export TORCH_HOME="$PWD/../.cache/torch"
export MPLCONFIGDIR="$PWD/../.cache/matplotlib"
export TMPDIR="$PWD/reproduced/tmp"
mkdir -p "$TMPDIR"

../.venv/bin/python -u run_exp2.py reproduced/run1 > reproduced/run1.log 2>&1
../.venv/bin/python -u run_exp2.py reproduced/run2 reproduced/run1/results.json > reproduced/run2.log 2>&1
../.venv/bin/python compare_runs.py reproduced
../.venv/bin/python report_exp2.py reproduced
sh ../experiment3/reproduce_exp3.sh
