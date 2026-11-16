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
export XDG_CACHE_HOME="$PWD/.cache"
export XDG_CONFIG_HOME="$PWD/.cache/config"
export TORCH_HOME="$PWD/.cache/torch"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TMPDIR="$PWD/reproduced/tmp"
mkdir -p .cache/models "$TMPDIR"
model_host=$(../.venv/bin/python -c 'from trustmark.trustmark import MODEL_REMOTE_HOST; assert MODEL_REMOTE_HOST == "https://cai-watermark.adobe.net/watermarking/trustmark-models/"; print(MODEL_REMOTE_HOST)')
for filename in trustmark_rm_Q.yaml trustmark_rm_Q.ckpt; do
    if [ ! -f ".cache/models/$filename" ]; then
        curl --fail --silent --show-error --proto =https --output ".cache/models/$filename" "${model_host}${filename}"
    fi
    ../.venv/bin/python -c 'import hashlib, sys; from pathlib import Path; from trustmark.trustmark import MODEL_CHECKSUMS; p=Path(sys.argv[1]); expected={"trustmark_rm_Q.yaml":"8476bcd4092abf302272868f3b4c2e38", "trustmark_rm_Q.ckpt":"760337a5596e665aed2ab5c49aa5284f"}; digest=hashlib.md5(p.read_bytes()).hexdigest(); assert digest == MODEL_CHECKSUMS[p.name] == expected[p.name], p.name; print(p.name, "MD5 match", digest)' ".cache/models/$filename"
done
for run in run1 run2; do
    ../.venv/bin/python -u run_exp3.py "reproduced/$run" --models .cache/models --inputs ../images --reference ../results/run1/results.json > "reproduced/$run.log" 2>&1
done
../.venv/bin/python compare_runs.py reproduced
