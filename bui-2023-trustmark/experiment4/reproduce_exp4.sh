#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x ../.venv/bin/python ]; then
    printf '%s\n' 'Install the parent Python environment first; see README.md.' >&2
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
export TMPDIR="$PWD/.cache/tmp"
mkdir -p .cache/models "$XDG_CONFIG_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$TMPDIR" results
model_host=$(../.venv/bin/python -c 'from trustmark.trustmark import MODEL_REMOTE_HOST; assert MODEL_REMOTE_HOST == "https://cai-watermark.adobe.net/watermarking/trustmark-models/"; print(MODEL_REMOTE_HOST)')
for filename in encoder_Q.ckpt decoder_Q.ckpt trustmark_Q.yaml trustmark_rm_Q.yaml trustmark_rm_Q.ckpt; do
    if [ ! -f ".cache/models/$filename" ]; then
        curl --fail --silent --show-error --proto =https --output ".cache/models/$filename" "${model_host}${filename}"
    fi
    ../.venv/bin/python -c 'import hashlib, sys; from pathlib import Path; from measure_networks import EXPECTED; from trustmark.trustmark import MODEL_CHECKSUMS; p=Path(sys.argv[1]); digest=hashlib.md5(p.read_bytes()).hexdigest(); assert digest == MODEL_CHECKSUMS[p.name] == EXPECTED[p.name], p.name; print(p.name, "MD5 match", digest)' ".cache/models/$filename"
done
../.venv/bin/python -u measure_networks.py > results/measurement.log 2>&1
printf '%s\n' 'Network measurements written to experiment4/results/.'
