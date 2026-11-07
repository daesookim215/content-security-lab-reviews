# Reversible Watermark Using the Difference Expansion of a Generalized Integer Transform

A. M. Alattar (2004). *Reversible Watermark Using the Difference Expansion of a Generalized Integer Transform*. [DOI: 10.1109/TIP.2004.828418](https://doi.org/10.1109/TIP.2004.828418).
[Published review](https://chickendays.com/alattar-2004-integer-transform/).

A generalized integer transform separates a rounded-down group average from signed differences. Safe differences meeting a signed threshold are expanded. A DEFLATE-compressed selection map, saved digits and a 96-bit header support exact inverse recovery.

We did not find the author's experimental code. This folder contains our re-implementation.

## What we tested

Airplane and Boat at 512x512; seed 200408; thresholds 12, 16, 24, 32, 48, 64. Triplets use columns then rows; quads use one 2x2 pass. All 24 cases use the signed difference criterion and check zero bit and pixel errors with an independent decoder. The table shows threshold 24.

## Reproduce

With Python 3.11 and the root requirements installed, run from the repository root:

```sh
python fetch_images.py
python alattar-2004-integer-transform/reproduce.py
```

The full reproduction runs twice, verifies numeric correctness and determinism,
and compares every recorded JSON value and CSV cell exactly with `results/`.
Only working runs in `outputs/` are replaced. To run the stages individually:

```sh
python alattar-2004-integer-transform/run.py signed-run1
python alattar-2004-integer-transform/run.py signed-run2
python alattar-2004-integer-transform/verify.py
```

All inputs come from root `images/`; generated images, arrays and numeric files
stay in this folder's ignored `outputs/`. Verification checks generated files.
The recorded reference is `results/signed-run1/metrics.json`. Environment-only metadata and file hashes
of excluded image artifacts are omitted from recorded results.

## Results

| **Image** | Group | Bits | PSNR |
|---|---|---|---|
| **Airplane** | Triplets | 275,590 | 28.73 dB |
| **Airplane** | Quads | 166,107 | 32.74 dB |
| **Boat** | Triplets | 247,296 | 26.76 dB |
| **Boat** | Quads | 157,106 | 31.58 dB |

The displayed precision matches the published review. Full-precision measurements
and all tested settings are in `results/`; comparisons use the full values.

## Conditions and limits

Exact reversal requires the embedded recovery data and pixel values to remain intact. We tested lossless image storage and extraction, with no compression attack, cropping, or added noise. This experiment measures reversible storage, rather than survival after editing.

Our spatial tests use grayscale images, with one brightness value per pixel. Our test uses equal weights, fixed grouping, and threshold sweeps. It uses DEFLATE, a lossless compressor, for the packed map and a length header to separate the stream's parts. These choices affect useful capacity. We tested two triplet passes and one quad pass. We did not test grouping red, green, and blue values together, adding more embedding rounds, adjusting thresholds to reach a target message size, or detecting image changes. These measurements cover two images, with random messages generated from the fixed starting value 200408.
