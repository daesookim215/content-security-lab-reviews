# A Digital Watermark

R. G. van Schyndel, A. Z. Tirkel, C. F. Osborne (1994). *A Digital Watermark*. [DOI: 10.1109/ICIP.1994.413536](https://doi.org/10.1109/ICIP.1994.413536).
[Published review](https://chickendays.com/vanschyndel-1994-digital-watermark/).

A 511-state maximal recurrence is extended to a balanced 512-bit sequence. Nine message bits select a circular shift per image row. Coding inserts into the low bit after adaptive histogram grouping; addition inserts the same pattern into full grayscale values and uses filtered row correlation.

We did not find the authors' experimental code. This folder contains our re-implementation.

## What we tested

Airplane and Boat at 512x512; seed 199409; identical nine-bit row messages for coding and addition. All 24 cases cover clean storage and JPEG qualities 100/95/85/75/50. An independent integer dot-product decoder checks the FFT-based decoder and every metric.

## Reproduce

With Python 3.11 and the root requirements installed, run from the repository root:

```sh
python fetch_images.py
python vanschyndel-1994-digital-watermark/reproduce.py
```

The full reproduction runs twice, verifies numeric correctness and determinism,
and compares every recorded JSON value and CSV cell exactly with `results/`.
Only working runs in `outputs/` are replaced. To run the stages individually:

```sh
python vanschyndel-1994-digital-watermark/run.py run1
python vanschyndel-1994-digital-watermark/run.py run2
python vanschyndel-1994-digital-watermark/verify.py
```

All inputs come from root `images/`; generated images, arrays and numeric files
stay in this folder's ignored `outputs/`. Verification checks generated files.
The recorded reference is `results/run1/metrics.json`. Environment-only metadata and file hashes
of excluded image artifacts are omitted from recorded results.

## Results

| **Image / method** | PSNR dB | Bits % | Rows % |
|---|---|---|---|
| **Airplane / coding** | 48.35 | 100.00 | 100.00 |
| **Boat / coding** | 47.46 | 100.00 | 100.00 |
| **Airplane / addition** | 51.14 | 54.62 | 10.16 |
| **Boat / addition** | 51.14 | 52.00 | 3.91 |

The displayed precision matches the published review. Full-precision measurements
and all tested settings are in `results/`; comparisons use the full values.

## Conditions and limits

The decoders require the reference sequence, its bit convention, and the correct row boundaries. This test keeps the image dimensions and alignment fixed. It covers one sequence, one message assignment, two photographs, and JPEG compression. It does not measure resizing, rotation, cropping, combined marks, or ownership authentication.

Our coding preparation merges adjacent histogram groups and maps their averages to even brightness values before insertion. Addition uses full grayscale values, one filter pass, and the largest filtered value without a detection threshold. Values above 255 are limited to 255; no tested addition exceeded 255. Restoration subtracts the saved inserted pattern before JPEG compression; it does not use the decoder's recovered message. The results apply to this decoder and do not show equivalence to the authors' detector.
