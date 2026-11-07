# Secure Spread Spectrum Watermarking for Multimedia

I. J. Cox, J. Kilian, F. T. Leighton, T. Shamoon (1997). *Secure Spread Spectrum Watermarking for Multimedia*. [DOI: 10.1109/83.650120](https://doi.org/10.1109/83.650120).
[Published review](https://chickendays.com/cox-1997-spread-spectrum-watermarking/).

A Gaussian watermark multiplicatively changes the 1,000 largest non-DC full-image DCT coefficients at strength 0.1. Detection uses those coefficients from the pristine original and scores the recovered vector against candidate marks.

We did not find the authors' experimental code. This folder contains our re-implementation.

## What we tested

Airplane and Boat at 256x256; seed 199712; ten cases per image and 1,000 candidates per case. Tests cover clean insertion, half-scaling, JPEG 10/5, raw and mean-centered dithering, known-coordinate cropping, JPEG plus cropping, five successive marks and averaging five separately marked copies.

## Reproduce

With Python 3.11 and the root requirements installed, run from the repository root:

```sh
python fetch_images.py
python cox-1997-spread-spectrum/reproduce.py
```

The full reproduction runs twice, verifies numeric correctness and determinism,
and compares every recorded JSON value and CSV cell exactly with `results/`.
Only working runs in `outputs/` are replaced. To run the stages individually:

```sh
python cox-1997-spread-spectrum/run_pair.py pair_001
python cox-1997-spread-spectrum/run_pair.py pair_002
python cox-1997-spread-spectrum/verify_pair.py
```

All inputs come from root `images/`; generated images, arrays and numeric files
stay in this folder's ignored `outputs/`. Verification checks generated files.
The recorded reference is `results/pair_summary.json`. Environment-only metadata and file hashes
of excluded image artifacts are omitted from recorded results.

## Results

| **Test** | Airplane | Boat |
|---|---|---|
| **Undistorted** | 30.5 | 32.4 |
| **Half-scale** | 20.6 | 23.7 |
| **JPEG 10** | 17.1 | 18.4 |
| **JPEG 5** | 9.5 | 9.3 |
| **Dither, no mean removal** | 17.9 | 18.3 |
| **Dither, mean removed** | 18.2 | 18.7 |
| **Crop** | 13.6 | 12.2 |
| **JPEG 10 + crop** | 8.2 | 8.1 |

Clean-image PSNR is 35.90 dB for Airplane and 34.76 dB for Boat. The 20 embedded responses in the multiple-mark cases range from 12.8 to 14.9 after rounding to one decimal. Both pair runs contain 20,000 detector responses.

The displayed precision matches the published review. Full-precision measurements
and all tested settings are in `results/`; comparisons use the full values.

## Conditions and limits

The authors' detector requires the original unmarked image. It also requires registration: geometric changes must be inverted before extraction. Our scale and crop tests use known dimensions and coordinates. We did not measure automatic registration or resilience to residual misalignment.

We did not run the authors' print-xerox-scan test. We also ran no rotation or translation sweeps. Our scope is 2 images and one master seed.
