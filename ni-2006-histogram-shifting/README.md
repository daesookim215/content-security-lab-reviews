# Reversible Data Hiding

Z. Ni, Y.-Q. Shi, N. Ansari, W. Su (2006). *Reversible Data Hiding*. [DOI: 10.1109/TCSVT.2006.869964](https://doi.org/10.1109/TCSVT.2006.869964).
[Published review](https://chickendays.com/ni-2006-histogram-shifting/).

Histogram peaks carry bits by shifting disjoint brightness intervals toward empty or minimally populated bins. Positions at occupied minima are recorded inside the stream; pair parameters and occupied-minimum flags are supplied separately.

We did not find the authors' experimental code. This folder contains our re-implementation.

## What we tested

Airplane and Boat at 512x512, one and two disjoint peak/minimum pairs, with every selected peak filled using seed 200603. Metrics account for occupied-minimum position records; a separate scalar decoder checks exact pixels and message bits.

## Reproduce

With Python 3.11 and the root requirements installed, run from the repository root:

```sh
python fetch_images.py
python ni-2006-histogram-shifting/reproduce.py
```

The full reproduction runs twice, verifies numeric correctness and determinism,
and compares every recorded JSON value and CSV cell exactly with `results/`.
Only working runs in `outputs/` are replaced. To run the stages individually:

```sh
python ni-2006-histogram-shifting/run.py run1
python ni-2006-histogram-shifting/run.py run2
python ni-2006-histogram-shifting/verify.py
```

All inputs come from root `images/`; generated images, arrays and numeric files
stay in this folder's ignored `outputs/`. Verification checks generated files.
The recorded reference is `results/run1/results.json`. Environment-only metadata and file hashes
of excluded image artifacts are omitted from recorded results.

## Results

| Case | Net bits | PSNR |
|---|---|---|
| **Airplane, 1 pair** | 8,312 | 53.93 dB |
| **Airplane, 2 pairs** | 16,040 | 48.27 dB |
| **Boat, 1 pair** | 5,796 | 51.97 dB |
| **Boat, 2 pairs** | 11,392 | 48.23 dB |

The displayed precision matches the published review. Full-precision measurements
and all tested settings are in `results/`; comparisons use the full values.

## Conditions and limits

Recovery requires the marked pixel values to remain intact. The decoder also needs the pair parameters and the same scan order. An occupied minimum needs its saved pixel positions. Visual similarity alone does not establish that these requirements hold.

Our test covers these two images, full use of the selected peaks, and one random seed. We supplied pair parameters separately and used a fixed encoding for position records. We tested no compression, cropping, noise, or other attacks. The restoration result applies to the losslessly saved marked images. We did not test a complete authentication system or protection against message tampering.
