# Reversible Data Embedding Using a Difference Expansion

J. Tian (2003). *Reversible Data Embedding Using a Difference Expansion*. [DOI: 10.1109/TCSVT.2003.815962](https://doi.org/10.1109/TCSVT.2003.815962).
[Published review](https://chickendays.com/tian-2003-difference-expansion/).

Horizontal pixel pairs retain their integer mean while selected signed differences are doubled and receive one message bit. A compressed location map, saved low bits, length framing and an unkeyed SHA-256 digest allow exact extraction and restoration.

We did not find the author's experimental code. This folder contains our re-implementation.

## What we tested

Airplane and Boat at 512x512, one horizontal pass and payloads 16,384, 32,768, 65,536 and 98,304 bits. Each case uses the first sufficient integer threshold, saves a lossless marked image and checks zero pixel and bit errors through a separate scalar decoder.

## Reproduce

With Python 3.11 and the root requirements installed, run from the repository root:

```sh
python fetch_images.py
python tian-2003-difference-expansion/reproduce.py
```

The full reproduction runs twice, verifies numeric correctness and determinism,
and compares every recorded JSON value and CSV cell exactly with `results/`.
Only working runs in `outputs/` are replaced. To run the stages individually:

```sh
python tian-2003-difference-expansion/run.py run-a
python tian-2003-difference-expansion/run.py run-b
python tian-2003-difference-expansion/verify.py
```

All inputs come from root `images/`; generated images, arrays and numeric files
stay in this folder's ignored `outputs/`. Verification checks generated files.
The recorded reference is `results/run-a/results.json`. Environment-only metadata and file hashes
of excluded image artifacts are omitted from recorded results.

## Results

| **User bits** | bpp | Airplane | Boat |
|---|---|---|---|
| **16,384** | 0.0625 | 45.38 | 38.35 |
| **32,768** | 0.1250 | 44.15 | 37.65 |
| **65,536** | 0.2500 | 41.03 | 35.88 |
| **98,304** | 0.3750 | 37.02 | 33.43 |

The displayed precision matches the published review. Full-precision measurements
and all tested settings are in `results/`; comparisons use the full values.

## Conditions and limits

The decoder needs the same pairing and reading order, with encoded pixel values intact. It reads the recovery map from the image. Our test saved embedded images as lossless PNGs and applied no lossy compression, resizing, cropping, or other image changes.

We use horizontal pairing, one embedding pass, and map compression that records lengths of repeated map values. We keep every pair selected at the first sufficient integer threshold, without fine-tuning individual selections, and fill unused space with zeros. SHA-256 produces a 256-bit fingerprint from pixel bytes without a secret key; it does not identify the creator. We did not test authentication using a secret key, encryption to conceal embedded bits, additional embedding passes, or preservation of appearance after halving both image dimensions.
