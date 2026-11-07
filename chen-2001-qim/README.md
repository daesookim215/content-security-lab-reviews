# Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding

B. Chen, G. W. Wornell (2001). *Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding*. [DOI: 10.1109/18.923725](https://doi.org/10.1109/18.923725).
[Published review](https://chickendays.com/chen-2001-qim/).

Dithered scalar quantization embeds one bit in DCT coefficient (2, 3) of each 8x8 block. Two offset quantizers represent the bit values; decoding chooses the nearer quantizer without the original image.

We did not find the authors' experimental code. This folder contains our re-implementation.

## What we tested

Airplane and Boat at 512x512; seed 2001; 4,096 uncoded bits; steps 8, 16, 32, 64; clean images, JPEG qualities 90/70/50/30 and additive-noise standard deviations 1/3/6. An independent cosine projection and parity decoder check all 64 cases.

## Reproduce

With Python 3.11 and the root requirements installed, run from the repository root:

```sh
python fetch_images.py
python chen-2001-qim/reproduce.py
```

The full reproduction runs twice, verifies numeric correctness and determinism,
and compares every recorded JSON value and CSV cell exactly with `results/`.
Only working runs in `outputs/` are replaced. To run the stages individually:

```sh
python chen-2001-qim/run.py run1
python chen-2001-qim/run.py run2
python chen-2001-qim/verify.py
```

All inputs come from root `images/`; generated images, arrays and numeric files
stay in this folder's ignored `outputs/`. Verification checks generated files.
The recorded reference is `results/run1/results.json`. Environment-only metadata and file hashes
of excluded image artifacts are omitted from recorded results.

## Results

| **Image / step** | PSNR dB | JPEG BER | Noise BER |
|---|---|---|---|
| **Airplane / 8** | 57.99 | 53.12% | 46.66% |
| **Airplane / 16** | 52.15 | 45.24% | 19.90% |
| **Airplane / 32** | 46.62 | 0.54% | 0.71% |
| **Airplane / 64** | 40.72 | 0.00% | 0.00% |
| **Boat / 8** | 57.97 | 54.03% | 48.66% |
| **Boat / 16** | 52.29 | 44.41% | 21.34% |
| **Boat / 32** | 46.69 | 0.49% | 1.12% |
| **Boat / 64** | 40.86 | 0.00% | 0.00% |

The displayed precision matches the published review. Full-precision measurements
and all tested settings are in `results/`; comparisons use the full values.

## Conditions and limits

We apply an orthonormal DCT, which preserves total squared error before pixel rounding. Our implementation embeds one bit in coefficient `(2, 3)` of each 8x8 block, counting indices from zero. The decoder shares the offsets, coefficient position, block alignment, and step size. The reference bits score BER; the original image is used for PSNR alone.

These results cover two grayscale images and uncoded messages. We rounded reconstructed pixels to integers and clipped them to the brightness range before saving and attacking them. We tested JPEG and additive noise, with no resizing, rotation, cropping, print-and-scan processing, error correction, distortion compensation, or deliberately targeted removal. PSNR describes pixel error rather than a human visibility judgment.
