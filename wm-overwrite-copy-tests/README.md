# Overwriting and copying open-source image watermarks

Code and our measurements for the Research Notes post
[Overwriting and Copying Open-Source Image Watermarks: TrustMark and Video Seal Tests](https://chickendays.com/wm-overwrite-copy-tests/).

The tests use the released packages: TrustMark 0.9.2 (variant Q, BCH_5, 61-bit
payload, strength 1.0) and Video Seal at commit `870ca7f` (default `videoseal`
model, 256-bit message, image mode). `configs/attenuation.yaml` is copied from
the Video Seal repository at that commit because the pip install does not
include it.

## Scenarios

`config.yaml` selects one scenario:

- `baseline`: each method embedded, then decoded after no edit, JPEG q80, JPEG q50,
  resize to 256 and back, a centre crop keeping 50% of the area, and a lossless PNG
  round trip; unmarked originals as controls.
- `overwrite_same`: TrustMark payload P1, then a second TrustMark payload P2.
- `cross_method`: TrustMark then Video Seal, and Video Seal then TrustMark.
- `copy_attack`: residual = marked minus a 5x5 Wiener or 3x3 median filtered copy,
  added to the next unmarked image at strengths 1, 2 and 3 with local
  standard-deviation weighting, then decoded with TrustMark.

Video Seal has no presence threshold in its public API. Here, Video Seal
"detected" means at least 167 of 256 decoded bits match the embedded message
(binomial upper tail at most 1e-6 for fair random bits).

## Running

Needs Python 3.12, an NVIDIA GPU with CUDA 12.8 drivers, and the first 200
images of [COCO val2017](https://cocodataset.org/) in `coco/val2017`
(not included).

```sh
bash reproduce.sh
```

`run.sh` creates a virtual environment (`.venv`, or `$VENV`), installs the pinned
requirements, and runs `main.py` on one config file.

## Results

`results/<scenario>/results.json` holds the aggregate numbers, settings and
package versions from our run on one RTX 5090. `per_image.csv` holds per-image
rows for the overwrite, cross-method and copy-attack scenarios; the baseline
folder has the aggregates only.
