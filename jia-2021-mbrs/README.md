# MBRS review reproduction

CPU measurements of Jia et al. (2021) official MBRS models, source commit `2bab0df66480f66cfdef534d38c851354ee2c6aa`.

This folder contains only our scripts and numeric/architecture records. It contains no official source files, paper files, images or weights. The official repository states no license. Source and model files are obtained separately; the upstream authors retain their rights.

## Offline execution

Use Python 3.11 and the existing pinned dependencies, or `uv run` where dependency resolution is available:

```bash
uv run reproduce.py --local-inputs /absolute/path/to/jia-2021-mbrs
```

The input root must contain `code/MBRS/` (the official source at the pinned commit), `weights/SHA256SUMS`, `weights/MBRS_256_m256/{EC_42.pth,D_42.pth}`, `weights/MBRS_Diffusion_128_m30/{EC_114.pth,D_114.pth}`, and `data/{4.2.05.tiff,boat.512.tiff}`.

`reproduce.py` runs the experiment twice, compares all six output files with each other and with the packaged run1 results. CSV files and numeric JSON are compared byte-for-byte. Only `environment.python` and `environment.packages` version fields in settings.json are excluded; platform remains part of the comparison; seeds, messages, transforms and all numbers must match exactly. Hardware/library floating-point differences cause a reported mismatch rather than an implicit tolerance.

Compare an existing run without recomputation:

```bash
python reproduce.py --check-results /path/to/experiment/results/run1
```

## Download mode

```bash
uv run reproduce.py
```

Downloads use stdlib urllib and stay in the selected cache. The official source
archive for the pinned commit, the four Google Drive weight files and the two
USC-SIPI images each must match a recorded SHA-256. The source archive is
extracted only after its hash matches. A Drive error page or changed file fails
the hash check.

## Protocol

- CPU float32, evaluation mode, deterministic algorithms, two intra-op threads and one inter-op thread. Base seed 20261126; model/image offsets 0,1,100,101.
- Official `EncoderDecoder` or `EncoderDecoder_Diffusion` and `Discriminator` load with strict=True. EC is encoder+decoder; D is discriminator. All official network and selected noise APIs work unchanged; api_adaptations.json records an empty patch list.
- Official test preprocessing: RGB conversion, bilinear resize to 110% of target (281 or 140), seeded random crop (256 or 128), ToTensor and Normalize(.5,.5). Boat's grayscale is repeated into RGB by conversion. Crop coordinates and complete binary messages are in settings.json.
- Encoder output strength=1. Clamp [-1,1], scale to [0,255], nearest rounding (ties to even), uint8 PNG, reopen, then decode. Every edit starts from that PNG. Bits are scores strictly greater than .5.
- Real JPEG via Pillow, qualities 90/70/50/30/10, subsampling=2 (4:2:0). Official GF(2) with7×7 kernel, Crop(.19,.19), Cropout(sqrt(.3),sqrt(.3)) and Dropout(.3) cover Table2's types/strengths. Additional GN(.01), MF(3), SP(.01) strengths are ours because the table supplies none. All are applied to both models as diagnostic inputs.
- Official Crop preserves its original canvas and fills with normalized zero (midgray); paper text says white. Cropout restores cover outside retained rectangle. Dropout retains encoded per channel element with probability .3. SP writes normalized0 or1. GN is unbounded Gaussian noise in [-1,1] units.
- Resize uses antialiased bilinear half-side then restoration to required input dimensions. Official non-JPEG noise outputs remain float32 at decode; PNG attack previews are for viewing, not reread as another quantization stage.
- PSNR and SSIM compare received RGB to the preprocessed cover, data_range=1. SSIM is scikit-image uniform7×7, sample covariance. Original-cover PSNR is mathematically infinite and represented as JSON null/blank CSV; SSIM=1. Our reported SSIM differs from the deprecated official metric harness.
- The deterministic run1/run2 equality receipt is measurement_checks.json. Architecture records include all checkpoint keys/shapes, hooked module shapes, parameter counts and full module trees.

## Files

`experiment/run_test.py` owns the run; `common.py` owns provenance/loading; `attacks.py` owns image conversion and independent edits. `inspect_models.py` records official loaded structures. `results/run1` and `results/run2` contain results.json, measurements.csv, quality.json/csv, settings.json and example_values.json. No source file in this package is copied from the official MBRS repository.
