# StegaStamp SavedModel image tests

CPU inference for the [StegaStamp review](https://chickendays.com/tancik-2020-stegastamp/), using the released encoder-decoder SavedModel via **community mirror Gu-f; original identity unverified**. Its graph, variable shard and index are byte-identical to the second mirror ytfrdfiw (3/3 files). Agreement between the mirrors does not authenticate the authors' original release.

This folder contains scripts and numeric records for quality, separate image changes, combined changes and network measurements. It contains no images, weights or paper PDFs. The detector, training critic and physical print/camera pipeline are outside these tests.

## Reproduction

Run on Linux x86_64 with micromamba or conda and `curl` installed:

```sh
sh reproduce.sh
```

The script creates `env/` with `python=3.7` using micromamba, or conda if micromamba is unavailable. It installs the exact packages in [results/pip_freeze.txt](results/pip_freeze.txt), including:

| Package | Version |
|---|---|
| TensorFlow (CPU) | 1.15.5 |
| NumPy | 1.18.5 |
| protobuf | 3.20.3 |
| Pillow | 9.5.0 |
| bchlib | 0.14.0 |
| scikit-image | 0.19.3 |
| opencv-python-headless | 4.5.5.64 |

The recorded Python version is 3.7.12. `CUDA_VISIBLE_DEVICES=""` and the TensorFlow session's GPU count of zero select CPU inference. TensorFlow uses eight intra-op threads and one inter-op thread; OMP, OpenBLAS, MKL and NumExpr limits are eight. OpenCV uses one thread for the A/B/C experiment. The recorded environment is in [results/environment.json](results/environment.json); its environment prefix is relative.

The script downloads and verifies the pinned official source archive, both input images and all six mirror files before inference. Downloads stay in ignored `.cache/`. It measures the network in `run-output/network/`, then runs the complete A/B/C experiment in two separate processes into ignored `run-output/run1/` and `run-output/run2/`. It compares the outputs and references, prints PASS or FAIL, and exits nonzero on failure. Existing downloads are rechecked; a checksum mismatch fails. Repeating the script replaces the named generated output files. Reference `results/` files are read-only inputs to the scripts.

Individual commands, after the environment and downloads are prepared:

```sh
env/bin/python code/measure_networks.py
env/bin/python code/run_experiment.py run1
env/bin/python code/run_experiment.py run2
env/bin/python code/compare_runs.py
env/bin/python code/report.py
```

## Weights and source

M1 denotes Gu-f and M3 denotes ytfrdfiw in the recorded files.

| File / pinned URL | Bytes | SHA-256 |
|---|---:|---|
| [weights/m1/saved_models/stegastamp_pretrained/saved_model.pb](https://media.githubusercontent.com/media/Gu-f/StegaStamp/d95ea307c73410c96557200656d2e0a59574eedb/saved_models/stegastamp_pretrained/saved_model.pb) | 241,765 | `42c4be663021a1d98f82eed425ac6643aa149d57e8bb5a60be2247e19937f095` |
| [weights/m1/saved_models/stegastamp_pretrained/variables/variables.data-00000-of-00001](https://media.githubusercontent.com/media/Gu-f/StegaStamp/d95ea307c73410c96557200656d2e0a59574eedb/saved_models/stegastamp_pretrained/variables/variables.data-00000-of-00001) | 217,018,916 | `1ca5cd940906f57454fcf3d05d4f81edd87066903f7ced2ba57399614bca3bf0` |
| [weights/m1/saved_models/stegastamp_pretrained/variables/variables.index](https://media.githubusercontent.com/media/Gu-f/StegaStamp/d95ea307c73410c96557200656d2e0a59574eedb/saved_models/stegastamp_pretrained/variables/variables.index) | 2,416 | `8c038f8fa37dba430c59dd91a654f2aedffff27acf0efed657d2d15f46d5149c` |
| [weights/m3/stegastamp_pretrained.zip.001](https://raw.githubusercontent.com/ytfrdfiw/StegaStamp/84e34ef46e27b5fe847d474853a133b730a1ec03/saved_models/stegastamp_pretrained.zip.001) | 100,000,000 | `b01fc5b39042965fccfac8149539e7fccca115ae565f2c426ef45e2d531cdb05` |
| [weights/m3/stegastamp_pretrained.zip.002](https://raw.githubusercontent.com/ytfrdfiw/StegaStamp/84e34ef46e27b5fe847d474853a133b730a1ec03/saved_models/stegastamp_pretrained.zip.002) | 100,000,000 | `23f3054b3cb76337d2c9748ff2410aba09c282c370eadaf1d2df24aa88a1462b` |
| [weights/m3/stegastamp_pretrained.zip.003](https://raw.githubusercontent.com/ytfrdfiw/StegaStamp/84e34ef46e27b5fe847d474853a133b730a1ec03/saved_models/stegastamp_pretrained.zip.003) | 17,264,075 | `0a48d3c68f5836e181b01d828abe28e20a716e9153593dce226b65c80e10bdab` |

The M3 parts are concatenated in 001/002/003 order: 217,264,075 bytes, SHA-256 `cd4aeb5dcd7c97be8d0907143d7e5103ea9f3b9e509490f0cf20f67bf6d8bcde`. Archive members are checked before extraction, then all three extracted model files are checked against M1 by SHA-256 and byte comparison. [results/weights_provenance.json](results/weights_provenance.json) retains the URLs, sizes, hashes and cross-check; [results/archive_listing.json](results/archive_listing.json) records the ZIP members.

Official source is pinned to [tancik/StegaStamp, commit c984446048b826587ca1875027b0a1dc1885fb30](https://github.com/tancik/StegaStamp/tree/c984446048b826587ca1875027b0a1dc1885fb30), MIT license. The [source archive](https://api.github.com/repos/tancik/StegaStamp/tarball/c984446048b826587ca1875027b0a1dc1885fb30) is 213,307 bytes, SHA-256 `70bb2c5a83e52abfdc6c6e82df1f64357e4e50d4db7f75ecbfd36e81e675ed4f`. [code/source_manifest.json](code/source_manifest.json) pins the archive and selected source-file hashes. Only the selected source files and license are extracted. The combined chain imports the unchanged official `utils.py` JPEG surrogate; inference loads the SavedModel with `tf.compat.v1.saved_model.loader.load`, tag `serve`, signature `serving_default`.

## Inputs and setup

| Image | Source | Original SHA-256 |
|---|---|---|
| Airplane | [USC-SIPI 4.2.05.tiff](https://sipi.usc.edu/database/misc/4.2.05.tiff) | `ae9eba12b80df955e47585bf4a16332ea0dadc753668d3457c509564934dbd40` |
| Boat | [USC-SIPI boat.512.tiff](https://sipi.usc.edu/database/misc/boat.512.tiff) | `6bc7319b1e3b44ce0b2a571ac39a28bac94437d812057d1afb2cf78ea3a9a35f` |

USC-SIPI lists the images' source and copyright status as unknown in its [copyright notes](https://sipi.usc.edu/database/copyright.php). Images are downloaded for local testing and are not redistributed here. The input URLs and hashes are also in [code/input_manifest.json](code/input_manifest.json).

Both images are converted to RGB and fitted to 400×400 with `ImageOps.fit(..., (400, 400))`, using Pillow's default bicubic sampling, then converted to float32 / 255. Boat's grayscale values are repeated in RGB. The secret is `Stega!!`; BCH(137,5) produces 56 data bits + 40 ECC bits + four zero padding bits = 100 network bits. The ECC bytes are `88a9fb4e40`. The seed is 20231130. The full packet and operation order are in [results/run1/settings.json](results/run1/settings.json).

The encoded image uses the official PNG pixel conversion: clip to [0,1], multiply by 255, truncate to uint8, then use float32 / 255 for decoding and changes. The harness uses those pixel arrays directly and saves NumPy arrays. Raw recovery counts compare all 100 bits, including padding. BCH uses the first 96 bits; exact recovery requires successful BCH decoding and the text `Stega!!`.

## A: quality

RGB PSNR and SSIM compare the encoded uint8 image with the fitted input. PSNR uses data_range=1. SSIM uses scikit-image 0.19.3, channel_axis=2, a 7×7 uniform window, sample covariance and the RGB-channel mean. Float-output metrics, residual and pixel-delta statistics, clipping counts and array hashes are also retained in [results/run1/quality.json](results/run1/quality.json).

| Image | PSNR (dB) | SSIM |
|---|---:|---:|
| airplane | 28.947071 | 0.911483 |
| boat | 28.955576 | 0.907496 |

## B: separate image changes

Every case starts from the encoded uint8 image. The clean case decodes the unchanged image.

| Change family | Parameters |
|---|---|
| JPEG | Pillow quality 90, 70, 50, 30; 4:2:0; optimize=False |
| Gaussian blur | OpenCV sigma 1 or 2 pixels; automatic kernel; reflect101 borders |
| Gaussian noise | Standard deviation 0.02 or 0.05; float32; clip [0,1]; seed = 20231130 + int(std × 10000) |
| Brightness | Pillow uint8 ImageEnhance factors 0.7, 1.3 |
| Contrast | Pillow uint8 ImageEnhance factors 0.5, 1.5 |
| Saturation | Pillow uint8 ImageEnhance.Color factors 0, 1.5 |
| Hue | Float32 OpenCV HSV; +0.05 turn (+18 degrees), wrap at 360 |
| Resize | Lanczos 400→200→400 |
| Rotation | Pillow bicubic, 5 or 10 degrees counterclockwise; fixed 400×400 canvas; black fill |
| Perspective | OpenCV bilinear; white fill; source corners [(0,0),(399,0),(399,399),(0,399)] to [(20,15),(380,0),(399,380),(10,399)] |
| Crop | Center box [20,20,380,380], retaining 90% of each side; Lanczos resize to 400×400 |
| Flip | Horizontal |

Exact recovery below means the full secret after BCH. Per-case parameters, decoded bit strings, BCH corrections and changed-tensor hashes are in [results/run1/single_attacks.json](results/run1/single_attacks.json).

| Change | Airplane bits/100; exact recovery | Boat bits/100; exact recovery |
|---|---|---|
| clean | 100; yes | 100; yes |
| jpeg_90 | 100; yes | 100; yes |
| jpeg_70 | 100; yes | 100; yes |
| jpeg_50 | 100; yes | 100; yes |
| jpeg_30 | 100; yes | 100; yes |
| blur_1 | 100; yes | 100; yes |
| blur_2 | 100; yes | 100; yes |
| noise_0.02 | 100; yes | 100; yes |
| noise_0.05 | 100; yes | 100; yes |
| brightness_0.7 | 100; yes | 100; yes |
| brightness_1.3 | 99; yes | 100; yes |
| contrast_0.5 | 100; yes | 100; yes |
| contrast_1.5 | 100; yes | 100; yes |
| saturation_0 | 100; yes | 100; yes |
| saturation_1.5 | 100; yes | 100; yes |
| hue_0.05 | 100; yes | 100; yes |
| resize_0.5 | 100; yes | 100; yes |
| rotation_5 | 92; no | 90; no |
| rotation_10 | 63; no | 56; no |
| perspective_moderate | 100; yes | 100; yes |
| crop_0.9 | 88; no | 100; yes |
| horizontal_flip | 55; no | 57; no |

## C: combined changes

Twenty draws per image apply perspective → blur → noise and clipping → contrast plus RGB/brightness offsets and clipping → desaturation → the official differentiable JPEG surrogate.

| Operation | Sampling and boundary settings |
|---|---|
| Perspective | bound = floor(400 × 0.1 × U[0,1)); four corner offsets independently U[-bound,+bound], with corner coordinates 0/400; inverse homography; bilinear; white fill |
| Blur | 7×7 channel-diagonal normalized kernel; 25% Gaussian sigma U[1,3], 25% motion, 50% identity; SAME zero padding |
| Motion kernel | Angle U[0,2π), transverse sigma U[0.25,1], Manhattan threshold U[3,3.1] |
| Noise | Independent Gaussian pixels, standard deviation U[0,0.02]; clip [0,1] |
| Color | Contrast U[0.5,1.5]; independent RGB offsets U[-0.1,0.1] plus shared brightness U[-0.3,0.3]; clip [0,1] |
| Desaturation | Interpolation U[0,1] toward luminance with weights [0.3,0.6,0.1] |
| JPEG surrogate | Quality U[50,100]; factor = (200 − 2q)/100 + 0.0001; round_only_at_0; 4:2:0 |

The noise maximum **0.02 is the code's setting** in `train.py`; paper Section 3.4 states 0.2. These draws sample the full configured ranges without curriculum reduction. This post-encoding chain directly warps the saved encoded pixels; training constructs the encoded image in a warped coordinate frame and inverse-warps it. NumPy RandomState supplies explicit draws rather than TensorFlow random operators. The surrogate uses `utils.jpeg_compress_decompress` and is separate from the real JPEG cases in B.

Airplane draw seeds are 20231130 + the zero-based draw index; Boat adds 1000. [results/run1/combined_draws.json](results/run1/combined_draws.json) records each seed, transform matrix, scalar parameter, kernel/noise hash, changed-tensor hash and decoding result.

| Image | Draws | Exact recovery | Mean bits/100 | Minimum bits/100 |
|---|---:|---:|---:|---:|
| airplane | 20 | 17/20 | 98.70 | 91 |
| boat | 20 | 20/20 | 99.90 | 99 |

## Network measurements

The deployment graph has 60 variables. Measurements fetch 128 finite intermediate float tensors using one fitted Airplane input and the fixed packet. Counts exclude the separate detector and training critic.

| Module | Parameters |
|---|---:|
| Secret path | 757,500 |
| Encoder U-Net / residual | 991,747 |
| Decoder STN | 41,054,150 |
| Decoder bit head | 11,451,332 |
| Total | 54,254,729 |

The secret path projects 100 bits to 7,500 values, reshapes to 50×50×3 and upsamples to 400×400×3. The encoder receives 400×400×6 and outputs an RGB residual. STN localization flattens 50×50×128 to 320,000 values and predicts six affine values. The bit head reaches 13×13×128, flattens to 21,632, then maps through 512 values to 100 bits.

[results/network_layers.csv](results/network_layers.csv), [results/shapes_params.json](results/shapes_params.json) and [results/signatures.json](results/signatures.json) contain the layer counts, shapes, graph attributes and serving tensors. [results/graph.pbtxt](results/graph.pbtxt) and [results/graph_ops.json](results/graph_ops.json) retain the graph and operation list.

The export follows conv10→residual; pinned `models.py` computes conv10 but returns residual(conv9). The export samples uncentered input in its STN and subtracts 0.5 afterward; the pinned source subtracts 0.5 before sampling. The measured decoder flatten size is 21,632; supplement Table 6 lists 20,000. These measurements describe the saved export.

## Determinism and runtime

The recorded run1 and run2 have **15/15 byte-identical outputs**, excluding `timing.json`: nine CSV/JSON result/settings files and six NumPy arrays. [results/determinism.json](results/determinism.json) retains both SHA-256 values and byte-comparison verdicts for every file. All nine deterministic numeric/settings files are distributed under `results/run1/`; the arrays are omitted and their hashes remain in the determinism record. The recorded run1 timing is also included.

`code/compare_runs.py` compares all 15 fresh run1/run2 files byte for byte, compares both runs' nine numeric/settings files byte for byte against `results/run1/`, and checks all 15 against the recorded hashes. It also compares regenerated layer measurements, shapes, signatures, weight provenance and archive listing with their reference files. Timings are excluded. A difference prints FAIL and returns a nonzero exit code. Exact floating-point and JPEG results can depend on the CPU and numerical/image libraries.

| Runtime | Run1 seconds | Run2 seconds |
|---|---:|---:|
| Model + attack graph loading | 0.233683 | 0.222793 |
| Encoding, attacks, decoding | 3.766951 | 3.538340 |
| Script total including writes | 4.013065 | 3.771998 |

Network measurement: 5.187507 seconds.

These script timers exclude interpreter imports/startup and downloads; network measurement includes repeated intermediate fetches and graph dumps. They are not isolated inference-latency measurements. The results cover two fixed images and one fixed secret.

Paper: Matthew Tancik, Ben Mildenhall and Ren Ng, [StegaStamp: Invisible Hyperlinks in Physical Photographs, arXiv:1904.05343v2](https://arxiv.org/abs/1904.05343v2), 2020.

Review: [chickendays.com/tancik-2020-stegastamp](https://chickendays.com/tancik-2020-stegastamp/).
