# TrustMark: two-image recovery experiment

Test of the authors' released TrustMark 0.9.2 Q encoder/decoder. Paper: Tu Bui, Shruti Agarwal, and John Collomosse (2023), *TrustMark: Universal Watermarking for Arbitrary Resolution Images*, https://arxiv.org/abs/2311.18297. Official code and model license: MIT, https://github.com/adobe/trustmark.

Run `sh reproduce.sh` in a new checkout with Python 3.12 and the standard `patch` command on Linux. `PYTHON` can select another Python 3.12 executable. The script creates its own environment, installs the package from PyPI and CPU Torch from its official index, applies the BCH compatibility patch described below, downloads the two USC-SIPI inputs with SHA-256 checks, and obtains the official Q checkpoints. No account or paid service is required. Existing reference results are retained. New outputs go into `reproduced/run1` and `reproduced/run2`; choose a fresh output directory to repeat again.

The tested 0.9.2 installation already included BCH fixes. `bch_compatibility.patch` reproduces them exactly: it makes received parity mutable, validates error-root counts and corrected parity, and rejects invalid zero-padding. `apply_bch_compat.py` checks the two original files against PyPI hashes before patching and checks the results against the measured installation's hashes. The encoder, decoder, and main TrustMark API are byte-identical to PyPI 0.9.2, and the released Q weights retain their official checksums. The patch and upstream source use the accompanying MIT license.

The fixed 61-bit message is `1011001010010110011100010100101011010001110001101010010110101`. BCH_5 adds 35 parity and four schema bits, making 100 network bits. Seed 20231130 initializes Python, NumPy, and Torch. CPU uses eight intra-op threads and one inter-op thread. Reference versions: Python 3.12.3, Torch 2.5.1+cpu, NumPy 2.2.6, Pillow 11.3.0. New library versions and CPU platforms may change exact pixels; matching the two fresh runs is the determinism check.

Airplane remains RGB. Boat repeats its grayscale values in three channels. Both are marked at 512×512 and at 2048×2048 after Lanczos enlargement. Changes are separate: unchanged; JPEG quality 90, 70, 50 (4:2:0); direct decoding at half size; Gaussian blur radius 2; brightness factor 1.5; centered 80%-side crop rounded to whole pixels. Encoder strength and bilinear residual merge retain package defaults. Remover and bounding-box models are not loaded.

`bit_accuracy` is the fraction of the first 61 raw network bits matching the payload, before BCH. `exact_payload` compares the public API's post-correction output with the full message. `detected` is the API flag. `candidate_payload` retains the lower-level attempted BCH output, including unsuccessful cases. Failed public decoding returns an empty string. `quality.csv` records PSNR against the prepared input at full marking resolution. PSNR uses the mean squared error across all RGB values and a peak of 255.

| Image | Size | PSNR, dB | Exact cases |
|---|---|---|---|
| Airplane | 512 | 41.04 | 7/8 |
| Airplane | 2048 | 40.97 | 7/8 |
| Boat | 512 | 43.32 | 8/8 |
| Boat | 2048 | 43.26 | 8/8 |

Airplane's brightness case fails at both sizes, with 50/61 raw payload bits correct. Boat recovers exactly in every tested case. Both unmarked inputs return `detected=false` and an empty public payload at both sizes. The two reference result sets are identical, including saved-image hashes. JSON includes source and model hashes. The folder distributes numeric records and scripts; generated images are created only when reproducing.

The resolution-scaling example uses Boat at 2048×2048 with the same seed, payload, Q model, and settings. After `reproduce.sh`, run `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python capture_example.py reproduced/run1 reproduced/example`. This checks the recorded source and model hashes, recomputes encoding, requires every marked pixel to match the saved run, and saves the normalized working cover, clipped prediction, channel means, 256×256 mean-adjusted residual, bilinearly resized 2048×2048 residual, full cover, and marked arrays as `.npy`. Array layout is height, width, RGB channel; residuals use normalized encoder units. Multiply by 127.5 for 8-bit pixel units. No supplied package or model file is modified.

`results/example_values.json` records the figure's numeric values, coordinates, ranges, labels, and example paragraph. `results/example_capture.json` records settings and array SHA-256 checksums. At zero-based red-channel pixel (904, 993), the input is 155 and the normalized residual is 0.0087890625. The residual in pixel units is exactly 1.12060546875; adding it gives 156.12060546875, then the release clips and truncates to 156. The 256×256 residual range across RGB is -0.1193251833319664 to 0.1372978836297989. Its 8×8 scale footprint blends neighboring samples; it does not replicate a constant residual into a block. Reference intermediate arrays stay with the local experiment and are generated by the script; this public folder contains no images.
