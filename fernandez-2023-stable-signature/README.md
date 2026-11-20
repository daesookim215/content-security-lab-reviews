# Stable Signature: released-decoder VAE reconstruction test

CPU inference code and numeric results for the Stable Signature review. Images,
weights and paper files are not included in this folder.

```sh
uv run reproduce.py
```

The runner downloads the two Meta checkpoints, the SD 2.1 base VAE component from
`sd2-community/stable-diffusion-2-1-base` at revision
`4e63672c03103b6c636b8fb4119ba982469b2955`, and Airplane/Boat from USC-SIPI.
Every file must match its recorded SHA-256. The original Stability AI host returned
HTTP 401 on 2026-11-19. Downloads use `.cache/`.
Newly computed images and results use `run-output/`. These paths are ignored.
For already acquired inputs, use `python reproduce.py --local-inputs PATH --output PATH`.
The offline branch performs no network requests.

The inputs are photographs passed through the VAE, not text-generated samples.
Boat gray values are repeated in RGB. Both decoders receive the same encoder
posterior mode; no diffusion scale is applied. The official CompVis KL-f8
architecture is loaded directly from its pinned source definitions. Mirror state
names are converted bijectively; attention matrices acquire singleton convolution
axes. All mirror tensors are consumed. Released decoder and post-quantization
weights load strictly. `experiment/mirror_check.json` records every tensor's
relative L2 difference and the key conversion. Architecture compatibility does
not establish identity with the authors' plain baseline.

Inference is CPU float32, two threads, batch one, evaluation mode, deterministic
algorithms, seed 230315435. The whitened TorchScript extractor uses ImageNet
normalization and score >0 for a one bit. The fixed key is
`111010110101000001010111010011010100010000100111`.

`results/results.json` records 52 received-image rows, 48 scores and bits per row,
saved-image hashes, dimensions, transformations and threshold comparisons.
`quality.json` records RGB PSNR and SSIM for marked/plain, marked/original and
plain/original pairs. PSNR uses peak 255 and all RGB bytes. Crops and resizes are
decoded at native dimensions; their full-original PSNR is inapplicable. Infinite
PSNR for identical pixels and inapplicable PSNR are JSON null, distinguished by
attack and dimensions. Same-sized rotation PSNR uses the unrotated original as a direct pixel reference. `example_values.json` contains the preselected Boat
red-channel block y252:256,x252:256, raw floats, bytes, residuals and clean scores.
Every attack starts independently from the reopened watermarked PNG.

Crop and resize parameters are retained area. Side length is floor(512√area),
matching the official code. Resizing uses Pillow bicubic. JPEG uses 4:2:0;
brightness, contrast, saturation and sharpness use Pillow ImageEnhance.
Gaussian blur uses float32 normalized kernels, radius ceil(3σ), reflection
padding and round-to-even serialization. Text is a fixed local Pillow overlay,
rather than the paper's randomized AugLy overlay. Rotation and Gaussian blur
are additional transformations. The main extractor is used throughout.

The primary paper count rules are M≥41 for detection and M≥44 for a 1,000-user
registry at nominal global FPR 1e-6. Equation 1 and the notebook's inclusive
binomial tail support this interpretation. Equation 2 prints P(M>τ); strict
alternative flags are also retained. The original evaluation comparator is
uncertain. Two photographs provide counts, not a population FPR estimate.

The reproduction command compares result JSON/CSV files byte for byte. It parses
settings and ignores only `environment.Python` and `environment.torch`; all other
settings, versions and numeric values must match exactly. Timing is a separate
diagnostic, not a deterministic result. No numeric tolerances are applied.
Different CPU backends can cause an exact-comparison failure; the report preserves
that failure.

Sources and terms:

- Fernandez et al. (2023), [paper v2](https://arxiv.org/abs/2303.15435v2).
- [Official code](https://github.com/facebookresearch/stable_signature/tree/c91217c06e953a4eaf270ef9de4a3939bcea3366),
  commit `c91217c06e953a4eaf270ef9de4a3939bcea3366`. The repository states CC BY-NC
  4.0 with MIT portions in `src/ldm` and `src/taming`. The included unmodified
  `src/ldm/modules/diffusionmodules/model.py` is one of those MIT portions.
  The repository license text is retained in `code/stable_signature/LICENSE`.
- [Meta extractor](https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt)
  and [decoder](https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth),
  linked from the official README. No separate per-file weight terms were supplied
  in the local material; no MIT license for the Meta checkpoints is inferred.
- [Mirror](https://huggingface.co/sd2-community/stable-diffusion-2-1-base/tree/4e63672c03103b6c636b8fb4119ba982469b2955),
  model metadata license `openrail++`. Applicable base-model restrictions remain.
- [USC-SIPI](https://sipi.usc.edu/database/), Airplane `4.2.05.tiff` and Boat
  `boat.512.tiff`; [source copyright conditions](https://sipi.usc.edu/database/copyright.php).
  Source-image copyright is not asserted to be public domain.
