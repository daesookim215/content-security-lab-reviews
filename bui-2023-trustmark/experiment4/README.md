# Released Q network measurements (experiment 4)

This experiment inventories the TrustMark 0.9.2 Q embedder, extractor and
`trustmark_rm_Q` remover. It dumps both YAML configurations, observed settings,
the complete module trees, recursive and direct parameter counts for every
module, and forward-hook input/output shapes. It runs CPU inference only, with
evaluation mode, gradients disabled, deterministic algorithms, seed 20231130,
eight intra-op threads and one inter-op thread.

The embedder receives one synthetic float32 NCHW image `[1, 3, 256, 256]`
spanning [-1, 1] and an alternating 100-bit message. Its output feeds the remover.
The extractor receives a constant RGB 256-square image through the public API's
bilinear resize to 245, followed by `SecretDecoder`'s internal resize to 224.
A second hook pass records `subimage_decode` on the same image; its stage shapes
match the direct wrapper call. These are architecture measurements, not
watermark quality, recovery, latency or training measurements. No image files
are needed or written.

## Run locally

From this directory, run:

```sh
sh reproduce_exp4.sh
```

The wrapper uses `../.venv/bin/python` from the parent reproduction. If that
environment does not exist, create it using the same package installation and
BCH compatibility patch as the parent `reproduce.sh`, without running the
image experiments. Run these commands from this directory on Linux with
Python 3.12 and the standard `patch` command:

```sh
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/.cache"
export XDG_CONFIG_HOME="$PWD/.cache/config"
export TORCH_HOME="$PWD/.cache/torch"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export PIP_CACHE_DIR="$PWD/.cache/pip"
mkdir -p "$XDG_CONFIG_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
python3.12 -m venv ../.venv
../.venv/bin/python -m pip install --upgrade pip
../.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
../.venv/bin/python -m pip install trustmark==0.9.2 numpy==2.2.6 pillow==11.3.0
../.venv/bin/python ../apply_bch_compat.py
sh reproduce_exp4.sh
```

The wrapper sets CPU/thread limits and keeps cache, configuration and temporary
files in `experiment4/.cache/`. It downloads missing files from the package's
official HTTPS model host into `.cache/models/`, checks every file against both
the pinned MD5 below and `MODEL_CHECKSUMS`, and stops on a mismatch.
The loader inherits the public baseline's checksum-only loading method and
redirects all five paths to this cache, preventing implicit package downloads.
Neither experiment2 nor experiment3 needs to be run first.

New measurements overwrite `results/config_dump.json`, `module_tree.txt`,
`shapes_params.json`, the two YAML copies, and `measurement.log`. Save a copy
of `results/` first if you want to retain the supplied capture. Checkpoint
files remain in the cache and are not distributed here.

## Summary

All shapes below use NCHW, including batch size 1; vectors use batch by features.
Multiple inputs separated by `+` are separate tensors, including decoder skip
inputs. Parameter counts exclude buffers such as BatchNorm running statistics.

| Module | Parameters | Input → output |
|---|---:|---|
| embedder (Unet1) | 8,642,179 | `[1, 3, 256, 256] + [1, 100] → [1, 3, 256, 256]` |
| extractor (SecretDecoder) | 23,712,932 | `[1, 3, 245, 245] → [1, 100]` |
| remover (WMRemover) | 38,321,379 | `[1, 3, 256, 256] → [1, 3, 256, 256]` |

The extractor's 256 → 245 public preprocessing and 245 → 224 internal
interpolation are functional operations rather than separate child modules.
The hook on `SecretDecoder` sees the 245 input; `decoder.conv1` sees 224.

## Embedder stage shapes

Taken from `results/shapes_params.json` (`hooks.embedder`). Counts are recursive;
rows overlap, so summing them would double-count parameters.

| Module | Parameters | Input → output |
|---|---:|---|
| `secret2image.dense` | 77,568 | `[1, 100] → [1, 768]` |
| `secret2image.upsample` | 0 | `[1, 3, 16, 16] → [1, 3, 256, 256]` |
| `secret2image.activ` | 0 | `[1, 3, 256, 256] → [1, 3, 256, 256]` |
| `pre` | 1,760 | `[1, 6, 256, 256] → [1, 32, 256, 256]` |
| `enc.0` | 18,496 | `[1, 32, 256, 256] → [1, 64, 128, 128]` |
| `enc.1` | 73,856 | `[1, 64, 128, 128] → [1, 128, 64, 64]` |
| `enc.2` | 295,168 | `[1, 128, 64, 64] → [1, 256, 32, 32]` |
| `enc.3` | 1,180,160 | `[1, 256, 32, 32] → [1, 512, 16, 16]` |
| `middle.0` | 4,719,616 | `[1, 512, 16, 16] → [1, 512, 16, 16]` |
| `dec.0` | 1,704,448 | `[1, 512, 16, 16] + [1, 256, 32, 32] → [1, 256, 32, 32]` |
| `dec.1` | 426,240 | `[1, 256, 32, 32] + [1, 128, 64, 64] → [1, 128, 64, 64]` |
| `dec.2` | 106,624 | `[1, 128, 64, 64] + [1, 64, 128, 128] → [1, 64, 128, 128]` |
| `dec.3` | 28,416 | `[1, 64, 128, 128] + [1, 38, 256, 256] → [1, 32, 256, 256]` |
| `post.0` | 9,248 | `[1, 32, 256, 256] → [1, 32, 256, 256]` |
| `post.1` | 528 | `[1, 32, 256, 256] → [1, 16, 256, 256]` |
| `post.2` | 51 | `[1, 16, 256, 256] → [1, 3, 256, 256]` |
| `(whole network)` | 8,642,179 | `[1, 3, 256, 256] + [1, 100] → [1, 3, 256, 256]` |

## Extractor stage shapes

Taken from `results/shapes_params.json` (`hooks.extractor`). Counts are recursive;
rows overlap, so summing them would double-count parameters.

| Module | Parameters | Input → output |
|---|---:|---|
| `decoder.conv1` | 9,408 | `[1, 3, 224, 224] → [1, 64, 112, 112]` |
| `decoder.bn1` | 128 | `[1, 64, 112, 112] → [1, 64, 112, 112]` |
| `decoder.relu` | 0 | `[1, 64, 112, 112] → [1, 64, 112, 112]` |
| `decoder.maxpool` | 0 | `[1, 64, 112, 112] → [1, 64, 56, 56]` |
| `decoder.layer1` | 215,808 | `[1, 64, 56, 56] → [1, 256, 56, 56]` |
| `decoder.layer2` | 1,219,584 | `[1, 256, 56, 56] → [1, 512, 28, 28]` |
| `decoder.layer3` | 7,098,368 | `[1, 512, 28, 28] → [1, 1024, 14, 14]` |
| `decoder.layer4` | 14,964,736 | `[1, 1024, 14, 14] → [1, 2048, 7, 7]` |
| `decoder.avgpool` | 0 | `[1, 2048, 7, 7] → [1, 2048, 1, 1]` |
| `decoder.fc` | 204,900 | `[1, 2048] → [1, 100]` |
| `(whole network)` | 23,712,932 | `[1, 3, 245, 245] → [1, 100]` |

## Remover stage shapes

Taken from `results/shapes_params.json` (`hooks.remover`). Counts are recursive;
rows overlap, so summing them would double-count parameters.

| Module | Parameters | Input → output |
|---|---:|---|
| `denoise.intro` | 896 | `[1, 3, 256, 256] → [1, 32, 256, 256]` |
| `denoise.encoders.0` | 94,624 | `[1, 32, 256, 256] → [1, 32, 256, 256]` |
| `denoise.downs.0` | 8,256 | `[1, 32, 256, 256] → [1, 64, 128, 128]` |
| `denoise.encoders.1` | 213,568 | `[1, 64, 128, 128] → [1, 64, 128, 128]` |
| `denoise.downs.1` | 32,896 | `[1, 64, 128, 128] → [1, 128, 64, 64]` |
| `denoise.encoders.2` | 1,056,512 | `[1, 128, 64, 64] → [1, 128, 64, 64]` |
| `denoise.downs.2` | 131,328 | `[1, 128, 64, 64] → [1, 256, 32, 32]` |
| `denoise.encoders.3` | 5,859,328 | `[1, 256, 32, 32] → [1, 256, 32, 32]` |
| `denoise.downs.3` | 524,800 | `[1, 256, 32, 32] → [1, 512, 16, 16]` |
| `denoise.middle_blks` | 27,400,704 | `[1, 512, 16, 16] → [1, 512, 16, 16]` |
| `denoise.ups.0` | 524,288 | `[1, 512, 16, 16] → [1, 256, 32, 32]` |
| `denoise.decoders.0` | 1,464,832 | `[1, 256, 32, 32] → [1, 256, 32, 32]` |
| `denoise.ups.1` | 131,072 | `[1, 256, 32, 32] → [1, 128, 64, 64]` |
| `denoise.decoders.1` | 528,256 | `[1, 128, 64, 64] → [1, 128, 64, 64]` |
| `denoise.ups.2` | 32,768 | `[1, 128, 64, 64] → [1, 64, 128, 128]` |
| `denoise.decoders.2` | 213,568 | `[1, 64, 128, 128] → [1, 64, 128, 128]` |
| `denoise.ups.3` | 8,192 | `[1, 64, 128, 128] → [1, 32, 256, 256]` |
| `denoise.decoders.3` | 94,624 | `[1, 32, 256, 256] → [1, 32, 256, 256]` |
| `denoise.ending` | 867 | `[1, 32, 256, 256] → [1, 3, 256, 256]` |
| `denoise` | 38,321,379 | `[1, 3, 256, 256] → [1, 3, 256, 256]` |
| `(whole network)` | 38,321,379 | `[1, 3, 256, 256] → [1, 3, 256, 256]` |

The embedder message projection reshapes 768 features to `[1, 3, 16, 16]`
before upsampling. Concatenating the cover and message image gives the six
channels entering `pre`. The final decoder skip has 38 channels (32 from
`pre` plus the six-channel fused input). The body has one bottleneck residual
block and no channel normalization. Downsampling layers use ReLU despite the
configuration's `activ=silu`; the pre/body decoder uses SiLU and the final
post layer uses tanh.

The extractor is ResNet-50 with bottleneck counts 3/4/6/3. Its FC has 204,900
parameters and returns 100 logits. The remover uses lightweight KBNet_s,
width 32, ffn_scale 1.5, encoder block counts [2, 2, 4, 8], 12 middle blocks
and decoder block counts [2, 2, 2, 2]. Its upsamplers contain a 1×1 convolution
and PixelShuffle(2). Encoder skips are added, KBNet adds the input after its
ending convolution, and the WMRemover wrapper clamps to [-1, 1]. LayerNorm2d
and multiplicative SimpleGate operations are included in the full module tree.

## Paper and released code

This comparison follows the local review of the paper
(arXiv:2311.18297v1, Sections 3.1.1, 3.1.2 and 3.2, Figure 1,
Table 4 and Supplement D.2) and the recorded released Q implementation.

| Item | Paper description | Released Q implementation |
|---|---|---|
| Body | MUNIT-based encoder/decoder with channel normalization removed, doubled depth and halved internal dimension; d and n are symbolic | `Unet1` with explicit skips, d=32, n=32, four down/up stages and one bottleneck residual block |
| E_post | Multiple 1×1 convolutions separated by SiLU, ending in tanh | A leading 3×3 32→32 convolution with SiLU (9,248 parameters), then 1×1 32→16 with SiLU (528), then 1×1 16→3 with tanh (51) |
| Bit head | An l-dimensional sigmoid-activated fully connected layer | 100 raw logits; public inference thresholds logits >0. A separate reference sigmoid has the equivalent threshold >0.5 |
| Decoder input | 256-square working image | Public resize to 245, then wrapper resize to 224 |
| Ablation scope | Table 4 uses TrustMark-B | These dimensions and counts describe the released Q networks |

## Model files and MD5

Official host:
`https://cai-watermark.adobe.net/watermarking/trustmark-models/`.
Each file is downloaded only if missing and verified before loading.

| File | Bytes | MD5 |
|---|---:|---|
| `decoder_Q.ckpt` | 47652460 | `4ced90e9cfe13e3295ad082887fe9187` |
| `encoder_Q.ckpt` | 17302074 | `700328b8754db934b2f6cb5e5185d81f` |
| `trustmark_Q.yaml` | 2009 | `fe40df84a7feeebfceb7a7678d7e6ec6` |
| `trustmark_rm_Q.yaml` | 1658 | `8476bcd4092abf302272868f3b4c2e38` |
| `trustmark_rm_Q.ckpt` | 77081406 | `760337a5596e665aed2ab5c49aa5284f` |

## Supplied results

The six files in `results/` are copied from the original CPU measurement:
Python 3.12.3, Torch 2.5.1+cpu, torchvision 0.20.1+cpu and TrustMark 0.9.2.
Private path strings are replaced with relative or neutral placeholders.
All numeric values, model hashes, source hashes, module inventories and hook
records are preserved. Only the path field in the environment block is redacted;
version and thread fields are unchanged.

The recorded `script_sha256` identifies the original capture script, not this
public adaptation. A new run records the public script's hash and its installed
package source hashes; it does not require private baseline files or source
locations. Package versions and CPU platforms can affect floating-point
decoding results.

- `results/config_dump.json`: resolved YAML and observed architecture settings.
- `results/module_tree.txt`: printed embedder, extractor and remover trees.
- `results/shapes_params.json`: every hook, per-module parameter counts, model
  MD5/SHA-256, package source hashes and environment.
- `results/trustmark_Q.yaml` and `results/trustmark_rm_Q.yaml`: unchanged
  released configurations, retaining the model MD5 values above.
- `results/measurement.log`: original stdout/stderr with its private path
  prefix replaced by a neutral package location.
