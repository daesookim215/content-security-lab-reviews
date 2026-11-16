# TrustMark extended experiments

These scripts extend the two-image recovery experiment with TrustMark 0.9.2,
the official Q encoder/decoder, BCH_5 and seed 20231130. The fixed 61-bit payload is
`1011001010010110011100010100101011010001110001101010010110101`.
Airplane and Boat are converted to RGB at 512 x 512 and enlarged with Lanczos
to 2048 x 2048. Strength is 1.0 except in E.

| Experiment | Measurement |
|---|---|
| A: single changes | Apply each of 17 settings separately to each marked image/size. |
| B: combined draws | Apply 20 seeded geometry/perturbation pipelines per image/size. |
| C: removal and re-watermarking | The wrapper runs the complete C1/C2/C3 experiments in [../experiment3/](../experiment3/) after A/B/D/E. |
| D: I-FGSM | Attack the decoder through input gradients, then decode a saved PNG through the normal API. |
| E: strength sweep | Encode at five residual strengths; decode unchanged and after JPEG50. |

## Run locally

First run `sh reproduce.sh` in the parent folder as described in its README
(Python 3.12 and the standard `patch` command). Then, from that same folder:

```sh
sh experiment2/reproduce_exp2.sh
```

The wrapper reuses `../.venv/bin/python` and `../images/manifest.json` plus the
downloaded TIFF inputs, relative to this directory. Checkpoints are loaded from
the installed package's `trustmark/models` directory, populated by the parent's
`download_models.py`. The final experiment3 step downloads exactly the two
official Q remover files into `../experiment3/.cache/models`, with MD5 checks.
`baseline_run1.py` imports the public parent's checksum loader and decode function.

Runs execute serially on CPU with `CUDA_VISIBLE_DEVICES=""`, eight intra-op
threads, one inter-op thread and BLAS/OpenMP limits of eight. Outputs go to
`reproduced/run1` and `reproduced/run2`; an existing run directory causes the
wrapper to stop. Cache/config writes use the parent's `.cache`, and temporary
files use `reproduced/tmp`. The wrapper invokes `compare_runs.py reproduced`
and writes `reproduced/determinism.json`, then generates
`reproduced/summary.md`. A comparison difference produces a failing exit status.
To regenerate reference tables without running experiments, use
`python3 report_exp2.py results` from this directory.

The runner checks package source/checkpoint hashes and all four baseline marked
PNG hashes against the public `../results/run1/results.json`. Those hashes are
present in the public reference; no separate reference file is needed.
Library versions and CPU platforms can change exact pixels, so this historical
baseline check may fail on another platform. It is separate from the comparison
between the two fresh runs.

## Settings

These are deterministic Pillow/NumPy analogues of the paper's perturbations,
not an exact reproduction of Kornia or the paper's aggregate benchmark.
Paper locations: Table 6 and Supplement C, Eq. 7 (PDF page 13), geometry in
Supplement A.2 (page 11), I-FGSM in Section 4.3 (page 7).

| Single change (A) | Value | Paper setting | Implementation |
|---|---|---|---|
| Gaussian noise | std 0.08 in [0,1] | std 0.08 | Independent RGB `x + N(0,0.08^2)`, clip and round |
| Contrast | 1.5 and 0.5 | 0.5-1.5 | PIL Contrast: blend with constant mean luminance |
| Saturation | 0.5 and 1.5 | 0.5-1.5 | PIL Color: blend RGB with its luminance |
| Hue | +0.05 turn | 0.05 | PIL HSV H channel shifted by round(0.05*256)=13 modulo 256; effective 13/256 turn |
| Sharpness | 2.5 | 2.5 | PIL Sharpness: blend SMOOTH-filtered image with original, factor 2.5 |
| Median blur | kernel 3 | k=3 | PIL MedianFilter(3) |
| Box blur | kernel 7 | k=7 | PIL BoxBlur(radius=3), normalized 7x7 box |
| Motion blur | kernel 9, angle 45 degrees, direction 0 | k=3-9, angle -90..90, direction -1..1 | `y[i,j,c] = mean(x[i+d,j+d,c], d=-4..4)`; replicated edges, symmetric weights; +45 degrees in image coordinates |
| Posterize | 3 bits | 3 | Keep top three bits: `x & 224` |
| RGB shift | (+0.1,-0.1,+0.1) | shift limit 0.1 | Add fixed channel offsets in [0,1], clip and round |
| Grayscale | forced conversion | application p=0.5 | PIL luminance conversion, copied into RGB channels |
| Horizontal flip | forced mirror | geometry p=0.5 | PIL horizontal mirror |
| Brightness | 0.5 and 1.5 | 0.5-1.5 | PIL Brightness; 1.5 repeats run1 for comparison |
| JPEG | quality 40 | minimum q=40 | Pillow, 4:2:0, optimize=False, progressive=False |

NumPy float operations are clipped to [0,1] and quantized with
`rint(255*x)` (ties to even). PIL operations use Pillow's native integer
rounding. Hue quantization, PIL contrast/sharpness definitions, and motion blur's
uniform diagonal kernel differ from exact Kornia implementations. A uses the
17 settings listed above; Table 6's additional color-jiggle/Gaussian-blur
operators are not included. Gaussian noise uses NumPy PCG64; every measurement stores
its seed. A's seed is `20231130 + size*1000 + image_index*100 + setting_index`,
with Airplane index 0, Boat index 1 and the order shown in settings JSON.

B draws 20 pipelines per image/size. Each draw stores its seed, flip, resize
parameters, dimensions, crop rectangle and ordered perturbation names. Its seed
is `20231130 + 100000 + size*100 + image_index*10000 + draw_index`.
Apply mirror with probability 0.5, sample linear scale s uniformly in [0.8,1.0]
and aspect a log-uniformly in [3/4,4/3], then bilinear-resize to
`(round(W*s*sqrt(a)), round(H*s/sqrt(a)))`. Here s scales linear dimensions,
not area. Crop a rectangle
with sides `round(new_side*244/256)` at a uniformly random valid integer origin.
This is a randomly placed crop of the stated relative size; no final resize is
added. Select two distinct *types* uniformly without replacement from list A,
then select an endpoint uniformly where contrast/saturation/brightness have two
settings. Apply in sampled order, at full strength (including forced grayscale
and flip if chosen). Thus a flip chosen as a perturbation can undo the geometry
flip. All random decisions, including Gaussian samples, use the same per-draw
generator. This is a documented analogue of the training/test pipeline.

| Other settings | Definition |
|---|---|
| C: removal | Run the standalone experiment3 wrapper with checksum-verified `trustmark_rm_Q` files; measure marked removal, an unmarked control, both second-message embedding paths, and JPEG50 removal. |
| D: objective | Minimize BCEWithLogits toward the flipped 100-bit ECC packet; freeze decoder parameters and differentiate only the input. |
| D: update | Start from the marked image in float32 [0,1]; subtract `(1/255)*sign(gradient)`, project into its L-infinity ball of radius 8/255, and clip to [0,1]. |
| D: decoder input | Bilinear resize to 245 x 245 with `align_corners=False, antialias=True`, then `2*x-1`. The float path omits Pillow's intermediate uint8 rounding; the initial resize difference is recorded. |
| D: stopping | Evaluate iteration zero and each update; stop at 100-bit packet accuracy strictly below `0.5+(8/255)/2 = 0.5156862745`, or after 500 updates. |
| D: saved-image check | Quantize with `rint(255*x)`, save PNG, enforce the 8/255 limit again, and decode through the normal Pillow/API path. Float and PNG outcomes may differ. |
| D: sizes | Always attack both 512 inputs. Attack both 2048 inputs only if the first Airplane-512 attack takes under 300 seconds; run2 replays run1's decision. |
| E: strengths | `WM_STRENGTH = 0.5, 0.75, 1.0, 1.25, 1.5`; retain the package's bilinear residual merge and border feathering. |
| E: measurements | Decode unchanged and JPEG50 (4:2:0, optimize=False, progressive=False). Embedding PSNR is against the cover before JPEG, including on JPEG rows. |

Completed removal and re-watermarking records are in
[../experiment3/results/](../experiment3/results/) and its README. The original
A/B/D/E records below retain their historical availability metadata and row counts;
experiment3 supplies the complete C measurements using the official Q remover.

Raw payload accuracy counts the first 61 network bits before BCH. Packet
accuracy counts all 100 bits. Exact recovery compares the API output after BCH
with the full payload; detection is the API flag. JSON null PSNR means identical
pixels (infinite PSNR). C, when available, records PSNR against both the cover
and marked image; D records it against the marked image.

## Recorded results

These tables summarize the completed CPU runs. Full per-setting values and
attack traces are in [results/summary.md](results/summary.md) and the run CSV/JSON.

| Image | Size | A: exact recovery failures (raw payload correct) | B: exact recovery | D: steps; payload correct after PNG |
|---|---:|---|---:|---|
| Airplane | 512 | Posterize3 (55/61), brightness1.5 (50/61); 15/17 exact | 18/20 | 2; 33/61 |
| Airplane | 2048 | Posterize3 (56/61), brightness1.5 (50/61); 15/17 exact | 19/20 | 2; 33/61 |
| Boat | 512 | Noise0.08 (57/61); 16/17 exact | 14/20 | 4; 27/61 |
| Boat | 2048 | None; 17/17 exact | 19/20 | 2; 29/61 |

All four D outputs have detection=false and exact recovery=false after PNG.
The final float packet accuracies are 46%, 46%, 45% and 49% in the table's order;
all meet the float stopping criterion. The separate experiment3 results contain
20 C measurements per run, including controls and both re-watermarking paths.

| E: strength | Unchanged exact (four image/size pairs) | JPEG50 exact | Embedding PSNR range, dB |
|---|---:|---:|---:|
| 0.5 | 2/4 | 0/4 | 46.14-47.81 |
| 0.75 | 4/4 | 3/4 | 43.24-45.38 |
| 1.0 | 4/4 | 4/4 | 40.97-43.32 |
| 1.25 | 4/4 | 4/4 | 39.13-41.59 |
| 1.5 | 4/4 | 4/4 | 37.61-40.12 |

At strength 0.5, unchanged Airplane has 56/61 correct at both sizes and fails
exact recovery. Boat recovers unchanged at both sizes but fails after JPEG50
(36/61 at 512, 35/61 at 2048). At 0.75, Boat-2048 after JPEG50 is the only
failure (56/61).

| CPU run | Total seconds | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| run1 | 81.47 | 27.78 | 28.01 | 0.00 | 1.07 | 17.63 |
| run2 | 81.24 | 27.89 | 27.91 | 0.00 | 1.12 | 17.63 |

Times are from the recorded eight-thread CPU runs; initialization, baseline and
control work also contribute to totals. Other machines may take longer.
Both recorded runs enabled the 2048 attacks. Each has 200 rows:
A 68, B 80, C 0, D 4, E 40, baseline 4 and control 4.

## Determinism and distributed records

The original comparison found 221 byte-identical files per run plus identical
logs, excluding only `timings.json`. It included full-resolution PNG/JPEG files
and thumbnails. `results/determinism.json` retains that historical comparison;
`results/validation.json` retains the original audit, including original image
inspections. Neither records a new execution of the public scripts.

This folder distributes scripts and numeric CSV/JSON/Markdown only.
Image paths and image hashes in the records identify files created when
reproducing under `reproduced/run1` or `reproduced/run2`; no images,
thumbnails or run logs are included in `results/`.
Full-size saved images are the decode/hash artifacts; generated thumbnails
(maximum side 600 pixels) are previews. JPEG40 rows decode a lossless PNG of
the JPEG-decoded pixels; the generated JPEG source is separately hashed.

Absolute removal-search paths were replaced with relative package/cache paths.
Publication notes and the baseline-check label were added to the copied JSON;
all numeric measurements are retained. Historical file hashes describe the
original files before these metadata edits. The recorded `script_sha256`
values describe the original executed scripts, not this adapted public copy.
The public scripts have not been used to rerun the experiments during packaging.
