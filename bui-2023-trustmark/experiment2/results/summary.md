## Observed results

These are the recorded CPU runs, not a new execution of the public scripts. Image filenames identify artifacts created when reproducing; images, thumbnails and logs are not distributed. The historical determinism hashes precede the publication metadata and relative-path changes.

Raw accuracy below refers to the 61 payload bits unless explicitly labeled 100-bit float.

### A: single changes

Cells: raw correct/61, exact BCH recovery (Y/N).

| Change | Airplane 512 | Airplane 2048 | Boat 512 | Boat 2048 |
|---|---|---|---|---|
| noise008 | 60/61, Y | 61/61, Y | 57/61, N | 60/61, Y |
| contrast150 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| contrast050 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| saturation050 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| saturation150 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| hue005 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| sharpness250 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| median3 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| box7 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| motion9_45 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| posterize3 | 55/61, N | 56/61, N | 61/61, Y | 61/61, Y |
| rgb010 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| grayscale | 58/61, Y | 58/61, Y | 58/61, Y | 58/61, Y |
| hflip | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| brightness050 | 61/61, Y | 61/61, Y | 61/61, Y | 61/61, Y |
| brightness150 | 50/61, N | 50/61, N | 61/61, Y | 61/61, Y |
| jpeg40 | 61/61, Y | 61/61, Y | 58/61, Y | 59/61, Y |

| A aggregate | Mean raw accuracy | Minimum | Exact recovery |
|---|---:|---:|---:|
| airplane 512 | 97.97% | 81.97% | 15/17 |
| airplane 2048 | 98.17% | 81.97% | 15/17 |
| boat 512 | 99.04% | 93.44% | 16/17 |
| boat 2048 | 99.42% | 95.08% | 17/17 |

### B: combined changes

| Image | Size | Mean raw accuracy | Minimum | Exact recovery |
|---|---:|---:|---:|---:|
| airplane | 512 | 99.67% | 96.72% | 18/20 |
| airplane | 2048 | 99.18% | 91.80% | 19/20 |
| boat | 512 | 94.34% | 75.41% | 14/20 |
| boat | 2048 | 98.85% | 90.16% | 19/20 |

### C: removal

API present: True. Status: skipped. Q removal configuration and weights are absent from the released package files and local cache; no download attempted


### D: I-FGSM

| Image | Size | Steps | Float 100-bit raw | PNG payload correct | Detected | Exact | PSNR vs marked | Float criterion |
|---|---:|---:|---:|---:|---|---|---:|---|
| airplane | 512 | 2 | 46.0% | 33/61 (54.10%) | False | False | 44.29 | True |
| airplane | 2048 | 2 | 46.0% | 33/61 (54.10%) | False | False | 44.30 | True |
| boat | 512 | 4 | 45.0% | 27/61 (44.26%) | False | False | 42.11 | True |
| boat | 2048 | 2 | 49.0% | 29/61 (47.54%) | False | False | 45.15 | True |

### E: inference strength

PSNR is pre-JPEG embedding quality. Accuracy is correct/61; Y/N denotes exact BCH recovery.

| Image | Size | Lambda | PSNR vs input | Unchanged raw, exact | JPEG50 raw, exact |
|---|---:|---:|---:|---|---|
| airplane | 512 | 0.5 | 46.21 | 56/61, N | 52/61, N |
| airplane | 512 | 0.75 | 43.31 | 61/61, Y | 60/61, Y |
| airplane | 512 | 1.0 | 41.04 | 61/61, Y | 61/61, Y |
| airplane | 512 | 1.25 | 39.21 | 61/61, Y | 61/61, Y |
| airplane | 512 | 1.5 | 37.69 | 61/61, Y | 61/61, Y |
| airplane | 2048 | 0.5 | 46.14 | 56/61, N | 55/61, N |
| airplane | 2048 | 0.75 | 43.24 | 61/61, Y | 61/61, Y |
| airplane | 2048 | 1.0 | 40.97 | 61/61, Y | 61/61, Y |
| airplane | 2048 | 1.25 | 39.13 | 61/61, Y | 61/61, Y |
| airplane | 2048 | 1.5 | 37.61 | 61/61, Y | 61/61, Y |
| boat | 512 | 0.5 | 47.81 | 58/61, Y | 36/61, N |
| boat | 512 | 0.75 | 45.38 | 61/61, Y | 58/61, Y |
| boat | 512 | 1.0 | 43.32 | 61/61, Y | 58/61, Y |
| boat | 512 | 1.25 | 41.59 | 61/61, Y | 59/61, Y |
| boat | 512 | 1.5 | 40.12 | 61/61, Y | 61/61, Y |
| boat | 2048 | 0.5 | 47.77 | 58/61, Y | 35/61, N |
| boat | 2048 | 0.75 | 45.33 | 61/61, Y | 56/61, N |
| boat | 2048 | 1.0 | 43.26 | 61/61, Y | 59/61, Y |
| boat | 2048 | 1.25 | 41.53 | 61/61, Y | 59/61, Y |
| boat | 2048 | 1.5 | 40.06 | 61/61, Y | 61/61, Y |

### Runtime and repeatability

| Run | Total seconds | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| run1 | 81.47 | 27.78 | 28.01 | 0.00 | 1.07 | 17.63 |
| run2 | 81.24 | 27.89 | 27.91 | 0.00 | 1.12 | 17.63 |

Byte-identical except timing: **True**. Compared 221 files per run plus the logs; only timings.json was excluded.
Differences: []

2048 attack eligibility: first 512 attack under 300 s; enabled=True.

The model/source hashes and all four marked PNG hashes matched the public baseline reference. C was skipped for all four image/size pairs if listed above; no other requested item is silently omitted.
