# TrustMark removal and re-watermarking (experiment C)

Two serial CPU runs completed C1 removal, C2 re-watermarking and C3 removal
after JPEG50 for Airplane and Boat at 512 and 2048 pixels square. Each run
contains 20 C measurements plus eight baseline records. The Q encoder/decoder,
prepared inputs and JPEG50 bytes matched the baseline's recorded hashes.

## Download provenance

Only these two files were downloaded, on the experiment machine, into this
experiment's `models/` directory. The official `MODEL_REMOTE_HOST` and
`MODEL_CHECKSUMS` came from the unchanged TrustMark checkout's `trustmark.py`.
Both MD5 checks passed before the experiment loaded the remover. The loader
rechecked both MD5 values against the package constants and the approved values.

| File / URL | Bytes | MD5 (match) | SHA-256 |
|---|---:|---|---|
| [trustmark_rm_Q.yaml](https://cai-watermark.adobe.net/watermarking/trustmark-models/trustmark_rm_Q.yaml) | 1658 | 8476bcd4092abf302272868f3b4c2e38 (yes) | d8ae92c8ad0135a5d085c295be8560fbb67af9b8f5ab46bbf25f4a1038a8626f |
| [trustmark_rm_Q.ckpt](https://cai-watermark.adobe.net/watermarking/trustmark-models/trustmark_rm_Q.ckpt) | 77081406 | 760337a5596e665aed2ab5c49aa5284f (yes) | cb23dc28b69104ae551286f63da6e965d269f715a87748e3ae7fe0d790f7dc9a |

The downloaded weights are not distributed here. `results/run1/results.json`
records URLs, byte lengths, MD5 matches and SHA-256 values as structured data.
No package model file was changed. `RemovalTrustMark.load_model` redirects only
the remover configuration/checkpoint to the experiment's model directory;
the inherited checksum-only loader prevents implicit downloads for every model.

## Settings and reproduction

`run_exp3.py` imports experiment2's `baseline_run1`, `records` and
`perturbations.jpeg`; those own the original seed/payload, public/raw decode,
PSNR, image recording and explicit JPEG settings. It uses TrustMark 0.9.2 Q,
BCH_5, strength 1.0, seed 20231130, RGB inputs, native 512 or Lanczos-enlarged
2048, and the released residual merge/feathering. The remover processes at 256.
The existing Q encoder/decoder/config cache is verified by official MD5 and
baseline SHA-256. Package source SHA-256 checks include the existing BCH fixes.

First payload: `1011001010010110011100010100101011010001110001101010010110101`.
Second payload: `0100110101101001100011101011010100101110001110010101101001010`.
The second is the bitwise NOT of the first. Both have 61 payload bits; the JSON
records both 100-bit BCH/schema packets. Each C row records raw payload and
packet strings, raw matches against both messages, packet accuracies, detected,
public decoded payload, exact matches and the recovered message classification.
`exact_payload` always refers to the first message; `exact_second_payload`
refers to the second. `recovered_payload` is first, second or neither.

PSNR uses RGB mean squared error with peak 255. `psnr_input_db` compares against
the original prepared unmarked input; `psnr_marked_db` compares against its
first marked image. Null PSNR denotes identical pixels. Each result is saved
as PNG and reopened for decoding. JPEG50 uses Pillow quality=50,
subsampling=2, optimize=False and progressive=False; C3 removes from its decoded
RGB pixels. Baseline JPEG50 file hashes must match before removal.

CPU only: `CUDA_VISIBLE_DEVICES=""`; Torch eight intra-op threads and one
inter-op thread; OpenMP/BLAS limits eight; deterministic algorithms enabled.
Reference environment: Python 3.12.3, Torch 2.5.1+cpu, NumPy 2.2.6,
Pillow 11.3.0. Cache/config/temp writes stay inside this experiment.

With the existing environment, model files and experiment2 support files,
run `python -u run_exp3.py run1`, then `python -u run_exp3.py run2`, capturing
each log as `run1.log` and `run2.log`; run `python compare_runs.py` afterward.
Set the CPU/thread/cache variables above first. `--models`, `--inputs` and
`--reference` can select existing files. The public copy provides
`sh reproduce_exp3.sh` after the parent baseline reproduction; it downloads
and verifies just the two remover files into its own `.cache/models`.
No other experiment is run by that wrapper.

## C1: removal and unmarked control

Payload and packet columns count matches to the first message before BCH.

| Image | Size | Input PSNR dB | Marked PSNR dB | Payload /61 | Packet /100 | Detected | Exact first |
|---|---:|---:|---:|---:|---:|---|---|
| Airplane | 512 | 40.58 | 46.53 | 39 | 70 | true | false |
| Airplane | 2048 | 40.52 | 46.40 | 40 | 71 | false | false |
| Boat | 512 | 43.36 | 47.33 | 45 | 77 | false | false |
| Boat | 2048 | 43.27 | 47.08 | 44 | 72 | false | false |

Airplane-512 decoded `100100100101000111000011010010000100100100011110101101100011111110010101111`.
This detected payload has a different length and equals neither fixed message.
All other removal outputs returned an empty public payload. Detection therefore
does not imply recovery of the embedded identifier.

| Unmarked control | Size | Input PSNR dB | Payload /61 | Packet /100 | Detected | Public payload |
|---|---:|---:|---:|---:|---|---|
| Airplane | 512 | 35.76 | 32 | 53 | false | empty |
| Airplane | 2048 | 37.64 | 32 | 54 | false | empty |
| Boat | 512 | 47.31 | 35 | 55 | false | empty |
| Boat | 2048 | 47.01 | 34 | 55 | false | empty |

## C2: second-message embedding

Both paths recovered the second payload exactly for every image/size, with
detected=true, first raw matches 0/61, second raw matches 61/61, first packet
matches 22/100 and second packet matches 100/100. First-message exact=false.

| Image | Size | Direct PSNR vs input dB | Removal-first PSNR vs input dB | Recovered (both paths) |
|---|---:|---:|---:|---|
| Airplane | 512 | 38.47 | 38.33 | second |
| Airplane | 2048 | 38.40 | 38.24 | second |
| Boat | 512 | 40.31 | 40.27 | second |
| Boat | 2048 | 40.25 | 40.18 | second |

## C3: removal after JPEG50

| Image | Size | Input PSNR dB | Marked PSNR dB | Payload /61 | Packet /100 | Detected | Exact first |
|---|---:|---:|---:|---:|---:|---|---|
| Airplane | 512 | 30.58 | 30.96 | 49 | 85 | false | false |
| Airplane | 2048 | 36.92 | 38.66 | 36 | 68 | false | false |
| Boat | 512 | 33.03 | 33.32 | 48 | 75 | false | false |
| Boat | 2048 | 40.06 | 41.84 | 46 | 76 | false | false |

All C3 public decoded payloads are empty.

## Determinism and runtime

All 42 non-timing files per run and both run logs are byte-identical. The
comparison includes 32 full-resolution PNG/JPEG artifacts, four thumbnails,
four per-group JSON files, `results.json` and `measurements.csv`. Only
`timings.json` is excluded. `results/determinism.json` lists every SHA-256 pair.
Runtime: run1 25.4383 seconds; run2 24.8614 seconds including model loading,
encoding, removal, decoding, serialization and saving. Hardware/library changes
can affect exact bytes; each new pair must pass its own comparison.

This local folder contains numeric CSV/JSON records, logs and just four
run1 thumbnails (maximum side 600 pixels). Full-resolution images remain with
the executed runs. The public copy distributes numeric records and this README
without images or logs; recorded file hashes identify reproduction artifacts.
