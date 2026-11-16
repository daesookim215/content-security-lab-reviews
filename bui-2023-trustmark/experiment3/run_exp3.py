# /// script
# requires-python = ">=3.11"
# dependencies = ["trustmark==0.9.2", "torch", "torchvision", "numpy", "pillow"]
# ///
# How to run: use the existing environment; see README.md and reproduce_exp3.sh.
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image
from trustmark import TrustMark
from trustmark.trustmark import MODEL_CHECKSUMS, MODEL_REMOTE_HOST

ROOT = Path(__file__).resolve().parent
SUPPORT = ROOT.parent / "experiment2"
if not SUPPORT.exists():
    SUPPORT = ROOT.parent / "exp2"
sys.path.insert(0, str(SUPPORT))
from baseline_run1 import ExistingModelsTrustMark, PAYLOAD, SEED
from perturbations import jpeg
from records import Extra, Identity, Recorder, psnr, sha256

SECOND = "".join("1" if bit == "0" else "0" for bit in PAYLOAD)
EXPECTED = {"trustmark_rm_Q.yaml": "8476bcd4092abf302272868f3b4c2e38",
            "trustmark_rm_Q.ckpt": "760337a5596e665aed2ab5c49aa5284f"}


class RemovalTrustMark(ExistingModelsTrustMark):
    remover_directory: Path

    def load_model(self, config_path: str, weight_path: str, device: str,
                   secret_len: int, part: str = "all"):
        if part == "remover":
            config_path = str(self.remover_directory / Path(config_path).name)
            weight_path = str(self.remover_directory / Path(weight_path).name)
        return super().load_model(config_path, weight_path, device, secret_len, part=part)


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--inputs", type=Path, default=SUPPORT / "images")
    parser.add_argument("--reference", type=Path, default=SUPPORT / "reference_run1.json")
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT) and output != ROOT
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "thumbnails").mkdir()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    downloads = []
    for name, expected in EXPECTED.items():
        path = args.models / name
        content = path.read_bytes()
        digest = hashlib.md5(content).hexdigest()
        assert digest == MODEL_CHECKSUMS[name] == expected, name
        downloads.append({"file": name, "url": MODEL_REMOTE_HOST + name,
                          "size_bytes": len(content), "md5": digest, "md5_match": True,
                          "sha256": hashlib.sha256(content).hexdigest()})
    RemovalTrustMark.remover_directory = args.models
    model = RemovalTrustMark(device="cpu", model_type="Q", use_ECC=True,
        encoding_type=TrustMark.Encoding.BCH_5, verbose=False, loadRemover=True,
        loadBBoxDetector=False)
    assert model.encoder is not None and model.decoder is not None and model.removal is not None
    for component in (model.encoder, model.decoder, model.removal):
        component.requires_grad_(False)
    assert model.schemaCapacity() == len(PAYLOAD) == len(SECOND) == 61
    packet, second_packet = ("".join(str(int(b)) for b in model.ecc.encode_binary([p])[0])
                             for p in (PAYLOAD, SECOND))
    reference = json.loads(args.reference.read_text())
    assert reference["settings"]["expected_packet"] == packet
    package = Path(sys.modules[TrustMark.__module__].__file__).resolve().parent
    for name, digest in reference["settings"]["source_sha256"].items():
        assert sha256(package / name) == digest, name
    for name, digest in reference["settings"]["checkpoint_sha256"].items():
        assert sha256(package / "models" / name) == digest, name
    recorder = Recorder(model, output, packet)
    inputs = json.loads((args.inputs / "manifest.json").read_text())
    for entry in inputs:
        path = args.inputs / entry["filename"]
        assert sha256(path) == entry["sha256"]
        with Image.open(path) as source:
            native = source.convert("RGB")
        assert native.size == (512, 512)
        for size in (512, 2048):
            cover = native.copy() if size == 512 else native.resize((size, size), Image.Resampling.LANCZOS)
            marked = model.encode(cover, PAYLOAD, MODE="binary", WM_STRENGTH=1.0)
            identity = Identity(image=entry["image"], resolution=size, experiment="baseline", change="marked")
            recorder.record(marked, identity, Extra(psnr_input_db=psnr(marked, cover)))
            assert recorder.rows[-1]["image_sha256"] == reference["image_sha256"][f"images/{entry['image']}-{size}-marked.png"]
            recorder.record(cover, Identity(**{**identity, "change": "input"}))
            stage = time.perf_counter()
            removed = model.remove_watermark(marked)
            control = model.remove_watermark(cover)
            direct = model.encode(marked, SECOND, MODE="binary", WM_STRENGTH=1.0)
            after = model.encode(removed, SECOND, MODE="binary", WM_STRENGTH=1.0)
            jpeg_path = output / "images" / f"{entry['image']}-{size}-jpeg50.jpg"
            jpeg_path.write_bytes(jpeg(marked, 50))
            assert sha256(jpeg_path) == reference["image_sha256"][f"images/{entry['image']}-{size}-jpeg50.jpg"]
            with Image.open(jpeg_path) as compressed:
                jpeg_removed = model.remove_watermark(compressed.convert("RGB"))
            for experiment, change, image in (
                ("C1", "removed", removed), ("C1", "unmarked-control", control),
                ("C2", "direct-second", direct), ("C2", "removed-second", after),
                ("C3", "jpeg50-removed", jpeg_removed)):
                recorder.record(image, Identity(**{**identity, "experiment": experiment, "change": change}),
                    Extra(psnr_input_db=psnr(image, cover), psnr_marked_db=psnr(image, marked)))
                row = recorder.rows[-1]
                second_bits = sum(a == b for a, b in zip(row["raw_payload"], SECOND, strict=True))
                second_packet_bits = sum(a == b for a, b in zip(row["raw_packet"], second_packet, strict=True))
                recovered = "neither"
                if row["decoded_payload"] == PAYLOAD:
                    recovered = "first"
                if row["decoded_payload"] == SECOND:
                    recovered = "second"
                row.update(second_payload_bits_correct=second_bits,
                    second_packet_bits_correct=second_packet_bits,
                    second_packet_accuracy=second_packet_bits / 100,
                    exact_second_payload=row["decoded_payload"] == SECOND,
                    recovered_payload=recovered)
            recorder.timings.append({"image": entry["image"], "resolution": size,
                                     "operations_seconds": time.perf_counter() - stage})
            if size == 512:
                recorder.thumbnail(removed, f"C1-{entry['image']}")
                recorder.thumbnail(after, f"C2-{entry['image']}-removed-second")
    fields = sorted({key for row in recorder.rows for key in row})
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in recorder.rows)
    for experiment in ("baseline", "C1", "C2", "C3"):
        (output / f"{experiment}.json").write_text(json.dumps(
            [r for r in recorder.rows if r["experiment"] == experiment], indent=2, sort_keys=True) + "\n")
    settings = {**reference["settings"], "second_payload": SECOND, "second_packet": second_packet,
        "second_payload_derivation": "bitwise NOT of first payload", "strength": 1.0,
        "device": "cpu", "threads": 8, "interop_threads": 1, "cuda_visible_devices": "",
        "remover": "trustmark_rm_Q", "remover_size": model.model_resolution_remove,
        "package_version": importlib.metadata.version("trustmark"), "torch_version": torch.__version__,
        "pillow_version": PIL.__version__, "numpy_version": np.__version__,
        "python_version": sys.version.split()[0], "jpeg_quality": 50, "jpeg_subsampling": 2,
        "jpeg_optimize": False, "jpeg_progressive": False}
    data = {"settings": settings, "inputs": inputs, "downloads": downloads, "rows": recorder.rows,
        "baseline_marked_hashes_match_run1": True, "jpeg50_hashes_match_run1": True,
        "image_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted((output / "images").iterdir())},
        "thumbnail_sha256": {p.name: sha256(p) for p in sorted((output / "thumbnails").iterdir())},
        "script_sha256": sha256(Path(__file__))}
    (output / "results.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (output / "timings.json").write_text(json.dumps({"total_seconds": time.perf_counter() - started,
        "measurements": recorder.timings}, indent=2, sort_keys=True) + "\n")
    print("PASS: baseline and JPEG50 hashes match; 20 C measurements", flush=True)


if __name__ == "__main__":
    main()
