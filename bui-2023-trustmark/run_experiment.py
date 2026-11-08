import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import PIL
import torch
from torchvision.transforms.functional import to_tensor
from trustmark import TrustMark
from trustmark.trustmark import MODEL_CHECKSUMS

SEED = 20231130
PAYLOAD = "1011001010010110011100010100101011010001110001101010010110101"
CHANGES = ("none", "jpeg90", "jpeg70", "jpeg50", "downscale50", "blur", "brightness", "crop80")


class ExistingModelsTrustMark(TrustMark):
    def check_and_download(self, filename: str) -> None:
        path = Path(filename)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != MODEL_CHECKSUMS[path.name]:
            raise RuntimeError(f"Existing model checksum mismatch: {path.name}")


def save_change(marked: Image.Image, change: str, destination: Path) -> Path:
    width, height = marked.size
    suffix = ".png"
    match change:
        case "none":
            result = marked.copy()
        case "jpeg90" | "jpeg70" | "jpeg50":
            path = destination.with_suffix(".jpg")
            marked.save(path, quality=int(change[4:]), subsampling=2,
                        optimize=False, progressive=False)
            return path
        case "downscale50":
            result = marked.resize((width // 2, height // 2), Image.Resampling.LANCZOS)
        case "blur":
            result = marked.filter(ImageFilter.GaussianBlur(radius=2.0))
        case "brightness":
            result = ImageEnhance.Brightness(marked).enhance(1.5)
        case "crop80":
            side = round(width * 0.8)
            left = (width - side) // 2
            top = (height - side) // 2
            result = marked.crop((left, top, left + side, top + side))
        case _:
            raise ValueError(change)
    path = destination.with_suffix(suffix)
    result.save(path)
    return path


def decode_record(model: TrustMark, image: Image.Image) -> dict:
    recovered, detected, schema = model.decode(image, MODE="binary")
    processed = model.get_the_image_for_processing(image)
    resized = processed.resize((model.model_resolution_dec,) * 2, Image.Resampling.BILINEAR)
    tensor = to_tensor(resized).unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        logits = model.decoder.decoder(tensor)
    raw = "".join(str(int(bit)) for bit in (logits > 0).numpy()[0])
    candidate, candidate_valid, candidate_schema = model.ecc.decode_bitstream(
        np.array([[int(bit) for bit in raw]]), MODE="binary")[0]
    correct = sum(a == b for a, b in zip(raw[:len(PAYLOAD)], PAYLOAD, strict=True))
    return {"raw_packet": raw, "raw_payload": raw[:len(PAYLOAD)],
            "payload_bits_correct": correct, "bit_accuracy": correct / len(PAYLOAD),
            "decoded_payload": recovered, "detected": bool(detected),
            "schema": int(schema), "exact_payload": recovered == PAYLOAD,
            "candidate_payload": candidate, "candidate_valid": bool(candidate_valid),
            "candidate_schema": int(candidate_schema)}


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    output.mkdir(exist_ok=False, parents=True)
    images = output / "images"
    images.mkdir()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the empty string")
    model = ExistingModelsTrustMark(device="cpu", model_type="Q", use_ECC=True,
                                   encoding_type=TrustMark.Encoding.BCH_5,
                                   verbose=False, loadRemover=False,
                                   loadBBoxDetector=False)
    assert len(PAYLOAD) == model.schemaCapacity() == 61
    packet = "".join(str(int(bit)) for bit in model.ecc.encode_binary([PAYLOAD])[0])
    package = Path(sys.modules[TrustMark.__module__].__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                     for name in ("trustmark.py", "datalayer.py", "bchecc.py", "model.py", "unet.py")}
    checkpoint_hashes = {name: hashlib.sha256((package / "models" / name).read_bytes()).hexdigest()
                         for name in ("encoder_Q.ckpt", "decoder_Q.ckpt", "trustmark_Q.yaml")}
    settings = {"seed": SEED, "payload": PAYLOAD, "payload_bits": len(PAYLOAD),
                "expected_packet": packet, "packet_bits": len(packet),
                "parity_bits": model.ecc.bch_encoder.get_ecc_bits(),
                "schema_bits": model.ecc.versionbits, "bch_mode": "BCH_5",
                "bch_polynomial": 137, "bch_correction_bits": 5,
                "model": "Q", "strength": 1.0, "merge": "bilinear",
                "encoder_size": model.model_resolution_enc,
                "decoder_size": model.model_resolution_dec,
                "concentrate_wm_region": model.concentrate_wm_region,
                "resolutions": [512, 2048], "resampling": "Lanczos",
                "jpeg_qualities": [90, 70, 50], "jpeg_subsampling": 2,
                "blur_radius": 2.0, "brightness_factor": 1.5,
                "crop_side_fraction": 0.8, "downscale_fraction": 0.5,
                "device": "cpu", "threads": 8, "interop_threads": 1,
                "package_version": importlib.metadata.version("trustmark"),
                "torch_version": torch.__version__, "pillow_version": PIL.__version__,
                "numpy_version": np.__version__, "python_version": sys.version.split()[0],
                "source_sha256": source_hashes, "checkpoint_sha256": checkpoint_hashes}
    rows, quality, unmarked = [], [], []
    input_manifest = json.loads((Path("images") / "manifest.json").read_text())
    for entry in input_manifest:
        original_path = Path("images") / entry["filename"]
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == entry["sha256"]
        with Image.open(original_path) as original:
            native = original.convert("RGB")
        for size in settings["resolutions"]:
            name = entry["image"]
            cover = native.copy() if size == 512 else native.resize((size, size), Image.Resampling.LANCZOS)
            stem = f"{name}-{size}"
            cover_path = images / f"{stem}-input.png"
            cover.save(cover_path)
            unmarked.append({"image": name, "resolution": size, **decode_record(model, cover)})
            marked = model.encode(cover, PAYLOAD, MODE="binary")
            marked_path = images / f"{stem}-marked.png"
            marked.save(marked_path)
            error = np.asarray(marked, dtype=np.float64) - np.asarray(cover, dtype=np.float64)
            mse = float(np.mean(error ** 2))
            quality.append({"image": name, "resolution": size, "mse": mse,
                            "psnr_db": 10 * math.log10(255 ** 2 / mse),
                            "input_file": str(cover_path.relative_to(output)),
                            "marked_file": str(marked_path.relative_to(output))})
            for change in CHANGES:
                path = save_change(marked, change, images / f"{stem}-{change}")
                with Image.open(path) as changed:
                    result = decode_record(model, changed.convert("RGB"))
                    dimensions = list(changed.size)
                rows.append({"image": name, "resolution": size, "change": change,
                             "dimensions": dimensions, "file": str(path.relative_to(output)), **result})
                print(stem, change, result["payload_bits_correct"], result["exact_payload"], flush=True)
    image_hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(images.iterdir())}
    data = {"settings": settings, "inputs": input_manifest, "quality": quality,
            "unmarked": unmarked, "rows": rows, "image_sha256": image_hashes}
    (output / "results.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    for name, records in (("measurements", rows), ("quality", quality), ("unmarked", unmarked)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
