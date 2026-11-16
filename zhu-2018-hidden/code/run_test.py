# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.4.6", "pillow==13.0.0"]
# ///
"""Run released weights only: .venv/bin/python experiment/run_test.py [repo data out]."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Final

import numpy as np
import PIL
import torch
from PIL import Image, features

from attacks import MESSAGE, SEED, blur, psnr, save_png, serialize, tensor

ROOT: Final = Path(__file__).resolve().parents[1]
REPO: Final = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "code/ando-HiDDeN"
DATA: Final = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data"
OUT: Final = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "experiment/results/run1"
sys.path.insert(0, str(REPO))
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration

INPUTS: Final = (
    ("Airplane", "4.2.05.tiff", "ae9eba12b80df955e47585bf4a16332ea0dadc753668d3457c509564934dbd40"),
    ("Boat", "boat.512.tiff", "6bc7319b1e3b44ce0b2a571ac39a28bac94437d812057d1afb2cf78ea3a9a35f"),
)
MODELS: Final = (
    ("combined", "experiments/combined-noise/checkpoints/combined-noise--epoch-400.pyt", "ef9741110627561a634bf54a9fae82d3c3b8b4621596a5770f5aa8f8a31b532d"),
    ("identity", "experiments/no-noise adam-eps-1e-4/checkpoints/no-noise--epoch-200.pyt", "71118462194d647f6f5c08dd35f95d4f53db11a9bd2758f3bd336b46aafa5916"),
)


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    config = HiDDenConfiguration(H=128, W=128, message_length=30, encoder_blocks=4,
        encoder_channels=64, decoder_blocks=7, decoder_channels=64, use_discriminator=True,
        use_vgg=False, discriminator_blocks=3, discriminator_channels=64,
        decoder_loss=1., encoder_loss=.7, adversarial_loss=.001)
    message = torch.tensor([[int(bit) for bit in MESSAGE]], dtype=torch.float32)
    rows, quality = [], []
    example = {}
    for model_name, checkpoint, expected_hash in MODELS:
        path = REPO / checkpoint
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        state = torch.load(path, map_location="cpu", weights_only=True)
        encoder, decoder = Encoder(config).eval(), Decoder(config).eval()
        weights = state["enc-dec-model"]
        encoder.load_state_dict({key.removeprefix("encoder."): value for key, value in weights.items() if key.startswith("encoder.")}, strict=True)
        decoder.load_state_dict({key.removeprefix("decoder."): value for key, value in weights.items() if key.startswith("decoder.")}, strict=True)
        for image_index, (name, filename, source_hash) in enumerate(INPUTS):
            assert hashlib.sha256((DATA / filename).read_bytes()).hexdigest() == source_hash
            with Image.open(DATA / filename) as source:
                cover = np.array(source.convert("RGB").crop((192, 192, 320, 320)))
            folder = OUT / "images" / name.lower() / model_name
            folder.mkdir(parents=True, exist_ok=True)
            save_png(folder / "cover.png", cover)
            with torch.inference_mode():
                raw = encoder(tensor(cover), message)
                raw_prediction = decoder(raw).squeeze(0).tolist()
            np.save(folder / "encoded_float.npy", raw.numpy())
            encoded = save_png(folder / "encoded.png", serialize(raw))
            raw_pixels = (raw.squeeze(0).permute(1, 2, 0).numpy() + 1) * 127.5
            quality_row = {"image": name, "model": model_name, "psnr_db": psnr((cover, encoded)),
                "raw_psnr_db": psnr((cover, raw_pixels)), "clipped_components": int(((raw.numpy() < -1) | (raw.numpy() > 1)).sum()),
                "raw_correct": int(np.sum((np.array(raw_prediction) >= .5) == np.array([int(b) for b in MESSAGE]))),
                "raw_predictions": raw_prediction, "checkpoint_sha256": expected_hash}
            quality.append(quality_row)

            def measure(attack: tuple[str, float, int], received: np.ndarray, metadata: dict) -> None:
                family, strength, draw = attack
                stem = f"{family}-{strength:g}-{draw:02d}"
                target = folder / f"{stem}.png"
                reopened = save_png(target, received)
                with torch.inference_mode():
                    prediction = decoder(tensor(reopened)).squeeze(0).tolist()
                bits = "".join(str(int(p >= .5)) for p in prediction)
                correct = sum(a == b for a, b in zip(MESSAGE, bits, strict=True))
                rows.append({"image": name, "model": model_name, "checkpoint_sha256": expected_hash,
                    "attack": family, "strength": strength, "draw": draw, "width": reopened.shape[1],
                    "height": reopened.shape[0], "correct": correct, "denominator": 30, "accuracy": correct / 30,
                    "exact": correct == 30, "predictions": prediction, "bits": bits,
                    "received": str(target.relative_to(OUT)), "embedding_psnr_db": quality_row["psnr_db"], **metadata})

            measure(("identity", 1., 0), encoded, {})
            if name == "Boat" and model_name == "combined":
                example = {"image": name, "model": model_name, "message": MESSAGE,
                    "crop_xyxy": [192, 192, 320, 320], "block_yx": [60, 64, 60, 64],
                    "channel": "red", "cover": cover[60:64, 60:64, 0].tolist(),
                    "encoded_float_pixel_units": raw_pixels[60:64, 60:64, 0].tolist(),
                    "signed_difference": (raw_pixels - cover)[60:64, 60:64, 0].tolist(),
                    "serialized": encoded[60:64, 60:64, 0].tolist(),
                    "predictions": rows[-1]["predictions"], "bits": rows[-1]["bits"], "correct": rows[-1]["correct"]}
            for q in (95, 75, 50, 30):
                jpeg = folder / f"jpeg-{q}.jpg"
                Image.fromarray(encoded).save(jpeg, quality=q, subsampling=2, optimize=False, progressive=False)
                with Image.open(jpeg) as image:
                    measure(("jpeg", q, 0), np.array(image.convert("RGB")), {"jpeg": str(jpeg.relative_to(OUT))})
            for index, p in enumerate((.7, .3, .1, .035)):
                side = round(128 * np.sqrt(p))
                start = (128 - side) // 2
                box = [start, start, start + side, start + side]
                measure(("crop", p, 0), encoded[start:start+side, start:start+side], {"box": box, "realized_retention": side ** 2 / 16384})
                for draw in range(20):
                    seed = SEED + image_index * 100000 + 10000 + index * 1000 + draw
                    x, y = np.random.default_rng(seed).integers(0, 129-side, size=2).tolist()
                    measure(("random_crop", p, draw), encoded[y:y+side, x:x+side], {"seed": seed,
                        "box": [x, y, x+side, y+side], "realized_retention": side ** 2 / 16384})
            for index, p in enumerate((.7, .3, .1)):
                for draw in range(20):
                    seed = SEED + image_index * 100000 + 20000 + index * 1000 + draw
                    mask = np.random.default_rng(seed).random((128, 128)) < p
                    mask_path = folder / f"mask-{p:g}-{draw:02d}.npy"
                    np.save(mask_path, mask)
                    measure(("dropout", p, draw), np.where(mask[:, :, None], encoded, cover),
                        {"seed": seed, "mask": str(mask_path.relative_to(OUT)), "realized_retention": float(mask.mean())})
            for sigma in (.5, 1., 2., 4.):
                measure(("blur", sigma, 0), blur(encoded, sigma), {"radius": int(np.ceil(3*sigma)), "padding": "reflect"})
            print(name, model_name, "PSNR", quality_row["psnr_db"], flush=True)
    settings = {"message": MESSAGE, "message_bits": [int(b) for b in MESSAGE], "threshold": .5,
        "crop_xyxy": [192, 192, 320, 320], "source_shape": [512, 512], "working_shape": [128, 128],
        "seed": SEED, "draws": 20, "device": "cpu", "dtype": "float32", "threads": 2,
        "normalization": "uint8 / 127.5 - 1", "serialization": "clip [-1,1]; (x+1)*127.5; numpy.rint (ties to even); uint8; reopen PNG",
        "jpeg": {"subsampling": 2, "optimize": False, "progressive": False, "libjpeg": features.version_codec("jpg")},
        "blur": "separable Gaussian; radius ceil(3*sigma); normalized float32 kernel; reflection padding; numpy.rint",
        "seed_offsets": "image_index*100000; random crop +10000; dropout +20000; strength_index*1000; +draw; shared across models",
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
        "commit": "556f76dd0602e4351ed4c19bd2ee87d50ef3c0de", "models": MODELS, "inputs": INPUTS}
    for filename, data in (("settings.json", settings), ("results.json", rows), ("quality.json", quality), ("example_values.json", example)):
        (OUT / filename).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    for filename, data in (("measurements.csv", rows), ("quality.csv", quality)):
        fields = list(dict.fromkeys(key for row in data for key in row))
        with (OUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    print(f"Saved {len(rows)} recovery rows, {len(quality)} quality rows to {OUT}")


if __name__ == "__main__":
    main()
