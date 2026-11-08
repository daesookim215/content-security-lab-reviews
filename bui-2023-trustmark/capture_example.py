"""Capture the release's resolution-scaling arrays without modifying TrustMark.

Run in the experiment directory: python capture_example.py run1 experiment/results/example
The run directory must contain results.json and its saved images.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

from run_experiment import ExistingModelsTrustMark
from trustmark import TrustMark


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    settings = json.loads((run / "results.json").read_text())["settings"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    torch.set_num_interop_threads(settings["interop_threads"])
    torch.use_deterministic_algorithms(True)
    model = ExistingModelsTrustMark(
        device="cpu", model_type=settings["model"], use_ECC=True,
        encoding_type=TrustMark.Encoding.BCH_5, verbose=False,
        loadRemover=False, loadBBoxDetector=False,
    )
    package = Path(sys.modules[TrustMark.__module__].__file__).resolve().parent
    for name, digest in settings["source_sha256"].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
    for name, digest in settings["checkpoint_sha256"].items():
        assert hashlib.sha256((package / "models" / name).read_bytes()).hexdigest() == digest
    assert model.model_resolution_enc == settings["encoder_size"]
    assert model.concentrate_wm_region == settings["concentrate_wm_region"]
    with Image.open(run / "images/boat-2048-input.png") as image:
        cover_image = image.convert("RGB")
    working_image = cover_image.resize((model.model_resolution_enc,) * 2, Image.Resampling.BILINEAR)
    from torchvision.transforms.functional import to_tensor

    cover = to_tensor(working_image).unsqueeze(0) * 2.0 - 1.0
    packet = model.ecc.encode_binary([settings["payload"]])
    assert "".join(str(int(bit)) for bit in packet[0]) == settings["expected_packet"]
    with torch.no_grad():
        prediction, _ = model.encoder(cover, torch.from_numpy(packet).float())
        prediction = prediction.clamp(-1, 1)
        raw = prediction - cover
        mean = raw.mean(dim=(2, 3), keepdim=True)
        working = raw - mean
        full = torch.nn.functional.interpolate(working, size=cover_image.size[::-1], mode=settings["merge"])
    full_array = full.permute(0, 2, 3, 1).numpy().astype("f4")[0]
    original = np.asarray(cover_image)
    merged = np.clip(full_array * settings["strength"] + original / 127.5 - 1.0, -1, 1) * 127.5 + 127.5
    marked = model.put_the_image_after_processing(merged, original).astype(np.uint8)
    with Image.open(run / "images/boat-2048-marked.png") as saved:
        assert np.array_equal(marked, np.asarray(saved)), "Recomputed marked image differs from the saved run"
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {
        "cover": original, "working_cover": cover[0].permute(1, 2, 0).numpy(),
        "prediction": prediction[0].permute(1, 2, 0).numpy(),
        "channel_mean": mean[0, :, 0, 0].numpy(),
        "residual_256": working[0].permute(1, 2, 0).numpy(),
        "residual_2048": full_array, "marked": marked,
    }
    for name, array in arrays.items():
        np.save(destination / f"{name}.npy", array, allow_pickle=False)
    metadata = {
        "settings": settings, "array_layout": "height, width, RGB channel",
        "residual_units": "normalized encoder units; multiply by 127.5 for 8-bit pixel units",
        "saved_marked_match": True,
        "array_sha256": {name: hashlib.sha256((destination / f"{name}.npy").read_bytes()).hexdigest() for name in arrays},
    }
    (destination / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("PASS: CPU-only intermediates saved; marked pixels exactly match run1 Boat 2048")


if __name__ == "__main__":
    main()
