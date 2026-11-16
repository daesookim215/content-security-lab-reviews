# /// script
# requires-python = ">=3.11"
# dependencies = ["trustmark==0.9.2", "torch", "torchvision", "numpy", "pillow"]
# ///
# How to run: sh reproduce_exp2.sh; reuses ../.venv without installing.
from __future__ import annotations

import importlib.metadata
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image
from trustmark import TrustMark

from attack import ifgsm
from baseline_run1 import ExistingModelsTrustMark, PAYLOAD, SEED
from perturbations import BY_NAME, KINDS, SETTINGS, Setting, geometry, jpeg, perturb
from records import Extra, Identity, Recorder, psnr, sha256


def main() -> None:
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    public = root.parent
    output = (root / sys.argv[1]).resolve()
    assert output.parent == root / "reproduced"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    output.mkdir(exist_ok=False)
    for directory in ("images", "thumbnails"):
        (output / directory).mkdir()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    reference = json.loads((public / "results" / "run1" / "results.json").read_text())
    model = ExistingModelsTrustMark(device="cpu", model_type="Q", use_ECC=True,
        encoding_type=TrustMark.Encoding.BCH_5, verbose=False, loadRemover=False,
        loadBBoxDetector=False)
    assert model.encoder is not None and model.decoder is not None
    assert model.schemaCapacity() == len(PAYLOAD) == 61
    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    packet = "".join(str(int(bit)) for bit in model.ecc.encode_binary([PAYLOAD])[0])
    assert packet == reference["settings"]["expected_packet"] and len(packet) == 100
    package = Path(sys.modules[TrustMark.__module__].__file__).resolve().parent
    for filename, expected in reference["settings"]["source_sha256"].items():
        assert sha256(package / filename) == expected
    for filename, expected in reference["settings"]["checkpoint_sha256"].items():
        assert sha256(package / "models" / filename) == expected
    rm_names = ("trustmark_rm_Q.yaml", "trustmark_rm_Q.ckpt")
    rm_files = {name: sorted((public / ".cache").rglob(name)) + list((package / "models").glob(name))
                for name in rm_names}
    removal_available = hasattr(model, "remove_watermark") and all(rm_files.values())
    removal = {"api_present": hasattr(model, "remove_watermark"),
        "searched": [os.path.relpath(public / ".cache", root),
                     os.path.relpath(package / "models", root)],
        "files": {name: [os.path.relpath(p, root) for p in paths] for name, paths in rm_files.items()},
        "status": "run" if removal_available else "skipped",
        "reason": "" if removal_available else "Q removal configuration and weights are absent from the released package files and local cache; no download attempted"}
    if removal_available:
        model.removal = model.load_model(str(rm_files[rm_names[0]][0]),
            str(rm_files[rm_names[1]][0]), "cpu", 100, part="remover")
        assert model.removal is not None
        model.removal.requires_grad_(False)
    recorder = Recorder(model, output, packet)
    inputs = json.loads((public / "images" / "manifest.json").read_text())
    pairs = []
    for entry in inputs:
        path = public / "images" / entry["filename"]
        assert sha256(path) == entry["sha256"]
        with Image.open(path) as source:
            native = source.convert("RGB")
        assert native.size == (512, 512)
        for size in (512, 2048):
            cover = native.copy() if size == 512 else native.resize((size, size), Image.Resampling.LANCZOS)
            name = entry["image"]
            marked = model.encode(cover, PAYLOAD, MODE="binary", WM_STRENGTH=1.0)
            identity = Identity(image=name, resolution=size, experiment="baseline", change="marked")
            recorder.record(marked, identity, Extra(psnr_input_db=psnr(marked, cover)))
            assert recorder.rows[-1]["image_sha256"] == reference["image_sha256"][f"images/{name}-{size}-marked.png"]
            recorder.record(cover, Identity(**{**identity, "experiment": "control", "change": "unmarked"}))
            pairs.append((identity, cover, marked))
            stage = time.perf_counter()
            for index, setting in enumerate(SETTINGS):
                seed = SEED + size * 1000 + inputs.index(entry) * 100 + index
                rng = np.random.default_rng(seed)
                changed = perturb(marked, setting, rng)
                rid = Identity(**{**identity, "experiment": "A", "change": setting.name})
                recorder.record(changed, rid, Extra(seed=seed))
                if setting.name == "jpeg40":
                    (output / "images" / f"A-{name}-{size}-jpeg40-source.jpg").write_bytes(jpeg(marked, 40))
                if size == 512 and setting.name == "motion9_45":
                    recorder.thumbnail(changed, f"A-{name}-{setting.name}")
            recorder.timings.append({"experiment": "A", "image": name, "resolution": size,
                                     "seconds": time.perf_counter() - stage})
            stage = time.perf_counter()
            for draw in range(20):
                seed = SEED + 100000 + size * 100 + inputs.index(entry) * 10000 + draw
                rng = np.random.default_rng(seed)
                changed, geometric = geometry(marked, rng)
                kinds = rng.choice(KINDS, 2, replace=False).tolist()
                selected = [str(rng.choice([s.name for s in SETTINGS if s.kind == kind])) for kind in kinds]
                for setting_name in selected:
                    changed = perturb(changed, BY_NAME[setting_name], rng)
                rid = Identity(**{**identity, "experiment": "B", "change": f"draw{draw:02}"})
                recorder.record(changed, rid, Extra(seed=seed, geometry=geometric, perturbations=selected))
                if size == 512 and draw == 0:
                    recorder.thumbnail(changed, f"B-{name}-draw00")
            recorder.timings.append({"experiment": "B", "image": name, "resolution": size,
                                     "seconds": time.perf_counter() - stage})
            if removal_available:
                stage = time.perf_counter()
                for tag, image in (("marked", marked), ("unmarked-control", cover)):
                    changed = model.remove_watermark(image)
                    rid = Identity(**{**identity, "experiment": "C", "change": tag})
                    recorder.record(changed, rid, Extra(psnr_input_db=psnr(changed, cover),
                                                         psnr_marked_db=psnr(changed, marked)))
                    if size == 512 and tag == "marked":
                        recorder.thumbnail(changed, f"C-{name}")
                recorder.timings.append({"experiment": "C", "image": name, "resolution": size,
                                         "seconds": time.perf_counter() - stage})
            stage = time.perf_counter()
            for strength in (0.5, 0.75, 1.0, 1.25, 1.5):
                image = model.encode(cover, PAYLOAD, MODE="binary", WM_STRENGTH=strength)
                for tag, changed in (("unchanged", image), ("jpeg50", perturb(image, Setting(
                        "jpeg50", "jpeg", 50, "run1 JPEG50"), np.random.default_rng(SEED)))):
                    rid = Identity(**{**identity, "experiment": "E", "change": f"lambda{strength}-{tag}"})
                    recorder.record(changed, rid, Extra(strength=strength, psnr_input_db=psnr(image, cover)))
                    if size == 512 and strength == 0.5 and tag == "jpeg50":
                        recorder.thumbnail(changed, f"E-{name}-lambda05-jpeg50")
            recorder.timings.append({"experiment": "E", "image": name, "resolution": size,
                                     "seconds": time.perf_counter() - stage})
    attack_gate = None
    previous = (root / sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    if previous is not None:
        attack_gate = json.loads(previous.read_text())["attack_2048_enabled"]
    for identity, cover, marked in sorted(pairs, key=lambda pair: pair[0]["resolution"]):
        if identity["resolution"] == 2048 and attack_gate is not True:
            continue
        attacked = ifgsm(model, marked, packet)
        recorder.record(attacked.image, Identity(**{**identity, "experiment": "D", "change": "ifgsm"}),
                        Extra(**attacked.info, psnr_marked_db=psnr(attacked.image, marked)))
        recorder.timings.append({"experiment": "D", "image": identity["image"],
                                 "resolution": identity["resolution"], "seconds": attacked.seconds})
        if identity["resolution"] == 512:
            recorder.thumbnail(attacked.image, f"D-{identity['image']}-ifgsm")
            if attack_gate is None:
                attack_gate = attacked.seconds < 300
    recorder.export()
    settings = {"seed": SEED, "payload": PAYLOAD, "expected_packet": packet,
        "resolutions": [512, 2048], "model": "Q", "bch_mode": "BCH_5", "strength": 1.0,
        "resampling": "Lanczos", "device": "cpu", "threads": 8, "interop_threads": 1,
        "decoder_size": model.model_resolution_dec, "encoder_size": model.model_resolution_enc,
        "concentrate_wm_region": model.concentrate_wm_region,
        "checkpoint_sha256": reference["settings"]["checkpoint_sha256"],
        "source_sha256": reference["settings"]["source_sha256"],
        "single_changes": [asdict(s) for s in SETTINGS],
        "package_version": importlib.metadata.version("trustmark"), "torch_version": torch.__version__,
        "pillow_version": PIL.__version__, "numpy_version": np.__version__,
        "attack_loss": "minimize BCEWithLogits toward flipped 100-bit ECC packet", "combined_draws": 20}
    data = {"settings": settings, "inputs": inputs, "removal": removal, "rows": recorder.rows,
        "summary": recorder.summary(), "attack_2048_enabled": attack_gate,
        "attack_2048_reason": "first 512 attack under 300 s" if attack_gate else "first 512 attack at least 300 s",
        "baseline_marked_hashes_match_public_reference": True,
        "image_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted((output / "images").iterdir())},
        "script_sha256": {p.name: sha256(p) for p in sorted(root.glob("*.py"))}}
    (output / "results.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (output / "timings.json").write_text(json.dumps({"total_seconds": time.perf_counter() - started,
        "measurements": recorder.timings}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
