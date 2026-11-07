# Run from the repository root: python vanschyndel-1994-digital-watermark/run.py

import json
import sys
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
from PIL import Image

from method import (QUALITIES, SEED, Bytes, adaptive_even_lookup, bits_of, decode,
                    extended_sequence, payload, psnr, watermark)

ROOT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from experiment_support import prepare_inputs
INPUTS = ROOT / "outputs/inputs"


class Measurement(TypedDict):
    image: str
    method: str
    attack: str
    psnr_db: float
    payload_bit_accuracy: float
    phase_accuracy: float
    lsb_bit_accuracy: float
    restored_pixel_accuracy: float
    saturated_pixels: int


def measure(original: Bytes, received: Bytes, mark: Bytes, folder: Path,
            name: str, method_name: str, attack: str, saturation: int) -> Measurement:
    phases = decode(received, extended_sequence(), addition=method_name == "addition")
    np.save(folder / f"{name}-{method_name}-{attack}-decoded.npy", phases)
    row: Measurement = {
        "image": name, "method": method_name, "attack": attack,
        "psnr_db": psnr(original, received),
        "payload_bit_accuracy": float(np.mean(bits_of(phases) == bits_of(payload())) * 100),
        "phase_accuracy": float(np.mean(phases == payload()) * 100),
        "lsb_bit_accuracy": float(np.mean((received & 1) == mark) * 100),
        "restored_pixel_accuracy": float(np.mean(received.astype(np.int16)
                                                  - mark.astype(np.int16) == original) * 100),
        "saturated_pixels": saturation,
    }
    return row


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ('run1', 'run2'):
        raise SystemExit("Choose a working run: run1, run2")
    prepare_inputs(ROOT)
    run_name = sys.argv[1]
    assert run_name in ("run1", "run2")
    folder = ROOT / "outputs" / run_name
    folder.mkdir(parents=True, exist_ok=False)
    sequence, phases = extended_sequence(), payload()
    mark = watermark(sequence, phases)
    np.save(folder / "sequence.npy", sequence)
    np.save(folder / "payload-phases.npy", phases)
    np.save(folder / "payload-bits.npy", bits_of(phases))
    np.save(folder / "watermark.npy", mark)
    config = {"seed": SEED, "width": 512, "height": 512, "grayscale_bits": 8,
              "sequence_length": 512, "payload_bits_per_row": 9,
              "jpeg_qualities": list(QUALITIES), "difference_factor": 32,
              "filter": [-1, -1, 4, -1, -1], "histogram_groups": 128}
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    for name in ("airplane", "boat"):
        with Image.open(INPUTS / f"{name}.png") as image:
            original = np.array(image)
        Image.fromarray(original).save(folder / f"{name}-original.png")
        lookup = adaptive_even_lookup(original)
        np.save(folder / f"{name}-histogram-lookup.npy", lookup)
        coding = lookup[original] | mark
        wide_addition = original.astype(np.uint16) + mark
        saturated = int(np.sum(wide_addition > 255))
        addition = np.minimum(wide_addition, 255).astype(np.uint8)
        restored = np.maximum(0, addition.astype(np.int16) - mark).astype(np.uint8)
        Image.fromarray(restored).save(folder / f"{name}-addition-restored.png")
        for method_name, marked, overflow in (("coding", coding, 0),
                                               ("addition", addition, saturated)):
            prefix = f"{name}-{method_name}"
            Image.fromarray(marked).save(folder / f"{prefix}-clean.png")
            rows.append(measure(original, marked, mark, folder, name, method_name, "clean", overflow))
            for quality in QUALITIES:
                attack = f"jpeg{quality}"
                jpeg = folder / f"{prefix}-{attack}.jpg"
                Image.fromarray(marked).save(jpeg, quality=quality, optimize=False,
                                             progressive=False, subsampling=0)
                with Image.open(jpeg) as compressed:
                    received = np.array(compressed.convert("L"))
                Image.fromarray(received).save(folder / f"{prefix}-{attack}.png")
                rows.append(measure(original, received, mark, folder, name, method_name, attack, overflow))
    (folder / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(json.dumps([row for row in rows if row["attack"] == "clean"], indent=2))
    print(f"Saved {len(rows)} measurement rows to {folder.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
