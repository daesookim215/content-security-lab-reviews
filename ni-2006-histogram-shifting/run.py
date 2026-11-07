# Run from the repository root: python ni-2006-histogram-shifting/run.py
"""Create deterministic raw arrays, lossless images, metadata, and measurements."""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

from method import SEED, bookkeeping, embed, extract, select_pairs

ROOT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from experiment_support import prepare_inputs
INPUTS = ROOT / "outputs/inputs"
SOURCES: Final = (("Airplane", "4.2.05.tiff"), ("Boat", "boat.512.tiff"))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ('run1', 'run2'):
        raise SystemExit("Choose a working run: run1, run2")
    destination = ROOT / "outputs" / sys.argv[1]
    destination.mkdir(parents=True, exist_ok=False)
    prepare_inputs(ROOT)
    rng = np.random.Generator(np.random.PCG64(SEED))
    rows = []
    for name, filename in SOURCES:
        from fetch_images import ROOT as REPO
        source = REPO / "images" / filename
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with Image.open(source) as opened:
            original = np.array(opened.convert("L"), dtype=np.uint8)
        assert original.shape == (512, 512)
        folder = destination / name.lower()
        folder.mkdir()
        Image.fromarray(original).save(folder / "original.png")
        histogram = np.bincount(original.ravel(), minlength=256)
        np.save(folder / "histogram.npy", histogram, allow_pickle=False)
        for count in (1, 2):
            pairs = select_pairs(original, count)
            header = bookkeeping(original, pairs)
            gross = sum(int(histogram[pair.peak]) for pair in pairs)
            net = gross-len(header)
            assert net > 0
            payload = rng.integers(0, 2, net, dtype=np.uint8)
            stream = np.concatenate((header, payload))
            marked = embed(original, pairs, stream)
            occupied = tuple(bool(histogram[pair.zero]) for pair in pairs)
            restored, decoded = extract(marked, pairs, occupied)
            error = marked.astype(np.int32)-original.astype(np.int32)
            squared_error = int(np.sum(error**2, dtype=np.int64))
            mse = squared_error/original.size
            row = {"image": name, "pairs": count, "points": [asdict(p) for p in pairs],
                   "minimum_counts": [int(histogram[p.zero]) for p in pairs],
                   "gross_bits": gross, "overhead_bits": len(header), "capacity_bits": net,
                   "squared_error": squared_error, "mse": mse,
                   "psnr_db": float(10*np.log10(255**2/mse)),
                   "changed_pixels": int(np.count_nonzero(error)),
                   "max_abs_change": int(np.abs(error).max()),
                   "restoration_errors": int(np.count_nonzero(original != restored)),
                   "bit_errors": int(np.count_nonzero(payload != decoded)),
                   "source_sha256": digest}
            assert row["restoration_errors"] == row["bit_errors"] == 0
            assert row["max_abs_change"] <= 1
            case = folder / f"pairs-{count}"
            case.mkdir()
            for label, pixels in (("marked", marked), ("restored", restored)):
                Image.fromarray(pixels).save(case / f"{label}.png")
            for label, bits in (("payload", payload), ("stream", stream)):
                np.save(case / f"{label}.npy", bits, allow_pickle=False)
            (case / "metadata.json").write_text(json.dumps(row, indent=2)+"\n")
            rows.append(row)
    result = {"seed": SEED, "size": [512, 512], "grayscale_bits": 8,
              "pair_counts": [1, 2], "difference_factor": 64,
              "rows": rows}
    (destination / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
