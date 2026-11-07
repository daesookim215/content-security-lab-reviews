# Run from the repository root: python alattar-2004-integer-transform/run.py
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Final

import numpy as np
from PIL import Image

from transform import Layout, decode, embed

ROOT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from experiment_support import prepare_inputs
INPUTS = ROOT / "outputs/inputs"
SEED: Final = 200408
THRESHOLDS: Final = (12, 16, 24, 32, 48, 64)
SOURCES: Final = (("Airplane", "4.2.05.tiff"), ("Boat", "boat.512.tiff"))


@dataclass(frozen=True, slots=True)
class Result:
    image: str
    vector_size: int
    threshold: int
    passes: int
    payload_bits: int
    bpp: float
    sse: int
    psnr_db: float
    restored_pixel_errors: int
    payload_bit_errors: int
    map_header_bits: int
    selected_vectors: int
    changeable_vectors: int
    case: str


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ('signed-run1', 'signed-run2'):
        raise SystemExit("Choose a working run: signed-run1, signed-run2")
    destination = ROOT / "outputs" / sys.argv[1]
    destination.mkdir(parents=True, exist_ok=False)
    prepare_inputs(ROOT)
    originals = INPUTS
    results = []
    for image_id, (name, filename) in enumerate(SOURCES):
        with Image.open(originals / f"{name.lower()}-gray.png") as picture:
            original = np.array(picture.convert("L"), dtype=np.int64)
        assert original.shape == (512, 512)
        Image.fromarray(original.astype(np.uint8)).save(originals / f"{name.lower()}-gray.png")
        for n in (3, 4):
            layouts: tuple[Layout, ...] = ("columns", "rows") if n == 3 else ("quads",)
            for threshold in THRESHOLDS:
                case = f"{name.lower()}-n{n}-t{threshold}"
                folder = destination / case
                folder.mkdir()
                rng = np.random.default_rng(np.random.SeedSequence([SEED, image_id, n, threshold]))
                current = original.copy()
                passes = []
                payloads = {}
                for number, layout in enumerate(layouts):
                    result = embed(current, layout, threshold, rng)
                    current = result.image
                    passes.append(result)
                    payloads[f"pass{number}"] = result.payload
                Image.fromarray(current.astype(np.uint8)).save(folder / "marked.png")
                np.savez(folder / "payloads.npz", **payloads)
                recovered = current.copy()
                bit_errors = 0
                for number in reversed(range(len(layouts))):
                    recovered, message = decode(recovered, layouts[number])
                    assert message.shape == passes[number].payload.shape
                    bit_errors += int(np.count_nonzero(message != passes[number].payload))
                pixel_errors = int(np.count_nonzero(original != recovered))
                assert pixel_errors == bit_errors == 0
                Image.fromarray(recovered.astype(np.uint8)).save(folder / "restored.png")
                payload_bits = sum(len(item.payload) for item in passes)
                sse = int(np.square(current - original).sum())
                row = Result(name, n, threshold, len(layouts), payload_bits,
                             payload_bits / original.size, sse,
                             float(10 * np.log10(255 ** 2 * original.size / sse)),
                             pixel_errors, bit_errors, sum(item.map_bits for item in passes),
                             sum(item.selected for item in passes), sum(item.changeable for item in passes), case)
                results.append(asdict(row))
                print(f"{case}: {payload_bits} bits, {row.psnr_db:.4f} dB, exact restoration")
    (destination / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
