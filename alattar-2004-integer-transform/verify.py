# Run from the repository root: python alattar-2004-integer-transform/verify.py
from itertools import product
import json
import math
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

from independent_decode import Arrangement, recover
from transform import embed, forward, inverse, safe

ROOT: Final = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT.parent))
from experiment_support import verify_inputs
INPUTS = ROOT / "outputs/inputs"


def check_signed_threshold() -> None:
    image = np.tile(np.array([[100, 100, 150, 82, 80, 148],
                              [100, 100, 147, 80, 83, 150]], dtype=np.int64), (16, 6))
    result = embed(image, "quads", 24, np.random.default_rng(200408))
    assert result.selected == 192, "Signed threshold must include safe negative differences and exclude positive differences above 24"
    decoded = recover(result.image, "quads", 24)
    assert np.array_equal(decoded.pixels, image)
    assert np.array_equal(decoded.payload, result.payload)
    print("Signed threshold: safe negative differences selected; positive over-threshold differences excluded PASS")


def read_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        assert image.mode == "L" and image.size == (512, 512)
        return np.array(image, dtype=np.int64)


def check_core() -> None:
    alphabet = (0, 1, 2, 127, 128, 253, 254, 255)
    total = 0
    for n in (3, 4):
        vectors = np.array(list(product(alphabet, repeat=n)), dtype=np.int64)
        means, differences = forward(vectors)
        assert np.array_equal(inverse(means, differences), vectors)
        for expanding in (False, True):
            baseline = 2 * differences if expanding else 2 * (differences // 2)
            answer = safe(means, baseline)
            for index, values in enumerate(vectors.tolist()):
                m = sum(values) // n
                delta = [value - values[0] for value in values[1:]]
                good = True
                for bits in product((0, 1), repeat=n - 1):
                    trial = [2 * (d if expanding else d // 2) + b for d, b in zip(delta, bits, strict=True)]
                    first = m - sum(trial) // n
                    original = [first] + [first + d for d in trial]
                    good = good and all(0 <= value <= 255 for value in original)
                assert bool(answer[index]) == good
                total += 1
    print(f"Core: integer inverse and {total} independently evaluated safety decisions PASS")


def main() -> None:
    check_signed_threshold()
    check_core()
    first, second = (ROOT / "outputs" / name for name in ("signed-run1", "signed-run2"))
    paths = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    assert paths == sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    for path in paths:
        assert (first / path).read_bytes() == (second / path).read_bytes(), f"Repeat differs: {path}"
    print(f"Repeat: all {len(paths)} result files are byte-identical PASS")
    verify_inputs()
    from run import THRESHOLDS
    rows = json.loads((first / "metrics.json").read_text())
    expected_cases = {(name, n, t) for name in ("Airplane", "Boat") for n in (3, 4) for t in THRESHOLDS}
    assert {(r["image"], r["vector_size"], r["threshold"]) for r in rows} == expected_cases
    assert len(rows) == len(expected_cases)
    for row in rows:
        folder = first / row["case"]
        original = read_pixels(INPUTS / f'{row["image"].lower()}-gray.png')
        marked = read_pixels(folder / "marked.png")
        decoded = marked.copy()
        arrangements: tuple[Arrangement, ...] = ("columns", "rows") if row["vector_size"] == 3 else ("quads",)
        payload_count = selected = changeable = overhead = bit_errors = 0
        with np.load(folder / "payloads.npz") as expected:
            for index in reversed(range(len(arrangements))):
                result = recover(decoded, arrangements[index], row["threshold"])
                decoded = result.pixels
                message = expected[f"pass{index}"]
                assert message.shape == result.payload.shape
                bit_errors += int(np.count_nonzero(message != result.payload))
                payload_count += len(result.payload)
                selected += result.selected
                changeable += result.changeable
                overhead += result.overhead
        pixel_errors = int(np.count_nonzero(decoded != original))
        assert pixel_errors == bit_errors == 0, row["case"]
        assert np.array_equal(decoded, read_pixels(folder / "restored.png"))
        squared_error = sum(int(x) for x in np.square(marked - original).reshape(-1))
        psnr = 10 * math.log10(255 ** 2 / (squared_error / original.size))
        assert math.isclose(psnr, row["psnr_db"], abs_tol=1e-12)
        assert row["sse"] == squared_error and row["payload_bits"] == payload_count
        assert row["bpp"] == payload_count / original.size
        assert row["passes"] == len(arrangements)
        assert row["selected_vectors"] == selected and row["changeable_vectors"] == changeable
        assert row["map_header_bits"] == overhead
        assert row["restored_pixel_errors"] == pixel_errors and row["payload_bit_errors"] == bit_errors
        print(f'{row["case"]}: {payload_count} bits; {psnr:.8f} dB; independent pixels/bits PASS')
    print(f"PASS: all numeric metric fields recomputed; {len(rows)} cases restored exactly")


if __name__ == "__main__":
    main()
