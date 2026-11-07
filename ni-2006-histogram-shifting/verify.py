# Run from the repository root: python ni-2006-histogram-shifting/verify.py
import hashlib
import json
import math
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

ROOT: Final = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT.parent))
from experiment_support import verify_inputs
INPUTS = ROOT / "outputs/inputs"


def read_image(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        assert image.mode == "L" and image.size == (512, 512)
        return np.array(image)


def decode_independently(marked: NDArray[np.uint8], metadata: dict) -> tuple[NDArray[np.uint8], list[int], list[int]]:
    values = [int(value) for value in marked.ravel()]
    recovered = values.copy()
    stream = []
    for pair in metadata["points"]:
        peak, minimum = pair["peak"], pair["zero"]
        step = 1 if peak < minimum else -1
        for index, value in enumerate(values):
            if value == peak:
                stream.append(0)
            if value == peak+step:
                stream.append(1)
            if peak < minimum and peak < value <= minimum:
                recovered[index] = value-1
            if minimum < peak and minimum <= value < peak:
                recovered[index] = value+1
    cursor = 0

    def take(width: int) -> int:
        nonlocal cursor
        result = 0
        assert cursor+width <= len(stream)
        for bit in stream[cursor:cursor+width]:
            result = 2*result+bit
        cursor += width
        return result

    for pair, population in zip(metadata["points"], metadata["minimum_counts"], strict=True):
        if population:
            minimum, count = take(8), take(19)
            assert minimum == pair["zero"] and count == population
            for _ in range(count):
                recovered[take(18)] = minimum
    return np.array(recovered, dtype=np.uint8).reshape(marked.shape), stream[cursor:], stream


def verify_case(run: Path, row: dict) -> None:
    image_folder = run / row["image"].lower()
    case = image_folder / f"pairs-{row['pairs']}"
    original = read_image(image_folder / "original.png")
    marked = read_image(case / "marked.png")
    saved_restored = read_image(case / "restored.png")
    assert json.loads((case / "metadata.json").read_text()) == row
    histogram = np.bincount(original.ravel(), minlength=256)
    assert np.array_equal(histogram, np.load(image_folder / "histogram.npy"))
    counts = [int(histogram[pair["zero"]]) for pair in row["points"]]
    filename = "4.2.05.tiff" if row["image"] == "Airplane" else "boat.512.tiff"
    assert hashlib.sha256((ROOT.parent / "images" / filename).read_bytes()).hexdigest() == row["source_sha256"]
    assert counts == row["minimum_counts"]
    gross = sum(int(histogram[pair["peak"]]) for pair in row["points"])
    overhead = sum(27+18*count for count in counts if count)
    assert (gross, overhead, gross-overhead) == (row["gross_bits"], row["overhead_bits"], row["capacity_bits"])
    recovered, payload, stream = decode_independently(marked, row)
    raw_payload = np.load(case / "payload.npy")
    assert len(raw_payload) == gross-overhead
    assert np.array_equal(stream, np.load(case / "stream.npy"))
    assert np.array_equal(payload, raw_payload)
    assert np.array_equal(recovered, original)
    assert np.array_equal(saved_restored, original)
    delta = marked.astype(np.int64)-original.astype(np.int64)
    squared_error = int(np.dot(delta.ravel(), delta.ravel()))
    mse = squared_error/original.size
    assert squared_error == row["squared_error"]
    assert mse == row["mse"]
    assert math.isclose(10*math.log10(255*255/mse), row["psnr_db"], abs_tol=1e-12)
    assert int(np.count_nonzero(delta)) == row["changed_pixels"]
    assert int(np.max(np.abs(delta))) == row["max_abs_change"] == 1
    assert int(np.count_nonzero(original != recovered)) == row["restoration_errors"] == 0
    assert int(np.count_nonzero(raw_payload != payload)) == row["bit_errors"] == 0
    damaged_payload = raw_payload.copy()
    damaged_payload[-1] ^= 1
    assert not np.array_equal(payload, damaged_payload)
    damaged_image = marked.copy()
    peak = row["points"][-1]["peak"]
    step = 1 if peak < row["points"][-1]["zero"] else -1
    carrier = int(np.flatnonzero((marked == peak) | (marked == peak+step))[-1])
    damaged_image.ravel()[carrier] = peak+step if int(marked.ravel()[carrier]) == peak else peak
    _, damaged_decoded, _ = decode_independently(damaged_image, row)
    assert not np.array_equal(damaged_decoded, raw_payload)
    print(f"PASS {row['image']} {row['pairs']} pair(s): {gross-overhead} bits, {row['psnr_db']:.8f} dB; exact pixels and bits; corruption controls rejected")


def main() -> None:
    first, second = (ROOT / "outputs" / name for name in ("run1", "run2"))
    files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    assert files == sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert all((first/path).read_bytes() == (second/path).read_bytes() for path in files)
    verify_inputs()
    for run in (first, second):
        result = json.loads((run / "results.json").read_text())
        assert result["seed"] == 200603 and result["size"] == [512, 512]
        assert result["grayscale_bits"] == 8 and result["pair_counts"] == [1, 2]
        assert result["difference_factor"] == 64
        assert {(row["image"], row["pairs"]) for row in result["rows"]} == {(name, count) for name in ("Airplane", "Boat") for count in (1, 2)}
        for row in result["rows"]:
            verify_case(run, row)
    print(f"PASS deterministic repeat: {len(files)} files match byte for byte")
    print("PASS source checksums, grayscale conversion, histogram counts, capacity, overhead, MSE, PSNR, change counts, independent decoding, restoration")


if __name__ == "__main__":
    main()
