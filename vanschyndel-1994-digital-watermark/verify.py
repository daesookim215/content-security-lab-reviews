# Run from the repository root: python vanschyndel-1994-digital-watermark/verify.py

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
RUN: Final = ROOT / "outputs/run1"


def pixels(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        assert image.size == (512, 512) and image.mode == "L", path
        return np.array(image)


def independent_decode(received: NDArray[np.uint8], sequence: NDArray[np.uint8],
                       *, addition: bool) -> NDArray[np.int64]:
    """Use direct integer dot products, independently of the FFT implementation."""
    templates = np.stack([np.roll(sequence, shift) for shift in range(512)])
    signed = templates.astype(np.int32) * 2 - 1
    observed = received.astype(np.int32) if addition else np.remainder(received, 2).astype(np.int32)
    scores = observed @ signed.T
    if addition:
        filtered = np.empty_like(scores)
        for shift in range(512):
            filtered[:, shift] = (4 * scores[:, shift] - scores[:, (shift - 2) % 512]
                                  - scores[:, (shift - 1) % 512] - scores[:, (shift + 1) % 512]
                                  - scores[:, (shift + 2) % 512])
        scores = filtered
    return np.argmax(scores, axis=1)


def payload_digits(symbols: NDArray[np.int64]) -> NDArray[np.uint8]:
    return np.array([[int(bit) for bit in f"{int(symbol):09b}"] for symbol in symbols], dtype=np.uint8)


def main() -> None:
    other = ROOT / "outputs/run2"
    files = sorted(p.name for p in RUN.iterdir())
    assert files == sorted(p.name for p in other.iterdir())
    for filename in files:
        assert (RUN / filename).read_bytes() == (other / filename).read_bytes(), filename
    verify_inputs()
    sequence = np.load(RUN / "sequence.npy")
    assert sequence.shape == (512,) and set(sequence.tolist()) == {0, 1}
    assert int(sequence.sum()) == 256
    windows = {tuple(int(sequence[(start + offset) % 512]) for offset in range(9))
               for start in range(512)}
    assert len(windows) == 512
    zero_run = next(start for start in range(512)
                    if all(sequence[(start + offset) % 512] == 0 for offset in range(9)))
    reduced = np.delete(sequence, zero_run)
    assert len({tuple(np.roll(reduced, -i)[:9]) for i in range(511)}) == 511
    for i in range(511):
        assert reduced[(i + 9) % 511] == (reduced[i] ^ reduced[(i + 4) % 511])
    phases = np.load(RUN / "payload-phases.npy")
    config = json.loads((RUN / "config.json").read_text())
    np.testing.assert_array_equal(phases, np.random.default_rng(config["seed"]).integers(0, 512, 512))
    truth = payload_digits(phases)
    np.testing.assert_array_equal(truth, np.load(RUN / "payload-bits.npy"))
    mark = np.stack([np.roll(sequence, int(shift)) for shift in phases])
    np.testing.assert_array_equal(mark, np.load(RUN / "watermark.npy"))
    rows = json.loads((RUN / "metrics.json").read_text())
    assert len(rows) == 24
    expected_cases = {(name, method, attack) for name in ("airplane", "boat")
                      for method in ("coding", "addition")
                      for attack in ["clean", *(f"jpeg{q}" for q in config["jpeg_qualities"])]}
    assert {(r["image"], r["method"], r["attack"]) for r in rows} == expected_cases
    for row in rows:
        name, method, attack = row["image"], row["method"], row["attack"]
        original = pixels(RUN / f"{name}-original.png")
        received = pixels(RUN / f"{name}-{method}-{attack}.png")
        clean = pixels(RUN / f"{name}-{method}-clean.png")
        if method == "coding":
            lookup = np.load(RUN / f"{name}-histogram-lookup.npy")
            assert np.all(lookup % 2 == 0)
            assert len(np.unique(lookup[original])) <= 128
            np.testing.assert_array_equal(clean, lookup[original] + mark)
            np.testing.assert_array_equal(clean % 2, mark)
        if method == "addition":
            expected = np.minimum(original.astype(np.int32) + mark, 255)
            np.testing.assert_array_equal(clean, expected)
            restored = np.maximum(clean.astype(np.int32) - mark, 0).astype(np.uint8)
            np.testing.assert_array_equal(restored, pixels(RUN / f"{name}-addition-restored.png"))
        if attack != "clean":
            with Image.open(RUN / f"{name}-{method}-{attack}.jpg") as jpeg:
                np.testing.assert_array_equal(received, np.array(jpeg.convert("L")))
        decoded = independent_decode(received, sequence, addition=method == "addition")
        np.testing.assert_array_equal(decoded, np.load(RUN / f"{name}-{method}-{attack}-decoded.npy"))
        squared_error = sum(int(v) ** 2 for v in (original.astype(np.int32) - received).ravel())
        expected_psnr = 10 * math.log10(255 * 255 * original.size / squared_error)
        recomputed = {
            "psnr_db": expected_psnr,
            "payload_bit_accuracy": float(np.count_nonzero(payload_digits(decoded) == truth) / truth.size * 100),
            "phase_accuracy": float(np.count_nonzero(decoded == phases) / len(phases) * 100),
            "lsb_bit_accuracy": float(np.count_nonzero(received % 2 == mark) / mark.size * 100),
            "restored_pixel_accuracy": float(np.mean(received.astype(np.int32) - mark == original) * 100),
            "saturated_pixels": int(np.count_nonzero(original.astype(np.int32) + mark > 255)) if method == "addition" else 0,
        }
        for key, value in recomputed.items():
            assert math.isclose(value, row[key], rel_tol=0, abs_tol=1e-11), (name, method, attack, key)
    for name in ("airplane", "boat"):
        clean_coding = next(r for r in rows if (r["image"], r["method"], r["attack"]) == (name, "coding", "clean"))
        assert clean_coding["payload_bit_accuracy"] == clean_coding["lsb_bit_accuracy"] == 100
        original = pixels(RUN / f"{name}-original.png")
        restored = pixels(RUN / f"{name}-addition-restored.png")
        np.testing.assert_array_equal(restored, original)
    print(f"PASS: {len(files)} files per run are byte-identical; all saved working files checked.")
    print("PASS: both source image checksums match.")
    print("PASS: 511-state maximal recurrence and 512-state extended sequence independently checked.")
    print("PASS: all 24 cases independently decoded by integer dot products; every raw metric recomputed.")
    print("PASS: clean LSB coding has exact coded bits and payload; known-pattern subtraction restores both addition images exactly.")
    print("LIMIT: exact subtraction uses the known inserted pattern, not the imperfect addition-decoder output.")


if __name__ == "__main__":
    main()
