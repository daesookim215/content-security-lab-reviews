# Run from the repository root: python chen-2001-qim/verify.py
import csv
import io
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
RUN: Final = ROOT / 'outputs/run1'


def parity_decode(values: NDArray[np.float64], dither: NDArray[np.float64], step: float) -> NDArray[np.uint8]:
    return (np.floor(2 * (values + step * dither) / step + 0.5).astype(np.int64) % 2).astype(np.uint8)


def project(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    coordinate = np.arange(8) + 0.5
    horizontal = 0.5 * np.cos(np.pi * 2 * coordinate / 8)
    vertical = 0.5 * np.cos(np.pi * 3 * coordinate / 8)
    kernel = horizontal[:, None] * vertical[None, :]
    tiles = image.reshape(64, 8, 64, 8).transpose(0, 2, 1, 3)
    return np.sum(tiles * kernel, axis=(-2, -1)).ravel()


def load_image(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        assert image.mode == 'L' and image.size == (512, 512), str(path)
        return np.array(image)


def check_runs() -> int:
    first = {path.name: path for path in RUN.iterdir()}
    second = {path.name: path for path in (ROOT / 'outputs/run2').iterdir()}
    assert set(first) == set(second), 'Run file inventories differ'
    for name, path in first.items():
        assert path.read_bytes() == second[name].read_bytes(), f'Runs differ: {name}'
    return len(first)


def check_inputs() -> None:
    verify_inputs()


def check_core() -> None:
    metadata = json.loads((RUN / 'results.json').read_text())
    for index, name in enumerate(('Airplane', 'Boat')):
        dither = np.load(RUN / f'{name}-dither.npy')
        expected = np.load(RUN / f'{name}-bits.npy')
        generator = np.random.default_rng(metadata['seed'] + index)
        assert np.array_equal(generator.integers(0, 2, 4096, dtype=np.uint8), expected)
        assert np.array_equal(generator.uniform(-0.5, 0.5, 4096), dither)
        assert np.array_equal(generator.normal(0, 1, (512, 512)), np.load(RUN / f'{name}-noise.npy'))
        original = load_image(INPUTS / f'{name.lower()}.png')
        host = project(original)
        for step in (8, 16, 32, 64):
            marked = np.load(RUN / f'{name}-d{step}-coefficients.npy')
            assert np.array_equal(parity_decode(marked, dither, step), expected)
            assert float(np.max(np.abs(marked - host))) <= step / 2 + 1e-10
            for perturbation in (-0.249, 0.249):
                assert np.array_equal(parity_decode(marked + perturbation * step, dither, step), expected)
            crossed = parity_decode(marked + 0.251 * step, dither, step)
            assert np.array_equal(crossed, 1 - expected), 'Boundary crossing did not flip bits'


def check_rows() -> int:
    result = json.loads((RUN / 'results.json').read_text())
    with (RUN / 'results.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(result['rows']) == len(csv_rows) == 64
    for row, csv_row in zip(result['rows'], csv_rows, strict=True):
        name, step = row['image'], row['step']
        original = load_image(INPUTS / f'{name.lower()}.png')
        received = load_image(RUN / row['file'])
        bits = np.load(RUN / f'{name}-bits.npy')
        dither = np.load(RUN / f'{name}-dither.npy')
        prediction = parity_decode(project(received), dither, step)
        saved = np.load(RUN / f"{name}-d{step}-{row['attack']}-decoded.npy")
        assert np.array_equal(prediction, saved), row
        errors = int(np.sum(prediction != bits))
        mse = float(np.sum((original.astype(float) - received.astype(float)) ** 2) / original.size)
        measured_psnr = 10 * math.log10(255 * 255 / mse)
        assert errors == row['errors'] == int(csv_row['errors']), row
        assert len(bits) == row['bits'] == int(csv_row['bits']) == 4096
        assert errors / len(bits) == row['ber'] == float(csv_row['ber']), row
        assert math.isclose(measured_psnr, row['psnr_db'], abs_tol=1e-12), row
        assert row['psnr_db'] == float(csv_row['psnr_db']), row
        marked = load_image(RUN / f'{name}-d{step}-clean.png')
        attack = row['attack']
        if attack.startswith('noise'):
            sigma = int(attack.removeprefix('noise'))
            noise = np.load(RUN / f'{name}-noise.npy')
            rebuilt = np.clip(np.rint(marked.astype(float) + sigma * noise), 0, 255).astype(np.uint8)
            assert np.array_equal(received, rebuilt), row
        if attack.startswith('jpeg'):
            quality = int(attack.removeprefix('jpeg'))
            buffer = io.BytesIO()
            Image.fromarray(marked).save(buffer, format='JPEG', quality=quality,
                                         subsampling=0, optimize=False, progressive=False)
            assert buffer.getvalue() == (RUN / row['file']).read_bytes(), row
    return len(result['rows'])


def main() -> None:
    check_inputs()
    print('PASS: both TIFF checksums and grayscale preparations')
    files = check_runs()
    print(f'PASS: two separate runs identical byte for byte across {files} files')
    check_core()
    print('PASS: independent parity decoder recovers every floating-point embedded bit')
    print('PASS: scalar perturbations of +/-0.249 step preserve bits; +0.251 step flips bits')
    count = check_rows()
    print(f'PASS: independent cosine decoder, error counts, BER and PSNR for all {count} conditions')
    print('PASS: CSV agrees with JSON; saved JPEG and noise attacks regenerate exactly')


if __name__ == "__main__":
    main()
