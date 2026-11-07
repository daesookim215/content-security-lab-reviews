# Run from the repository root: python chen-2001-qim/run.py
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

from qim import Key, decode, embed, psnr

ROOT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from experiment_support import prepare_inputs
INPUTS = ROOT / "outputs/inputs"
SEED: Final = 2001
STEPS: Final = (8, 16, 32, 64)
JPEG: Final = (90, 70, 50, 30)
NOISE: Final = (1, 3, 6)
IMAGES: Final = (('Airplane', '4.2.05.tiff'), ('Boat', 'boat.512.tiff'))


@dataclass(frozen=True, slots=True)
class Row:
    image: str
    step: int
    attack: str
    file: str
    errors: int
    bits: int
    ber: float
    psnr_db: float


def prepare_images() -> None:
    prepare_inputs(ROOT)


def run(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (name, _) in enumerate(IMAGES):
        rng = np.random.default_rng(SEED + index)
        bits = rng.integers(0, 2, 4096, dtype=np.uint8)
        dither = rng.uniform(-0.5, 0.5, 4096)
        noise = rng.normal(0, 1, (512, 512))
        np.save(folder / f'{name}-bits.npy', bits)
        np.save(folder / f'{name}-dither.npy', dither)
        np.save(folder / f'{name}-noise.npy', noise)
        with Image.open(INPUTS / f'{name.lower()}.png') as image:
            original = np.array(image)
        for step in STEPS:
            prefix = f'{name}-d{step}'
            key = Key(bits, dither, step)
            marked, floating = embed(original, key)
            np.save(folder / f'{prefix}-coefficients.npy', floating)
            Image.fromarray(marked).save(folder / f'{prefix}-clean.png')
            cases = [('clean', f'{prefix}-clean.png')]
            for quality in JPEG:
                filename = f'{prefix}-jpeg{quality}.jpg'
                Image.fromarray(marked).save(folder / filename, quality=quality,
                                             subsampling=0, optimize=False, progressive=False)
                cases.append((f'jpeg{quality}', filename))
            for sigma in NOISE:
                filename = f'{prefix}-noise{sigma}.png'
                attacked = np.clip(np.rint(marked.astype(float) + sigma * noise), 0, 255).astype(np.uint8)
                Image.fromarray(attacked).save(folder / filename)
                cases.append((f'noise{sigma}', filename))
            for attack, filename in cases:
                with Image.open(folder / filename) as image:
                    received = np.array(image)
                decoded = decode(received, key)
                np.save(folder / f'{prefix}-{attack}-decoded.npy', decoded)
                errors = int(np.count_nonzero(decoded != bits))
                rows.append(Row(name, step, attack, filename, errors, len(bits),
                                errors / len(bits), psnr(original, received)))
    payload = {'seed': SEED, 'steps': STEPS, 'jpeg_qualities': JPEG, 'noise_sigmas': NOISE,
               'size': [512, 512], 'block': [8, 8], 'coefficient': [2, 3], 'payload_bits': 4096,
               'rows': [asdict(row) for row in rows]}
    (folder / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    with (folder / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    print(f'{folder.name}: {len(rows)} conditions saved')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('run1', 'run2'):
        raise SystemExit("Choose a working run: run1, run2")
    prepare_images()
    run(ROOT / 'outputs' / sys.argv[1])
