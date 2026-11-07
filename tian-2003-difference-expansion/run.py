# Run from the repository root: python tian-2003-difference-expansion/run.py
import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

from tian import choose_plan, embed, extract

ROOT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from experiment_support import prepare_inputs
INPUTS = ROOT / "outputs/inputs"
SEED: Final = 2003815962
SIZES: Final = (16384, 32768, 65536, 98304)
IMAGES: Final = (('Airplane', '4.2.05.tiff'), ('Boat', 'boat.512.tiff'))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ('run-a', 'run-b'):
        raise SystemExit("Choose a working run: run-a, run-b")
    destination = ROOT / 'outputs' / sys.argv[1]
    destination.mkdir(parents=True, exist_ok=True)
    prepare_inputs(ROOT)
    rows = []
    for image_index, (name, _) in enumerate(IMAGES):
        with Image.open(INPUTS / f'{name.lower()}-gray.png') as source:
            original = np.array(source)
        for size in SIZES:
            rng = np.random.default_rng(np.random.SeedSequence([SEED, image_index, size]))
            payload = rng.integers(0, 2, size, dtype=np.uint8)
            plan = choose_plan(original, size)
            encoded = embed(original, payload, plan)
            restored, decoded = extract(encoded.image)
            assert np.array_equal(payload, decoded)
            assert np.array_equal(original, restored)
            case = f'{name.lower()}-{size}'
            folder = destination / case
            folder.mkdir(exist_ok=True)
            Image.fromarray(encoded.image).save(folder / 'embedded.png')
            Image.fromarray(restored).save(folder / 'restored.png')
            np.save(folder / 'payload.npy', payload, allow_pickle=False)
            np.save(folder / 'decoded.npy', decoded, allow_pickle=False)
            mse = float(np.mean((original.astype(np.float64)-encoded.image)**2))
            row = dict(case=case, image=name, payload_bits=size, bpp=size/original.size,
                psnr_db=float(10*np.log10(255**2/mse)), mse=mse,
                threshold=plan.threshold, map_bits=int(plan.map_bits.size),
                saved_lsb_bits=int(plan.saved.sum()), header_bits=16, length_bits=32,
                hash_bits=256, padding_bits=encoded.padding,
                gross_bits=int(plan.changeable.sum()), expanded_pairs=int(plan.selected.sum()),
                pixel_errors=int(np.count_nonzero(restored != original)),
                bit_errors=int(np.count_nonzero(decoded != payload)))
            rows.append(row)
            print(json.dumps(row), flush=True)
    (destination / 'results.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == "__main__":
    main()
