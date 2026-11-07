# Run from the repository root: python tian-2003-difference-expansion/verify.py
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from independent_decode import decode
from map_codec import decode_map, encode_map
from run import SEED, SIZES
from tian import make_plan

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT.parent))
from experiment_support import verify_inputs
INPUTS = ROOT / "outputs/inputs"


def pair_check() -> None:
    x, y = np.indices((256, 256), dtype=np.int32)
    mean, difference = (x+y)//2, x-y
    bound = np.minimum(2*(255-mean), 2*mean+1)
    expanded = np.ones(x.shape, dtype=bool)
    changeable = np.ones(x.shape, dtype=bool)
    for bit in (0, 1):
        for variant, mask in ((2*difference+bit, expanded),
                              (2*(difference//2)+bit, changeable)):
            a, b = mean+(variant+1)//2, mean-variant//2
            mask &= (a >= 0) & (a <= 255) & (b >= 0) & (b <= 255)
        expanded_h = 2*difference+bit
        assert np.array_equal(expanded_h//2, difference)
        assert np.all(expanded_h % 2 == bit)
    assert np.array_equal(expanded, np.maximum(abs(2*difference), abs(2*difference+1)) <= bound)
    assert np.array_equal(changeable, np.maximum(abs(2*(difference//2)), abs(2*(difference//2)+1)) <= bound)
    assert np.all(mean+(difference+1)//2 == x)
    assert np.all(mean-difference//2 == y)
    rng = np.random.default_rng(SEED)
    for bitmap in (np.zeros(131072, dtype=bool), np.ones(131072, dtype=bool),
                   rng.integers(0, 2, 131072).astype(bool)):
        packed = encode_map(bitmap)
        restored, used = decode_map(packed, bitmap.size)
        assert used == packed.size and np.array_equal(bitmap, restored)
    print('PASS: all 65,536 pixel pairs; negative differences; map endpoint cases')


def main() -> None:
    pair_check()
    a, b = ROOT / 'outputs/run-a', ROOT / 'outputs/run-b'
    files_a = sorted(p.relative_to(a) for p in a.rglob('*') if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob('*') if p.is_file())
    assert files_a == files_b
    for path in files_a:
        assert (a/path).read_bytes() == (b/path).read_bytes(), path
    print(f'PASS: {len(files_a)} saved files identical across separate runs')
    verify_inputs()
    rows = json.loads((a/'results.json').read_text())
    assert len(rows) == 8
    for row in rows:
        folder = a / row['case']
        image_index = ('Airplane', 'Boat').index(row['image'])
        with Image.open(INPUTS/f"{row['image'].lower()}-gray.png") as source:
            original = np.array(source)
        with Image.open(folder/'embedded.png') as source:
            embedded = np.array(source)
        with Image.open(folder/'restored.png') as source:
            saved_restored = np.array(source)
        restored, decoded = decode(embedded)
        payload = np.load(folder/'payload.npy', allow_pickle=False)
        saved_decoded = np.load(folder/'decoded.npy', allow_pickle=False)
        regenerated = np.random.default_rng(np.random.SeedSequence(
            [SEED, image_index, row['payload_bits']])).integers(0, 2, row['payload_bits'], dtype=np.uint8)
        assert np.array_equal(payload, regenerated)
        assert payload.size in SIZES
        assert np.array_equal(decoded, payload) and np.array_equal(saved_decoded, decoded)
        assert np.array_equal(restored, original) and np.array_equal(saved_restored, restored)
        assert row['pixel_errors'] == int(np.count_nonzero(restored != original)) == 0
        assert row['bit_errors'] == int(np.count_nonzero(decoded != payload)) == 0
        assert row['payload_bits'] == payload.size
        assert row['bpp'] == payload.size / original.size
        mse = float(np.mean((original.astype(np.float64)-embedded.astype(np.float64))**2))
        assert row['mse'] == mse
        assert math.isclose(row['psnr_db'], 10*math.log10(255**2/mse), abs_tol=1e-12)
        plan = make_plan(original, row['threshold'])
        assert row['map_bits'] == plan.map_bits.size
        assert row['saved_lsb_bits'] == plan.saved.sum()
        assert row['expanded_pairs'] == plan.selected.sum()
        assert row['gross_bits'] == plan.changeable.sum()
        assert row['header_bits'] == 16 and row['length_bits'] == 32 and row['hash_bits'] == 256
        assert row['padding_bits'] == plan.capacity-payload.size
        assert row['gross_bits'] == sum(row[key] for key in (
            'payload_bits','map_bits','saved_lsb_bits','header_bits','length_bits','hash_bits','padding_bits'))
        for lower in range(row['threshold']):
            assert make_plan(original, lower).capacity < payload.size
        print(f"PASS: {row['case']} {payload.size} bits, {row['bpp']:.4f} bpp, "
              f"{row['psnr_db']:.2f} dB; independent decode: 0 pixel / 0 bit errors")
    print('PASS: checksums, every raw metric, complete overhead, exact restoration and payload')


if __name__ == '__main__':
    main()
