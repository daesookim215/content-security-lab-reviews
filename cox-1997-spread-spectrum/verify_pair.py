# Run from the repository root: python cox-1997-spread-spectrum/verify_pair.py
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

from records import CASES, PARAMETERS, SEED

ROOT: Final = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT.parent))
INPUTS = ROOT / "outputs/inputs"
NAMES: Final = ('sipi_airplane', 'sipi_boat')


def main() -> None:
    log = []
    runs = []
    for run_id in ('pair_001', 'pair_002'):
        with (ROOT / 'outputs' / run_id / 'responses.csv').open(newline='') as stream:
            runs.append(list(csv.DictReader(stream)))
    rows, repeated = runs
    assert len(rows) == len(repeated) == 20000
    for left, right in zip(rows, repeated, strict=True):
        assert {k: v for k, v in left.items() if k != 'run_id'} == {
            k: v for k, v in right.items() if k != 'run_id'}
    log.append('PASS repeatability: 20000 rows; all scientific columns identical, excluding only run_id')
    summary = json.loads((ROOT / 'outputs/pair_summary.json').read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[(row['image'], row['experiment'])].append(row)
        assert int(row['seed']) == SEED
        parameters = json.loads(row['parameters'])
        assert all(parameters[key] == value for key, value in PARAMETERS.items())
        assert row['above_6'] == str(float(row['similarity']) > parameters['threshold'])
    expected_images = {}
    for name in NAMES:
        expected_cases = {}
        for case in CASES:
            group = groups[(name, case.name)]
            assert len(group) == 1000
            assert {int(r['candidate_index']) for r in group} == set(range(1000))
            embedded = sorted((r for r in group if r['embedded'] == 'True'), key=lambda r: int(r['mark_index']))
            multiple = case.name in ('rewatermark', 'collusion')
            assert len(embedded) == (5 if multiple else 1)
            assert [int(r['candidate_index']) for r in embedded] == ([99, 299, 499, 699, 899] if multiple else [199])
            unrelated = [float(r['similarity']) for r in group if r['embedded'] == 'False']
            fidelity = {float(r['psnr_db']) for r in group}
            assert len(fidelity) == 1
            expected_cases[case.name] = {
                'embedded_responses': [float(r['similarity']) for r in embedded],
                'embedded_indices': [int(r['candidate_index']) for r in embedded],
                'psnr_db': fidelity.pop(), 'unrelated_max': max(unrelated),
                'unrelated_count': len(unrelated), 'unrelated_above_6': sum(x > 6 for x in unrelated),
                'paper_response': case.paper_response,
            }
        expected_images[name] = {
            'cases': expected_cases,
            'unrelated_count': sum(c['unrelated_count'] for c in expected_cases.values()),
            'unrelated_max': max(c['unrelated_max'] for c in expected_cases.values()),
            'unrelated_above_6': sum(c['unrelated_above_6'] for c in expected_cases.values()),
        }
    params = json.loads(rows[0]['parameters'])
    rms = max(255 * 10 ** (-expected_images[n]['cases']['uniqueness']['psnr_db'] / 20) for n in NAMES)
    expected = {
        'images': expected_images, 'threshold': params['threshold'], 'shape': params['shape'],
        'difference_amplification': max(1, 2 ** math.floor(math.log2(32 / rms))),
        'unrelated_count': sum(v['unrelated_count'] for v in expected_images.values()),
        'unrelated_max': max(v['unrelated_max'] for v in expected_images.values()),
        'unrelated_above_6': sum(v['unrelated_above_6'] for v in expected_images.values()),
        'response_count': len(rows),
        'jpeg_quality': {c: int(json.loads(groups[(NAMES[0], c)][0]['parameters'])['operation']
                               .split('quality=')[1].split(',')[0]) for c in ('jpeg10', 'jpeg5', 'jpeg10_crop')},
    }
    assert summary == expected
    log.append('PASS summary: every numeric JSON field independently recomputed from CSV; paper values checked against records.py')
    image_seeds = [int(c.generate_state(1)[0]) for c in np.random.SeedSequence(SEED).spawn(2)]
    max_score_error = 0.0
    for run_id in ('pair_001', 'pair_002'):
        run = ROOT / 'outputs' / run_id
        manifest = json.loads((run / 'manifest.json').read_text())
        assert manifest['seed'] == SEED and manifest['parameters'] == PARAMETERS
        assert manifest['image_seeds'] == image_seeds
        assert [s['name'] for s in manifest['images']] == list(NAMES)
        for spec, image_seed in zip(manifest['images'], image_seeds, strict=True):
            assert hashlib.sha256((ROOT.parent / 'images' / spec['filename']).read_bytes()).hexdigest() == spec['sha256']
            directory = run / 'images' / spec['name']
            source = json.loads((directory / 'source.json').read_text())
            assert source['display'] == spec['display'] and source['image_seed'] == image_seed
            with Image.open(directory / 'original.png') as raster:
                original = np.asarray(raster, dtype=np.float64)
            assert original.shape == (256, 256)
            assert hashlib.sha256(original.astype(np.uint8).tobytes()).hexdigest() == source['prepared_sha256']
            if run_id == 'pair_001':
                with np.load(directory / 'secrets.npz') as secrets:
                    for case in CASES:
                        group = groups[(spec['name'], case.name)]
                        bank = secrets['multiple_candidates' if case.name in ('rewatermark', 'collusion') else 'single_candidates']
                        recovered = np.load(directory / f'{case.name}_extracted.npy')
                        actual = bank @ recovered / np.linalg.norm(recovered)
                        stored = np.array([float(r['similarity']) for r in group])
                        np.testing.assert_allclose(actual, stored, atol=1e-11, rtol=1e-12)
                        max_score_error = max(max_score_error, float(np.max(np.abs(actual - stored))))
                        with Image.open(directory / f'{case.name}.png') as raster:
                            attacked = np.asarray(raster, dtype=np.float64)
                        fidelity = 10 * np.log10(255**2 / np.mean((original - attacked)**2))
                        assert abs(fidelity - float(group[0]['psnr_db'])) < 1e-12
                        assert all(r['source_sha256'] == spec['sha256'] and
                                   r['prepared_sha256'] == source['prepared_sha256'] and
                                   int(r['image_seed']) == image_seed for r in group)
    log.append(f'PASS sources, ordered SeedSequence children, parameters, display metadata and rasters; max detector recomputation error {max_score_error:.3g}')
    print("\n".join(log))
    print("PASS verify_pair: all numeric checks passed")


if __name__ == '__main__':
    main()
