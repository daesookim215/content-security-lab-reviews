# Run from the repository root: python cox-1997-spread-spectrum/run_pair.py
"""Run the unchanged scientific implementation on the requested ordered pair."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from imaging import image_specs
from numerical_checks import numerical_checks
from records import PARAMETERS, SEED, write_rows
from pair_experiment import execute_image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from experiment_support import verify_inputs
INPUTS = ROOT / "outputs/inputs"


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ('pair_001', 'pair_002'):
        raise SystemExit("Choose a working run: pair_001, pair_002")
    verify_inputs()
    checks = numerical_checks()
    inventory = {spec.name: spec for spec in image_specs()}
    specs = tuple(inventory[name]
                  for name in ('sipi_airplane', 'sipi_boat'))
    run = ROOT / "outputs" / sys.argv[1]
    run.mkdir(parents=True, exist_ok=False)
    seeds = [int(child.generate_state(1)[0])
             for child in np.random.SeedSequence(SEED).spawn(2)]
    manifest = {
        'run_id': run.name, 'seed': SEED, 'parameters': PARAMETERS,
        'images': [asdict(spec) for spec in specs], 'image_seeds': seeds,
        'numerical_checks': checks,
    }
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    for spec, seed in zip(specs, seeds, strict=True):
        rows.extend(execute_image(spec, run, seed))
    write_rows(run / 'responses.csv', rows)
    from summarize_pair import compute_summary, read_rows
    summary = compute_summary(read_rows(run / 'responses.csv'))
    (ROOT / 'outputs/pair_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Completed {run.name}: {len(rows)} responses; source checksums verified')


if __name__ == '__main__':
    main()
