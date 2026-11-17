import argparse
import filecmp
import json
from pathlib import Path
from download_file import ROOT, digest


def compare(run1: Path, run2: Path, reference: Path) -> bool:
    recorded = json.loads((reference / 'determinism.json').read_text())
    expected = {row['file'] for row in recorded['files']}
    comparisons = []
    for run in (run1, run2):
        actual = {path.name for path in run.iterdir() if path.is_file()}
        comparisons.append({'check': run.name + ' file set',
                            'identical': actual == expected | {'timing.json'}})
    for row in recorded['files']:
        name = row['file']
        if Path(name).name != name:
            raise SystemExit('FAIL: invalid reference filename')
        a, b = run1 / name, run2 / name
        present = a.is_file() and b.is_file()
        same = present and filecmp.cmp(str(a), str(b), shallow=False)
        reference_match = (present and digest(a) == row['run1_sha256']
                           and digest(b) == row['run2_sha256'])
        saved = reference / 'run1' / name
        if name.endswith(('.csv', '.json')):
            reference_match = (reference_match and saved.is_file()
                               and filecmp.cmp(str(a), str(saved), shallow=False)
                               and filecmp.cmp(str(b), str(saved), shallow=False))
        comparisons.append({'check': name, 'identical': bool(same and reference_match),
                            'runs_byte_identical': bool(same),
                            'reference_match': bool(reference_match)})
    for name in ('shapes_params.json', 'network_layers.csv', 'signatures.json'):
        measured, saved = ROOT / 'run-output/network' / name, reference / name
        comparisons.append({'check': 'network/' + name,
                            'identical': measured.is_file() and filecmp.cmp(
                                str(measured), str(saved), shallow=False)})
    for name, generated in [('weights_provenance.json', ROOT / '.cache/weights_provenance.json'),
                            ('archive_listing.json', ROOT / '.cache/weights/m3/archive_listing.json')]:
        comparisons.append({'check': name, 'identical': generated.is_file() and filecmp.cmp(
            str(generated), str(reference / name), shallow=False)})
    success = all(row['identical'] for row in comparisons)
    report = {'pass': success, 'comparisons': comparisons,
              'excluded': ['run1/timing.json', 'run2/timing.json', 'network/measurement_timing.json'],
              'timing': [json.loads((run / 'timing.json').read_text()) for run in (run1, run2)]}
    (ROOT / 'run-output/comparison.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for row in comparisons:
        print(('PASS: ' if row['identical'] else 'FAIL: ') + row['check'])
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare all fixed outputs and recorded references.')
    parser.add_argument('--run1', type=Path, default=ROOT / 'run-output/run1')
    parser.add_argument('--run2', type=Path, default=ROOT / 'run-output/run2')
    parser.add_argument('--reference', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    success = compare(args.run1, args.run2, args.reference)
    print('PASS: reproduction matches' if success else 'FAIL: reproduction differs')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
