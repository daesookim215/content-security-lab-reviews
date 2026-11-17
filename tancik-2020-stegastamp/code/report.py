import json
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / 'results'


def tables() -> str:
    quality = json.loads((RESULTS / 'run1/quality.json').read_text())
    singles = json.loads((RESULTS / 'run1/single_attacks.json').read_text())
    combined = json.loads((RESULTS / 'run1/combined_summary.json').read_text())
    network = json.loads((RESULTS / 'shapes_params.json').read_text())
    times = json.loads((RESULTS / 'determinism.json').read_text())['timing']
    measurement = json.loads((RESULTS / 'measurement_timing.json').read_text())['seconds']
    lines = ['| Image | PSNR (dB) | SSIM |', '|---|---:|---:|']
    lines += ['| %s | %.6f | %.6f |' % (r['image'], r['psnr_db'], r['ssim']) for r in quality]
    lines += ['', '| Change | Airplane bits/100; exact recovery | Boat bits/100; exact recovery |',
              '|---|---|---|']
    for row in (r for r in singles if r['image'] == 'airplane'):
        boat = next(r for r in singles if r['image'] == 'boat' and r['attack'] == row['attack'])
        lines.append('| %s | %d; %s | %d; %s |' % (row['attack'], row['raw_bits_correct'],
                     'yes' if row['exact_secret'] else 'no', boat['raw_bits_correct'],
                     'yes' if boat['exact_secret'] else 'no'))
    lines += ['', '| Image | Draws | Exact recovery | Mean bits/100 | Minimum bits/100 |',
              '|---|---:|---:|---:|---:|']
    lines += ['| %s | %d | %d/%d | %.2f | %d |' % (r['image'], r['draws'],
              r['exact_recovery_count'], r['draws'], r['mean_bits'], r['min_bits']) for r in combined]
    lines += ['', '| Module | Parameters |', '|---|---:|']
    for label, scope in [('Secret path', 'gen_secret'), ('Encoder U-Net / residual', 'gen_encoder'),
                         ('Decoder STN', 'gen_stn'), ('Decoder bit head', 'gen_decoder')]:
        lines.append('| %s | %s |' % (label, format(network['params_by_module'][scope], ',')))
    lines += ['| Total | %s |' % format(network['params_total'], ','), '',
              '| Runtime | Run1 seconds | Run2 seconds |', '|---|---:|---:|']
    for label, key in [('Model + attack graph loading', 'model_and_chain_load_seconds'),
                       ('Encoding, attacks, decoding', 'inference_and_attacks_seconds'),
                       ('Script total including writes', 'total_seconds')]:
        lines.append('| %s | %.6f | %.6f |' % (label, times[0][key], times[1][key]))
    lines += ['', 'Network measurement: %.6f seconds.' % measurement]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    print(tables(), end='')
