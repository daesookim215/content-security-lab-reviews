# Run from the repository root: python cox-1997-spread-spectrum/summarize_pair.py
"""Build pair summaries from raw rows; no scientific execution or plotting."""
import csv
import json
import math
from pathlib import Path
from typing import Final, TypedDict

from records import CASES

ROOT: Final = Path(__file__).resolve().parent
NAMES: Final = ('sipi_airplane', 'sipi_boat')
LABELS: Final = ('Undistorted', 'Half-scale', 'JPEG 10', 'JPEG 5', 'Dither raw',
                 'Dither centered', 'Crop', 'JPEG 10 + crop', 'Re-watermarking', 'Collusion')


class CaseSummary(TypedDict):
    embedded_responses: list[float]
    embedded_indices: list[int]
    psnr_db: float
    unrelated_max: float
    unrelated_count: int
    unrelated_above_6: int
    paper_response: float | None


class ImageSummary(TypedDict):
    cases: dict[str, CaseSummary]
    unrelated_count: int
    unrelated_max: float
    unrelated_above_6: int


class Summary(TypedDict):
    images: dict[str, ImageSummary]
    threshold: float
    shape: list[int]
    difference_amplification: int
    unrelated_count: int
    unrelated_max: float
    unrelated_above_6: int
    response_count: int
    jpeg_quality: dict[str, int]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def compute_summary(rows: list[dict[str, str]]) -> Summary:
    """Aggregate responses without rounding; paper references come from records."""
    images: dict[str, ImageSummary] = {}
    for name in NAMES:
        cases: dict[str, CaseSummary] = {}
        for case in CASES:
            group = [r for r in rows if r['image'] == name and r['experiment'] == case.name]
            assert len(group) == 1000
            embedded = sorted((r for r in group if r['embedded'] == 'True'),
                              key=lambda r: int(r['mark_index']))
            unrelated = [float(r['similarity']) for r in group if r['embedded'] == 'False']
            cases[case.name] = {
                'embedded_responses': [float(r['similarity']) for r in embedded],
                'embedded_indices': [int(r['candidate_index']) for r in embedded],
                'psnr_db': float(group[0]['psnr_db']), 'unrelated_max': max(unrelated),
                'unrelated_count': len(unrelated),
                'unrelated_above_6': sum(v > 6 for v in unrelated),
                'paper_response': case.paper_response,
            }
        images[name] = {
            'cases': cases, 'unrelated_count': sum(c['unrelated_count'] for c in cases.values()),
            'unrelated_max': max(c['unrelated_max'] for c in cases.values()),
            'unrelated_above_6': sum(c['unrelated_above_6'] for c in cases.values()),
        }
    parameters = json.loads(rows[0]['parameters'])
    rms = max(255 * 10 ** (-images[n]['cases']['uniqueness']['psnr_db'] / 20) for n in NAMES)
    return {
        'images': images, 'threshold': parameters['threshold'], 'shape': parameters['shape'],
        'difference_amplification': max(1, 2 ** math.floor(math.log2(32 / rms))),
        'unrelated_count': sum(i['unrelated_count'] for i in images.values()),
        'unrelated_max': max(i['unrelated_max'] for i in images.values()),
        'unrelated_above_6': sum(i['unrelated_above_6'] for i in images.values()),
        'response_count': len(rows),
        'jpeg_quality': {case: int(json.loads(next(r['parameters'] for r in rows
                         if r['experiment'] == case))['operation'].split('quality=')[1].split(',')[0])
                         for case in ('jpeg10', 'jpeg5', 'jpeg10_crop')},
    }
