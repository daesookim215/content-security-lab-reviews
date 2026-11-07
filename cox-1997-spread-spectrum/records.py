# Run from the repository root: python cox-1997-spread-spectrum/records.py
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from watermark import ALPHA, LENGTH

SEED: Final = 199712
PARAMETERS: Final = {
    'alpha': ALPHA, 'n': LENGTH, 'shape': [256, 256],
    'dct': 'II orthonormal full-image float64',
    'selection': 'largest absolute AC; stable descending; row-major ties',
    'quantization': 'clip [0,255], rint nearest-even, uint8',
    'rng': 'PCG64', 'threshold': 6.0,
    'reference': 'pristine original, fixed support and order for all passes',
}


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    section: str
    paper_response: float | None
    paper_description: str
    operation: str
    postprocess: str = 'raw'


CASES: Final = (
    Case('uniqueness', 'V-A / V-B p.1682', 32.0, 'one spike / 1000 candidates', 'one insertion'),
    Case('scaling', 'V-B pp.1682-1683', 13.4, '0.5 linear scale, restore dimensions', '256->128->256 LANCZOS'),
    Case('jpeg10', 'V-C p.1683', 22.8, 'quality 10%, smoothing 0%', 'Pillow JPEG quality=10, smooth=0'),
    Case('jpeg5', 'V-C p.1683', 13.9, 'quality 5%, smoothing 0%', 'Pillow JPEG quality=5, smooth=0'),
    Case('dither_raw', 'V-D p.1683', 5.2, 'dithering', 'Floyd-Steinberg, 1 bit'),
    Case('dither_centered', 'V-D p.1683', 10.5, 'dithering, remove extracted mean', 'Floyd-Steinberg, 1 bit', 'subtract mean'),
    Case('crop', 'V-E p.1683', 14.6, 'central quarter, pristine exterior', 'central [64:192,64:192], pristine exterior'),
    Case('jpeg10_crop', 'V-E p.1683', 10.6, 'quality 10%, crop central quarter', 'JPEG quality=10,smooth=0 then crop; pristine exterior'),
    Case('rewatermark', 'V-G pp.1683-1685', None, 'five spikes; exact scores not tabulated', '5 successive insertions total, fixed support/order'),
    Case('collusion', 'V-H pp.1684-1685', None, 'five spikes; exact scores not tabulated', 'mean of 5 independently watermarked uint8 copies, requantize'),
)


@dataclass(frozen=True, slots=True)
class Row:
    run_id: str
    seed: int
    image: str
    dataset: str
    image_seed: int
    watermark_seed: int
    candidate_seed: int
    experiment: str
    parameters: str
    candidate_index: int
    embedded: bool
    mark_index: int
    similarity: float
    above_6: bool
    psnr_db: float
    extracted_mean: float
    extracted_norm: float
    clipped_fraction: float
    source_sha256: str
    prepared_sha256: str


def parameter_json(case: Case) -> str:
    return json.dumps(PARAMETERS | {'operation': case.operation, 'postprocess': case.postprocess}, sort_keys=True)


def write_rows(path: Path, rows: list[Row]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Row.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
