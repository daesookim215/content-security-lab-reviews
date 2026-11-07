# Run from the repository root: python cox-1997-spread-spectrum/imaging.py
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

from watermark import FloatArray

ROOT: Final = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT.parent))
INPUTS = ROOT / "outputs/inputs"


@dataclass(frozen=True, slots=True)
class ImageSpec:
    name: str
    dataset: str
    filename: str
    url: str
    sha256: str
    display: str


@dataclass(frozen=True, slots=True)
class ImageSource:
    name: str
    original: FloatArray
    source_sha256: str
    prepared_sha256: str
    source_size: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ChecksumError(ValueError):
    path: Path

    def __str__(self) -> str:
        return f'Image checksum mismatch: {self.path}'


def image_specs() -> tuple[ImageSpec, ...]:
    from fetch_images import SOURCES
    return tuple(ImageSpec("sipi_" + source.name, "USC-SIPI classics", source.filename,
                           "https://sipi.usc.edu/database/misc/" + source.filename,
                           source.digest, "USC-SIPI: source and copyright status unknown")
                 for source in SOURCES)


def save_image(pixels: FloatArray, path: Path) -> None:
    Image.fromarray(pixels.astype(np.uint8)).save(path)


def prepare(spec: ImageSpec, destination: Path) -> ImageSource:
    source_path = ROOT.parent / 'images' / spec.filename
    raw = source_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec.sha256:
        raise ChecksumError(source_path)
    with Image.open(source_path) as image:
        source_size = image.size
        width, height = source_size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        prepared = image.crop((left, top, left + side, top + side)).convert('L')
        prepared = prepared.resize((256, 256), Image.Resampling.LANCZOS)
        pixels = np.asarray(prepared, dtype=np.float64)
        prepared.save(destination / 'original.png')
    return ImageSource(spec.name, pixels, digest,
                       hashlib.sha256(pixels.astype(np.uint8).tobytes()).hexdigest(), source_size)


def scale(image: FloatArray) -> FloatArray:
    source = Image.fromarray(image.astype(np.uint8))
    reduced = source.resize((128, 128), Image.Resampling.LANCZOS)
    return np.asarray(reduced.resize((256, 256), Image.Resampling.LANCZOS), dtype=np.float64)


def jpeg(image: FloatArray, quality: int, destination: Path) -> FloatArray:
    Image.fromarray(image.astype(np.uint8)).save(
        destination, format='JPEG', quality=quality, smooth=0,
        subsampling=0, optimize=False, progressive=False,
    )
    with Image.open(destination) as decoded:
        destination.with_suffix('.tables.json').write_text(json.dumps(decoded.quantization, indent=2))
        return np.asarray(decoded.convert('L'), dtype=np.float64)


def dither(image: FloatArray) -> FloatArray:
    binary = Image.fromarray(image.astype(np.uint8)).convert('1', dither=Image.Dither.FLOYDSTEINBERG)
    return np.asarray(binary.convert('L'), dtype=np.float64)


def restore_crop(image: FloatArray, original: FloatArray) -> FloatArray:
    restored = original.copy()
    restored[64:192, 64:192] = image[64:192, 64:192]
    return restored
