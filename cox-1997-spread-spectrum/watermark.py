# Run from the repository root: python cox-1997-spread-spectrum/watermark.py
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dctn, idctn

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]
ALPHA: Final = 0.1
LENGTH: Final = 1000


@dataclass(frozen=True, slots=True)
class Reference:
    spectrum: FloatArray
    locations: IndexArray


@dataclass(frozen=True, slots=True)
class InvalidReferenceError(ValueError):
    available: int
    nonzero: int

    def __str__(self) -> str:
        return f'Need {LENGTH} nonzero AC coefficients; available={self.available}, nonzero={self.nonzero}'


def reference(image: FloatArray) -> Reference:
    spectrum = dctn(image, type=2, norm='ortho')
    magnitudes = np.abs(spectrum.ravel())
    locations = np.argsort(-magnitudes[1:], kind='stable')[:LENGTH] + 1
    nonzero = int(np.count_nonzero(magnitudes[locations]))
    if locations.size != LENGTH or nonzero != LENGTH:
        raise InvalidReferenceError(int(locations.size), nonzero)
    return Reference(spectrum, locations)


def embed_float(image: FloatArray, owner: Reference, mark: FloatArray) -> FloatArray:
    spectrum = dctn(image, type=2, norm='ortho')
    flat = spectrum.ravel()
    flat[owner.locations] *= 1.0 + ALPHA * mark
    return idctn(spectrum, type=2, norm='ortho')


def quantize(image: FloatArray) -> FloatArray:
    return np.rint(np.clip(image, 0, 255)).astype(np.uint8).astype(np.float64)


def extract(image: FloatArray, owner: Reference) -> FloatArray:
    observed = dctn(image, type=2, norm='ortho').ravel()[owner.locations]
    original = owner.spectrum.ravel()[owner.locations]
    return (observed / original - 1.0) / ALPHA


def responses(candidates: FloatArray, recovered: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(recovered))
    if norm == 0:
        return np.full(candidates.shape[0], np.nan)
    return candidates @ recovered / norm


def psnr(original: FloatArray, image: FloatArray) -> float:
    mse = float(np.mean(np.square(original - image)))
    return float(10 * np.log10(255**2 / mse)) if mse else float('inf')
