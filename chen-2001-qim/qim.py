# Run from the repository root: python chen-2001-qim/qim.py
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dctn, idctn

FloatArray = NDArray[np.float64]
ByteArray = NDArray[np.uint8]
BLOCK: Final = 8
FREQUENCY: Final = (2, 3)


@dataclass(frozen=True, slots=True)
class Key:
    bits: ByteArray
    base_dither: FloatArray
    step: float


def blocks(image: ByteArray) -> FloatArray:
    return image.astype(np.float64).reshape(64, BLOCK, 64, BLOCK).transpose(0, 2, 1, 3)


def coefficients(image: ByteArray) -> FloatArray:
    return dctn(blocks(image), axes=(-2, -1), norm='ortho')


def dither_pair(key: Key) -> tuple[FloatArray, FloatArray]:
    first = key.base_dither * key.step
    second = first + np.where(first < 0, key.step / 2, -key.step / 2)
    return first, second


def embed(image: ByteArray, key: Key) -> tuple[ByteArray, FloatArray]:
    transformed = coefficients(image)
    host = transformed[:, :, FREQUENCY[0], FREQUENCY[1]].ravel()
    first, second = dither_pair(key)
    shift = np.where(key.bits == 0, first, second)
    marked = key.step * np.floor((host + shift) / key.step + 0.5) - shift
    transformed[:, :, FREQUENCY[0], FREQUENCY[1]] = marked.reshape(64, 64)
    pixels = idctn(transformed, axes=(-2, -1), norm='ortho')
    pixels = pixels.transpose(0, 2, 1, 3).reshape(512, 512)
    return np.clip(np.rint(pixels), 0, 255).astype(np.uint8), marked


def decode(image: ByteArray, key: Key) -> ByteArray:
    observed = coefficients(image)[:, :, FREQUENCY[0], FREQUENCY[1]].ravel()
    distances = []
    for shift in dither_pair(key):
        nearest = key.step * np.floor((observed + shift) / key.step + 0.5) - shift
        distances.append(np.abs(observed - nearest))
    return (distances[1] < distances[0]).astype(np.uint8)


def psnr(original: ByteArray, received: ByteArray) -> float:
    mse = float(np.mean((original.astype(np.float64) - received.astype(np.float64)) ** 2))
    return float(10 * np.log10(255**2 / mse)) if mse > 0 else float('inf')
