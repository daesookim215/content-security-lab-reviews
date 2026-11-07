# Run from the repository root: python alattar-2004-integer-transform/transform.py
from dataclasses import dataclass
from itertools import product
import struct
from typing import Final, Literal, assert_never
import zlib

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.int64]
Layout = Literal["columns", "rows", "quads"]
HEADER: Final = struct.Struct(">4sII")
MAGIC: Final = b"A04D"


@dataclass(frozen=True, slots=True)
class CapacityError(Exception):
    available: int


@dataclass(frozen=True, slots=True)
class PassResult:
    image: Array
    payload: Array
    selected: int
    changeable: int
    map_bits: int


def positions(shape: tuple[int, int], layout: Layout) -> Array:
    h, w = shape
    grid = np.arange(h * w).reshape(h, w)
    match layout:
        case "columns":
            return grid[:h // 3 * 3].reshape(h // 3, 3, w).transpose(0, 2, 1).reshape(-1, 3)
        case "rows":
            return grid[:, :w // 3 * 3].reshape(-1, 3)
        case "quads":
            return grid.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3).reshape(-1, 4)
        case unreachable:
            assert_never(unreachable)


def forward(vectors: Array) -> tuple[Array, Array]:
    return vectors.sum(axis=1) // vectors.shape[1], vectors[:, 1:] - vectors[:, :1]


def inverse(mean: Array, differences: Array) -> Array:
    first = mean - differences.sum(axis=1) // (differences.shape[1] + 1)
    return np.column_stack((first, first[:, None] + differences))


def safe(mean: Array, base: Array) -> NDArray[np.bool_]:
    valid = np.ones(len(mean), dtype=bool)
    for pattern in product((0, 1), repeat=base.shape[1]):
        pixels = inverse(mean, base + np.array(pattern))
        valid &= ((pixels >= 0) & (pixels <= 255)).all(axis=1)
    return valid


def embed(image: Array, layout: Layout, threshold: int, rng: np.random.Generator) -> PassResult:
    index = positions(image.shape, layout)
    mean, delta = forward(image.ravel()[index])
    changeable = safe(mean, 2 * (delta // 2))
    selected = safe(mean, 2 * delta) & (delta <= threshold).all(axis=1)
    assert np.all(~selected | changeable)
    compressed = zlib.compress(np.packbits(selected).tobytes(), level=9)
    map_bits = (HEADER.size + len(compressed)) * 8
    payload_size = int(selected.sum()) * delta.shape[1] - map_bits
    if payload_size < 1:
        raise CapacityError(payload_size)
    payload = rng.integers(0, 2, payload_size, dtype=np.int64)
    header = HEADER.pack(MAGIC, len(compressed), payload_size)
    prefix = np.unpackbits(np.frombuffer(header + compressed, dtype=np.uint8)).astype(np.int64)
    saved = (delta[changeable & ~selected] % 2).ravel()
    stream = np.concatenate((prefix, saved, payload))
    assert len(stream) == int(changeable.sum()) * delta.shape[1]
    modified = delta.copy()
    modified[selected] *= 2
    modified[changeable & ~selected] = 2 * (modified[changeable & ~selected] // 2)
    modified[changeable] += stream.reshape(-1, delta.shape[1])
    result = image.copy()
    result.ravel()[index] = inverse(mean, modified)
    assert result.min() >= 0 and result.max() <= 255
    return PassResult(result, payload, int(selected.sum()), int(changeable.sum()), map_bits)


def decode(image: Array, layout: Layout) -> tuple[Array, Array]:
    index = positions(image.shape, layout)
    mean, delta = forward(image.ravel()[index])
    changeable = safe(mean, 2 * (delta // 2))
    bits = (delta[changeable] % 2).ravel()
    magic, map_bytes, payload_size = HEADER.unpack(np.packbits(bits[:96]).tobytes())
    assert magic == MAGIC
    end = 96 + map_bytes * 8
    packed_map = zlib.decompress(np.packbits(bits[96:end]).tobytes())
    selected = np.unpackbits(np.frombuffer(packed_map, dtype=np.uint8))[:len(index)].astype(bool)
    other = changeable & ~selected
    saved_size = int(other.sum()) * delta.shape[1]
    original_delta = delta.copy()
    original_delta[selected] //= 2
    original_delta[other] = 2 * (delta[other] // 2) + bits[end:end + saved_size].reshape(-1, delta.shape[1])
    restored = image.copy()
    restored.ravel()[index] = inverse(mean, original_delta)
    payload = bits[end + saved_size:end + saved_size + payload_size]
    assert len(payload) == payload_size and end + saved_size + payload_size == len(bits)
    return restored, payload
