# Run from the repository root: python alattar-2004-integer-transform/independent_decode.py
from dataclasses import dataclass
from itertools import product
from typing import Literal, assert_never
import zlib

import numpy as np
from numpy.typing import NDArray

Pixels = NDArray[np.int64]
Arrangement = Literal["columns", "rows", "quads"]


@dataclass(frozen=True, slots=True)
class Decoded:
    pixels: Pixels
    payload: Pixels
    selected: int
    changeable: int
    overhead: int


def recover(pixels: Pixels, arrangement: Arrangement, threshold: int) -> Decoded:
    h, w = pixels.shape
    match arrangement:
        case "columns":
            samples = np.stack([pixels[k:h // 3 * 3:3, :] for k in range(3)], axis=-1)
        case "rows":
            samples = np.stack([pixels[:, k:w // 3 * 3:3] for k in range(3)], axis=-1)
        case "quads":
            samples = np.stack([pixels[0::2, 0::2], pixels[0::2, 1::2],
                                pixels[1::2, 0::2], pixels[1::2, 1::2]], axis=-1)
        case unreachable:
            assert_never(unreachable)
    n = samples.shape[-1]
    v = samples.reshape(-1, n)
    differences = v[:, 1:] - v[:, 0, None]
    average = v[:, 0] + np.floor_divide(differences.sum(axis=1), n)
    eligible = np.full(len(v), True)
    base = np.floor_divide(differences, 2) * 2
    for flags in product(range(2), repeat=n - 1):
        trial = base + np.array(flags, dtype=np.int64)
        anchor = average - np.floor_divide(trial.sum(axis=1), n)
        eligible &= (anchor >= 0) & (anchor <= 255)
        eligible &= np.all((anchor[:, None] + trial >= 0) & (anchor[:, None] + trial <= 255), axis=1)
    extracted = np.remainder(differences[eligible].reshape(-1), 2)
    stream = np.packbits(extracted).tobytes()
    assert stream[:4] == b"A04D", "Independent decoder: invalid header"
    compressed_length = int.from_bytes(stream[4:8], "big")
    payload_length = int.from_bytes(stream[8:12], "big")
    map_end = 12 + compressed_length
    mask_bytes = zlib.decompress(stream[12:map_end])
    mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8))[:len(v)] == 1
    assert len(mask) == len(v) and np.all(eligible[mask])
    replacement = eligible & ~mask
    saved_count = int(replacement.sum()) * (n - 1)
    recovered_differences = differences.copy()
    recovered_differences[mask] = np.floor_divide(differences[mask], 2)
    saved = extracted[map_end * 8:map_end * 8 + saved_count]
    recovered_differences[replacement] = base[replacement] + saved.reshape(-1, n - 1)
    first = average - np.floor_divide(recovered_differences.sum(axis=1), n)
    restored_samples = np.empty_like(v)
    restored_samples[:, 0] = first
    for k in range(1, n):
        restored_samples[:, k] = first + recovered_differences[:, k - 1]
    expected_mask = np.max(recovered_differences, axis=1) <= threshold
    for flags in product(range(2), repeat=n - 1):
        trial = 2 * recovered_differences + np.array(flags)
        anchor = average - np.floor_divide(trial.sum(axis=1), n)
        expected_mask &= (anchor >= 0) & (anchor <= 255)
        expected_mask &= np.all((anchor[:, None] + trial >= 0) & (anchor[:, None] + trial <= 255), axis=1)
    assert np.array_equal(mask, expected_mask), "Selection differs from the paper's conditions"
    output = pixels.copy()
    shaped = restored_samples.reshape(samples.shape)
    match arrangement:
        case "columns":
            for k in range(3):
                output[k:h // 3 * 3:3, :] = shaped[:, :, k]
        case "rows":
            for k in range(3):
                output[:, k:w // 3 * 3:3] = shaped[:, :, k]
        case "quads":
            for k, (r, c) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                output[r::2, c::2] = shaped[:, :, k]
        case unreachable:
            assert_never(unreachable)
    start = map_end * 8 + saved_count
    assert start + payload_length == len(extracted)
    assert payload_length == (n - 1) * int(mask.sum()) - map_end * 8
    return Decoded(output, extracted[start:], int(mask.sum()), int(eligible.sum()), map_end * 8)
