# Run from the repository root: python vanschyndel-1994-digital-watermark/method.py

import heapq
from typing import Final

import numpy as np
from numpy.typing import NDArray

SEED: Final = 199409
WIDTH: Final = 512
QUALITIES: Final = (100, 95, 85, 75, 50)
Bytes = NDArray[np.uint8]
Integers = NDArray[np.int64]
Floats = NDArray[np.float64]


def extended_sequence() -> Bytes:
    """Extend a maximal recurrence by inserting a zero in its longest zero run."""
    bits = [1] * 9
    for index in range(511):
        bits.append(bits[index] ^ bits[index + 4])
    cycle = np.array(bits[:511], dtype=np.uint8)
    insertion = next(i + 8 for i in range(511)
                     if all(cycle[(i + j) % 511] == 0 for j in range(8)))
    return np.insert(cycle, insertion % 511, 0).astype(np.uint8)


def payload() -> Integers:
    return np.random.default_rng(SEED).integers(0, WIDTH, size=WIDTH, dtype=np.int64)


def watermark(sequence: Bytes, phases: Integers) -> Bytes:
    return sequence[(np.arange(WIDTH)[None, :] - phases[:, None]) % WIDTH]


def adaptive_even_lookup(image: Bytes) -> Bytes:
    """Merge adjacent histogram groups by weighted squared error; free the LSB."""
    histogram = np.bincount(image.ravel(), minlength=256)
    occupied = np.flatnonzero(histogram)
    groups = len(occupied)
    weights = histogram[occupied].astype(np.float64)
    sums = weights * occupied
    left = list(range(-1, groups - 1))
    right = list(range(1, groups + 1))
    right[-1] = -1
    active = [True] * groups
    version = [0] * groups
    members = [[int(v)] for v in occupied]
    queue: list[tuple[float, int, int, int, int]] = []

    def enqueue(a: int, b: int) -> None:
        if a < 0 or b < 0:
            return
        cost = weights[a] * weights[b] / (weights[a] + weights[b])
        cost *= (sums[a] / weights[a] - sums[b] / weights[b]) ** 2
        heapq.heappush(queue, (float(cost), a, b, version[a], version[b]))

    for a in range(groups - 1):
        enqueue(a, a + 1)
    while groups > 128:
        _, a, b, va, vb = heapq.heappop(queue)
        if not (active[a] and active[b] and version[a] == va and version[b] == vb
                and right[a] == b):
            continue
        weights[a] += weights[b]
        sums[a] += sums[b]
        members[a].extend(members[b])
        active[b] = False
        version[a] += 1
        right[a] = right[b]
        if right[b] >= 0:
            left[right[b]] = a
        groups -= 1
        enqueue(left[a], a)
        enqueue(a, right[a])
    lookup = np.arange(256, dtype=np.uint8) & 254
    for a in range(len(occupied)):
        if active[a]:
            representative = min(254, max(0, int(np.rint(sums[a] / weights[a] / 2)) * 2))
            lookup[members[a]] = representative
    return lookup


def correlate_rows(image: Bytes, sequence: Bytes, *, addition: bool) -> Floats:
    reference = 2 * sequence.astype(np.float64) - 1
    observed = image.astype(np.float64) if addition else (image & 1).astype(np.float64)
    correlation = np.fft.ifft(np.fft.fft(observed, axis=1)
                              * np.conj(np.fft.fft(reference))[None, :], axis=1).real
    if addition:
        correlation = (4 * correlation - np.roll(correlation, 1, axis=1)
                       - np.roll(correlation, -1, axis=1)
                       - np.roll(correlation, 2, axis=1)
                       - np.roll(correlation, -2, axis=1))
    return np.rint(correlation).astype(np.float64)


def decode(image: Bytes, sequence: Bytes, *, addition: bool) -> Integers:
    return np.argmax(correlate_rows(image, sequence, addition=addition), axis=1)


def bits_of(phases: Integers) -> Bytes:
    return ((phases[:, None] >> np.arange(8, -1, -1)) & 1).astype(np.uint8)


def psnr(original: Bytes, received: Bytes) -> float:
    error = np.mean((original.astype(np.float64) - received.astype(np.float64)) ** 2)
    return float(10 * np.log10(255**2 / error))
