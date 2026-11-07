# Run from the repository root: python ni-2006-histogram-shifting/method.py
"""Ni histogram shifting, with disjoint pairs and embedded minimum records."""
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

Pixels = NDArray[np.uint8]
SEED: Final = 200603


@dataclass(frozen=True, slots=True)
class Pair:
    peak: int
    zero: int


def select_pairs(image: Pixels, count: int) -> tuple[Pair, ...]:
    """Specialize II-B3 to two minima, retaining nonoverlapping intervals."""
    histogram = np.bincount(image.ravel(), minlength=256)
    global_peak = int(np.argmax(histogram))
    candidates = range(1, 255)
    rank = lambda value: (int(histogram[value]), abs(value-global_peak), value)
    if count == 1:
        zero = min((v for v in candidates if abs(v-global_peak) > 1), key=rank)
        return (Pair(global_peak, zero),)
    assert count == 2
    low = min(range(1, 128), key=rank)
    high = min(range(128, 255), key=rank)
    middle = sorted(range(low+1, high), key=lambda v: (-int(histogram[v]), -v))[:2]
    left_middle, right_middle = sorted(middle)
    left = max(range(0, low), key=lambda v: (int(histogram[v]), v))
    right = max(range(high+1, 256), key=lambda v: (int(histogram[v]), v))
    first = max((left, left_middle), key=lambda v: (int(histogram[v]), v))
    second = max((right_middle, right), key=lambda v: (int(histogram[v]), v))
    pairs = (Pair(first, low), Pair(second, high))
    assert max(min(p.peak, p.zero) for p in pairs) > min(max(p.peak, p.zero) for p in pairs)
    return pairs


def integer_bits(value: int, width: int) -> list[int]:
    assert 0 <= value < 2**width
    return [int(bit) for bit in f"{value:0{width}b}"]


def bookkeeping(image: Pixels, pairs: tuple[Pair, ...]) -> Pixels:
    records = []
    for pair in pairs:
        locations = np.flatnonzero(image.ravel() == pair.zero)
        if len(locations):
            assert abs(pair.peak-pair.zero) > 1
            records += integer_bits(pair.zero, 8) + integer_bits(len(locations), 19)
            for location in locations:
                records += integer_bits(int(location), 18)
    return np.array(records, dtype=np.uint8)


def embed(image: Pixels, pairs: tuple[Pair, ...], stream: Pixels) -> Pixels:
    """Fill every peak in a fixed scan order; minima with records stay put."""
    source = image.ravel().astype(np.int16)
    marked = source.copy()
    cursor = 0
    for pair in pairs:
        direction = 1 if pair.zero > pair.peak else -1
        inside = (source > min(pair.peak, pair.zero)) & (source < max(pair.peak, pair.zero))
        marked[inside] += direction
        locations = np.flatnonzero(source == pair.peak)
        end = cursor + len(locations)
        marked[locations] += direction * stream[cursor:end].astype(np.int16)
        cursor = end
    assert cursor == len(stream)
    assert int(marked.min()) >= 0 and int(marked.max()) <= 255
    return marked.astype(np.uint8).reshape(image.shape)


def extract(marked: Pixels, pairs: tuple[Pair, ...], occupied: tuple[bool, ...]) -> tuple[Pixels, Pixels]:
    """Vector decoder used by the run; verify.py has a separate scalar decoder."""
    source = marked.ravel().astype(np.int16)
    restored = source.copy()
    chunks = []
    for pair in pairs:
        direction = 1 if pair.zero > pair.peak else -1
        carriers = (source == pair.peak) | (source == pair.peak+direction)
        chunks.append((source[carriers] != pair.peak).astype(np.uint8))
        shifted = ((source-pair.peak)*direction > 0) & ((source-pair.zero)*direction <= 0)
        restored[shifted] -= direction
    stream = np.concatenate(chunks)
    cursor = 0
    for has_record in occupied:
        if has_record:
            zero = int("".join(map(str, stream[cursor:cursor+8])), 2)
            count = int("".join(map(str, stream[cursor+8:cursor+27])), 2)
            cursor += 27
            for _ in range(count):
                location = int("".join(map(str, stream[cursor:cursor+18])), 2)
                restored[location] = zero
                cursor += 18
    return restored.astype(np.uint8).reshape(marked.shape), stream[cursor:]
