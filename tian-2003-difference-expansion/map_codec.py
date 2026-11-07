# Run from the repository root: python tian-2003-difference-expansion/map_codec.py
from typing import Final

import numpy as np
from numpy.typing import NDArray

Bits = NDArray[np.uint8]
MAX_K: Final = 17


def int_bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in reversed(range(width))]


def encode_map(selected: NDArray[np.bool_]) -> Bits:
    edges = np.flatnonzero(np.diff(selected.astype(np.int8))) + 1
    lengths = np.diff(np.r_[0, edges, selected.size]).tolist()
    initial = int(selected[0])
    groups = [lengths[bit ^ initial::2] for bit in (0, 1)]
    parameters = [min(range(MAX_K + 1), key=lambda k: sum(
        (length >> k) + 1 + k for length in group)) for group in groups]
    stream = [initial] + int_bits(parameters[0], 5) + int_bits(parameters[1], 5)
    for index, length in enumerate(lengths + [0]):
        k = parameters[initial ^ (index % 2)]
        stream.extend([0] * (length >> k))
        stream.append(1)
        stream.extend(int_bits(length % (1 << k), k))
    return np.array(stream, dtype=np.uint8)


def decode_map(bits: Bits, count: int) -> tuple[NDArray[np.bool_], int]:
    cursor = 11
    bit = int(bits[0])
    parameters = [sum(int(b) << (4-i) for i, b in enumerate(bits[start:start+5]))
                  for start in (1, 6)]
    assert max(parameters) <= MAX_K
    output: list[int] = []
    while True:
        k = parameters[bit]
        quotient = 0
        while bits[cursor] == 0:
            quotient += 1
            cursor += 1
        cursor += 1
        remainder = 0
        for _ in range(k):
            remainder = 2 * remainder + int(bits[cursor])
            cursor += 1
        length = (quotient << k) + remainder
        if length == 0:
            break
        output.extend([bit] * length)
        assert len(output) <= count
        bit ^= 1
    assert len(output) == count
    return np.array(output, dtype=bool), cursor
