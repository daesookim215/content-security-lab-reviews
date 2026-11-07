# Run from the repository root: python tian-2003-difference-expansion/independent_decode.py
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray


def decode(encoded: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    pairs = encoded.astype(np.int64).reshape(-1, 2)
    means = (pairs[:, 0] + pairs[:, 1]) // 2
    differences = pairs[:, 0] - pairs[:, 1]
    base = differences - differences % 2
    left0 = means + (base + 1) // 2
    right0 = means - base // 2
    left1 = means + (base + 2) // 2
    right1 = means - (base + 1) // 2
    eligible = (left0 >= 0) & (left0 <= 255) & (right0 >= 0) & (right0 <= 255)
    eligible &= (left1 >= 0) & (left1 <= 255) & (right1 >= 0) & (right1 <= 255)
    bits = (differences[eligible] % 2).tolist()
    assert bits[:16] == [0]*16
    cursor = 16

    def read_integer(width: int) -> int:
        nonlocal cursor
        result = 0
        for _ in range(width):
            result = result * 2 + bits[cursor]
            cursor += 1
        return result

    color = read_integer(1)
    widths = (read_integer(5), read_integer(5))
    assert max(widths) <= 17
    location: list[int] = []
    while True:
        quotient = 0
        while read_integer(1) == 0:
            quotient += 1
        run = quotient * 2**widths[color] + read_integer(widths[color])
        if run == 0:
            break
        location += [color]*run
        assert len(location) <= len(differences)
        color = 1-color
    assert len(location) == len(differences)
    assert all(not flag or ok for flag, ok in zip(location, eligible, strict=True))
    restored = np.empty_like(pairs)
    for index, difference in enumerate(differences):
        if location[index]:
            original = int(difference) // 2
        elif eligible[index]:
            half = int(difference) // 2
            if half == 0:
                original = 1
            elif half == -1:
                original = -2
            else:
                original = 2*half + read_integer(1)
        else:
            original = int(difference)
        restored[index, 0] = means[index] + (original + 1) // 2
        restored[index, 1] = means[index] - original // 2
    length = read_integer(32)
    digest = bytes(read_integer(8) for _ in range(32))
    assert cursor + length <= len(bits)
    message = np.array(bits[cursor:cursor+length], dtype=np.uint8)
    assert all(bit == 0 for bit in bits[cursor+length:])
    assert restored.min() >= 0 and restored.max() <= 255
    output = restored.astype(np.uint8).reshape(encoded.shape)
    assert sha256(output.tobytes()).digest() == digest
    return output, message
