# Run from the repository root: python tian-2003-difference-expansion/tian.py
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from map_codec import Bits, decode_map, encode_map, int_bits

Pixels = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Plan:
    threshold: int
    selected: NDArray[np.bool_]
    changeable: NDArray[np.bool_]
    saved: NDArray[np.bool_]
    map_bits: Bits
    capacity: int


@dataclass(frozen=True, slots=True)
class Encoded:
    image: Pixels
    plan: Plan
    padding: int


def transform(image: Pixels) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    pairs = image.astype(np.int32).reshape(-1, 2)
    return (pairs[:, 0] + pairs[:, 1]) // 2, pairs[:, 0] - pairs[:, 1]


def inverse(mean: NDArray[np.int32], difference: NDArray[np.int32]) -> Pixels:
    pairs = np.column_stack((mean + (difference + 1) // 2, mean - difference // 2))
    assert pairs.min() >= 0 and pairs.max() <= 255
    return pairs.astype(np.uint8).reshape(512, 512)


def make_plan(image: Pixels, threshold: int) -> Plan:
    mean, difference = transform(image)
    bound = np.minimum(2 * (255 - mean), 2 * mean + 1)
    expandable = np.maximum(abs(2*difference), abs(2*difference+1)) <= bound
    low = 2 * (difference // 2)
    changeable = np.maximum(abs(low), abs(low+1)) <= bound
    selected = expandable & ((abs(difference) <= threshold) | (difference == -1))
    saved = changeable & ~selected & (difference != 1) & (difference != -2)
    map_bits = encode_map(selected)
    capacity = int(changeable.sum() - saved.sum() - map_bits.size - 16 - 32 - 256)
    return Plan(threshold, selected, changeable, saved, map_bits, capacity)


def choose_plan(image: Pixels, payload_size: int) -> Plan:
    for threshold in range(256):
        plan = make_plan(image, threshold)
        if plan.capacity >= payload_size:
            return plan
    raise RuntimeError(payload_size)


def embed(image: Pixels, payload: Bits, plan: Plan) -> Encoded:
    mean, difference = transform(image)
    padding = plan.capacity - payload.size
    assert padding >= 0
    digest = np.unpackbits(np.frombuffer(sha256(image.tobytes()).digest(), dtype=np.uint8))
    stream = np.concatenate((np.zeros(16, dtype=np.uint8), plan.map_bits,
        (difference[plan.saved] & 1).astype(np.uint8),
        np.array(int_bits(payload.size, 32), dtype=np.uint8), digest, payload,
        np.zeros(padding, dtype=np.uint8)))
    assert stream.size == plan.changeable.sum()
    modified = difference.copy()
    modified[plan.selected] *= 2
    replace = plan.changeable & ~plan.selected
    modified[replace] = 2 * (modified[replace] // 2)
    modified[plan.changeable] += stream
    return Encoded(inverse(mean, modified), plan, padding)


def extract(image: Pixels) -> tuple[Pixels, Bits]:
    mean, difference = transform(image)
    bound = np.minimum(2 * (255 - mean), 2 * mean + 1)
    low = 2 * (difference // 2)
    changeable = np.maximum(abs(low), abs(low + 1)) <= bound
    stream = (difference[changeable] & 1).astype(np.uint8)
    assert not stream[:16].any()
    selected, map_size = decode_map(stream[16:], difference.size)
    assert not np.any(selected & ~changeable)
    restored = difference.copy()
    restored[selected] //= 2
    other = changeable & ~selected
    positive = other & ((difference == 0) | (difference == 1))
    negative = other & ((difference == -2) | (difference == -1))
    saved = other & ~positive & ~negative
    cursor = 16 + map_size
    restored[positive] = 1
    restored[negative] = -2
    restored[saved] = 2 * (restored[saved] // 2) + stream[cursor:cursor+saved.sum()]
    cursor += int(saved.sum())
    size = int.from_bytes(np.packbits(stream[cursor:cursor+32]).tobytes(), 'big')
    cursor += 32
    digest = np.packbits(stream[cursor:cursor+256]).tobytes()
    cursor += 256
    original = inverse(mean, restored)
    assert sha256(original.tobytes()).digest() == digest
    assert cursor + size <= stream.size
    assert not stream[cursor+size:].any()
    return original, stream[cursor:cursor+size].copy()
