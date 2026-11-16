# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.4.6", "pillow==13.0.0"]
# ///
"""Pixel transformations for run_test.py; imported by that runner."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SEED: Final = 20180908
MESSAGE: Final = "101100101100101100101100101100"


def tensor(pixels: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1


def serialize(values: torch.Tensor) -> np.ndarray:
    return np.rint((values.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8)


def blur(pixels: np.ndarray, sigma: float) -> np.ndarray:
    radius = math.ceil(3 * sigma)
    axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-axis.square() / (2 * sigma * sigma))
    kernel /= kernel.sum()
    values = torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0).float()
    horizontal = kernel.reshape(1, 1, 1, -1).repeat(3, 1, 1, 1)
    vertical = kernel.reshape(1, 1, -1, 1).repeat(3, 1, 1, 1)
    values = F.conv2d(F.pad(values, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=3)
    values = F.conv2d(F.pad(values, (0, 0, radius, radius), mode="reflect"), vertical, groups=3)
    return np.rint(values.squeeze(0).permute(1, 2, 0).numpy().clip(0, 255)).astype(np.uint8)


def psnr(pair: tuple[np.ndarray, np.ndarray]) -> float:
    reference, received = pair
    mse = float(np.mean((reference.astype(np.float64) - received.astype(np.float64)) ** 2))
    return 10 * math.log10(255 ** 2 / mse) if mse else math.inf


def save_png(path: Path, pixels: np.ndarray) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))
