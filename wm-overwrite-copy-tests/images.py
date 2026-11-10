"""In-memory image preparation, quantization, and representative PNGs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

SIZE: Final = 512
SUFFIXES: Final = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        return rgb.crop((left, top, left + side, top + side)).resize(
            (SIZE, SIZE), Image.Resampling.LANCZOS
        )


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float64) / 255.0


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


@dataclass(frozen=True, slots=True)
class Example:
    original: Image.Image
    marked: Image.Image
    attacked: Image.Image


def save_example(folder: Path, label: str, example: Example) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    example.original.save(folder / f"{label}_original.png")
    example.marked.save(folder / f"{label}_marked.png")
    example.attacked.save(folder / f"{label}_attacked_or_forged.png")
    # Absolute RGB difference, x10; black means unchanged, clipped at white.
    difference = np.abs(to_array(example.attacked) - to_array(example.original))
    to_image(difference * 10).save(folder / f"{label}_difference_x10.png")
