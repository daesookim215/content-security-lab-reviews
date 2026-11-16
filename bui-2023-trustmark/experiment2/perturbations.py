# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
# How to run: imported by run_exp2.py using the existing TrustMark environment.
"""Deterministic PIL/NumPy perturbations and recorded random geometry."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Final, TypedDict, assert_never

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True, slots=True)
class Setting:
    name: str
    kind: str
    value: float
    paper: str


SETTINGS: Final = (
    Setting("noise008", "noise", 0.08, "Gaussian noise high std=0.08"),
    Setting("contrast150", "contrast", 1.5, "Contrast high 0.5-1.5: upper endpoint"),
    Setting("contrast050", "contrast", 0.5, "Contrast high 0.5-1.5: lower endpoint"),
    Setting("saturation050", "saturation", 0.5, "Saturation high 0.5-1.5: lower endpoint"),
    Setting("saturation150", "saturation", 1.5, "Saturation high 0.5-1.5: upper endpoint"),
    Setting("hue005", "hue", 0.05, "Hue high 0.05 turn"),
    Setting("sharpness250", "sharpness", 2.5, "Sharpness high 2.5"),
    Setting("median3", "median", 3, "Median blur high k=3"),
    Setting("box7", "box", 7, "Box blur high k=7"),
    Setting("motion9_45", "motion", 9, "Motion high k=3-9, angle -90..90; k=9, angle=45, direction=0"),
    Setting("posterize3", "posterize", 3, "Posterize high bits=3"),
    Setting("rgb010", "rgb", 0.1, "RGB shift high limit=0.1; signs (+,-,+)"),
    Setting("grayscale", "grayscale", 1, "Grayscale p=0.5; forced on for single change"),
    Setting("hflip", "hflip", 1, "Fixed geometry flip p=0.5; forced on for single change"),
    Setting("brightness050", "brightness", 0.5, "Brightness high 0.5-1.5: lower endpoint"),
    Setting("brightness150", "brightness", 1.5, "Brightness high 0.5-1.5: upper endpoint (run1)"),
    Setting("jpeg40", "jpeg", 40, "JPEG high minimum q=40"),
)
BY_NAME: Final = {s.name: s for s in SETTINGS}
KINDS: Final = tuple(dict.fromkeys(s.kind for s in SETTINGS))


def jpeg(image: Image.Image, quality: int) -> bytes:
    """Return the same explicit JPEG settings used by run1."""
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=quality, subsampling=2,
               optimize=False, progressive=False)
    return stream.getvalue()


def quantize(array: np.ndarray) -> Image.Image:
    """Round clipped [0,1] values to RGB uint8, ties to even."""
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def perturb(image: Image.Image, setting: Setting, rng: np.random.Generator) -> Image.Image:
    """Apply one named, deterministic strength; RNG is used only for noise."""
    value = setting.value
    match setting.kind:
        case "noise":
            array = np.asarray(image, dtype=np.float64) / 255
            return quantize(array + rng.normal(0, value, array.shape))
        case "contrast":
            return ImageEnhance.Contrast(image).enhance(value)
        case "saturation":
            return ImageEnhance.Color(image).enhance(value)
        case "brightness":
            return ImageEnhance.Brightness(image).enhance(value)
        case "sharpness":
            return ImageEnhance.Sharpness(image).enhance(value)
        case "hue":
            hsv = np.array(image.convert("HSV"))
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.uint16) + round(value * 256)) % 256
            return Image.fromarray(hsv, mode="HSV").convert("RGB")
        case "median":
            return image.filter(ImageFilter.MedianFilter(int(value)))
        case "box":
            return image.filter(ImageFilter.BoxBlur((value - 1) / 2))
        case "motion":
            # Nine equally weighted samples along x=y, edge-replicated boundaries.
            radius = int(value) // 2
            array = np.asarray(image, dtype=np.float64) / 255
            padded = np.pad(array, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
            height, width = array.shape[:2]
            result = np.zeros_like(array)
            for offset in range(int(value)):
                result += padded[offset:offset + height, offset:offset + width] / value
            return quantize(result)
        case "posterize":
            return ImageOps.posterize(image, int(value))
        case "rgb":
            return quantize(np.asarray(image, dtype=np.float64) / 255 + np.array([value, -value, value]))
        case "grayscale":
            return ImageOps.grayscale(image).convert("RGB")
        case "hflip":
            return ImageOps.mirror(image)
        case "jpeg":
            with Image.open(io.BytesIO(jpeg(image, int(value)))) as encoded:
                return encoded.convert("RGB")
        case unreachable:
            assert_never(unreachable)


class Geometry(TypedDict):
    flip: bool
    scale: float
    aspect: float
    resized_wh: list[int]
    crop_xyxy: list[int]
    crop_fraction: float


def geometry(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, Geometry]:
    """Flip, resize with linear scale/log-uniform aspect, then random fixed-size crop."""
    flip = bool(rng.random() < 0.5)
    if flip:
        image = ImageOps.mirror(image)
    scale = float(rng.uniform(0.8, 1.0))
    aspect = math.exp(float(rng.uniform(math.log(3 / 4), math.log(4 / 3))))
    width = max(1, round(image.width * scale * math.sqrt(aspect)))
    height = max(1, round(image.height * scale / math.sqrt(aspect)))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    cw, ch = round(width * 244 / 256), round(height * 244 / 256)
    left, top = int(rng.integers(width - cw + 1)), int(rng.integers(height - ch + 1))
    box = [left, top, left + cw, top + ch]
    return image.crop(tuple(box)), Geometry(flip=flip, scale=scale, aspect=aspect,
        resized_wh=[width, height], crop_xyxy=box, crop_fraction=244 / 256)
