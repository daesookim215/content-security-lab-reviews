"""Image transforms and blind watermark residual transfer."""

from enum import StrEnum
from io import BytesIO
from math import sqrt
from typing import assert_never

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import wiener

from images import SIZE, to_array, to_image


class Attack(StrEnum):
    NONE = "none"
    JPEG80 = "jpeg_q80"
    JPEG50 = "jpeg_q50"
    RESIZE = "resize_256_back"
    CROP = "center_crop_50pct_area"
    PNG = "screenshot_like_png"


class Estimator(StrEnum):
    WIENER = "wiener_5x5"
    MEDIAN = "median_3x3"


def round_trip(image: Image.Image, format_name: str, quality: int = 80) -> Image.Image:
    with BytesIO() as buffer:
        image.save(buffer, format=format_name, quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as recovered:
            return recovered.convert("RGB")


def apply_attack(image: Image.Image, attack: Attack) -> Image.Image:
    match attack:
        case Attack.NONE:
            return image.copy()
        case Attack.JPEG80:
            return round_trip(image, "JPEG", 80)
        case Attack.JPEG50:
            return round_trip(image, "JPEG", 50)
        case Attack.RESIZE:
            return image.resize((256, 256), Image.Resampling.LANCZOS).resize(
                (SIZE, SIZE), Image.Resampling.LANCZOS
            )
        case Attack.CROP:
            side = round(SIZE / sqrt(2))  # 362 pixels; approximately 50% area.
            offset = (SIZE - side) // 2
            return image.crop((offset, offset, offset + side, offset + side)).resize(
                (SIZE, SIZE), Image.Resampling.LANCZOS
            )
        case Attack.PNG:
            # Lossless RGB PNG only: a screenshot-like positive control, no UI/rendering.
            return round_trip(image, "PNG")
        case unreachable:
            assert_never(unreachable)


def estimate_residual(marked: Image.Image, estimator: Estimator) -> np.ndarray:
    source = to_array(marked)
    match estimator:
        case Estimator.WIENER:
            channels = []
            for channel in np.moveaxis(source, -1, 0):
                # scipy's adaptive Wiener estimates noise from local variance.
                # At zero-variance 0/0 locations, retain the input (zero residual).
                with np.errstate(divide="ignore", invalid="ignore"):
                    filtered = wiener(channel, (5, 5))
                channels.append(np.where(np.isfinite(filtered), filtered, channel))
            denoised = np.stack(channels, axis=-1)
        case Estimator.MEDIAN:
            denoised = median_filter(source, size=(3, 3, 1), mode="reflect")
        case unreachable:
            assert_never(unreachable)
    return source - denoised


def forge(target: Image.Image, residual: np.ndarray, strength: float) -> Image.Image:
    cover = to_array(target)
    # Per-channel 5x5 local std: sigma = sqrt(max(E[x^2] - E[x]^2, 0)).
    # Normalize jointly over pixels/channels: w = sigma / max(sigma).
    # For a constant target w=0. Forgery = clip(target + s*w*residual, 0, 1),
    # rounded to 8-bit RGB. Reflect padding; no channels are mixed.
    mean = uniform_filter(cover, size=(5, 5, 1), mode="reflect")
    variance = uniform_filter(cover**2, size=(5, 5, 1), mode="reflect") - mean**2
    sigma = np.sqrt(np.maximum(variance, 0))
    maximum = float(sigma.max())
    weights = sigma / maximum if maximum > 1e-12 else np.zeros_like(sigma)
    return to_image(cover + strength * weights * residual)
