# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "pillow"]
# ///
# How to run: imported by run_exp2.py in the existing TrustMark environment.
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from trustmark import TrustMark

EPSILON: Final = 8 / 255
STEP: Final = 1 / 255
THRESHOLD: Final = 0.5 + EPSILON / 2


class AttackInfo(TypedDict):
    iterations: int
    float_packet_accuracy: float
    float_payload_accuracy: float
    criterion_met: bool
    epsilon: float
    step: float
    threshold: float
    float_linf: float
    uint8_linf: float
    resize_max_abs_difference_01: float
    float_initial_packet_accuracy: float
    trace: list[dict[str, float | int]]


@dataclass(frozen=True, slots=True)
class AttackResult:
    image: Image.Image
    info: AttackInfo
    seconds: float


def ifgsm(model: TrustMark, marked: Image.Image, packet: str) -> AttackResult:
    start = time.perf_counter()
    base = to_tensor(marked).unsqueeze(0)
    assert model.get_the_image_for_processing(marked).size == marked.size
    assert model.concentrate_wm_region == 1.0
    target = torch.tensor([[int(bit) for bit in packet]], dtype=torch.float32)
    flipped = 1 - target
    decoder_size = (model.model_resolution_dec,) * 2

    def logits_for(image: torch.Tensor) -> torch.Tensor:
        # PIL bilinear downsamples with antialiasing. Float interpolation omits PIL's uint8 rounding.
        resized = F.interpolate(image, size=decoder_size, mode="bilinear",
                                align_corners=False, antialias=True)
        return model.decoder.decoder(resized * 2 - 1)

    with torch.no_grad():
        resized = F.interpolate(base, size=decoder_size, mode="bilinear",
                                align_corners=False, antialias=True)
        pil_resized = to_tensor(marked.resize(decoder_size, Image.Resampling.BILINEAR)).unsqueeze(0)
        mismatch = float((resized - pil_resized).abs().max())
    image = base.clone()
    trace = []
    initial = 0.0
    for iteration in range(501):
        image = image.detach().requires_grad_(True)
        logits = logits_for(image)
        bits = logits > 0
        accuracy = float((bits == target.bool()).float().mean())
        payload_accuracy = float((bits[:, :61] == target[:, :61].bool()).float().mean())
        if iteration == 0:
            initial = accuracy
        trace.append({"iteration": iteration, "packet_accuracy": accuracy,
                      "payload_accuracy": payload_accuracy})
        if accuracy < THRESHOLD or iteration == 500:
            break
        loss = F.binary_cross_entropy_with_logits(logits, flipped)
        gradient, = torch.autograd.grad(loss, image)
        assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
        with torch.no_grad():
            image = image - STEP * gradient.sign()
            image = torch.maximum(torch.minimum(image, base + EPSILON), base - EPSILON).clamp(0, 1)
    with torch.no_grad():
        array = image[0].permute(1, 2, 0).numpy()
        rounded = np.rint(array * 255).clip(0, 255).astype(np.uint8)
        result = Image.fromarray(rounded)
        uint8_linf = float(np.abs(rounded.astype(np.int16) - np.asarray(marked).astype(np.int16)).max()) / 255
        float_linf = float((image - base).abs().max())
    assert uint8_linf <= EPSILON + 1e-7 and float_linf <= EPSILON + 1e-7
    return AttackResult(result, AttackInfo(iterations=iteration,
        float_packet_accuracy=accuracy, float_payload_accuracy=payload_accuracy,
        criterion_met=accuracy < THRESHOLD, epsilon=EPSILON, step=STEP,
        threshold=THRESHOLD, float_linf=float_linf, uint8_linf=uint8_linf,
        resize_max_abs_difference_01=mismatch, float_initial_packet_accuracy=initial,
        trace=trace), time.perf_counter() - start)
