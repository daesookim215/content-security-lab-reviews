"""Adapters for the pinned upstream image watermark APIs."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import videoseal
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from trustmark import TrustMark

from errors import ExperimentError
from images import to_image
from settings import TrustMarkSettings


@dataclass(frozen=True, slots=True)
class Decoded:
    detected: bool | None
    bits: str
    score: float | None = None


class Watermark(Protocol):
    name: str
    payload_length: int

    def embed(self, image: Image.Image, bits: str) -> Image.Image: ...

    def decode(self, image: Image.Image) -> Decoded: ...


def payload(length: int, seed_key: tuple[int, int, int]) -> str:
    rng = np.random.default_rng(np.random.SeedSequence(seed_key))
    return "".join(str(bit) for bit in rng.integers(0, 2, size=length))


class TrustMarkWatermark:
    name = "trustmark"

    def __init__(self, device: str, settings: TrustMarkSettings) -> None:
        self.model = TrustMark(
            model_type=settings.model_variant,
            encoding_type=TrustMark.Encoding.BCH_5,
            device=device,
            verbose=False,
            loadRemover=False,
        )
        if self.model.encoder is None or self.model.decoder is None:
            raise ExperimentError("TrustMark model download/load failed")
        self.payload_length = int(self.model.schemaCapacity())
        if self.payload_length != 61:
            raise ExperimentError(f"Expected BCH_5 capacity 61, got {self.payload_length}")

    @torch.inference_mode()
    def embed(self, image: Image.Image, bits: str) -> Image.Image:
        # Omit WM_STRENGTH to retain upstream's default 1.0.
        return self.model.encode(image, bits, MODE="binary").convert("RGB")

    @torch.inference_mode()
    def decode(self, image: Image.Image) -> Decoded:
        bits, detected, _schema = self.model.decode(image, MODE="binary")
        return Decoded(bool(detected), bits or "")


class VideoSealWatermark:
    name = "videoseal"

    def __init__(self, device: str) -> None:
        self.device = device
        self.model = videoseal.load("videoseal").eval().to(device)
        self.model.chunk_size = 1
        self.payload_length = int(self.model.get_random_msg(1).shape[-1])
        if self.payload_length != 256:
            raise ExperimentError(f"Expected default VideoSeal 256 bits, got {self.payload_length}")

    def tensor(self, image: Image.Image) -> torch.Tensor:
        return pil_to_tensor(image).unsqueeze(0).to(self.device, torch.float32) / 255.0

    @torch.inference_mode()
    def embed(self, image: Image.Image, bits: str) -> Image.Image:
        message = torch.tensor([[int(bit) for bit in bits]], device=self.device).float()
        output = self.model.embed(self.tensor(image), msgs=message, is_video=False)
        array = output["imgs_w"][0].detach().cpu().permute(1, 2, 0).numpy()
        return to_image(array)

    @torch.inference_mode()
    def decode(self, image: Image.Image) -> Decoded:
        predictions = self.model.detect(self.tensor(image), is_video=False)["preds"][0]
        if predictions.shape != (self.payload_length + 1,):
            raise ExperimentError(f"Unexpected default model predictions: {predictions.shape}")
        # Official README: bits are preds[1:] > 0 (zero-logit threshold).
        # Official docs/torchscript.md: channel 0 is unused. There is NO official
        # presence threshold. Return its raw score and detected=None; metrics.py
        # optionally supplies explicitly labeled known-message verification.
        bits = "".join(str(int(bit)) for bit in (predictions[1:] > 0).cpu().tolist())
        return Decoded(None, bits, float(predictions[0].item()))
