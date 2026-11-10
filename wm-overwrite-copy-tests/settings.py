"""Parse the small experiment configuration once at the CLI boundary."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Scenario(StrEnum):
    BASELINE = "baseline"
    OVERWRITE = "overwrite_same"
    CROSS = "cross_method"
    COPY = "copy_attack"


class DetectionMode(StrEnum):
    VERIFY = "message_verification"
    SCORE = "score_only"


class TrustMarkSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model_variant: Literal["Q"] = "Q"
    encoding: Literal["BCH_5"] = "BCH_5"
    strength: float = Field(default=1.0, ge=1.0, le=1.0)


class VideoSealSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: Literal["videoseal"] = "videoseal"
    detection: DetectionMode = DetectionMode.VERIFY
    verification_p_threshold: float = Field(default=1e-6, gt=0, lt=1)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    scenario: Scenario
    n_images: int = Field(default=200, ge=3)
    image_dir: Path
    seed: int = Field(default=0, ge=0)
    trustmark: TrustMarkSettings = TrustMarkSettings()
    videoseal: VideoSealSettings = VideoSealSettings()
    output_dir: Path = Path("outputs")


def load_config(path: Path) -> Config:
    with path.open(encoding="utf-8") as handle:
        return Config.model_validate(yaml.safe_load(handle))
