"""Per-observation scoring with separate API and message-verification detection."""

from dataclasses import asdict, dataclass
from math import log10
from statistics import fmean
from typing import assert_never

import numpy as np
from PIL import Image
from scipy.stats import binom

from images import to_array
from runtime import JsonValue
from settings import DetectionMode, VideoSealSettings
from wm import Decoded


@dataclass(frozen=True, slots=True)
class Observation:
    scenario: str
    image_index: int
    source_file: str
    target_file: str
    condition: str
    method: str
    reference: str
    reference_bits: str
    decoded_bits: str
    api_detected: bool | None
    detection_score: float | None
    detected: bool | None
    detection_basis: str
    verification_pvalue: float | None
    whole_message_match: bool
    bit_accuracy: float | None
    returned_bits: int
    compared_bits: int
    payload_length: int
    psnr_db: float | None
    psnr_infinite: bool


@dataclass(frozen=True, slots=True)
class Context:
    scenario: str
    image_index: int
    source_file: str
    target_file: str
    condition: str
    method: str
    reference: str


def psnr(original: Image.Image, modified: Image.Image) -> tuple[float | None, bool]:
    mse = float(np.mean((to_array(original) - to_array(modified)) ** 2))
    return (None, True) if mse == 0 else (-10 * log10(mse), False)


class Recorder:
    """Accumulate small scalar rows; image/tensor data never enters the ledger."""

    def __init__(self, settings: VideoSealSettings) -> None:
        self.settings = settings
        self.rows: list[Observation] = []

    def record(self, context: Context, reading: tuple[Decoded, str], quality: tuple[float | None, bool]) -> None:
        decoded, reference = reading
        compared = min(len(decoded.bits), len(reference))
        matches = sum(a == b for a, b in zip(decoded.bits[:compared], reference[:compared]))
        detected = decoded.detected
        basis = "api_ecc"
        pvalue = None
        if detected is None:
            basis = "raw_score_only"
            match self.settings.detection:
                case DetectionMode.VERIFY:
                    # Known-message binomial upper tail under independent fair bits.
                    # Not an official VideoSeal threshold or blind presence detector.
                    pvalue = float(binom.sf(matches - 1, len(reference), 0.5))
                    detected = compared == len(reference) and pvalue <= self.settings.verification_p_threshold
                    basis = "known_message_binomial"
                case DetectionMode.SCORE:
                    pass
                case unreachable:
                    assert_never(unreachable)
        self.rows.append(Observation(
            **asdict(context), reference_bits=reference, decoded_bits=decoded.bits,
            api_detected=decoded.detected, detection_score=decoded.score,
            detected=detected, detection_basis=basis, verification_pvalue=pvalue,
            whole_message_match=decoded.bits == reference,
            bit_accuracy=matches / compared if compared else None,
            returned_bits=len(decoded.bits), compared_bits=compared,
            payload_length=len(reference), psnr_db=quality[0], psnr_infinite=quality[1],
        ))

    def aggregates(self) -> list[dict[str, JsonValue]]:
        groups: dict[tuple[str, str, str], list[Observation]] = {}
        for row in self.rows:
            groups.setdefault((row.condition, row.method, row.reference), []).append(row)
        aggregates: list[dict[str, JsonValue]] = []
        for (condition, method, reference), rows in groups.items():
            flags = [int(r.detected) for r in rows if r.detected is not None]
            api_flags = [int(r.api_detected) for r in rows if r.api_detected is not None]
            accuracies = [r.bit_accuracy for r in rows if r.bit_accuracy is not None]
            scores = [r.detection_score for r in rows if r.detection_score is not None]
            qualities = [r.psnr_db for r in rows if r.psnr_db is not None]
            aggregates.append({
                "condition": condition, "method": method, "reference": reference,
                "n": len(rows), "detection_basis": rows[0].detection_basis,
                "detection_rate": fmean(flags) if flags else None,
                "api_detection_rate": fmean(api_flags) if api_flags else None,
                "false_detection_rate": (fmean(flags) if flags else None) if condition == "unmarked_control" else None,
                "mean_detection_score": fmean(scores) if scores else None,
                "whole_message_match_rate": fmean(r.whole_message_match for r in rows),
                "mean_bit_accuracy": fmean(accuracies) if accuracies else None,
                "bit_accuracy_n": len(accuracies),
                "empty_decode_n": sum(r.returned_bits == 0 for r in rows),
                "mean_compared_bits": fmean(r.compared_bits for r in rows),
                "mean_psnr_db": fmean(qualities) if qualities else None,
                "psnr_infinite_n": sum(r.psnr_infinite for r in rows),
            })
        return aggregates
