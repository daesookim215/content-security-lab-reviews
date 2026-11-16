# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "pillow"]
# ///
# How to run: imported by run_exp2.py in the existing TrustMark environment.
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import Image
from trustmark import TrustMark

from baseline_run1 import PAYLOAD, decode_record


class Identity(TypedDict):
    image: str
    resolution: int
    experiment: str
    change: str


class Extra(TypedDict, total=False):
    strength: float
    psnr_input_db: float | None
    psnr_marked_db: float | None
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
    seed: int
    geometry: dict[str, bool | float | list[int]]
    perturbations: list[str]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def psnr(first: Image.Image, second: Image.Image) -> float | None:
    error = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    mse = float(np.mean(error ** 2))
    return 10 * math.log10(255 ** 2 / mse) if mse > 0 else None


@dataclass(frozen=True, slots=True)
class Recorder:
    model: TrustMark
    output: Path
    packet: str
    rows: list = field(default_factory=list)
    timings: list = field(default_factory=list)

    def record(self, image: Image.Image, identity: Identity, extra: Extra | None = None) -> None:
        start = time.perf_counter()
        stem = "-".join(str(identity[k]) for k in ("experiment", "image", "resolution", "change"))
        path = self.output / "images" / f"{stem}.png"
        if path.exists():
            with Image.open(path) as saved:
                assert np.array_equal(np.asarray(saved), np.asarray(image))
        else:
            image.save(path)
        with Image.open(path) as saved:
            metrics = decode_record(self.model, saved.convert("RGB"))
        packet_correct = sum(a == b for a, b in zip(metrics["raw_packet"], self.packet, strict=True))
        row = {**identity, **(extra or {}), **metrics, "packet_bits_correct": packet_correct,
               "packet_accuracy": packet_correct / 100, "dimensions": list(image.size),
               "file": str(path.relative_to(self.output)), "image_sha256": sha256(path)}
        self.rows.append(row)
        self.timings.append({**identity, "decode_and_save_seconds": time.perf_counter() - start})
        print(stem, metrics["payload_bits_correct"], metrics["detected"], metrics["exact_payload"], flush=True)

    def thumbnail(self, image: Image.Image, name: str) -> None:
        thumb = image.copy()
        thumb.thumbnail((600, 600), Image.Resampling.LANCZOS)
        thumb.save(self.output / "thumbnails" / f"{name}.png")

    def export(self) -> None:
        fields = sorted({key for row in self.rows for key in row})
        with (self.output / "measurements.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                encoded = {key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict))
                           else value for key, value in row.items()}
                writer.writerow(encoded)
        for experiment in ("A", "B", "C", "D", "E", "baseline", "control"):
            selected = [row for row in self.rows if row["experiment"] == experiment]
            (self.output / f"{experiment}.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")

    def summary(self) -> list:
        summary = []
        for image in ("airplane", "boat"):
            for size in (512, 2048):
                for experiment in ("A", "B", "C", "D", "E"):
                    rows = [r for r in self.rows if (r["image"], r["resolution"], r["experiment"])
                            == (image, size, experiment)]
                    if rows:
                        summary.append({"image": image, "resolution": size, "experiment": experiment,
                            "count": len(rows), "mean_raw_accuracy": float(np.mean([r["bit_accuracy"] for r in rows])),
                            "min_raw_accuracy": min(r["bit_accuracy"] for r in rows),
                            "exact_recovery_count": sum(r["exact_payload"] for r in rows)})
        return summary
