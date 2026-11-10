"""Run one configured scenario and write the experiment artifacts."""

import argparse
import csv
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import assert_never

import torch

from errors import ExperimentError
from images import SUFFIXES, load_image
from metrics import Observation, Recorder
from runtime import configure, memory_stats, versions
from scenarios import Experiment, ImageCase
from settings import Config, Scenario, load_config
from wm import TrustMarkWatermark, VideoSealWatermark


def selected_images(config: Config) -> list[Path]:
    paths = sorted(
        (path for path in config.image_dir.iterdir()
         if path.suffix.lower() in SUFFIXES and not path.is_symlink() and path.is_file()),
        key=lambda path: path.name,
    )[:config.n_images]
    if len(paths) != config.n_images:
        raise ExperimentError(f"Need {config.n_images} images; found {len(paths)} in {config.image_dir}")
    return paths


def run(config: Config) -> None:
    provenance = versions()
    print("VERSIONS " + json.dumps(provenance, sort_keys=True), flush=True)
    image_dir = config.image_dir.resolve()
    output_dir = config.output_dir.resolve()
    if not output_dir.is_relative_to(Path.cwd().resolve()) or output_dir.is_relative_to(image_dir):
        raise ExperimentError("output_dir must be inside this repository and outside image_dir")
    paths = selected_images(config)
    device = configure(config.seed)
    recorder = Recorder(config.videoseal)
    experiment = Experiment(config, recorder, TrustMarkWatermark(device, config.trustmark))
    match config.scenario:
        case Scenario.BASELINE | Scenario.CROSS:
            experiment.videoseal = VideoSealWatermark(device)
        case Scenario.OVERWRITE | Scenario.COPY:
            pass
        case unreachable:
            assert_never(unreachable)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(paths):
        target = paths[(index + 1) % len(paths)] if config.scenario is Scenario.COPY else source
        case = ImageCase(index, source, target, load_image(source))
        match config.scenario:
            case Scenario.BASELINE:
                experiment.baseline(case)
            case Scenario.OVERWRITE:
                experiment.overwrite(case)
            case Scenario.CROSS:
                experiment.cross_method(case)
            case Scenario.COPY:
                experiment.copy_attack(case)
            case unreachable:
                assert_never(unreachable)
        if (index + 1) % 10 == 0 or index + 1 == len(paths):
            print(f"Processed {index + 1}/{len(paths)} images", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize(0)
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(Observation)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in recorder.rows)
    results = {
        "config": config.model_dump(mode="json"), "versions": provenance,
        "n": len(paths), "filenames": [path.name for path in paths],
        "payload_lengths": {
            "trustmark": experiment.trustmark.payload_length,
            "videoseal": experiment.videoseal.payload_length if experiment.videoseal else None,
        },
        "memory": memory_stats(), "aggregates": recorder.aggregates(),
        "conventions": {
            "images": "Sorted filenames; center square crop; Lanczos resize; 512x512 RGB; all transforms in memory; 8-bit quantization after every embedding/forgery.",
            "seed": "NumPy SeedSequence([seed, sorted image index, stream]); TrustMark streams 11/12; VideoSeal streams 21/22; independent per-image messages.",
            "bit_accuracy": "Compare min(returned bits, reference payload length) aligned prefix bits regardless of detection. Empty output has null accuracy and is excluded from the mean; bit_accuracy_n, empty_decode_n and mean_compared_bits disclose coverage. Extra returned bits are ignored for bit accuracy.",
            "whole_message_match": "Exact length and all payload bits equal, irrespective of detection; empty output is a mismatch. Denominator is all images.",
            "detection": "TrustMark: native ECC flag, independent of reference match. VideoSeal: no official presence threshold; raw channel-0 score retained, api_detected null. Optional experiment-level known-message binomial p <= configured threshold supplies detection_rate; this is message verification, not blind presence detection. score_only leaves VideoSeal rates null.",
            "false_detection": "unmarked_control detection_rate on the same original images; VideoSeal verification tests each original against its independent assigned random message. These rates are not directly equivalent to TrustMark blind ECC detection.",
            "psnr": "10*log10(1/MSE) on RGB [0,1]. Identical images: null psnr_db and psnr_infinite=true; aggregate mean_psnr_db averages finite values only and psnr_infinite_n counts excluded infinities. Baseline none gives marked-vs-original PSNR; other conditions give attacked-vs-original PSNR.",
            "copy_attack": "Blind residual = marked - denoised(marked). Wiener5 or median3 per channel. Pair i with (i+1) mod n. sigma is target per-channel 5x5 local std; weight=sigma/max(sigma) jointly over pixels/channels, zero on constant targets; forged=clip(target+s*weight*residual,0,1), rounded to uint8. Strengths 1,2,3.",
            "png": "Lossless RGB PNG round trip; no display resampling, browser capture or color-management simulation.",
            "examples": "First three image indices. Difference is abs(final-original)*10, clipped. Copy examples: original=unmarked target; marked=marked donor; donor_original also saved; representative strength=2 for both estimators. Baseline examples use JPEG80; cross examples show both orders.",
        },
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("RESULT_JSON " + json.dumps(results, separators=(",", ":"), allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    arguments = parser.parse_args()
    run(load_config(arguments.config))


if __name__ == "__main__":
    main()
