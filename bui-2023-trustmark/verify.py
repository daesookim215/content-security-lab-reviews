import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent


def main(run_root: Path | None = None) -> None:
    result_root = ROOT / "results" if run_root is None else run_root
    folders = [result_root / run for run in ("run1", "run2")]
    first, second = [json.loads((folder / "results.json").read_text()) for folder in folders]
    assert first == second, "Repeated raw JSON differs"
    for filename in ("results.json", "measurements.csv", "quality.csv", "unmarked.csv"):
        assert (folders[0] / filename).read_bytes() == (folders[1] / filename).read_bytes(), filename
    settings = first["settings"]
    payload = settings["payload"]
    assert len(payload) == settings["payload_bits"]
    assert len(settings["expected_packet"]) == sum(settings[key] for key in ("payload_bits", "parity_bits", "schema_bits"))
    assert settings["expected_packet"].startswith(payload)
    for entry in first["inputs"]:
        source = ROOT / "images" / entry["filename"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]
        with Image.open(source) as source_image:
            rgb = source_image.convert("RGB")
        with Image.open(folders[0] / f"images/{entry['image']}-512-input.png") as saved:
            assert np.array_equal(np.asarray(rgb), np.asarray(saved))
            if entry["image"] == "boat":
                array = np.asarray(saved)
                assert np.array_equal(array[:, :, 0], array[:, :, 1])
                assert np.array_equal(array[:, :, 1], array[:, :, 2])
        with Image.open(folders[0] / f"images/{entry['image']}-2048-input.png") as enlarged:
            assert np.array_equal(np.asarray(rgb.resize((2048, 2048), Image.Resampling.LANCZOS)), np.asarray(enlarged))
    for folder in folders:
        for filename, digest in first["image_sha256"].items():
            assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest, filename
    assert len(first["rows"]) == len(first["inputs"]) * len(settings["resolutions"]) * 8
    for row in first["rows"] + first["unmarked"]:
        raw = row["raw_packet"]
        assert len(raw) == settings["packet_bits"] and set(raw) <= {"0", "1"}
        assert row["raw_payload"] == raw[:len(payload)]
        correct = sum(a == b for a, b in zip(raw, payload))
        assert row["payload_bits_correct"] == correct
        assert row["bit_accuracy"] == correct / len(payload)
        assert row["exact_payload"] == (row["decoded_payload"] == payload)
        if row["detected"]:
            assert row["candidate_valid"]
            assert row["decoded_payload"] == row["candidate_payload"]
        else:
            assert row["decoded_payload"] == "" and row["schema"] == -1
    for row in first["quality"]:
        with Image.open(folders[0] / row["input_file"]) as cover, Image.open(folders[0] / row["marked_file"]) as marked:
            histogram = ImageChops.difference(cover, marked).histogram()
            squared_error = sum((index % 256) ** 2 * count for index, count in enumerate(histogram))
            mse = squared_error / (cover.width * cover.height * len(cover.getbands()))
            psnr = 20 * math.log10(255) - 10 * math.log10(mse)
            assert math.isclose(mse, row["mse"], abs_tol=1e-12)
            assert math.isclose(psnr, row["psnr_db"], abs_tol=1e-12)
    for row in first["rows"]:
        with Image.open(folders[0] / row["file"]) as saved:
            assert list(saved.size) == row["dimensions"]
            with Image.open(folders[0] / f"images/{row['image']}-{row['resolution']}-marked.png") as marked:
                size = row["resolution"]
                match row["change"]:
                    case "none":
                        expected = marked.copy()
                    case "downscale50":
                        expected = marked.resize((size // 2, size // 2), Image.Resampling.LANCZOS)
                    case "crop80":
                        side = round(size * settings["crop_side_fraction"])
                        offset = (size - side) // 2
                        expected = marked.crop((offset, offset, offset + side, offset + side))
                    case "brightness":
                        expected = ImageEnhance.Brightness(marked).enhance(settings["brightness_factor"])
                    case "blur":
                        expected = marked.filter(ImageFilter.GaussianBlur(settings["blur_radius"]))
                    case "jpeg90" | "jpeg70" | "jpeg50":
                        expected = None
                        assert saved.format == "JPEG"
                    case _:
                        raise ValueError(row["change"])
                if expected is not None:
                    assert np.array_equal(np.asarray(saved), np.asarray(expected)), row["file"]
    for name, key in (("measurements", "rows"), ("quality", "quality"), ("unmarked", "unmarked")):
        with (folders[0] / f"{name}.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        assert len(records) == len(first[key])
        for record, raw in zip(records, first[key], strict=True):
            assert record == {key: str(value) for key, value in raw.items()}, name
    print("PASS: both complete raw runs and all saved image bytes are identical")
    print("PASS: input checksums, RGB conversion, and Lanczos enlargement")
    print("PASS: all 32 bit accuracies, exact payload flags, and raw API returns")
    print("PASS: four PSNR values independently recomputed from saved image histograms")
    print("PASS: change dimensions/pixels and JSON/CSV consistency")


if __name__ == "__main__":
    main()
