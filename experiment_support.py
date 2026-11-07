"""Shared image preparation and exact comparison of recorded measurements."""
import csv
import json
import subprocess
import sys
from enum import StrEnum
from pathlib import Path
from typing import assert_never

from PIL import Image
from fetch_images import ROOT, SOURCES, verify_file


class ResultFormat(StrEnum):
    JSON = ".json"
    CSV = ".csv"
    TEXT = ".txt"


def verify_inputs() -> None:
    for source in SOURCES:
        verify_file(source)


def prepare_inputs(paper: Path) -> None:
    """Verify fetched inputs and put lossless grayscale working files in outputs."""
    destination = paper / "outputs/inputs"
    destination.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        with Image.open(verify_file(source)) as image:
            if image.size != (512, 512):
                raise SystemExit(f"FAIL input dimensions: {source.filename}")
            gray = image.convert("L")
            gray.save(destination / f"{source.name}.png")
            gray.save(destination / f"{source.name}-gray.png")


def run_script(paper: Path, arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-B", str(paper / arguments[0]), *arguments[1:]],
                   cwd=ROOT, check=True)


def compare_results(reference: Path, regenerated: Path) -> int:
    """Require exact JSON values and CSV cells, including run identifiers."""
    files = sorted(path for path in reference.rglob("*") if path.is_file())
    if not files:
        raise SystemExit("FAIL empty reference results")
    for expected in files:
        relative = expected.relative_to(reference)
        actual = regenerated / relative
        match ResultFormat(expected.suffix):
            case ResultFormat.JSON:
                equal = json.loads(expected.read_text()) == json.loads(actual.read_text())
            case ResultFormat.CSV:
                with expected.open(newline="") as left, actual.open(newline="") as right:
                    left_rows, right_rows = list(csv.DictReader(left)), list(csv.DictReader(right))
                equal = left_rows == right_rows
            case ResultFormat.TEXT:
                equal = expected.read_bytes() == actual.read_bytes()
            case unreachable:
                assert_never(unreachable)
        if not equal:
            raise SystemExit(f"FAIL recorded measurements differ: {relative}")
    return len(files)
