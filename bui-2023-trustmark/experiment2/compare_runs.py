# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: ../.venv/bin/python compare_runs.py reproduced
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "reproduced")
    for run in ("run1", "run2"):
        for name in ("results.json", "measurements.csv", "timings.json"):
            if not (root / run / name).is_file():
                raise FileNotFoundError(root / run / name)
    files = [sorted(str(p.relative_to(root / run)) for p in (root / run).rglob("*") if p.is_file())
             for run in ("run1", "run2")]
    differences = []
    checked = []
    for relative in sorted(set(files[0]) | set(files[1])):
        if relative == "timings.json":
            continue
        paths = [root / run / relative for run in ("run1", "run2")]
        digests = [hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in paths]
        checked.append({"file": relative, "run1": digests[0], "run2": digests[1]})
        if digests[0] != digests[1]:
            differences.append(checked[-1])
    log_hashes = [hashlib.sha256((root / f"{run}.log").read_bytes()).hexdigest() for run in ("run1", "run2")]
    if log_hashes[0] != log_hashes[1]:
        differences.append({"file": "run.log", "run1": log_hashes[0], "run2": log_hashes[1]})
    data = {"byte_identical_except_timing": len(differences) == 0, "differences": differences,
            "excluded_files": ["timings.json"], "checked_files": checked, "log_sha256": log_hashes,
            "policy": "All output files including full-resolution images are byte-compared. Only timing files excluded."
            " 2048 gate measured in run1 and replayed in run2."}
    (root / "determinism.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k != "checked_files"}, indent=2), flush=True)
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
