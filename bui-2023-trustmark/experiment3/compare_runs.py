# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python compare_runs.py [directory containing run1 and run2].
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    files = sorted({str(p.relative_to(root / run)) for run in ("run1", "run2")
                    for p in (root / run).rglob("*") if p.is_file()} - {"timings.json"})
    checked = []
    for name in files:
        digests = [hashlib.sha256((root / run / name).read_bytes()).hexdigest()
                   for run in ("run1", "run2")]
        checked.append({"file": name, "run1": digests[0], "run2": digests[1]})
    logs = [hashlib.sha256((root / f"{run}.log").read_bytes()).hexdigest() for run in ("run1", "run2")]
    differences = [r for r in checked if r["run1"] != r["run2"]]
    data = {"byte_identical_except_timing": not differences and logs[0] == logs[1],
            "checked_files": checked, "differences": differences, "log_sha256": logs,
            "excluded_files": ["timings.json"]}
    (root / "determinism.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    assert data["byte_identical_except_timing"], data
    print(f"PASS: {len(checked)} files and logs byte-identical; only timings.json excluded")


if __name__ == "__main__":
    main()
