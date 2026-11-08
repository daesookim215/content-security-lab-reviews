import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
specification = importlib.util.find_spec("trustmark")
if specification is None or specification.origin is None:
    raise ImportError("Install trustmark==0.9.2 first")
package = Path(specification.origin).resolve().parent
records = json.loads((root / "bch_compatibility.json").read_text())
digests = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in records}
if not all(digests[name] == entry["tested_sha256"] for name, entry in records.items()):
    for name, entry in records.items():
        assert digests[name] == entry["pypi_sha256"], f"Unexpected source: {name}"
    subprocess.run(["patch", "--batch", "--forward", "-p1", "-d", str(package),
                    "-i", str(root / "bch_compatibility.patch")], check=True)
for name, entry in records.items():
    assert hashlib.sha256((package / name).read_bytes()).hexdigest() == entry["tested_sha256"]
print("BCH compatibility files match the tested source hashes")
