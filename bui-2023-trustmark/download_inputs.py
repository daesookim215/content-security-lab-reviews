import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parent
target = root / "images"
target.mkdir(exist_ok=True)
manifest = json.loads((root / "input_manifest.json").read_text())
for entry in manifest:
    path = target / entry["filename"]
    if not path.exists():
        with urllib.request.urlopen(entry["url"], timeout=90) as response:
            path.write_bytes(response.read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
(target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Two input images downloaded and verified")
