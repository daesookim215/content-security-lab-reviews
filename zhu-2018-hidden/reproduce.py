# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.4.6", "pillow==13.0.0", "httpx2[http2,brotli,zstd]"]
# ///
"""Run: uv run reproduce.py. Fetch pinned inputs, run CPU inference, compare numeric files."""
from __future__ import annotations
import hashlib
import json
import os
import socket
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Final
import httpx2

ROOT: Final = Path(__file__).resolve().parent
COMMIT: Final = "556f76dd0602e4351ed4c19bd2ee87d50ef3c0de"
SOURCES: Final = (
    (f"https://codeload.github.com/ando-khachatryan/HiDDeN/zip/{COMMIT}",
        "ando-pinned.zip", "13ded3eb62de8ac996c90aa161a7a0856cdd9f2497b5953439f079b4eb753ca2"),
    ("https://sipi.usc.edu/database/misc/4.2.05.tiff", "4.2.05.tiff",
        "ae9eba12b80df955e47585bf4a16332ea0dadc753668d3457c509564934dbd40"),
    ("https://sipi.usc.edu/database/misc/boat.512.tiff", "boat.512.tiff",
        "6bc7319b1e3b44ce0b2a571ac39a28bac94437d812057d1afb2cf78ea3a9a35f"),
)

def main() -> None:
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    if "--fetch-with-ego" in sys.argv:
        module = (ROOT/"fetch_sources.mjs").as_uri()
        subprocess.run(["ego-browser","nodejs","-e",
            f"const source = await import({json.dumps(module)}); await source.run();"],check=True)
    limits = httpx2.Limits(max_connections=200,max_keepalive_connections=40,keepalive_expiry=30.)
    transport = httpx2.HTTPTransport(http2=True,retries=3,limits=limits,
        socket_options=[(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)])
    timeout = httpx2.Timeout(connect=5.,read=30.,write=10.,pool=10.)
    with httpx2.Client(transport=transport,timeout=timeout,follow_redirects=True) as client:
        for url,name,expected in SOURCES:
            path = cache / name
            if not path.exists():
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(response.content)
                print("Fetched",name,response.status_code)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    repo = cache / f"HiDDeN-{COMMIT}"
    with zipfile.ZipFile(cache / "ando-pinned.zip") as archive:
        prefix = f"HiDDeN-{COMMIT}/"
        selected = [name for name in archive.namelist() if name.startswith(prefix) and (
            name.startswith(prefix+"model/") or name==prefix+"options.py" or
            name.endswith("/combined-noise--epoch-400.pyt") or name.endswith("/no-noise--epoch-200.pyt"))]
        for name in selected:
            assert ".." not in Path(name).parts
            archive.extract(name,cache)
    output = ROOT / "run-output"
    environment = {**os.environ,"PYTHONDONTWRITEBYTECODE":"1"}
    subprocess.run([sys.executable,str(ROOT/"code/run_test.py"),str(repo),str(cache),str(output)],
        check=True,env=environment)
    comparisons = []
    for name in ("results.json","quality.json","example_values.json","measurements.csv","quality.csv","settings.json"):
        expected = (ROOT / "results" / name).read_bytes()
        actual = (output / name).read_bytes()
        if name == "settings.json":
            # Interpreter patch versions may differ; compare every other setting exactly.
            strip = lambda raw: {k: v for k, v in json.loads(raw).items() if k != "python"}
            equal = strip(actual) == strip(expected)
        else:
            equal = actual == expected
        comparisons.append({"file":name,"identical_bytes":equal,
            "sha256":hashlib.sha256(actual).hexdigest()})
    report = {"all_identical":all(item["identical_bytes"] for item in comparisons),
        "files":comparisons,"recovery_rows":len(json.loads((output/"results.json").read_text())),
        "checkpoint_loading":"weights_only=True; strict encoder and decoder; no training",
        "fetch_method":"ego-browser Node fetch" if "--fetch-with-ego" in sys.argv else "httpx2 or verified cache"}
    (ROOT / "reproduce_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    assert report["all_identical"]

if __name__ == "__main__":
    main()
