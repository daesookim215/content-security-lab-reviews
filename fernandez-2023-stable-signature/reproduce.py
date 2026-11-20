"""Fetch hash-pinned inputs, run the CPU VAE path, compare numeric results exactly."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "numpy==2.4.6", "Pillow==13.0.0", "safetensors==0.8.0", "scipy==1.17.1", "scikit-image==0.26.0"]
# ///
# Run: uv run reproduce.py
# Offline: python reproduce.py --local-inputs /path/to/review --output /path/to/run
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent
REVISION: Final = "4e63672c03103b6c636b8fb4119ba982469b2955"
SOURCES: Final = (
    ("https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt",
     "weights/dec_48b_whit.torchscript.pt", "77cd0a2040b9391233bbcd79c1adf00816b196089cbb844da40035f854637a04"),
    ("https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth",
     "weights/sd2_decoder.pth", "36f3a926ba080a0fe29952cd3895661736da68619a71c8166d5c44de3612965a"),
    (f"https://huggingface.co/sd2-community/stable-diffusion-2-1-base/resolve/{REVISION}/vae/config.json",
     "weights/sd21base-vae/config.json", "424117cb534ce03497c41305ed868980123917b2b6abba4bbaa615e968772903"),
    (f"https://huggingface.co/sd2-community/stable-diffusion-2-1-base/resolve/{REVISION}/vae/diffusion_pytorch_model.safetensors",
     "weights/sd21base-vae/diffusion_pytorch_model.safetensors", "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"),
    ("https://sipi.usc.edu/database/misc/4.2.05.tiff", "data/4.2.05.tiff",
     "ae9eba12b80df955e47585bf4a16332ea0dadc753668d3457c509564934dbd40"),
    ("https://sipi.usc.edu/database/misc/boat.512.tiff", "data/boat.512.tiff",
     "6bc7319b1e3b44ce0b2a571ac39a28bac94437d812057d1afb2cf78ea3a9a35f"),
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def fetch_inputs(cache: Path) -> None:
    import shutil
    import urllib.request
    for url,name,expected in SOURCES:
        destination=cache/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            temporary=destination.with_suffix(destination.suffix+".part")
            request=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(request,timeout=60) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response,handle,1<<20)
            assert digest(temporary)==expected,name
            temporary.replace(destination)
        assert digest(destination)==expected,name
    (cache/"weights/SHA256SUMS").write_bytes((ROOT/"weights/SHA256SUMS").read_bytes())


def main() -> None:
    local="--local-inputs" in sys.argv
    inputs=Path(sys.argv[sys.argv.index("--local-inputs")+1]).resolve() if local else ROOT/".cache"
    output=Path(sys.argv[sys.argv.index("--output")+1]).resolve() if "--output" in sys.argv else ROOT/"run-output"
    if local:
        for _,name,expected in SOURCES:assert digest(inputs/name)==expected,name
    else:
        fetch_inputs(inputs)
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","STABLE_SIGNATURE_INPUT_ROOT":str(inputs)}
    subprocess.run([sys.executable,str(ROOT/"experiment/run_test.py"),str(output)],check=True,env=environment)
    comparisons=[]
    for name in ("results.json","quality.json","example_values.json","measurements.csv","quality.csv","settings.json","load_report.json","serialization.json"):
        expected=(ROOT/"results"/name).read_bytes();actual=(output/name).read_bytes()
        equal=actual==expected
        if name=="settings.json":
            left=json.loads(expected);right=json.loads(actual)
            for value in (left,right):
                value["environment"].pop("Python")
                value["environment"].pop("torch")
            equal=left==right
        comparisons.append({"file":name,"identical":equal,"actual_sha256":hashlib.sha256(actual).hexdigest()})
    report={"all_identical":all(row["identical"] for row in comparisons),"files":comparisons,
            "comparison":"Byte-exact results; settings parsed with only environment.Python and environment.torch excluded",
            "download_branch_exercised":not local,"received_images":len(json.loads((output/"results.json").read_text()))}
    (ROOT/"reproduce_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    assert report["all_identical"],"Exact reproduction differs; inspect files instead of applying tolerances"


if __name__=="__main__":
    main()
