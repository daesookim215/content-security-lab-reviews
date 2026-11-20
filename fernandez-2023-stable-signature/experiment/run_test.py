"""CPU VAE-only inference, saved received pixels, numeric evidence."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "safetensors", "Pillow", "numpy", "scipy", "scikit-image"]
# ///
# Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiment/run_test.py
from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import PIL
import torch
from PIL import Image
from scipy.stats import binom
from skimage.metrics import structural_similarity

from attacks import attacks
from model_io import INPUT_ROOT, KEY, REVISION, ROOT, load_pair, sha256

INPUTS: Final = (
    ("Airplane","4.2.05.tiff","ae9eba12b80df955e47585bf4a16332ea0dadc753668d3457c509564934dbd40"),
    ("Boat","boat.512.tiff","6bc7319b1e3b44ce0b2a571ac39a28bac94437d812057d1afb2cf78ea3a9a35f"))


def pixels(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()[None]/255


def psnr(left: np.ndarray, right: np.ndarray) -> float | None:
    mse = np.mean((left.astype(np.float64)-right.astype(np.float64))**2)
    return float(10*math.log10(255**2/mse)) if mse else None


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"experiment/results/run1"
    images = out/"images"
    images.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(230315435)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    vae,marked,extractor = load_pair()
    plain = {k:v.clone() for k,v in vae.state_dict().items()
             if k.startswith(("decoder.","post_quant_conv."))}
    load_seconds = time.perf_counter()-started
    mean = torch.tensor([.485,.456,.406])[None,:,None,None]
    std = torch.tensor([.229,.224,.225])[None,:,None,None]
    rows,quality,times,serialization = [],[],[],[]
    example = {}
    for name,filename,digest in INPUTS:
        source = INPUT_ROOT/"data"/filename
        assert sha256(source)==digest, filename
        with Image.open(source) as handle:
            original = handle.convert("RGB")
        assert original.size==(512,512)
        original.save(images/f"{name}-original.png")
        clock = time.perf_counter()
        with torch.inference_mode():
            latent = vae.encode_mode(pixels(original)*2-1)
            times.append({"image":name,"encode_seconds":time.perf_counter()-clock})
            torch.save(latent,out/f"{name}-latent.pt")
            decoded = {}
            clipping = {}
            raw = {}
            for variant,state in (("plain",plain),("watermarked",marked)):
                vae.decoder.load_state_dict({k.removeprefix("decoder."):v for k,v in state.items()
                                            if k.startswith("decoder.")},strict=True)
                vae.post_quant_conv.load_state_dict({k.removeprefix("post_quant_conv."):v for k,v in state.items()
                                                    if k.startswith("post_quant_conv.")},strict=True)
                decode_clock=time.perf_counter()
                value = (vae.decode(latent)+1)/2
                times.append({"image":name,"variant":variant,"decode_seconds":time.perf_counter()-decode_clock})
                clipping[variant] = int(((value<0)|(value>1)).sum())
                raw[variant] = value[0].permute(1,2,0).numpy().copy()
                np.save(out/f"{name}-{variant}-float.npy",raw[variant])
                array = value.clamp(0,1).mul(255).round().byte()[0].permute(1,2,0).numpy()
                Image.fromarray(array).save(images/f"{name}-{variant}.png")
                with Image.open(images/f"{name}-{variant}.png") as handle:
                    decoded[variant] = handle.convert("RGB")
            times.append({"image":name,"encode_and_two_decodes_seconds":time.perf_counter()-clock})
            serialization.append({"image":name,"clipped_values":clipping,
                                  "rounding":"clip [0,1], multiply 255, round ties to even, uint8"})
            for left,right in (("watermarked","plain"),("watermarked","original"),("plain","original")):
                a=np.asarray(decoded[left]); b=np.asarray(original if right=="original" else decoded[right])
                quality.append({"image":name,"comparison":f"{left}_vs_{right}","psnr_db":psnr(a,b),
                                "ssim":float(structural_similarity(a,b,channel_axis=2,data_range=255))})
            received = [("original","none",original,"Original photograph",None),
                        ("plain","none",decoded["plain"],"Plain VAE reconstruction",None)]
            attack_clock=time.perf_counter()
            received += [(r.attack,r.strength,r.image,r.details,r.jpeg) for r in attacks(decoded["watermarked"])]
            times.append({"image":name,"all_attack_construction_seconds":time.perf_counter()-attack_clock})
            for attack,strength,image,details,jpeg_bytes in received:
                path=images/f"{name}-{attack}-{strength}.png"
                image.save(path)
                if jpeg_bytes is not None:
                    path.with_suffix(".jpg").write_bytes(jpeg_bytes)
                clock=time.perf_counter()
                logits=extractor((pixels(image)-mean)/std)[0].tolist()
                bits="".join("1" if value>0 else "0" for value in logits)
                matches=sum(a==b for a,b in zip(bits,KEY,strict=True))
                row={"image":name,"source_sha256":digest,"attack":attack,"strength":strength,
                     "width":image.width,"height":image.height,"received_png":str(path.relative_to(out)),
                     "received_sha256":sha256(path),"details":details,"logits":logits,"bits":bits,
                     "matching_bits":matches,"denominator":48,"accuracy":matches/48,
                     "detection_pass":matches>=41,"identification_1000_pass":matches>=44,
                     "paper_detection_strict_gt41":matches>41,"paper_identification_1000_strict_gt44":matches>44,
                     "eq1_inclusive_ge41":matches>=41,"eq1_inclusive_ge44":matches>=44,
                     "inclusive_null_tail":float(binom.sf(matches-1,48,.5)),
                     "psnr_vs_original_db":psnr(np.asarray(image),np.asarray(original)) if image.size==original.size else None,
                     "psnr_vs_watermarked_db":psnr(np.asarray(image),np.asarray(decoded["watermarked"]))
                                              if image.size==original.size and attack!="identity" else None}
                rows.append(row)
                times.append({"image":name,"attack":attack,"strength":strength,
                              "extract_seconds":time.perf_counter()-clock})
                if name=="Boat" and attack=="identity":
                    example={"image":"Boat","coordinates":"red channel y252:256,x252:256; zero-based",
                             "original":np.asarray(original)[252:256,252:256,0].tolist(),
                             "plain":np.asarray(decoded["plain"])[252:256,252:256,0].tolist(),
                             "watermarked":np.asarray(decoded["watermarked"])[252:256,252:256,0].tolist(),
                             "difference":(np.asarray(decoded["watermarked"]).astype(int)-np.asarray(decoded["plain"]).astype(int))[252:256,252:256,0].tolist(),
                             "raw_plain":raw["plain"][252:256,252:256,0].tolist(),
                             "raw_watermarked":raw["watermarked"][252:256,252:256,0].tolist(),
                             "key":KEY,"extracted_bits":bits,"logits":logits,"matching_bits":matches,
                             "mismatches_1based":[i+1 for i,(a,b) in enumerate(zip(bits,KEY,strict=True)) if a!=b],
                             "quality":[r for r in quality if r["image"]==name],"clipping":clipping}
    settings={"key":KEY,"key_booleans":[b=="1" for b in KEY],"seed":230315435,"threads":2,
              "device":"CPU float32","posterior":"mode; no diffusion scaling",
              "preprocessing":"RGB/255; ImageNet mean .485,.456,.406 std .229,.224,.225; no resizing",
              "serialization":"clip [0,1], multiply 255, round ties to even, uint8, PNG, reopen",
              "crop_and_resize":"parameters are retained area, native decoding without restoration",
              "mirror_revision":REVISION,"code_commit":"c91217c06e953a4eaf270ef9de4a3939bcea3366",
              "weight_sha256":(INPUT_ROOT/"weights/SHA256SUMS").read_text(),
              "paper_threshold_rule":"Primary: Eq1 M>=41 detection, M>=44 for N=1000 identification, at global nominal FPR 1e-6. These are the minimal inclusive cutoffs using the notebook's inclusive binomtest p-value. Eq2 prints >; both comparator flags retained. Original evaluation implementation unavailable.",
              "psnr_null":"Identical images have infinite PSNR; unaligned dimensions are inapplicable; both serialized as null, distinguishable by attack and dimensions.",
              "tails":{"ge41":float(binom.sf(40,48,.5)),"gt41":float(binom.sf(41,48,.5)),
                       "ge44":float(binom.sf(43,48,.5)),"gt44":float(binom.sf(44,48,.5))},
              "environment":{"Python":platform.python_version(),"torch":torch.__version__,"Pillow":PIL.__version__,"numpy":np.__version__}}
    for filename,value in (("settings.json",settings),("results.json",rows),("quality.json",quality),
                           ("example_values.json",example),("timing.json",{"load_seconds":load_seconds,"steps":times}),
                           ("serialization.json",serialization),
                           ("load_report.json",{"mirror_all_keys_consumed":True,"decoder_strict":True,"post_quant_conv_strict":True})):
        (out/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    for filename,values in (("measurements.csv",rows),("quality.csv",quality)):
        with (out/filename).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    print(json.dumps({"rows":len(rows),"quality":quality,"tails":settings["tails"]},indent=2))


if __name__=="__main__":
    main()
