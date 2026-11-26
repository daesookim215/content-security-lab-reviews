# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "torchvision==0.29.2", "kornia==0.9.2", "Pillow==13.0.0", "numpy==2.4.6", "scipy==1.17.1", "scikit-image==0.26.0"]
# ///
# Run: imported by run_test.py and inspect_models.py
"""Local provenance, deterministic runtime and official model loading."""
from dataclasses import dataclass
from pathlib import Path
from typing import Final
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import numpy as np
import torch

ROOT: Final = Path(os.environ.get('MBRS_INPUTS', Path(__file__).resolve().parents[1])).resolve()
SEED: Final = 20261126
IMAGES: Final = {'Airplane': ('4.2.05.tiff','ae9eba12b80df955e47585bf4a16332ea0dadc753668d3457c509564934dbd40'), 'Boat': ('boat.512.tiff','6bc7319b1e3b44ce0b2a571ac39a28bac94437d812057d1afb2cf78ea3a9a35f')}

@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    side: int
    length: int
    epoch: int
    diffusion: bool

MODELS: Final = (ModelSpec('MBRS_256_m256',256,256,42,False), ModelSpec('MBRS_Diffusion_128_m30',128,30,114,True))

def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def verify_inputs() -> None:
    for filename, expected in IMAGES.values():
        assert hashlib.sha256((ROOT/'data'/filename).read_bytes()).hexdigest() == expected, filename
    for line in (ROOT/'weights/SHA256SUMS').read_text().splitlines():
        expected, filename = line.split()
        assert hashlib.sha256((ROOT/'weights'/filename).read_bytes()).hexdigest() == expected, filename

def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def configure() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    seed_all()

def environment():
    return {'python':sys.version, 'packages':{n:importlib.metadata.version(n) for n in ('torch','torchvision','kornia','Pillow','numpy','scipy','scikit-image','pypdfium2')},'platform':sys.platform}

def load_model(spec: ModelSpec):
    sys.path.insert(0,str(ROOT/'code/MBRS'))
    from network.Encoder_MP_Decoder import EncoderDecoder, EncoderDecoder_Diffusion
    from network.Discriminator import Discriminator
    factory = EncoderDecoder_Diffusion if spec.diffusion else EncoderDecoder
    net = factory(spec.side,spec.side,spec.length,['Identity()']).cpu().float().eval()
    dis = Discriminator().cpu().float().eval()
    folder = ROOT/'weights'/spec.name
    ec_state = torch.load(folder/f'EC_{spec.epoch}.pth',map_location='cpu',weights_only=True)
    d_state = torch.load(folder/f'D_{spec.epoch}.pth',map_location='cpu',weights_only=True)
    ec_report = net.load_state_dict(ec_state,strict=True)
    d_report = dis.load_state_dict(d_state,strict=True)
    assert not ec_report.missing_keys and not ec_report.unexpected_keys
    assert not d_report.missing_keys and not d_report.unexpected_keys
    return net,dis,ec_state,d_state
