"""Load local weights into the official KL-f8 architecture, without training imports."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "safetensors", "numpy"]
# ///
# Run: .venv/bin/python experiment/run_test.py
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

ROOT: Final = Path(__file__).resolve().parents[1]
INPUT_ROOT: Final = Path(os.environ.get("STABLE_SIGNATURE_INPUT_ROOT", str(ROOT)))
KEY: Final = "111010110101000001010111010011010100010000100111"
REVISION: Final = "4e63672c03103b6c636b8fb4119ba982469b2955"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Vae(nn.Module):
    """The four first-stage modules used in the official autoencoding path."""

    def __init__(self) -> None:
        super().__init__()
        source = ROOT / "code/stable_signature/src/ldm/modules/diffusionmodules/model.py"
        names = {"nonlinearity", "Normalize", "Upsample", "Downsample", "ResnetBlock",
                 "AttnBlock", "make_attn", "Encoder", "Decoder"}
        definitions = [n for n in ast.parse(source.read_text()).body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in definitions} == names
        module = ModuleType("official_vae_architecture")
        module.__dict__.update(torch=torch, nn=nn, np=np, XFORMERS_IS_AVAILBLE=False)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"),
             module.__dict__)
        config = dict(ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2,
                      attn_resolutions=(), dropout=0., in_channels=3, resolution=256,
                      z_channels=4, double_z=True)
        self.encoder = module.Encoder(**config)
        self.decoder = module.Decoder(**config)
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)

    def encode_mode(self, image: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(image)).chunk(2, dim=1)[0]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(latent))


def mirror_key(key: str) -> str:
    """Rename original CompVis stages to the mirror's Diffusers 0.10 state names."""
    key = re.sub(r"\.down\.(\d+)\.block\.(\d+)\.", r".down_blocks.\1.resnets.\2.", key)
    key = re.sub(r"\.down\.(\d+)\.downsample\.", r".down_blocks.\1.downsamplers.0.", key)
    key = re.sub(r"\.up\.(\d+)\.block\.(\d+)\.",
                 lambda m: f".up_blocks.{3-int(m[1])}.resnets.{m[2]}.", key)
    key = re.sub(r"\.up\.(\d+)\.upsample\.",
                 lambda m: f".up_blocks.{3-int(m[1])}.upsamplers.0.", key)
    key = key.replace(".mid.block_1.", ".mid_block.resnets.0.")
    key = key.replace(".mid.block_2.", ".mid_block.resnets.1.")
    key = key.replace(".nin_shortcut.", ".conv_shortcut.")
    key = key.replace(".norm_out.", ".conv_norm_out.")
    for old, new in (("norm", "group_norm"), ("q", "query"), ("k", "key"),
                     ("v", "value"), ("proj_out", "proj_attn")):
        key = key.replace(f".mid.attn_1.{old}.", f".mid_block.attentions.0.{new}.")
    return key


def load_pair() -> tuple[Vae, dict[str, torch.Tensor], torch.jit.ScriptModule]:
    """Require a bijective mirror mapping and complete released decoding state."""
    for line in (INPUT_ROOT / "weights/SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert sha256(INPUT_ROOT / "weights" / name) == digest, name
    vae = Vae().eval()
    mirror = load_file(INPUT_ROOT / "weights/sd21base-vae/diffusion_pytorch_model.safetensors")
    converted = {}
    mapping = []
    for key, template in vae.state_dict().items():
        source_key = mirror_key(key)
        value = mirror[source_key]
        assert value.numel() == template.numel(), key
        # Diffusers stores attention 1×1 convolutions as matrices; only singleton axes change.
        assert tuple(d for d in value.shape if d != 1) == tuple(d for d in template.shape if d != 1), key
        converted[key] = value.reshape(template.shape)
        mapping.append({"official": key, "mirror": source_key,
                        "official_shape": list(template.shape), "mirror_shape": list(value.shape)})
    assert {r["mirror"] for r in mapping} == set(mirror)
    vae.load_state_dict(converted, strict=True)
    marked = torch.load(INPUT_ROOT / "weights/sd2_decoder.pth", map_location="cpu", weights_only=True)
    intended = {k: v for k, v in converted.items()
                if k.startswith(("decoder.", "post_quant_conv."))}
    assert set(marked) == set(intended)
    rows = []
    numerator = denominator = 0.
    for key, value in marked.items():
        reference = intended[key]
        assert value.shape == reference.shape, key
        difference_sq = (value.double() - reference.double()).square().sum().item()
        reference_sq = reference.double().square().sum().item()
        numerator += difference_sq
        denominator += reference_sq
        rows.append({"tensor": key, "shape": list(value.shape), "parameters": value.numel(),
                     "difference_l2": difference_sq**.5, "reference_l2": reference_sq**.5,
                     "relative_l2": (difference_sq/reference_sq)**.5})
    layers = {}
    for row in rows:
        layer = row["tensor"].rsplit(".",1)[0]
        entry = layers.setdefault(layer,{"layer":layer,"difference_sq":0.,"reference_sq":0.})
        entry["difference_sq"] += row["difference_l2"]**2
        entry["reference_sq"] += row["reference_l2"]**2
    layer_rows = [{"layer":v["layer"],"relative_l2":(v["difference_sq"]/v["reference_sq"])**.5,
                   "difference_l2":v["difference_sq"]**.5,"reference_l2":v["reference_sq"]**.5}
                  for v in layers.values()]
    report = {"mirror_revision": REVISION,
              "mirror_vae_config":json.loads((INPUT_ROOT/"weights/sd21base-vae/config.json").read_text()),
              "keys_identical_after_documented_mapping": True,
              "shapes_identical_after_attention_reshape": True, "tensor_count": len(rows),
              "parameters": sum(v.numel() for v in marked.values()),
              "overall_relative_l2": (numerator/denominator)**.5,
              "overall_difference_l2": numerator**.5, "overall_reference_l2": denominator**.5,
              "largest_relative_changes": sorted(rows, key=lambda r: r["relative_l2"], reverse=True)[:15],
              "largest_absolute_changes": sorted(rows, key=lambda r: r["difference_l2"], reverse=True)[:15],
              "largest_changed_layers":sorted(layer_rows,key=lambda r:r["relative_l2"],reverse=True)[:15],
              "per_tensor": rows, "key_mapping": mapping,
              "interpretation": "Compatible architecture; difference includes watermark adaptation and any unknown base-weight differences. Mirror identity with the authors' private baseline is unestablished."}
    (ROOT / "experiment/mirror_check.json").write_text(json.dumps(report, indent=2)+"\n")
    extractor = torch.jit.load(INPUT_ROOT / "weights/dec_48b_whit.torchscript.pt", map_location="cpu").eval()
    (ROOT / "experiment/released_module_tree.txt").write_text(str(extractor)+"\n"+str(vae.decoder))
    shapes = json.loads((ROOT / "experiment/shapes_params.json").read_text())
    shapes["released checkpoint"] = {
        "decoder": {"parameters": sum(v.numel() for k,v in marked.items() if k.startswith("decoder.")),
                    "input_hwc": [64,64,4], "output_hwc": [512,512,3],
                    "blocks": "conv_in; ResNet-attention-ResNet; 4 up stages of 3 ResNet blocks, 3 nearest-neighbor+conv upsamplers; GroupNorm-SiLU-conv_out",
                    "state_shapes": {k:list(v.shape) for k,v in marked.items()}},
        "post_quant_conv": {"parameters":20,"shape":[4,4,1,1]},
        "extractor_whitened": {"parameters":sum(p.numel() for p in extractor.parameters()),
             "input_hwc":[512,512,3],"output":[48],"conv_blocks_64":8,"conv_blocks_48":1,
             "block":"3×3 convolution, BatchNorm eps .001, GELU", "pool":"global average",
             "linear_layers":[[48,48],[48,48]],
             "state_shapes":{k:list(v.shape) for k,v in extractor.state_dict().items()}},
        "mirror_encoder": {"parameters":sum(p.numel() for p in vae.encoder.parameters()),
                           "moments_hwc":[64,64,8],"mode_hwc":[64,64,4]}}
    (ROOT / "experiment/shapes_params.json").write_text(json.dumps(shapes,indent=2)+"\n")
    return vae, marked, extractor
