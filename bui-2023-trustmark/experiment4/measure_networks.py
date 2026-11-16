# /// script
# requires-python = ">=3.11"
# dependencies = ["trustmark==0.9.2", "torch==2.5.1", "torchvision==0.20.1", "omegaconf", "pillow==11.3.0"]
# ///
# How to run: sh reproduce_exp4.sh (uses the parent .venv and verified cache).
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, TypeAlias, TypedDict, Union, assert_never

ROOT: Final = Path(__file__).resolve().parent
BASE: Final = ROOT.parent
MODELS: Final = ROOT / ".cache/models"
RESULTS: Final = ROOT / "results"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable, directory in (
    ("XDG_CACHE_HOME", ROOT / ".cache"),
    ("XDG_CONFIG_HOME", ROOT / ".cache/config"),
    ("TORCH_HOME", ROOT / ".cache/torch"),
    ("MPLCONFIGDIR", ROOT / ".cache/matplotlib"),
    ("TMPDIR", ROOT / ".cache/tmp"),
):
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import to_tensor
from trustmark import TrustMark
from trustmark.trustmark import MODEL_CHECKSUMS

sys.path.insert(0, str(BASE))
from run_experiment import ExistingModelsTrustMark

EXPECTED: Final = {
    "decoder_Q.ckpt": "4ced90e9cfe13e3295ad082887fe9187",
    "encoder_Q.ckpt": "700328b8754db934b2f6cb5e5185d81f",
    "trustmark_Q.yaml": "fe40df84a7feeebfceb7a7678d7e6ec6",
    "trustmark_rm_Q.yaml": "8476bcd4092abf302272868f3b4c2e38",
    "trustmark_rm_Q.ckpt": "760337a5596e665aed2ab5c49aa5284f",
}
TensorValue: TypeAlias = Union[torch.Tensor, tuple["TensorValue", ...], list["TensorValue"], None]
Shape: TypeAlias = Union[list[int], list["Shape"], None]
Attribute: TypeAlias = str | int | float | bool | tuple[int, ...] | tuple[float, ...]


class HookRecord(TypedDict):
    module: str
    type: str
    input: Shape
    output: Shape


class ModuleInfo(TypedDict):
    type: str
    parameters: int
    direct_parameters: int
    attributes: dict[str, Attribute]


class CachedModelsTrustMark(ExistingModelsTrustMark):
    """Redirect every model to this experiment's MD5-checked cache."""

    # The argument list is the package's load_model override contract.
    def load_model(self, config_path: str, weight_path: str, device: str,
                   secret_len: int, part: str = "all") -> torch.nn.Module:
        return super().load_model(
            str(MODELS / Path(config_path).name),
            str(MODELS / Path(weight_path).name), device, secret_len, part=part)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_shapes(value: TensorValue) -> Shape:
    match value:
        case torch.Tensor():
            return list(value.shape)
        case tuple() | list():
            return [tensor_shapes(item) for item in value]
        case None:
            return None
        case unreachable:
            assert_never(unreachable)


def hook_for(records: list[HookRecord], name: str) -> Callable[[torch.nn.Module, tuple[TensorValue, ...], TensorValue], None]:
    """Collect tensor shapes without retaining tensors or changing outputs."""
    def record(module: torch.nn.Module, inputs: tuple[TensorValue, ...], output: TensorValue) -> None:
        records.append({"module": name, "type": type(module).__name__,
                        "input": tensor_shapes(inputs), "output": tensor_shapes(output)})
    return record


def inventory(model: torch.nn.Module) -> Mapping[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    for name, module in model.named_modules():
        attributes = {}
        for key in ("in_channels", "out_channels", "kernel_size", "stride", "padding",
                    "in_features", "out_features", "scale_factor", "mode", "resolution",
                    "secret_len", "ndown", "arch", "padder_size"):
            if hasattr(module, key):
                attributes[key] = getattr(module, key)
        modules[name] = {"type": type(module).__name__,
                         "parameters": sum(p.numel() for p in module.parameters()),
                         "direct_parameters": sum(p.numel() for p in module.parameters(recurse=False)),
                         "attributes": attributes}
    return modules


def main() -> None:
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20231130)
    torch.use_deterministic_algorithms(True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    package = Path(sys.modules[TrustMark.__module__].__file__).resolve().parent
    hashes, configs = [], {}
    for name, expected in EXPECTED.items():
        path = MODELS / name
        content = path.read_bytes()
        md5 = hashlib.md5(content).hexdigest()
        assert md5 == MODEL_CHECKSUMS[name] == expected, name
        hashes.append({"file": name, "path": str(path.relative_to(BASE)), "bytes": len(content),
                       "md5": md5, "official_md5_match": True, "sha256": digest(path)})
        if path.suffix == ".yaml":
            (RESULTS / name).write_bytes(content)
            configs[name] = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    source_hashes = {str(p.relative_to(package)): digest(p)
                     for p in sorted(package.rglob("*.py"))}
    model = CachedModelsTrustMark(device="cpu", model_type="Q", use_ECC=True,
                                encoding_type=TrustMark.Encoding.BCH_5, verbose=False,
                                loadRemover=True, loadBBoxDetector=False)
    assert model.encoder is not None and model.decoder is not None and model.removal is not None
    networks = {"embedder": model.encoder.encoder, "extractor": model.decoder.decoder,
                "remover": model.removal}
    trees, records, parameter_counts = [], {}, {}
    handles = []
    for name, network in networks.items():
        network.eval().requires_grad_(False)
        assert all(p.device.type == "cpu" for p in network.parameters())
        trees.append(f"=== {name} ===\n{network}\n")
        records[name] = []
        parameter_counts[name] = {
            "total": sum(p.numel() for p in network.parameters()),
            "top_level": {key: sum(p.numel() for p in child.parameters())
                          for key, child in network.named_children()},
            "modules": inventory(network)}
        for key, child in network.named_modules():
            handles.append(child.register_forward_hook(hook_for(records[name], key)))
    cover = torch.linspace(-1, 1, 3 * 256 * 256).reshape(1, 3, 256, 256)
    message = (torch.arange(model.secret_len) % 2).float().unsqueeze(0)
    image = Image.new("RGB", (256, 256), (128, 128, 128))
    public_input = to_tensor(image.resize((model.model_resolution_dec,) * 2,
                                         Image.Resampling.BILINEAR)).unsqueeze(0) * 2 - 1
    with torch.inference_mode():
        encoded = networks["embedder"](cover, message)
        logits = networks["extractor"](public_input)
        probabilities = logits.sigmoid()
        removed = networks["remover"](encoded)
    for handle in handles:
        handle.remove()
    public_records = []
    public_handles = [child.register_forward_hook(hook_for(public_records, key))
                      for key, child in networks["extractor"].named_modules()]
    with torch.inference_mode():
        public_result = model.subimage_decode(image, "binary")
    for handle in public_handles:
        handle.remove()
    encoder = networks["embedder"]
    config_dump = {"yaml": configs, "observed": {
        "message_length_l": model.secret_len, "E_pre_d": encoder.pre.conv.out_channels,
        "body_output_n": encoder.dec[-1].conv2.conv.out_channels,
        "downsampling_stages": len(encoder.enc), "upsampling_stages": len(encoder.dec),
        "residual_blocks": sum(type(m).__name__ == "ResBlock" for m in encoder.modules()),
        "E_post": [str(layer) for layer in encoder.post],
        "activation_types": {name: sorted({type(m).__name__ for m in network.modules()
                                          if type(m).__name__ in
                                          {"ReLU", "SiLU", "Tanh", "GELU", "Sigmoid", "SimpleGate"}})
                             for name, network in networks.items()},
        "decoder_api_resize": model.model_resolution_dec,
        "decoder_wrapper_resolution": networks["extractor"].resolution,
        "encoder_resolution": model.model_resolution_enc,
        "remover_resolution": model.model_resolution_remove}}
    data = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
             "torchvision": importlib.metadata.version("torchvision"),
             "trustmark": importlib.metadata.version("trustmark"), "device": "cpu",
             "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
             "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "pythonpath": ""},
            "model_files": hashes, "source_sha256": source_hashes,
            "script_sha256": digest(Path(__file__)), "config": config_dump["observed"],
            "parameters": parameter_counts, "hooks": records,
            "public_extractor_hooks": public_records,
            "public_decode_result": [str(item) for item in public_result],
            "functional_outputs": {"cover": tensor_shapes(cover), "message": tensor_shapes(message),
                "encoded": tensor_shapes(encoded), "wrapper_logits": tensor_shapes(logits),
                "reference_sigmoid": tensor_shapes(probabilities),
                "removed": tensor_shapes(removed)},
            "notes": {"sigmoid": "Reference transform only: package returns logits and public API thresholds >0.",
                      "decoder_paths": "Public subimage_decode resizes to 245; SecretDecoder.forward internally resizes to 224.",
                      "shapes": "Raw tensors are NCHW, vectors are batch by features; diagrams use H×W×C.",
                      "message": "Alternating synthetic 100-bit input, not an image-quality or recovery experiment."}}
    (RESULTS / "module_tree.txt").write_text("\n".join(trees), encoding="utf-8")
    (RESULTS / "config_dump.json").write_text(json.dumps(config_dump, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "shapes_params.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parameters": {k: v["total"] for k, v in parameter_counts.items()},
                      "config": config_dump["observed"]}, indent=2))


if __name__ == "__main__":
    main()
