"""Runtime provenance and a conservative CUDA allocator budget."""

import hashlib
import importlib.metadata
import json
import platform
import random
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


def versions() -> dict[str, JsonValue]:
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    direct_url = importlib.metadata.distribution("videoseal").read_text("direct_url.json")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "trustmark": importlib.metadata.version("trustmark"),
        "videoseal": importlib.metadata.version("videoseal"),
        "videoseal_source": json.loads(direct_url) if direct_url else None,
        "requirements_sha256": hashlib.sha256(Path("requirements.txt").read_bytes()).hexdigest(),
        "packages": packages,
    }


def configure(seed: int) -> str:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        return "cpu"
    # This caps caching-allocator reservations at 6 decimal GB, leaving almost
    # 2 GB below the requested 8 GB for CUDA context/library allocations.
    total = torch.cuda.get_device_properties(0).total_memory
    budget = min(6_000_000_000, int(total * 0.7))
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.manual_seed_all(seed)
    return "cuda:0"


def memory_stats() -> dict[str, JsonValue]:
    return {
        "batch_size": 1,
        "allocator_limit_bytes": 6_000_000_000 if torch.cuda.is_available() else 0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0) if torch.cuda.is_available() else 0,
        "scope": "PyTorch allocator only; excludes CUDA context/library allocations",
    }
