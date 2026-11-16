# /// script
# requires-python = ">=3.11"
# dependencies = ["trustmark==0.9.2", "torch", "torchvision", "numpy", "pillow"]
# ///
# How to run: imported by run_exp2.py with ../.venv/bin/python.
"""Reuse the public baseline's checksum loader and decode measurements."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_experiment import ExistingModelsTrustMark, PAYLOAD, SEED, decode_record

__all__ = ["ExistingModelsTrustMark", "PAYLOAD", "SEED", "decode_record"]
