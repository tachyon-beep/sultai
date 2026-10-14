"""M0 environment provenance and the worker receipt envelope, without importing the Gate 3 screen."""

from __future__ import annotations

import importlib.metadata
import os
import time

import numpy as np
import torch

from sultai_screen.contracts import PilotHold, Record
from sultai_screen.host import flags
from sultai_screen.m0.identity import source_identity


def environment_identity() -> Record:
    """Admission checks distinct distribution, runtime and CUDA fields without GPU initialization."""
    actual: Record = {
        "torch_wheel": importlib.metadata.version("torch"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "numpy_wheel": importlib.metadata.version("numpy"),
        "numpy": np.__version__,
    }
    expected: Record = {
        "torch_wheel": "2.9.1",
        "torch": "2.9.1+cu128",
        "cuda": "12.8",
        "numpy_wheel": "2.3.5",
        "numpy": "2.3.5",
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise PilotHold(f"pinned environment mismatch: {key}: expected {value}, observed {actual[key]}")
    return actual


def provenance() -> Record:
    result = source_identity()
    result.update(environment_identity())
    result.update({"determinism": flags(), "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"]})
    return result


def envelope(action: str, identity: Record, payload: Record, seal: Record, started: float, peak: int) -> Record:
    """The `result.json` receipt of a successful worker; the caller publishes it."""
    return {
        "action": action,
        "software_valid": True,
        "identity": identity,
        "payload": payload,
        "data_opens": seal,
        "wall_seconds": time.monotonic() - started,
        "peak_cuda_bytes": peak,
    }
