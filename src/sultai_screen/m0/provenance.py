"""M0 environment provenance and the worker receipt envelope, without importing the Gate 3 screen."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import time
from typing import Final

import numpy as np
import torch

from sultai_screen.contracts import Json, PilotHold, Record
from sultai_screen.host import flags
from sultai_screen.m0.identity import source_identity

# The five pinned fields: a mismatch holds before custody (§7).
PINNED_ENVIRONMENT: Final[dict[str, str]] = {
    "torch_wheel": "2.9.1",
    "torch": "2.9.1+cu128",
    "cuda": "12.8",
    "numpy_wheel": "2.3.5",
    "numpy": "2.3.5",
}
# Amendment 5 (A5.2 item 5): the determinism-relevant variables every child records as it saw them.
ENVIRONMENT_VARIABLES: Final = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
    "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_CUDNN_V8_API_DISABLED",
)


def _driver_and_gpus() -> tuple[str | None, list[Json] | None]:
    """The NVIDIA driver version and the GPU names from nvidia-smi (no CUDA context); absent when it cannot answer."""
    try:
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    rows = [line.split(", ") for line in query.strip().splitlines() if line.strip()]
    if not rows or any(len(row) != 2 for row in rows):
        return None, None
    return rows[0][1].strip(), [row[0].strip() for row in rows]


def environment_identity() -> Record:
    """Admission checks distinct distribution, runtime and CUDA fields without GPU initialization; amendment 5
    (A5.2 item 1) records Python, driver, cuDNN and the GPU names beside them, never comparing those."""
    actual: Record = {
        "torch_wheel": importlib.metadata.version("torch"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "numpy_wheel": importlib.metadata.version("numpy"),
        "numpy": np.__version__,
    }
    for key, value in PINNED_ENVIRONMENT.items():
        if actual[key] != value:
            raise PilotHold(f"pinned environment mismatch: {key}: expected {value}, observed {actual[key]}")
    driver, gpu_names = _driver_and_gpus()
    cudnn: object = torch.backends.cudnn.version()
    actual.update(
        {
            "python": platform.python_version(),
            "driver": driver,
            "cudnn": cudnn if isinstance(cudnn, int) else None,
            "gpu_names": gpu_names,
        }
    )
    return actual


def provenance() -> Record:
    result = source_identity()
    result.update(environment_identity())
    result.update({"determinism": flags(), "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"]})
    result["environment_variables"] = {
        name: (os.environ[name] if name in os.environ else None) for name in ENVIRONMENT_VARIABLES
    }
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
