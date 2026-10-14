"""M0 S-static arm: the reference host trained with the 16x16 module present from step zero."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from sultai.trust import ContractViolation, InputDataError, t3_boundary
from sultai_screen.contracts import PilotHold, Record, file_sha
from sultai_screen.data import Dataset, as_tensor, normalized
from sultai_screen.fit import Adapter, mean_loss
from sultai_screen.host import (
    REFERENCE_WIDTHS,
    Features,
    Host,
    constant,
    freeze,
    load_host,
    make_host,
    recipe_optimizer,
    state_hash,
    train_epochs,
)
from sultai_screen.m0.calibrate import RECALIBRATION_CHUNK
from sultai_screen.m0.data import features_from_images

MODULE_SHAPES = {"w": (16, 16), "b": (16,)}


class StaticHost(nn.Module):
    """A host whose site carries w·tanh(z) + b; the module is zero-initialised after the host, with no RNG draw."""

    def __init__(self, host: Host) -> None:
        super().__init__()
        self.host = host
        device = next(host.parameters()).device
        self.w = nn.Parameter(torch.zeros(16, 16, device=device))
        self.b = nn.Parameter(torch.zeros(16, device=device))

    def forward(self, images: Tensor) -> Tensor:
        z = as_tensor(self.host.bottleneck(as_tensor(self.host.front(images))))
        # Exactly Adapter((w, b), nonlinear=True).apply(z, z), so the trained tensors are directly an Adapter.
        site = z + torch.einsum("ij,bjhw->bihw", self.w, torch.tanh(z)) + self.b[None, :, None, None]
        return self.host.classify(site)


def _seal(static: StaticHost) -> None:
    if not static.host.sealed:
        freeze(static.host)
    static.eval()
    for parameter in (static.w, static.b):
        parameter.requires_grad_(False)
        parameter.grad = None


def train_static(
    seed: int, data: Dataset, device: torch.device, widths: tuple[int, int, int, int] = REFERENCE_WIDTHS
) -> tuple[StaticHost, list[str], Tensor, float, str]:
    """The registered ten-epoch recipe with w in the weight-decay group and b in the affine group; returned sealed.

    The last element is the inner host's state hash at step zero, before any step (final-review M8).
    """
    host = make_host(seed, 0, device, widths=widths)
    initial = state_hash(host)
    static = StaticHost(host)
    hashes, stream, elapsed = train_epochs(
        static,
        static.host,
        data,
        seed,
        epochs=range(10),
        rates=constant(0.05),
        optimizer=recipe_optimizer(static.parameters()),
    )
    _seal(static)
    return static, hashes, stream, elapsed, initial


def static_adapter(static: StaticHost) -> Adapter:
    return Adapter((static.w.detach().clone(), static.b.detach().clone()), nonlinear=True)


def static_features(static: StaticHost, data: Dataset, indices: Tensor) -> Features:
    """Inner-host site features; evaluate the complete model as losses(static.host, features, static_adapter(static))."""
    return features_from_images(static.host, data, indices)


def recalibrate_static(static: StaticHost, images: Tensor, indices: Tensor) -> StaticHost:
    """Amendment 4 (A4.1): R-bn of the complete static model, the module in the forward, over the fit examples.

    The same cumulative pass as `calibrate.recalibrate_bn` (chunks of RECALIBRATION_CHUNK, momentum=None), run through
    `StaticHost.forward` so every statistic downstream of the site is the designed-in network's; parameters and the
    module are untouched; returned sealed. An already sealed model is refused as the plain rule refuses a sealed host.
    """
    if static.host.sealed:
        raise PilotHold(
            "static recalibration requires an unsealed static model; a sealed one would keep stale statistics"
        )
    if indices.ndim != 1 or len(indices) == 0:
        raise ContractViolation("recalibration requires a nonempty one-dimensional index tensor")
    norms = [module for module in static.host.modules() if isinstance(module, nn.BatchNorm2d)]
    if not norms:
        raise ContractViolation("static host has no BatchNorm layers to recalibrate")
    for module in norms:
        module.momentum = None
        module.reset_running_stats()
    static.train()
    with torch.no_grad():
        for start in range(0, len(indices), RECALIBRATION_CHUNK):
            ids = indices[start : start + RECALIBRATION_CHUNK].to(images.device)
            static(normalized(images[ids]))
    _seal(static)
    return static


@dataclass(frozen=True)
class StaticChoice:
    """The no-op rule's outcome on the complete static model: the chosen model, its label and both selection losses."""

    model: StaticHost
    label: str
    raw_selection_loss: float
    recalibrated_selection_loss: float


def static_no_op(raw: StaticHost, recalibrated: StaticHost, data: Dataset) -> StaticChoice:
    """Amendment 4 (A4.1): §4's no-op rule on the complete static model, each evaluated on selection features
    recomputed from itself with its own module; the lower mean selection loss wins and a tie keeps raw."""

    def selection(static: StaticHost) -> float:
        return mean_loss(static.host, static_features(static, data, data.selection), static_adapter(static))

    raw_loss, recalibrated_loss = selection(raw), selection(recalibrated)
    if recalibrated_loss < raw_loss:
        return StaticChoice(recalibrated, "R-bn", raw_loss, recalibrated_loss)
    return StaticChoice(raw, "raw", raw_loss, recalibrated_loss)


def static_hash(static: StaticHost) -> str:
    digest = hashlib.sha256()
    digest.update(f"host:{state_hash(static.host)}:".encode())
    for name, tensor in (("w", static.w), ("b", static.b)):
        digest.update(f"module:{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_static(static: StaticHost, output: Path) -> Record:
    torch.save(dict(static.host.state_dict()), output / "host.pt")
    torch.save(
        {"w": static.w.detach().cpu().clone(), "b": static.b.detach().cpu().clone()}, output / "static-module.pt"
    )
    return {
        "static_sha256": static_hash(static),
        "host_sha256": state_hash(static.host),
        "module_sha256": file_sha(output / "static-module.pt"),
    }


@t3_boundary(
    test="tests/test_m0_retune.py::StaticTests.test_static_module_boundary_rejects_missing_keys_wrong_shape_and_nonfinite",
    fingerprint="7b0c50c846f10fe5afdb27d02139c8a84b62e673090499d929f1890a974c4c73",
)
def static_module(value: object) -> dict[str, Tensor]:
    if not isinstance(value, dict) or set(value) != set(MODULE_SHAPES):
        raise InputDataError("static module must contain exactly w and b")
    result: dict[str, Tensor] = {}
    for key, shape in MODULE_SHAPES.items():
        item: object = value[key]
        if not isinstance(item, Tensor) or item.dtype != torch.float32 or tuple(item.shape) != shape:
            raise InputDataError("static module requires float32 w [16,16] and b [16]")
        if not bool(torch.isfinite(item).all()):
            raise InputDataError("static module tensors must be finite")
        result[key] = item
    return result


def load_static(
    root: Path,
    seed: int,
    widths: tuple[int, int, int, int],
    device: torch.device,
    expected: str,
    *,
    frozen: bool = True,
) -> StaticHost:
    """The published static model by its recorded identity; `frozen=False` returns it unsealed for recalibration."""
    host = load_host(root / "host.pt", seed, 0, device, widths=widths, frozen=frozen)
    unknown: object = torch.load(root / "static-module.pt", map_location=device, weights_only=True)
    module = static_module(unknown)
    static = StaticHost(host)
    with torch.no_grad():
        static.w.copy_(module["w"])
        static.b.copy_(module["b"])
    if frozen:
        _seal(static)
    if static_hash(static) != expected:
        raise PilotHold("static host or module differs from its recorded identity")
    return static
