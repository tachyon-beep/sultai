"""M0 calibration arms: BatchNorm recalibration (R-bn), temperature (R-temp), class bias (R-cal) and the N0 rule."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sultai.trust import ContractViolation
from sultai_screen.contracts import PilotHold
from sultai_screen.data import Dataset, normalized
from sultai_screen.fit import mean_loss
from sultai_screen.host import Host, cached, freeze, load_host, state_hash

RECALIBRATION_CHUNK: Final = 128
LBFGS_ITERATIONS: Final = 50
NOOP_FILE: Final = "noop.pt"


def recalibrate_bn(host: Host, images: Tensor, indices: Tensor) -> Host:
    """R-bn: reset every BatchNorm and re-estimate cumulative statistics over the fit examples, then freeze.

    One no-grad train-mode pass in `indices` order, chunks of RECALIBRATION_CHUNK with the partial last chunk;
    momentum=None gives the equal-weight cumulative average of the chunk statistics. Parameters are untouched.
    """
    if host.sealed:
        raise PilotHold("BatchNorm recalibration requires an unsealed host; a sealed host would keep stale statistics")
    if indices.ndim != 1 or len(indices) == 0:
        raise ContractViolation("recalibration requires a nonempty one-dimensional index tensor")
    norms = [module for module in host.modules() if isinstance(module, nn.BatchNorm2d)]
    if not norms:
        raise ContractViolation("host has no BatchNorm layers to recalibrate")
    for module in norms:
        module.momentum = None
        module.reset_running_stats()
    host.train()
    with torch.no_grad():
        for start in range(0, len(indices), RECALIBRATION_CHUNK):
            ids = indices[start : start + RECALIBRATION_CHUNK].to(images.device)
            host(normalized(images[ids]))
    freeze(host)
    return host


def _checked(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2 or logits.shape[1] != 10 or len(logits) == 0 or labels.shape != (len(logits),):
        raise ContractViolation("calibration requires nonempty [N, 10] logits and [N] labels")
    if not bool(torch.isfinite(logits).all()):
        raise ContractViolation("calibration requires finite logits")
    if labels.dtype != torch.long or bool(((labels < 0) | (labels > 9)).any()):
        raise ContractViolation("calibration labels must be int64 classes 0..9")
    return logits.detach().to("cpu", torch.float64), labels.detach().to("cpu", torch.long)


def _fit(logits: Tensor, labels: Tensor, *, bias: bool) -> tuple[float, Tensor]:
    """LBFGS on CPU float64 of mean cross-entropy of logits / exp(log_t) + b, from log_t = 0 and b = 0."""
    values, targets = _checked(logits, labels)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    shift = torch.zeros(10, dtype=torch.float64, requires_grad=bias)
    parameters = [log_t, shift] if bias else [log_t]
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=LBFGS_ITERATIONS,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure() -> float:
        loss = F.cross_entropy(values / log_t.exp() + shift, targets)
        for parameter, gradient in zip(parameters, torch.autograd.grad(loss, parameters), strict=True):
            parameter.grad = gradient
        return float(loss.detach())

    # LBFGS.step is unannotated in torch; bind it to its documented signature. It runs the closure under enable_grad.
    step: Callable[[Callable[[], float]], object] = optimizer.step
    step(closure)
    t = float(log_t.detach().exp())
    fitted = shift.detach().clone()
    if not (math.isfinite(t) and t > 0 and bool(torch.isfinite(fitted).all())):
        raise PilotHold("nonfinite logit calibration")
    return t, fitted


def temperature(logits: Tensor, labels: Tensor) -> float:
    """R-temp: the single temperature t > 0 minimising cross-entropy of logits / t on the fit examples."""
    return _fit(logits, labels, bias=False)[0]


def class_bias(logits: Tensor, labels: Tensor) -> tuple[float, Tensor]:
    """R-cal: temperature and a 10-vector class bias (11 numbers) fitted jointly on the fit examples."""
    return _fit(logits, labels, bias=True)


def apply_calibration(logits: Tensor, t: float, bias: Tensor) -> Tensor:
    """logits / t + bias in the logits' dtype and device."""
    if bias.shape != (10,) or logits.ndim != 2 or logits.shape[1] != 10:
        raise ContractViolation("calibration applies a [10] bias to [N, 10] logits")
    return logits / t + bias.to(logits.device, logits.dtype)


@dataclass(frozen=True)
class NoOpChoice:
    """The N0 outcome: the chosen host, its label and both selection losses that decided it (amendment 5, A5.2)."""

    model: Host
    label: str
    raw_selection_loss: float
    recalibrated_selection_loss: float


def no_op(raw: Host, recalibrated: Host, data: Dataset) -> NoOpChoice:
    """N0: the lower mean selection loss of raw and R-bn, each on selection features recomputed from itself.

    R-bn changes the front's BatchNorm buffers, so the raw host's cached site is not the recalibrated host's.
    Only the selection split decides; ties keep the raw host.
    """
    raw_loss = mean_loss(raw, cached(raw, data).selection, None)
    recalibrated_loss = mean_loss(recalibrated, cached(recalibrated, data).selection, None)
    if recalibrated_loss < raw_loss:
        return NoOpChoice(recalibrated, "R-bn", raw_loss, recalibrated_loss)
    return NoOpChoice(raw, "raw", raw_loss, recalibrated_loss)


def save_noop(host: Host, output: Path) -> str:
    """Publish the N0 host as output/noop.pt; every later arm on the anchor loads it by the returned state hash."""
    torch.save(dict(host.state_dict()), output / NOOP_FILE)
    return state_hash(host)


def load_noop(
    root: Path,
    seed: int,
    widths: tuple[int, int, int, int],
    expected_hash: str,
    device: torch.device,
) -> Host:
    """Load the published N0 host frozen; refuse it unless its state hash is the one N0 recorded."""
    host = load_host(root / NOOP_FILE, seed, 0, device, widths=widths, frozen=True)
    if state_hash(host) != expected_hash:
        raise PilotHold("noop.pt state hash differs from the recorded N0 choice")
    return host
