"""M0 re-tune arms: SGD on declared host-parameter subsets, complement hashing, the re-tune grid and post-fit R-bn."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
import time
from dataclasses import dataclass
from typing import Final, TypeAlias

import torch
from torch import Tensor
from torch.nn import functional as F

from sultai.trust import ContractViolation
from sultai_screen.contracts import OracleUnavailable, Record
from sultai_screen.data import Dataset, as_tensor, generator, normalized
from sultai_screen.fit import _passes, at_edge, losses
from sultai_screen.host import Host, assert_frozen, freeze, make_host, state_hash
from sultai_screen.m0.calibrate import recalibrate_bn
from sultai_screen.m0.data import features_from_images
from sultai_screen.m0.identity import CHECKPOINTS, RETUNE_RATES

Subset: TypeAlias = dict[str, Tensor]
BATCH: Final = 256
# Parameters downstream of the cached site; only these may be trained on cached site features.
SITE_DOWNSTREAM: Final = ("tail.", "classifier.")


def _mask(host: Host, name: str) -> Tensor:
    return torch.zeros_like(dict(host.named_parameters())[name], dtype=torch.bool)


def R272S(host: Host) -> Subset:
    """Centre taps of output channels 0-15 of tail[0] over its 16 input channels (256) and their 16 biases."""
    weight, bias = _mask(host, "tail.0.weight"), _mask(host, "tail.0.bias")
    weight[0:16, :, 1, 1] = True
    bias[0:16] = True
    return {"tail.0.weight": weight, "tail.0.bias": bias}


def RAFF(host: Host) -> Subset:
    """Producer BatchNorm affine (front.8), first downstream BatchNorm affine (tail.1) and the classifier bias."""
    names = ("front.8.weight", "front.8.bias", "tail.1.weight", "tail.1.bias", "classifier.bias")
    return {name: _mask(host, name).logical_not() for name in names}


def RALL(host: Host) -> Subset:
    """Every host parameter."""
    return {name: _mask(host, name).logical_not() for name, _ in host.named_parameters()}


def count(subset: Subset) -> int:
    """Number of declared entries; a subset covering nothing is refused."""
    if any(mask.dtype != torch.bool for mask in subset.values()):
        raise ContractViolation("subset masks must be boolean")
    total = sum(int(mask.sum()) for mask in subset.values())
    if total == 0:
        raise ContractViolation("a re-tune subset must cover at least one parameter entry")
    return total


def _checked(host: Host, subset: Subset) -> int:
    named = dict(host.named_parameters())
    for name, mask in subset.items():
        if name not in named:
            raise ContractViolation(f"subset names an unknown parameter: {name}")
        if mask.dtype != torch.bool or mask.shape != named[name].shape:
            raise ContractViolation(f"subset mask for {name} must be boolean with the parameter's shape")
    return count(subset)


def complement_hash(host: Host, subset: Subset) -> str:
    """state_hash with the subset's entries zeroed: invariant inside the subset, sensitive to everything else."""
    _checked(host, subset)
    digest = hashlib.sha256()
    for kind, pairs in (("parameter", host.named_parameters()), ("buffer", host.named_buffers())):
        for name, tensor in pairs:
            value = tensor.detach().cpu()
            if kind == "parameter" and name in subset:
                value = torch.where(subset[name].cpu(), torch.zeros_like(value), value)
            digest.update(f"{kind}:{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _copy(host: Host, seed: int, *, frozen: bool) -> Host:
    """A fresh host of the same width and tail widths carrying host's state, in evaluation mode."""
    device = next(host.parameters()).device
    # The init seed is irrelevant: load_state_dict overwrites every parameter and buffer.
    clone = make_host(seed, host.width, device, widths=host.widths)
    clone.load_state_dict(host.state_dict(), strict=True)
    clone.eval()
    clone.requires_grad_(False)
    if frozen:
        freeze(clone)
    return clone


def _selection(host: Host, data: Dataset, indices: Tensor) -> float | None:
    """Mean selection loss of a frozen host on features recomputed from its own front; None if nonfinite."""
    values = losses(host, features_from_images(host, data, indices), None, check_finite=False)
    return float(values.double().mean()) if bool(torch.isfinite(values).all()) else None


@dataclass(frozen=True)
class RetuneCandidate:
    host: Host
    rate: float
    seed: int
    steps: int
    selection: float | None
    training_losses: tuple[float, ...]
    wall_seconds: float
    example_passes: int
    parameter_steps: int
    snapshots: tuple[tuple[int, str, float | None], ...]
    complement: str
    # Frozen hosts aligned with snapshots, so a grid can select an intermediate checkpoint.
    snapshot_hosts: tuple[Host, ...] = ()
    pre_recalibration_selection: float | None = None
    pre_recalibration_hash: str | None = None

    def report(self) -> Record:
        result: Record = {
            "rate": self.rate,
            "seed": self.seed,
            "steps": self.steps,
            "selection_loss": self.selection,
            "finite": self.selection is not None,
            "training_losses": list(self.training_losses),
            "wall_seconds": self.wall_seconds,
            "example_passes": self.example_passes,
            "parameter_steps": self.parameter_steps,
            "state_sha256": state_hash(self.host),
            "complement_sha256": self.complement,
            "snapshots": [
                {"step": step, "state_sha256": digest, "selection_loss": loss} for step, digest, loss in self.snapshots
            ],
        }
        if self.pre_recalibration_hash is not None:
            result["pre_recalibration_selection_loss"] = self.pre_recalibration_selection
            result["pre_recalibration_state_sha256"] = self.pre_recalibration_hash
        return result


def fit_retune(
    host: Host,
    subset: Subset,
    *,
    data: Dataset,
    fit_indices: Tensor,
    selection_indices: Tensor,
    seed: int,
    rate: float,
    steps: int,
    checkpoints: tuple[int, ...],
    from_images: bool,
) -> RetuneCandidate:
    """Plain SGD on the subset's entries of a clone of the frozen host, BatchNorm statistics frozen (eval mode).

    The minibatch stream is fit_sgd's (oracle-order permutation each pass, batch 256). from_images=False trains
    on the site cached once from the host (legitimate only for entries downstream of the site); True re-runs
    the whole host on normalised, unaugmented images. Each checkpoint snapshot is a frozen copy whose selection
    loss is computed on features recomputed from that copy. A nonfinite loss or weight ends the run with
    selection None (retained, never selected).
    """
    if rate <= 0 or not math.isfinite(rate) or steps <= 0:
        raise ContractViolation("positive finite rate and positive steps required")
    if any(c <= 0 for c in checkpoints) or any(a >= b for a, b in itertools.pairwise(checkpoints)):
        raise ContractViolation("checkpoints must be positive and strictly increasing")
    if checkpoints and checkpoints[-1] > steps:
        raise ContractViolation("checkpoints cannot exceed the run length")
    for indices in (fit_indices, selection_indices):
        if indices.ndim != 1 or len(indices) == 0:
            raise ContractViolation("fit and selection indices must be nonempty and one-dimensional")
    entries = _checked(host, subset)
    if not from_images and not all(name.startswith(SITE_DOWNSTREAM) for name in subset):
        raise ContractViolation("cached-site training requires every subset parameter downstream of the site")
    expected = state_hash(host)
    assert_frozen(host, expected)
    began = time.monotonic()
    device = data.images.device
    clone = _copy(host, seed, frozen=False)
    named = dict(clone.named_parameters())
    trained = tuple(named[name] for name in subset)
    masks = tuple(subset[name].to(device) for name in subset)
    for parameter in trained:
        parameter.requires_grad_(True)
    fit_ids = fit_indices.to(device)
    site = None if from_images else features_from_images(host, data, fit_indices).base
    n = len(fit_indices)
    chunks = math.ceil(n / BATCH)
    order = torch.arange(n, device=device)
    draws = generator(seed, "oracle-order")
    traces: list[Tensor] = []
    snapshots: list[tuple[int, str, float | None]] = []
    snapshot_hosts: list[Host] = []
    examples = 0
    for step in range(steps):
        if step % chunks == 0:
            order = torch.randperm(n, generator=draws).to(device)
        offset = (step % chunks) * BATCH
        positions = order[offset : offset + BATCH]
        ids = fit_ids[positions]
        logits = as_tensor(clone(normalized(data.images[ids]))) if site is None else clone.classify(site[positions])
        loss = F.cross_entropy(logits, data.labels[ids])
        gradients = torch.autograd.grad(loss, trained)
        with torch.no_grad():
            for parameter, mask, gradient in zip(trained, masks, gradients, strict=True):
                # The mask selects the update itself: a float32 product rate * 0 is NaN once rate overflows
                # float32 (and inf * 0 is NaN), so multiplying by the mask would leak into the complement.
                parameter.copy_(torch.where(mask, parameter - rate * gradient, parameter))
        examples += len(positions)
        traces.append(loss.detach())
        if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(p).all()) for p in trained):
            freeze(clone)
            assert_frozen(host, expected)
            return RetuneCandidate(
                clone,
                rate,
                seed,
                steps,
                None,
                (),
                time.monotonic() - began,
                examples,
                entries * (step + 1),
                tuple(snapshots),
                complement_hash(clone, subset),
                tuple(snapshot_hosts),
            )
        if step + 1 in checkpoints:
            snapshot = _copy(clone, seed, frozen=True)
            snapshots.append((step + 1, state_hash(snapshot), _selection(snapshot, data, selection_indices)))
            snapshot_hosts.append(snapshot)
    if clone.training:
        raise ContractViolation("re-tune clone left evaluation mode")
    freeze(clone)
    selected = (
        snapshots[-1][2] if snapshots and snapshots[-1][0] == steps else _selection(clone, data, selection_indices)
    )
    history = tuple(float(v) for v in torch.stack(traces).cpu())
    assert_frozen(host, expected)
    return RetuneCandidate(
        clone,
        rate,
        seed,
        steps,
        selected,
        history,
        time.monotonic() - began,
        examples,
        entries * steps,
        tuple(snapshots),
        complement_hash(clone, subset),
        tuple(snapshot_hosts),
    )


@dataclass(frozen=True)
class RetuneGrid:
    best: RetuneCandidate
    candidates: tuple[RetuneCandidate, ...]
    edge_of_grid: bool
    plateau: bool | None
    plateau_evidence: str

    def report(self) -> Record:
        return {
            "best": self.best.report(),
            "candidates": [c.report() for c in self.candidates],
            "edge_of_grid": self.edge_of_grid,
            "plateau_certified": self.plateau,
            "plateau_evidence": self.plateau_evidence,
        }


def _choose(candidates: tuple[RetuneCandidate, ...]) -> RetuneCandidate:
    finite = [c for c in candidates if c.selection is not None]
    if not finite:
        raise OracleUnavailable(
            "all re-tune candidates nonfinite; anchor retained, arm unavailable",
            candidates=tuple(c.report() for c in candidates),
        )
    return min(finite, key=lambda c: math.inf if c.selection is None else c.selection)


def retune_grid(
    host: Host,
    subset: Subset,
    *,
    data: Dataset,
    fit_indices: Tensor,
    selection_indices: Tensor,
    seed: int,
    rates: tuple[float, ...] = RETUNE_RATES,
    checkpoints: tuple[int, ...] = CHECKPOINTS,
    from_images: bool,
) -> RetuneGrid:
    """One run per rate to checkpoints[-1]; every (rate, checkpoint) snapshot is a candidate, best on selection.

    Bookkeeping follows fit.grid: each run's wall time is booked once, on its final-checkpoint candidate; a
    checkpoint a failed run never reached is a nonfinite candidate carrying what the run spent. The plateau rule
    needs per-example traces the snapshot hashes do not carry, so it is not applied to re-tune arms.
    """
    if not rates or not checkpoints:
        raise ContractViolation("nonempty rate set and checkpoints required")
    n = len(fit_indices)
    entries = count(subset)
    candidates: list[RetuneCandidate] = []
    for rate in rates:
        run = fit_retune(
            host,
            subset,
            data=data,
            fit_indices=fit_indices,
            selection_indices=selection_indices,
            seed=seed,
            rate=rate,
            steps=checkpoints[-1],
            checkpoints=checkpoints,
            from_images=from_images,
        )
        pairs = zip(run.snapshots, run.snapshot_hosts, strict=True)
        reached = {snapshot[0]: (snapshot, state) for snapshot, state in pairs}
        for step in checkpoints:
            wall = run.wall_seconds if step == checkpoints[-1] else 0.0
            if step in reached:
                (_, digest, loss), state = reached[step]
                candidate = RetuneCandidate(
                    state,
                    rate,
                    seed,
                    step,
                    loss,
                    run.training_losses[:step] if run.training_losses else (),
                    wall,
                    _passes(n, step, True),
                    entries * step,
                    ((step, digest, loss),),
                    complement_hash(state, subset),
                    (state,),
                )
            else:
                candidate = dataclasses.replace(run, steps=step, selection=None, wall_seconds=wall)
            candidates.append(candidate)
    bank = tuple(candidates)
    best = _choose(bank)
    return RetuneGrid(best, bank, at_edge(best.rate, rates), None, "not applied to re-tune arms")


def recalibrate_after(
    candidate: RetuneCandidate, data: Dataset, fit_indices: Tensor, selection_indices: Tensor
) -> RetuneCandidate:
    """R-all's post-fit R-bn (spec §5): recalibrate an unsealed copy of the candidate's host on the fit examples.

    The returned candidate carries the recalibrated frozen host and its selection loss, and records the
    pre-recalibration selection loss and state hash. Its complement hash stays the pre-recalibration one: the
    complement invariant is checked before recalibration, never across it.
    """
    if candidate.selection is None:
        raise ContractViolation("only a finite re-tune candidate is recalibrated")
    previous = state_hash(candidate.host)
    assert_frozen(candidate.host, previous)
    clone = _copy(candidate.host, candidate.seed, frozen=False)
    recalibrate_bn(clone, data.images, fit_indices)
    assert_frozen(candidate.host, previous)
    return dataclasses.replace(
        candidate,
        host=clone,
        selection=_selection(clone, data, selection_indices),
        pre_recalibration_selection=candidate.selection,
        pre_recalibration_hash=previous,
    )
