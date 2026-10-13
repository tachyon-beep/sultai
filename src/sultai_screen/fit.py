"""Conventional ridge and first-order fits, selection-only candidate choice, and plateau evidence."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from sultai.trust import ContractViolation, FitUnavailable, t2_operation

from .contracts import CHEAP_RATES, ORACLE_RATES, OracleUnavailable, PilotHold, Record
from .data import as_tensor, generator
from .host import Features, Host, assert_frozen, state_hash


@dataclass(frozen=True)
class Adapter:
    weights: tuple[Tensor, ...]
    nonlinear: bool = True

    def apply(self, read: Tensor, base: Tensor) -> Tensor:
        if len(self.weights) == 2:
            w, b = self.weights
            features = torch.tanh(read) if self.nonlinear else read
            return base + torch.einsum("ij,bjhw->bihw", w, features) + b[None, :, None, None]
        if len(self.weights) == 4:
            inside, bias_in, outside, bias_out = self.weights
            hidden = torch.tanh(torch.einsum("ij,bjhw->bihw", inside, read) + bias_in[None, :, None, None])
            return base + torch.einsum("ij,bjhw->bihw", outside, hidden) + bias_out[None, :, None, None]
        raise ContractViolation("adapter must have two or four parameter tensors")

    def detached(self) -> Adapter:
        return Adapter(tuple(w.detach().clone() for w in self.weights), self.nonlinear)


EVAL_CHUNK = 128


def zero_adapter(device: torch.device, *, dtype: torch.dtype = torch.float32) -> Adapter:
    return Adapter((torch.zeros((16, 16), device=device, dtype=dtype), torch.zeros(16, device=device, dtype=dtype)))


def _padded(values: Tensor) -> Tensor:
    """v4.1 §3: pad a partial chunk to EVAL_CHUNK rows by repeating its last real row; same dtype/device/layout."""
    missing = EVAL_CHUNK - len(values)
    if missing <= 0:
        return values
    return torch.cat((values, values[-1:].expand(missing, *values.shape[1:])))


def losses(host: Host, data: Features, adapter: Adapter | None, *, check_finite: bool = True) -> Tensor:
    """Shared frozen evaluator: every chunk has exactly EVAL_CHUNK rows; exactly N losses in caller order."""
    count = len(data.labels)
    if count == 0 or len(data.read) != count or len(data.base) != count:
        raise ContractViolation("nonempty aligned read/base/labels required")
    expected = state_hash(host)
    assert_frozen(host, expected)
    result = []
    with torch.no_grad():
        for start in range(0, count, EVAL_CHUNK):
            stop = min(start + EVAL_CHUNK, count)
            read, base = _padded(data.read[start:stop]), _padded(data.base[start:stop])
            labels = _padded(data.labels[start:stop])
            values = base if adapter is None else adapter.apply(read, base)
            chunk = F.cross_entropy(host.classify(values), labels, reduction="none")
            result.append(chunk[: stop - start])
    values = torch.cat(result)
    assert_frozen(host, expected)
    if check_finite and not bool(torch.isfinite(values).all()):
        raise PilotHold("nonfinite no-op or selected-candidate evaluation")
    return values


def logits(host: Host, data: Features, adapter: Adapter | None) -> Tensor:
    """Sibling of losses(): the same padded EVAL_CHUNK loop, returning [N, 10] logits in caller order."""
    count = len(data.labels)
    if count == 0 or len(data.read) != count or len(data.base) != count:
        raise ContractViolation("nonempty aligned read/base/labels required")
    expected = state_hash(host)
    assert_frozen(host, expected)
    result = []
    with torch.no_grad():
        for start in range(0, count, EVAL_CHUNK):
            stop = min(start + EVAL_CHUNK, count)
            read, base = _padded(data.read[start:stop]), _padded(data.base[start:stop])
            values = base if adapter is None else adapter.apply(read, base)
            result.append(host.classify(values)[: stop - start])
    values = torch.cat(result)
    assert_frozen(host, expected)
    return values


def accuracy(host: Host, data: Features, adapter: Adapter | None) -> Tensor:
    """Per-example 0/1 correctness [N] through the padded frozen evaluator."""
    return (logits(host, data, adapter).argmax(1) == data.labels).float()


def mean_accuracy(host: Host, data: Features, adapter: Adapter | None) -> float:
    return float(accuracy(host, data, adapter).double().mean())


def mean_loss(host: Host, data: Features, adapter: Adapter | None) -> float:
    return float(losses(host, data, adapter).double().mean())


def identity_check(host: Host, data: Features, seed: int) -> None:
    baseline = losses(host, data, None)
    if not torch.equal(baseline, losses(host, data, zero_adapter(data.read.device))):
        raise PilotHold("zero module is not bitwise identity")
    order = torch.randperm(len(data.labels), generator=generator(seed, "noop-evaluation-twin")).to(data.read.device)
    twin = Features(data.read[order], data.base[order], data.labels[order])
    if not torch.equal(baseline, losses(host, twin, None)[torch.argsort(order)]):
        raise PilotHold("resampled no-op evaluation twin is not bitwise")


@t2_operation(
    invariants="Owned finite matched NCHW site pairs with 16 channels; positive ridge 1e-8; a singular numeric solve is recoverable.",
    failures=(FitUnavailable,),
)
def ridge(read: Tensor, base: Tensor, target: Tensor, *, nonlinear: bool) -> Adapter:
    if read.shape != base.shape or target.shape != base.shape or read.ndim != 4 or read.shape[1] != 16:
        raise ContractViolation("matched [N,16,H,W] pairs required")
    if not all(bool(torch.isfinite(t).all()) for t in (read, base, target)):
        raise ContractViolation("finite owned pairs required")
    features = (read.tanh() if nonlinear else read).permute(0, 2, 3, 1).reshape(-1, 16).double()
    residual = (target - base).permute(0, 2, 3, 1).reshape(-1, 16).double()
    x = torch.cat((features, torch.ones((len(features), 1), device=read.device, dtype=torch.float64)), dim=1)
    gram = x.T @ x + torch.eye(17, device=read.device, dtype=torch.float64) * 1e-8
    solved, info = torch.linalg.solve_ex(gram, x.T @ residual, check_errors=False)
    solution = as_tensor(solved)
    if bool(as_tensor(info) != 0):
        raise FitUnavailable("ridge solve unavailable at machine precision")
    if not bool(torch.isfinite(solution).all()):
        raise FitUnavailable("ridge solve produced nonfinite coefficients")
    return Adapter((solution[:16].T.to(read.dtype), solution[16].to(read.dtype)), nonlinear)


@dataclass(frozen=True)
class Candidate:
    adapter: Adapter
    rate: float
    seed: int
    steps: int
    selection: float | None
    training_losses: tuple[float, ...]
    wall_seconds: float
    example_passes: int
    parameter_steps: int

    def report(self) -> Record:
        return {
            "rate": self.rate,
            "seed": self.seed,
            "steps": self.steps,
            "selection_loss": self.selection,
            "finite": self.selection is not None,
            "training_losses": list(self.training_losses),
            "wall_seconds": self.wall_seconds,
            "example_passes": self.example_passes,
            "parameter_steps": self.parameter_steps,
        }


def gradient_step(host: Host, data: Features, adapter: Adapter, rate: float, indices: Tensor) -> tuple[Adapter, Tensor]:
    weights = tuple(w.detach().requires_grad_(True) for w in adapter.weights)
    current = Adapter(weights, adapter.nonlinear)
    totals = [torch.zeros_like(w) for w in weights]
    total_loss = torch.zeros((), device=data.read.device)
    # Chunking is an exact mean-gradient accumulation, never eight SGD updates.
    # It bounds autograd activation memory for the 2,000-example full-batch step.
    for start in range(0, len(indices), 128):
        ids = indices[start : start + 128]
        logits = host.classify(current.apply(data.read[ids], data.base[ids]))
        loss = F.cross_entropy(logits, data.labels[ids], reduction="sum") / len(indices)
        gradients = torch.autograd.grad(loss, weights)
        totals = [a + b for a, b in zip(totals, gradients, strict=True)]
        total_loss = total_loss + loss.detach()
    return Adapter(
        tuple((w - rate * grad).detach() for w, grad in zip(weights, totals, strict=True)), adapter.nonlinear
    ), total_loss


def fit_sgd(
    host: Host,
    conditioning: Features,
    selection: Features,
    *,
    seed: int,
    rate: float,
    steps: int,
    minibatch: bool,
    wide: bool = False,
) -> Candidate:
    if rate <= 0 or not math.isfinite(rate) or steps <= 0 or not len(conditioning.labels):
        raise ContractViolation("positive finite rate, positive steps and nonempty conditioning required")
    expected = state_hash(host)
    assert_frozen(host, expected)
    device = conditioning.read.device
    rng = generator(seed, "oracle-init")
    if wide:
        shapes = ((64, 16), (64,), (16, 64), (16,))
        adapter = Adapter(tuple((torch.randn(shape, generator=rng) * 1e-3).to(device) for shape in shapes))
    elif minibatch:
        adapter = Adapter(tuple((torch.randn(shape, generator=rng) * 1e-3).to(device) for shape in ((16, 16), (16,))))
    else:
        adapter = zero_adapter(device)
    n = len(conditioning.labels)
    order = torch.arange(n, device=device)
    draws = generator(seed, "oracle-order")
    traces: list[Tensor] = []
    examples = 0
    start = time.monotonic()
    for step in range(steps):
        if minibatch:
            if step % math.ceil(n / 256) == 0:
                order = torch.randperm(n, generator=draws).to(device)
            offset = (step % math.ceil(n / 256)) * 256
            indices = order[offset : offset + 256]
        else:
            indices = order
        adapter, loss = gradient_step(host, conditioning, adapter, rate, indices)
        examples += len(indices)
        traces.append(loss)
        # A candidate failure is recoverable and cannot be selected; no anchor disappears.
        if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(w).all()) for w in adapter.weights):
            assert_frozen(host, expected)
            return Candidate(
                adapter,
                rate,
                seed,
                steps,
                None,
                (),
                time.monotonic() - start,
                examples,
                sum(w.numel() for w in adapter.weights) * (step + 1),
            )
    values = losses(host, selection, adapter, check_finite=False)
    selected = float(values.double().mean()) if bool(torch.isfinite(values).all()) else None
    history = tuple(float(v) for v in torch.stack(traces).cpu())
    assert_frozen(host, expected)
    return Candidate(
        adapter.detached(),
        rate,
        seed,
        steps,
        selected,
        history,
        time.monotonic() - start,
        examples,
        sum(w.numel() for w in adapter.weights) * steps,
    )


@t2_operation(
    invariants="Complete owned candidate bank; candidates explicitly mark numerical failures; at least one finite candidate is required for selection.",
    failures=(OracleUnavailable,),
)
def choose(candidates: tuple[Candidate, ...]) -> Candidate:
    if not candidates:
        raise ContractViolation("owned candidate bank must be nonempty")
    finite = [c for c in candidates if c.selection is not None]
    if not finite:
        raise OracleUnavailable(
            "all candidates nonfinite; anchor retained, arm unavailable",
            candidates=tuple(c.report() for c in candidates),
        )
    return min(finite, key=lambda c: math.inf if c.selection is None else c.selection)


def paired_se(difference: Tensor, seed: int, *, samples: int = 10000) -> float:
    if difference.ndim != 1 or len(difference) < 2 or not bool(torch.isfinite(difference).all()):
        raise ContractViolation("finite paired per-example differences required")
    values = difference.detach().double().cpu()
    rng = generator(seed, "paired-bootstrap")
    means: list[Tensor] = []
    for _ in range(0, samples, 100):
        indices = torch.randint(len(values), (min(100, samples - len(means)), len(values)), generator=rng)
        means.extend(values[indices].mean(dim=1).unbind())
    return float(torch.stack(means).std(unbiased=True))


@dataclass(frozen=True)
class Oracle:
    best: Candidate
    candidates: tuple[Candidate, ...]
    plateau: bool
    selection_se: float | None
    loss_tail_se: float | None
    plateau_evidence: str

    def report(self) -> Record:
        return {
            "best": self.best.report(),
            "candidates": [c.report() for c in self.candidates],
            "plateau_certified": self.plateau,
            "selection_paired_bootstrap_se": self.selection_se,
            "training_tail_se": self.loss_tail_se,
            "plateau_evidence": self.plateau_evidence,
        }


def oracle(host: Host, conditioning: Features, selection: Features, seed: int, *, seeds: int = 3) -> Oracle:
    if seeds not in (1, 2, 3):
        raise ContractViolation("oracle seed count must be a timing-only pilot decision")
    candidates = tuple(
        fit_sgd(host, conditioning, selection, seed=derive_oracle(seed, s), rate=rate, steps=steps, minibatch=True)
        for steps in (200, 800)
        for s in range(seeds)
        for rate in ORACLE_RATES
    )
    best = choose(candidates)
    finite1 = tuple(c for c in candidates if c.steps == 200 and c.selection is not None)
    finite4 = tuple(c for c in candidates if c.steps == 800 and c.selection is not None)
    if not finite1 or not finite4:
        return Oracle(
            best, candidates, False, None, None, "schedule comparator unavailable; finite overall winner retained"
        )
    best1, best4 = choose(finite1), choose(finite4)
    difference = losses(host, selection, best1.adapter) - losses(host, selection, best4.adapter)
    se = paired_se(difference, seed)
    tail = torch.tensor(best4.training_losses[720:800], dtype=torch.float64)
    tail_se = float(tail.std(unbiased=True) / math.sqrt(len(tail)))
    change = float((tail[:40].mean() - tail[40:].mean()).abs())
    if best1.selection is None or best4.selection is None:
        raise ContractViolation("selected candidates must be finite")
    plateau = best1.selection - best4.selection < 2 * se and change < 2 * tail_se
    return Oracle(best, candidates, plateau, se, tail_se, "both finite schedule comparators available")


def derive_oracle(seed: int, index: int) -> int:
    from .contracts import derive

    return derive(seed, f"oracle:{index}")


def cheap_bank(
    host: Host, conditioning: Features, selection: Features, seed: int, *, steps: int = 4
) -> tuple[Candidate, ...]:
    return tuple(
        fit_sgd(host, conditioning, selection, seed=seed, rate=rate, steps=steps, minibatch=False)
        for rate in CHEAP_RATES
    )
