"""Conventional ridge and first-order fits, selection-only candidate choice, and plateau evidence."""

from __future__ import annotations

import itertools
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
    """Two tensors: (w, b) on tanh(read) or, affine, on read; three: M-exp (w1, w2, b); four: hidden module."""

    weights: tuple[Tensor, ...]
    nonlinear: bool = True

    def apply(self, read: Tensor, base: Tensor) -> Tensor:
        if len(self.weights) == 2:
            w, b = self.weights
            features = torch.tanh(read) if self.nonlinear else read
            return base + torch.einsum("ij,bjhw->bihw", w, features) + b[None, :, None, None]
        if not self.nonlinear and len(self.weights) in (3, 4):
            # E6: adapter_shapes refuses affine M-exp and hidden modules; a hand-built one never applies tanh silently.
            raise ContractViolation("the affine path exists only for the two-tensor adapter")
        if len(self.weights) == 3:
            # base + (w1 @ w2) tanh(read) + b, without forming the 16x16 product.
            w1, w2, b = self.weights
            inner = torch.einsum("ij,bjhw->bihw", w2, torch.tanh(read))
            return base + torch.einsum("ij,bjhw->bihw", w1, inner) + b[None, :, None, None]
        if len(self.weights) == 4:
            inside, bias_in, outside, bias_out = self.weights
            hidden = torch.tanh(torch.einsum("ij,bjhw->bihw", inside, read) + bias_in[None, :, None, None])
            return base + torch.einsum("ij,bjhw->bihw", outside, hidden) + bias_out[None, :, None, None]
        raise ContractViolation("adapter must have two, three or four parameter tensors")

    def detached(self) -> Adapter:
        return Adapter(tuple(w.detach().clone() for w in self.weights), self.nonlinear)

    def count(self) -> int:
        return sum(w.numel() for w in self.weights)


EVAL_CHUNK = 128
EXPANDED_WIDTH = 64
PLATEAU_WINDOW = 40


def zero_adapter(device: torch.device, *, dtype: torch.dtype = torch.float32, nonlinear: bool = True) -> Adapter:
    return Adapter(
        (torch.zeros((16, 16), device=device, dtype=dtype), torch.zeros(16, device=device, dtype=dtype)), nonlinear
    )


def adapter_shapes(*, hidden: int | None, nonlinear: bool, expanded: bool) -> tuple[tuple[int, ...], ...]:
    """Fixed tensor order of each adapter kind; the oracle-init draw order follows it."""
    if hidden is not None and (type(hidden) is not int or hidden <= 0):
        raise ContractViolation("hidden width must be a positive integer")
    if expanded and hidden is not None:
        raise ContractViolation("expanded and hidden-width adapters are exclusive")
    if not nonlinear and (expanded or hidden is not None):
        raise ContractViolation("the affine path exists only for the two-tensor adapter")
    if expanded:
        return ((16, EXPANDED_WIDTH), (EXPANDED_WIDTH, 16), (16,))
    if hidden is not None:
        return ((hidden, 16), (hidden,), (16, hidden), (16,))
    return ((16, 16), (16,))


def initial_adapter(seed: int, *, hidden: int | None, nonlinear: bool, expanded: bool, device: torch.device) -> Adapter:
    shapes = adapter_shapes(hidden=hidden, nonlinear=nonlinear, expanded=expanded)
    rng = generator(seed, "oracle-init")
    return Adapter(tuple((torch.randn(shape, generator=rng) * 1e-3).to(device) for shape in shapes), nonlinear)


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
    snapshots: tuple[tuple[int, Adapter, float | None], ...] = ()
    step0_selection: float | None = None
    # E2 (ruling R33): a grid checkpoint a run reached before it diverged keeps its weights and selection loss but no
    # trace (fit_sgd reports none for a failed run, Gate 3's contract).
    trace_truncated: bool = False

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
        }
        # M0-only keys appear only when set, so Gate 3 candidate receipts keep their bytes.
        if self.snapshots:
            result["snapshots"] = [{"step": step, "selection_loss": loss} for step, _, loss in self.snapshots]
        if self.step0_selection is not None:
            result["step0_selection_loss"] = self.step0_selection
        if self.trace_truncated:
            result["trace_truncated"] = True
        return result


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


def _selection(host: Host, selection: Features, adapter: Adapter) -> float | None:
    values = losses(host, selection, adapter, check_finite=False)
    return float(values.double().mean()) if bool(torch.isfinite(values).all()) else None


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
    start: Adapter | None = None,
    hidden: int | None = None,
    nonlinear: bool = True,
    expanded: bool = False,
    checkpoints: tuple[int, ...] = (),
) -> Candidate:
    """One SGD run; a snapshot is the detached adapter after the step whose 1-based count is a checkpoint.

    The order stream and the selection evaluator are unaffected by checkpoints, so each snapshot equals a
    separate run of that length bitwise.
    """
    if rate <= 0 or not math.isfinite(rate) or steps <= 0 or not len(conditioning.labels):
        raise ContractViolation("positive finite rate, positive steps and nonempty conditioning required")
    if any(c <= 0 for c in checkpoints) or any(a >= b for a, b in itertools.pairwise(checkpoints)):
        raise ContractViolation("checkpoints must be positive and strictly increasing")
    if checkpoints and checkpoints[-1] > steps:
        raise ContractViolation("checkpoints cannot exceed the run length")
    if wide:
        if hidden not in (None, 64):
            raise ContractViolation("wide means hidden width 64")
        hidden = 64
    shapes = adapter_shapes(hidden=hidden, nonlinear=nonlinear, expanded=expanded)
    expected = state_hash(host)
    assert_frozen(host, expected)
    device = conditioning.read.device
    if start is not None:
        if start.nonlinear != nonlinear or tuple(tuple(w.shape) for w in start.weights) != shapes:
            raise ContractViolation("warm start must match the declared adapter kind")
        adapter = start.detached()
    elif minibatch or hidden is not None or expanded:
        adapter = initial_adapter(seed, hidden=hidden, nonlinear=nonlinear, expanded=expanded, device=device)
    else:
        adapter = zero_adapter(device, nonlinear=nonlinear)
    began = time.monotonic()
    step0 = None if start is None else _selection(host, selection, adapter)
    n = len(conditioning.labels)
    order = torch.arange(n, device=device)
    draws = generator(seed, "oracle-order")
    traces: list[Tensor] = []
    snapshots: list[tuple[int, Adapter, float | None]] = []
    examples = 0
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
                time.monotonic() - began,
                examples,
                adapter.count() * (step + 1),
                tuple(snapshots),
                step0,
            )
        if step + 1 in checkpoints:
            snapshots.append((step + 1, adapter.detached(), _selection(host, selection, adapter)))
    selected = snapshots[-1][2] if snapshots and snapshots[-1][0] == steps else _selection(host, selection, adapter)
    history = tuple(float(v) for v in torch.stack(traces).cpu())
    assert_frozen(host, expected)
    return Candidate(
        adapter.detached(),
        rate,
        seed,
        steps,
        selected,
        history,
        time.monotonic() - began,
        examples,
        adapter.count() * steps,
        tuple(snapshots),
        step0,
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


def plateau(early: Candidate, late: Candidate, selection: Features, seed: int, host: Host) -> tuple[bool, float, float]:
    """v4 Table B rule for any two schedules: (certified, paired selection SE, training tail SE).

    Certified when the late schedule improves selection by less than 2 paired SEs and the last two
    40-step training windows of the late schedule differ by less than 2 tail SEs.
    """
    if early.selection is None or late.selection is None:
        raise ContractViolation("selected candidates must be finite")
    if len(late.training_losses) < 2 * PLATEAU_WINDOW:
        raise ContractViolation("late schedule must hold two training-loss windows")
    difference = losses(host, selection, early.adapter) - losses(host, selection, late.adapter)
    return table_b(early.selection, late.selection, difference, late.training_losses, seed)


def table_b(
    early_selection: float, late_selection: float, difference: Tensor, late_training: tuple[float, ...], seed: int
) -> tuple[bool, float, float]:
    """plateau()'s v4 Table B arithmetic on any two selected schedules (M0's re-tune arms share it, ruling R32).

    `difference` is the per-example early-minus-late selection loss; `late_training` the late run's step losses.
    """
    if len(late_training) < 2 * PLATEAU_WINDOW:
        raise ContractViolation("late schedule must hold two training-loss windows")
    se = paired_se(difference, seed)
    tail = torch.tensor(late_training[-2 * PLATEAU_WINDOW :], dtype=torch.float64)
    tail_se = float(tail.std(unbiased=True) / math.sqrt(len(tail)))
    change = float((tail[:PLATEAU_WINDOW].mean() - tail[PLATEAU_WINDOW:].mean()).abs())
    return early_selection - late_selection < 2 * se and change < 2 * tail_se, se, tail_se


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
    certified, se, tail_se = plateau(choose(finite1), choose(finite4), selection, seed, host)
    return Oracle(best, candidates, certified, se, tail_se, "both finite schedule comparators available")


def at_edge(rate: float, rates: tuple[float, ...]) -> bool:
    return rate in (max(rates), min(rates))


def _passes(n: int, steps: int, minibatch: bool) -> int:
    """Example passes of the first `steps` steps of fit_sgd's schedule."""
    if not minibatch:
        return n * steps
    chunks = math.ceil(n / 256)
    return sum(min(256, n - (step % chunks) * 256) for step in range(steps))


@dataclass(frozen=True)
class Grid:
    best: Candidate
    candidates: tuple[Candidate, ...]
    edge_of_grid: bool
    plateau: bool
    selection_se: float | None
    loss_tail_se: float | None
    plateau_evidence: str

    def report(self) -> Record:
        return {
            "best": self.best.report(),
            "candidates": [c.report() for c in self.candidates],
            "edge_of_grid": self.edge_of_grid,
            "plateau_certified": self.plateau,
            "selection_paired_bootstrap_se": self.selection_se,
            "training_tail_se": self.loss_tail_se,
            "plateau_evidence": self.plateau_evidence,
        }


def grid(
    host: Host,
    conditioning: Features,
    selection: Features,
    seed: int,
    *,
    rates: tuple[float, ...],
    checkpoints: tuple[int, ...],
    start: Adapter | None = None,
    hidden: int | None = None,
    nonlinear: bool = True,
    expanded: bool = False,
) -> Grid:
    """One minibatch run per rate to checkpoints[-1]; every (rate, checkpoint) snapshot is a candidate.

    Each run's wall time is booked once, on its final-checkpoint candidate, so candidate times sum to the run
    times. A checkpoint a failed run never reached is a nonfinite candidate carrying what the run spent; a
    checkpoint it did reach keeps its snapshot (weights and selection loss, bitwise a separate run of that length)
    but no training trace, since fit_sgd reports none for a failed run (ruling R33), and is reported with
    `trace_truncated: true`. That never matters to plateau(): the late comparator is the final checkpoint, reached
    only by a finite run.
    """
    if not rates or not checkpoints:
        raise ContractViolation("nonempty rate set and checkpoints required")
    n = len(conditioning.labels)
    candidates: list[Candidate] = []
    for rate in rates:
        run = fit_sgd(
            host,
            conditioning,
            selection,
            seed=seed,
            rate=rate,
            steps=checkpoints[-1],
            minibatch=True,
            start=start,
            hidden=hidden,
            nonlinear=nonlinear,
            expanded=expanded,
            checkpoints=checkpoints,
        )
        reached = {step: (adapter, loss) for step, adapter, loss in run.snapshots}
        for step in checkpoints:
            wall = run.wall_seconds if step == checkpoints[-1] else 0.0
            truncated = False
            if step in reached:
                adapter, loss = reached[step]
                trace = run.training_losses[:step] if run.training_losses else ()
                truncated = not run.training_losses
                passes, parameter_steps = _passes(n, step, True), adapter.count() * step
            else:
                adapter, loss, trace = run.adapter, None, ()
                passes, parameter_steps = run.example_passes, run.parameter_steps
            candidates.append(
                Candidate(
                    adapter,
                    rate,
                    seed,
                    step,
                    loss,
                    trace,
                    wall,
                    passes,
                    parameter_steps,
                    (),
                    run.step0_selection,
                    truncated,
                )
            )
    bank = tuple(candidates)
    best = choose(bank)
    edge = at_edge(best.rate, rates)
    if len(checkpoints) < 2:
        return Grid(best, bank, edge, False, None, None, "single checkpoint; no schedule comparator")
    early = tuple(c for c in bank if c.steps == checkpoints[-2] and c.selection is not None)
    late = tuple(c for c in bank if c.steps == checkpoints[-1] and c.selection is not None)
    if not early or not late:
        return Grid(
            best, bank, edge, False, None, None, "schedule comparator unavailable; finite overall winner retained"
        )
    certified, se, tail_se = plateau(choose(early), choose(late), selection, seed, host)
    return Grid(best, bank, edge, certified, se, tail_se, "both finite schedule comparators available")


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
