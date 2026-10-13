"""M0 statistics: per-anchor quantities (§6), anchor bootstrap, exact sign test, sizing and the frontier slope.

Pure statistics over per-anchor records; never imports `sultai_screen.screen` or `sultai_screen.pilot`.
Interval conventions are fixed by the implementation plan (Task 13) so two implementers compute the same numbers:
paired per-example bounds are normal (mean ∓ 1.645 · paired SE); anchor bounds are percentile bootstraps of the
median read at the 1-based 500th / 9,500th order statistics of 10,000 resamples, no interpolation.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

from sultai.trust import ContractViolation
from sultai_screen.contracts import Json, Record, anchor_count
from sultai_screen.data import generator
from sultai_screen.fit import paired_se
from sultai_screen.m0.identity import HARM

Z_ONE_SIDED_95: Final = 1.645
BOOTSTRAP_SAMPLES: Final = 10_000
SIGN_TEST_MINIMUM: Final = 6
# Parameter counts of the three core frontier widths s = 0.5, 1.0, 1.5; the reference host is s = 1.0.
CORE_PARAMETERS: Final = (85_274, 284_634, 613_018)
REFERENCE_PARAMETERS: Final = 284_634
FRONTIER_LABEL: Final = "same-recipe frontier, 10 epochs"
DESCRIPTIVE_LABEL: Final = "same-recipe frontier, 10 epochs, descriptive"
SURVIVAL_WIDTHS: Final = (16, 8, 4, 2)

CORE_ARMS: Final = (
    "M-sq",
    "M-aff",
    "R-272s",
    "R-aff",
    "T-cool",
    "M-sq|T-cool",
    "M-wide",
    "M-wide→8:rank:refit",
    "M-wide→8:rank:sliced",
    "M-wide→8:random:refit",
    "M-wide→8:random:sliced",
    "M-wide→16:sliced",
    "M-wide→4:sliced",
    "M-wide→2:sliced",
    "M-wide→16:refit",
    "M-wide→4:refit",
    "M-wide→2:refit",
    "M-wide→8:ablation",
    "M-small",
    "M-small-B",
    "M-sq-fresh",
    "R-temp",
    "R-cal",
)
OPTIONAL_ARMS: Final = ("R-all", "M-exp", "lottery", "S-static", "S-static:refit")
KNOWN_ARMS: Final = frozenset(CORE_ARMS + OPTIONAL_ARMS)
DENOMINATOR_REASON: Final = "denominator ≤ 0"


def _vector(tensor: Tensor, name: str) -> Tensor:
    if tensor.ndim != 1 or len(tensor) == 0 or not tensor.is_floating_point():
        raise ContractViolation(f"{name} must be a nonempty 1-D floating vector")
    if not bool(torch.isfinite(tensor).all()):
        raise ContractViolation(f"{name} must be finite")
    return tensor


def _indicator(tensor: Tensor, name: str) -> Tensor:
    _vector(tensor, name)
    if not bool(((tensor == 0) | (tensor == 1)).all()):
        raise ContractViolation(f"{name} must be 0/1 correctness")
    return tensor


def _mean(tensor: Tensor) -> float:
    return float(tensor.detach().double().mean())


@dataclass(frozen=True, eq=False)
class ArmResult:
    """One arm on one anchor: per-example dev loss and 0/1 correctness in caller order, plus the selection report."""

    loss: Tensor
    correct: Tensor
    certified: bool | None
    rate: float | None
    edge_of_grid: bool | None

    def __post_init__(self) -> None:
        _vector(self.loss, "arm loss")
        _indicator(self.correct, "arm correctness")
        if self.loss.shape != self.correct.shape:
            raise ContractViolation("arm loss and correctness must be aligned")


@dataclass(frozen=True, eq=False)
class AnchorResult:
    """Everything §6 needs for one anchor; every vector is aligned with `noop_loss` (the anchor's N0)."""

    arms: dict[str, ArmResult]
    noop_loss: Tensor
    noop_correct: Tensor
    tcool_noop_loss: Tensor
    fold_residual: float
    survivors: dict[str, tuple[int, ...]]
    unfolded: dict[str, Tensor]
    raw_candidates: dict[str, tuple[tuple[float, float], ...]]

    def __post_init__(self) -> None:
        shape = _vector(self.noop_loss, "N0 loss").shape
        if _indicator(self.noop_correct, "N0 correctness").shape != shape:
            raise ContractViolation("N0 correctness must be aligned with N0 loss")
        if _vector(self.tcool_noop_loss, "T-cool N0 loss").shape != shape:
            raise ContractViolation("T-cool N0 loss must be aligned with N0 loss")
        unknown = sorted((set(self.arms) | set(self.raw_candidates)) - KNOWN_ARMS)
        if unknown:
            raise ContractViolation(f"unknown arm names: {unknown}")
        for name, arm in self.arms.items():
            if arm.loss.shape != shape:
                raise ContractViolation(f"arm {name} is not aligned with N0")
        for name, loss in self.unfolded.items():
            if _vector(loss, f"unfolded {name}").shape != shape:
                raise ContractViolation(f"unfolded {name} is not aligned with N0")
        if not math.isfinite(self.fold_residual):
            raise ContractViolation("fold residual must be finite")


def _optional(value: float | None, reason: str | None, key: str) -> Record:
    return {key: value, f"{key}_reason": reason}


def quantities(result: AnchorResult) -> Record:
    """Per-anchor §6 quantities in nats; G(arm) = mean(N0 loss) - mean(arm loss). Optional values carry a reason."""
    missing = [name for name in CORE_ARMS if name not in result.arms]
    if missing:
        raise ContractViolation(f"core arms missing: {missing}")
    arms = result.arms
    noop = _mean(result.noop_loss)

    def loss(name: str) -> float:
        return _mean(arms[name].loss)

    def gain(name: str) -> float:
        return noop - loss(name)

    def absent(*names: str) -> str | None:
        for name in names:
            if name not in arms:
                return f"{name} not run"
        return None

    record: Record = {
        "noop_loss": noop,
        "noop_accuracy": _mean(result.noop_correct),
        "tcool_noop_loss": _mean(result.tcool_noop_loss),
        "arms": {
            name: {
                "loss": loss(name),
                "accuracy": _mean(arm.correct),
                "gain": gain(name),
                "certified": arm.certified,
                "rate": arm.rate,
                "edge_of_grid": arm.edge_of_grid,
            }
            for name, arm in arms.items()
        },
        "D1": gain("M-sq") - gain("M-aff"),
        "D1prime": gain("M-sq") - max(gain("R-272s"), gain("R-aff")),
        "D2": gain("M-wide→8:rank:refit") - gain("M-small-B"),
        "D2s": gain("M-wide→8:rank:sliced") - gain("M-small"),
        "D2_random": gain("M-wide→8:random:refit") - gain("M-small-B"),
        "D2s_random": gain("M-wide→8:random:sliced") - gain("M-small"),
        "D3": _mean(result.tcool_noop_loss) - loss("M-sq|T-cool"),
        "G_tcool": gain("T-cool"),
        "D4": gain("M-sq-fresh") - gain("M-sq"),
        "r_aff": result.fold_residual,
        "delta_acc": _mean(arms["M-sq"].correct) - _mean(result.noop_correct),
        "G_rtemp": gain("R-temp"),
        "G_rcal": gain("R-cal"),
        "leave_one_out": gain("M-wide→8:ablation"),
        "unfolded": {
            name: {"loss": _mean(value), "gain": noop - _mean(value)} for name, value in result.unfolded.items()
        },
        "survivors": {name: list(units) for name, units in result.survivors.items()},
    }
    for key, name in (("D5", "S-static:refit"), ("D5_norefit", "S-static")):
        reason = absent(name)
        record |= _optional(None if reason else loss("M-sq") - loss(name), reason, key)
    for key, name in (("G_mexp", "M-exp"), ("G_lottery", "lottery")):
        reason = absent(name)
        record |= _optional(None if reason else gain(name), reason, key)
    reason = absent("R-all") or (DENOMINATOR_REASON if gain("R-all") <= 0 else None)
    record |= _optional(None if reason else gain("M-sq") / gain("R-all"), reason, "retune_fraction")
    record |= _survival(gain)
    record["harm"] = _harm(result, noop)
    return record


def _survival(gain: Callable[[str], float]) -> Record:
    wide = gain("M-wide")
    if wide <= 0:
        return {"survival": None, "survival_reason": DENOMINATOR_REASON}
    curves: dict[str, Json] = {}
    for kind in ("sliced", "refit"):
        names = {h: f"M-wide→{h}:rank:{kind}" if h == 8 else f"M-wide→{h}:{kind}" for h in SURVIVAL_WIDTHS}
        curves[kind] = {str(h): gain(name) / wide for h, name in names.items()}
    return {"survival": curves, "survival_reason": None}


def _harm(result: AnchorResult, noop: float) -> list[Json]:
    """Inherited v4 harm screen: every raw candidate whose dev loss exceeds N0's by more than HARM (nonfinite too)."""
    harm: list[Json] = []
    for arm, candidates in result.raw_candidates.items():
        for candidate, value in candidates:
            if not math.isfinite(value):
                harm.append([arm, candidate, None])
            elif value - noop > HARM:
                harm.append([arm, candidate, value - noop])
    return harm


def paired_bound(difference: Tensor, seed: int, *, lower: bool) -> float:
    """One-sided 95 % normal bound on a per-example paired difference: mean ∓ 1.645 · fit.paired_se (v4 §7)."""
    mean = _mean(difference)
    se = paired_se(difference, seed)
    return mean - Z_ONE_SIDED_95 * se if lower else mean + Z_ONE_SIDED_95 * se


def _values(values: Sequence[float], minimum: int = 1) -> list[float]:
    data = [float(v) for v in values]
    if len(data) < minimum or not all(math.isfinite(v) for v in data):
        raise ContractViolation(f"at least {minimum} finite values required")
    return data


def median(values: Sequence[float]) -> float:
    """statistics.median: the mean of the two middle order statistics at even n."""
    return float(statistics.median(_values(values)))


def sd(values: Sequence[float]) -> float:
    """statistics.stdev: the sample SD with n - 1."""
    return float(statistics.stdev(_values(values, 2)))


def _row_medians(rows: Tensor) -> Tensor:
    """statistics.median of each row: sort, then the middle value or the mean of the two middle values."""
    ordered = rows.sort(dim=1).values
    n = rows.shape[1]
    return ordered[:, n // 2] if n % 2 else (ordered[:, n // 2 - 1] + ordered[:, n // 2]) / 2


def _order_statistic(replicates: Tensor, *, lower: bool) -> float:
    """1-based 5 % / 95 % order statistic of the replicates (500th / 9,500th of 10,000), no interpolation."""
    samples = len(replicates)
    tail = samples // 20
    ordered = replicates.sort().values
    return float(ordered[tail - 1] if lower else ordered[samples - tail - 1])


def _check_samples(samples: int) -> None:
    if samples < 20 or samples % 20:
        raise ContractViolation("bootstrap sample count must be a positive multiple of 20")


def anchor_bound(values: Sequence[float], seed: int, *, lower: bool, samples: int = BOOTSTRAP_SAMPLES) -> float:
    """Percentile anchor bootstrap of the median: one (samples, n) draw under generator(seed, "anchor-bootstrap")."""
    _check_samples(samples)
    data = torch.tensor(_values(values), dtype=torch.float64)
    indices = torch.randint(len(data), (samples, len(data)), generator=generator(seed, "anchor-bootstrap"))
    return _order_statistic(_row_medians(data[indices]), lower=lower)


def achieved_half_width(values: Sequence[float], seed: int) -> float:
    centre = median(values)
    return max(centre - anchor_bound(values, seed, lower=True), anchor_bound(values, seed, lower=False) - centre)


def _sign_rank(n: int) -> int:
    """The largest k with P(Bin(n, 1/2) < k) ≤ 0.05, in exact integer arithmetic."""
    k, below = 0, 0
    while 20 * (below + math.comb(n, k)) <= 2**n:
        below += math.comb(n, k)
        k += 1
    return k


def sign_test_bound(values: Sequence[float], *, lower: bool) -> tuple[float | None, str | None]:
    """Exact one-sided 95 % sign-test bound on the population median by order statistics.

    Lower: the k-th smallest; upper: the (n - k + 1)-th smallest. Zeros are kept (never dropped), so an exact zero
    counts against the direction tested: a bound that lands on it is 0, which is not > 0 (or < 0).
    """
    data = sorted(_values(values))
    n = len(data)
    if n < SIGN_TEST_MINIMUM:
        return None, f"n < {SIGN_TEST_MINIMUM}"
    k = _sign_rank(n)
    return (data[k - 1] if lower else data[n - k]), None


def confirmatory_count(sd_d1: float) -> tuple[int, int]:
    """(n*, max(32, n*)) for the median D1 under the v4 inflation: contracts.anchor_count."""
    return anchor_count(sd_d1)


@dataclass(frozen=True)
class Slope:
    """Loss decrease per doubling of parameters (the negated OLS slope of mean dev loss on log2 parameters)."""

    per_doubling: float
    lower: float
    upper: float
    widths: tuple[float, ...]
    label: str
    samples: int

    def report(self) -> Record:
        return {
            "per_doubling": self.per_doubling,
            "lower": self.lower,
            "upper": self.upper,
            "widths": list(self.widths),
            "label": self.label,
            "samples": self.samples,
            "units": "nats of mean dev loss per doubling of parameters",
        }


@dataclass(frozen=True)
class Fraction:
    """The module gain as a fraction of one doubling: median of the joint bootstrap with 5 % / 95 % bounds."""

    value: float | None
    lower: float | None
    upper: float | None
    point: float | None
    reason: str | None

    def report(self) -> Record:
        return {
            "value": self.value,
            "lower": self.lower,
            "upper": self.upper,
            "point": self.point,
            "reason": self.reason,
        }


def _frontier(points: dict[float, tuple[float, ...]]) -> tuple[list[float], list[Tensor]]:
    widths = sorted(points)
    if any(not math.isfinite(p) or p <= 0 for p in widths):
        raise ContractViolation("frontier widths must be finite positive parameter counts")
    return widths, [torch.tensor(_values(points[p]), dtype=torch.float64) for p in widths]


def _per_doubling(widths: list[float], means: Tensor) -> Tensor:
    """Negated OLS slope of the width means (columns, in `widths` order) on log2(parameters), per row."""
    x = torch.tensor([math.log2(p) for p in widths], dtype=torch.float64)
    centred = x - x.mean()
    return -(means * (centred / (centred * centred).sum())).sum(dim=1)


def _draws(seeds: list[Tensor], seed: int, samples: int) -> list[Tensor]:
    """Within-width seed resamples, one (samples, m) draw per width in ascending parameter order."""
    rng = generator(seed, "frontier-bootstrap")
    return [torch.randint(len(v), (samples, len(v)), generator=rng) for v in seeds]


def _fit(points: dict[float, tuple[float, ...]], seed: int, label: str, samples: int) -> Slope:
    _check_samples(samples)
    widths, seeds = _frontier(points)
    point = float(_per_doubling(widths, torch.stack([v.mean() for v in seeds]).unsqueeze(0))[0])
    draws = _draws(seeds, seed, samples)
    replicates = _per_doubling(widths, torch.stack([v[i].mean(dim=1) for v, i in zip(seeds, draws, strict=True)], 1))
    return Slope(
        per_doubling=point,
        lower=_order_statistic(replicates, lower=True),
        upper=_order_statistic(replicates, lower=False),
        widths=tuple(widths),
        label=label,
        samples=samples,
    )


def _core(points: dict[float, tuple[float, ...]]) -> None:
    if sorted(points) != list(CORE_PARAMETERS):
        raise ContractViolation(f"the frontier slope uses exactly the three core widths {CORE_PARAMETERS}")


def frontier_slope(points: dict[float, tuple[float, ...]], seed: int, *, samples: int = BOOTSTRAP_SAMPLES) -> Slope:
    """Slope over the three core widths (parameter count → its seeds' N0 dev losses); no extrapolation."""
    _core(points)
    return _fit(points, seed, FRONTIER_LABEL, samples)


def frontier_slope_descriptive(
    points: dict[float, tuple[float, ...]], seed: int, *, samples: int = BOOTSTRAP_SAMPLES
) -> Slope:
    """The same fit over every 10-epoch width that ran (2 to 5); the caller never passes the cooled 20-epoch point."""
    if not 2 <= len(points) <= 5:
        raise ContractViolation("descriptive frontier needs 2 to 5 ten-epoch widths")
    return _fit(points, seed, DESCRIPTIVE_LABEL, samples)


def gain_fraction(
    gain_per_anchor: tuple[float, ...],
    points: dict[float, tuple[float, ...]],
    seed: int,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
) -> Fraction:
    """Mean reference-host module gain over the per-doubling slope, bootstrapped jointly.

    `gain_per_anchor` is aligned with `points[REFERENCE_PARAMETERS]` (the same seeds); each replicate uses one
    reference-width index draw for both the gain mean and the reference loss mean, the other widths independently.
    """
    _core(points)
    _check_samples(samples)
    widths, seeds = _frontier(points)
    gains = torch.tensor(_values(gain_per_anchor), dtype=torch.float64)
    reference = widths.index(REFERENCE_PARAMETERS)
    if len(gains) != len(seeds[reference]):
        raise ContractViolation("gains must be aligned with the reference width's seeds")
    slope = float(_per_doubling(widths, torch.stack([v.mean() for v in seeds]).unsqueeze(0))[0])
    if slope <= 0:
        return Fraction(None, None, None, None, "slope ≤ 0")
    draws = _draws(seeds, seed, samples)
    slopes = _per_doubling(widths, torch.stack([v[i].mean(dim=1) for v, i in zip(seeds, draws, strict=True)], 1))
    numerators = gains[draws[reference]].mean(dim=1)
    infinite = torch.copysign(torch.full_like(numerators, math.inf), numerators)
    fractions = torch.where(slopes == 0, infinite, numerators / slopes)
    return Fraction(
        value=float(_row_medians(fractions.unsqueeze(0))[0]),
        lower=_order_statistic(fractions, lower=True),
        upper=_order_statistic(fractions, lower=False),
        point=float(gains.mean()) / slope,
        reason=None,
    )
