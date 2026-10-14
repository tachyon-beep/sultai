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
from sultai_screen.contracts import Json, Record, anchor_count, mapping, number
from sultai_screen.data import generator
from sultai_screen.fit import paired_se
from sultai_screen.m0.identity import HARM

Z_ONE_SIDED_95: Final = 1.645
BOOTSTRAP_SAMPLES: Final = 10_000
# The anchor-bootstrap, frontier-bootstrap and per-example paired-bootstrap seed of every M0 statistic (ruling R18 f):
# the first development anchor. Every per-anchor SE in `quantities` is fit.paired_se(difference, BOOTSTRAP_SEED), so
# each key and anchor draws the same 10,000 index resamples (generator(BOOTSTRAP_SEED, "paired-bootstrap")).
BOOTSTRAP_SEED: Final = 3000
SIGN_TEST_MINIMUM: Final = 6
# Parameter counts of the three core frontier widths s = 0.5, 1.0, 1.5; the reference host is s = 1.0.
CORE_PARAMETERS: Final = (85_274, 284_634, 613_018)
REFERENCE_PARAMETERS: Final = 284_634
FRONTIER_LABEL: Final = "same-recipe frontier, 10 epochs"
DESCRIPTIVE_LABEL: Final = "same-recipe frontier, 10 epochs, descriptive"
SURVIVAL_WIDTHS: Final = (16, 8, 4, 2)
SURVIVAL_KINDS: Final = ("sliced", "refit")

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
# Arm sets each stage runs by design; `quantities` raises when one of these is missing and reports every quantity
# whose arms are absent as None with reason "<arm> not run".
PILOT_ARMS: Final = frozenset(CORE_ARMS)
# §10 confirmatory set: N0 and calibrations, M-sq, M-aff (with residual), R-272s, R-aff, T-cool, M-sq | T-cool,
# M-wide, M-wide→8 (both rankings, sliced and refit), M-small, M-small-B, and (amendment 3, A3.1) the survival-curve
# widths M-wide→16/4/2 sliced and refit. S-static stays optional (14 h flag only).
CONFIRMATORY_ARMS: Final = frozenset(
    (
        *(f"M-wide→{h}:{kind}" for h in (16, 4, 2) for kind in ("sliced", "refit")),
        "R-temp",
        "R-cal",
        "M-sq",
        "M-aff",
        "R-272s",
        "R-aff",
        "T-cool",
        "M-sq|T-cool",
        "M-wide",
        "M-wide→8:rank:sliced",
        "M-wide→8:rank:refit",
        "M-wide→8:random:sliced",
        "M-wide→8:random:refit",
        "M-small",
        "M-small-B",
    )
)
# §5 extended item 1: the starved 0.5-host pair reports D1 and D3 only.
STARVED_PAIR_ARMS: Final = frozenset(("M-sq", "M-aff", "T-cool", "M-sq|T-cool", "R-temp", "R-cal"))
DENOMINATOR_REASON: Final = "denominator ≤ 0"
# §6: the two-arm quantities, each reported per anchor with its per-example paired-bootstrap SE (loss and accuracy).
PAIRED_SE_KEYS: Final = ("D1", "D1prime", "D2", "D2s", "D2_random", "D2s_random", "D3", "D4", "D5", "D5_norefit")
# key -> (the arm whose gain is added, the comparator arm whose gain is subtracted); D1prime and D3 are built apart.
PAIRS: Final[dict[str, tuple[str, str]]] = {
    "D1": ("M-sq", "M-aff"),
    "D2": ("M-wide→8:rank:refit", "M-small-B"),
    "D2s": ("M-wide→8:rank:sliced", "M-small"),
    "D2_random": ("M-wide→8:random:refit", "M-small-B"),
    "D2s_random": ("M-wide→8:random:sliced", "M-small"),
    "D4": ("M-sq-fresh", "M-sq"),
    "D5": ("S-static:refit", "M-sq"),
    "D5_norefit": ("S-static", "M-sq"),
}


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


def _survival_names(kind: str) -> dict[int, str]:
    return {h: f"M-wide→{h}:rank:{kind}" if h == 8 else f"M-wide→{h}:{kind}" for h in SURVIVAL_WIDTHS}


def quantities(result: AnchorResult, required: frozenset[str] = PILOT_ARMS) -> Record:
    """Per-anchor §6 quantities in nats; G(arm) = mean(N0 loss) - mean(arm loss).

    `required` is the arm set the stage runs by design (PILOT_ARMS, CONFIRMATORY_ARMS or STARVED_PAIR_ARMS); a
    missing required arm raises. Every quantity `k` carries `k_reason`: None when computed, otherwise
    "<arm> not run" for the first absent arm it needs, or "denominator ≤ 0" for a ratio whose denominator is not
    positive, with `k` itself None.
    """
    unknown = sorted(required - KNOWN_ARMS)
    if unknown:
        raise ContractViolation(f"unknown required arms: {unknown}")
    missing = sorted(required - set(result.arms))
    if missing:
        raise ContractViolation(f"required arms missing: {missing}")
    arms = result.arms
    noop = _mean(result.noop_loss)

    def loss(name: str) -> float:
        return _mean(arms[name].loss)

    def gain(name: str) -> float:
        return noop - loss(name)

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
        "unfolded": {
            name: {"loss": _mean(value), "gain": noop - _mean(value)} for name, value in result.unfolded.items()
        },
        "survivors": {name: list(units) for name, units in result.survivors.items()},
    }

    def put(key: str, needs: tuple[str, ...], compute: Callable[[], Json]) -> None:
        absent = next((name for name in needs if name not in arms), None)
        reason = None if absent is None else f"{absent} not run"
        record[key] = None if reason else compute()
        record[f"{key}_reason"] = reason

    def ratio(key: str, numerator: str, denominator: str) -> None:
        if denominator in arms and numerator in arms and gain(denominator) <= 0:
            record[key], record[f"{key}_reason"] = None, DENOMINATOR_REASON
        else:
            put(key, (numerator, denominator), lambda: gain(numerator) / gain(denominator))

    put("D1", ("M-sq", "M-aff"), lambda: gain("M-sq") - gain("M-aff"))
    put("D1prime", ("M-sq", "R-272s", "R-aff"), lambda: gain("M-sq") - max(gain("R-272s"), gain("R-aff")))
    put("D2", ("M-wide→8:rank:refit", "M-small-B"), lambda: gain("M-wide→8:rank:refit") - gain("M-small-B"))
    put("D2s", ("M-wide→8:rank:sliced", "M-small"), lambda: gain("M-wide→8:rank:sliced") - gain("M-small"))
    put(
        "D2_random",
        ("M-wide→8:random:refit", "M-small-B"),
        lambda: gain("M-wide→8:random:refit") - gain("M-small-B"),
    )
    put(
        "D2s_random",
        ("M-wide→8:random:sliced", "M-small"),
        lambda: gain("M-wide→8:random:sliced") - gain("M-small"),
    )
    put("D3", ("M-sq|T-cool",), lambda: _mean(result.tcool_noop_loss) - loss("M-sq|T-cool"))
    put("G_tcool", ("T-cool",), lambda: gain("T-cool"))
    put("D4", ("M-sq-fresh", "M-sq"), lambda: gain("M-sq-fresh") - gain("M-sq"))
    put("D5", ("M-sq", "S-static:refit"), lambda: loss("M-sq") - loss("S-static:refit"))
    put("D5_norefit", ("M-sq", "S-static"), lambda: loss("M-sq") - loss("S-static"))
    put("r_aff", ("M-aff",), lambda: result.fold_residual)
    put("delta_acc", ("M-sq",), lambda: _mean(arms["M-sq"].correct) - _mean(result.noop_correct))
    put("G_rtemp", ("R-temp",), lambda: gain("R-temp"))
    put("G_rcal", ("R-cal",), lambda: gain("R-cal"))
    put("G_mexp", ("M-exp",), lambda: gain("M-exp"))
    put("G_lottery", ("lottery",), lambda: gain("lottery"))
    put("leave_one_out", ("M-wide→8:ablation",), lambda: gain("M-wide→8:ablation"))
    ratio("retune_fraction", "M-sq", "R-all")
    # Amendment 3 (A3.2): the curve reports every width that ran and names the absent arms, by width then kind; it
    # is undefined only when M-wide is absent or its gain is not positive.
    record["survival_absent"] = [name for name in _survival_arms() if name not in arms]
    if "M-wide" in arms and gain("M-wide") <= 0:
        record["survival"], record["survival_reason"] = None, DENOMINATOR_REASON
    else:
        put("survival", ("M-wide",), lambda: _survival(gain, frozenset(arms)))
    record["harm"] = _harm(result, noop)
    record.update(_paired_ses(result, gain))
    return record


def _paired_ses(result: AnchorResult, gain: Callable[[str], float]) -> Record:
    """§6 (A4): per-anchor per-example paired-bootstrap SEs of every two-arm quantity, oriented as the quantity: loss
    difference L(comparator) - L(arm) and accuracy difference acc(arm) - acc(comparator). D1prime uses the anchor's
    larger re-tune gain (R-272s on a tie, as max() does); D3 the T-cool host's own N0 (its accuracy is the T-cool arm,
    that N0 evaluated). None where an arm is absent. Reporting only: no verdict reads these."""
    arms = result.arms
    pairs: dict[str, tuple[Tensor, Tensor | None] | None] = {}
    for key, (name, comparator) in PAIRS.items():
        present = name in arms and comparator in arms
        pairs[key] = (
            (arms[comparator].loss - arms[name].loss, arms[name].correct - arms[comparator].correct)
            if present
            else None
        )
    pairs["D1prime"] = None
    if all(name in arms for name in ("M-sq", "R-272s", "R-aff")):
        retune = "R-272s" if gain("R-272s") >= gain("R-aff") else "R-aff"
        pairs["D1prime"] = (arms[retune].loss - arms["M-sq"].loss, arms["M-sq"].correct - arms[retune].correct)
    pairs["D3"] = None
    if "M-sq|T-cool" in arms:
        finished = arms["M-sq|T-cool"]
        accuracy = finished.correct - arms["T-cool"].correct if "T-cool" in arms else None
        pairs["D3"] = (result.tcool_noop_loss - finished.loss, accuracy)
    record: Record = {}
    for key in PAIRED_SE_KEYS:
        pair = pairs[key]
        record[f"{key}_se"] = None if pair is None else paired_se(pair[0], BOOTSTRAP_SEED)
        record[f"{key}_acc_se"] = None if pair is None or pair[1] is None else paired_se(pair[1], BOOTSTRAP_SEED)
    return record


def _survival_arms() -> tuple[str, ...]:
    return tuple(
        f"M-wide→{h}:rank:{kind}" if h == 8 else f"M-wide→{h}:{kind}"
        for h in SURVIVAL_WIDTHS
        for kind in SURVIVAL_KINDS
    )


def _survival(gain: Callable[[str], float], present: frozenset[str]) -> Json:
    wide = gain("M-wide")
    return {
        kind: {str(h): gain(name) / wide for h, name in _survival_names(kind).items() if name in present}
        for kind in SURVIVAL_KINDS
    }


def survival_value(record: Record, kind: str, h: int) -> float | None:
    """One survival ratio G(M-wide→h)/G(M-wide) from a `quantities` record, or None where the curve or that width is
    undefined (amendment 3, A3.2: the per-width value the confirmatory bounds and the pilot's medians read)."""
    if kind not in SURVIVAL_KINDS or h not in SURVIVAL_WIDTHS:
        raise ContractViolation(f"no survival ratio {kind} at h = {h}")
    if record["survival"] is None:
        return None
    curve = mapping(mapping(record["survival"])[kind])
    return None if str(h) not in curve else number(curve[str(h)])


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
