"""Bounded seen-family lineage assay; oracle pairs contain gradient information.

All settings below are fixed before evaluation. No task fitting occurs in
FrozenGenerator.form; full ridge teachers are offline training supervision.
"""

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol as TypingProtocol

from .contracts import PartitionValidation, SplitValidation, numeric_matrix, numeric_vector
from .repair import CHANNELS, Adapter, Episode, Example, FrozenHost, finite, fit, mse, vector
from .report_types import (
    CostLedger,
    FormationConfig,
    FormationGates,
    FormationReport,
    LineageReport,
    MethodMetrics,
    MethodSummary,
    PhaseReport,
    count,
    loss,
    named,
    parse_CostLedger,
    parse_PhaseReport,
    validate_isolation,
    validate_phase,
)
from .trust import ContractViolation, FitUnavailable, InputDataError, _fit_finite, t2_operation, t3_boundary


@dataclass(frozen=True)
class Protocol:
    train_seeds: tuple[int, ...] = tuple(range(1000, 1032))
    dev_seeds: tuple[int, ...] = tuple(range(2000, 2008))
    test_seeds: tuple[int, ...] = tuple(range(3000, 3008))
    conditioning_count: int = 64
    selection_count: int = 24
    test_count: int = 64
    coefficient_bound: float = 0.35
    input_bound: float = 2.0
    teacher_ridge: float = 1e-8
    meta_ridge: float = 0.01
    admission_improvement: float = 1e-10
    sgd_learning_rate: float = 4.0
    probe_count: int = 32
    shuffled_lineage_rotation: int = 1
    mismatched_rotation: int = 1


FIXED = Protocol()
# Existing known-positive unit-test tolerance, promoted to the intended gate
# after the first audit by independent review; no model/data settings changed.
REPRESENTABILITY_TOLERANCE = 1e-12
FAMILY = "public-four-template-nonlinear-16"
EVIDENCE_MODES = ("coarse", "paired", "paired_probes")
COST_KEYS = (
    "oracle_conditioning_queries",
    "identity_probe_queries",
    "teacher_fits",
    "teacher_fit_examples",
    "meta_regression_fits",
    "meta_training_rows",
    "full_ridge_fits",
    "full_ridge_fit_examples",
    "sgd_steps",
    "sgd_example_steps",
    "generator_forwards",
    "retrieval_distance_evaluations",
    "selection_queries",
    "final_audit_queries",
    "affine_ridge_fits",
    "affine_ridge_fit_examples",
)


def costs(**entries: int) -> CostLedger:
    if set(entries) - set(COST_KEYS) or any(type(v) is not int or v < 0 for v in entries.values()):
        raise ValueError("invalid cost entry")
    return parse_CostLedger({key: entries[key] if key in entries else 0 for key in COST_KEYS})


@t3_boundary(
    test="tests/test_trust.py::BoundaryTests.test_formation_object_boundaries",
    fingerprint="b5eece8fe3e171bd5c634077a4077ebb6178e8c420aa7c88d0ba748a3abcc5ad",
)
def add_costs(*ledgers: object) -> CostLedger:
    if not ledgers:
        raise InputDataError("cost aggregation requires complete ledgers")
    validated = tuple(named(parse_CostLedger(ledger), count) for ledger in ledgers)
    return parse_CostLedger({key: sum(ledger[key] for ledger in validated) for key in COST_KEYS})


def public_templates() -> tuple[Adapter, ...]:
    """Four mutually orthogonal parameter directions, each squared norm 16."""
    zero = (0.0,) * CHANNELS
    diagonal = tuple(tuple(float(i == j) for j in range(CHANNELS)) for i in range(CHANNELS))
    shift = tuple(tuple(float(j == (i + 1) % CHANNELS) for j in range(CHANNELS)) for i in range(CHANNELS))
    alternate = tuple(
        tuple((1.0 if i % 2 == 0 else -1.0) * (j == (i + 2) % CHANNELS) for j in range(CHANNELS))
        for i in range(CHANNELS)
    )
    return (
        Adapter(diagonal, zero),
        Adapter(shift, zero),
        Adapter(alternate, zero),
        Adapter((zero,) * CHANNELS, tuple(1.0 if i % 2 == 0 else -1.0 for i in range(CHANNELS))),
    )


def materialize(coefficients: Sequence[float]) -> Adapter:
    if len(coefficients) != 4:
        raise ValueError("expected four public-template coefficients")
    coefficients = tuple(finite(c) for c in coefficients)
    templates = public_templates()
    return Adapter(
        tuple(
            tuple(
                sum(c * a.weights[i][j] for c, a in zip(coefficients, templates, strict=True)) for j in range(CHANNELS)
            )
            for i in range(CHANNELS)
        ),
        tuple(sum(c * a.bias[i] for c, a in zip(coefficients, templates, strict=True)) for i in range(CHANNELS)),
    )


def make_episode(seed: int, *, healthy: bool = False) -> Episode:
    """Seed is fixture construction only; it never enters generator evidence."""
    rng = random.Random(seed)
    coefficients = tuple(rng.uniform(-FIXED.coefficient_bound, FIXED.coefficient_bound) for _ in range(4))
    planted = Adapter.zero() if healthy else materialize(coefficients)
    lineage = f"procedural-lineage-{seed}" + ("-healthy" if healthy else "")

    def examples(split: str, count: int) -> tuple[Example, ...]:
        result = []
        for index in range(count):
            h = tuple(rng.uniform(-FIXED.input_bound, FIXED.input_bound) for _ in range(CHANNELS))
            result.append(Example(f"{lineage}:{split}:{index}", h, planted.apply(h)))
        return tuple(result)

    return Episode(
        lineage,
        FAMILY,
        examples("conditioning", FIXED.conditioning_count),
        examples("selection", FIXED.selection_count),
        examples("test", FIXED.test_count),
    )


def guard_lineage_splits(splits: Mapping[str, Sequence[Episode]]) -> SplitValidation | None:
    """Shared family is intentional; lineage AND sample IDs must be disjoint."""
    lineages: set[str] = set()
    samples: set[str] = set()
    for episodes in splits.values():
        for episode in episodes:
            if episode.host_lineage in lineages:
                raise ValueError("host lineage overlap")
            lineages.add(episode.host_lineage)
            ids = {x.sample_id for part in (episode.conditioning, episode.selection, episode.test) for x in part}
            if samples & ids:
                raise ValueError("sample ID overlap")
            samples.update(ids)
    if set(splits) != {"train", "development", "test"}:
        return None

    def partition(name: str) -> PartitionValidation:
        episodes = splits[name]
        ids = tuple(x.sample_id for e in episodes for part in (e.conditioning, e.selection, e.test) for x in part)
        digest = hashlib.sha256(json.dumps(ids, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return PartitionValidation(tuple(e.host_lineage for e in episodes), len(ids), digest)

    result = SplitValidation(partition("train"), partition("development"), partition("test"))
    validate_isolation(result.report())
    return result


def _basis(h: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    x = tuple(math.tanh(v) for v in h)
    signs = tuple(1.0 if i % 2 == 0 else -1.0 for i in range(CHANNELS))
    return (
        x,
        tuple(x[(i + 1) % CHANNELS] for i in range(CHANNELS)),
        tuple(signs[i] * x[(i + 2) % CHANNELS] for i in range(CHANNELS)),
        signs,
    )


def conditioning_evidence(conditioning: Sequence[Example], mode: str) -> tuple[float, ...]:
    """Coarse loses input/residual pairing; paired cross-moments recover gradients.

    Residuals are used exactly as supplied, including outside-family residuals.
    The probe bank executes identity host outputs; it adds no evidence features.
    """
    if mode not in EVIDENCE_MODES or not conditioning:
        raise ValueError("known mode and nonempty conditioning required")
    sums = [0.0] * 10
    for example in conditioning:
        residual = tuple(t - h for t, h in zip(example.target, example.h, strict=True))
        basis = _basis(example.h)
        sums[0] += sum(example.h)
        sums[1] += sum(v * v for v in example.h)
        sums[2] += sum(basis[0])
        sums[3] += sum(v * v for v in basis[0])
        sums[4] += sum(residual)
        sums[5] += sum(v * v for v in residual)
        for j in range(4):
            sums[6 + j] += sum(q * r for q, r in zip(basis[j], residual, strict=True))
    if mode == "paired_probes":
        host = FrozenHost()
        for channel in range(CHANNELS):
            for sign in (-1.0, 1.0):
                probe = tuple(sign if i == channel else 0.0 for i in range(CHANNELS))
                if host.output(probe) != probe:
                    raise ValueError("identity probe fixture changed")
    width = 6 if mode == "coarse" else 10
    return tuple(finite(value / (len(conditioning) * CHANNELS)) for value in sums[:width])


def _teacher_coefficients(adapter: Adapter) -> tuple[float, ...]:
    # Projection of OFFLINE teacher weights for regression supervision only.
    return tuple(
        finite(
            (
                sum(
                    w * q
                    for row, template in zip(adapter.weights, a.weights, strict=True)
                    for w, q in zip(row, template, strict=True)
                )
                + sum(b * q for b, q in zip(adapter.bias, a.bias, strict=True))
            )
            / CHANNELS
        )
        for a in public_templates()
    )


@t2_operation(
    invariants="Finite rectangular nonempty matched arrays and finite positive ridge; zero numeric pivot and nonfinite arithmetic are recoverable.",
    failures=(FitUnavailable,),
)
def _regress(
    rows: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], ridge: float = FIXED.meta_ridge
) -> tuple[tuple[float, ...], ...]:
    try:
        ridge = finite(ridge)
    except InputDataError as error:
        raise ContractViolation("finite positive ridge is required") from error
    if ridge <= 0 or not rows or not targets or len(rows) != len(targets):
        raise ContractViolation("nonempty matched regression arrays and positive ridge required")
    try:
        rows = numeric_matrix(rows, len(rows), len(rows[0]))
        targets = numeric_matrix(targets, len(targets), len(targets[0]))
    except InputDataError as error:
        raise ContractViolation("finite rectangular regression arrays are required") from error
    if not rows[0] or not targets[0]:
        raise ContractViolation("regression columns must be nonempty")
    width = len(rows[0]) + 1
    outputs = len(targets[0])
    augmented = [[ridge if i == j else 0.0 for j in range(width)] + [0.0] * outputs for i in range(width)]
    for row, target in zip(rows, targets, strict=True):
        x = (1.0, *tuple(row))
        for i in range(width):
            for j in range(width):
                augmented[i][j] += x[i] * x[j]
            for j in range(outputs):
                augmented[i][width + j] += x[i] * target[j]
    for column in range(width):
        pivot = max(range(column, width), key=lambda i: abs(augmented[i][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = _fit_finite(augmented[column][column])
        if divisor == 0:
            raise FitUnavailable("singular meta regression at machine precision")
        augmented[column] = [_fit_finite(v / divisor) for v in augmented[column]]
        for row_index in range(width):
            if row_index != column:
                factor = augmented[row_index][column]
                augmented[row_index] = [
                    _fit_finite(a - factor * b) for a, b in zip(augmented[row_index], augmented[column], strict=True)
                ]
    return tuple(tuple(row[width:]) for row in augmented)


@dataclass(frozen=True)
class FrozenGenerator:
    mode: str
    weights: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.mode not in EVIDENCE_MODES:
            raise ValueError("unknown evidence mode")
        object.__setattr__(self, "weights", numeric_matrix(self.weights, 7 if self.mode == "coarse" else 11, 4))

    def form(self, conditioning: Sequence[Example]) -> Adapter:
        x = (1.0, *conditioning_evidence(conditioning, self.mode))
        return materialize(
            tuple(finite(sum(value * row[j] for value, row in zip(x, self.weights, strict=True))) for j in range(4))
        )


@dataclass(frozen=True)
class TeacherBank:
    evidence: tuple[tuple[float, ...], ...]
    adapters: tuple[Adapter, ...]

    def __post_init__(self) -> None:
        if (
            not self.adapters
            or len(self.evidence) != len(self.adapters)
            or any(not isinstance(a, Adapter) for a in self.adapters)
        ):
            raise ValueError("teacher bank requires matched nonempty evidence and adapters")
        object.__setattr__(self, "evidence", numeric_matrix(self.evidence, len(self.adapters), 10))
        object.__setattr__(self, "adapters", tuple(self.adapters))


@dataclass(frozen=True)
class AffineControl:
    """Descriptive272-weight affine residual; distinct from primary Adapter."""

    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", numeric_matrix(self.weights, CHANNELS, CHANNELS))
        object.__setattr__(self, "bias", numeric_vector(self.bias, CHANNELS))

    def apply(self, h: Sequence[float]) -> tuple[float, ...]:
        h = vector(h)
        return vector(
            tuple(
                finite(value + sum(w * x for w, x in zip(row, h, strict=True)) + b)
                for value, row, b in zip(h, self.weights, self.bias, strict=True)
            )
        )


def _fit_affine(conditioning: Sequence[Example]) -> AffineControl:
    solution = _regress(
        tuple(x.h for x in conditioning),
        tuple(tuple(t - h for t, h in zip(x.target, x.h, strict=True)) for x in conditioning),
        FIXED.teacher_ridge,
    )
    return AffineControl(
        tuple(tuple(solution[1 + j][i] for j in range(CHANNELS)) for i in range(CHANNELS)), tuple(solution[0])
    )


class ConditioningSource(TypingProtocol):
    @property
    def conditioning(self) -> tuple[Example, ...]: ...


class EpisodeSource(ConditioningSource, TypingProtocol):
    @property
    def host_lineage(self) -> str: ...
    @property
    def bottleneck_family(self) -> str: ...
    @property
    def selection(self) -> tuple[Example, ...]: ...
    @property
    def test(self) -> tuple[Example, ...]: ...


def train_generators(
    training: Sequence[ConditioningSource],
) -> tuple[dict[str, FrozenGenerator], TeacherBank, CostLedger]:
    if not training:
        raise ValueError("nonempty training lineages required")
    teachers = tuple(fit(e.conditioning, FIXED.teacher_ridge) for e in training)
    targets = tuple(_teacher_coefficients(a) for a in teachers)
    rows = {mode: tuple(conditioning_evidence(e.conditioning, mode) for e in training) for mode in EVIDENCE_MODES}
    generators = {mode: FrozenGenerator(mode, _regress(rows[mode], targets)) for mode in EVIDENCE_MODES}
    n = sum(len(e.conditioning) for e in training)
    return (
        generators,
        TeacherBank(rows["paired"], teachers),
        costs(
            oracle_conditioning_queries=4 * n,
            identity_probe_queries=FIXED.probe_count * len(training),
            teacher_fits=len(training),
            teacher_fit_examples=n,
            meta_regression_fits=3,
            meta_training_rows=3 * len(training),
        ),
    )


def _sgd(conditioning: Sequence[Example], steps: int) -> Adapter:
    count(steps)
    if not conditioning:
        raise ValueError("nonempty conditioning required")
    weights = [[0.0] * CHANNELS for _ in range(CHANNELS)]
    bias = [0.0] * CHANNELS
    scale = 2 * FIXED.sgd_learning_rate / (len(conditioning) * CHANNELS)
    for _ in range(steps):
        gw = [[0.0] * CHANNELS for _ in range(CHANNELS)]
        gb = [0.0] * CHANNELS
        for example in conditioning:
            x = tuple(math.tanh(v) for v in example.h)
            for i in range(CHANNELS):
                error = (
                    example.h[i] + sum(w * q for w, q in zip(weights[i], x, strict=True)) + bias[i] - example.target[i]
                )
                gb[i] += error
                for j in range(CHANNELS):
                    gw[i][j] += error * x[j]
        for i in range(CHANNELS):
            bias[i] -= scale * gb[i]
            for j in range(CHANNELS):
                weights[i][j] -= scale * gw[i][j]
    return Adapter(tuple(tuple(row) for row in weights), tuple(bias))


def _retrieve(evidence: Sequence[float], bank: TeacherBank, neighbors: int) -> Adapter:
    if type(neighbors) is not int or not 1 <= neighbors <= len(bank.adapters):
        raise ValueError("neighbor count must fit the teacher bank")
    evidence = numeric_vector(evidence, 10)
    indices = sorted(
        range(len(bank.evidence)),
        key=lambda i: (sum((a - b) ** 2 for a, b in zip(evidence[6:], bank.evidence[i][6:], strict=True)), i),
    )[:neighbors]
    # Fixed equal-weight mean; distances use paired gradient moments only.
    return Adapter(
        tuple(
            tuple(sum(bank.adapters[k].weights[i][j] for k in indices) / len(indices) for j in range(CHANNELS))
            for i in range(CHANNELS)
        ),
        tuple(sum(bank.adapters[k].bias[i] for k in indices) / len(indices) for i in range(CHANNELS)),
    )


@dataclass(frozen=True)
class Candidate:
    name: str
    adapter: Adapter | AffineControl
    ledger: tuple[tuple[str, int], ...]


def candidate_bank(
    conditioning: Sequence[Example],
    generators: Mapping[str, FrozenGenerator],
    teachers: TeacherBank,
    *,
    shuffled_conditioning: Sequence[Example],
) -> tuple[Candidate, ...]:
    """All candidates are complete before any selection or test examples enter."""
    n = len(conditioning)
    result = []

    def append(name: str, adapter: Adapter | AffineControl, **ledger: int) -> None:
        result.append(Candidate(name, adapter, tuple(named(costs(**ledger), count).items())))

    append("no_op", Adapter.zero())
    for mode in EVIDENCE_MODES:
        append(
            mode,
            generators[mode].form(conditioning),
            oracle_conditioning_queries=n,
            identity_probe_queries=FIXED.probe_count if mode == "paired_probes" else 0,
            generator_forwards=1,
        )
    append(
        "full_ridge",
        fit(conditioning, FIXED.teacher_ridge),
        oracle_conditioning_queries=n,
        full_ridge_fits=1,
        full_ridge_fit_examples=n,
    )
    append(
        "affine_ridge",
        _fit_affine(conditioning),
        oracle_conditioning_queries=n,
        affine_ridge_fits=1,
        affine_ridge_fit_examples=n,
    )
    for steps in (1, 2, 4):
        append(
            f"sgd_{steps}",
            _sgd(conditioning, steps),
            oracle_conditioning_queries=n * steps,
            sgd_steps=steps,
            sgd_example_steps=n * steps,
        )
    for name, neighbor_count in (("retrieval", 1), ("interpolation_3", 3)):
        evidence = conditioning_evidence(conditioning, "paired")
        append(
            name,
            _retrieve(evidence, teachers, neighbor_count),
            oracle_conditioning_queries=n,
            retrieval_distance_evaluations=len(teachers.adapters),
        )
    append(
        "shuffled_conditioning",
        generators["paired"].form(shuffled_conditioning),
        oracle_conditioning_queries=len(shuffled_conditioning),
        generator_forwards=1,
    )
    mismatched = mismatched_residuals(conditioning)
    append(
        "mismatched_pairing", generators["paired"].form(mismatched), oracle_conditioning_queries=n, generator_forwards=1
    )
    return tuple(result)


def mismatched_residuals(conditioning: Sequence[Example]) -> tuple[Example, ...]:
    """Preserve identity and residual marginals; break input/residual pairing."""
    if not conditioning:
        raise ValueError("nonempty conditioning required")
    result = []
    for i, x in enumerate(conditioning):
        donor = conditioning[(i + FIXED.mismatched_rotation) % len(conditioning)]
        residual = tuple(t - h for t, h in zip(donor.target, donor.h, strict=True))
        result.append(Example(x.sample_id, x.h, tuple(h + r for h, r in zip(x.h, residual, strict=True))))
    return tuple(result)


@dataclass(frozen=True)
class Admission:
    adapter: Adapter | AffineControl
    acted: bool
    raw_selection_mse: float
    no_op_selection_mse: float
    selection_queries: int


def admit(
    candidate: Adapter | AffineControl, selection: Sequence[Example], *, baseline_loss: float | None = None
) -> Admission:
    raw = mse(candidate, selection)
    baseline = mse(Adapter.zero(), selection) if baseline_loss is None else loss(baseline_loss)
    acted = raw < baseline - FIXED.admission_improvement
    return Admission(
        candidate if acted else Adapter.zero(),
        acted,
        raw,
        baseline,
        len(selection) * (2 if baseline_loss is None else 1),
    )


def _evaluate(
    episodes: Sequence[EpisodeSource], generators: Mapping[str, FrozenGenerator], teachers: TeacherBank
) -> PhaseReport:
    if len(episodes) < 2:
        raise ValueError("conditioning derangement requires at least two lineages")
    # Complete every bank in the phase before reading any selection/test pairs.
    banks = tuple(
        candidate_bank(
            e.conditioning,
            generators,
            teachers,
            shuffled_conditioning=episodes[(i + FIXED.shuffled_lineage_rotation) % len(episodes)].conditioning,
        )
        for i, e in enumerate(episodes)
    )
    rows: list[LineageReport] = []
    for index, (episode, candidates) in enumerate(zip(episodes, banks, strict=True)):
        baseline_selection = mse(Adapter.zero(), episode.selection)
        # Admission is finished for every fixed candidate before final audits.
        decisions = tuple(admit(c.adapter, episode.selection, baseline_loss=baseline_selection) for c in candidates)
        raw = tuple(mse(c.adapter, episode.test) for c in candidates)
        admitted = tuple(mse(d.adapter, episode.test) for d in decisions)
        no_op_test = raw[0]
        methods: dict[str, MethodMetrics] = {}
        for candidate, decision, raw_loss, admitted_loss in zip(candidates, decisions, raw, admitted, strict=True):
            ledger = add_costs(
                dict(candidate.ledger),
                costs(selection_queries=decision.selection_queries, final_audit_queries=2 * len(episode.test)),
            )
            methods[candidate.name] = {
                "candidate_count": 0 if candidate.name == "no_op" else 1,
                "raw_selection_mse": decision.raw_selection_mse,
                "no_op_selection_mse": decision.no_op_selection_mse,
                "raw_test_mse": raw_loss,
                "admitted_test_mse": admitted_loss,
                "acted": decision.acted,
                "realized_harm": decision.acted and admitted_loss > no_op_test,
                "harm_mse_delta": admitted_loss - no_op_test,
                "costs": ledger,
            }
        rows.append(
            {
                "host_lineage": episode.host_lineage,
                "family": episode.bottleneck_family,
                "shuffled_conditioning_donor": episodes[
                    (index + FIXED.shuffled_lineage_rotation) % len(episodes)
                ].host_lineage,
                "methods": methods,
                "costs": add_costs(
                    *(m["costs"] for m in methods.values()), costs(selection_queries=len(episode.selection))
                ),
            }
        )
    summary: dict[str, MethodSummary] = {}
    for method in rows[0]["methods"]:
        values = [row["methods"][method] for row in rows]
        summary[method] = {
            "raw_mean_mse": finite(sum(v["raw_test_mse"] for v in values) / len(values)),
            "admitted_mean_mse": finite(sum(v["admitted_test_mse"] for v in values) / len(values)),
            "acted_lineages": sum(v["acted"] for v in values),
            "realized_harms": sum(v["realized_harm"] for v in values),
        }
    return {"lineages": rows, "summary": summary, "costs": add_costs(*(row["costs"] for row in rows))}


@t3_boundary(
    test="tests/test_trust.py::BoundaryTests.test_formation_object_boundaries",
    fingerprint="b5eece8fe3e171bd5c634077a4077ebb6178e8c420aa7c88d0ba748a3abcc5ad",
)
def formation_acceptance(
    development: object, test: object, healthy: object, *, isolation: SplitValidation | None = None
) -> FormationGates:
    """Require complete evidence and the guard's fixed-split validation result."""
    if isolation is None:
        raise InputDataError("validated split provenance is required")
    provenance = isolation.report()
    validate_isolation(provenance)
    dev = parse_PhaseReport(development)
    heldout = parse_PhaseReport(test)
    control = parse_PhaseReport(healthy)
    validate_phase(dev, provenance["development"]["lineages"])
    validate_phase(heldout, provenance["test"]["lineages"])
    validate_phase(control, tuple(name + "-healthy" for name in provenance["test"]["lineages"]))
    return {
        "paired_reduces_test_mean_mse": heldout["summary"]["paired"]["raw_mean_mse"]
        < heldout["summary"]["no_op"]["raw_mean_mse"],
        "full_ridge_representability": all(
            row["methods"]["full_ridge"]["raw_test_mse"] < REPRESENTABILITY_TOLERANCE for row in heldout["lineages"]
        ),
        "healthy_all_no_op": all(v["acted_lineages"] == 0 for v in control["summary"].values()),
        "lineage_and_sample_isolation": True,
        "finite_metrics": all(
            math.isfinite(v["raw_mean_mse"]) and math.isfinite(v["admitted_mean_mse"])
            for phase in (dev, heldout, control)
            for v in phase["summary"].values()
        ),
    }


def run_formation() -> tuple[FormationReport, FrozenGenerator]:
    training = tuple(make_episode(seed) for seed in FIXED.train_seeds)
    development = tuple(make_episode(seed) for seed in FIXED.dev_seeds)
    evaluation = tuple(make_episode(seed) for seed in FIXED.test_seeds)
    healthy_episodes = tuple(make_episode(seed, healthy=True) for seed in FIXED.test_seeds)
    # Healthy evidence participates in isolation before any teacher is fitted.
    # The legacy v2 report retains its three-part provenance schema; the current
    # correction report additionally records every healthy cohort's identity.
    guard_lineage_splits(
        {"train": training, "development": development, "test": evaluation, "healthy": healthy_episodes}
    )
    isolation = guard_lineage_splits({"train": training, "development": development, "test": evaluation})
    if isolation is None:
        raise ValueError("fixed split validation missing")
    generators, teachers, offline = train_generators(training)
    # No settings are selected from development; evaluate it as a separate audit.
    dev = _evaluate(development, generators, teachers)
    test = _evaluate(evaluation, generators, teachers)
    healthy = _evaluate(healthy_episodes, generators, teachers)
    config: FormationConfig = {
        "train_seeds": FIXED.train_seeds,
        "dev_seeds": FIXED.dev_seeds,
        "test_seeds": FIXED.test_seeds,
        "conditioning_count": FIXED.conditioning_count,
        "selection_count": FIXED.selection_count,
        "test_count": FIXED.test_count,
        "coefficient_bound": FIXED.coefficient_bound,
        "input_bound": FIXED.input_bound,
        "teacher_ridge": FIXED.teacher_ridge,
        "meta_ridge": FIXED.meta_ridge,
        "admission_improvement": FIXED.admission_improvement,
        "sgd_learning_rate": FIXED.sgd_learning_rate,
        "probe_count": FIXED.probe_count,
        "shuffled_lineage_rotation": FIXED.shuffled_lineage_rotation,
        "mismatched_rotation": FIXED.mismatched_rotation,
        "lineages": {"train": 32, "development": 8, "test": 8},
        "healthy_control_lineages": 8,
        "shared_family": FAMILY,
        "adapter_stored_parameters": 272,
        "effective_output_dimension": 4,
        "retrieval_retained_teacher_parameters": 32 * 272,
        "public_template_materialized_values": 4 * 272,
        "generator_parameters": {
            "coarse": sum(len(row) for row in generators["coarse"].weights),
            "paired": sum(len(row) for row in generators["paired"].weights),
            "paired_probes": sum(len(row) for row in generators["paired_probes"].weights),
        },
        "template_squared_parameter_norms": [16, 16, 16, 16],
        "template_definitions": ["W[i,i]=1", "W[i,(i+1)%16]=1", "W[i,(i+2)%16]=(-1)^i", "b[i]=(-1)^i"],
        "evidence_dimensions": {"coarse": 6, "paired": 10, "paired_probes": 10},
        "coarse_features": [
            "mean_h",
            "mean_h_squared",
            "mean_tanh_h",
            "mean_tanh_h_squared",
            "mean_residual",
            "mean_residual_squared",
        ],
        "paired_features": "coarse plus four mean basis-response/residual cross-moments",
        "probe_bank": "32 identity outputs at signed coordinate unit vectors; no added features",
        "retrieval": "Euclidean distance on four paired moments; training teachers only",
        "interpolation": "equal-weight mean of nearest three training teachers",
        "affine_control": "full272 ridge using h rather than tanh(h); descriptive",
        "shuffle": "within each phase lineage i receives conditioning from (i+1)%8; no held-out training",
        "mispair": "rotate residuals by one conditioning sample, target_i=h_i+(target_j-h_j)",
        "meta_intercept": "included and ridge-regularized like every coefficient",
        "full_ridge_representability_tolerance": REPRESENTABILITY_TOLERANCE,
        "representability_gate_provenance": "independent-review correction after first audit; existing1e-12 unit-test tolerance; no model/data tuning",
        "hyperparameter_selection": "none; all constants predeclared",
        "candidate_bank": "one candidate per method plus no-op, K=1 except no-op K=0",
    }
    all_costs = add_costs(offline, dev["costs"], test["costs"], healthy["costs"])
    report: FormationReport = {
        "isolation": isolation.report(),
        "assay": "seen-family synthetic lineage holdout",
        "config": config,
        "development": dev,
        "test": test,
        "healthy": healthy,
        "costs": {
            "offline": offline,
            "development": dev["costs"],
            "test": test["costs"],
            "healthy": healthy["costs"],
            "total": all_costs,
            "units": "example-output queries; repeated conditioning reads/SGD passes charged per method",
            "selection_baseline": "one shared no-op selection pass per lineage, charged at lineage level",
            "audit": "raw and admitted adapter outputs both executed for every method",
            "meta_regression": "3 pivoted normal-equation solves: 7/11/11 feature columns,4 outputs",
            "teachers": "32 conventional full272 ridge fits; retrieval retains32 teacher adapters",
        },
        "acceptance": formation_acceptance(dev, test, healthy, isolation=isolation),
        "summary": {
            "paired_raw_test_mean_mse": test["summary"]["paired"]["raw_mean_mse"],
            "paired_admitted_test_mean_mse": test["summary"]["paired"]["admitted_mean_mse"],
            "no_op_test_mean_mse": test["summary"]["no_op"]["raw_mean_mse"],
            "full_ridge_test_mean_mse": test["summary"]["full_ridge"]["raw_mean_mse"],
            "affine_ridge_test_mean_mse": test["summary"]["affine_ridge"]["raw_mean_mse"],
            "paired_acted_lineages": test["summary"]["paired"]["acted_lineages"],
            "paired_realized_harms": test["summary"]["paired"]["realized_harms"],
        },
        "limitations": [
            "Direct oracle residual observations are privileged evidence.",
            "Paired cross-moments contain task-gradient information.",
            "Identity-host probes add no identifying information.",
            "Shared public four-template family: no unseen-family or trained-host transfer claim.",
            "Admission is empirical, with no calibrated population harm-rate guarantee.",
            "All baseline comparisons are descriptive and retained, including unfavorable outcomes.",
            "Frozen form accepts outside-family residuals unchanged but emits only four-template adapters.",
            "Conditioning derangement uses another lineage's permitted conditioning in the same phase.",
            "Residual mismatching preserves identity and residual marginals while breaking their pairing.",
        ],
    }
    return report, generators["paired"]
