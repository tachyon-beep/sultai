"""Frozen bounded correction assay; development and confirmation run separately.

The learned maps consume conditioning only. All fixed cohorts are validated
before fitting, and all candidate banks are frozen before selection is read.
False admission/rejection labels describe these finite, clean-audit fixtures.
"""

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .contracts import numeric_matrix
from .formation import (
    FIXED,
    REPRESENTABILITY_TOLERANCE,
    AffineControl,
    EpisodeSource,
    FrozenGenerator,
    TeacherBank,
    _basis,
    _regress,
    _teacher_coefficients,
    add_costs,
    candidate_bank,
    costs,
    make_episode,
    materialize,
    train_generators,
)
from .repair import CHANNELS, PARAMETERS, Adapter, Episode, Example, finite, mse
from .report_types import METHODS, CostLedger, count, named, parse_CostLedger
from .trust import ContractViolation, FitUnavailable, _fit_finite, t2_operation

CONTROL_METHODS = (*METHODS, "analytic4x4", "marginal96", "frozen_zero", "frozen_negated", "frozen_norm_random")
AUDIT_EPSILON = 1e-12
NOISE_SIGMA = 0.1
RANDOM_ROLE = "sultai-formation-random-v1"
NOISE_ROLE = "sultai-noisy-healthy-v1"
Phase = Literal["development", "confirmation"]


@dataclass(frozen=True)
class ControlCost:
    """Complete historical ledger plus separately charged analytic fitting."""

    legacy: tuple[tuple[str, int], ...]
    analytic4x4_fits: int
    analytic4x4_fit_examples: int
    derived_candidates: int = 0
    negation_transforms: int = 0
    random_direction_draws: int = 0
    norm_rescalings: int = 0

    def __post_init__(self) -> None:
        if len(dict(self.legacy)) != len(self.legacy):
            raise ContractViolation("duplicate cost keys")
        parse_CostLedger(dict(self.legacy))
        count(self.analytic4x4_fits)
        count(self.analytic4x4_fit_examples)
        count(self.derived_candidates)
        count(self.negation_transforms)
        count(self.random_direction_draws)
        count(self.norm_rescalings)


def _cost(
    ledger: CostLedger,
    *,
    analytic_fits: int = 0,
    analytic_examples: int = 0,
    derived: int = 0,
    negations: int = 0,
    random_draws: int = 0,
    rescalings: int = 0,
) -> ControlCost:
    return ControlCost(
        tuple(named(ledger, count).items()),
        analytic_fits,
        analytic_examples,
        derived,
        negations,
        random_draws,
        rescalings,
    )


def _total_cost(values: Sequence[ControlCost]) -> ControlCost:
    return _cost(
        add_costs(*(dict(v.legacy) for v in values)),
        analytic_fits=sum(v.analytic4x4_fits for v in values),
        analytic_examples=sum(v.analytic4x4_fit_examples for v in values),
        derived=sum(v.derived_candidates for v in values),
        negations=sum(v.negation_transforms for v in values),
        random_draws=sum(v.random_direction_draws for v in values),
        rescalings=sum(v.norm_rescalings for v in values),
    )


@dataclass(frozen=True)
class CohortIdentity:
    name: str
    lineages: tuple[str, ...]
    conditioning_count: int
    selection_count: int
    audit_count: int
    sample_count: int
    sample_ids_sha256: str
    pairs_sha256: str


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def guard_control_cohorts(cohorts: Mapping[str, Sequence[EpisodeSource]]) -> tuple[CohortIdentity, ...]:
    """Check every sample, including within-episode collisions, before a fit."""
    seen_lineages: set[str] = set()
    seen_samples: set[str] = set()
    result = []
    for name, episodes in cohorts.items():
        if not episodes:
            raise ContractViolation("nonempty control cohort required")
        ids = []
        pairs = []
        for episode in episodes:
            if episode.host_lineage in seen_lineages:
                raise ContractViolation("host lineage overlap")
            seen_lineages.add(episode.host_lineage)
            for partition, examples in (
                ("conditioning", episode.conditioning),
                ("selection", episode.selection),
                ("audit", episode.test),
            ):
                if not examples:
                    raise ContractViolation("nonempty episode partition required")
                for example in examples:
                    if not isinstance(example, Example):
                        raise ContractViolation("paired Example required")
                    if example.sample_id in seen_samples:
                        raise ContractViolation("sample ID overlap")
                    seen_samples.add(example.sample_id)
                    ids.append(example.sample_id)
                    pairs.append((episode.host_lineage, partition, example.sample_id, example.h, example.target))
        conditioning = sum(len(e.conditioning) for e in episodes)
        selection = sum(len(e.selection) for e in episodes)
        audit = sum(len(e.test) for e in episodes)
        result.append(
            CohortIdentity(
                name,
                tuple(e.host_lineage for e in episodes),
                conditioning,
                selection,
                audit,
                len(ids),
                _digest(ids),
                _digest(pairs),
            )
        )
    return tuple(result)


def marginal_evidence(conditioning: Sequence[Example]) -> tuple[float, ...]:
    """96 per-channel moments, with no input/residual cross-products."""
    if not conditioning:
        raise ContractViolation("nonempty conditioning required")
    groups = (
        tuple(e.h for e in conditioning),
        tuple(tuple(math.tanh(v) for v in e.h) for e in conditioning),
        tuple(tuple(t - h for t, h in zip(e.target, e.h, strict=True)) for e in conditioning),
    )
    return tuple(
        finite(math.fsum(row[channel] ** power for row in group) / len(conditioning))
        for group in groups
        for power in (1, 2)
        for channel in range(CHANNELS)
    )


@dataclass(frozen=True)
class FrozenMarginalGenerator:
    weights: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", numeric_matrix(self.weights, 97, 4))

    def form(self, conditioning: Sequence[Example]) -> Adapter:
        x = (1.0, *marginal_evidence(conditioning))
        return materialize(
            tuple(finite(math.fsum(v * row[j] for v, row in zip(x, self.weights, strict=True))) for j in range(4))
        )


@t2_operation(
    invariants="Nonempty owned finite conditioning pairs and fixed positive ridge; zero pivot and nonfinite arithmetic are recoverable.",
    failures=(FitUnavailable,),
)
def analytic_template_fit(conditioning: Sequence[Example]) -> Adapter:
    """Empirical four-template ridge normal equations, without an intercept."""
    if not conditioning:
        raise ContractViolation("nonempty conditioning required")

    def finite_sum(values: Iterable[float]) -> float:
        # Catch only the summation operation's documented numeric overflow.
        # Finite residuals prevent inf/-inf cancellation from reaching fsum.
        try:
            result = math.fsum(values)
        except OverflowError as error:
            raise FitUnavailable("analytic summation overflow at machine precision") from error
        return _fit_finite(result)

    augmented = [[FIXED.teacher_ridge if i == j else 0.0 for j in range(4)] + [0.0] for i in range(4)]
    for example in conditioning:
        basis = _basis(example.h)
        residual = tuple(_fit_finite(t - h) for t, h in zip(example.target, example.h, strict=True))
        for i in range(4):
            for j in range(4):
                augmented[i][j] += finite_sum(a * b for a, b in zip(basis[i], basis[j], strict=True))
            augmented[i][4] += finite_sum(a * b for a, b in zip(basis[i], residual, strict=True))
    for column in range(4):
        pivot = max(range(column, 4), key=lambda i: abs(augmented[i][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = _fit_finite(augmented[column][column])
        if divisor == 0.0:
            raise FitUnavailable("singular analytic template system")
        augmented[column] = [_fit_finite(v / divisor) for v in augmented[column]]
        for index in range(4):
            if index != column:
                factor = augmented[index][column]
                augmented[index] = [
                    _fit_finite(a - factor * b) for a, b in zip(augmented[index], augmented[column], strict=True)
                ]
    return materialize(tuple(row[4] for row in augmented))


def _seed(role: str, identity: str, partition: str | None = None) -> int:
    text = f"{role}:{identity}" if partition is None else f"{role}:{identity}:{partition}"
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def norm_matched_random(adapter: Adapter, lineage: str) -> Adapter:
    """One fixed random template direction; no observed outcomes choose it."""
    norm = math.fsum(w * w for row in adapter.weights for w in row) + math.fsum(b * b for b in adapter.bias)
    if norm == 0.0:
        return Adapter.zero()
    rng = random.Random(_seed(RANDOM_ROLE, lineage))
    direction = tuple(rng.gauss(0.0, 1.0) for _ in range(4))
    scale = math.sqrt(norm / (CHANNELS * math.fsum(v * v for v in direction)))
    return materialize(tuple(finite(scale * v) for v in direction))


def make_noisy_healthy_episode(seed: int) -> Episode:
    base = make_episode(seed, healthy=True)
    lineage = f"procedural-lineage-{seed}-noisy-healthy"

    def noisy(partition: str, examples: Sequence[Example]) -> tuple[Example, ...]:
        rng = random.Random(_seed(NOISE_ROLE, lineage, partition))
        return tuple(
            Example(f"{lineage}:{partition}:{index}", e.h, tuple(h + rng.gauss(0.0, NOISE_SIGMA) for h in e.h))
            for index, e in enumerate(examples)
        )

    return Episode(
        lineage,
        base.bottleneck_family,
        noisy("conditioning", base.conditioning),
        noisy("selection", base.selection),
        tuple(Example(f"{lineage}:audit:{i}", e.h, e.h) for i, e in enumerate(base.test)),
    )


@dataclass(frozen=True)
class ControlCandidate:
    name: str
    adapter: Adapter | AffineControl
    cost: ControlCost


def control_candidate_bank(
    episode: EpisodeSource,
    generators: Mapping[str, FrozenGenerator],
    teachers: TeacherBank,
    marginal: FrozenMarginalGenerator,
    *,
    shuffled_conditioning: Sequence[Example],
) -> tuple[ControlCandidate, ...]:
    historical = candidate_bank(episode.conditioning, generators, teachers, shuffled_conditioning=shuffled_conditioning)
    result = [ControlCandidate(c.name, c.adapter, _cost(parse_CostLedger(dict(c.ledger)))) for c in historical]
    paired = next(c.adapter for c in historical if c.name == "paired")
    if not isinstance(paired, Adapter):
        raise TypeError("paired generator must produce an Adapter")
    n = len(episode.conditioning)
    nonzero_paired = any(w != 0.0 for row in paired.weights for w in row) or any(b != 0.0 for b in paired.bias)
    result.extend(
        (
            ControlCandidate(
                "analytic4x4",
                analytic_template_fit(episode.conditioning),
                _cost(costs(oracle_conditioning_queries=n), analytic_fits=1, analytic_examples=n),
            ),
            ControlCandidate(
                "marginal96",
                marginal.form(episode.conditioning),
                _cost(costs(oracle_conditioning_queries=n, generator_forwards=1)),
            ),
            ControlCandidate("frozen_zero", Adapter.zero(), _cost(costs(), derived=1)),
            ControlCandidate(
                "frozen_negated",
                Adapter(tuple(tuple(-w for w in row) for row in paired.weights), tuple(-b for b in paired.bias)),
                _cost(costs(), derived=1, negations=1),
            ),
            ControlCandidate(
                "frozen_norm_random",
                norm_matched_random(paired, episode.host_lineage),
                _cost(costs(), derived=1, random_draws=4 if nonzero_paired else 0, rescalings=int(nonzero_paired)),
            ),
        )
    )
    if tuple(c.name for c in result) != CONTROL_METHODS:
        raise ContractViolation("complete fixed candidate inventory required")
    return tuple(result)


@dataclass(frozen=True)
class RandomStream:
    role: str
    partition: str
    fixture_identity: str
    seed: int


@dataclass(frozen=True)
class ControlMethodReport:
    name: str
    candidate_count: int
    raw_selection_mse: float
    no_op_selection_mse: float
    raw_audit_mse: float
    admitted_audit_mse: float
    acted: bool
    no_op: bool
    realized_harm: bool
    harm_mse_delta: float
    audit_improvement: float
    audit_class: Literal["helpful", "harmful", "neutral"]
    exact_null: bool
    false_admission: bool
    harmful_admission: bool
    neutral_admission: bool
    false_rejection: bool
    cost: ControlCost


@dataclass(frozen=True)
class ControlLineageReport:
    host_lineage: str
    family: str
    shuffled_conditioning_donor: str
    no_op_audit_mse: float
    methods: tuple[ControlMethodReport, ...]
    random_streams: tuple[RandomStream, ...]
    cost: ControlCost


@dataclass(frozen=True)
class ControlMethodSummary:
    name: str
    raw_mean_mse: float
    admitted_mean_mse: float
    acted_lineages: int
    realized_harms: int
    false_admissions: int
    harmful_admissions: int
    neutral_admissions: int
    false_rejections: int
    no_op_lineages: int
    helpful_candidates: int
    harmful_candidates: int
    neutral_candidates: int
    exact_null_candidates: int


@dataclass(frozen=True)
class ControlCohortReport:
    name: str
    lineages: tuple[ControlLineageReport, ...]
    summary: tuple[ControlMethodSummary, ...]
    cost: ControlCost


def _evaluate_frozen_cohort(
    name: str,
    episodes: Sequence[EpisodeSource],
    banks: Sequence[Sequence[ControlCandidate]],
    *,
    protocol_generated_noise: bool = False,
) -> ControlCohortReport:
    rows = []
    for index, (episode, candidates) in enumerate(zip(episodes, banks, strict=True)):
        baseline_selection = mse(Adapter.zero(), episode.selection)
        selection = tuple(mse(c.adapter, episode.selection) for c in candidates)
        decisions = tuple(raw < baseline_selection - FIXED.admission_improvement for raw in selection)
        # No audit evaluation participates in selection or candidate construction.
        raw = tuple(mse(c.adapter, episode.test) for c in candidates)
        admitted = tuple(
            mse(c.adapter if acted else Adapter.zero(), episode.test)
            for c, acted in zip(candidates, decisions, strict=True)
        )
        baseline_audit = raw[0]
        methods = []
        for candidate, selection_loss, acted, raw_loss, admitted_loss in zip(
            candidates, selection, decisions, raw, admitted, strict=True
        ):
            improvement = finite(baseline_audit - raw_loss)
            audit_class: Literal["helpful", "harmful", "neutral"] = (
                "helpful" if improvement > AUDIT_EPSILON else "harmful" if improvement < -AUDIT_EPSILON else "neutral"
            )
            null = all(w == 0.0 for row in candidate.adapter.weights for w in row) and all(
                b == 0.0 for b in candidate.adapter.bias
            )
            method_cost = _total_cost(
                (
                    candidate.cost,
                    _cost(costs(selection_queries=len(episode.selection), final_audit_queries=2 * len(episode.test))),
                )
            )
            methods.append(
                ControlMethodReport(
                    candidate.name,
                    0 if candidate.name == "no_op" else 1,
                    selection_loss,
                    baseline_selection,
                    raw_loss,
                    admitted_loss,
                    acted,
                    not acted,
                    acted and admitted_loss > baseline_audit,
                    finite(admitted_loss - baseline_audit),
                    improvement,
                    audit_class,
                    null,
                    acted and audit_class != "helpful",
                    acted and audit_class == "harmful",
                    acted and audit_class == "neutral",
                    not acted and audit_class == "helpful",
                    method_cost,
                )
            )
        streams = [
            RandomStream(RANDOM_ROLE, "candidate", episode.host_lineage, _seed(RANDOM_ROLE, episode.host_lineage))
        ]
        if name == "noisy_healthy" and protocol_generated_noise:
            streams.extend(
                RandomStream(
                    NOISE_ROLE, partition, episode.host_lineage, _seed(NOISE_ROLE, episode.host_lineage, partition)
                )
                for partition in ("conditioning", "selection")
            )
        rows.append(
            ControlLineageReport(
                episode.host_lineage,
                episode.bottleneck_family,
                episodes[(index + FIXED.shuffled_lineage_rotation) % len(episodes)].host_lineage,
                baseline_audit,
                tuple(methods),
                tuple(streams),
                _total_cost((*tuple(m.cost for m in methods), _cost(costs(selection_queries=len(episode.selection))))),
            )
        )
    summaries = []
    for method_index, method in enumerate(CONTROL_METHODS):
        values = tuple(row.methods[method_index] for row in rows)
        summaries.append(
            ControlMethodSummary(
                method,
                finite(math.fsum(v.raw_audit_mse for v in values) / len(values)),
                finite(math.fsum(v.admitted_audit_mse for v in values) / len(values)),
                sum(v.acted for v in values),
                sum(v.realized_harm for v in values),
                sum(v.false_admission for v in values),
                sum(v.harmful_admission for v in values),
                sum(v.neutral_admission for v in values),
                sum(v.false_rejection for v in values),
                sum(v.no_op for v in values),
                sum(v.audit_class == "helpful" for v in values),
                sum(v.audit_class == "harmful" for v in values),
                sum(v.audit_class == "neutral" for v in values),
                sum(v.exact_null for v in values),
            )
        )
    return ControlCohortReport(name, tuple(rows), tuple(summaries), _total_cost(tuple(row.cost for row in rows)))


@dataclass(frozen=True)
class FormationAssets:
    adapter_stored_parameters: int
    effective_output_dimension: int
    public_template_materialized_values: int
    retained_teacher_parameters: int
    retained_teacher_evidence_values: int
    historical_generator_parameters: tuple[tuple[str, int], ...]
    marginal_generator_parameters: int
    candidate_bank_stored_values: int
    candidate_count_per_lineage: int
    corpus_numeric_values: int


@dataclass(frozen=True)
class FormationControlConfig:
    teacher_ridge: float
    meta_ridge: float
    admission_improvement: float
    audit_classification_epsilon: float
    noisy_healthy_sigma: float
    marginal_features: str
    analytic_fit: str
    random_seed_derivation: str
    noise_seed_derivation: str
    evidence_limit: str


@dataclass(frozen=True)
class FormationControlReport:
    protocol: str
    phase: Literal["development", "confirmation", "fixture"]
    config: FormationControlConfig
    provenance: tuple[CohortIdentity, ...]
    cohorts: tuple[ControlCohortReport, ...]
    offline_cost: ControlCost
    total_cost: ControlCost
    assets: FormationAssets
    clean_healthy_all_no_op: bool
    analytic_in_family_representability: bool
    finite_metrics: bool
    software_valid: bool


def evaluate_formation_controls(
    training: Sequence[EpisodeSource],
    evaluation: Sequence[EpisodeSource],
    clean_healthy: Sequence[EpisodeSource],
    noisy_healthy: Sequence[EpisodeSource],
    *,
    phase: Literal["development", "confirmation", "fixture"] = "fixture",
) -> tuple[FormationControlReport, FrozenGenerator]:
    """Evaluate supplied unit fixtures or the predeclared current-phase cohorts."""
    if phase not in ("development", "confirmation", "fixture"):
        raise ContractViolation("unknown formation control phase")
    cohorts = {
        "train": training,
        "evaluation": evaluation,
        "clean_healthy": clean_healthy,
        "noisy_healthy": noisy_healthy,
    }
    provenance = guard_control_cohorts(cohorts)
    if len(training) < 3 or any(len(episodes) < 2 for episodes in (evaluation, clean_healthy, noisy_healthy)):
        raise ContractViolation("three teachers and two lineages per evaluated cohort required")
    if any(x.target != x.h for e in (*clean_healthy, *noisy_healthy) for x in e.test):
        raise ContractViolation("healthy audit must use clean latent identity")
    if any(x.target != x.h for e in clean_healthy for part in (e.conditioning, e.selection) for x in part):
        raise ContractViolation("clean healthy conditioning and selection must be identity")
    generators, teachers, offline = train_generators(training)
    marginal = FrozenMarginalGenerator(
        _regress(
            tuple(marginal_evidence(e.conditioning) for e in training),
            tuple(_teacher_coefficients(a) for a in teachers.adapters),
            FIXED.meta_ridge,
        )
    )
    offline_cost = _cost(
        add_costs(
            offline,
            costs(
                oracle_conditioning_queries=sum(len(e.conditioning) for e in training),
                meta_regression_fits=1,
                meta_training_rows=len(training),
            ),
        )
    )
    evaluated = tuple((name, episodes) for name, episodes in cohorts.items() if name != "train")
    # Freeze all candidate banks across every cohort before reading selection.
    all_banks = tuple(
        tuple(
            control_candidate_bank(
                episode,
                generators,
                teachers,
                marginal,
                shuffled_conditioning=episodes[(i + FIXED.shuffled_lineage_rotation) % len(episodes)].conditioning,
            )
            for i, episode in enumerate(episodes)
        )
        for _, episodes in evaluated
    )
    reports = tuple(
        _evaluate_frozen_cohort(name, episodes, banks, protocol_generated_noise=phase != "fixture")
        for (name, episodes), banks in zip(evaluated, all_banks, strict=True)
    )
    assets = FormationAssets(
        PARAMETERS,
        4,
        4 * PARAMETERS,
        len(teachers.adapters) * PARAMETERS,
        len(teachers.evidence) * 10,
        tuple((name, sum(len(row) for row in generator.weights)) for name, generator in generators.items()),
        sum(len(row) for row in marginal.weights),
        sum(len(bank) for cohort in all_banks for bank in cohort) * PARAMETERS,
        len(CONTROL_METHODS) - 1,
        sum(p.sample_count for p in provenance) * CHANNELS * 2,
    )
    clean_valid = all(not m.acted and m.admitted_audit_mse == 0.0 for row in reports[1].lineages for m in row.methods)
    analytic_valid = all(
        next(m for m in row.methods if m.name == "analytic4x4").raw_audit_mse < REPRESENTABILITY_TOLERANCE
        for row in reports[0].lineages
    )
    finite_valid = all(
        math.isfinite(m.raw_audit_mse) and math.isfinite(m.admitted_audit_mse)
        for cohort in reports
        for row in cohort.lineages
        for m in row.methods
    )
    report = FormationControlReport(
        "sultai-formation-controls-v1",
        phase,
        FormationControlConfig(
            FIXED.teacher_ridge,
            FIXED.meta_ridge,
            FIXED.admission_improvement,
            AUDIT_EPSILON,
            NOISE_SIGMA,
            "96: per-channel first/second moments of h, tanh(h), supplied residual; no cross-products",
            "empirical four-template 4x4 Gram plus ridge; no intercept; per-episode fit",
            "integer SHA256('sultai-formation-random-v1:'+lineage); candidate partition; Gaussian four-template direction",
            "integer SHA256('sultai-noisy-healthy-v1:'+lineage+':'+partition); independent conditioning/selection streams",
            "one shared procedural family; privileged oracle residuals; finite-fixture audit labels; no calibrated safety",
        ),
        provenance,
        reports,
        offline_cost,
        _total_cost((offline_cost, *tuple(c.cost for c in reports))),
        assets,
        clean_valid,
        analytic_valid,
        finite_valid,
        clean_valid and analytic_valid and finite_valid,
    )
    return report, generators["paired"]


def run_formation_controls(phase: Phase) -> tuple[FormationControlReport, FrozenGenerator]:
    """Construct only training and the named phase's predeclared cohorts."""
    if phase not in ("development", "confirmation"):
        raise ContractViolation("phase must be development or confirmation")
    evaluation_root, clean_root, noisy_root = (4000, 14000, 24000) if phase == "development" else (5000, 15000, 25000)
    training = tuple(make_episode(seed) for seed in FIXED.train_seeds)
    evaluation = tuple(make_episode(seed) for seed in range(evaluation_root, evaluation_root + 8))
    clean = tuple(make_episode(seed, healthy=True) for seed in range(clean_root, clean_root + 8))
    noisy = tuple(make_noisy_healthy_episode(seed) for seed in range(noisy_root, noisy_root + 8))
    return evaluate_formation_controls(training, evaluation, clean, noisy, phase=phase)
