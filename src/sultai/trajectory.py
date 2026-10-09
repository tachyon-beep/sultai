"""Fixed-window same-feature controls; recovery is descriptive, never rescue.

Training receives audit inputs only. Final targets are consumed after every arm
has completed its fixed task updates. The former is supplied already frozen;
its offline fits and retained assets belong to the calling formation report.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from . import lifecycle
from .formation import materialize
from .repair import CHANNELS, PARAMETERS, Adapter, Example, finite
from .trust import ContractViolation

DEVELOPMENT_ROOTS = (8000, 8010)
CONFIRMATION_ROOTS = (9000, 9010, 9020, 9030)
EPSILON = 1e-12
FROZEN_ARMS = ("frozen_zero_taper", "frozen_negated_taper", "frozen_random_taper")
ARMS = (*lifecycle.ARMS, "no_growth_lr_matched", *FROZEN_ARMS)
COMPARATORS = ("no_growth", "no_growth_lr_matched", "early_static_retained", "early_static_taper", *FROZEN_ARMS)
STEPS = tuple(range(lifecycle.INITIAL_STEPS, lifecycle.REMOVAL_STEP + 1))
Predictions = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class Checkpoint:
    completed_steps: int
    alpha: float
    allocated_parameters: int
    mse: float


@dataclass(frozen=True)
class RateEquivalence:
    completed_steps: int
    max_absolute_error: float


@dataclass(frozen=True)
class ArmCost:
    host_gradient_steps: int
    auxiliary_gradient_steps: int
    training_example_uses: int
    auxiliary_example_gradient_uses: int
    host_parameter_gradient_steps: int
    auxiliary_parameter_gradient_steps: int
    allocated_parameter_steps: int
    audit_queries: int
    conditioning_host_forward_examples: int


@dataclass(frozen=True)
class TrajectoryArm:
    name: str
    checkpoints: tuple[Checkpoint, ...]
    trajectory_mean_mse: float
    pre_insertion_mse: float
    insertion_mse: float
    insertion_improvement: float
    immediate_removal_mse: float
    final_recovery_mse: float
    endpoint_below_descriptive_target: bool
    maximum_trajectory_harm_vs_no_growth: float
    initial_parameters: int
    peak_parameters: int
    final_parameters: int
    serialized_parameter_count: int
    final_serialization_keys: tuple[str, ...]
    final_serialization_sha256: str
    adapter_absent: bool
    optimizer_state_empty: bool
    host_unchanged_at_insertion: bool
    training_pairs_sha256: str
    formation_conditioning_sha256: str | None
    candidate_sha256: str | None
    candidate_squared_norm: float | None
    cost: ArmCost


@dataclass(frozen=True)
class Contrast:
    comparator: str
    improvement: float
    strictly_improved: bool
    maximum_trajectory_harm: float


@dataclass(frozen=True)
class StreamCost:
    host_gradient_steps: int
    auxiliary_gradient_steps: int
    training_example_uses: int
    auxiliary_example_gradient_uses: int
    host_parameter_gradient_steps: int
    auxiliary_parameter_gradient_steps: int
    allocated_parameter_steps: int
    audit_queries: int
    conditioning_host_forward_examples: int
    formation_calls: int
    formation_conditioning_examples: int
    candidate_count: int
    candidate_parameter_values: int
    selection_queries: int
    final_retained_model_parameters: int


@dataclass(frozen=True)
class KnownAnswers:
    rate_equivalence: bool
    zero_matches_no_growth: bool
    actual_removal: bool
    serialized_counts_match: bool
    complete_record_window: bool
    fixed_training_costs: bool
    insertion_preserves_host: bool
    no_optimizer_state: bool


@dataclass(frozen=True)
class StreamReport:
    root: int
    partition_seeds: tuple[int, int, int, int]
    partition_counts: tuple[int, int, int]
    partition_pair_digests: tuple[str, ...]
    partition_id_digests: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    arms: tuple[TrajectoryArm, ...]
    contrasts: tuple[Contrast, ...]
    rate_equivalence: tuple[RateEquivalence, ...]
    rate_equivalence_verified: bool
    diagnostic_specificity: bool
    meaningful_benefit_established: bool
    worthwhile_margin: float | None
    cost: StreamCost
    known_answers: KnownAnswers
    software_valid: bool


@dataclass(frozen=True)
class ContrastSummary:
    comparator: str
    mean_improvement: float
    minimum_improvement: float
    maximum_improvement: float
    positive_streams: int
    neutral_streams: int
    negative_streams: int


@dataclass(frozen=True)
class TrajectoryReport:
    phase: Literal["development", "confirmation"]
    streams: tuple[StreamReport, ...]
    contrast_summary: tuple[ContrastSummary, ...]
    diagnostic_specificity: bool
    meaningful_benefit_established: bool
    worthwhile_margin: float | None
    limits: tuple[str, ...]
    known_answers: KnownAnswers
    software_valid: bool


@dataclass(frozen=True)
class _Record:
    completed_steps: int
    alpha: float
    allocated_parameters: int
    predictions: Predictions


@dataclass(frozen=True)
class _TrainedArm:
    name: str
    records: tuple[_Record, ...]
    pre_insertion: Predictions
    final_predictions: Predictions
    equivalence_predictions: tuple[Predictions, ...]
    initial_parameters: int
    peak_parameters: int
    final_parameters: int
    serialized_parameter_count: int
    final_serialization_keys: tuple[str, ...]
    final_serialization_sha256: str
    adapter_absent: bool
    optimizer_state_empty: bool
    host_unchanged_at_insertion: bool
    formation_conditioning_sha256: str | None
    candidate_sha256: str | None
    candidate_squared_norm: float | None
    cost: ArmCost


def _target() -> Adapter:
    return materialize(lifecycle.TASK_COEFFICIENTS)


def _fixture(root: int) -> tuple[tuple[Example, ...], tuple[Example, ...], tuple[Example, ...]]:
    target = _target()
    return (
        lifecycle._examples(root + 1, f"{root}:training", lifecycle.TOTAL_STEPS * lifecycle.BATCH_SIZE, target),
        lifecycle._examples(root + 2, f"{root}:conditioning", lifecycle.CONDITIONING_COUNT, target),
        lifecycle._examples(root + 3, f"{root}:audit", lifecycle.AUDIT_COUNT, target),
    )


def _pairs(examples: Sequence[Example]) -> str:
    return lifecycle._digest(tuple((e.sample_id, e.h, e.target) for e in examples))


def _norm_squared(adapter: Adapter) -> float:
    return finite(sum(value * value for row in adapter.weights for value in row) + sum(v * v for v in adapter.bias))


def _scaled(adapter: Adapter, factor: float) -> Adapter:
    return Adapter(
        tuple(tuple(factor * value for value in row) for row in adapter.weights),
        tuple(factor * value for value in adapter.bias),
    )


def _random_trainable(root: int) -> Adapter:
    rng = random.Random(root + 4)
    return Adapter(
        tuple(tuple(rng.uniform(-0.025, 0.025) for _ in range(CHANNELS)) for _ in range(CHANNELS)),
        tuple(rng.uniform(-0.025, 0.025) for _ in range(CHANNELS)),
    )


def _replacements(candidate: Adapter, root: int) -> tuple[Adapter, Adapter, Adapter]:
    rng = random.Random(root + 4)
    direction = materialize(tuple(rng.gauss(0.0, 1.0) for _ in range(4)))
    norm = _norm_squared(direction)
    if norm == 0.0:
        raise ContractViolation("random replacement direction has zero norm")
    random_candidate = _scaled(direction, (_norm_squared(candidate) / norm) ** 0.5)
    return Adapter.zero(), _scaled(candidate, -1.0), random_candidate


def _train_arm(
    name: str,
    former: Callable[[Sequence[Example]], Adapter],
    candidate: Adapter | None,
    root: int,
    training: Sequence[Example],
    conditioning: Sequence[Example],
    audit_inputs: Sequence[Sequence[float]],
    conditioning_digest: str | None,
) -> _TrainedArm:
    model = lifecycle._Model()
    if name.startswith("early_static"):
        model.add(Adapter.zero(), trainable=True)
    initial_parameters = peak_parameters = model.parameters
    records: list[_Record] = []
    equivalence: list[Predictions] = []
    pre_insertion: Predictions = ()
    final_predictions: Predictions = ()
    auxiliary_steps = parameter_steps = audit_queries = conditioning_forwards = 0
    unchanged = True
    candidate_digest: str | None = None
    candidate_norm: float | None = None
    rate_arm = name in ("no_growth_lr_matched", "early_static_retained")

    def predict(alpha: float) -> Predictions:
        return tuple(model.predict(h, alpha) for h in audit_inputs)

    for step in range(lifecycle.TOTAL_STEPS + 1):
        influence = 1.0 if name == "early_static_retained" else lifecycle._alpha(step)
        if name in ("no_growth", "no_growth_lr_matched"):
            influence = 0.0
        if step == lifecycle.INITIAL_STEPS:
            pre_insertion = predict(influence)
            audit_queries += len(audit_inputs)
            before = lifecycle._digest(model.host.serialized())
            if name == "formed_taper":
                residual = tuple(
                    Example(
                        e.sample_id,
                        e.h,
                        tuple(
                            h + target - output
                            for h, target, output in zip(e.h, e.target, model.predict(e.h, 0.0), strict=True)
                        ),
                    )
                    for e in conditioning
                )
                conditioning_forwards = len(conditioning)
                conditioning_digest = _pairs(residual)
                candidate = former(residual)
                if not isinstance(candidate, Adapter):
                    raise ContractViolation("former must return a repair.Adapter")
            if name == "formed_taper" or name in FROZEN_ARMS:
                if candidate is None:
                    raise ContractViolation("frozen arm requires a candidate")
                model.add(candidate, trainable=False)
                candidate_digest = lifecycle._digest({"weights": candidate.weights, "bias": candidate.bias})
                candidate_norm = _norm_squared(candidate)
            elif name == "cold_zero_taper":
                model.add(Adapter.zero(), trainable=True)
            elif name == "random_taper":
                model.add(_random_trainable(root), trainable=True)
            unchanged = before == lifecycle._digest(model.host.serialized())
            peak_parameters = max(peak_parameters, model.parameters)
        if step == lifecycle.REMOVAL_STEP and name != "early_static_retained":
            model.remove()
        prediction: Predictions = ()
        if step in STEPS or step == lifecycle.TOTAL_STEPS or rate_arm:
            prediction = predict(influence)
            audit_queries += len(audit_inputs)
        if step in STEPS:
            records.append(_Record(step, influence, model.parameters, prediction))
        if rate_arm:
            equivalence.append(prediction)
        if step == lifecycle.TOTAL_STEPS:
            final_predictions = prediction
            break
        parameter_steps += model.parameters
        batch = training[step * lifecycle.BATCH_SIZE : (step + 1) * lifecycle.BATCH_SIZE]
        updated = lifecycle._sgd_step(
            model, batch, influence, learning_rate=1.6 if name == "no_growth_lr_matched" else 0.8
        )
        auxiliary_steps += updated - 1
    final = model.serialized()
    serialized_parameters = sum(len(row) for row in final["host"]["weights"]) + len(final["host"]["bias"])
    if "adapter" in final:
        serialized_parameters += sum(len(row) for row in final["adapter"]["weights"]) + len(final["adapter"]["bias"])
    return _TrainedArm(
        name,
        tuple(records),
        pre_insertion,
        final_predictions,
        tuple(equivalence),
        initial_parameters,
        peak_parameters,
        model.parameters,
        serialized_parameters,
        tuple(sorted(final)),
        lifecycle._digest(final),
        "adapter" not in final,
        final["optimizer_state"] == {},
        unchanged,
        conditioning_digest,
        candidate_digest,
        candidate_norm,
        ArmCost(
            lifecycle.TOTAL_STEPS,
            auxiliary_steps,
            len(training),
            auxiliary_steps * lifecycle.BATCH_SIZE,
            lifecycle.TOTAL_STEPS * PARAMETERS,
            auxiliary_steps * PARAMETERS,
            parameter_steps,
            audit_queries,
            conditioning_forwards,
        ),
    )


def _arm(arms: Sequence[TrajectoryArm], name: str) -> TrajectoryArm:
    return next(arm for arm in arms if arm.name == name)


def _contrasts(arms: Sequence[TrajectoryArm]) -> tuple[Contrast, ...]:
    formed = _arm(arms, "formed_taper")
    result = []
    for name in COMPARATORS:
        comparator = _arm(arms, name)
        improvement = finite(comparator.trajectory_mean_mse - formed.trajectory_mean_mse)
        maximum_harm = max(
            0.0, *(a.mse - b.mse for a, b in zip(formed.checkpoints, comparator.checkpoints, strict=True))
        )
        result.append(Contrast(name, improvement, improvement > EPSILON, finite(maximum_harm)))
    return tuple(result)


def _specificity(arms: Sequence[TrajectoryArm]) -> bool:
    return _arm(arms, "formed_taper").insertion_improvement > EPSILON and all(
        contrast.strictly_improved for contrast in _contrasts(arms)
    )


def _known_answers(arms: Sequence[TrajectoryArm], equivalence: Sequence[RateEquivalence]) -> KnownAnswers:
    zero, baseline = _arm(arms, "frozen_zero_taper"), _arm(arms, "no_growth")
    return KnownAnswers(
        tuple(p.completed_steps for p in equivalence) == tuple(range(lifecycle.TOTAL_STEPS + 1))
        and all(p.max_absolute_error <= EPSILON for p in equivalence),
        tuple(p.mse for p in zero.checkpoints) == tuple(p.mse for p in baseline.checkpoints)
        and zero.final_serialization_sha256 == baseline.final_serialization_sha256,
        all(
            a.adapter_absent and "adapter" not in a.final_serialization_keys and a.final_parameters == PARAMETERS
            for a in arms
            if a.name != "early_static_retained"
        ),
        all(a.serialized_parameter_count == a.final_parameters for a in arms),
        all(tuple(p.completed_steps for p in a.checkpoints) == STEPS for a in arms),
        all(
            a.cost.host_gradient_steps == lifecycle.TOTAL_STEPS
            and a.cost.training_example_uses == lifecycle.TOTAL_STEPS * lifecycle.BATCH_SIZE
            for a in arms
        ),
        all(a.host_unchanged_at_insertion for a in arms),
        all(a.optimizer_state_empty for a in arms),
    )


def _valid(checks: KnownAnswers) -> bool:
    return (
        checks.rate_equivalence
        and checks.zero_matches_no_growth
        and checks.actual_removal
        and checks.serialized_counts_match
        and checks.complete_record_window
        and checks.fixed_training_costs
        and checks.insertion_preserves_host
        and checks.no_optimizer_state
    )


def _run_stream(
    former: Callable[[Sequence[Example]], Adapter],
    root: int,
    training: Sequence[Example],
    conditioning: Sequence[Example],
    audit: Sequence[Example],
) -> StreamReport:
    if (len(training), len(conditioning), len(audit)) != (
        lifecycle.TOTAL_STEPS * lifecycle.BATCH_SIZE,
        lifecycle.CONDITIONING_COUNT,
        lifecycle.AUDIT_COUNT,
    ):
        raise ContractViolation("fixed trajectory example counts required")
    ids = tuple(e.sample_id for group in (training, conditioning, audit) for e in group)
    if len(ids) != len(set(ids)):
        raise ContractViolation("trajectory splits must have unique, disjoint sample IDs")
    candidates: tuple[Adapter, ...] = ()

    def capture(examples: Sequence[Example]) -> Adapter:
        nonlocal candidates
        formed = former(examples)
        if not isinstance(formed, Adapter):
            raise ContractViolation("former must return a repair.Adapter")
        candidates = (formed, *_replacements(formed, root))
        return formed

    inputs = tuple(e.h for e in audit)
    formed_raw = _train_arm("formed_taper", capture, None, root, training, conditioning, inputs, None)
    replacement_map = dict(zip(FROZEN_ARMS, candidates[1:], strict=True))
    raw_arms = tuple(
        formed_raw
        if name == "formed_taper"
        else _train_arm(
            name,
            former,
            replacement_map[name] if name in replacement_map else None,
            root,
            training,
            conditioning,
            inputs,
            formed_raw.formation_conditioning_sha256 if name in FROZEN_ARMS else None,
        )
        for name in ARMS
    )
    # All task training is complete before reading any audit targets.
    baseline = next(raw for raw in raw_arms if raw.name == "no_growth")
    baseline_mse = tuple(lifecycle._prediction_mse(record.predictions, audit) for record in baseline.records)
    arms = []
    for raw in raw_arms:
        checkpoints = tuple(
            Checkpoint(
                r.completed_steps, r.alpha, r.allocated_parameters, lifecycle._prediction_mse(r.predictions, audit)
            )
            for r in raw.records
        )
        before = lifecycle._prediction_mse(raw.pre_insertion, audit)
        final_mse = lifecycle._prediction_mse(raw.final_predictions, audit)
        harm = max(0.0, *(p.mse - loss for p, loss in zip(checkpoints, baseline_mse, strict=True)))
        arms.append(
            TrajectoryArm(
                raw.name,
                checkpoints,
                finite(sum(p.mse for p in checkpoints) / len(STEPS)),
                before,
                checkpoints[0].mse,
                finite(before - checkpoints[0].mse),
                checkpoints[-1].mse,
                final_mse,
                final_mse <= lifecycle.POST_REMOVAL_MSE_TARGET,
                finite(harm),
                raw.initial_parameters,
                raw.peak_parameters,
                raw.final_parameters,
                raw.serialized_parameter_count,
                raw.final_serialization_keys,
                raw.final_serialization_sha256,
                raw.adapter_absent,
                raw.optimizer_state_empty,
                raw.host_unchanged_at_insertion,
                _pairs(training),
                raw.formation_conditioning_sha256,
                raw.candidate_sha256,
                raw.candidate_squared_norm,
                raw.cost,
            )
        )
    retained = next(raw for raw in raw_arms if raw.name == "early_static_retained")
    matched = next(raw for raw in raw_arms if raw.name == "no_growth_lr_matched")
    equivalence = tuple(
        RateEquivalence(
            step,
            finite(
                max(
                    abs(a - b)
                    for left, right in zip(retained_predictions, matched_predictions, strict=True)
                    for a, b in zip(left, right, strict=True)
                )
            ),
        )
        for step, (retained_predictions, matched_predictions) in enumerate(
            zip(retained.equivalence_predictions, matched.equivalence_predictions, strict=True)
        )
    )
    arm_tuple = tuple(arms)
    costs = tuple(arm.cost for arm in arms)
    cost = StreamCost(
        sum(c.host_gradient_steps for c in costs),
        sum(c.auxiliary_gradient_steps for c in costs),
        sum(c.training_example_uses for c in costs),
        sum(c.auxiliary_example_gradient_uses for c in costs),
        sum(c.host_parameter_gradient_steps for c in costs),
        sum(c.auxiliary_parameter_gradient_steps for c in costs),
        sum(c.allocated_parameter_steps for c in costs),
        sum(c.audit_queries for c in costs),
        sum(c.conditioning_host_forward_examples for c in costs),
        1,
        len(conditioning),
        4,
        4 * PARAMETERS,
        0,
        sum(arm.final_parameters for arm in arms),
    )
    groups = (training, conditioning, audit)
    checks = _known_answers(arm_tuple, equivalence)
    return StreamReport(
        root,
        (root + 1, root + 2, root + 3, root + 4),
        (len(training), len(conditioning), len(audit)),
        tuple(_pairs(group) for group in groups),
        tuple(lifecycle._digest(tuple(e.sample_id for e in group)) for group in groups),
        tuple(f"trajectory-{root}-{name}" for name in ("training", "conditioning", "audit")),
        arm_tuple,
        _contrasts(arm_tuple),
        equivalence,
        all(p.max_absolute_error <= EPSILON for p in equivalence),
        _specificity(arm_tuple),
        False,
        None,
        cost,
        checks,
        _valid(checks),
    )


def run_trajectory(
    former: Callable[[Sequence[Example]], Adapter], phase: Literal["development", "confirmation"]
) -> TrajectoryReport:
    """Run a fixed phase only after caller scientific-review/source-freeze clearance."""
    if phase not in ("development", "confirmation"):
        raise ContractViolation("trajectory phase must be development or confirmation")
    roots = DEVELOPMENT_ROOTS if phase == "development" else CONFIRMATION_ROOTS
    fixtures = tuple(_fixture(root) for root in roots)
    ids = tuple(e.sample_id for fixture in fixtures for group in fixture for e in group)
    if len(ids) != len(set(ids)):
        raise ContractViolation("trajectory phase streams must have disjoint sample IDs")
    streams = tuple(_run_stream(former, root, *fixture) for root, fixture in zip(roots, fixtures, strict=True))
    summaries = []
    for name in COMPARATORS:
        values = tuple(next(c.improvement for c in s.contrasts if c.comparator == name) for s in streams)
        summaries.append(
            ContrastSummary(
                name,
                finite(sum(values) / len(values)),
                min(values),
                max(values),
                sum(v > EPSILON for v in values),
                sum(abs(v) <= EPSILON for v in values),
                sum(v < -EPSILON for v in values),
            )
        )
    checks = KnownAnswers(
        all(s.known_answers.rate_equivalence for s in streams),
        all(s.known_answers.zero_matches_no_growth for s in streams),
        all(s.known_answers.actual_removal for s in streams),
        all(s.known_answers.serialized_counts_match for s in streams),
        all(s.known_answers.complete_record_window for s in streams),
        all(s.known_answers.fixed_training_costs for s in streams),
        all(s.known_answers.insertion_preserves_host for s in streams),
        all(s.known_answers.no_optimizer_state for s in streams),
    )
    return TrajectoryReport(
        phase,
        streams,
        tuple(summaries),
        all(s.diagnostic_specificity for s in streams),
        False,
        None,
        (
            "one procedural same-feature task; streams are independent conditional replicates",
            "strict diagnostic epsilon is numerical discrimination, with no worthwhile benefit margin",
            "endpoint <= .01 and diagnostic specificity do not establish benefit after removal",
            "offline frozen former fitting/search/retained assets are accounted by calling formation report",
            "no independently trained host, unseen-family transfer, calibrated safety, or half-size success",
        ),
        checks,
        _valid(checks),
    )
