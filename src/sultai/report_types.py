"""Owned report schemas and fail-closed parsers at the evidence publication seam."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from typing import TypedDict, TypeVar

T = TypeVar("T")

METHODS = (
    "no_op",
    "coarse",
    "paired",
    "paired_probes",
    "full_ridge",
    "affine_ridge",
    "sgd_1",
    "sgd_2",
    "sgd_4",
    "retrieval",
    "interpolation_3",
    "shuffled_conditioning",
    "mismatched_pairing",
)
ARMS = ("no_growth", "cold_zero_taper", "random_taper", "early_static_retained", "early_static_taper", "formed_taper")
STAGES = ("pre_add", "immediate_add", "mid_taper", "pre_remove", "immediate_remove", "post_learning")

SOURCE_FILES = (
    "src/sultai/__init__.py",
    "src/sultai/repair.py",
    "src/sultai/formation.py",
    "src/sultai/lifecycle.py",
    "src/sultai/hybrid.py",
    "src/sultai/contracts.py",
    "src/sultai/report_types.py",
    "src/sultai/smoke.py",
)


def object_map(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("expected string object keys")
        result[key] = item
    return result


def required(value: object, keys: tuple[str, ...]) -> dict[str, object]:
    result = object_map(value)
    if set(result) != set(keys):
        raise ValueError(
            f"required fields differ: missing={set(keys) - set(result)}, unknown={set(result) - set(keys)}"
        )
    return result


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("expected a finite real number") from error
    if not math.isfinite(result):
        raise ValueError("expected a finite real number")
    return result


def loss(value: object) -> float:
    result = number(value)
    if result < 0:
        raise ValueError("expected a nonnegative MSE")
    return result


def count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected a nonnegative integer count")
    return value


def boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("expected an explicit boolean")
    return value


def string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected a nonempty string")
    return value


def nullable_string(value: object) -> str | None:
    return None if value is None else string(value)


def sequence(value: object, parse: Callable[[object], T]) -> list[T]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected an array")
    return [parse(item) for item in value]


def named(value: object, parse: Callable[[object], T], keys: tuple[str, ...] | None = None) -> dict[str, T]:
    data = object_map(value) if keys is None else required(value, keys)
    return {key: parse(item) for key, item in data.items()}


class CostLedger(TypedDict):
    oracle_conditioning_queries: int
    identity_probe_queries: int
    teacher_fits: int
    teacher_fit_examples: int
    meta_regression_fits: int
    meta_training_rows: int
    full_ridge_fits: int
    full_ridge_fit_examples: int
    sgd_steps: int
    sgd_example_steps: int
    generator_forwards: int
    retrieval_distance_evaluations: int
    selection_queries: int
    final_audit_queries: int
    affine_ridge_fits: int
    affine_ridge_fit_examples: int


class MethodMetrics(TypedDict):
    candidate_count: int
    raw_selection_mse: float
    no_op_selection_mse: float
    raw_test_mse: float
    admitted_test_mse: float
    harm_mse_delta: float
    acted: bool
    realized_harm: bool
    costs: CostLedger


class MethodSummary(TypedDict):
    raw_mean_mse: float
    admitted_mean_mse: float
    acted_lineages: int
    realized_harms: int


class LineageReport(TypedDict):
    host_lineage: str
    family: str
    shuffled_conditioning_donor: str
    methods: dict[str, MethodMetrics]
    costs: CostLedger


class PhaseReport(TypedDict):
    lineages: list[LineageReport]
    summary: dict[str, MethodSummary]
    costs: CostLedger


class PartitionIdentity(TypedDict):
    lineages: tuple[str, ...]
    sample_count: int
    sample_ids_sha256: str


class SplitIdentity(TypedDict):
    train: PartitionIdentity
    development: PartitionIdentity
    test: PartitionIdentity


class LineageCounts(TypedDict):
    train: int
    development: int
    test: int


class EvidenceModeCounts(TypedDict):
    coarse: int
    paired: int
    paired_probes: int


class FormationConfig(TypedDict):
    train_seeds: tuple[int, ...]
    dev_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    conditioning_count: int
    selection_count: int
    test_count: int
    probe_count: int
    shuffled_lineage_rotation: int
    mismatched_rotation: int
    healthy_control_lineages: int
    adapter_stored_parameters: int
    effective_output_dimension: int
    retrieval_retained_teacher_parameters: int
    public_template_materialized_values: int
    coefficient_bound: float
    input_bound: float
    teacher_ridge: float
    meta_ridge: float
    admission_improvement: float
    sgd_learning_rate: float
    full_ridge_representability_tolerance: float
    shared_family: str
    paired_features: str
    probe_bank: str
    retrieval: str
    interpolation: str
    affine_control: str
    shuffle: str
    mispair: str
    meta_intercept: str
    representability_gate_provenance: str
    hyperparameter_selection: str
    candidate_bank: str
    lineages: LineageCounts
    generator_parameters: EvidenceModeCounts
    evidence_dimensions: EvidenceModeCounts
    template_squared_parameter_norms: list[int]
    template_definitions: list[str]
    coarse_features: list[str]


class FormationCosts(TypedDict):
    offline: CostLedger
    development: CostLedger
    test: CostLedger
    healthy: CostLedger
    total: CostLedger
    units: str
    selection_baseline: str
    audit: str
    meta_regression: str
    teachers: str


class FormationSummary(TypedDict):
    paired_raw_test_mean_mse: float
    paired_admitted_test_mean_mse: float
    no_op_test_mean_mse: float
    full_ridge_test_mean_mse: float
    affine_ridge_test_mean_mse: float
    paired_acted_lineages: int
    paired_realized_harms: int


class FormationGates(TypedDict):
    paired_reduces_test_mean_mse: bool
    full_ridge_representability: bool
    healthy_all_no_op: bool
    lineage_and_sample_isolation: bool
    finite_metrics: bool


class FormationReport(TypedDict):
    assay: str
    config: FormationConfig
    development: PhaseReport
    test: PhaseReport
    healthy: PhaseReport
    costs: FormationCosts
    acceptance: FormationGates
    summary: FormationSummary
    limitations: list[str]
    isolation: SplitIdentity


class ParameterState(TypedDict):
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]


class StageState(TypedDict):
    completed_host_updates: int
    alpha: float
    allocated_parameters: int


class ParameterCounts(TypedDict):
    before: int
    at_step_zero: int
    peak: int
    final: int


class TrainingArm(TypedDict):
    counts: ParameterCounts
    serialized_parameter_count: int
    host_updates: int
    auxiliary_gradient_updates: int
    training_examples: int
    training_example_ids_sha256: str
    training_pairs_sha256: str
    initial_host_sha256: str
    host_unchanged_at_insertion: bool
    formation_conditioning_sha256: str | None
    formation_examples: int
    additional_module_training: str
    stage_state: dict[str, StageState]
    final_serialization_keys: list[str]
    final_serialization_sha256: str
    adapter_absent: bool
    optimizer_state_empty: bool
    retained_budget_matches_host: bool


class WithdrawalReport(TypedDict, total=False):
    frozen_host_withdrawal_mse: dict[str, float]


class ArmReport(WithdrawalReport):
    counts: ParameterCounts
    serialized_parameter_count: int
    host_updates: int
    auxiliary_gradient_updates: int
    training_examples: int
    training_example_ids_sha256: str
    training_pairs_sha256: str
    initial_host_sha256: str
    host_unchanged_at_insertion: bool
    formation_conditioning_sha256: str | None
    formation_examples: int
    additional_module_training: str
    stage_state: dict[str, StageState]
    final_serialization_keys: list[str]
    final_serialization_sha256: str
    adapter_absent: bool
    optimizer_state_empty: bool
    retained_budget_matches_host: bool
    mse: dict[str, float]
    post_removal_target_met: bool


class LifecycleConfig(TypedDict):
    initial_steps: int
    hold_steps: int
    taper_steps: int
    removal_step: int
    post_steps: int
    total_host_updates: int
    batch_size: int
    training_seed: int
    conditioning_seed: int
    audit_seed: int
    random_adapter_seed: int
    conditioning_examples: int
    audit_examples: int
    learning_rate: float
    post_removal_mse_target: float
    loss: str
    task_coefficients: tuple[float, ...]


class LifecycleCost(TypedDict):
    host_gradient_steps: int
    auxiliary_gradient_steps: int
    task_example_gradient_uses: int
    formation_calls: int
    formation_conditioning_examples: int
    selection_queries: int
    final_audit_queries: int
    fixed_candidate_count: int


class LifecycleSummary(TypedDict):
    post_learning_mse: dict[str, float]
    formed_immediate_add_mse: float
    formed_immediate_remove_mse: float
    formed_peak_parameters: int
    formed_final_parameters: int
    retained_static_final_parameters: int


class LifecycleGates(TypedDict):
    formed_post_removal_target_met: bool
    all_final_removal_arms_absent: bool
    insertion_preserves_host_weights: bool
    serialized_counts_match_allocations: bool
    no_adapter_optimizer_state: bool
    common_future_examples: bool
    common_host_update_count: bool


class LifecycleReport(TypedDict):
    assay: str
    formation_label: str
    insertion_policy: str
    config: LifecycleConfig
    arms: dict[str, ArmReport]
    cost: LifecycleCost
    acceptance: LifecycleGates
    summary: LifecycleSummary
    limits: list[str]


class AffineDiagnostic(TypedDict):
    max_absolute_error: float
    identity_verified: bool
    folded_linear_map: tuple[tuple[float, ...], ...]
    adapter_bias: tuple[float, ...]
    interpretation: str
    convolution_caveat: str


class SignedDiagnostic(TypedDict):
    epsilon: float
    a_plus: float
    b_minus: float
    a_plus_change: float
    b_plus_change: float
    sum_same_sign_changes: float
    passive_evidence_a: tuple[float, ...]
    passive_evidence_b: tuple[float, ...]
    passive_identical: bool
    signed_probe_identity_verified: bool
    finite_changes_are_opposite: bool
    interpretation: str
    limit: str


class Diagnostics(TypedDict):
    affine_folding: AffineDiagnostic
    signed_probe: SignedDiagnostic


class SourceIdentity(TypedDict):
    files: dict[str, str]
    aggregate_sha256: str


class RuntimeIdentity(TypedDict):
    python: str
    implementation: str


class Claims(TypedDict):
    scope: str
    diffusion: bool
    unseen_family_transfer: bool
    population_safety_guarantee: bool
    learned_removal_policy: bool
    half_parameter_goal_demonstrated: bool
    near_empty_host: bool


class HybridReport(TypedDict):
    protocol: str
    source_identity: SourceIdentity
    runtime: RuntimeIdentity
    acceptance: dict[str, bool]
    passed: bool
    formation: FormationReport
    lifecycle: LifecycleReport
    diagnostics: Diagnostics
    claims: Claims


class HybridSummary(TypedDict):
    protocol: str
    source_identity: SourceIdentity
    runtime: RuntimeIdentity
    acceptance: dict[str, bool]
    passed: bool
    claims: Claims
    formation_summary: FormationSummary
    lifecycle_summary: LifecycleSummary


class SmokeEpisodeReport(TypedDict):
    examples: dict[str, int]
    conditioning_feature_rank: int
    feature_columns: int
    learned_adapter_parameters: int
    retained_adapter_parameters: int
    candidate_bank_parameters_before_selection: int
    host_learned_parameters: int
    retained_optimizer_state_values: int
    fit_example_uses: int
    ridge_solves: int
    candidate_index: int
    selection_queries: int
    selection_scalar_predictions: int
    test_queries: int
    test_scalar_predictions: int
    alpha_control: float
    method: str
    ridge_candidates: list[float]
    selection_losses: list[float]
    test_mse: dict[str, float]


class SmokeReport(TypedDict):
    instrument: str
    evidence_status: str
    positive_planted: SmokeEpisodeReport
    healthy_no_repair_control: SmokeEpisodeReport


def float_tuple(value: object) -> tuple[float, ...]:
    return tuple(sequence(value, number))


def parse_CostLedger(value: object) -> CostLedger:
    data = required(
        value,
        (
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
        ),
    )
    result: CostLedger = {
        "oracle_conditioning_queries": count(data["oracle_conditioning_queries"]),
        "identity_probe_queries": count(data["identity_probe_queries"]),
        "teacher_fits": count(data["teacher_fits"]),
        "teacher_fit_examples": count(data["teacher_fit_examples"]),
        "meta_regression_fits": count(data["meta_regression_fits"]),
        "meta_training_rows": count(data["meta_training_rows"]),
        "full_ridge_fits": count(data["full_ridge_fits"]),
        "full_ridge_fit_examples": count(data["full_ridge_fit_examples"]),
        "sgd_steps": count(data["sgd_steps"]),
        "sgd_example_steps": count(data["sgd_example_steps"]),
        "generator_forwards": count(data["generator_forwards"]),
        "retrieval_distance_evaluations": count(data["retrieval_distance_evaluations"]),
        "selection_queries": count(data["selection_queries"]),
        "final_audit_queries": count(data["final_audit_queries"]),
        "affine_ridge_fits": count(data["affine_ridge_fits"]),
        "affine_ridge_fit_examples": count(data["affine_ridge_fit_examples"]),
    }
    return result


def parse_MethodMetrics(value: object) -> MethodMetrics:
    data = required(
        value,
        (
            "candidate_count",
            "raw_selection_mse",
            "no_op_selection_mse",
            "raw_test_mse",
            "admitted_test_mse",
            "harm_mse_delta",
            "acted",
            "realized_harm",
            "costs",
        ),
    )
    result: MethodMetrics = {
        "candidate_count": count(data["candidate_count"]),
        "raw_selection_mse": loss(data["raw_selection_mse"]),
        "no_op_selection_mse": loss(data["no_op_selection_mse"]),
        "raw_test_mse": loss(data["raw_test_mse"]),
        "admitted_test_mse": loss(data["admitted_test_mse"]),
        "harm_mse_delta": number(data["harm_mse_delta"]),
        "acted": boolean(data["acted"]),
        "realized_harm": boolean(data["realized_harm"]),
        "costs": parse_CostLedger(data["costs"]),
    }
    return result


def parse_MethodSummary(value: object) -> MethodSummary:
    data = required(value, ("raw_mean_mse", "admitted_mean_mse", "acted_lineages", "realized_harms"))
    result: MethodSummary = {
        "raw_mean_mse": loss(data["raw_mean_mse"]),
        "admitted_mean_mse": loss(data["admitted_mean_mse"]),
        "acted_lineages": count(data["acted_lineages"]),
        "realized_harms": count(data["realized_harms"]),
    }
    return result


def parse_LineageReport(value: object) -> LineageReport:
    data = required(value, ("host_lineage", "family", "shuffled_conditioning_donor", "methods", "costs"))
    result: LineageReport = {
        "host_lineage": string(data["host_lineage"]),
        "family": string(data["family"]),
        "shuffled_conditioning_donor": string(data["shuffled_conditioning_donor"]),
        "methods": named(data["methods"], parse_MethodMetrics),
        "costs": parse_CostLedger(data["costs"]),
    }
    return result


def parse_PhaseReport(value: object) -> PhaseReport:
    data = required(value, ("lineages", "summary", "costs"))
    result: PhaseReport = {
        "lineages": sequence(data["lineages"], parse_LineageReport),
        "summary": named(data["summary"], parse_MethodSummary),
        "costs": parse_CostLedger(data["costs"]),
    }
    if not result["lineages"]:
        raise ValueError("phase requires nonempty lineage evidence")
    required(result["summary"], METHODS)
    for row in result["lineages"]:
        required(row["methods"], METHODS)
    return result


def parse_PartitionIdentity(value: object) -> PartitionIdentity:
    data = required(value, ("lineages", "sample_count", "sample_ids_sha256"))
    result: PartitionIdentity = {
        "lineages": tuple(sequence(data["lineages"], string)),
        "sample_count": count(data["sample_count"]),
        "sample_ids_sha256": string(data["sample_ids_sha256"]),
    }
    return result


def parse_SplitIdentity(value: object) -> SplitIdentity:
    data = required(value, ("train", "development", "test"))
    result: SplitIdentity = {
        "train": parse_PartitionIdentity(data["train"]),
        "development": parse_PartitionIdentity(data["development"]),
        "test": parse_PartitionIdentity(data["test"]),
    }
    return result


def parse_LineageCounts(value: object) -> LineageCounts:
    data = required(value, ("train", "development", "test"))
    result: LineageCounts = {
        "train": count(data["train"]),
        "development": count(data["development"]),
        "test": count(data["test"]),
    }
    if result != {"train": 32, "development": 8, "test": 8}:
        raise ValueError("fixed lineage counts differ")
    return result


def parse_EvidenceModeCounts(value: object) -> EvidenceModeCounts:
    data = required(value, ("coarse", "paired", "paired_probes"))
    return {
        "coarse": count(data["coarse"]),
        "paired": count(data["paired"]),
        "paired_probes": count(data["paired_probes"]),
    }


def parse_FormationConfig(value: object) -> FormationConfig:
    data = required(
        value,
        (
            "train_seeds",
            "dev_seeds",
            "test_seeds",
            "conditioning_count",
            "selection_count",
            "test_count",
            "probe_count",
            "shuffled_lineage_rotation",
            "mismatched_rotation",
            "healthy_control_lineages",
            "adapter_stored_parameters",
            "effective_output_dimension",
            "retrieval_retained_teacher_parameters",
            "public_template_materialized_values",
            "coefficient_bound",
            "input_bound",
            "teacher_ridge",
            "meta_ridge",
            "admission_improvement",
            "sgd_learning_rate",
            "full_ridge_representability_tolerance",
            "shared_family",
            "paired_features",
            "probe_bank",
            "retrieval",
            "interpolation",
            "affine_control",
            "shuffle",
            "mispair",
            "meta_intercept",
            "representability_gate_provenance",
            "hyperparameter_selection",
            "candidate_bank",
            "lineages",
            "generator_parameters",
            "evidence_dimensions",
            "template_squared_parameter_norms",
            "template_definitions",
            "coarse_features",
        ),
    )
    result: FormationConfig = {
        "train_seeds": tuple(sequence(data["train_seeds"], count)),
        "dev_seeds": tuple(sequence(data["dev_seeds"], count)),
        "test_seeds": tuple(sequence(data["test_seeds"], count)),
        "conditioning_count": count(data["conditioning_count"]),
        "selection_count": count(data["selection_count"]),
        "test_count": count(data["test_count"]),
        "probe_count": count(data["probe_count"]),
        "shuffled_lineage_rotation": count(data["shuffled_lineage_rotation"]),
        "mismatched_rotation": count(data["mismatched_rotation"]),
        "healthy_control_lineages": count(data["healthy_control_lineages"]),
        "adapter_stored_parameters": count(data["adapter_stored_parameters"]),
        "effective_output_dimension": count(data["effective_output_dimension"]),
        "retrieval_retained_teacher_parameters": count(data["retrieval_retained_teacher_parameters"]),
        "public_template_materialized_values": count(data["public_template_materialized_values"]),
        "coefficient_bound": number(data["coefficient_bound"]),
        "input_bound": number(data["input_bound"]),
        "teacher_ridge": number(data["teacher_ridge"]),
        "meta_ridge": number(data["meta_ridge"]),
        "admission_improvement": number(data["admission_improvement"]),
        "sgd_learning_rate": number(data["sgd_learning_rate"]),
        "full_ridge_representability_tolerance": number(data["full_ridge_representability_tolerance"]),
        "shared_family": string(data["shared_family"]),
        "paired_features": string(data["paired_features"]),
        "probe_bank": string(data["probe_bank"]),
        "retrieval": string(data["retrieval"]),
        "interpolation": string(data["interpolation"]),
        "affine_control": string(data["affine_control"]),
        "shuffle": string(data["shuffle"]),
        "mispair": string(data["mispair"]),
        "meta_intercept": string(data["meta_intercept"]),
        "representability_gate_provenance": string(data["representability_gate_provenance"]),
        "hyperparameter_selection": string(data["hyperparameter_selection"]),
        "candidate_bank": string(data["candidate_bank"]),
        "lineages": parse_LineageCounts(data["lineages"]),
        "generator_parameters": parse_EvidenceModeCounts(data["generator_parameters"]),
        "evidence_dimensions": parse_EvidenceModeCounts(data["evidence_dimensions"]),
        "template_squared_parameter_norms": sequence(data["template_squared_parameter_norms"], count),
        "template_definitions": sequence(data["template_definitions"], string),
        "coarse_features": sequence(data["coarse_features"], string),
    }
    if result["generator_parameters"] != {"coarse": 28, "paired": 44, "paired_probes": 44}:
        raise ValueError("fixed generator parameter counts differ")
    if result["evidence_dimensions"] != {"coarse": 6, "paired": 10, "paired_probes": 10}:
        raise ValueError("fixed evidence dimensions differ")
    return result


def parse_FormationCosts(value: object) -> FormationCosts:
    data = required(
        value,
        (
            "offline",
            "development",
            "test",
            "healthy",
            "total",
            "units",
            "selection_baseline",
            "audit",
            "meta_regression",
            "teachers",
        ),
    )
    result: FormationCosts = {
        "offline": parse_CostLedger(data["offline"]),
        "development": parse_CostLedger(data["development"]),
        "test": parse_CostLedger(data["test"]),
        "healthy": parse_CostLedger(data["healthy"]),
        "total": parse_CostLedger(data["total"]),
        "units": string(data["units"]),
        "selection_baseline": string(data["selection_baseline"]),
        "audit": string(data["audit"]),
        "meta_regression": string(data["meta_regression"]),
        "teachers": string(data["teachers"]),
    }
    return result


def parse_FormationSummary(value: object) -> FormationSummary:
    data = required(
        value,
        (
            "paired_raw_test_mean_mse",
            "paired_admitted_test_mean_mse",
            "no_op_test_mean_mse",
            "full_ridge_test_mean_mse",
            "affine_ridge_test_mean_mse",
            "paired_acted_lineages",
            "paired_realized_harms",
        ),
    )
    result: FormationSummary = {
        "paired_raw_test_mean_mse": loss(data["paired_raw_test_mean_mse"]),
        "paired_admitted_test_mean_mse": loss(data["paired_admitted_test_mean_mse"]),
        "no_op_test_mean_mse": loss(data["no_op_test_mean_mse"]),
        "full_ridge_test_mean_mse": loss(data["full_ridge_test_mean_mse"]),
        "affine_ridge_test_mean_mse": loss(data["affine_ridge_test_mean_mse"]),
        "paired_acted_lineages": count(data["paired_acted_lineages"]),
        "paired_realized_harms": count(data["paired_realized_harms"]),
    }
    return result


def parse_FormationGates(value: object) -> FormationGates:
    data = required(
        value,
        (
            "paired_reduces_test_mean_mse",
            "full_ridge_representability",
            "healthy_all_no_op",
            "lineage_and_sample_isolation",
            "finite_metrics",
        ),
    )
    result: FormationGates = {
        "paired_reduces_test_mean_mse": boolean(data["paired_reduces_test_mean_mse"]),
        "full_ridge_representability": boolean(data["full_ridge_representability"]),
        "healthy_all_no_op": boolean(data["healthy_all_no_op"]),
        "lineage_and_sample_isolation": boolean(data["lineage_and_sample_isolation"]),
        "finite_metrics": boolean(data["finite_metrics"]),
    }
    return result


def parse_FormationReport(value: object) -> FormationReport:
    data = required(
        value,
        (
            "assay",
            "config",
            "development",
            "test",
            "healthy",
            "costs",
            "acceptance",
            "summary",
            "limitations",
            "isolation",
        ),
    )
    result: FormationReport = {
        "assay": string(data["assay"]),
        "config": parse_FormationConfig(data["config"]),
        "development": parse_PhaseReport(data["development"]),
        "test": parse_PhaseReport(data["test"]),
        "healthy": parse_PhaseReport(data["healthy"]),
        "costs": parse_FormationCosts(data["costs"]),
        "acceptance": parse_FormationGates(data["acceptance"]),
        "summary": parse_FormationSummary(data["summary"]),
        "limitations": sequence(data["limitations"], string),
        "isolation": parse_SplitIdentity(data["isolation"]),
    }
    validate_formation_evidence(result)
    return result


def parse_StageState(value: object) -> StageState:
    data = required(value, ("completed_host_updates", "alpha", "allocated_parameters"))
    result: StageState = {
        "completed_host_updates": count(data["completed_host_updates"]),
        "alpha": number(data["alpha"]),
        "allocated_parameters": count(data["allocated_parameters"]),
    }
    return result


def parse_ParameterCounts(value: object) -> ParameterCounts:
    data = required(value, ("before", "at_step_zero", "peak", "final"))
    result: ParameterCounts = {
        "before": count(data["before"]),
        "at_step_zero": count(data["at_step_zero"]),
        "peak": count(data["peak"]),
        "final": count(data["final"]),
    }
    return result


def parse_ArmReport(value: object) -> ArmReport:
    raw = object_map(value)
    has_withdrawal = "frozen_host_withdrawal_mse" in raw
    withdrawal = raw.pop("frozen_host_withdrawal_mse") if has_withdrawal else None
    data = required(
        raw,
        (
            "counts",
            "serialized_parameter_count",
            "host_updates",
            "auxiliary_gradient_updates",
            "training_examples",
            "training_example_ids_sha256",
            "training_pairs_sha256",
            "initial_host_sha256",
            "host_unchanged_at_insertion",
            "formation_conditioning_sha256",
            "formation_examples",
            "additional_module_training",
            "stage_state",
            "final_serialization_keys",
            "final_serialization_sha256",
            "adapter_absent",
            "optimizer_state_empty",
            "retained_budget_matches_host",
            "mse",
            "post_removal_target_met",
        ),
    )
    result: ArmReport = {
        "counts": parse_ParameterCounts(data["counts"]),
        "serialized_parameter_count": count(data["serialized_parameter_count"]),
        "host_updates": count(data["host_updates"]),
        "auxiliary_gradient_updates": count(data["auxiliary_gradient_updates"]),
        "training_examples": count(data["training_examples"]),
        "training_example_ids_sha256": string(data["training_example_ids_sha256"]),
        "training_pairs_sha256": string(data["training_pairs_sha256"]),
        "initial_host_sha256": string(data["initial_host_sha256"]),
        "host_unchanged_at_insertion": boolean(data["host_unchanged_at_insertion"]),
        "formation_conditioning_sha256": nullable_string(data["formation_conditioning_sha256"]),
        "formation_examples": count(data["formation_examples"]),
        "additional_module_training": string(data["additional_module_training"]),
        "stage_state": named(data["stage_state"], parse_StageState),
        "final_serialization_keys": sequence(data["final_serialization_keys"], string),
        "final_serialization_sha256": string(data["final_serialization_sha256"]),
        "adapter_absent": boolean(data["adapter_absent"]),
        "optimizer_state_empty": boolean(data["optimizer_state_empty"]),
        "retained_budget_matches_host": boolean(data["retained_budget_matches_host"]),
        "mse": named(data["mse"], loss),
        "post_removal_target_met": boolean(data["post_removal_target_met"]),
    }
    if has_withdrawal:
        result["frozen_host_withdrawal_mse"] = named(withdrawal, loss, ("with_adapter", "without_adapter"))
    required(result["mse"], STAGES)
    required(result["stage_state"], STAGES)
    return result


def parse_LifecycleConfig(value: object) -> LifecycleConfig:
    data = required(
        value,
        (
            "initial_steps",
            "hold_steps",
            "taper_steps",
            "removal_step",
            "post_steps",
            "total_host_updates",
            "batch_size",
            "training_seed",
            "conditioning_seed",
            "audit_seed",
            "random_adapter_seed",
            "conditioning_examples",
            "audit_examples",
            "learning_rate",
            "post_removal_mse_target",
            "loss",
            "task_coefficients",
        ),
    )
    result: LifecycleConfig = {
        "initial_steps": count(data["initial_steps"]),
        "hold_steps": count(data["hold_steps"]),
        "taper_steps": count(data["taper_steps"]),
        "removal_step": count(data["removal_step"]),
        "post_steps": count(data["post_steps"]),
        "total_host_updates": count(data["total_host_updates"]),
        "batch_size": count(data["batch_size"]),
        "training_seed": count(data["training_seed"]),
        "conditioning_seed": count(data["conditioning_seed"]),
        "audit_seed": count(data["audit_seed"]),
        "random_adapter_seed": count(data["random_adapter_seed"]),
        "conditioning_examples": count(data["conditioning_examples"]),
        "audit_examples": count(data["audit_examples"]),
        "learning_rate": number(data["learning_rate"]),
        "post_removal_mse_target": number(data["post_removal_mse_target"]),
        "loss": string(data["loss"]),
        "task_coefficients": tuple(sequence(data["task_coefficients"], number)),
    }
    return result


def parse_LifecycleCost(value: object) -> LifecycleCost:
    data = required(
        value,
        (
            "host_gradient_steps",
            "auxiliary_gradient_steps",
            "task_example_gradient_uses",
            "formation_calls",
            "formation_conditioning_examples",
            "selection_queries",
            "final_audit_queries",
            "fixed_candidate_count",
        ),
    )
    result: LifecycleCost = {
        "host_gradient_steps": count(data["host_gradient_steps"]),
        "auxiliary_gradient_steps": count(data["auxiliary_gradient_steps"]),
        "task_example_gradient_uses": count(data["task_example_gradient_uses"]),
        "formation_calls": count(data["formation_calls"]),
        "formation_conditioning_examples": count(data["formation_conditioning_examples"]),
        "selection_queries": count(data["selection_queries"]),
        "final_audit_queries": count(data["final_audit_queries"]),
        "fixed_candidate_count": count(data["fixed_candidate_count"]),
    }
    return result


def parse_LifecycleSummary(value: object) -> LifecycleSummary:
    data = required(
        value,
        (
            "post_learning_mse",
            "formed_immediate_add_mse",
            "formed_immediate_remove_mse",
            "formed_peak_parameters",
            "formed_final_parameters",
            "retained_static_final_parameters",
        ),
    )
    result: LifecycleSummary = {
        "post_learning_mse": named(data["post_learning_mse"], loss),
        "formed_immediate_add_mse": loss(data["formed_immediate_add_mse"]),
        "formed_immediate_remove_mse": loss(data["formed_immediate_remove_mse"]),
        "formed_peak_parameters": count(data["formed_peak_parameters"]),
        "formed_final_parameters": count(data["formed_final_parameters"]),
        "retained_static_final_parameters": count(data["retained_static_final_parameters"]),
    }
    return result


def parse_LifecycleGates(value: object) -> LifecycleGates:
    data = required(
        value,
        (
            "formed_post_removal_target_met",
            "all_final_removal_arms_absent",
            "insertion_preserves_host_weights",
            "serialized_counts_match_allocations",
            "no_adapter_optimizer_state",
            "common_future_examples",
            "common_host_update_count",
        ),
    )
    result: LifecycleGates = {
        "formed_post_removal_target_met": boolean(data["formed_post_removal_target_met"]),
        "all_final_removal_arms_absent": boolean(data["all_final_removal_arms_absent"]),
        "insertion_preserves_host_weights": boolean(data["insertion_preserves_host_weights"]),
        "serialized_counts_match_allocations": boolean(data["serialized_counts_match_allocations"]),
        "no_adapter_optimizer_state": boolean(data["no_adapter_optimizer_state"]),
        "common_future_examples": boolean(data["common_future_examples"]),
        "common_host_update_count": boolean(data["common_host_update_count"]),
    }
    return result


def parse_LifecycleReport(value: object) -> LifecycleReport:
    data = required(
        value,
        ("assay", "formation_label", "insertion_policy", "config", "arms", "cost", "acceptance", "summary", "limits"),
    )
    result: LifecycleReport = {
        "assay": string(data["assay"]),
        "formation_label": string(data["formation_label"]),
        "insertion_policy": string(data["insertion_policy"]),
        "config": parse_LifecycleConfig(data["config"]),
        "arms": named(data["arms"], parse_ArmReport),
        "cost": parse_LifecycleCost(data["cost"]),
        "acceptance": parse_LifecycleGates(data["acceptance"]),
        "summary": parse_LifecycleSummary(data["summary"]),
        "limits": sequence(data["limits"], string),
    }
    required(result["arms"], ARMS)
    required(result["summary"]["post_learning_mse"], ARMS)
    for name, arm in result["arms"].items():
        if ("frozen_host_withdrawal_mse" in arm) != (name == "formed_taper"):
            raise ValueError("withdrawal evidence is required exactly for the formed arm")
        if (arm["formation_conditioning_sha256"] is not None) != (name == "formed_taper"):
            raise ValueError("formation provenance is required exactly for the formed arm")
    return result


def parse_AffineDiagnostic(value: object) -> AffineDiagnostic:
    data = required(
        value,
        (
            "max_absolute_error",
            "identity_verified",
            "folded_linear_map",
            "adapter_bias",
            "interpretation",
            "convolution_caveat",
        ),
    )
    result: AffineDiagnostic = {
        "max_absolute_error": number(data["max_absolute_error"]),
        "identity_verified": boolean(data["identity_verified"]),
        "folded_linear_map": tuple(sequence(data["folded_linear_map"], float_tuple)),
        "adapter_bias": tuple(sequence(data["adapter_bias"], number)),
        "interpretation": string(data["interpretation"]),
        "convolution_caveat": string(data["convolution_caveat"]),
    }
    if (
        len(result["folded_linear_map"]) != 2
        or any(len(row) != 2 for row in result["folded_linear_map"])
        or len(result["adapter_bias"]) != 2
    ):
        raise ValueError("affine diagnostic requires a 2 by 2 map and two biases")
    return result


def parse_SignedDiagnostic(value: object) -> SignedDiagnostic:
    data = required(
        value,
        (
            "epsilon",
            "a_plus",
            "b_minus",
            "a_plus_change",
            "b_plus_change",
            "sum_same_sign_changes",
            "passive_evidence_a",
            "passive_evidence_b",
            "passive_identical",
            "signed_probe_identity_verified",
            "finite_changes_are_opposite",
            "interpretation",
            "limit",
        ),
    )
    result: SignedDiagnostic = {
        "epsilon": number(data["epsilon"]),
        "a_plus": number(data["a_plus"]),
        "b_minus": number(data["b_minus"]),
        "a_plus_change": number(data["a_plus_change"]),
        "b_plus_change": number(data["b_plus_change"]),
        "sum_same_sign_changes": number(data["sum_same_sign_changes"]),
        "passive_evidence_a": tuple(sequence(data["passive_evidence_a"], number)),
        "passive_evidence_b": tuple(sequence(data["passive_evidence_b"], number)),
        "passive_identical": boolean(data["passive_identical"]),
        "signed_probe_identity_verified": boolean(data["signed_probe_identity_verified"]),
        "finite_changes_are_opposite": boolean(data["finite_changes_are_opposite"]),
        "interpretation": string(data["interpretation"]),
        "limit": string(data["limit"]),
    }
    if len(result["passive_evidence_a"]) != 3 or len(result["passive_evidence_b"]) != 3:
        raise ValueError("signed diagnostic requires complete passive observations")
    return result


def parse_Diagnostics(value: object) -> Diagnostics:
    data = required(value, ("affine_folding", "signed_probe"))
    result: Diagnostics = {
        "affine_folding": parse_AffineDiagnostic(data["affine_folding"]),
        "signed_probe": parse_SignedDiagnostic(data["signed_probe"]),
    }
    return result


def sha256_digest(value: object) -> str:
    result = string(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError("expected a lowercase SHA-256 digest")
    return result


def parse_SourceIdentity(value: object) -> SourceIdentity:
    data = required(value, ("files", "aggregate_sha256"))
    files = named(data["files"], sha256_digest, SOURCE_FILES)
    aggregate = sha256_digest(data["aggregate_sha256"])
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != aggregate:
        raise ValueError("source identity aggregate differs from file digests")
    return {"files": files, "aggregate_sha256": aggregate}


def parse_RuntimeIdentity(value: object) -> RuntimeIdentity:
    data = required(value, ("python", "implementation"))
    result: RuntimeIdentity = {
        "python": string(data["python"]),
        "implementation": string(data["implementation"]),
    }
    return result


def parse_Claims(value: object) -> Claims:
    data = required(
        value,
        (
            "scope",
            "diffusion",
            "unseen_family_transfer",
            "population_safety_guarantee",
            "learned_removal_policy",
            "half_parameter_goal_demonstrated",
            "near_empty_host",
        ),
    )
    result: Claims = {
        "scope": string(data["scope"]),
        "diffusion": boolean(data["diffusion"]),
        "unseen_family_transfer": boolean(data["unseen_family_transfer"]),
        "population_safety_guarantee": boolean(data["population_safety_guarantee"]),
        "learned_removal_policy": boolean(data["learned_removal_policy"]),
        "half_parameter_goal_demonstrated": boolean(data["half_parameter_goal_demonstrated"]),
        "near_empty_host": boolean(data["near_empty_host"]),
    }
    return result


def parse_HybridReport(value: object) -> HybridReport:
    data = required(
        value,
        (
            "protocol",
            "source_identity",
            "runtime",
            "acceptance",
            "passed",
            "formation",
            "lifecycle",
            "diagnostics",
            "claims",
        ),
    )
    result: HybridReport = {
        "protocol": string(data["protocol"]),
        "source_identity": parse_SourceIdentity(data["source_identity"]),
        "runtime": parse_RuntimeIdentity(data["runtime"]),
        "acceptance": named(data["acceptance"], boolean),
        "passed": boolean(data["passed"]),
        "formation": parse_FormationReport(data["formation"]),
        "lifecycle": parse_LifecycleReport(data["lifecycle"]),
        "diagnostics": parse_Diagnostics(data["diagnostics"]),
        "claims": parse_Claims(data["claims"]),
    }
    expected = (
        tuple(
            "formation." + key
            for key in (
                "paired_reduces_test_mean_mse",
                "full_ridge_representability",
                "healthy_all_no_op",
                "lineage_and_sample_isolation",
                "finite_metrics",
            )
        )
        + tuple(
            "lifecycle." + key
            for key in (
                "formed_post_removal_target_met",
                "all_final_removal_arms_absent",
                "insertion_preserves_host_weights",
                "serialized_counts_match_allocations",
                "no_adapter_optimizer_state",
                "common_future_examples",
                "common_host_update_count",
            )
        )
        + DIAGNOSTIC_GATES
    )
    required(result["acceptance"], expected)
    if result["acceptance"] != combined_gates(
        result["formation"]["acceptance"], result["lifecycle"]["acceptance"], result["diagnostics"]
    ):
        raise ValueError("combined gates disagree with required assay evidence")
    if result["passed"] != all(result["acceptance"].values()):
        raise ValueError("passed flag disagrees with acceptance")
    return result


DIAGNOSTIC_GATES = (
    "diagnostic.affine_folding",
    "diagnostic.passive_equality",
    "diagnostic.cross_host_probe_equality",
    "diagnostic.nonlinear_changes_need_not_be_opposite",
)


def validate_isolation(value: SplitIdentity) -> None:
    lineages: set[str] = set()
    for part, expected in ((value["train"], 32), (value["development"], 8), (value["test"], 8)):
        if len(part["lineages"]) != expected or len(set(part["lineages"])) != expected:
            raise ValueError("required lineage identity counts differ")
        if part["sample_count"] != expected * (64 + 24 + 64):
            raise ValueError("required sample identity counts differ")
        digest = part["sample_ids_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid sample identity digest")
        if lineages.intersection(part["lineages"]):
            raise ValueError("split identity overlap")
        lineages.update(part["lineages"])


def validate_phase(phase: PhaseReport, expected_lineages: tuple[str, ...]) -> None:
    if tuple(row["host_lineage"] for row in phase["lineages"]) != expected_lineages:
        raise ValueError("phase lineage provenance differs")
    for method in METHODS:
        values = [row["methods"][method] for row in phase["lineages"]]
        summary = phase["summary"][method]
        if summary["acted_lineages"] != sum(v["acted"] for v in values) or summary["realized_harms"] != sum(
            v["realized_harm"] for v in values
        ):
            raise ValueError("summary outcome counts differ from lineage evidence")
        if summary["raw_mean_mse"] != sum(v["raw_test_mse"] for v in values) / len(values) or summary[
            "admitted_mean_mse"
        ] != sum(v["admitted_test_mse"] for v in values) / len(values):
            raise ValueError("summary metrics differ from lineage evidence")


def validate_formation_evidence(report: FormationReport) -> None:
    validate_isolation(report["isolation"])
    for name in ("development", "test"):
        validate_phase(report[name], report["isolation"][name]["lineages"])
    validate_phase(report["healthy"], tuple(name + "-healthy" for name in report["isolation"]["test"]["lineages"]))


class OptionalModelState(TypedDict, total=False):
    adapter: ParameterState


class ModelState(OptionalModelState):
    host: ParameterState
    optimizer_state: dict[str, float]


class ArmAudit(TypedDict):
    predictions: dict[str, tuple[tuple[float, ...], ...]]
    withdrawal: dict[str, tuple[tuple[float, ...], ...]] | None


def combined_gates(formation: FormationGates, lifecycle: LifecycleGates, diagnostics: Diagnostics) -> dict[str, bool]:
    result = {
        "diagnostic.affine_folding": diagnostics["affine_folding"]["identity_verified"],
        "diagnostic.passive_equality": diagnostics["signed_probe"]["passive_identical"],
        "diagnostic.cross_host_probe_equality": diagnostics["signed_probe"]["signed_probe_identity_verified"],
        "diagnostic.nonlinear_changes_need_not_be_opposite": not diagnostics["signed_probe"][
            "finite_changes_are_opposite"
        ],
    }
    result.update({"formation." + key: value for key, value in named(formation, boolean).items()})
    result.update({"lifecycle." + key: value for key, value in named(lifecycle, boolean).items()})
    return result
