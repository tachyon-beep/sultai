"""Explicit conservative projection of every frozen confirmatory cost-ledger line."""

from __future__ import annotations

from .contracts import PilotHold, Record, json_value, number


def project_cost(
    count: int,
    *,
    timing: tuple[Record, ...],
    intact_seconds: tuple[float, ...],
    bottleneck_seconds: tuple[float, ...],
    oracle_worker_seconds: tuple[float, ...],
    posthoc_seconds: tuple[float, ...],
) -> Record:
    """Use observed maxima, with explicit workload multipliers; never omit an arm.

    These are projections from pilot kernels, not measurements of confirmation.
    Whole-worker oracle and posthoc timings deliberately duplicate some work.
    A retrieval pair includes both alignments, moments and transport. Charging
    every ordered pair covers nearest-other search independently of its winner.
    """
    if (
        count < 32
        or not timing
        or any(not values for values in (intact_seconds, bottleneck_seconds, oracle_worker_seconds, posthoc_seconds))
    ):
        raise PilotHold("complete measured cost inputs and required anchor count needed")
    if any(t["fit_finite"] is not True or t["wide_grid_finite"] is not True for t in timing):
        raise PilotHold("cost timing lacks a finite reference fit")
    measured = {
        "intact_host_seconds": max(intact_seconds),
        "bottleneck_host_seconds": max(bottleneck_seconds),
        "oracle_case_seconds": max((*oracle_worker_seconds, *posthoc_seconds)),
        "full_batch_step_seconds": max(number(t["full_batch_step_seconds"]) for t in timing),
        "wide_grid_seconds": max(number(t["wide_grid_seconds"]) for t in timing),
        "retrieval_pair_seconds": max(number(t["retrieval_pair_seconds"]) for t in timing),
        "posthoc_anchor_seconds": max(posthoc_seconds),
        "development_evaluation_seconds": max(number(t["development_evaluation_seconds"]) for t in timing),
    }
    if any(number(value) <= 0 for value in measured.values()):
        raise PilotHold("positive finite measured cost required for every line")
    cases = 2 * count + 64  # P5/P6 on every bottleneck; P8 on all intact hosts.
    oracle_cases = cases + 64  # P1 alpha-low on every intact host as well.
    full_steps = 5 * (1 + 4 + 16 + 64) + (1 + 4 + 16)
    pairs = 2 * count * (count - 1) + 64 * 63
    # Includes every raw candidate for harm reporting, all posthoc cases and B5.
    evaluations = cases * 50 + 64 * 54 + 16
    lines: Record = {
        "intact_training": 64 * measured["intact_host_seconds"],
        "bottleneck_training": count * measured["bottleneck_host_seconds"],
        "B3_P1_P5_P6_P8": oracle_cases * measured["oracle_case_seconds"],
        "B1_B2_B2prime_B6": cases * full_steps * measured["full_batch_step_seconds"],
        "B5_eight_P6_anchors": 8 * measured["wide_grid_seconds"],
        "A1_A2_A3_A7_A8": 64 * measured["posthoc_anchor_seconds"],
        "B4_all_ordered_other_anchor_pairs": pairs * measured["retrieval_pair_seconds"],
        "all_candidate_development_evaluations": evaluations * measured["development_evaluation_seconds"],
    }
    total = sum(number(value) for value in lines.values())
    return {
        "method": "sum of explicit frozen workloads times observed pilot maxima; overlapping whole-worker charges retained",
        "measured_unit_seconds": json_value(measured),
        "line_gpu_process_seconds": lines,
        "headroom_host_cases": cases,
        "oracle_host_cases": oracle_cases,
        "full_batch_steps_per_headroom_case": full_steps,
        "ordered_retrieval_pairs": pairs,
        "development_evaluations": evaluations,
        "projected_gpu_process_seconds": total,
        "is_confirmatory_measurement": False,
    }
