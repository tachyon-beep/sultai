"""M0 cost projection (§8, §9 stages 4 and 6, §10) from measured one-per-card workloads, and extended admission.

Every projected line is a count of measured units: no booking from the §8 table appears here as a number. The units
are measured once per card in pilot stages 1-4 (one process per card is what the ceiling counts) and passed in as
`Workloads`. Never imports `sultai_screen.screen` or `sultai_screen.pilot`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from sultai.trust import ContractViolation
from sultai_screen.contracts import Record, json_value
from sultai_screen.m0.identity import (
    CHECKPOINTS,
    M0_CONFIRMATORY_SECONDS,
    M0_CONFIRMATORY_STATIC_SECONDS,
    RALL_CHECKPOINTS,
    RETUNE_RATES,
)
from sultai_screen.m0.stats import CONFIRMATORY_ARMS, KNOWN_ARMS, PILOT_ARMS, STARVED_PAIR_ARMS

DEVELOPMENT_ANCHORS: Final = 6
CONFIRMATORY_MINIMUM: Final = 32
GRID_KINDS: Final = frozenset(("module", "module-B", "aff", "retune", "refit", "refit-bound"))
STATIC_ARMS: Final = frozenset(("S-static", "S-static:refit"))

Arms = tuple[str, ...] | frozenset[str]


@dataclass(frozen=True)
class Workloads:
    """Measured one-per-card process-seconds per unit of work, each the pilot's maximum over its measurements.

    `host` per width scale (one 10-epoch host); `tcool` one T-cool resume; `static` one S-static host trained from
    step zero; `grid` per grid kind: "module" (three-rate checkpointed module grid), "module-B" (M-small-B's doubled
    schedule), "aff" (M-aff five-rate grid with fold and residual), "retune" (cached-site re-tune grid, R-272s),
    "refit" (M-wide→8 refit with the full module grid), "refit-bound" (one rate across the last checkpoint, the
    bound on the h = 16/4/2 refit at M-wide's selected (rate, step) pair, amendment 5); `front_rerun` seconds per 1,000 front-re-running optimiser steps (the first front-re-running arm is
    measured in stage 4 and divided by its steps, R9); `fold_and_calibrations`
    one anchor's N0 and its twin, R-temp and R-cal, cuts, folds, leave-one-out and their evaluations (the §8 30 s
    line); `evaluation` one development evaluation of a new host (from images); `fixed_v0` the fixed V0 stage total;
    `noop_evaluate` per width scale the whole noop child plus the whole evaluate child that follow one trained host
    (ruling R23: a host item is costed as its three whole children, not as the in-process evaluation kernel).
    """

    host: dict[float, float]
    tcool: float
    static: float
    grid: dict[str, float]
    front_rerun: float
    fold_and_calibrations: float
    evaluation: float
    fixed_v0: float
    noop_evaluate: dict[float, float]

    def __post_init__(self) -> None:
        if not self.host or not self.noop_evaluate:
            raise ContractViolation("at least one measured host width and its noop and evaluate children are required")
        if set(self.grid) != GRID_KINDS:
            raise ContractViolation(f"grid workloads must name exactly {sorted(GRID_KINDS)}")
        scalars = (
            self.tcool,
            self.static,
            self.front_rerun,
            self.fold_and_calibrations,
            self.evaluation,
            self.fixed_v0,
        )
        for value in (*scalars, *self.host.values(), *self.grid.values(), *self.noop_evaluate.values()):
            if not math.isfinite(value) or value <= 0:
                raise ContractViolation("every measured workload must be positive and finite")


# Front re-running arms are costed in units of 1,000 optimiser steps (R9): R-aff runs each of the five re-tune rates
# to the last checkpoint (5 x 3,200 = 16 units); R-all runs each rate to 800 steps (§5 item 2: 5 x 800 = 4 units).
FRONT_UNIT_STEPS: Final = 1_000
R_ALL_STEPS: Final = RALL_CHECKPOINTS[-1]
# Ruling R23: R-all is also booked for its from-images evaluations, one per (rate, checkpoint): 5 x 20 = 100 per anchor,
# each at the measured development-evaluation kernel of a new host.
R_ALL_EVALUATIONS: Final = len(RETUNE_RATES) * len(RALL_CHECKPOINTS)
R_AFF_FRONT_UNITS: Final = len(RETUNE_RATES) * CHECKPOINTS[-1] // FRONT_UNIT_STEPS
R_ALL_FRONT_UNITS: Final = len(RETUNE_RATES) * R_ALL_STEPS // FRONT_UNIT_STEPS
if (len(RETUNE_RATES) * CHECKPOINTS[-1]) % FRONT_UNIT_STEPS or (len(RETUNE_RATES) * R_ALL_STEPS) % FRONT_UNIT_STEPS:
    raise ContractViolation("front re-run step counts must be whole units of 1,000 steps")

# The per-anchor work every arm set needs: the anchor's reference host (s = 1.0) and its N0, twin, calibrations,
# cuts, folds and evaluations. R-temp and R-cal, and every sliced or ablated cut, are inside the second unit.
ANCHOR_UNITS: Final = ("host:1.0", "fold_and_calibrations")
ARM_UNITS: Final[dict[str, tuple[str, ...]]] = {
    "R-temp": (),
    "R-cal": (),
    "M-sq": ("grid:module",),
    "M-aff": ("grid:aff",),
    "R-272s": ("grid:retune",),
    "R-aff": ("front_rerun",) * R_AFF_FRONT_UNITS,
    "T-cool": ("tcool",),
    "M-sq|T-cool": ("grid:module",),
    "M-wide": ("grid:module",),
    "M-wide→8:rank:refit": ("grid:refit",),
    "M-wide→8:rank:sliced": (),
    "M-wide→8:random:refit": ("grid:refit",),
    "M-wide→8:random:sliced": (),
    "M-wide→16:sliced": (),
    "M-wide→4:sliced": (),
    "M-wide→2:sliced": (),
    # Amendment 5 (A5.1): each survival refit is one run bounded by the last checkpoint plus one development evaluation.
    "M-wide→16:refit": ("grid:refit-bound", "evaluation"),
    "M-wide→4:refit": ("grid:refit-bound", "evaluation"),
    "M-wide→2:refit": ("grid:refit-bound", "evaluation"),
    "M-wide→8:ablation": (),
    "M-small": ("grid:module",),
    "M-small-B": ("grid:module-B",),
    "M-sq-fresh": ("grid:module",),
    "R-all": ("front_rerun",) * R_ALL_FRONT_UNITS + ("evaluation",) * R_ALL_EVALUATIONS,
    "M-exp": ("grid:module",),
    "lottery": ("grid:refit",),
    "S-static": ("static",),
    "S-static:refit": ("grid:module",),
}
if set(ARM_UNITS) != KNOWN_ARMS:
    raise ContractViolation("every known arm must be costed exactly once")

# §5 extended set, in priority order.
EXTENDED_ORDER: Final = ("starved-pair", "R-all", "M-exp", "lottery", "cooled-20", "F(0.25)", "F(0.75)")
# Ruling R23: extended work is admitted only up to (1 - EXTENDED_RESERVE) x the ceiling, at stage 4 and at run time.
EXTENDED_RESERVE: Final = 0.05
FRONTIER_ITEMS: Final = {"F(0.25)": 0.25, "F(0.75)": 0.75}
# Arms the §9 stage 5 core cases run on each development anchor. T-cool's resume belongs to the hosts stage, and the
# S-static refit is a core module-grid case (§8) although S-static is optional in the confirmatory.
CORE_CASE_ARMS: Final = (PILOT_ARMS - {"T-cool"}) | {"S-static:refit"}


@dataclass(frozen=True)
class Stage:
    """A pilot stage's workload: `units` repeated `anchors` times."""

    anchors: int
    units: tuple[str, ...]


# Pilot stage name -> its measured workload. Extended items (EXTENDED_ORDER) are stages too, costed by extended_cost.
# §9 stages 1-3 are never projected: stage 4 projects from the charge so far (R19), which already holds the replay,
# timing, acceptance and instrument children (the six s = 1.0 reference hosts among them, inside fixed_v0, R17) and
# the stage-3 frontier, T-cool and S-static host children (B6).
STAGE_WORKLOADS: Final[dict[str, Stage]] = {
    # §9 stage 5: every core case on the anchor's already-trained hosts.
    "core": Stage(
        DEVELOPMENT_ANCHORS,
        ("fold_and_calibrations", *(unit for arm in sorted(CORE_CASE_ARMS) for unit in ARM_UNITS[arm])),
    ),
}


def unit_seconds(w: Workloads, unit: str) -> float:
    """The measured seconds of one named unit ("host:<s>", "grid:<kind>" or a scalar field)."""
    kind, _, key = unit.partition(":")
    if kind == "host" and key:
        scale = float(key)
        if scale not in w.host:
            raise ContractViolation(f"no measured host workload at width scale {scale}")
        return w.host[scale]
    if kind == "grid" and key:
        if key not in w.grid:
            raise ContractViolation(f"no measured grid workload {key!r}")
        return w.grid[key]
    scalars = {
        "tcool": w.tcool,
        "static": w.static,
        "front_rerun": w.front_rerun,
        "fold_and_calibrations": w.fold_and_calibrations,
        "evaluation": w.evaluation,
        "fixed_v0": w.fixed_v0,
    }
    if unit not in scalars:
        raise ContractViolation(f"unknown workload unit {unit!r}")
    return scalars[unit]


def arm_seconds(w: Workloads, arms: Arms) -> float:
    """The named arms' measured units only (no anchor host, N0 or calibrations line): what a child running them costs
    beyond the anchor's own work (ruling R30 scales the measured confirmatory term by these)."""
    names = tuple(arms)
    if len(set(names)) != len(names):
        raise ContractViolation("an arm may be named once per anchor")
    unknown = sorted(set(names) - set(ARM_UNITS))
    if unknown:
        raise ContractViolation(f"no measured workload for arms {unknown}")
    return sum(unit_seconds(w, unit) for arm in names for unit in ARM_UNITS[arm])


def per_anchor(w: Workloads, arms: Arms) -> float:
    """One reference-host anchor: its host, N0 and calibrations line, plus each named arm's measured units."""
    return sum(unit_seconds(w, unit) for unit in ANCHOR_UNITS) + arm_seconds(w, arms)


def _noop_evaluate(w: Workloads, scale: float) -> float:
    if scale not in w.noop_evaluate:
        raise ContractViolation(f"no measured noop and evaluate children at width scale {scale}")
    return w.noop_evaluate[scale]


def extended_cost(item: str, w: Workloads) -> float:
    """One §5 extended item from measured units."""
    if item == "starved-pair":
        # Six 0.5 hosts x (N0 and calibrations + T-cool + M-sq grid + M-aff grid + M-sq | T-cool grid). The 0.5 hosts
        # are the core frontier hosts already trained and charged in stage 3, so no host term (R9).
        return DEVELOPMENT_ANCHORS * (unit_seconds(w, "fold_and_calibrations") + arm_seconds(w, STARVED_PAIR_ARMS))
    if item == "R-all":
        # Six anchors x (4 front re-run units (R9) + 100 from-images evaluations (R23)).
        return DEVELOPMENT_ANCHORS * arm_seconds(w, ("R-all",))
    if item in ("M-exp", "lottery"):
        # Six anchors x one grid: M-exp a module grid; the lottery retrain the h = 8 refit grid.
        return DEVELOPMENT_ANCHORS * arm_seconds(w, (item,))
    if item == "cooled-20":
        # Six 20-epoch F(1.0) hosts as whole children (R23): twice the measured 10-epoch train child, plus the noop
        # and evaluate children that follow every trained host.
        return DEVELOPMENT_ANCHORS * (2 * unit_seconds(w, "host:1.0") + _noop_evaluate(w, 1.0))
    if item in FRONTIER_ITEMS:
        # Six F(s) hosts as whole children (R23): train, noop and evaluate. The 0.25 and 0.75 hosts are trained only
        # after admission, so they are costed at the measured 0.5 children (§8: "F(0.25) 210 s (launch-bound, as the
        # 0.5 host); F(0.75) 210 s"); a measured width-s child is used instead when one is present.
        scale = FRONTIER_ITEMS[item]
        host = f"host:{scale}" if scale in w.host else "host:0.5"
        return DEVELOPMENT_ANCHORS * (
            unit_seconds(w, host) + _noop_evaluate(w, scale if scale in w.noop_evaluate else 0.5)
        )
    raise ContractViolation(f"unknown extended item {item!r}")


def stage_seconds(stage: str, w: Workloads) -> float:
    if stage in STAGE_WORKLOADS:
        workload = STAGE_WORKLOADS[stage]
        return workload.anchors * sum(unit_seconds(w, unit) for unit in workload.units)
    if stage in EXTENDED_ORDER:
        return extended_cost(stage, w)
    raise ContractViolation(f"unknown pilot stage {stage!r}")


def project_pilot(charged: float, remaining: tuple[str, ...], w: Workloads) -> Record:
    """Charged so far plus the remaining stages' measured workloads; no completed stage is counted twice."""
    if not math.isfinite(charged) or charged < 0:
        raise ContractViolation("charged process-seconds must be finite and nonnegative")
    if len(set(remaining)) != len(remaining):
        raise ContractViolation("a remaining stage may be named once")
    lines: Record = {}
    total = charged
    for stage in remaining:
        seconds = stage_seconds(stage, w)
        lines[stage] = seconds
        total += seconds
    return {
        "method": "charged so far plus the remaining stages' measured one-per-card workloads",
        "charged": charged,
        "remaining_lines": lines,
        "projected_gpu_process_seconds": total,
    }


def core_over_ceiling(charged: float, core_remaining: float, ceiling: float) -> bool:
    """§8: if the core alone projects over the ceiling the pilot stops before V1 is opened."""
    return charged + core_remaining > ceiling


def extended_limit(ceiling: float) -> float:
    """Ruling R23: extended work fits only under (1 - EXTENDED_RESERVE) x the ceiling."""
    return (1 - EXTENDED_RESERVE) * ceiling


def extended_fits(projected: float, cost: float, ceiling: float) -> bool:
    return projected + cost <= extended_limit(ceiling)


def admit_extended(charged: float, core_remaining: float, w: Workloads, ceiling: float) -> list[str]:
    """The admitted prefix of EXTENDED_ORDER: admission stops at the first item that does not project to fit under
    the reserve (R23)."""
    projected = charged + core_remaining
    admitted: list[str] = []
    if projected > ceiling:
        return admitted
    for item in EXTENDED_ORDER:
        cost = extended_cost(item, w)
        if not extended_fits(projected, cost, ceiling):
            break
        projected += cost
        admitted.append(item)
    return admitted


def measured_confirmatory(n: int, per_anchor_measured: float, fixed: float, ceiling: int) -> Record:
    """Ruling I3: the §10 projection from the pilot's measured whole-child charges per development anchor."""
    if n < CONFIRMATORY_MINIMUM:
        raise ContractViolation(f"the confirmatory count is at least {CONFIRMATORY_MINIMUM}")
    for value in (per_anchor_measured, fixed):
        if not math.isfinite(value) or value <= 0:
            raise ContractViolation("measured per-anchor and fixed charges must be positive and finite")
    total = fixed + n * per_anchor_measured
    return {
        "method": "fixed replay and acceptance charge plus n x the largest development anchor's whole-child charge",
        "anchors": n,
        "per_anchor_gpu_process_seconds": per_anchor_measured,
        "fixed_v0_gpu_process_seconds": fixed,
        "projected_gpu_process_seconds": total,
        "ceiling_seconds": ceiling,
        "within_ceiling": total <= ceiling,
    }


def _confirmatory(n: int, w: Workloads, arms: Arms, fixed: float, ceiling: int) -> Record:
    if n < CONFIRMATORY_MINIMUM:
        raise ContractViolation(f"the confirmatory count is at least {CONFIRMATORY_MINIMUM}")
    if not math.isfinite(fixed) or fixed < 0:
        raise ContractViolation("the fixed V0 term must be finite and nonnegative")
    anchor = per_anchor(w, arms)
    total = fixed + n * anchor
    return {
        "method": "fixed V0 term plus n x the measured per-anchor workload of the arm set",
        "anchors": n,
        "arms": json_value(sorted(arms)),
        "per_anchor_gpu_process_seconds": anchor,
        "fixed_v0_gpu_process_seconds": fixed,
        "projected_gpu_process_seconds": total,
        "ceiling_seconds": ceiling,
        "within_ceiling": total <= ceiling,
    }


def project_confirmatory(n: int, w: Workloads, arms: Arms, fixed: float) -> Record:
    """§10 projection under the 12-hour ceiling; only the confirmatory arm set may be projected."""
    extra = sorted(set(arms) - CONFIRMATORY_ARMS)
    if extra:
        raise ContractViolation(f"arms outside the confirmatory set: {extra}")
    return _confirmatory(n, w, arms, fixed, M0_CONFIRMATORY_SECONDS)


def with_static(n: int, w: Workloads, arms_with_static: Arms, fixed: float) -> Record:
    """The §10 projection with S-static (its measured host training and refit), against the 14-hour ceiling.

    S-static is excluded under 12 hours by the §10 rule, not by this arithmetic: the record says it requires 14 h.
    """
    names = set(arms_with_static)
    if not STATIC_ARMS <= names:
        raise ContractViolation("with_static requires both S-static and S-static:refit")
    extra = sorted(names - CONFIRMATORY_ARMS - STATIC_ARMS)
    if extra:
        raise ContractViolation(f"arms outside the confirmatory set: {extra}")
    record = _confirmatory(n, w, arms_with_static, fixed, M0_CONFIRMATORY_STATIC_SECONDS)
    record["requires_14h"] = True
    return record
