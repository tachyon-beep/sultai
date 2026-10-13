"""M0 confirmatory launcher (preregistration v4 §10): fresh anchors, the §10 arm whitelist, a 12 GPU-hour hard stop.

A supervisor process with its own custody receipt and its own `Supervisor` instance (one process per card). It reads a
clean pilot result hash-listed in the admission certificate, draws n anchors from the confirmatory block (the restart
block only under a listed disposition), runs V0 and then the §10 arm set on every anchor, reads V1 and V2 once and
publishes `confirmatory-result.json`. Like the pilot it never imports `sultai_screen.screen`, `former_bridge` or
`alignment`; the data root goes to the `sultai_screen.m0.worker` children, which keep the seal.
"""

from __future__ import annotations

import argparse
import signal
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Final

from sultai_screen.contracts import (
    FeasibilityHold,
    InstrumentHold,
    Json,
    PilotHold,
    Record,
    file_sha,
    integer,
    json_value,
    mapping,
    number,
    publish,
    read_record,
    sequence,
    text,
)
from sultai_screen.m0 import stats, verdicts
from sultai_screen.m0.costs import (
    M0_CONFIRMATORY_STATIC_SECONDS,
    STATIC_ARMS,
    Workloads,
    project_confirmatory,
    with_static,
)
from sultai_screen.m0.identity import (
    CERTIFIED_FRACTION,
    CHECKPOINTS,
    CHECKPOINTS_B,
    FOLD_TOLERANCE,
    M0_CONFIRMATORY_SECONDS,
    protocol_identity,
)
from sultai_screen.m0.identity import REPO as M0_REPO
from sultai_screen.m0.pilot import (
    WORKER,
    Run,
    axis,
    check_cards,
    check_certificate,
    fold_check,
    job,
    m0_stop_report,
    raise_only,
    slicing_known_answer,
    stage_replay,
)
from sultai_screen.m0.provenance import environment_identity
from sultai_screen.m0.receipts import (
    BOOTSTRAP_SEED,
    STATIC_FILES,
    TRAINING_FILES,
    Anchor,
    anchor,
    bound,
    schedule_matches,
    source,
)
from sultai_screen.m0.seeds import BLOCKS, Lineage, admit
from sultai_screen.m0.worker import MODULE_ARMS, RETUNE_ARMS
from sultai_screen.pilot import Job, Supervisor, check_output_root, claim_attempt

REPO: Path = M0_REPO
# Never the pilot's (.weft/m0-pilot-attempt.json) nor a Gate 3 receipt (.weft/gate3-pilot*.json).
CUSTODY: Final = ".weft/m0-confirmatory-attempt.json"
# A restart after a failed attempt claims this receipt and binds the first one by hash (the linked_pilot_* pattern).
RESTART_CUSTODY: Final = ".weft/m0-confirmatory-restart-attempt.json"
# Every action a confirmatory child may run: the reused V0 replay and acceptance (i-iv), the fresh anchors' hosts,
# N0, module and re-tune grids, T-cool and the fold-check. S-static's host only under the 14 h flag.
ACTIONS: Final = frozenset(("replay", "acceptance", "train", "noop", "modules", "retune", "tcool", "fold-check"))
# §7 V1 and V2: the arms whose certification fraction gates a reading.
CERTIFIED: Final = ("M-aff", "M-sq", "M-small-B", "M-wide→8:rank:refit")
BOUNDED: Final = ("D1", "D1prime", "D2", "D3", "delta_acc")
STOP_ONLY_PILOT: Final = ("pilot_clean", "confirmatory_ready")


@dataclass(frozen=True)
class Plan:
    """Everything admitted before custody: the anchors, the axis, the arm set, the ceiling and the custody binding."""

    lineage: Lineage
    seeds: tuple[int, ...]
    schedule: tuple[tuple[int, ...], tuple[int, ...]]
    arms: frozenset[str]
    static: bool
    ceiling: int
    custody: str
    binding: Record
    projection: Record

    def block(self) -> Record:
        return {"lineage": self.lineage, "first": self.seeds[0], "last": self.seeds[-1], "count": len(self.seeds)}

    def flags(self) -> tuple[str, ...]:
        return ("--checkpoints", axis(self.schedule[0]), "--checkpoints-b", axis(self.schedule[1]))


def whitelist(jobs: list[Job], arms: frozenset[str]) -> list[Job]:
    """Ruling R2-revised: every modules/retune child names its --arms, all inside the §10 set; no other action runs."""
    actions = ACTIONS | ({"static"} if STATIC_ARMS <= arms else set())
    for item in jobs:
        if item.action not in actions:
            raise PilotHold(f"{item.name}: action {item.action} is outside the §10 confirmatory")
        if item.action in ("modules", "retune"):
            if "--arms" not in item.extra:
                raise PilotHold(f"{item.name}: a {item.action} child must restrict its --arms to the §10 whitelist")
            outside = sorted(set(item.extra[item.extra.index("--arms") + 1].split(",")) - arms)
            if outside:
                raise PilotHold(f"{item.name}: arm(s) {outside} are outside the §10 whitelist")
    return jobs


@dataclass
class Guarded(Run):
    """The pilot's Run with every job list, the reused helpers' included, checked against the whitelist."""

    arms: frozenset[str]

    def run(self, jobs: list[Job]) -> None:
        super().run(whitelist(jobs, self.arms))


def evidenced(approved: Record, path: Path, what: str) -> Record:
    """`path` is inside the repository and listed, with its current SHA-256, in the certificate's evidence."""
    if not path.resolve().is_relative_to(REPO.resolve()) or not path.is_file():
        raise PilotHold(f"{what} ({path}) must be a committed file inside the repository")
    relative, sha = str(path.resolve().relative_to(REPO.resolve())), file_sha(path)
    entries = [mapping(entry) for entry in sequence(approved["evidence"])]
    if not any(entry["path"] == relative and entry["sha256"] == sha for entry in entries):
        raise PilotHold(f"{what} ({relative}) is not hash-listed in the certificate evidence")
    return {relative: sha}


def workloads_of(record: Record) -> Workloads:
    """The pilot's measured kernels from 04-admission.json (the inverse of receipts.workloads_record)."""
    return Workloads(
        host={float(scale): number(value) for scale, value in mapping(record["host"]).items()},
        tcool=number(record["tcool"]),
        static=number(record["static"]),
        grid={kind: number(value) for kind, value in mapping(record["grid"]).items()},
        front_rerun=number(record["front_rerun"]),
        fold_and_calibrations=number(record["fold_and_calibrations"]),
        evaluation=number(record["evaluation"]),
        fixed_v0=number(record["fixed_v0"]),
    )


def admit_launch(approved: Record, pilot_result: Path, static: bool, restart: Path | None) -> Plan:
    """Every refusal before custody, so a refused launch never consumes the single attempt."""
    binding: Record = {"pilot_result": evidenced(approved, pilot_result, "the pilot result")}
    pilot = read_record(pilot_result)
    if pilot["pilot_clean"] is not True:
        raise PilotHold("the confirmatory requires a clean pilot result")
    for key, value in protocol_identity().items():
        if key not in pilot or pilot[key] != value:
            raise PilotHold(f"the pilot result carries another protocol identity ({key})")
    admission = pilot_result.parent / "04-admission.json"
    binding["pilot_admission"] = evidenced(approved, admission, "the pilot's measured kernels")
    if static and pilot["static_admitted"] is not True:
        raise PilotHold("S-static joins only if the pilot's SD(D5) rule admitted it")
    if static and not ("confirmatory_ceiling_hours" in approved and approved["confirmatory_ceiling_hours"] == 14):
        raise PilotHold("S-static requires John's 14 GPU-hour confirmatory ceiling in the admission certificate")
    lineage: Lineage = "confirmatory"
    custody = CUSTODY
    if restart is not None:
        lineage, custody = "restart", RESTART_CUSTODY
        prior = REPO / CUSTODY
        if not prior.is_file():
            raise PilotHold("a restart requires the prior confirmatory custody receipt")
        binding["restart_disposition"] = evidenced(approved, restart, "the restart disposition")
        binding["linked_confirmatory_custody"] = {CUSTODY: file_sha(prior)}
    schedule = mapping(pilot["schedule"])
    checkpoints = raise_only(tuple(integer(c) for c in sequence(schedule["checkpoints"])), CHECKPOINTS)
    checkpoints_b = raise_only(tuple(integer(c) for c in sequence(schedule["checkpoints_b"])), CHECKPOINTS_B)
    if checkpoints_b != tuple(2 * c for c in checkpoints):
        raise PilotHold("M-small-B's axis must stay at twice the module axis")
    n = integer(pilot["confirmatory_count"])
    start = BLOCKS[lineage].start
    seeds = tuple(admit(seed, lineage) for seed in range(start, start + n))
    w = workloads_of(mapping(read_record(admission)["workloads"]))
    arms = stats.CONFIRMATORY_ARMS | (STATIC_ARMS if static else frozenset())
    projection = (with_static if static else project_confirmatory)(n, w, arms, w.fixed_v0)
    if projection["within_ceiling"] is not True:
        raise FeasibilityHold("the confirmatory projects over its ceiling from the pilot's measured kernels")
    ceiling = M0_CONFIRMATORY_STATIC_SECONDS if static else M0_CONFIRMATORY_SECONDS
    return Plan(lineage, seeds, (checkpoints, checkpoints_b), arms, static, ceiling, custody, binding, projection)


# --- V0 on the fresh anchors ---------------------------------------------------------------------------------------


def stage_v0(run: Run, plan: Plan) -> Record:
    """Every anchor's host and N0; M-aff with the fold residual on every anchor; the fold-check after the first."""
    lineage, seeds = plan.lineage, plan.seeds
    slicing = slicing_known_answer()
    if slicing["known_answer"] is not True:
        raise InstrumentHold("V0 slicing known answer failed")
    run.run([job(f"train-{s}", "train", s, lineage) for s in seeds])
    for s in seeds:
        bound(run.path(f"train-{s}"), run.payload(f"train-{s}", "train", s, lineage), TRAINING_FILES)
    run.run([job(f"noop-{s}", "noop", s, lineage, training=run.path(f"train-{s}")) for s in seeds])
    maff = [
        job(f"maff-{s}", "modules", s, lineage, "--arms", "M-aff", "--noop-root", str(run.path(f"noop-{s}")),
            *plan.flags())
        for s in seeds
    ]  # fmt: skip
    run.run(maff[:2])
    checked = fold_check(run, f"fold-check-{seeds[0]}", f"maff-{seeds[0]}", seed=seeds[0], lineage=lineage)
    run.run(maff[2:])
    residuals: Record = {}
    for s in seeds:
        payload = run.payload(f"maff-{s}", "modules", s, lineage)
        schedule_matches(f"maff-{s}", payload, *plan.schedule)
        if mapping(mapping(payload["arms"])["M-aff"])["status"] != "available" or payload["r_aff"] is None:
            raise InstrumentHold(f"M-aff unavailable on anchor {s}; the V0 fold residual cannot be read")
        r_aff = number(payload["r_aff"])
        if abs(r_aff) > FOLD_TOLERANCE:
            raise InstrumentHold(f"V0 fold residual clause: |r_aff| = {abs(r_aff)} > {FOLD_TOLERANCE} on anchor {s}")
        residuals[str(s)] = r_aff
    return {
        "slicing": slicing,
        "fold_residuals": residuals,
        "fold_tolerance": FOLD_TOLERANCE,
        "fold_check": checked,
        "charged_gpu_process_seconds": run.supervisor.charged,
    }


def stage_hosts(run: Run, plan: Plan) -> Record:
    """T-cool resumes each anchor's own host; S-static's host from step zero under the 14 h flag only."""
    lineage, seeds = plan.lineage, plan.seeds
    run.run(
        [job(f"tcool-{s}", "tcool", s, lineage, training=run.path(f"train-{s}")) for s in seeds]
        + ([job(f"static-{s}", "static", s, lineage) for s in seeds] if plan.static else [])
    )
    tcool: Record = {}
    static: Record = {}
    for s in seeds:
        resumed = run.payload(f"tcool-{s}", "tcool", s, lineage)
        if resumed["source_final_state_sha256"] != run.payload(f"train-{s}", "train", s, lineage)["final_state_sha256"]:
            raise PilotHold(f"T-cool on anchor {s} resumed another host")
        tcool[str(s)] = resumed["final_state_sha256"]
        if plan.static:
            payload = run.payload(f"static-{s}", "static", s, lineage)
            bound(run.path(f"static-{s}"), payload, STATIC_FILES)
            static[str(s)] = payload["static_sha256"]
    return {"tcool_final_state_sha256": tcool, "static_sha256": static if plan.static else None}


def stage_core(run: Run, plan: Plan) -> None:
    """The §10 module arms (M-aff ran in V0) and the two re-tune arms on every anchor, never dropping one."""
    module_arms = ",".join(name for name in MODULE_ARMS if name in plan.arms - {"M-aff"})
    retune_arms = ",".join(name for name in RETUNE_ARMS if name in plan.arms)
    kinds = ("noop", "tcool", "static") if plan.static else ("noop", "tcool")

    def roots(s: int, *names: str) -> tuple[str, ...]:
        return tuple(value for kind in names for value in (f"--{kind}-root", str(run.path(f"{kind}-{s}"))))

    run.run(
        [job(f"modules-{s}", "modules", s, plan.lineage, "--arms", module_arms, *roots(s, *kinds), *plan.flags())
         for s in plan.seeds]
        + [job(f"retune-{s}", "retune", s, plan.lineage, "--arms", retune_arms, *roots(s, "noop"), *plan.flags())
           for s in plan.seeds]
    )  # fmt: skip


def aggregate(run: Run, plan: Plan) -> tuple[list[Anchor], list[Record]]:
    """Ruling R11 through receipts.anchor: a required arm unavailable on any anchor holds; then V0's last clauses."""
    required = plan.arms
    anchors: list[Anchor] = []
    for s in plan.seeds:
        sources = tuple(
            source(run.root, f"{kind}-{s}", action, s, plan.lineage)
            for kind, action in (("modules", "modules"), ("retune", "retune"), ("maff", "modules"))
        )
        for item in sources:
            schedule_matches(item.name, item.payload, *plan.schedule)
        anchors.append(anchor(s, sources, required))
    for clause in ("printed_counts", "complement_unchanged"):
        failed = [item.seed for item in anchors if item.report[clause] is not True]
        if failed:
            raise InstrumentHold(f"V0 failed at aggregation: {clause} on anchors {failed}")
    return anchors, [stats.quantities(item.result, required) for item in anchors]


# --- V1 and V2, read once ------------------------------------------------------------------------------------------


def read_once(plan: Plan, anchors: list[Anchor], quantities: list[Record]) -> Record:
    """§7 at confirmatory n: medians with one-sided 95 % anchor-bootstrap bounds, then V1, labels, V2 and static."""

    def column(key: str) -> list[float]:
        values = [q[key] for q in quantities]
        if None in values:
            raise InstrumentHold(f"{key} is undefined on an anchor; no verdict is read")
        return [number(value) for value in values]

    def interval(key: str) -> Record:
        values = column(key)
        return {
            "median": stats.median(values),
            "lower": stats.anchor_bound(values, BOOTSTRAP_SEED, lower=True),
            "upper": stats.anchor_bound(values, BOOTSTRAP_SEED, lower=False),
        }

    bounded = BOUNDED + (("D5",) if plan.static else ())
    intervals = {key: interval(key) for key in bounded}
    d1 = column("D1")
    lower, lower_reason = stats.sign_test_bound(d1, lower=True)
    upper, upper_reason = stats.sign_test_bound(d1, lower=False)
    intervals["D1"].update(sign_test_lower=lower, sign_test_upper=upper, sign_test_reason=lower_reason or upper_reason)
    n = len(anchors)
    fractions = {name: sum(item.result.arms[name].certified is True for item in anchors) / n for name in CERTIFIED}

    def bound_of(key: str, side: str) -> float:
        return number(intervals[key][side])

    v1 = verdicts.v1(bound_of("D1", "lower"), bound_of("D1", "upper"), fractions["M-aff"], fractions["M-sq"])
    d3_lower, d3_upper = bound_of("D3", "lower"), bound_of("D3", "upper")
    labels = verdicts.labels(bound_of("delta_acc", "lower"), d3_lower, d3_upper, bound_of("D1prime", "lower"), None)
    v2 = verdicts.v2(
        bound_of("D2", "lower"), bound_of("D2", "upper"), fractions["M-small-B"], fractions["M-wide→8:rank:refit"]
    )
    widths = {
        key: stats.achieved_half_width(column(key), BOOTSTRAP_SEED) if key != "D5" or plan.static else None
        for key in ("D2", "D3", "D5")
    }
    return {
        "v1": v1,
        "v2": v2,
        "labels": json_value(labels),
        "static_reading": verdicts.static_reading(bound_of("D5", "lower"), True, n) if plan.static else None,
        "programme_reading": verdicts.programme_reading(v1, labels),
        "intervals": json_value(intervals),
        "certified_fractions": json_value(fractions),
        "certified_fraction_required": CERTIFIED_FRACTION,
        "achieved_half_width": json_value(widths),
        "achieved_half_width_reasons": {"D5": None if plan.static else "S-static is not in this confirmatory set"},
        "D4": None,
        "D4_reason": "M-sq-fresh is a pilot-only descriptive (§10); no fresh-data label is read",
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


# --- the attempt ---------------------------------------------------------------------------------------------------


def stopped(error: Exception, stage: str, run: Run, plan: Plan) -> Record:
    """The stop report with the M0 identity; any trained anchor burns the block (§10), so the record says so."""
    report = m0_stop_report(error, stage, run.supervisor.charged)
    launched = {text(mapping(entry)["job"]) for entry in run.supervisor.ledger}
    return {
        **{key: value for key, value in report.items() if key not in STOP_ONLY_PILOT},
        "seed_block": plan.block(),
        "seed_block_burnt": any(f"train-{s}" in launched for s in plan.seeds),
        "ceiling_gpu_process_seconds": plan.ceiling,
    }


def confirmatory(
    root: Path, data: Path, certificate: Path, pilot_result: Path, *, static: bool = False, restart: Path | None = None
) -> int:
    """One M0 confirmatory attempt: admission, cards, custody, then V0, hosts, core, aggregation and one reading."""
    check_output_root(root)
    if root.exists():
        raise PilotHold("one attempt requires a new output root; no automatic resume")
    approved = read_record(certificate)
    identity = check_certificate(approved)
    plan = admit_launch(approved, pilot_result, static, restart)
    environment = environment_identity()
    cards = check_cards()
    certificate_sha = file_sha(certificate)
    # Durable one-attempt custody, never removed or reset; a restart binds the first receipt by hash.
    custody = dict(identity, environment=environment, certificate_sha256=certificate_sha, **plan.binding)
    claim_attempt(REPO / plan.custody, root, custody)
    root.mkdir(parents=True, exist_ok=False)
    supervisor = Supervisor(root, data, ceiling=plan.ceiling, per_card=1, module=WORKER)
    run = Guarded(root, supervisor, plan.arms)
    stage = "opening"
    try:
        publish(
            root / "00-opening.json",
            {
                "cards": cards,
                "environment": environment,
                "implementation_commit": text(identity["commit"]),
                "certificate_sha256": certificate_sha,
                "custody": plan.custody,
                "binding": plan.binding,
                "confirmatory_gpu_process_seconds_ceiling": plan.ceiling,
                "processes_per_card": 1,
                "worker": WORKER,
                "seed_block": plan.block(),
                "arms": json_value(sorted(plan.arms)),
                "static": plan.static,
                "schedule": {"checkpoints": list(plan.schedule[0]), "checkpoints_b": list(plan.schedule[1])},
                "projection": plan.projection,
                **protocol_identity(),
            },
        )
        stage = "replay"
        publish(root / "01-replay.json", stage_replay(run, timed=False))
        stage = "v0"
        v0 = stage_v0(run, plan)
        publish(root / "02-v0.json", v0)
        stage = "hosts"
        publish(root / "03-hosts.json", stage_hosts(run, plan))
        stage = "core"
        stage_core(run, plan)
        stage = "aggregation"
        anchors, quantities = aggregate(run, plan)
        reports: list[Json] = [{**item.report, "quantities": q} for item, q in zip(anchors, quantities, strict=True)]
        publish(root / "04-anchors.json", reports)
        stage = "read"
        reading = read_once(plan, anchors, quantities)
        publish(
            root / "confirmatory-result.json",
            {
                "software_valid": True,
                "instrument_valid": True,
                "verdict": "confirmatory-read",
                "v0": {
                    "passed": True,
                    "replay_both_cards_bitwise": True,
                    "acceptance_on_both_cards": True,
                    "slicing_known_answer": True,
                    "fold_residuals_within_tolerance": True,
                    "fold_check_matches": True,
                    "printed_counts": True,
                    "complement_unchanged": True,
                    "not_rerun": "P1 instrument (A1, B3) and zero-module identity: development-anchor clauses of "
                    "the clean pilot's V0 (§7)",
                },
                **reading,
                "anchors": list(plan.seeds),
                "seed_block": plan.block(),
                "static": plan.static,
                "arms": json_value(sorted(plan.arms)),
                "ceiling_gpu_process_seconds": plan.ceiling,
                "charged_gpu_process_seconds": run.supervisor.charged,
                **protocol_identity(),
            },
        )
        return 0
    except Exception as error:
        publish(root / "confirmatory-result.json", stopped(error, stage, run, plan))
        return 2


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--review-certificate", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True, help="the committed, clean pilot-result.json")
    parser.add_argument("--static", action="store_true", help="S-static under John's 14 GPU-hour ceiling")
    parser.add_argument("--restart", type=Path, metavar="DISPOSITION", help="restart block under a listed disposition")
    args = parser.parse_args()
    return confirmatory(
        args.output.resolve(),
        args.data_root.resolve(),
        args.review_certificate.resolve(),
        args.pilot_result.resolve(),
        static=args.static,
        restart=None if args.restart is None else args.restart.resolve(),
    )


def _terminate(signum: int, _frame: FrameType | None) -> None:
    raise PilotHold(f"supervisor interrupted by signal {signum}; stop owned children and retain the attempt")


if __name__ == "__main__":
    raise SystemExit(main())
