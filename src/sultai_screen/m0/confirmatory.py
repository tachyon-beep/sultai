"""M0 confirmatory launcher (preregistration v4 §10): fresh anchors, the §10 arm whitelist, a 12 GPU-hour hard stop.

A supervisor process with its own custody receipt and its own `Supervisor` instance (one process per card). It reads a
clean pilot result hash-listed in the admission certificate, draws n anchors from the confirmatory block (the restart
block only under a listed disposition), runs V0 and then the §10 arm set on every anchor, reads V1 and V2 once and
publishes `confirmatory-result.json`. Like the pilot it never itself imports `sultai_screen.screen`, `former_bridge` or
`alignment` (they reach the supervisor's process only through the reused Gate 3 `sultai_screen.pilot`); the data root
goes to the `sultai_screen.m0.worker` children, which keep the seal.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
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
    STATIC_ARMS,
    Workloads,
    measured_confirmatory,
    project_confirmatory,
    with_static,
)
from sultai_screen.m0.identity import (
    CERTIFIED_FRACTION,
    CHECKPOINTS,
    CHECKPOINTS_B,
    FOLD_TOLERANCE,
    M0_CONFIRMATORY_SECONDS,
    M0_CONFIRMATORY_STATIC_SECONDS,
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
    implementation,
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
)
from sultai_screen.m0.seeds import BLOCKS, Lineage, admit
from sultai_screen.m0.worker import MODULE_ARMS, RETUNE_ARMS, ZERO_MODULE_KINDS
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
# Amendment 3 (A3.2): the quantities a verdict or label reads hold the reading when undefined on an anchor; every other
# §6 quantity the §10 set defines is published beside with bounds and the sign test, undefined with the anchor named.
VERDICT_KEYS: Final = ("D1", "D1prime", "D2", "D3", "delta_acc")
BESIDE_KEYS: Final = ("D2s", "D2_random", "D2s_random", "G_tcool", "r_aff", "G_rtemp", "G_rcal")
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
            if item.extra.count("--arms") != 1:
                raise PilotHold(f"{item.name}: a {item.action} child must name exactly one --arms in the §10 whitelist")
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
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=REPO, capture_output=True)
    if tracked.returncode != 0:
        raise PilotHold(f"{what} ({relative}) is not tracked by git; only committed evidence is read")
    return {relative: sha}


def field(record: Record, key: str, where: str) -> Json:
    """A required key of a pilot receipt; a missing one holds before custody (never a KeyError)."""
    if key not in record:
        raise PilotHold(f"{where} lacks {key!r}")
    return record[key]


def failed_prior(custody: Record) -> Record | None:
    """§11: a restart follows a failed attempt only; the prior result, when published, is bound by hash."""
    result = Path(text(field(custody, "output", "the prior confirmatory custody receipt"))) / "confirmatory-result.json"
    if not result.is_file():
        return None
    if field(read_record(result), "verdict", "the prior confirmatory result") == "confirmatory-read":
        raise PilotHold("the prior confirmatory attempt read V1 and V2; a restart follows a failed attempt only")
    return {str(result): file_sha(result)}


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
        noop_evaluate={float(scale): number(value) for scale, value in mapping(record["noop_evaluate"]).items()},
    )


def source_binding(approved: Record, identity: Record, pilot: Record) -> Record:
    """Ruling R27 (§10): the confirmatory runs the pilot's implementation source, or a hash-listed disposition in its
    certificate (`source_change_disposition`) names the new version by its source hash (R28)."""
    pilot_source = text(field(pilot, "implementation_source_sha256", "the pilot result"))
    current = text(identity["source_sha256"])
    record: Record = {
        "pilot_source_sha256": pilot_source,
        "pilot_commit": field(pilot, "implementation_commit", "the pilot result"),
        "confirmatory_source_sha256": current,
        "same_source": current == pilot_source,
    }
    if current != pilot_source:
        if "source_change_disposition" not in approved:
            raise PilotHold(
                "the confirmatory implementation source differs from the pilot's; a hash-listed "
                "source_change_disposition is required (§10)"
            )
        disposition = REPO / text(approved["source_change_disposition"])
        record["source_change_disposition"] = evidenced(approved, disposition, "the source-change disposition")
        # R28 minor 1: the disposition names the version it admits by its source hash.
        if current not in disposition.read_text():
            raise PilotHold(f"the source-change disposition does not name the confirmatory's source_sha256 {current}")
    return record


def admit_launch(approved: Record, identity: Record, pilot_result: Path, static: bool, restart: Path | None) -> Plan:
    """Every refusal before custody, so a refused launch never consumes the single attempt."""
    binding: Record = {"pilot_result": evidenced(approved, pilot_result, "the pilot result")}
    pilot = read_record(pilot_result)
    if field(pilot, "pilot_clean", "the pilot result") is not True:
        raise PilotHold("the confirmatory requires a clean pilot result")
    for key, value in protocol_identity().items():
        if key not in pilot or pilot[key] != value:
            raise PilotHold(f"the pilot result carries another protocol identity ({key})")
    binding["implementation_source"] = source_binding(approved, identity, pilot)
    admission = pilot_result.parent / "04-admission.json"
    binding["pilot_admission"] = evidenced(approved, admission, "the pilot's measured kernels")
    if static and field(pilot, "static_admitted", "the pilot result") is not True:
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
        binding["linked_confirmatory_result"] = failed_prior(read_record(prior))
    schedule = mapping(field(pilot, "schedule", "the pilot result"))
    axes = (field(schedule, key, "the pilot schedule") for key in ("checkpoints", "checkpoints_b"))
    checkpoints, checkpoints_b = (
        raise_only(tuple(integer(c) for c in sequence(values)), default)
        for values, default in zip(axes, (CHECKPOINTS, CHECKPOINTS_B), strict=True)
    )
    if checkpoints_b != tuple(2 * c for c in checkpoints):
        raise PilotHold("M-small-B's axis must stay at twice the module axis")
    n = integer(field(pilot, "confirmatory_count", "the pilot result"))
    start = BLOCKS[lineage].start
    seeds = tuple(admit(seed, lineage) for seed in range(start, start + n))
    w = workloads_of(mapping(field(read_record(admission), "workloads", "the pilot's 04-admission.json")))
    arms = stats.CONFIRMATORY_ARMS | (STATIC_ARMS if static else frozenset())
    ceiling = M0_CONFIRMATORY_STATIC_SECONDS if static else M0_CONFIRMATORY_SECONDS
    # I3 and M7: the fixed term is the pilot's replay and acceptance charge; the larger of the kernel projection and
    # the measured whole-child projection must fit.
    carried = mapping(field(pilot, "confirmatory_projection", "the pilot result"))
    fixed = number(field(carried, "fixed_v0_gpu_process_seconds", "the pilot's confirmatory projection"))
    per = "static_per_anchor_gpu_process_seconds" if static else "per_anchor_gpu_process_seconds"
    kernel = (with_static if static else project_confirmatory)(n, w, arms, fixed)
    measured = measured_confirmatory(
        n, number(field(carried, per, "the pilot's confirmatory projection")), fixed, ceiling
    )
    larger = max(number(kernel["projected_gpu_process_seconds"]), number(measured["projected_gpu_process_seconds"]))
    projection: Record = {
        "kernel": kernel,
        "measured": measured,
        "projected_gpu_process_seconds": larger,
        "ceiling_seconds": ceiling,
        "within_ceiling": larger <= ceiling,
    }
    if larger > ceiling:
        raise FeasibilityHold("the confirmatory projects over its ceiling from the pilot's kernels or measured charges")
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
    # R25: every fresh anchor's noop child checked the zero module of every form against N0, bitwise.
    failed: list[str] = []
    recorded: Record = {}
    for s in seeds:
        flags = mapping(run.payload(f"noop-{s}", "noop", s, lineage)["zero_module_identity"])
        if set(flags) != set(ZERO_MODULE_KINDS):
            raise InstrumentHold(f"anchor {s}'s noop child did not check every module form for zero-module identity")
        recorded[str(s)] = flags
        failed.extend(f"{s}:{kind}" for kind, value in flags.items() if value is not True)
    if failed:
        raise InstrumentHold(f"V0 zero-module identity (bitwise, every module form) failed on {failed}")
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
        "zero_module_identity": True,
        # B3: the per-anchor flags the clause was read from (pilot M3/R25), not the literal beside them.
        "zero_module_identity_flags": recorded,
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
            run.source(f"{kind}-{s}", action, s, plan.lineage)
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
    """§7 at confirmatory n: every defined §6 quantity as its median with one-sided 95 % anchor-bootstrap bounds and
    the exact sign-test bound beside it (amendment 3, A3.2), then V1, labels, V2 and static."""

    def column(key: str) -> list[float]:
        values = [q[key] for q in quantities]
        if None in values:
            raise InstrumentHold(f"{key} is undefined on an anchor; no verdict is read")
        return [number(value) for value in values]

    def interval(values: list[float]) -> Record:
        lower, lower_reason = stats.sign_test_bound(values, lower=True)
        upper, upper_reason = stats.sign_test_bound(values, lower=False)
        return {
            "median": stats.median(values),
            "lower": stats.anchor_bound(values, BOOTSTRAP_SEED, lower=True),
            "upper": stats.anchor_bound(values, BOOTSTRAP_SEED, lower=False),
            "sign_test_lower": lower,
            "sign_test_upper": upper,
            "sign_test_reason": lower_reason or upper_reason,
        }

    def beside(name: str, values: list[float | None], reasons: list[Json]) -> Record:
        """A no-verdict quantity: its interval, or undefined naming the first anchor it is undefined on."""
        for item, value, reason in zip(anchors, values, reasons, strict=True):
            if value is None:
                return {"median": None, "reason": f"{name} undefined on anchor {item.seed} ({reason})"}
        return interval([number(value) for value in values])

    verdict_keys = VERDICT_KEYS + (("D5",) if plan.static else ())
    beside_keys = BESIDE_KEYS + (("D5_norefit",) if plan.static else ())
    intervals: dict[str, Record] = {key: interval(column(key)) for key in verdict_keys}
    for key in beside_keys:
        intervals[key] = beside(
            key,
            [None if q[key] is None else number(q[key]) for q in quantities],
            [q[f"{key}_reason"] for q in quantities],
        )
    intervals["survival"] = {
        kind: {
            str(h): beside(
                f"survival {kind} h = {h}",
                [stats.survival_value(q, kind, h) for q in quantities],
                [q["survival_reason"] if q["survival_reason"] is not None else "not run" for q in quantities],
            )
            for h in stats.SURVIVAL_WIDTHS
        }
        for kind in stats.SURVIVAL_KINDS
    }
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
        "verdict_keys": list(verdict_keys),
        "certified_fractions": json_value(fractions),
        "certified_fraction_required": CERTIFIED_FRACTION,
        "achieved_half_width": json_value(widths),
        "achieved_half_width_reasons": {"D5": None if plan.static else "S-static is not in this confirmatory set"},
        "D4": None,
        "D4_reason": "M-sq-fresh is a pilot-only descriptive (§10); no fresh-data label is read",
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


# --- the attempt ---------------------------------------------------------------------------------------------------


def stopped(error: Exception, stage: str, run: Run, plan: Plan, identity: Record) -> Record:
    """The stop report with the M0 identity; custody consumed the block (ruling R22, §11), so every stop burns it.

    Ruling R21: a pinned-hash replay mismatch is `execution-invalid` (not the certified software or hardware); a failed
    instrument clause (acceptance matrix, slicing, identities, fold residual, fold-check) is `instrument-failed`.
    """
    report = m0_stop_report(error, stage, run.supervisor.charged, identity)
    return {
        **{key: value for key, value in report.items() if key not in STOP_ONLY_PILOT},
        "seed_block": plan.block(),
        "seed_block_burnt": True,
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
    plan = admit_launch(approved, identity, pilot_result, static, restart)
    environment = environment_identity()
    cards = check_cards()
    certificate_sha = file_sha(certificate)
    # Durable one-attempt custody, never removed or reset; a restart binds the first receipt by hash.
    custody = dict(identity, environment=environment, certificate_sha256=certificate_sha, **plan.binding)
    claim_attempt(REPO / plan.custody, root, custody)
    root.mkdir(parents=True, exist_ok=False)
    supervisor = Supervisor(root, data, ceiling=plan.ceiling, per_card=1, module=WORKER)
    run = Guarded(root, supervisor, text(identity["source_sha256"]), plan.arms)
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
                    "zero_module_identity": True,
                    "not_rerun": "P1 instrument (A1, B3): development-anchor clauses of the clean pilot's V0 (§7)",
                },
                **reading,
                "anchors": list(plan.seeds),
                "seed_block": plan.block(),
                "static": plan.static,
                "arms": json_value(sorted(plan.arms)),
                "ceiling_gpu_process_seconds": plan.ceiling,
                "charged_gpu_process_seconds": run.supervisor.charged,
                # B4 (R27): the certified implementation source this reading was produced by.
                **implementation(identity),
                **protocol_identity(),
            },
        )
        return 0
    except Exception as error:
        publish(root / "confirmatory-result.json", stopped(error, stage, run, plan, identity))
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
