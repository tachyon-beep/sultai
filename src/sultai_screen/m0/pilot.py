"""M0 pilot supervisor (preregistration v4 §9): its own custody, one process per card, 14,400 s, seven ordered stages.

A supervisor process. It reuses the frozen Gate 3 `Job`, `Supervisor` and custody helpers and never imports
`sultai_screen.screen`, `former_bridge` or `alignment` itself. It reads child receipts and `vectors.pt` only; the data
root is handed to the `sultai_screen.m0.worker` children, which keep the seal.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Final

import torch

from sultai.trust import FitUnavailable
from sultai_screen.contracts import (
    FeasibilityHold,
    InstrumentHold,
    Json,
    OracleUnavailable,
    PilotHold,
    Record,
    file_sha,
    integer,
    mapping,
    number,
    publish,
    read_record,
    sequence,
    text,
)
from sultai_screen.fit import Adapter
from sultai_screen.m0 import stats, verdicts
from sultai_screen.m0.costs import (
    STATIC_ARMS,
    Workloads,
    admit_extended,
    core_over_ceiling,
    project_confirmatory,
    project_pilot,
    stage_seconds,
    with_static,
)
from sultai_screen.m0.identity import (
    CERTIFIED_FRACTION,
    CHECKPOINTS,
    CHECKPOINTS_B,
    FOLD_TOLERANCE,
    M0_PILOT_SECONDS,
    S_STATIC_SD_LIMIT,
    protocol_identity,
    source_identity,
)
from sultai_screen.m0.identity import REPO as M0_REPO
from sultai_screen.m0.instrument import check_instrument, materiality_report
from sultai_screen.m0.modules import activation_means, cut, random_cut, rank_cut
from sultai_screen.m0.provenance import environment_identity
from sultai_screen.m0.receipts import (
    BOOTSTRAP_SEED,
    STATIC_FILES,
    TRAINING_FILES,
    Anchor,
    Source,
    anchor,
    bound,
    derive_workloads,
    ledger_charges,
    receipt,
    schedule_matches,
    source,
    workloads_record,
)
from sultai_screen.m0.seeds import BLOCKS, Lineage, width_scale
from sultai_screen.m0.worker import ATTEMPT4_FINAL_STATE, MODULE_ARMS, RETUNE_ARMS
from sultai_screen.pilot import GPU_UUIDS, Job, Supervisor, check_output_root, claim_attempt

REPO: Path = M0_REPO
# Never one of the five Gate 3 receipts (.weft/gate3-pilot*.json), which this module never reads, writes or resets.
CUSTODY: Final = ".weft/m0-pilot-attempt.json"
WORKER: Final = "sultai_screen.m0.worker"
ANCHORS: Final = tuple(BLOCKS["anchor"])
ACCEPTANCE: Final = {
    "resume_bitwise": "(i) resume versus uninterrupted",
    "checkpoint_bitwise": "(ii) checkpointed grid versus separate fits",
    "static_step0_bitwise": "(iii) complete static model at step 0",
    "evaluation_consistent": "(iv) shared padded evaluation",
}
REPLAY_KEYS: Final = ("epoch_state_sha256", "loss_stream_sha256", "final_state_sha256", "forward_backward")
# Ruling R15: raise iff stage-2 M-aff certifies on fewer than CERTIFIED_FRACTION of anchors; double every checkpoint.
RAISE_FACTOR: Final = 2
SPREAD_KEYS: Final = ("D1", "D2", "D3", "D5")
MEDIAN_KEYS: Final = (
    "D1",
    "D1prime",
    "D2",
    "D2s",
    "D2_random",
    "D2s_random",
    "D3",
    "G_tcool",
    "D4",
    "D5",
    "D5_norefit",
    "r_aff",
    "delta_acc",
    "G_rtemp",
    "G_rcal",
    "G_mexp",
    "G_lottery",
    "leave_one_out",
    "retune_fraction",
)
# §5 extended host items, in EXTENDED_ORDER, and the lineage each trains.
EXTENDED_HOSTS: Final[tuple[tuple[str, Lineage], ...]] = (
    ("cooled-20", "cooled-20"),
    ("F(0.25)", "frontier-0.25"),
    ("F(0.75)", "frontier-0.75"),
)
# The slicing known answer (§7 V0): the reviewed tests/test_m0_modules.py fixture, run in the supervisor on CPU (R18 b).
SLICING_UNITS: Final = (3, 9, 17, 22, 30, 41, 50, 63)
SLICING_WEIGHT_SEED: Final = 7
SLICING_SITE_SEED: Final = 1
RANDOM_CUT_SEED: Final = 0


def check_cards() -> Record:
    """Both pinned cards present and idle, no compute apps. Gate 3's Simic and date checks are not repeated."""
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    rows = [tuple(value.strip() for value in line.split(",")) for line in query.strip().splitlines()]
    if len(rows) != 2 or any(
        rows[i][0] != str(i) or rows[i][1] != GPU_UUIDS[i] or int(rows[i][2]) != 0 for i in range(2)
    ):
        raise PilotHold("pinned GPUs are unavailable, busy or identities differ")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    if apps.strip():
        raise PilotHold("GPU compute processes already active; preserve them")
    return {
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_query": query,
        "compute_apps": apps,
        "grant": "both cards granted to Sultai after Simic rung 4; attempt 4 recorded the Simic terminal receipts",
    }


def check_certificate(approved: Record) -> Record:
    """The admission certificate against the M0 identity and the committed source (Gate 3 field names)."""
    for key, value in protocol_identity().items():
        if key not in approved or approved[key] != value:
            raise PilotHold(f"review certificate protocol identity missing or mismatched: {key}")
    if (
        approved["quality_passed"] is not True
        or approved["runtime_tests_passed"] is not True
        or approved["independent_panel_accepted"] is not True
    ):
        raise PilotHold("required quality/runtime/independent review certificate missing")
    identity = source_identity()
    if approved["implementation_source_sha256"] != identity["source_sha256"]:
        raise PilotHold("review certificate belongs to another implementation source")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", text(approved["implementation_commit"]), text(identity["commit"])],
        cwd=REPO,
        check=True,
    )
    evidence = [mapping(entry) for entry in sequence(approved["evidence"])]
    for entry in evidence:
        path = REPO / text(entry["path"])
        if not path.resolve().is_relative_to(REPO.resolve()) or not path.is_file():
            raise PilotHold("review evidence missing or outside the repository")
        if file_sha(path) != text(entry["sha256"]):
            raise PilotHold("review evidence changed after certification")
    if len(evidence) < 3:
        raise PilotHold("quality, runtime and independent panel evidence required")
    return identity


def m0_stop_report(error: Exception, stage: str, charged: float) -> Record:
    """Gate 3's stop_report shape carrying the M0 identity."""
    instrument_failure = isinstance(error, (InstrumentHold, OracleUnavailable, FitUnavailable))
    feasibility_failure = isinstance(error, FeasibilityHold)
    return {
        "software_valid": instrument_failure or feasibility_failure,
        "instrument_valid": False if instrument_failure else None,
        "verdict": "instrument-failed"
        if instrument_failure
        else "hold-feasibility"
        if feasibility_failure
        else "execution-invalid",
        "pilot_clean": False,
        "confirmatory_ready": False,
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "charged_gpu_process_seconds": charged,
        **protocol_identity(),
    }


def slicing_known_answer() -> Record:
    """§7: 56 zero V columns; the criterion cut to 8 with fold reproduces the output, a random and a wrong cut do not."""
    rng = torch.Generator().manual_seed(SLICING_WEIGHT_SEED)
    shapes = ((64, 16), (64,), (16, 64), (16,))
    inside, bias_in, outside, bias_out = (torch.randn(shape, generator=rng) * 0.1 for shape in shapes)
    mask = torch.zeros(64)
    mask[list(SLICING_UNITS)] = 1.0
    module = Adapter((inside, bias_in, outside * mask[None, :], bias_out))
    site = torch.Generator().manual_seed(SLICING_SITE_SEED)
    read, base = (torch.randn((4, 16, 16, 16), generator=site) for _ in range(2))
    original = module.apply(read, base)

    def difference(adapter: Adapter) -> float:
        return float((adapter.apply(read, base) - original).abs().max())

    ranked = rank_cut(module, read, len(SLICING_UNITS))
    randomised = random_cut(module, read, len(SLICING_UNITS), RANDOM_CUT_SEED)
    zero = torch.tensor([unit for unit in range(64) if unit not in SLICING_UNITS][: len(SLICING_UNITS)])
    wrong = cut(module, zero, activation_means(module, read))
    rank_d, random_d, wrong_d = difference(ranked.folded), difference(randomised.folded), difference(wrong)
    return {
        "known_answer": ranked.survivors == SLICING_UNITS and rank_d <= 1e-6 and random_d > 1e-4 and wrong_d > 1e-4,
        "rank_cut_survivors": list(ranked.survivors),
        "rank_cut_max_difference": rank_d,
        "random_cut_max_difference": random_d,
        "wrong_cut_max_difference": wrong_d,
        "device": "cpu (supervisor, ruling R18 b)",
    }


def already_run(w: Workloads, raised: bool) -> Record:
    """Ruling R19: core work stage 2 already charged: the six N0 children and, on the default axis, the M-aff grids."""
    maff = 0.0 if raised else len(ANCHORS) * w.grid["aff"]
    n0 = len(ANCHORS) * w.fold_and_calibrations
    return {
        "N0": {"names": [f"noop-{s}" for s in ANCHORS], "seconds": n0},
        "M-aff": {"names": [] if raised else [f"maff-{s}" for s in ANCHORS], "seconds": maff},
        "seconds": n0 + maff,
        "remaining_core_seconds": stage_seconds("core", w) - n0 - maff,
    }


def admission(charged: float, w: Workloads, ceiling: float, done: float) -> Record:
    """§8 stage 4: charged so far plus the exact remaining core (core less `done`, R19) and the admitted extended work."""
    core = stage_seconds("core", w) - done
    over = core_over_ceiling(charged, core, ceiling)
    admitted = [] if over else admit_extended(charged, core, w, ceiling)
    extended = project_pilot(charged, tuple(admitted), w)
    return {
        "core_over_ceiling": over,
        "admitted_extended": list(admitted),
        "projection": {
            "method": "charged so far plus the remaining core (already-run work subtracted) and admitted extended items",
            "charged": charged,
            "remaining_lines": {"core": core, **mapping(extended["remaining_lines"])},
            "projected_gpu_process_seconds": number(extended["projected_gpu_process_seconds"]) + core,
        },
        "ceiling_gpu_process_seconds": ceiling,
    }


def axis(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def raise_only(raised: tuple[int, ...], default: tuple[int, ...]) -> tuple[int, ...]:
    if len(raised) != len(default) or any(a < b for a, b in zip(raised, default, strict=True)):
        raise PilotHold(f"schedule axis {raised} is not a raise-only extension of {default}")
    return raised


def job(name: str, action: str, seed: int, lineage: Lineage, *flags: str, training: Path | None = None) -> Job:
    """An M0 worker job; its card is assigned by position in Run.run."""
    extra = ("--lineage", lineage, "--scale", str(width_scale(lineage)), *flags)
    return Job(name, action, seed, 0, 0, training, None, extra)


def anchor_host(seed: int) -> str:
    return "replay-3000-gpu0" if seed == ANCHORS[0] else f"replay-{seed}"


@dataclass
class Run:
    root: Path
    supervisor: Supervisor

    def run(self, jobs: list[Job]) -> None:
        """One process per card: job i runs on card i % 2, two at a time, in order."""
        placed = [dataclasses.replace(item, gpu=index % 2) for index, item in enumerate(jobs)]
        for start in range(0, len(placed), 2):
            self.supervisor.run(tuple(placed[start : start + 2]))

    def path(self, name: str) -> Path:
        return self.root / name

    def payload(self, name: str, action: str, seed: int, lineage: Lineage) -> Record:
        return receipt(self.root, name, action, seed, lineage)

    def charges(self, names: list[str]) -> tuple[float, ...]:
        return ledger_charges(self.supervisor.ledger, tuple(names))


# --- stage 1 -----------------------------------------------------------------------------------------------------


def stage_replay(run: Run) -> Record:
    """Six anchors replayed against attempt 4 (3000 on both cards), the acceptance matrix (i-iv) and timing."""
    replays = [
        job("replay-3000-gpu0", "replay", ANCHORS[0], "anchor"),
        job("replay-3000-gpu1", "replay", ANCHORS[0], "anchor"),
        *(job(f"replay-{seed}", "replay", seed, "anchor") for seed in ANCHORS[1:]),
    ]
    run.run(replays)
    finals: Record = {}
    for item in replays:
        payload = run.payload(item.name, "replay", item.seed, "anchor")
        bound(run.path(item.name), payload, TRAINING_FILES)
        if payload["final_state_sha256"] != ATTEMPT4_FINAL_STATE[item.seed]:
            raise PilotHold(f"bitwise host replay mismatch with attempt 4's pinned state: {item.name}")
        finals[item.name] = payload["final_state_sha256"]
    cards = [run.payload(f"replay-3000-gpu{card}", "replay", ANCHORS[0], "anchor") for card in (0, 1)]
    for key in REPLAY_KEYS:
        if cards[0][key] != cards[1][key]:
            raise PilotHold(f"both-card bitwise host replay mismatch: {key}")
    checks = [
        job(f"acceptance-{card}", "acceptance", ANCHORS[0], "anchor", training=run.path(f"replay-3000-gpu{card}"))
        for card in (0, 1)
    ]
    timing = [
        job(f"timing-{i}", "timing", ANCHORS[0], "anchor", training=run.path(anchor_host(ANCHORS[0]))) for i in range(3)
    ]
    run.run(checks + timing)
    accepted = [run.payload(item.name, "acceptance", ANCHORS[0], "anchor") for item in checks]
    for card, record in enumerate(accepted):
        for key, label in ACCEPTANCE.items():
            if record[key] is not True:
                raise InstrumentHold(f"GPU acceptance {label} failed on card {card}")
    for key in ("straight_epoch_state_sha256", "checkpoint_rate"):
        if accepted[0][key] != accepted[1][key]:
            raise PilotHold(f"GPU acceptance replay differs across cards: {key}")
    return {
        "attempt4_final_state_sha256": finals,
        "both_cards_bitwise": True,
        "acceptance": list(accepted),
        "timing": [run.payload(item.name, "timing", ANCHORS[0], "anchor") for item in timing],
        "charged_gpu_process_seconds": run.supervisor.charged,
    }


# --- stage 2 -----------------------------------------------------------------------------------------------------


def fold_check(run: Run, name: str, modules: str) -> Record:
    """Acceptance (v), rulings R13/R16: the first anchor's selected M-aff fold residual recomputed on card."""
    seed = ANCHORS[0]
    flags = ("--noop-root", str(run.path(f"noop-{seed}")), "--modules-root", str(run.path(modules)))
    run.run([job(name, "fold-check", seed, "anchor", *flags)])
    payload = run.payload(name, "fold-check", seed, "anchor")
    if (
        payload["fold_residual_matches_receipt"] is not True
        or payload["receipt_r_aff"] != run.payload(modules, "modules", seed, "anchor")["r_aff"]
    ):
        raise InstrumentHold(f"GPU acceptance (v): the M-aff fold residual recomputed on card differs from {modules}")
    return payload


def stage_instrument(run: Run) -> Record:
    """P1 known answer, zero-module identity, slicing known answer, N0, and the M-aff fold residual on every anchor."""
    run.run(
        [job(f"instrument-{s}", "instrument", s, "anchor", training=run.path(anchor_host(s))) for s in ANCHORS]
        + [job(f"noop-{s}", "noop", s, "anchor", training=run.path(anchor_host(s))) for s in ANCHORS]
    )
    records = [run.payload(f"instrument-{s}", "instrument", s, "anchor") for s in ANCHORS]
    check_instrument(tuple(mapping(record["posthoc"]) for record in records))
    slicing = slicing_known_answer()
    if slicing["known_answer"] is not True:
        raise InstrumentHold("V0 slicing known answer failed")
    default = ("--checkpoints", axis(CHECKPOINTS), "--checkpoints-b", axis(CHECKPOINTS_B))
    maff = [
        job(f"maff-{s}", "modules", s, "anchor", "--arms", "M-aff", "--noop-root", str(run.path(f"noop-{s}")), *default)
        for s in ANCHORS
    ]
    run.run(maff[:2])
    checked = fold_check(run, f"fold-check-{ANCHORS[0]}", f"maff-{ANCHORS[0]}")
    run.run(maff[2:])
    residuals: Record = {}
    certified: Record = {}
    n0: Record = {}
    for s in ANCHORS:
        payload = run.payload(f"maff-{s}", "modules", s, "anchor")
        schedule_matches(f"maff-{s}", payload, CHECKPOINTS, CHECKPOINTS_B)
        arm = mapping(mapping(payload["arms"])["M-aff"])
        if arm["status"] != "available" or payload["r_aff"] is None:
            raise InstrumentHold(f"M-aff unavailable on anchor {s}; the V0 fold residual cannot be read")
        r_aff = number(payload["r_aff"])
        if abs(r_aff) > FOLD_TOLERANCE:
            raise InstrumentHold(f"V0 fold residual clause: |r_aff| = {abs(r_aff)} > {FOLD_TOLERANCE} on anchor {s}")
        residuals[str(s)] = r_aff
        certified[str(s)] = arm["certified"] is True
        n0[str(s)] = number(mapping(payload["n0"])["development_loss"])
    return {
        "materiality": materiality_report(tuple(mapping(record["materiality"]) for record in records)),
        "posthoc": [record["posthoc"] for record in records],
        "instrument_passed": True,
        "a1_within_1e-4": True,
        "zero_module_identity": True,
        "zero_module_identity_evidence": "every instrument child's materiality ran fit.identity_check (ruling R18 c)",
        "slicing": slicing,
        "fold_residuals": residuals,
        "fold_tolerance": FOLD_TOLERANCE,
        "fold_check": checked,
        "maff_certified": certified,
        "n0_development_loss": n0,
        "charged_gpu_process_seconds": run.supervisor.charged,
    }


# --- stage 3 -----------------------------------------------------------------------------------------------------


def train_hosts(run: Run, lineages: tuple[Lineage, ...]) -> dict[str, list[tuple[int, int, float]]]:
    """Train each lineage's six hosts, apply N0, evaluate N0 on development (R18 e): (seed, parameters, N0 loss)."""
    hosts = [(lineage, seed) for lineage in lineages for seed in BLOCKS[lineage]]

    def action(lineage: Lineage) -> str:
        return "cooled20" if lineage == "cooled-20" else "train"

    run.run(
        [
            job(
                f"{lineage}-{seed}",
                action(lineage),
                seed,
                lineage,
                "--epochs",
                "20" if lineage == "cooled-20" else "10",
            )
            for lineage, seed in hosts
        ]
    )
    parameters: dict[int, int] = {}
    for lineage, seed in hosts:
        payload = run.payload(f"{lineage}-{seed}", action(lineage), seed, lineage)
        bound(run.path(f"{lineage}-{seed}"), payload, TRAINING_FILES)
        parameters[seed] = integer(payload["parameters"])
    run.run([job(f"noop-{s}", "noop", s, lineage, training=run.path(f"{lineage}-{s}")) for lineage, s in hosts])
    run.run(
        [
            job(f"evaluate-{s}", "modules", s, lineage, "--arms", "R-temp", "--noop-root", str(run.path(f"noop-{s}")))
            for lineage, s in hosts
        ]
    )
    result: dict[str, list[tuple[int, int, float]]] = {}
    for lineage, seed in hosts:
        payload = run.payload(f"evaluate-{seed}", "modules", seed, lineage)
        loss = number(mapping(payload["n0"])["development_loss"])
        result.setdefault(lineage, []).append((seed, parameters[seed], loss))
    return result


def points_of(hosts: list[tuple[int, int, float]]) -> tuple[float, tuple[float, ...]]:
    counts = {count for _, count, _ in hosts}
    if len(counts) != 1:
        raise PilotHold("one lineage's hosts disagree on their parameter count")
    return float(counts.pop()), tuple(loss for _, _, loss in hosts)


def stage_hosts(run: Run, reference: list[tuple[int, int, float]]) -> tuple[Record, dict[float, tuple[float, ...]]]:
    """Frontier 0.5 and 1.5 hosts, T-cool resumes, S-static hosts; the core frontier slope."""
    frontier = train_hosts(run, ("frontier-0.5", "frontier-1.5"))
    run.run(
        [job(f"tcool-{s}", "tcool", s, "anchor", training=run.path(anchor_host(s))) for s in ANCHORS]
        + [job(f"static-{s}", "static", s, "anchor") for s in ANCHORS]
    )
    tcool: Record = {}
    static: Record = {}
    for s in ANCHORS:
        resumed = run.payload(f"tcool-{s}", "tcool", s, "anchor")
        if resumed["source_final_state_sha256"] != ATTEMPT4_FINAL_STATE[s]:
            raise PilotHold(f"T-cool on anchor {s} resumed another host")
        tcool[str(s)] = resumed["final_state_sha256"]
        payload = run.payload(f"static-{s}", "static", s, "anchor")
        bound(run.path(f"static-{s}"), payload, STATIC_FILES)
        static[str(s)] = payload["static_sha256"]
    points = dict(points_of(hosts) for hosts in (frontier["frontier-0.5"], reference, frontier["frontier-1.5"]))
    slope = stats.frontier_slope(points, BOOTSTRAP_SEED)
    record: Record = {
        "frontier": {lineage: [list(host) for host in hosts] for lineage, hosts in frontier.items()},
        "tcool_final_state_sha256": tcool,
        "static_sha256": static,
        "frontier_slope": slope.report(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "charged_gpu_process_seconds": run.supervisor.charged,
    }
    return record, points


# --- stage 4 -----------------------------------------------------------------------------------------------------


def stage_admission(
    run: Run, certified: list[bool], timing: tuple[Record, ...], fixed_v0: float
) -> tuple[tuple[int, ...], tuple[int, ...], list[str], Workloads]:
    """The schedule axis (R15), the measured workloads (R17), the projection and the extended admission."""
    fraction = sum(certified) / len(certified)
    raised = fraction < CERTIFIED_FRACTION
    factor = RAISE_FACTOR if raised else 1
    publish(
        run.path("04-schedule.json"),
        {
            "rule": "raise iff stage-2 M-aff certifies (plateau 800 vs 3,200) on fewer than CERTIFIED_FRACTION of "
            "the six anchors; the raised axis doubles every checkpoint, M-small-B's in step (ruling R15)",
            "maff_certified_fraction": fraction,
            "certified_fraction_required": CERTIFIED_FRACTION,
            "raised": raised,
            "checkpoints": [factor * c for c in CHECKPOINTS],
            "checkpoints_b": [factor * c for c in CHECKPOINTS_B],
            "default_checkpoints": list(CHECKPOINTS),
            "default_checkpoints_b": list(CHECKPOINTS_B),
        },
    )
    # Every later arm reads its axis from the published receipt.
    schedule = read_record(run.path("04-schedule.json"))
    checkpoints = raise_only(tuple(integer(c) for c in sequence(schedule["checkpoints"])), CHECKPOINTS)
    checkpoints_b = raise_only(tuple(integer(c) for c in sequence(schedule["checkpoints_b"])), CHECKPOINTS_B)
    if checkpoints_b != tuple(2 * c for c in checkpoints):
        raise PilotHold("M-small-B's axis must stay at twice the module axis (matched steps and example passes)")
    charges = {
        "host:0.5": run.charges([f"frontier-0.5-{s}" for s in BLOCKS["frontier-0.5"]]),
        "host:1.0": run.charges([anchor_host(s) for s in ANCHORS] + ["replay-3000-gpu1"]),
        "host:1.5": run.charges([f"frontier-1.5-{s}" for s in BLOCKS["frontier-1.5"]]),
        "tcool": run.charges([f"tcool-{s}" for s in ANCHORS]),
        "static": run.charges([f"static-{s}" for s in ANCHORS]),
        "aff": run.charges([f"maff-{s}" for s in ANCHORS]),
        "fold_and_calibrations": run.charges([f"noop-{s}" for s in ANCHORS]),
    }
    w, sources = derive_workloads(timing, charges, fixed_v0=fixed_v0, ratio=checkpoints[-1] / CHECKPOINTS[-1])
    done = already_run(w, raised)
    record = admission(run.supervisor.charged, w, M0_PILOT_SECONDS, number(done["seconds"]))
    record.update(
        {
            "schedule": {"checkpoints": list(checkpoints), "checkpoints_b": list(checkpoints_b), "raised": raised},
            "workloads": workloads_record(w),
            "workload_sources": sources,
            "already_run": {**done, "V0": "stages 1-2, in charged", "hosts": "stage 3, in charged"},
        }
    )
    publish(run.path("04-admission.json"), record)
    if record["core_over_ceiling"] is True:
        raise FeasibilityHold("the core alone projects over the 14,400 s ceiling; the pilot stops before V1")
    return checkpoints, checkpoints_b, [text(item) for item in sequence(record["admitted_extended"])], w


# --- stage 5 -----------------------------------------------------------------------------------------------------


def stage_core(
    run: Run, schedule: tuple[tuple[int, ...], tuple[int, ...]], admitted: list[str], raised: bool
) -> list[Anchor]:
    """Every core case on the six anchors; the axis from 04-schedule.json; M-aff reused from stage 2 unless raised."""
    checkpoints, checkpoints_b = schedule
    flags = ("--checkpoints", axis(checkpoints), "--checkpoints-b", axis(checkpoints_b))
    wanted = stats.PILOT_ARMS | STATIC_ARMS | {item for item in ("M-exp", "lottery", "R-all") if item in admitted}
    if not raised:
        wanted -= {"M-aff"}
    module_arms = ",".join(name for name in MODULE_ARMS if name in wanted)
    retune_arms = ",".join(name for name in RETUNE_ARMS if name in wanted)

    def roots(s: int, *kinds: str) -> tuple[str, ...]:
        return tuple(value for kind in kinds for value in (f"--{kind}-root", str(run.path(f"{kind}-{s}"))))

    modules = [
        job(f"modules-{s}", "modules", s, "anchor", "--arms", module_arms, *roots(s, "noop", "tcool", "static"), *flags)
        for s in ANCHORS
    ]
    retunes = [
        job(f"retune-{s}", "retune", s, "anchor", "--arms", retune_arms, *roots(s, "noop"), *flags) for s in ANCHORS
    ]
    run.run(modules[:2])
    if raised:
        fold_check(run, f"fold-check-raised-{ANCHORS[0]}", f"modules-{ANCHORS[0]}")
    run.run(modules[2:] + retunes)
    anchors: list[Anchor] = []
    for s in ANCHORS:
        sources = [source(run.root, f"modules-{s}", "modules", s, "anchor")]
        sources.append(source(run.root, f"retune-{s}", "retune", s, "anchor"))
        for item in sources:
            schedule_matches(item.name, item.payload, checkpoints, checkpoints_b)
        if not raised:
            sources.append(source(run.root, f"maff-{s}", "modules", s, "anchor"))
        anchors.append(anchor(s, tuple(sources), stats.PILOT_ARMS))
    return anchors


def spreads(anchors: list[Anchor], quantities: list[Record], acceptance: list[Json], slicing: bool) -> Record:
    """V0 at aggregation, then the per-anchor SDs of D1, D2, D3, D5 and S-static's admission, before any median."""
    v0 = verdicts.v0(
        {"both_cards_bitwise": True},
        {"a1_within_1e-4": True, "b3_fraction_every_anchor": True},
        {
            "zero_module_identity": True,
            "printed_counts": all(item.report["printed_counts"] is True for item in anchors),
            "complement_hash": all(item.report["complement_unchanged"] is True for item in anchors),
            "static_step0": all(mapping(record)["static_step0_bitwise"] is True for record in acceptance),
        },
        {"known_answer": slicing},
        tuple(item.result.fold_residual for item in anchors),
    )
    if v0["passed"] is not True:
        raise InstrumentHold(f"V0 failed at aggregation: {v0['failed']}")
    sd: Record = {}
    reasons: Record = {}
    for key in SPREAD_KEYS:
        values = [q[key] for q in quantities]
        sd[key] = stats.sd([number(v) for v in values if v is not None]) if None not in values else None
        reasons[key] = None if sd[key] is not None else f"{key} undefined on an anchor"
    sd_d5 = sd["D5"]
    certified: dict[str, list[bool]] = {}
    for item in anchors:
        for name, arm in item.result.arms.items():
            if arm.certified is not None:
                certified.setdefault(name, []).append(arm.certified)
    return {
        "v0": v0,
        "sd": sd,
        "sd_reasons": reasons,
        "sd_d5": sd_d5,
        "s_static_sd_limit": S_STATIC_SD_LIMIT,
        "static_admitted": sd_d5 is not None and number(sd_d5) <= S_STATIC_SD_LIMIT,
        "plateau_fractions": {name: sum(flags) / len(flags) for name, flags in sorted(certified.items())},
        "anchors": [{**item.report, "quantities": q} for item, q in zip(anchors, quantities, strict=True)],
    }


def starved_host(item: Source, seed: int) -> Record:
    """R11 for a descriptive item: an unavailable starved-pair arm is reported, never a stop."""
    records = mapping(item.payload["arms"])
    absent = sorted(
        name
        for name in stats.STARVED_PAIR_ARMS
        if name not in records or mapping(records[name])["status"] != "available"
    )
    if absent:
        causes: Record = {
            name: {key: mapping(records[name])[key] for key in ("cause", "error", "candidates")}
            for name in absent
            if name in records
        }
        return {"seed": seed, "status": "unavailable", "unavailable_arms": list(absent), "causes": causes}
    host = anchor(seed, (item,), stats.STARVED_PAIR_ARMS)
    quantities = stats.quantities(host.result, stats.STARVED_PAIR_ARMS)
    return {"status": "available", **host.report, "quantities": quantities}


def stage_extended(run: Run, admitted: list[str], schedule: tuple[tuple[int, ...], tuple[int, ...]]) -> Record:
    """The admitted §5 extended host items (R-all, M-exp and the lottery ran inside the anchors' children)."""
    flags = ("--checkpoints", axis(schedule[0]), "--checkpoints-b", axis(schedule[1]))
    record: Record = {"admitted": list(admitted)}
    if "starved-pair" in admitted:
        seeds = BLOCKS["frontier-0.5"]
        lineage: Lineage = "frontier-0.5"
        arms = ",".join(name for name in MODULE_ARMS if name in stats.STARVED_PAIR_ARMS)
        run.run([job(f"tcool-{s}", "tcool", s, lineage, training=run.path(f"{lineage}-{s}")) for s in seeds])
        run.run(
            [
                job(f"starved-{s}", "modules", s, lineage, "--arms", arms, "--noop-root", str(run.path(f"noop-{s}")),
                    "--tcool-root", str(run.path(f"tcool-{s}")), *flags)
                for s in seeds
            ]
        )  # fmt: skip
        record["starved-pair"] = [
            starved_host(source(run.root, f"starved-{s}", "modules", s, lineage), s) for s in seeds
        ]
    lineages = tuple(lineage for item, lineage in EXTENDED_HOSTS if item in admitted)
    if lineages:
        hosts_by_lineage = train_hosts(run, lineages)
        record["hosts"] = {key: [list(host) for host in hosts] for key, hosts in hosts_by_lineage.items()}
    return record


# --- stages 6 and 7 ----------------------------------------------------------------------------------------------


def confirmatory_count(quantities: list[Record], w: Workloads, static_admitted: bool) -> Record:
    """§9 stage 6: n = max(32, n*) sized on D1 only; the projection includes the fixed V0 term (R20 half-widths)."""
    for seed, q in zip(ANCHORS, quantities, strict=True):
        if q["D1"] is None:
            raise InstrumentHold(f"D1 is undefined on anchor {seed} ({q['D1_reason']}); the count cannot be sized")
    sd_d1 = stats.sd([number(q["D1"]) for q in quantities])
    required, n = stats.confirmatory_count(sd_d1)
    projection = project_confirmatory(n, w, stats.CONFIRMATORY_ARMS, w.fixed_v0)
    static = with_static(n, w, stats.CONFIRMATORY_ARMS | STATIC_ARMS, w.fixed_v0) if static_admitted else None
    return {
        "confirmatory_count": n,
        "n_star": required,
        "sd_d1": sd_d1,
        "projection": projection,
        "static_admitted": static_admitted,
        "static_projection": static,
        "within_ceiling": projection["within_ceiling"],
        **half_widths(quantities),
    }


def half_widths(quantities: list[Record]) -> Record:
    """§9 stage 6 (R20): D2, D3 and D5 at their achieved half-widths over the development anchors."""
    widths: Record = {}
    reasons: Record = {}
    for key in ("D2", "D3", "D5"):
        values = [q[key] for q in quantities]
        defined = None not in values
        widths[key] = (
            stats.achieved_half_width([number(v) for v in values if v is not None], BOOTSTRAP_SEED) if defined else None
        )
        reasons[key] = None if defined else f"{key} undefined on an anchor"
    return {
        "achieved_half_width": widths,
        "achieved_half_width_reasons": reasons,
        "achieved_half_width_development_only": True,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def median_or_none(values: list[Json]) -> Json:
    return stats.median([number(v) for v in values if v is not None]) if values and None not in values else None


def development_medians(quantities: list[Record], points: dict[float, tuple[float, ...]], extended: Record) -> Record:
    """§9 stage 7: medians for the record only; no verdict is read in the pilot."""
    gains = tuple(number(mapping(mapping(q["arms"])["M-sq"])["gain"]) for q in quantities)
    record: Record = {
        "development_only": True,
        "confirmatory_verdict": None,
        "medians": {key: median_or_none([q[key] for q in quantities]) for key in MEDIAN_KEYS},
        "frontier_fraction": stats.gain_fraction(gains, points, BOOTSTRAP_SEED).report(),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if "starved-pair" in extended:
        hosts = [mapping(host) for host in sequence(extended["starved-pair"])]
        starved = [mapping(host["quantities"]) for host in hosts if host["status"] == "available"]
        record["starved_pair_medians"] = (
            {key: median_or_none([q[key] for q in starved]) for key in ("D1", "D3")} if starved else None
        )
        record["starved_pair_hosts"] = {"available": len(starved), "run": len(hosts)}
    if "hosts" in extended:
        trained = mapping(extended["hosts"])
        widths = dict(points)
        for lineage in ("frontier-0.25", "frontier-0.75"):
            if lineage in trained:
                parsed = [
                    (integer(h[0]), integer(h[1]), number(h[2])) for h in map(sequence, sequence(trained[lineage]))
                ]
                count, losses = points_of(parsed)
                widths[count] = losses
        if len(widths) > len(points):
            record["frontier_descriptive"] = stats.frontier_slope_descriptive(widths, BOOTSTRAP_SEED).report()
        if "cooled-20" in trained:
            # The 20-epoch point never enters either frontier fit (§6).
            losses20 = [number(sequence(host)[2]) for host in sequence(trained["cooled-20"])]
            record["cooled_20_median_n0_loss"] = stats.median(losses20)
    return record


# --- the attempt -------------------------------------------------------------------------------------------------


def pilot(root: Path, data: Path, certificate: Path) -> int:
    """One M0 pilot attempt: certificate, cards, custody, then §9 stages 1-7 in order."""
    check_output_root(root)
    if root.exists():
        raise PilotHold("one attempt requires a new output root; no automatic resume")
    approved = read_record(certificate)
    identity = check_certificate(approved)
    environment = environment_identity()
    cards = check_cards()
    certificate_sha = file_sha(certificate)
    # Durable one-attempt custody across output roots and supervisor restarts; never removed or reset.
    claim_attempt(REPO / CUSTODY, root, dict(identity, environment=environment, certificate_sha256=certificate_sha))
    root.mkdir(parents=True, exist_ok=False)
    run = Run(root, Supervisor(root, data, ceiling=M0_PILOT_SECONDS, per_card=1, module=WORKER))
    stage = "opening"
    try:
        # Everything after custody is inside the attempt, so any failure publishes pilot-result.json.
        publish(
            root / "00-opening.json",
            {
                "cards": cards,
                "environment": environment,
                "implementation_commit": text(identity["commit"]),
                "certificate_sha256": certificate_sha,
                "custody": CUSTODY,
                "pilot_gpu_process_seconds_ceiling": M0_PILOT_SECONDS,
                "processes_per_card": 1,
                "worker": WORKER,
                **protocol_identity(),
            },
        )
        stage = "replay-timing"
        replay = stage_replay(run)
        publish(root / "01-replay-timing.json", replay)
        stage = "instrument"
        instrument = stage_instrument(run)
        publish(root / "02-instrument.json", instrument)
        fixed_v0 = run.supervisor.charged
        stage = "hosts"
        n0 = mapping(instrument["n0_development_loss"])
        reference = [(s, integer(run.payload(anchor_host(s), "replay", s, "anchor")["parameters"]), number(n0[str(s)]))
                     for s in ANCHORS]  # fmt: skip
        hosts, points = stage_hosts(run, reference)
        publish(root / "03-hosts.json", hosts)
        stage = "admission"
        certified = [value is True for value in mapping(instrument["maff_certified"]).values()]
        timing = tuple(mapping(record) for record in sequence(replay["timing"]))
        checkpoints, checkpoints_b, admitted, w = stage_admission(run, certified, timing, fixed_v0)
        raised = checkpoints != CHECKPOINTS
        stage = "core"
        anchors = stage_core(run, (checkpoints, checkpoints_b), admitted, raised)
        quantities = [stats.quantities(item.result, stats.PILOT_ARMS) for item in anchors]
        slicing = mapping(instrument["slicing"])["known_answer"] is True
        spread = spreads(anchors, quantities, sequence(replay["acceptance"]), slicing)
        publish(root / "05-spreads.json", spread)
        stage = "extended"
        extended = stage_extended(run, admitted, (checkpoints, checkpoints_b))
        publish(root / "05-extended.json", extended)
        stage = "confirmatory-count"
        count = confirmatory_count(quantities, w, spread["static_admitted"] is True)
        publish(root / "06-confirmatory-count.json", count)
        if count["within_ceiling"] is not True:
            raise FeasibilityHold("the confirmatory count projects over its ceiling; stop before freeze, John's choice")
        stage = "development-medians"
        publish(root / "07-development-medians.json", development_medians(quantities, points, extended))
        publish(
            root / "pilot-result.json",
            {
                "software_valid": True,
                "instrument_valid": True,
                "verdict": "pilot-clean; confirmatory verdict pending",
                "pilot_clean": True,
                "confirmatory_ready": False,
                "reason": "pilot-filled values must be reviewed and committed before the confirmatory run",
                "schedule": {"checkpoints": list(checkpoints), "checkpoints_b": list(checkpoints_b), "raised": raised},
                "admitted_extended": list(admitted),
                "confirmatory_count": count["confirmatory_count"],
                "static_admitted": spread["static_admitted"],
                "charged_gpu_process_seconds": run.supervisor.charged,
                **protocol_identity(),
            },
        )
        return 0
    except Exception as error:
        publish(root / "pilot-result.json", m0_stop_report(error, stage, run.supervisor.charged))
        return 2


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--review-certificate", type=Path, required=True)
    args = parser.parse_args()
    return pilot(args.output.resolve(), args.data_root.resolve(), args.review_certificate.resolve())


def _terminate(signum: int, _frame: FrameType | None) -> None:
    raise PilotHold(f"supervisor interrupted by signal {signum}; stop owned children and retain the attempt")


if __name__ == "__main__":
    raise SystemExit(main())
