"""One-attempt ordered pilot supervisor; conservative aggregate GPU process-time budget."""

from __future__ import annotations

import argparse
import datetime
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from sultai.trust import FitUnavailable

from .contracts import (
    CHEAP_RATES,
    PILOT_SECONDS,
    PREREG_SHA,
    RHO,
    FeasibilityHold,
    InstrumentHold,
    Json,
    OracleUnavailable,
    PilotHold,
    Record,
    anchor_count,
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
from .costs import project_cost
from .screen import BOTTLENECK_SEEDS, INTACT_SEEDS, REPO, source_identity

GPU_UUIDS = ("GPU-6198007d-505f-920d-088c-957818fc0988", "GPU-d5b931ca-445e-0e51-004f-ead3b68d6ece")


@dataclass(frozen=True)
class Job:
    name: str
    action: str
    seed: int
    width: int
    gpu: int
    training: Path | None = None
    magnitudes: Path | None = None


@dataclass
class Running:
    job: Job
    process: subprocess.Popen[bytes]
    start: float


def _signal_owned(process: subprocess.Popen[bytes], signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        # The child may exit between poll() and killpg(); it is still reaped below.
        pass


def check_handover() -> Record:
    now = datetime.datetime.now(datetime.timezone.utc)
    if now < datetime.datetime(2026, 10, 11, 8, tzinfo=datetime.timezone.utc):
        raise PilotHold("Sultai's authorized GPU boundary has not arrived")
    terminal: Record = {}
    for study in ("fleet-c1", "fleet-c1s"):
        root = Path("/home/john/simic-worktrees") / study / "runs" / study
        receipt = read_record(root / "launch-finished.json")
        units = [mapping(u) for u in sequence(receipt["units"])]
        if (
            len(units) != 768
            or len({integer(u["seed"]) for u in units}) != 768
            or any(u["returncode"] != 0 for u in units)
        ):
            raise PilotHold(f"{study} terminal receipt is not 768 successful unique units")
        terminal[study] = {
            "finished_unix": number(receipt["finished_unix"]),
            "terminal_sha256": file_sha(root / "launch-finished.json"),
            "units": 768,
        }
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    rows = [tuple(value.strip() for value in line.split(",")) for line in query.strip().splitlines()]
    if len(rows) != 2 or any(
        rows[i][0] != str(i) or rows[i][1] != GPU_UUIDS[i] or int(rows[i][2]) != 0 for i in range(2)
    ):
        raise PilotHold("pinned GPUs are unavailable or identities differ")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    if apps.strip():
        raise PilotHold("GPU compute processes already active; preserve them")
    return {
        "observed_utc": now.isoformat(),
        "terminal": terminal,
        "gpu_query": query,
        "compute_apps": apps,
        "owner_release": "Explicit Simic owner release relayed by Emmy before handover; no training killed.",
    }


def payload(root: Path, name: str) -> Record:
    result = read_record(root / name / "result.json")
    if result["software_valid"] is not True:
        raise PilotHold("child receipt is not software valid")
    return mapping(result["payload"])


def verify_replay(expected: Record, actual: Record) -> None:
    for key in ("epoch_state_sha256", "loss_stream_sha256", "final_state_sha256", "forward_backward"):
        if actual[key] != expected[key]:
            raise PilotHold(f"bitwise host replay mismatch: {key}")


class Supervisor:
    def __init__(self, root: Path, data: Path) -> None:
        self.root = root
        self.data = data
        self.charged = 0.0
        self.ledger: list[Json] = []
        self.serial = 0

    def run(self, jobs: tuple[Job, ...]) -> None:
        if (
            not jobs
            or any(j.gpu not in (0, 1) for j in jobs)
            or any(sum(j.gpu == gpu for j in jobs) > 3 for gpu in (0, 1))
            or len({j.name for j in jobs}) != len(jobs)
        ):
            raise PilotHold("unique jobs, pinned cards and reviewed maximum three processes/card required")
        remaining = PILOT_SECONDS - self.charged
        if remaining <= 10 * len(jobs):
            raise PilotHold("aggregate pilot GPU-hour cap exhausted")
        # n concurrent GPU processes consume n seconds of this conservative budget per second.
        # Charge wall from each process creation through wait, including load/idle/cleanup.
        deadline = time.monotonic() + remaining / len(jobs) - 10
        running: list[Running] = []
        failure: str | None = None
        try:
            for job in jobs:
                output = self.root / job.name
                if output.exists():
                    raise PilotHold("duplicate child output; pilot never resumes or overwrites")
                command = [
                    "nice",
                    "-n",
                    "19",
                    sys.executable,
                    "-B",
                    "-m",
                    "sultai_screen.screen",
                    job.action,
                    "--output",
                    str(output),
                    "--data-root",
                    str(self.data),
                    "--seed",
                    str(job.seed),
                    "--width",
                    str(job.width),
                ]
                if job.training is not None:
                    command += ["--training-root", str(job.training)]
                if job.magnitudes is not None:
                    command += ["--magnitudes", str(job.magnitudes)]
                env = dict(os.environ)
                env.update(
                    CUDA_VISIBLE_DEVICES=str(job.gpu),
                    CUBLAS_WORKSPACE_CONFIG=":4096:8",
                    PYTHONPATH=str(REPO / "src"),
                    PYTHONDONTWRITEBYTECODE="1",
                    OMP_NUM_THREADS="2",
                    MKL_NUM_THREADS="2",
                )
                started = time.monotonic()
                with (
                    (self.root / f"{job.name}.stdout").open("xb") as stdout,
                    (self.root / f"{job.name}.stderr").open("xb") as stderr,
                ):
                    process = subprocess.Popen(
                        command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, start_new_session=True
                    )
                running.append(Running(job, process, started))
                publish(
                    self.root / f"{job.name}.process.json",
                    {
                        "pid": process.pid,
                        "gpu": job.gpu,
                        "command": json_value(command),
                        "started_unix": time.time(),
                        "aggregate_budget_remaining_seconds": remaining,
                    },
                )
            while True:
                active = False
                for entry in running:
                    code = entry.process.poll()
                    if code is None:
                        active = True
                    elif code != 0:
                        receipt = self.root / entry.job.name / "failure.json"
                        if receipt.is_file():
                            failed = read_record(receipt)
                            if failed["software_valid"] is True and failed["instrument_valid"] is False:
                                failure = (
                                    f"instrument unavailable: {text(failed['error_type'])}: {text(failed['error'])}"
                                )
                                break
                        failure = f"child {entry.job.name} exited {code}"
                        break
                if failure is not None or not active:
                    break
                if time.monotonic() >= deadline:
                    failure = "aggregate GPU-hour timeout"
                    break
                time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))
        finally:
            # Signal every owned session before waiting: cleanup latency must not multiply
            # by the number of co-tenants. Reserve ten seconds/process before the deadline.
            for entry in running:
                if entry.process.poll() is None:
                    _signal_owned(entry.process, signal.SIGTERM)
            cleanup_deadline = time.monotonic() + 3
            for entry in running:
                if entry.process.poll() is None:
                    try:
                        entry.process.wait(timeout=max(0.001, cleanup_deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        _signal_owned(entry.process, signal.SIGKILL)
            for entry in running:
                if entry.process.poll() is None:
                    entry.process.wait()
            for entry in running:
                wall = time.monotonic() - entry.start
                self.charged += wall
                self.ledger.append(
                    {
                        "job": entry.job.name,
                        "pid": entry.process.pid,
                        "gpu": entry.job.gpu,
                        "returncode": entry.process.returncode,
                        "charged_gpu_process_seconds": wall,
                    }
                )
            publish(
                self.root / f"ledger-{self.serial:03}.json",
                {
                    "charged_gpu_process_seconds": self.charged,
                    "ceiling_gpu_process_seconds": PILOT_SECONDS,
                    "entries": list(self.ledger),
                },
            )
            self.serial += 1
        if failure is not None:
            if failure.startswith("instrument unavailable:"):
                raise InstrumentHold(failure)
            raise PilotHold(failure)
        if self.charged > PILOT_SECONDS:
            raise PilotHold("aggregate cap reached during cleanup; no further work")
        for entry in running:
            record = read_record(self.root / entry.job.name / "result.json")
            if record["software_valid"] is not True or text(mapping(record["identity"])["physical_gpu"]) != str(
                entry.job.gpu
            ):
                raise PilotHold("child success/card evidence missing")


def magnitude_decision(records: tuple[Record, ...]) -> Record:
    if len(records) != 6:
        raise PilotHold("all six development anchors required")
    chosen: Record = {}
    for family, ladders in (
        ("P1", ((0.1, 0.3), (0.03, 0.1))),
        ("P2", ((0.1,), (0.03,))),
        ("P3", ((0.25,), (0.5,), (1.0,))),
    ):
        found: tuple[float, ...] | None = None
        for magnitudes in ladders:
            means = [
                sum(number(mapping(mapping(r["rungs"])[family])[str(m)]) for m in magnitudes) / len(magnitudes)
                for r in records
            ]
            if all(gain >= 0.05 for gain in means):
                found = magnitudes
                break
        if found is None:
            raise InstrumentHold(f"{family} exhausted its materiality ladder; no additional rung authorized")
        if family == "P1":
            chosen["alpha_low"], chosen["alpha_high"] = found[0], found[1]
        else:
            chosen["s_min" if family == "P2" else "kappa"] = found[0]
    return chosen


def check_posthoc(records: tuple[Record, ...]) -> None:
    former = records[0]["former"]
    for record in records:
        if record["former"] != former:
            raise PilotHold("A7 artifact drift across workers")
        for family in ("P1", "P2", "P3", "P3prime"):
            cases = [mapping(case) for case in sequence(mapping(record["families"])[family])]
            if sum(number(c["noop_loss"]) - number(mapping(c["arm_losses"])["A1"]) for c in cases) / len(cases) < 0.05:
                raise InstrumentHold(f"pilot materiality failed: {family}")
            if any(
                number(mapping(c["arm_losses"])["A3"]) - number(mapping(c["arm_losses"])["A1"]) > 0.01 for c in cases
            ):
                raise InstrumentHold(f"supervised known answer failed: {family}")
        low = mapping(sequence(mapping(record["families"])["P1"])[0])
        fraction = low["oracle_gain_fraction_of_A3"]
        if fraction is None or number(fraction) < RHO:
            raise InstrumentHold("B3 did not recover rho of A3 on every development anchor")


def sizing_receipt(h_values: tuple[float, ...], per_bottleneck_seconds: float, fixed_seconds: float) -> Record:
    if len(h_values) != 6:
        raise PilotHold("six H-prime observations required for sizing")
    sd = statistics.stdev(h_values)
    required, count = anchor_count(sd)
    projected = fixed_seconds + count * per_bottleneck_seconds
    return {
        "sample_sd_h_prime": sd,
        "n_star": required,
        "n_p6": count,
        "projected_confirmatory_gpu_process_seconds": projected,
        "ceiling_seconds": 16 * 3600,
        "within_ceiling": projected <= 16 * 3600 and count <= 200,
    }


def check_timing(records: tuple[Record, ...]) -> Record:
    if len(records) < 3:
        raise PilotHold("at least three timing/replay receipts required")
    timing = records[0]
    for actual in records:
        if actual["fit_finite"] is not True:
            raise InstrumentHold("nonfinite timing reference fit; no timing/replay admission")
        if any(actual[key] != timing[key] for key in ("fit_state_sha256", "fit_trace_sha256")):
            raise PilotHold("B3 timing-fit replay mismatch")
    return timing


def stop_report(error: Exception, stage: str, charged: float) -> Record:
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
        "preregistration_sha256": PREREG_SHA,
    }


def check_certificate(approved: Record) -> Record:
    if (
        approved["preregistration_sha256"] != PREREG_SHA
        or approved["quality_passed"] is not True
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
    for entry in sequence(approved["evidence"]):
        evidence = mapping(entry)
        path = REPO / text(evidence["path"])
        if not path.resolve().is_relative_to(REPO.resolve()) or file_sha(path) != text(evidence["sha256"]):
            raise PilotHold("review evidence missing, outside repository or changed")
    if len(sequence(approved["evidence"])) < 3:
        raise PilotHold("quality, runtime and independent panel evidence required")
    return identity


def claim_attempt(path: Path, root: Path, identity: Record) -> None:
    path.parent.mkdir(exist_ok=True)
    publish(path, {"pid": os.getpid(), "output": str(root), "identity": identity, "started_unix": time.time()})


def check_output_root(root: Path) -> None:
    if not root.is_absolute() or not (
        root.resolve().is_relative_to((REPO / "runs").resolve())
        or root.resolve().is_relative_to(Path("/tmp").resolve())
    ):
        raise PilotHold(
            "stage raw pilot output in ignored runs/ or /tmp; publish sanitized docs/results evidence afterward"
        )


def selected_development(candidate: Record) -> float:
    value = candidate["development_loss"]
    if value is None:
        raise PilotHold("nonfinite development evaluation of a selected cheap candidate")
    return number(value)


def cheap_rate_decision(cheap: tuple[Record, ...]) -> Record:
    if len(cheap) != 6:
        raise PilotHold("six development anchors required for global cheap rate")
    mean_selection: Record = {}
    for rate in CHEAP_RATES:
        selected = [mapping(mapping(record["P6"])[str(rate)]) for record in cheap]
        values = [mapping(c["candidate"])["selection_loss"] for c in selected]
        mean_selection[str(rate)] = (
            statistics.mean(number(v) for v in values) if all(v is not None for v in values) else None
        )
    finite_rates = [rate for rate in CHEAP_RATES if mean_selection[str(rate)] is not None]
    if not finite_rates:
        raise InstrumentHold("no cheap rate is finite on all development anchors")
    eta = min(finite_rates, key=lambda rate: number(mean_selection[str(rate)]))
    fixed: list[Json] = []
    for record in cheap:
        status: Record = {"seed": integer(record["seed"])}
        for family in ("P5", "P6"):
            candidate = mapping(mapping(record[family])[str(eta)])
            if mapping(candidate["candidate"])["selection_loss"] is None:
                if candidate["development_evaluated"] is not False or candidate["development_loss"] is not None:
                    raise PilotHold("failed fit carries inconsistent development evidence")
                status[family] = {"status": "fit-unavailable", "development_loss": None}
            else:
                status[family] = {"status": "evaluated", "development_loss": selected_development(candidate)}
        fixed.append(status)
    return {"eta_B": eta, "mean_selection_loss": mean_selection, "fixed_rate_results": fixed}


def pilot(root: Path, data: Path, certificate: Path) -> int:
    check_output_root(root)
    if root.exists():
        raise PilotHold("one attempt requires a new output root; no automatic resume")
    approved = read_record(certificate)
    identity = check_certificate(approved)
    source_commit = text(identity["commit"])
    handover = check_handover()
    # Durable one-attempt custody across output paths and supervisor restarts.
    # Never remove/reset this receipt to make a second invocation appear fresh.
    attempt = REPO / ".weft/gate3-pilot-attempt.json"
    claim_attempt(attempt, root, identity)
    root.mkdir(parents=True, exist_ok=False)
    publish(
        root / "admission.json",
        {
            "handover": handover,
            "implementation_commit": source_commit,
            "certificate_sha256": file_sha(certificate),
            "preregistration_sha256": PREREG_SHA,
            "pilot_gpu_hour_ceiling": 3,
            "oracle_seeds": 3,
            "confirmatory_cotenancy_processes_per_card": 3,
        },
    )
    supervisor = Supervisor(root, data)
    stage = "timing-replay"
    try:
        # The six intact and six bottleneck hosts share seed labels only for a descriptive
        # pilot deficit reference; the confirmatory ranges remain disjoint as preregistered.
        for index, seed in enumerate(INTACT_SEEDS):
            supervisor.run(
                (
                    Job(f"intact-{seed}", "train", seed, 0, index % 2),
                    Job(f"bottle4-{seed}", "train", seed, 4, (index + 1) % 2),
                )
            )
        original = payload(root, "intact-3000")
        for name, gpu in (("replay-card0", 0), ("replay-card1-a", 1), ("replay-card1-b", 1)):
            supervisor.run((Job(name, "train", 3000, 0, gpu),))
            verify_replay(original, payload(root, name))
        for gpu in (0, 1):
            jobs = tuple(Job(f"cotenancy-{gpu}-{i}", "train", 3000, 0, gpu) for i in range(3))
            supervisor.run(jobs)
            for job in jobs:
                verify_replay(original, payload(root, job.name))
        for name, gpu in (("timing-0-a", 0), ("timing-0-b", 0), ("timing-1", 1)):
            supervisor.run((Job(name, "timing", 3000, 0, gpu, root / "intact-3000"),))
        fit_cotenancy = tuple(
            Job(f"cotenancy-fit-{gpu}-{i}", "timing", 3000, 0, gpu, root / "intact-3000")
            for gpu in (0, 1)
            for i in range(3)
        )
        supervisor.run(fit_cotenancy)
        timing_records = tuple(payload(root, name) for name in ("timing-0-a", "timing-0-b", "timing-1")) + tuple(
            payload(root, job.name) for job in fit_cotenancy
        )
        timing = check_timing(timing_records)
        publish(
            root / "01-timing-replay-decision.json",
            {
                "host_bitwise_replay": True,
                "fit_bitwise_replay": True,
                "oracle_seeds": 3,
                "seed_reduction": "none",
                "timing": timing,
                "charged_gpu_process_seconds": supervisor.charged,
            },
        )
        stage = "materiality"
        for index, seed in enumerate(INTACT_SEEDS):
            supervisor.run((Job(f"materiality-{seed}", "materiality", seed, 0, index % 2, root / f"intact-{seed}"),))
        chosen = magnitude_decision(tuple(payload(root, f"materiality-{seed}") for seed in INTACT_SEEDS))
        publish(root / "02-magnitudes.json", chosen)
        stage = "posthoc-known-answer"
        for index, seed in enumerate(INTACT_SEEDS):
            supervisor.run(
                (
                    Job(
                        f"posthoc-{seed}",
                        "posthoc",
                        seed,
                        0,
                        index % 2,
                        root / f"intact-{seed}",
                        root / "02-magnitudes.json",
                    ),
                )
            )
        posthoc = tuple(payload(root, f"posthoc-{seed}") for seed in INTACT_SEEDS)
        check_posthoc(posthoc)
        publish(root / "03-posthoc-known-answer.json", {"passed": True, "former": posthoc[0]["former"]})
        stage = "bottleneck-known-answer"
        width = 4
        bottles: tuple[Record, ...] = ()
        for trial in (4, 2):
            if trial == 2:
                for index, seed in enumerate(BOTTLENECK_SEEDS):
                    supervisor.run((Job(f"bottle2-{seed}", "train", seed, 2, index % 2),))
            for index, seed in enumerate(BOTTLENECK_SEEDS):
                supervisor.run(
                    (
                        Job(
                            f"oracle{trial}-{seed}",
                            "bottleneck",
                            seed,
                            trial,
                            index % 2,
                            root / f"bottle{trial}-{seed}",
                        ),
                    )
                )
            bottles = tuple(payload(root, f"oracle{trial}-{seed}") for seed in BOTTLENECK_SEEDS)
            if all(number(record["paired_gain_difference"]) >= 0.05 for record in bottles):
                width = trial
                break
        else:
            raise InstrumentHold("finite w4 and w2 failed the known answer; w8 needs explicit degeneration disposition")
        publish(
            root / "04-width.json",
            {"width": width, "known_answer_passed_every_anchor": True, "oracle_reports": list(bottles)},
        )
        stage = "cheap-rate"
        for index, seed in enumerate(BOTTLENECK_SEEDS):
            supervisor.run((Job(f"cheap-{seed}", "cheap", seed, width, index % 2, root / f"bottle{width}-{seed}"),))
        cheap = tuple(payload(root, f"cheap-{seed}") for seed in BOTTLENECK_SEEDS)
        rate_decision = cheap_rate_decision(cheap)
        eta = number(rate_decision["eta_B"])
        publish(root / "05-cheap-rate.json", rate_decision)
        stage = "spread-sizing"
        h = []
        paired_h = []
        for bottle, bank in zip(bottles, cheap, strict=True):
            family_values: dict[str, float] = {}
            for family in ("P5", "P6"):
                candidates = [mapping(mapping(bank[family])[str(rate)]) for rate in CHEAP_RATES]
                finite = [c for c in candidates if mapping(c["candidate"])["selection_loss"] is not None]
                if not finite:
                    raise InstrumentHold("all cheap candidates failed on an anchor")
                best = min(finite, key=lambda c: number(mapping(c["candidate"])["selection_loss"]))
                family_values[family] = selected_development(best) - number(mapping(bottle[family])["oracle_loss"])
            h.append(family_values["P6"])
            paired_h.append(family_values["P6"] - family_values["P5"])
        sd = statistics.stdev(h)
        required, count = anchor_count(sd)
        projection = project_cost(
            count,
            timing=timing_records,
            intact_seconds=tuple(
                number(read_record(root / f"intact-{seed}/result.json")["wall_seconds"]) for seed in INTACT_SEEDS
            ),
            bottleneck_seconds=tuple(
                number(read_record(root / f"bottle{width}-{seed}/result.json")["wall_seconds"])
                for seed in BOTTLENECK_SEEDS
            ),
            oracle_worker_seconds=tuple(
                number(read_record(root / f"oracle{width}-{seed}/result.json")["wall_seconds"])
                for seed in BOTTLENECK_SEEDS
            ),
            posthoc_seconds=tuple(
                number(read_record(root / f"posthoc-{seed}/result.json")["wall_seconds"]) for seed in INTACT_SEEDS
            ),
        )
        projected = number(projection["projected_gpu_process_seconds"])
        sizing: Record = {
            "sample_sd_h_prime": sd,
            "n_star": required,
            "n_p6": count,
            "projection": projection,
            "projected_confirmatory_gpu_process_seconds": projected,
            "ceiling_seconds": 16 * 3600,
            "within_ceiling": projected <= 16 * 3600 and count <= 200,
        }
        publish(root / "06-spread-sizing.json", sizing)
        if sizing["within_ceiling"] is not True:
            raise FeasibilityHold(
                "required anchor count exceeds measured compute/allocated seed ceiling; operator decision required"
            )
        publish(
            root / "07-development-medians.json",
            {
                "h_prime_median": statistics.median(h),
                "specificity_median": statistics.median(paired_h),
                "development_only": True,
                "confirmatory_verdict": None,
            },
        )
        publish(
            root / "pilot-result.json",
            {
                "software_valid": True,
                "instrument_valid": True,
                "verdict": "pilot-clean; confirmatory verdict pending",
                "pilot_clean": True,
                "confirmatory_ready": False,
                "reason": "pilot-filled preregistration and evidence must be reviewed and committed before confirmation",
                "width": width,
                "magnitudes": chosen,
                "eta_B": eta,
                "sizing": sizing,
                "former": posthoc[0]["former"],
                "charged_gpu_process_seconds": supervisor.charged,
                "preregistration_sha256": PREREG_SHA,
            },
        )
        return 0
    except Exception as error:
        publish(
            root / "pilot-result.json",
            stop_report(error, stage, supervisor.charged),
        )
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
    raise PilotHold(f"supervisor interrupted by signal {signum}; stop owned children and retain attempt")


if __name__ == "__main__":
    raise SystemExit(main())
