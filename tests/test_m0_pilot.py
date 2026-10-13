"""CPU controls for the M0 pilot supervisor: custody, stage order, receipts, ceiling and admission (Task 17).

Children are patched exactly where Gate 3's supervisor test patches them (`sultai_screen.pilot.subprocess.Popen`):
the fake writes the M0 receipt each stage reads, then a real `/bin/true` process stands in for the GPU child. The
GPU, `nvidia-smi` and `git` are never invoked.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import Tensor

from sultai.trust import InputDataError
from sultai_screen.contracts import (
    FeasibilityHold,
    InstrumentHold,
    Json,
    PilotHold,
    Record,
    file_sha,
    mapping,
    read_record,
)
from sultai_screen.m0 import pilot as m0_pilot
from sultai_screen.m0.confirmatory import CUSTODY as CONFIRMATORY_CUSTODY
from sultai_screen.m0.confirmatory import RESTART_CUSTODY, Guarded, confirmatory, whitelist
from sultai_screen.m0.costs import (
    EXTENDED_ORDER,
    M0_CONFIRMATORY_STATIC_SECONDS,
    STATIC_ARMS,
    Workloads,
    admit_extended,
    extended_cost,
    extended_limit,
    measured_confirmatory,
    per_anchor,
    stage_seconds,
)
from sultai_screen.m0.identity import (
    CHECKPOINTS,
    CHECKPOINTS_B,
    M0_CONFIRMATORY_SECONDS,
    M0_PILOT_SECONDS,
    protocol_identity,
)
from sultai_screen.m0.modules import FRONTIER_WIDTHS
from sultai_screen.m0.pilot import (
    ANCHORS,
    CUSTODY,
    admission,
    already_run,
    check_cards,
    check_certificate,
    job,
    m0_stop_report,
    pilot,
    slicing_known_answer,
    spreads,
)
from sultai_screen.m0.receipts import Source, anchor, arm_vectors, derive_workloads, receipt, workloads_record
from sultai_screen.m0.stats import CONFIRMATORY_ARMS, KNOWN_ARMS, PILOT_ARMS, quantities
from sultai_screen.m0.worker import ATTEMPT4_FINAL_STATE, MODULE_ARMS, RETUNE_ARMS, ZERO_MODULE_KINDS
from sultai_screen.pilot import GPU_UUIDS, Job, Supervisor

REAL_POPEN = subprocess.Popen
SRC = Path(__file__).resolve().parents[1] / "src"
N = 20
PARAMETERS = {0.25: 23_066, 0.5: 85_274, 0.75: 166_538, 1.0: 284_634, 1.5: 613_018}
FIXED_ARMS = frozenset(("T-cool", "R-temp", "R-cal", "S-static"))
RETUNE = frozenset(("R-272s", "R-aff", "R-all"))
IDENTITY: Record = {"commit": "0" * 40, "source_sha256": "1" * 64, **protocol_identity()}
RECEIPTS = (
    "00-opening.json",
    "01-replay-timing.json",
    "02-instrument.json",
    "03-hosts.json",
    "04-schedule.json",
    "04-admission.json",
    "05-spreads.json",
    "06-confirmatory-count.json",
    "07-development-medians.json",
    "pilot-result.json",
)


def hidden(arm: str) -> int | None:
    if arm.startswith("M-wide→"):
        return int(arm.split("→")[1].split(":")[0])
    return 8 if arm == "lottery" else None


def noop_loss(seed: int, scale: float) -> Tensor:
    return 1.5 - 0.2 * scale + 0.005 * (seed % 6) + 0.001 * torch.arange(N, dtype=torch.float32)


def arm_record(
    arm: str, seed: int, n0: Tensor, *, r_aff: float, certified: bool, boost: float = 0.0
) -> tuple[Record, dict[str, Tensor]]:
    index = sorted(KNOWN_ARMS).index(arm)
    gain = 0.02 + 0.001 * (index % 10) + 0.002 * ((seed + index) % 4) + boost
    loss = n0 - gain
    correct = (torch.arange(N) % 3 != 0).float()
    vectors = {"loss": loss, "correct": correct}
    record: Record = {
        "status": "available",
        "development_loss": float(loss.double().mean()),
        "development_accuracy": float(correct.double().mean()),
        "certified": None,
        "rate": None,
        "steps": None,
        "edge_of_grid": None,
        "step0_selection_loss": None,
        "retained_parameters": 272,
        "report": None,
        "harm_inventory": {},
    }
    width = hidden(arm)
    sliced = arm.endswith((":sliced", ":ablation"))
    if arm not in FIXED_ARMS and not sliced:
        record.update(
            certified=certified,
            rate=0.3,
            steps=3200,
            edge_of_grid=False,
            step0_selection_loss=1.0,
            harm_inventory={"0.3@3200": float(loss.double().mean()), "1.0@200": None},
        )
    if width is not None:
        record.update(retained_parameters=33 * width + 16, survivors=list(range(width)))
        if sliced:
            record["unfolded_development_loss"] = float(loss.double().mean()) + 0.001
            vectors["unfolded_loss"] = loss + 0.001
    # The real modules' sizes, so the §4 parameter-step disclosure (M5) reproduces 23.1 M against 5.4 M.
    if arm == "M-wide":
        record["retained_parameters"] = 33 * 64 + 16
    if arm in ("M-small", "M-small-B"):
        record["retained_parameters"] = 33 * 8 + 16
    if arm not in FIXED_ARMS and not sliced:
        # M5: final-checkpoint booking of the arm's three runs (module grids 3 x 3,200 steps; M-small-B 3 x 6,400).
        steps = 3 * (6400 if arm == "M-small-B" else 3200)
        record["booking"] = {
            "example_passes": 250 * steps,
            "parameter_steps": int(str(record["retained_parameters"])) * steps,
            "wall_seconds": 1.0,
        }
    if arm in RETUNE:
        record.update(retained_parameters=0, complement_sha256="c" * 64, complement_unchanged=True)
    if arm == "M-aff":
        record["r_aff"] = r_aff
    return record, vectors


def write_files(output: Path, files: dict[str, str], seed: int) -> Record:
    for name in files:
        (output / name).write_bytes(f"{name}-{seed}".encode())
    return {key: file_sha(output / name) for name, key in files.items()}


class FakeCluster:
    """The Popen side effect: one M0 receipt per launched child, then a real `/bin/true` process."""

    def __init__(
        self, repo: Path, custody: str = CUSTODY, ordered: str = "04-schedule.json", **scenario: object
    ) -> None:
        self.repo = repo
        self.custody = custody
        self.ordered = ordered
        self.replay_mismatch = scenario.pop("replay_mismatch", frozenset())
        self.r_aff = scenario.pop("r_aff", {})
        self.uncertified = scenario.pop("uncertified", frozenset())
        self.module_seconds = scenario.pop("module_seconds", 10.0)
        self.unavailable = scenario.pop("unavailable", {})
        self.fold_mismatch = scenario.pop("fold_mismatch", False)
        self.failing = scenario.pop("failing", frozenset())
        self.acceptance_failed = scenario.pop("acceptance_failed", frozenset())
        self.boost = scenario.pop("boost", {})
        self.zero_identity_failed = scenario.pop("zero_identity_failed", frozenset())
        self.static_step0_mismatch = scenario.pop("static_step0_mismatch", frozenset())
        if scenario:
            raise AssertionError(f"unknown scenario keys {sorted(scenario)}")
        self.launched: list[tuple[str, str, str, dict[str, str], str]] = []
        self.violations: list[str] = []

    def __call__(self, command: list[str], **kwargs: dict[str, str]) -> subprocess.Popen[bytes]:
        if command[0] == "git":
            # The confirmatory's tracked-evidence check runs real git in the temporary repository.
            return REAL_POPEN(command, **kwargs)
        module = command[command.index("-m") + 1]
        action = command[command.index("-m") + 2]
        tokens = command[command.index("-m") + 3 :]
        flags = dict(zip(tokens[0::2], tokens[1::2], strict=True))
        output = Path(flags["--output"])
        gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        if not (self.repo / self.custody).is_file():
            self.violations.append(f"{output.name} launched before custody")
        if output.name.startswith("modules-") and not (output.parent / self.ordered).is_file():
            self.violations.append(f"{output.name} launched before {self.ordered}")
        self.launched.append((output.name, action, gpu, flags, module))
        output.mkdir()
        if output.name in self.failing:
            # A child that exits nonzero and publishes no receipt (not a ceiling stop).
            return REAL_POPEN(["/bin/false"], **kwargs)
        envelope = {
            "action": action,
            "software_valid": True,
            "identity": {"physical_gpu": gpu, **IDENTITY},
            "payload": self.payload(action, flags, output),
        }
        (output / "result.json").write_text(json.dumps(envelope))
        return REAL_POPEN(["/bin/true"], **kwargs)

    def names(self) -> list[str]:
        return [entry[0] for entry in self.launched]

    def payload(self, action: str, flags: dict[str, str], output: Path) -> Record:
        seed, lineage, scale = int(flags["--seed"]), flags["--lineage"], float(flags["--scale"])
        base: Record = {"seed": seed, "lineage": lineage, "scale": scale, "widths": list(FRONTIER_WIDTHS[scale])}
        axis = flags["--checkpoints"] if "--checkpoints" in flags else "200,800,3200"
        axis_b = flags["--checkpoints-b"] if "--checkpoints-b" in flags else "400,1600,6400"
        schedule: Record = {
            "checkpoints": [int(x) for x in axis.split(",")],
            "checkpoints_b": [int(x) for x in axis_b.split(",")],
        }
        if action in ("replay", "train", "cooled20"):
            state = ATTEMPT4_FINAL_STATE[seed] if action == "replay" else f"state-{lineage}-{seed}"
            if output.name in self.replay_mismatch:
                state = "0" * 64
            files = {"host.pt": "checkpoint_sha256", "optimizer.pt": "optimizer_sha256"}
            return {
                **base,
                **write_files(output, {**files, "loss-stream.bin": "loss_stream_sha256"}, seed),
                "epochs": 20 if action == "cooled20" else 10,
                "final_state_sha256": state,
                "initial_state_sha256": f"init-{scale}-{seed}",
                "epoch_state_sha256": [f"epoch-{seed}"],
                "forward_backward": {"output_sha256": f"fb-{seed}"},
                "parameters": PARAMETERS[scale],
                "training_seconds": 1.0,
            }
        if action == "acceptance":
            items = ("resume_bitwise", "checkpoint_bitwise", "static_step0_bitwise", "evaluation_consistent")
            passed = dict.fromkeys(items, True) | (
                {"static_step0_bitwise": False} if output.name in self.acceptance_failed else {}
            )
            return {**base, **passed, "straight_epoch_state_sha256": ["s"], "checkpoint_rate": 0.3}
        if action == "timing":
            return {
                **base,
                **schedule,
                "module_grid_seconds": self.module_seconds,
                "module_grid_steps": 9600,
                "front_rerun_seconds_per_1000_steps": 1.0,
                "development_evaluation_seconds": 0.01,
            }
        if action == "instrument":
            posthoc = {
                "seed": seed,
                "noop_loss": 1.0,
                "a1_drift": 0.0,
                "arm_losses": {"A1": 0.8, "A2": 0.82, "A3": 0.805, "B3": 0.81},
                "oracle_gain_fraction_of_A3": 0.95,
            }
            rungs = {"0.1": 0.2, "0.3": 0.3, "0.03": 0.06}
            return {**base, "materiality": {"seed": seed, "rungs": {"P1": rungs}}, "posthoc": posthoc}
        if action == "noop":
            failed = seed in self.zero_identity_failed
            identity = {kind: not (failed and kind == "hidden-8") for kind in ZERO_MODULE_KINDS}
            return {**base, "noop_choice": "raw", "noop_sha256": f"noop-{seed}", "zero_module_identity": identity}
        if action == "tcool":
            source = read_record(Path(flags["--training-root"]) / "result.json")["payload"]
            files = write_files(output, {"host.pt": "checkpoint_sha256", "loss-stream.bin": "loss_stream_sha256"}, seed)
            return {
                **base,
                **files,
                "source_final_state_sha256": source["final_state_sha256"],
                "final_state_sha256": f"tcool-{seed}",
                "noop_sha256": f"tcool-noop-{seed}",
            }
        if action == "static":
            files = {"host.pt": "checkpoint_sha256", "static-module.pt": "module_sha256"}
            written = write_files(output, {**files, "loss-stream.bin": "loss_stream_sha256"}, seed)
            initial = "0" * 64 if seed in self.static_step0_mismatch else f"init-{scale}-{seed}"
            return {**base, **written, "static_sha256": f"static-{seed}", "initial_state_sha256": initial}
        if action in ("modules", "retune"):
            n0 = noop_loss(seed, scale)
            records: Record = {}
            vectors: dict[str, dict[str, Tensor]] = {"N0": {"loss": n0, "correct": (torch.arange(N) % 2).float()}}
            r_aff = self.r_aff[seed] if seed in self.r_aff else 0.001
            failed = self.unavailable[seed] if seed in self.unavailable else frozenset()
            for arm in flags["--arms"].split(","):
                if arm in failed:
                    candidates = [{"rate": 1.0, "steps": 200, "selection_loss": None, "finite": False}]
                    records[arm] = {
                        "status": "unavailable",
                        "cause": arm,
                        "error": "nonfinite",
                        "candidates": candidates,
                    }
                    continue
                certified = seed not in self.uncertified
                boost = self.boost[arm] if arm in self.boost else 0.0
                records[arm], vectors[arm] = arm_record(arm, seed, n0, r_aff=r_aff, certified=certified, boost=boost)
            torch.save(vectors, output / "vectors.pt")
            return {
                **base,
                **schedule,
                "noop_sha256": f"noop-{seed}",
                "arms_run": flags["--arms"].split(","),
                "arms": records,
                "unavailable_arms": [],
                "n0": {"development_loss": float(n0.double().mean())},
                "r_aff": r_aff if "M-aff" in vectors else None,
                "vectors_sha256": file_sha(output / "vectors.pt"),
            }
        if action == "fold-check":
            receipt = read_record(Path(flags["--modules-root"]) / "result.json")["payload"]
            return {
                **base,
                "receipt_r_aff": receipt["r_aff"],
                "recomputed_r_aff": receipt["r_aff"],
                "fold_residual_matches_receipt": not self.fold_mismatch,
            }
        raise AssertionError(f"unexpected action {action}")


class PilotRunTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.certificate = self.base / "certificate.json"
        self.certificate.write_text("{}")

    def launch(
        self, root_name: str = "run", *, bumps: dict[str, float] | None = None, item_cost: float | None = None,
        **scenario: object,
    ) -> tuple[int, Path, FakeCluster, list[str]]:  # fmt: skip
        """`bumps` raises the measured charge to a value just before the named job's batch runs (fake children charge
        milliseconds); `item_cost` replaces the run-time cost of the extended host items (stage 4 is unchanged)."""
        cluster = FakeCluster(self.repo, **scenario)
        published: list[str] = []
        real_publish = m0_pilot.publish

        def record(path: Path, value: Json) -> None:
            published.append(path.name)
            real_publish(path, value)

        raised = {} if bumps is None else bumps

        class Bumping(Supervisor):
            def run(self, jobs: tuple[Job, ...]) -> None:
                for item in jobs:
                    if item.name in raised:
                        self.charged = max(self.charged, raised[item.name])
                super().run(jobs)

        def runtime_cost(item: str, w: Workloads) -> float:
            hosts = item in ("cooled-20", "F(0.25)", "F(0.75)")
            return item_cost if hosts and item_cost is not None else extended_cost(item, w)

        root = self.base / root_name
        with (
            patch("sultai_screen.pilot.subprocess.Popen", side_effect=cluster),
            patch("sultai_screen.m0.pilot.REPO", self.repo),
            patch("sultai_screen.m0.pilot.check_certificate", return_value=IDENTITY),
            patch("sultai_screen.m0.pilot.environment_identity", return_value={"torch": "fake"}),
            patch("sultai_screen.m0.pilot.check_cards", return_value={"gpu_query": "fake"}),
            patch("sultai_screen.m0.pilot.publish", side_effect=record),
            patch("sultai_screen.m0.pilot.Supervisor", Bumping),
            patch("sultai_screen.m0.pilot.extended_cost", side_effect=runtime_cost),
        ):
            code = pilot(root, self.base / "data", self.certificate)
        return code, root, cluster, published

    def assert_stop(self, root: Path, stage: str, verdict: str) -> Record:
        result = read_record(root / "pilot-result.json")
        self.assertEqual(result["stage"], stage)
        self.assertEqual(result["verdict"], verdict)
        self.assertEqual(result["protocol_version"], "m0-v4")
        self.assertEqual(result["amendment_2_sha256"], protocol_identity()["amendment_2_sha256"])
        self.assertFalse(result["pilot_clean"])
        return result

    def test_clean_pilot_publishes_every_receipt_in_stage_order_under_the_m0_ceiling(self) -> None:
        code, root, cluster, published = self.launch()
        self.assertEqual(code, 0, read_record(root / "pilot-result.json"))
        self.assertEqual(cluster.violations, [])
        self.assertEqual([name for name in published if name in RECEIPTS], list(RECEIPTS))
        result = read_record(root / "pilot-result.json")
        self.assertTrue(result["pilot_clean"])
        self.assertEqual(result["protocol_version"], "m0-v4")
        # Every child is the M0 worker, carries its lineage and scale, and one job list never doubles a card.
        for name, _, _, flags, module in cluster.launched:
            self.assertEqual(module, "sultai_screen.m0.worker", name)
            self.assertIn("--lineage", flags)
            self.assertIn("--scale", flags)
        ledgers = sorted(root.glob("ledger-*.json"))
        previous = 0
        for path in ledgers:
            ledger = read_record(path)
            self.assertEqual(ledger["ceiling_gpu_process_seconds"], M0_PILOT_SECONDS)
            self.assertEqual(M0_PILOT_SECONDS, 14_400)
            batch = ledger["entries"][previous:]
            previous = len(ledger["entries"])
            cards = [entry["gpu"] for entry in batch]
            self.assertEqual(len(cards), len(set(cards)), path.name)
        # Stage 1: all six anchors replay; seed 3000 on both cards.
        names = cluster.names()
        self.assertIn("replay-3000-gpu0", names)
        self.assertIn("replay-3000-gpu1", names)
        launched = {name: gpu for name, _, gpu, _, _ in cluster.launched}
        self.assertEqual((launched["replay-3000-gpu0"], launched["replay-3000-gpu1"]), ("0", "1"))
        # Stage 2 (R16): fold-check right after anchor 3000's M-aff child and before any stage-3 host.
        self.assertLess(names.index("maff-3000"), names.index("fold-check-3000"))
        self.assertLess(names.index("fold-check-3000"), names.index("maff-3002"))
        # Stage 4: the default axis; stage 5 reuses the stage-2 M-aff and reads the axis from 04-schedule.json.
        schedule = read_record(root / "04-schedule.json")
        self.assertFalse(schedule["raised"])
        self.assertEqual(schedule["checkpoints"], list(CHECKPOINTS))
        self.assertEqual(schedule["checkpoints_b"], list(CHECKPOINTS_B))
        admitted = read_record(root / "04-admission.json")
        self.assertEqual(admitted["already_run"]["M-aff"]["names"], [f"maff-{seed}" for seed in range(3000, 3006)])
        workload = admitted["workloads"]
        self.assertAlmostEqual(admitted["already_run"]["M-aff"]["seconds"], 6 * workload["grid"]["aff"])
        self.assertAlmostEqual(admitted["already_run"]["N0"]["seconds"], 6 * workload["fold_and_calibrations"])
        self.assertAlmostEqual(
            admitted["projection"]["remaining_lines"]["core"], admitted["already_run"]["remaining_core_seconds"]
        )
        self.assertEqual(admitted["admitted_extended"], list(EXTENDED_ORDER))
        for name, action, _, flags, _ in cluster.launched:
            if name.startswith("modules-"):
                self.assertEqual(action, "modules")
                self.assertNotIn("M-aff", flags["--arms"].split(","))
                self.assertEqual(flags["--checkpoints"], "200,800,3200")
                self.assertEqual(flags["--checkpoints-b"], "400,1600,6400")
        # Stage 5: the spreads before any median; S-static admission beside SD(D5).
        spreads = read_record(root / "05-spreads.json")
        self.assertEqual(set(spreads["sd"]), {"D1", "D2", "D3", "D5"})
        self.assertIsInstance(spreads["static_admitted"], bool)
        self.assertEqual(spreads["static_admitted"], spreads["sd"]["D5"] <= 0.0275)
        self.assertTrue(spreads["v0"]["passed"])
        count = read_record(root / "06-confirmatory-count.json")
        self.assertGreaterEqual(count["confirmatory_count"], 32)
        self.assertEqual(set(count["achieved_half_width"]), {"D2", "D3", "D5"})
        self.assertTrue(all(math.isfinite(value) for value in count["achieved_half_width"].values()))
        self.assertTrue(count["achieved_half_width_development_only"])
        medians = read_record(root / "07-development-medians.json")
        self.assertTrue(medians["development_only"])
        self.assertIsNone(medians["confirmatory_verdict"])
        self.assertIn("D1", medians["medians"])
        self.assertEqual(set(medians["headline"]), {"frontier_fraction", "frontier_slope"})
        self.assertIn("bootstrap", medians["headline"]["frontier_fraction"])
        self.assertIn("plug-in", medians["headline"]["frontier_slope"])
        # R26: two timing children, one per card. R24: every anchor-bound child runs on its anchor's card.
        timing = sorted((name, gpu) for name, action, gpu, _, _ in cluster.launched if action == "timing")
        self.assertEqual(timing, [("timing-0", "0"), ("timing-1", "1")])
        for name, _, gpu, flags, _ in cluster.launched:
            if flags["--lineage"] == "anchor" and not name.startswith(("replay-3000-gpu", "acceptance-", "timing-")):
                self.assertEqual(gpu, str(ANCHORS.index(int(flags["--seed"])) % 2), name)
        # M3, R25, M8: the recorded stage 1-3 V0 flags, never literals.
        instrument = read_record(root / "02-instrument.json")
        self.assertEqual(set(instrument["zero_module_identity_flags"]), {str(seed) for seed in ANCHORS})
        self.assertTrue(all(all(flags.values()) for flags in instrument["zero_module_identity_flags"].values()))
        self.assertIs(instrument["b3_fraction_every_anchor"], True)
        hosts = read_record(root / "03-hosts.json")
        self.assertTrue(all(entry["matches"] for entry in hosts["static_step0"].values()))
        self.assertEqual(hosts["static_step0"]["3000"]["anchor"], "init-1.0-3000")
        self.assertTrue(all(spreads["v0_recorded"].values()))
        # M5: §4's parameter-step disclosure of the two D2 routes, 23.1 M against 5.4 M per anchor.
        ratio = spreads["d2_parameter_step_ratio"]
        self.assertAlmostEqual(ratio["median"], (2128 + 280) * 9600 / (280 * 19200))
        self.assertEqual(spreads["anchors"][0]["harm_baseline"]["M-sq|T-cool"], "baseline: anchor N0")
        # R23: the reserve at stage 4; every extended item re-projected and run.
        self.assertEqual(admitted["extended_reserve_fraction"], 0.05)
        self.assertEqual(admitted["extended_limit_gpu_process_seconds"], extended_limit(M0_PILOT_SECONDS))
        extended = read_record(root / "05-extended.json")
        self.assertEqual(extended["completed"], ["starved-pair", "cooled-20", "F(0.25)", "F(0.75)"])
        self.assertEqual((extended["not_admitted_at_runtime"], extended["extended_truncated"]), ([], False))
        # I3 and M7: the measured projection beside the kernel one, the larger deciding; the fixed term is what the
        # confirmatory re-runs (replays, acceptance, fold-check), read from the ledger.
        entries = {e["job"]: e["charged_gpu_process_seconds"] for e in read_record(ledgers[-1])["entries"]}
        # R23: an extended host item costs its three whole stage-3 children (train, noop, evaluate maxima).
        half = range(6000, 6006)
        three = sum(max(entries[f"{kind}-{seed}"] for seed in half) for kind in ("frontier-0.5", "noop", "evaluate"))
        self.assertAlmostEqual(workload["host"]["0.5"] + workload["noop_evaluate"]["0.5"], three)
        measured = count["measured_charges"]
        replays = ["replay-3000-gpu0", "replay-3000-gpu1", *(f"replay-{seed}" for seed in ANCHORS[1:])]
        self.assertEqual(measured["fixed_v0_children"], [*replays, "acceptance-0", "acceptance-1", "fold-check-3000"])
        fixed = sum(entries[name] for name in measured["fixed_v0_children"])
        self.assertAlmostEqual(measured["fixed_v0_gpu_process_seconds"], fixed)
        self.assertAlmostEqual(count["projection"]["fixed_v0_gpu_process_seconds"], fixed)
        # R28: no replay in the per-anchor term (replays are the fixed term); the anchor's own host is the kernel
        # host:1.0; the modules and retune children are scaled to the §10 arm set from their receipts' arms_run.
        self.assertNotIn("train", measured["per_anchor_children"])
        whole = workload["host"]["1.0"] + sum(entries[f"{kind}-3000"] for kind in ("noop", "maff", "tcool"))
        grids = sum(entries[f"{kind}-3000"] for kind in ("modules", "retune"))
        scale = measured["scale"]["3000"]
        self.assertLess(scale, 1.0)
        self.assertAlmostEqual(measured["per_anchor_raw"]["3000"], whole + grids)
        self.assertAlmostEqual(measured["per_anchor"]["3000"], whole + scale * grids)
        larger = max(
            count["projection"]["projected_gpu_process_seconds"],
            count["measured_projection"]["projected_gpu_process_seconds"],
        )
        self.assertEqual(count["projected_gpu_process_seconds"], larger)
        self.assertEqual(count["within_ceiling"], larger <= M0_CONFIRMATORY_SECONDS)
        # R27, I3: the pilot result binds its source and carries both projections to the confirmatory.
        self.assertEqual(result["implementation_source_sha256"], IDENTITY["source_sha256"])
        self.assertEqual(result["implementation_commit"], IDENTITY["commit"])
        self.assertIs(result["extended_truncated"], False)
        carried = result["confirmatory_projection"]
        self.assertEqual(carried["projected_gpu_process_seconds"], larger)
        self.assertEqual(carried["fixed_v0_gpu_process_seconds"], measured["fixed_v0_gpu_process_seconds"])
        self.assertEqual(carried["per_anchor_gpu_process_seconds"], max(measured["per_anchor"].values()))

    def test_raised_axis_doubles_both_axes_in_one_receipt_and_reruns_maff(self) -> None:
        code, root, cluster, _ = self.launch(uncertified=frozenset((3000, 3001)))
        self.assertEqual(code, 0, read_record(root / "pilot-result.json"))
        self.assertEqual(cluster.violations, [])
        schedule = read_record(root / "04-schedule.json")
        self.assertTrue(schedule["raised"])
        self.assertAlmostEqual(schedule["maff_certified_fraction"], 4 / 6)
        self.assertEqual(schedule["checkpoints"], [400, 1600, 6400])
        self.assertEqual(schedule["checkpoints_b"], [800, 3200, 12800])
        names = cluster.names()
        self.assertLess(names.index("modules-3000"), names.index("fold-check-raised-3000"))
        for name, _, _, flags, _ in cluster.launched:
            if name.startswith("modules-"):
                self.assertIn("M-aff", flags["--arms"].split(","))
                self.assertEqual(flags["--checkpoints"], "400,1600,6400")
                self.assertEqual(flags["--checkpoints-b"], "800,3200,12800")
        admitted = read_record(root / "04-admission.json")
        self.assertEqual(admitted["already_run"]["M-aff"], {"names": [], "seconds": 0.0})
        k = 10.0 / 9600
        self.assertAlmostEqual(admitted["workloads"]["grid"]["module"], k * 9600 * 2)

    def test_a_false_zero_module_flag_holds_at_stage_two_before_any_maff_grid(self) -> None:
        code, root, cluster, _ = self.launch(zero_identity_failed=frozenset((3002,)))
        self.assertEqual(code, 2)
        result = self.assert_stop(root, "instrument", "instrument-failed")
        self.assertIn("zero-module identity", result["error"])
        self.assertIn("3002:hidden-8", result["error"])
        self.assertEqual(result["implementation_source_sha256"], IDENTITY["source_sha256"])
        self.assertFalse(any(name.startswith("maff-") for name in cluster.names()))

    def test_a_static_step_zero_state_other_than_the_anchors_holds_at_stage_three(self) -> None:
        code, root, cluster, _ = self.launch(static_step0_mismatch=frozenset((3004,)))
        self.assertEqual(code, 2)
        result = self.assert_stop(root, "hosts", "instrument-failed")
        self.assertIn("step 0", result["error"])
        self.assertIn("3004", result["error"])
        self.assertFalse((root / "03-hosts.json").exists())
        self.assertFalse(any(name.startswith("modules-") for name in cluster.names()))

    def test_runtime_reprojection_stops_at_the_first_extended_item_that_no_longer_fits(self) -> None:
        # Stage 4 admits every item; the starved pair runs; the measured charge has then grown so that cooled-20,
        # costed 1,000 s at run time, no longer fits under the reserve. The pilot stays clean (R23).
        limit = extended_limit(M0_PILOT_SECONDS)
        code, root, cluster, published = self.launch(bumps={"starved-6000": limit - 500.0}, item_cost=1000.0)
        self.assertEqual(code, 0, read_record(root / "pilot-result.json"))
        self.assertEqual([name for name in published if name in RECEIPTS], list(RECEIPTS))
        self.assertEqual(read_record(root / "04-admission.json")["admitted_extended"], list(EXTENDED_ORDER))
        extended = read_record(root / "05-extended.json")
        self.assertEqual(extended["completed"], ["starved-pair"])
        refused = extended["not_admitted_at_runtime"]
        self.assertEqual([entry["item"] for entry in refused], ["cooled-20", "F(0.25)", "F(0.75)"])
        projection = refused[0]["projection"]
        self.assertEqual(projection["item_gpu_process_seconds"], 1000.0)
        self.assertGreaterEqual(projection["charged_gpu_process_seconds"], limit - 500.0)
        self.assertGreater(projection["projected_gpu_process_seconds"], projection["limit_gpu_process_seconds"])
        self.assertEqual(projection["limit_gpu_process_seconds"], limit)
        self.assertNotIn("projection", refused[1])
        self.assertIs(extended["extended_truncated"], False)
        hosts = ("cooled-20-", "frontier-0.25-", "frontier-0.75-")
        self.assertFalse(any(name.startswith(hosts) for name in cluster.names()))
        self.assertIs(read_record(root / "pilot-result.json")["pilot_clean"], True)

    def test_a_ceiling_stop_inside_an_extended_item_truncates_the_set_and_the_pilot_stays_clean(self) -> None:
        # The F(0.75) item is admitted at run time, then the Supervisor's own cap check stops its first batch.
        code, root, cluster, published = self.launch(bumps={"frontier-0.75-6006": M0_PILOT_SECONDS - 5.0})
        self.assertEqual(code, 0, read_record(root / "pilot-result.json"))
        self.assertEqual([name for name in published if name in RECEIPTS], list(RECEIPTS))
        extended = read_record(root / "05-extended.json")
        self.assertIs(extended["extended_truncated"], True)
        self.assertEqual(extended["completed"], ["starved-pair", "cooled-20", "F(0.25)"])
        truncated = extended["truncated"]
        self.assertEqual((truncated["item"], truncated["completed"]), ("F(0.75)", False))
        self.assertEqual(truncated["error"], "aggregate pilot GPU-hour cap exhausted")
        self.assertNotIn("frontier-0.75-6006", cluster.names())
        result = read_record(root / "pilot-result.json")
        self.assertIs(result["pilot_clean"], True)
        self.assertIs(result["extended_truncated"], True)
        self.assertTrue((root / "06-confirmatory-count.json").is_file())
        self.assertTrue((root / "07-development-medians.json").is_file())

    def test_a_child_failure_inside_an_extended_item_still_stops_the_pilot(self) -> None:
        # R23 catches only the Supervisor's three ceiling stops: a child exiting nonzero is execution-invalid.
        code, root, cluster, _ = self.launch(failing=frozenset(("cooled-20-6018",)))
        self.assertEqual(code, 2)
        result = self.assert_stop(root, "extended", "execution-invalid")
        self.assertIn("cooled-20-6018", result["error"])
        self.assertIn("cooled-20-6018", cluster.names())
        for name in ("05-extended.json", "06-confirmatory-count.json", "07-development-medians.json"):
            self.assertFalse((root / name).exists(), name)

    def test_only_the_supervisors_ceiling_stops_are_caught_in_the_extended_stage(self) -> None:
        for message in m0_pilot.CEILING_STOPS:
            self.assertTrue(m0_pilot.ceiling_stop(PilotHold(message)), message)
        self.assertFalse(m0_pilot.ceiling_stop(PilotHold("child cooled-20-6018 exited 1")))
        self.assertFalse(m0_pilot.ceiling_stop(InstrumentHold("aggregate GPU-hour timeout")))
        self.assertFalse(m0_pilot.ceiling_stop(FeasibilityHold("aggregate GPU-hour timeout")))

    def test_replay_mismatch_holds_at_stage_one(self) -> None:
        code, root, cluster, _ = self.launch(replay_mismatch=frozenset(("replay-3000-gpu1",)))
        self.assertEqual(code, 2)
        result = self.assert_stop(root, "replay-timing", "execution-invalid")
        self.assertEqual(result["error_type"], "PilotHold")
        self.assertFalse((root / "02-instrument.json").exists())
        self.assertFalse(any(name.startswith(("instrument-", "noop-")) for name in cluster.names()))

    def test_fold_residual_above_tolerance_holds_at_stage_two_naming_the_fold_clause(self) -> None:
        code, root, cluster, _ = self.launch(r_aff={3003: 0.006})
        self.assertEqual(code, 2)
        result = self.assert_stop(root, "instrument", "instrument-failed")
        self.assertIn("fold residual", result["error"])
        self.assertIn("3003", result["error"])
        self.assertFalse((root / "03-hosts.json").exists())
        self.assertFalse(any(name.startswith(("tcool-", "frontier-")) for name in cluster.names()))

    def test_fold_check_mismatch_holds_at_stage_two_as_acceptance_item_five(self) -> None:
        code, root, cluster, _ = self.launch(fold_mismatch=True)
        self.assertEqual(code, 2)
        result = self.assert_stop(root, "instrument", "instrument-failed")
        self.assertIn("(v)", result["error"])
        self.assertNotIn("maff-3002", cluster.names())

    def test_an_unavailable_optional_core_arm_is_recorded_and_the_pilot_continues(self) -> None:
        code, root, _, published = self.launch(unavailable={3002: frozenset(("M-exp",))})
        self.assertEqual(code, 0, read_record(root / "pilot-result.json"))
        self.assertEqual([name for name in published if name in RECEIPTS], list(RECEIPTS))
        anchors = {entry["seed"]: entry for entry in read_record(root / "05-spreads.json")["anchors"]}
        self.assertEqual(anchors[3002]["unavailable_arms"], ["M-exp"])
        self.assertEqual(anchors[3001]["unavailable_arms"], [])
        self.assertIsNone(read_record(root / "07-development-medians.json")["medians"]["G_mexp"])

    def test_an_unavailable_starved_pair_arm_is_reported_and_never_stops_the_pilot(self) -> None:
        code, root, _, published = self.launch(unavailable={6002: frozenset(("M-aff",))})
        self.assertEqual(code, 0, read_record(root / "pilot-result.json"))
        self.assertEqual([name for name in published if name in RECEIPTS], list(RECEIPTS))
        hosts = {entry["seed"]: entry for entry in read_record(root / "05-extended.json")["starved-pair"]}
        self.assertEqual(hosts[6002]["status"], "unavailable")
        self.assertEqual(hosts[6002]["unavailable_arms"], ["M-aff"])
        self.assertEqual(hosts[6002]["causes"]["M-aff"]["cause"], "M-aff")
        self.assertEqual(hosts[6003]["status"], "available")
        medians = read_record(root / "07-development-medians.json")
        self.assertEqual(medians["starved_pair_hosts"], {"available": 5, "run": 6})
        self.assertIsNotNone(medians["starved_pair_medians"]["D1"])

    def test_core_over_ceiling_stops_at_stage_four_before_any_core_child(self) -> None:
        code, root, cluster, _ = self.launch(module_seconds=400.0)
        self.assertEqual(code, 2)
        self.assert_stop(root, "admission", "hold-feasibility")
        admitted = read_record(root / "04-admission.json")
        self.assertTrue(admitted["core_over_ceiling"])
        self.assertEqual(admitted["admitted_extended"], [])
        self.assertFalse(any(name.startswith(("modules-", "retune-")) for name in cluster.names()))
        self.assertFalse((root / "05-spreads.json").exists())

    def test_confirmatory_hold_keeps_the_spreads_and_publishes_no_medians(self) -> None:
        code, root, _, published = self.launch(module_seconds=160.0)
        self.assertEqual(code, 2)
        self.assert_stop(root, "confirmatory-count", "hold-feasibility")
        self.assertIn("static_admitted", read_record(root / "05-spreads.json"))
        count = read_record(root / "06-confirmatory-count.json")
        self.assertFalse(count["projection"]["within_ceiling"])
        self.assertFalse((root / "07-development-medians.json").exists())
        self.assertLess(published.index("05-spreads.json"), published.index("06-confirmatory-count.json"))

    def test_custody_is_claimed_before_any_child_and_a_new_root_cannot_reset_it(self) -> None:
        code, _, cluster, _ = self.launch(replay_mismatch=frozenset(("replay-3000-gpu0",)))
        self.assertEqual(code, 2)
        self.assertTrue(cluster.launched)
        self.assertEqual(cluster.violations, [])
        custody = read_record(self.repo / CUSTODY)
        self.assertEqual(custody["identity"]["protocol_version"], "m0-v4")
        with self.assertRaisesRegex(PilotHold, "custody"):
            self.launch("another-root")
        self.assertFalse((self.base / "another-root").exists())

    def test_an_existing_m0_custody_receipt_holds_before_any_child(self) -> None:
        (self.repo / ".weft").mkdir()
        (self.repo / CUSTODY).write_text("{}")
        cluster = FakeCluster(self.repo)
        with (
            patch("sultai_screen.pilot.subprocess.Popen", side_effect=cluster),
            patch("sultai_screen.m0.pilot.REPO", self.repo),
            patch("sultai_screen.m0.pilot.check_certificate", return_value=IDENTITY),
            patch("sultai_screen.m0.pilot.environment_identity", return_value={}),
            patch("sultai_screen.m0.pilot.check_cards", return_value={}),
            self.assertRaisesRegex(PilotHold, "custody"),
        ):
            pilot(self.base / "run", self.base / "data", self.certificate)
        self.assertEqual(cluster.launched, [])
        self.assertFalse((self.base / "run").exists())

    def test_a_gate3_custody_receipt_is_irrelevant_and_untouched(self) -> None:
        (self.repo / ".weft").mkdir()
        gate3 = self.repo / ".weft/gate3-pilot-attempt.json"
        gate3.write_text('{"gate3": true}\n')
        before = gate3.read_bytes()
        code, root, cluster, _ = self.launch(replay_mismatch=frozenset(("replay-3000-gpu1",)))
        self.assertEqual(code, 2)
        self.assertTrue(cluster.launched)
        self.assertEqual(gate3.read_bytes(), before)
        self.assertTrue((self.repo / CUSTODY).is_file())
        self.assertNotEqual(CUSTODY, ".weft/gate3-pilot-attempt.json")
        self.assertFalse(CUSTODY.startswith(".weft/gate3-pilot"))
        self.assertFalse((root / "01-replay-timing.json").exists())


def workloads() -> Workloads:
    return Workloads(
        host={0.25: 20.0, 0.5: 35.0, 0.75: 40.0, 1.0: 45.0, 1.5: 85.0},
        tcool=25.0,
        static=50.0,
        grid={"module": 60.0, "module-B": 120.0, "aff": 100.0, "retune": 80.0, "refit": 60.0, "refit-single": 10.0},
        front_rerun=15.625,
        fold_and_calibrations=30.0,
        evaluation=0.5,
        fixed_v0=1500.0,
        noop_evaluate={0.5: 40.0, 1.0: 45.0, 1.5: 50.0},
    )


class AdmissionTests(unittest.TestCase):
    def test_projection_is_charged_plus_remaining_work_after_two_stages_have_run(self) -> None:
        w = workloads()
        charged = stage_seconds("V0", w) + stage_seconds("hosts", w)
        done = already_run(w, raised=False)
        self.assertEqual(done["N0"]["seconds"], 6 * w.fold_and_calibrations)
        self.assertEqual(done["M-aff"]["seconds"], 6 * w.grid["aff"])
        record = admission(charged, w, M0_PILOT_SECONDS, done["seconds"])
        self.assertEqual(record["extended_limit_gpu_process_seconds"], 0.95 * M0_PILOT_SECONDS)
        projection = record["projection"]
        lines = projection["remaining_lines"]
        admitted = record["admitted_extended"]
        self.assertFalse(record["core_over_ceiling"])
        self.assertEqual(set(lines), {"core", *admitted})
        self.assertNotIn("V0", lines)
        self.assertNotIn("hosts", lines)
        remaining_core = stage_seconds("core", w) - 6 * w.fold_and_calibrations - 6 * w.grid["aff"]
        self.assertAlmostEqual(lines["core"], remaining_core)
        self.assertAlmostEqual(done["remaining_core_seconds"], remaining_core)
        expected = charged + remaining_core + sum(extended_cost(item, w) for item in admitted)
        self.assertAlmostEqual(projection["projected_gpu_process_seconds"], expected)
        raised = already_run(w, raised=True)
        self.assertEqual(raised["M-aff"], {"names": [], "seconds": 0.0})
        self.assertAlmostEqual(raised["remaining_core_seconds"] - remaining_core, 6 * w.grid["aff"])
        self.assertLessEqual(expected, extended_limit(M0_PILOT_SECONDS))
        self.assertEqual(admitted, list(EXTENDED_ORDER[: len(admitted)]))
        over = admission(M0_PILOT_SECONDS - 1.0, w, M0_PILOT_SECONDS, done["seconds"])
        self.assertTrue(over["core_over_ceiling"])
        self.assertEqual(over["admitted_extended"], [])

    def test_workloads_take_one_per_card_maxima_and_scale_the_grids_with_a_raised_axis(self) -> None:
        timing = (
            {"module_grid_seconds": 48.0, "module_grid_steps": 9600, "front_rerun_seconds_per_1000_steps": 15.0,
             "development_evaluation_seconds": 0.4},
            {"module_grid_seconds": 60.0, "module_grid_steps": 9600, "front_rerun_seconds_per_1000_steps": 12.0,
             "development_evaluation_seconds": 0.5},
        )  # fmt: skip
        charges = {
            "host:0.5": (30.0, 35.0),
            "host:1.0": (44.0, 45.0),
            "host:1.5": (85.0,),
            "tcool": (25.0,),
            "static": (50.0,),
            "aff": (90.0, 100.0),
            "fold_and_calibrations": (30.0,),
            "noop:0.5": (5.0, 6.0),
            "noop:1.5": (7.0,),
            "evaluate:0.5": (3.0,),
            "evaluate:1.5": (4.0, 4.5),
        }
        w, sources = derive_workloads(timing, charges, fixed_v0=1200.0, ratio=1.0)
        # R23: a host item's noop and evaluate children; width 1.0 pairs the anchors' noop with the larger evaluate.
        self.assertEqual(w.noop_evaluate, {0.5: 9.0, 1.0: 34.5, 1.5: 11.5})
        self.assertEqual(w.host, {0.5: 35.0, 1.0: 45.0, 1.5: 85.0})
        self.assertAlmostEqual(w.grid["module"], 60.0)
        self.assertAlmostEqual(w.grid["module-B"], 120.0)
        self.assertAlmostEqual(w.grid["refit"], 60.0)
        self.assertAlmostEqual(w.grid["refit-single"], 60.0 * 800 / 9600)
        self.assertAlmostEqual(w.grid["retune"], 60.0 * 16000 / 9600)
        self.assertEqual(w.grid["aff"], 100.0)
        self.assertEqual((w.front_rerun, w.evaluation, w.fixed_v0), (15.0, 0.5, 1200.0))
        self.assertEqual(sources["per_step_module_seconds"], 60.0 / 9600)
        raised, _ = derive_workloads(timing, charges, fixed_v0=1200.0, ratio=2.0)
        self.assertAlmostEqual(raised.grid["module"], 120.0)
        self.assertAlmostEqual(raised.grid["aff"], 200.0)
        self.assertAlmostEqual(raised.front_rerun, 30.0)
        self.assertEqual(raised.host, w.host)


def source(name: str, seed: int, arms: dict[str, Record], vectors: dict[str, dict[str, Tensor]]) -> Source:
    n0 = noop_loss(seed, 1.0)
    full = {"N0": {"loss": n0, "correct": (torch.arange(N) % 2).float()}, **vectors}
    return Source(name, {"seed": seed, "noop_sha256": f"noop-{seed}", "arms": arms}, full)


def anchor_sources(seed: int, arms: frozenset[str]) -> tuple[dict[str, Record], dict[str, dict[str, Tensor]]]:
    records: dict[str, Record] = {}
    vectors: dict[str, dict[str, Tensor]] = {}
    for arm in sorted(arms):
        records[arm], vectors[arm] = arm_record(arm, seed, noop_loss(seed, 1.0), r_aff=0.001, certified=True)
    return records, vectors


class MeasuredProjectionTests(unittest.TestCase):
    def test_the_measured_term_is_scaled_to_the_confirmatory_arm_set_at_section_8_magnitudes(self) -> None:
        # R28: at the §8 bookings (the workloads() fixture: 1,010 s per anchor for the §10 set), the pilot's modules
        # and retune children run every pilot arm plus S-static and the admitted extended arms (M-exp, lottery,
        # R-all): about 1.33 x the §10 set. Unscaled, n = 32 would project over 43,200 s; scaled, it fits.
        w = workloads()
        wanted = PILOT_ARMS | STATIC_ARMS | {"M-exp", "lottery", "R-all"}
        modules = tuple(name for name in MODULE_ARMS if name in wanted - {"M-aff"})
        retune = tuple(name for name in RETUNE_ARMS if name in wanted)
        self.assertEqual(admit_extended(0.0, 0.0, w, 1e9), list(EXTENDED_ORDER))
        grids = per_anchor(w, modules) + per_anchor(w, retune) - 2 * per_anchor(w, ())
        whole = w.host[1.0] + w.fold_and_calibrations + w.grid["aff"] + w.tcool
        terms = m0_pilot.scaled_anchor(w, modules + retune, whole, grids, w.static)
        self.assertAlmostEqual(terms["scale"], per_anchor(w, CONFIRMATORY_ARMS) / per_anchor(w, modules + retune))
        self.assertGreater(1 / terms["scale"], 1.3)
        self.assertAlmostEqual(terms["raw"], whole + grids)
        self.assertAlmostEqual(terms["scaled"], whole + terms["scale"] * grids)
        fixed = 7 * w.host[1.0] + 2 * 60.0 + 20.0  # seven replays, two acceptance children, the fold-check
        scaled = measured_confirmatory(32, terms["scaled"], fixed, M0_CONFIRMATORY_SECONDS)
        raw = measured_confirmatory(32, terms["raw"], fixed, M0_CONFIRMATORY_SECONDS)
        self.assertIs(scaled["within_ceiling"], True)
        self.assertIs(raw["within_ceiling"], False)
        static = STATIC_ARMS | CONFIRMATORY_ARMS
        self.assertAlmostEqual(
            terms["static_scaled"], whole + w.static + grids * per_anchor(w, static) / per_anchor(w, modules + retune)
        )


class ReceiptTests(unittest.TestCase):
    def test_a_receipt_of_another_implementation_source_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "noop-3000").mkdir()
            envelope = {
                "action": "noop",
                "software_valid": True,
                "identity": {"physical_gpu": "0", **IDENTITY},
                "payload": {"seed": 3000, "lineage": "anchor"},
            }
            (root / "noop-3000/result.json").write_text(json.dumps(envelope))
            certified = str(IDENTITY["source_sha256"])
            self.assertEqual(receipt(root, "noop-3000", "noop", 3000, "anchor", certified)["seed"], 3000)
            with self.assertRaisesRegex(PilotHold, "another implementation source"):
                receipt(root, "noop-3000", "noop", 3000, "anchor", "2" * 64)

    def test_recorded_v0_reads_the_per_anchor_identity_flags_and_step_zero_comparisons(self) -> None:
        replay: Record = {"both_cards_bitwise": True, "acceptance": [{"static_step0_bitwise": True}] * 2}
        flags = {str(seed): dict.fromkeys(ZERO_MODULE_KINDS, True) for seed in ANCHORS}
        instrument: Record = {
            "a1_within_1e-4": True,
            "b3_fraction_every_anchor": True,
            "zero_module_identity": True,
            "zero_module_identity_flags": flags,
            "slicing": {"known_answer": True},
        }
        step0 = {str(seed): {"static": "s", "anchor": "s", "matches": True} for seed in ANCHORS}
        hosts: Record = {"static_step0_matches": True, "static_step0": step0}
        self.assertTrue(all(m0_pilot.recorded_v0(replay, instrument, hosts).values()))
        flags["3003"]["hidden-64"] = False
        self.assertIs(m0_pilot.recorded_v0(replay, instrument, hosts)["zero_module_identity"], False)
        flags["3003"]["hidden-64"] = True
        step0["3001"]["matches"] = False
        self.assertIs(m0_pilot.recorded_v0(replay, instrument, hosts)["static_step0"], False)

    def test_a_recorded_false_v0_flag_fails_v0_at_aggregation(self) -> None:
        anchors = []
        for seed in ANCHORS:
            records, vectors = anchor_sources(seed, PILOT_ARMS)
            anchors.append(anchor(seed, (source(f"modules-{seed}", seed, records, vectors),), PILOT_ARMS))
        computed = [quantities(item.result, PILOT_ARMS) for item in anchors]
        recorded: Record = {
            "both_cards_bitwise": True,
            "a1_within_1e-4": True,
            "b3_fraction_every_anchor": True,
            "zero_module_identity": True,
            "static_step0": True,
            "known_answer": True,
        }
        clean = spreads(anchors, computed, recorded)
        self.assertIs(mapping(clean["v0"])["passed"], True)
        self.assertEqual(clean["v0_recorded"], recorded)
        for key, clause in (
            ("zero_module_identity", "zero-module identity"),
            ("a1_within_1e-4", "A1"),
            ("b3_fraction_every_anchor", "B3"),
            ("both_cards_bitwise", "replay"),
            ("static_step0", "S-static step 0"),
            ("known_answer", "slicing known answer"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(InstrumentHold, f"V0 failed at aggregation.*{clause}"):
                spreads(anchors, computed, {**recorded, key: False})

    def test_vectors_boundary_requires_n0_and_aligned_floating_loss_and_correctness(self) -> None:
        good = {
            "N0": {"loss": torch.ones(4), "correct": torch.zeros(4)},
            "M-sq": {"loss": torch.ones(4), "correct": torch.ones(4)},
        }
        self.assertEqual(set(arm_vectors(good)), {"N0", "M-sq"})
        for bad in (
            [torch.ones(4)],
            {"M-sq": {"loss": torch.ones(4), "correct": torch.ones(4)}},
            {"N0": {"loss": torch.ones(4)}},
            {"N0": {"loss": torch.ones(4), "correct": torch.ones(3)}},
            {"N0": {"loss": torch.ones(4, 1), "correct": torch.ones(4, 1)}},
            {"N0": {"loss": torch.ones(4, dtype=torch.int64), "correct": torch.ones(4)}},
            {"N0": {"loss": [1.0] * 4, "correct": torch.ones(4)}},
            {1: {"loss": torch.ones(4), "correct": torch.ones(4)}},
        ):  # fmt: skip
            with self.assertRaises(InputDataError):
                arm_vectors(bad)

    def test_an_unavailable_required_arm_holds_naming_anchor_and_arm(self) -> None:
        records, vectors = anchor_sources(3004, PILOT_ARMS)
        records["M-small-B"] = {"status": "unavailable", "cause": "M-small-B", "error": "nonfinite", "candidates": []}
        del vectors["M-small-B"]
        with self.assertRaisesRegex(InstrumentHold, r"3004.*M-small-B"):
            anchor(3004, (source("modules-3004", 3004, records, vectors),), PILOT_ARMS)

    def test_raw_candidates_parse_the_rate_and_unavailable_arms_list_their_nonfinite_candidates(self) -> None:
        records, vectors = anchor_sources(3001, PILOT_ARMS)
        candidates = [
            {"rate": 1.0, "steps": 200, "selection_loss": None, "finite": False},
            {"rate": 0.3, "steps": 3200, "selection_loss": None, "finite": False},
        ]
        records["M-exp"] = {"status": "unavailable", "cause": "M-exp", "error": "nonfinite", "candidates": candidates}
        vectors["M-wide→16:sliced"]["unfolded_loss"] = torch.full((N,), math.inf)
        result = anchor(3001, (source("modules-3001", 3001, records, vectors),), PILOT_ARMS)
        raw = result.result.raw_candidates
        self.assertEqual(raw["M-exp"][0][0], 1.0)
        self.assertTrue(all(math.isnan(value) for _, value in raw["M-exp"]))
        values = dict(raw["M-sq"])
        self.assertEqual(sorted(values), [0.3, 1.0])
        self.assertTrue(math.isnan(values[1.0]))
        self.assertTrue(math.isfinite(values[0.3]))
        self.assertEqual(result.report["harm_inventory"]["M-exp"], {"1.0@200": None, "0.3@3200": None})
        self.assertIn("1.0@200", result.report["harm_inventory"]["M-sq"])
        self.assertEqual(result.report["unfolded_nonfinite"], ["M-wide→16:sliced"])
        self.assertNotIn("M-wide→16:sliced", result.result.unfolded)
        self.assertEqual(result.report["unavailable_arms"], ["M-exp"])
        self.assertTrue(result.report["printed_counts"])

    def test_m_aff_reused_from_another_receipt_must_share_the_n0(self) -> None:
        records, vectors = anchor_sources(3002, PILOT_ARMS - {"M-aff"})
        maff, maff_vectors = anchor_sources(3002, frozenset(("M-aff",)))
        joined = anchor(
            3002,
            (source("modules-3002", 3002, records, vectors), source("maff-3002", 3002, maff, maff_vectors)),
            PILOT_ARMS,
        )
        self.assertEqual(joined.result.fold_residual, 0.001)
        other = source("maff-3002", 3002, maff, maff_vectors)
        other.payload["noop_sha256"] = "another"
        with self.assertRaisesRegex(PilotHold, "N0"):
            anchor(3002, (source("modules-3002", 3002, records, vectors), other), PILOT_ARMS)


def pilot_record(**changes: Json) -> Record:
    """A clean pilot-result.json as Task 17 publishes it."""
    record: Record = {
        "software_valid": True,
        "instrument_valid": True,
        "verdict": "pilot-clean; confirmatory verdict pending",
        "pilot_clean": True,
        "confirmatory_ready": False,
        "reason": "pilot-filled values must be reviewed and committed before the confirmatory run",
        "schedule": {"checkpoints": list(CHECKPOINTS), "checkpoints_b": list(CHECKPOINTS_B), "raised": False},
        "admitted_extended": list(EXTENDED_ORDER),
        "extended_truncated": False,
        "confirmatory_count": 32,
        "confirmatory_projection": carried_projection(),
        "static_admitted": False,
        "charged_gpu_process_seconds": 9000.0,
        "implementation_source_sha256": IDENTITY["source_sha256"],
        "implementation_commit": IDENTITY["commit"],
        **protocol_identity(),
    }
    record.update(changes)
    return record


def carried_projection(**changes: Json) -> Record:
    """The confirmatory projection pair a clean pilot result carries (I3): kernel 600 + 32 x 1,010, measured below."""
    record: Record = {
        "fixed_v0_gpu_process_seconds": 600.0,
        "per_anchor_gpu_process_seconds": 1_000.0,
        "static_per_anchor_gpu_process_seconds": 1_100.0,
        "kernel_gpu_process_seconds": 32_920.0,
        "measured_gpu_process_seconds": 32_600.0,
        "projected_gpu_process_seconds": 32_920.0,
    }
    record.update(changes)
    return record


class ConfirmatoryTests(unittest.TestCase):
    """§10 under its own custody and ceiling; children faked exactly as PilotRunTests fakes them."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base / "repo"
        self.results = self.repo / "docs/results/m0-pilot"
        self.results.mkdir(parents=True)
        self.pilot_result = self.results / "pilot-result.json"
        self.write_pilot()
        (self.results / "04-admission.json").write_text(json.dumps({"workloads": workloads_record(workloads())}))
        self.disposition = self.repo / "docs/certificates/m0-confirmatory-restart-disposition.md"
        self.disposition.parent.mkdir(parents=True)
        self.disposition.write_text("John: one restart from the restart block.\n")
        self.source_disposition = self.repo / "docs/certificates/m0-source-change-disposition.md"
        self.source_disposition.write_text("John: the confirmatory runs implementation version 2 (§10).\n")
        # Evidence must be tracked: a real index in the temporary repository, with runs/ ignored as in Sultai.
        (self.repo / ".gitignore").write_text("runs/\n")
        self.git("init", "-q")
        self.git("add", ".gitignore", "docs")

    def git(self, *arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=self.repo, check=True, capture_output=True)

    def prior_attempt(self, verdict: str) -> Path:
        """The first attempt's custody receipt, pointing at an output root whose result carries `verdict`."""
        first = self.base / "first"
        first.mkdir()
        (first / "confirmatory-result.json").write_text(json.dumps({"verdict": verdict}))
        prior = self.repo / CONFIRMATORY_CUSTODY
        prior.parent.mkdir()
        prior.write_text(json.dumps({"output": str(first)}))
        return first / "confirmatory-result.json"

    def assert_burnt_stop(self, root: Path, verdict: str) -> Record:
        result = read_record(root / "confirmatory-result.json")
        self.assertEqual(result["verdict"], verdict)
        self.assertIs(result["seed_block_burnt"], True)
        self.assertEqual(result["seed_block"], {"lineage": "confirmatory", "first": 7500, "last": 7531, "count": 32})
        for key in ("v1", "v2", "labels", "intervals", "programme_reading", "achieved_half_width"):
            self.assertNotIn(key, result)
        self.assertEqual(result["protocol_sha256"], protocol_identity()["protocol_sha256"])
        self.assertTrue((self.repo / CONFIRMATORY_CUSTODY).is_file())
        return result

    def write_pilot(self, **changes: Json) -> None:
        self.pilot_result.write_text(json.dumps(pilot_record(**changes)))

    def certificate(self, hours: int = 12, *, disposition: bool = False, source_change: str | None = None) -> Path:
        """`source_change`: "named" names the source-change disposition only; "listed" also hash-lists it."""
        paths = [self.pilot_result, self.results / "04-admission.json", *([self.disposition] if disposition else [])]
        paths += [self.source_disposition] if source_change == "listed" else []
        evidence = [{"path": str(path.relative_to(self.repo)), "sha256": file_sha(path)} for path in paths]
        named = (
            {}
            if source_change is None
            else {"source_change_disposition": str(self.source_disposition.relative_to(self.repo))}
        )
        path = self.base / "certificate.json"
        path.write_text(json.dumps({**IDENTITY, "evidence": evidence, "confirmatory_ceiling_hours": hours, **named}))
        return path

    def launch(
        self, certificate: Path, root_name: str = "run", *, static: bool = False, restart: Path | None = None,
        **scenario: object,
    ) -> tuple[int, Path, FakeCluster]:  # fmt: skip
        custody = CONFIRMATORY_CUSTODY if restart is None else RESTART_CUSTODY
        cluster = FakeCluster(self.repo, custody=custody, ordered="00-opening.json", **scenario)
        root = self.base / root_name
        with (
            patch("sultai_screen.pilot.subprocess.Popen", side_effect=cluster),
            patch("sultai_screen.m0.confirmatory.REPO", self.repo),
            patch("sultai_screen.m0.confirmatory.check_certificate", return_value=IDENTITY),
            patch("sultai_screen.m0.confirmatory.environment_identity", return_value={"torch": "fake"}),
            patch("sultai_screen.m0.confirmatory.check_cards", return_value={"gpu_query": "fake"}),
        ):
            code = confirmatory(
                root, self.base / "data", certificate, self.pilot_result, static=static, restart=restart
            )
        return code, root, cluster

    def assert_refused(self, certificate: Path, pattern: str, **options: object) -> None:
        with self.assertRaisesRegex(PilotHold, pattern):
            self.launch(certificate, **options)
        # Refused before custody: the single attempt is not consumed and no child (nor its root) exists.
        self.assertFalse((self.repo / CONFIRMATORY_CUSTODY).exists())
        self.assertFalse((self.repo / RESTART_CUSTODY).exists())
        self.assertFalse((self.base / "run").exists())

    def ledger_ceilings(self, root: Path) -> set[float]:
        ledgers = sorted(root.glob("ledger-*.json"))
        self.assertTrue(ledgers)
        return {read_record(path)["ceiling_gpu_process_seconds"] for path in ledgers}

    def arms_of(self, cluster: FakeCluster, name: str) -> set[str]:
        flags = next(flags for launched, _, _, flags, _ in cluster.launched if launched == name)
        return set(flags["--arms"].split(","))

    def test_refuses_a_pilot_result_that_is_not_clean_or_carries_another_identity(self) -> None:
        self.write_pilot(pilot_clean=False)
        self.assert_refused(self.certificate(), "clean pilot")
        self.write_pilot(protocol_sha256="0" * 64)
        self.assert_refused(self.certificate(), "protocol_sha256")
        self.write_pilot()
        changed = self.certificate()
        self.pilot_result.write_text(json.dumps(pilot_record(confirmatory_count=33)))
        self.assert_refused(changed, "evidence")
        missing = pilot_record()
        del missing["confirmatory_count"]
        self.pilot_result.write_text(json.dumps(missing))
        self.assert_refused(self.certificate(), "lacks 'confirmatory_count'")
        self.write_pilot(schedule={"checkpoints": list(CHECKPOINTS)})
        self.assert_refused(self.certificate(), "lacks 'checkpoints_b'")

    def test_the_confirmatory_source_is_the_pilots_unless_a_listed_disposition_names_the_change(self) -> None:
        # R27: the pilot result binds its implementation source; a mixed-source confirmatory needs a disposition.
        self.write_pilot(implementation_source_sha256="2" * 64)
        self.assert_refused(self.certificate(), "implementation source differs from the pilot's")
        self.assert_refused(self.certificate(source_change="named"), "source-change disposition .* not hash-listed")
        missing = pilot_record()
        del missing["implementation_source_sha256"]
        self.pilot_result.write_text(json.dumps(missing))
        self.assert_refused(self.certificate(), "lacks 'implementation_source_sha256'")
        self.write_pilot(implementation_source_sha256="2" * 64)
        # Listed, but the disposition does not name the confirmatory's source hash: refused (R28 minor 1).
        self.assert_refused(self.certificate(source_change="listed"), "does not name the confirmatory's source")
        source = str(IDENTITY["source_sha256"])
        self.source_disposition.write_text(f"John: the confirmatory runs implementation source {source} (§10).\n")
        code, root, _ = self.launch(self.certificate(source_change="listed"))
        self.assertEqual(code, 0, read_record(root / "confirmatory-result.json"))
        binding = read_record(self.repo / CONFIRMATORY_CUSTODY)["identity"]["implementation_source"]
        self.assertEqual((binding["pilot_source_sha256"], binding["same_source"]), ("2" * 64, False))
        listed = str(self.source_disposition.relative_to(self.repo))
        self.assertEqual(binding["source_change_disposition"], {listed: file_sha(self.source_disposition)})

    def test_the_larger_of_the_kernel_and_measured_projections_must_fit(self) -> None:
        # I3: the kernel projection fits (600 + 32 x 1,010); the measured per-anchor charge does not (600 + 32 x 1,400).
        self.write_pilot(confirmatory_projection=carried_projection(per_anchor_gpu_process_seconds=1_400.0))
        with self.assertRaisesRegex(FeasibilityHold, "measured"):
            self.launch(self.certificate())
        self.assertFalse((self.repo / CONFIRMATORY_CUSTODY).exists())
        self.assertFalse((self.base / "run").exists())

    def test_evidence_must_be_tracked_not_merely_hash_listed(self) -> None:
        ignored = self.repo / "runs/m0-pilot"
        ignored.mkdir(parents=True)
        for name in ("pilot-result.json", "04-admission.json"):
            (ignored / name).write_bytes((self.results / name).read_bytes())
        self.results, self.pilot_result = ignored, ignored / "pilot-result.json"
        self.assert_refused(self.certificate(), "the pilot result .* not tracked by git")

    def test_static_requires_the_pilot_admission_and_johns_14_hour_certificate(self) -> None:
        self.assert_refused(self.certificate(14), "S-static", static=True)
        self.write_pilot(static_admitted=True)
        self.assert_refused(self.certificate(12), "14", static=True)

    def test_the_whitelist_refuses_an_extended_arm_and_an_unrestricted_arm_list(self) -> None:
        allowed = frozenset(CONFIRMATORY_ARMS)
        for arms in ("M-sq,M-exp", "M-sq-fresh", "lottery", "M-wide→16:refit", "S-static"):
            with self.assertRaisesRegex(PilotHold, "whitelist"):
                whitelist([job("modules-7500", "modules", 7500, "confirmatory", "--arms", arms)], allowed)
        with self.assertRaisesRegex(PilotHold, "whitelist"):
            whitelist([job("retune-7500", "retune", 7500, "confirmatory", "--arms", "R-272s,R-all")], allowed)
        with self.assertRaisesRegex(PilotHold, "exactly one --arms"):
            whitelist([job("modules-7500", "modules", 7500, "confirmatory")], allowed)
        with self.assertRaisesRegex(PilotHold, "exactly one --arms"):
            whitelist(
                [job("modules-7500", "modules", 7500, "confirmatory", "--arms", "M-sq", "--arms", "M-exp")], allowed
            )
        for action in ("instrument", "timing", "cooled20", "static"):
            with self.assertRaisesRegex(PilotHold, action):
                whitelist([job(f"{action}-7500", action, 7500, "confirmatory")], allowed)
        whitelist([job("static-7500", "static", 7500, "confirmatory")], allowed | STATIC_ARMS)
        # Every launch, the reused pilot helpers' included, passes the whitelist before any child starts.
        cluster = FakeCluster(self.repo, custody=CONFIRMATORY_CUSTODY)
        root = self.base / "guarded"
        root.mkdir()
        supervisor = Supervisor(root, self.base / "data", ceiling=M0_CONFIRMATORY_SECONDS, per_card=1)
        run = Guarded(root, supervisor, str(IDENTITY["source_sha256"]), allowed)
        with (
            patch("sultai_screen.pilot.subprocess.Popen", side_effect=cluster),
            self.assertRaisesRegex(PilotHold, "whitelist"),
        ):
            run.run([job("modules-7500", "modules", 7500, "confirmatory", "--arms", "M-sq,M-exp")])
        self.assertEqual(cluster.launched, [])
        self.assertEqual(list(root.iterdir()), [])

    def test_restart_requires_the_listed_disposition_and_the_prior_custody_receipt(self) -> None:
        self.assert_refused(self.certificate(disposition=True), "prior", restart=self.disposition)
        failed = self.prior_attempt("instrument-failed")
        prior = self.repo / CONFIRMATORY_CUSTODY
        before = prior.read_bytes()
        with self.assertRaisesRegex(PilotHold, "the restart disposition .* not hash-listed"):
            self.launch(self.certificate(), restart=self.disposition)
        self.assertFalse((self.repo / RESTART_CUSTODY).exists())
        code, root, cluster = self.launch(self.certificate(disposition=True), restart=self.disposition)
        self.assertEqual(code, 0, read_record(root / "confirmatory-result.json"))
        self.assertEqual(cluster.violations, [])
        trained = [int(flags["--seed"]) for _, action, _, flags, _ in cluster.launched if action == "train"]
        self.assertEqual(trained, list(range(7700, 7732)))
        custody = read_record(self.repo / RESTART_CUSTODY)["identity"]
        self.assertEqual(custody["linked_confirmatory_custody"], {CONFIRMATORY_CUSTODY: file_sha(prior)})
        disposition = str(self.disposition.relative_to(self.repo))
        self.assertEqual(custody["restart_disposition"], {disposition: file_sha(self.disposition)})
        self.assertEqual(custody["linked_confirmatory_result"], {str(failed): file_sha(failed)})
        self.assertEqual(prior.read_bytes(), before)
        self.assertEqual(read_record(root / "confirmatory-result.json")["seed_block"]["lineage"], "restart")

    def test_restart_refuses_after_a_prior_attempt_that_read_v1_and_v2(self) -> None:
        self.prior_attempt("confirmatory-read")
        with self.assertRaisesRegex(PilotHold, "follows a failed attempt only"):
            self.launch(self.certificate(disposition=True), restart=self.disposition)
        self.assertFalse((self.repo / RESTART_CUSTODY).exists())
        self.assertFalse((self.base / "run").exists())

    def test_clean_confirmatory_runs_the_whitelist_on_fresh_anchors_under_its_own_ceiling(self) -> None:
        (self.repo / ".weft").mkdir()
        gate3 = self.repo / ".weft/gate3-pilot-attempt.json"
        gate3.write_text('{"gate3": true}\n')
        code, root, cluster = self.launch(self.certificate())
        result = read_record(root / "confirmatory-result.json")
        self.assertEqual(code, 0, result)
        self.assertEqual(cluster.violations, [])
        # Its own hard stop: 43,200 s in every ledger, one process per card, the M0 worker for every child.
        self.assertEqual(self.ledger_ceilings(root), {M0_CONFIRMATORY_SECONDS})
        self.assertEqual(M0_CONFIRMATORY_SECONDS, 43_200)
        previous = 0
        for path in sorted(root.glob("ledger-*.json")):
            entries = read_record(path)["entries"]
            cards = [entry["gpu"] for entry in entries[previous:]]
            previous = len(entries)
            self.assertEqual(len(cards), len(set(cards)), path.name)
        self.assertEqual({module for *_, module in cluster.launched}, {"sultai_screen.m0.worker"})
        # Fresh anchors exactly 7500..7531 at s = 1.0; only the V0 replay touches the development anchors.
        trained = [int(flags["--seed"]) for _, action, _, flags, _ in cluster.launched if action == "train"]
        self.assertEqual(trained, list(range(7500, 7532)))
        for name, action, _, flags, _ in cluster.launched:
            self.assertEqual(flags["--scale"], "1.0", name)
            if action in ("replay", "acceptance"):
                self.assertEqual(flags["--lineage"], "anchor", name)
            else:
                self.assertEqual(flags["--lineage"], "confirmatory", name)
                self.assertIn(int(flags["--seed"]), range(7500, 7532), name)
            self.assertNotIn(action, ("timing", "instrument", "static", "cooled20"), name)
        names = cluster.names()
        self.assertIn("replay-3000-gpu1", names)
        self.assertLess(names.index("maff-7500"), names.index("fold-check-7500"))
        self.assertLess(names.index("fold-check-7500"), names.index("maff-7502"))
        # The §10 whitelist and nothing else: M-aff in V0, the rest in the core children.
        for seed in range(7500, 7532):
            ran = self.arms_of(cluster, f"maff-{seed}")
            ran |= self.arms_of(cluster, f"modules-{seed}") | self.arms_of(cluster, f"retune-{seed}")
            self.assertEqual(ran, set(CONFIRMATORY_ARMS), seed)
            self.assertNotIn("M-aff", self.arms_of(cluster, f"modules-{seed}"))
        # V1 and V2 read once, with the M0 identity and the R20 half-widths beside the D1 interval.
        for key, value in protocol_identity().items():
            self.assertEqual(result[key], value)
        self.assertIn(result["v1"], ("present", "re-tuning equivalent", "indeterminate"))
        self.assertIn(result["v2"], ("helps", "hurts", "indeterminate"))
        self.assertNotIn("fresh-data effect", result["labels"])
        self.assertIsNone(result["static_reading"])
        self.assertEqual(result["anchors"], list(range(7500, 7532)))
        self.assertEqual(result["seed_block"], {"lineage": "confirmatory", "first": 7500, "last": 7531, "count": 32})
        interval = result["intervals"]["D1"]
        self.assertLessEqual(interval["lower"], interval["median"])
        self.assertLessEqual(interval["median"], interval["upper"])
        self.assertIn("sign_test_lower", interval)
        widths = result["achieved_half_width"]
        self.assertEqual(set(widths), {"D2", "D3", "D5"})
        self.assertTrue(math.isfinite(widths["D2"]) and math.isfinite(widths["D3"]))
        self.assertIsNone(widths["D5"])
        self.assertEqual(result["ceiling_gpu_process_seconds"], M0_CONFIRMATORY_SECONDS)
        self.assertIs(result["v0"]["zero_module_identity"], True)
        # I3 and M7: the launch took the larger of the kernel projection (on the pilot's replay and acceptance fixed
        # term) and the measured one.
        projection = read_record(root / "00-opening.json")["projection"]
        self.assertEqual(projection["kernel"]["fixed_v0_gpu_process_seconds"], 600.0)
        self.assertEqual(projection["kernel"]["projected_gpu_process_seconds"], 600.0 + 32 * 1_010.0)
        self.assertEqual(projection["measured"]["projected_gpu_process_seconds"], 600.0 + 32 * 1_000.0)
        self.assertEqual(projection["projected_gpu_process_seconds"], 600.0 + 32 * 1_010.0)
        # Its own custody; the pilot's and Gate 3's receipts are never written.
        custody = read_record(self.repo / CONFIRMATORY_CUSTODY)
        self.assertEqual(custody["identity"]["protocol_version"], "m0-v4")
        listed = {"docs/results/m0-pilot/pilot-result.json": file_sha(self.pilot_result)}
        self.assertEqual(custody["identity"]["pilot_result"], listed)
        self.assertFalse((self.repo / CUSTODY).exists())
        self.assertEqual(gate3.read_text(), '{"gate3": true}\n')
        with self.assertRaisesRegex(PilotHold, "custody"):
            self.launch(self.certificate(), "another-root")

    def test_static_runs_s_static_under_the_14_hour_ceiling(self) -> None:
        self.write_pilot(static_admitted=True)
        code, root, cluster = self.launch(self.certificate(14), static=True, boost={"S-static:refit": 0.05})
        result = read_record(root / "confirmatory-result.json")
        self.assertEqual(code, 0, result)
        self.assertEqual(self.ledger_ceilings(root), {M0_CONFIRMATORY_STATIC_SECONDS})
        self.assertEqual(M0_CONFIRMATORY_STATIC_SECONDS, 50_400)
        static = sorted(name for name in cluster.names() if name.startswith("static-"))
        self.assertEqual(static, [f"static-{seed}" for seed in range(7500, 7532)])
        self.assertLessEqual(STATIC_ARMS, self.arms_of(cluster, "modules-7500"))
        # D5 = loss(M-sq) - loss(S-static:refit) ~ 0.05 > M_s on every anchor, so the reading is read at n = 32.
        self.assertGreater(result["intervals"]["D5"]["lower"], 0.02)
        self.assertEqual(result["static_reading"], "design-in beats insertion")
        self.assertTrue(math.isfinite(result["achieved_half_width"]["D5"]))
        self.assertEqual(result["ceiling_gpu_process_seconds"], M0_CONFIRMATORY_STATIC_SECONDS)

    def test_a_v0_failure_publishes_instrument_failed_burns_the_block_and_reads_nothing(self) -> None:
        code, root, cluster = self.launch(self.certificate(), r_aff={7503: 0.006})
        self.assertEqual(code, 2)
        result = self.assert_burnt_stop(root, "instrument-failed")
        self.assertEqual(result["stage"], "v0")
        self.assertIn("7503", result["error"])
        self.assertFalse(any(name.startswith(("tcool-", "modules-", "retune-")) for name in cluster.names()))

    def test_a_replay_mismatch_is_execution_invalid_and_still_burns_the_block(self) -> None:
        # Rulings R21 (inherited Gate 3 class) and R22 (custody consumed the block before any anchor trained).
        code, root, cluster = self.launch(self.certificate(), replay_mismatch=frozenset(("replay-3002",)))
        self.assertEqual(code, 2)
        result = self.assert_burnt_stop(root, "execution-invalid")
        self.assertIs(result["software_valid"], False)
        self.assertEqual(result["stage"], "replay")
        self.assertFalse(any(action == "train" for _, action, _, _, _ in cluster.launched))

    def test_an_acceptance_failure_is_instrument_failed_and_burns_the_block(self) -> None:
        code, root, cluster = self.launch(self.certificate(), acceptance_failed=frozenset(("acceptance-1",)))
        self.assertEqual(code, 2)
        result = self.assert_burnt_stop(root, "instrument-failed")
        self.assertEqual(result["stage"], "replay")
        self.assertIn("(iii)", result["error"])
        self.assertFalse(any(action == "train" for _, action, _, _, _ in cluster.launched))

    def test_the_pilot_count_sets_the_anchors_exactly(self) -> None:
        self.write_pilot(confirmatory_count=33)
        code, root, cluster = self.launch(self.certificate())
        result = read_record(root / "confirmatory-result.json")
        self.assertEqual(code, 0, result)
        trained = [int(flags["--seed"]) for _, action, _, flags, _ in cluster.launched if action == "train"]
        self.assertEqual(trained, list(range(7500, 7533)))
        self.assertEqual(result["anchors"], list(range(7500, 7533)))
        # R24 at odd n: every anchor's children (train, noop, maff, fold-check, tcool, modules, retune) share its card.
        cards: dict[int, set[str]] = {}
        for _, _, gpu, flags, _ in cluster.launched:
            if flags["--lineage"] == "confirmatory":
                cards.setdefault(int(flags["--seed"]), set()).add(gpu)
        self.assertEqual(set(cards), set(range(7500, 7533)))
        for seed, used in cards.items():
            self.assertEqual(used, {str((seed - 7500) % 2)}, seed)
        # One process per card in every Supervisor.run call.
        previous = 0
        for path in sorted(root.glob("ledger-*.json")):
            entries = read_record(path)["entries"]
            batch = [entry["gpu"] for entry in entries[previous:]]
            previous = len(entries)
            self.assertEqual(len(batch), len(set(batch)), path.name)

    def test_an_unavailable_required_arm_holds_at_aggregation_and_drops_no_anchor(self) -> None:
        code, root, cluster = self.launch(self.certificate(), unavailable={7510: frozenset(("M-small",))})
        self.assertEqual(code, 2)
        result = read_record(root / "confirmatory-result.json")
        self.assertEqual((result["verdict"], result["stage"]), ("instrument-failed", "aggregation"))
        self.assertIn("7510", result["error"])
        self.assertIn("M-small", result["error"])
        self.assertNotIn("v1", result)
        launched = set(cluster.names())
        self.assertTrue(all(f"modules-{seed}" in launched for seed in range(7500, 7532)))


class SupervisionTests(unittest.TestCase):
    def test_check_cards_requires_both_pinned_idle_cards_and_no_compute_apps(self) -> None:
        good = f"0, {GPU_UUIDS[0]}, 0, 1\n1, {GPU_UUIDS[1]}, 0, 1\n"
        for query, apps, ok in (
            (good, "", True),
            (good.replace(", 0, 1\n1", ", 7, 1\n1"), "", False),
            (good.replace(GPU_UUIDS[1], "GPU-other"), "", False),
            (good, f"{GPU_UUIDS[0]}, 123\n", False),
        ):
            with patch("sultai_screen.m0.pilot.subprocess.check_output", side_effect=(query, apps)):
                if ok:
                    self.assertEqual(check_cards()["gpu_query"], query)
                else:
                    with self.assertRaises(PilotHold):
                        check_cards()

    def test_certificate_must_carry_the_m0_identity_not_gate3(self) -> None:
        from sultai_screen.contracts import protocol_identity as gate3_identity

        approved: Record = {**gate3_identity(), "quality_passed": True}
        with self.assertRaisesRegex(PilotHold, "protocol identity"):
            check_certificate(approved)

    def test_stop_report_has_the_gate3_shape_with_the_m0_identity(self) -> None:
        report = m0_stop_report(FeasibilityHold("over"), "admission", 12.5, IDENTITY)
        self.assertEqual(report["verdict"], "hold-feasibility")
        self.assertEqual(report["charged_gpu_process_seconds"], 12.5)
        self.assertEqual(report["implementation_source_sha256"], IDENTITY["source_sha256"])
        self.assertEqual(report["implementation_commit"], IDENTITY["commit"])
        for key, value in protocol_identity().items():
            self.assertEqual(report[key], value)
        self.assertEqual(m0_stop_report(InstrumentHold("x"), "core", 0.0, IDENTITY)["verdict"], "instrument-failed")
        self.assertEqual(m0_stop_report(PilotHold("x"), "core", 0.0, IDENTITY)["verdict"], "execution-invalid")

    def test_slicing_known_answer_passes_on_the_cpu_fixture(self) -> None:
        record = slicing_known_answer()
        self.assertTrue(record["known_answer"])
        self.assertLess(record["rank_cut_max_difference"], 1e-6)
        self.assertGreater(record["random_cut_max_difference"], 1e-4)
        self.assertGreater(record["wrong_cut_max_difference"], 1e-4)

    def test_import_isolation(self) -> None:
        code = (
            "import sys, sultai_screen.m0.receipts; "
            "bad = [m for m in ('sultai_screen.screen', 'sultai_screen.pilot', 'sultai_screen.former_bridge', "
            "'sultai_screen.alignment') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], env={"PYTHONPATH": str(SRC)}, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        tree = ast.parse((SRC / "sultai_screen/m0/pilot.py").read_text())
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        for forbidden in ("sultai_screen.screen", "sultai_screen.former_bridge", "sultai_screen.alignment"):
            self.assertNotIn(forbidden, imported)
        self.assertIn("sultai_screen.pilot", imported)


if __name__ == "__main__":
    unittest.main()
