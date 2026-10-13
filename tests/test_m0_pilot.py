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
from sultai_screen.contracts import FeasibilityHold, InstrumentHold, Json, PilotHold, Record, file_sha, read_record
from sultai_screen.m0 import pilot as m0_pilot
from sultai_screen.m0.costs import EXTENDED_ORDER, Workloads, extended_cost, stage_seconds
from sultai_screen.m0.identity import CHECKPOINTS, CHECKPOINTS_B, M0_PILOT_SECONDS, protocol_identity
from sultai_screen.m0.modules import FRONTIER_WIDTHS
from sultai_screen.m0.pilot import (
    CUSTODY,
    admission,
    already_run,
    check_cards,
    check_certificate,
    m0_stop_report,
    pilot,
    slicing_known_answer,
)
from sultai_screen.m0.receipts import Source, anchor, arm_vectors, derive_workloads
from sultai_screen.m0.stats import KNOWN_ARMS, PILOT_ARMS
from sultai_screen.m0.worker import ATTEMPT4_FINAL_STATE
from sultai_screen.pilot import GPU_UUIDS

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


def arm_record(arm: str, seed: int, n0: Tensor, *, r_aff: float, certified: bool) -> tuple[Record, dict[str, Tensor]]:
    index = sorted(KNOWN_ARMS).index(arm)
    gain = 0.02 + 0.001 * (index % 10) + 0.002 * ((seed + index) % 4)
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

    def __init__(self, repo: Path, **scenario: object) -> None:
        self.repo = repo
        self.replay_mismatch = scenario.pop("replay_mismatch", frozenset())
        self.r_aff = scenario.pop("r_aff", {})
        self.uncertified = scenario.pop("uncertified", frozenset())
        self.module_seconds = scenario.pop("module_seconds", 10.0)
        self.unavailable = scenario.pop("unavailable", {})
        self.fold_mismatch = scenario.pop("fold_mismatch", False)
        if scenario:
            raise AssertionError(f"unknown scenario keys {sorted(scenario)}")
        self.launched: list[tuple[str, str, str, dict[str, str], str]] = []
        self.violations: list[str] = []

    def __call__(self, command: list[str], **kwargs: dict[str, str]) -> subprocess.Popen[bytes]:
        module = command[command.index("-m") + 1]
        action = command[command.index("-m") + 2]
        tokens = command[command.index("-m") + 3 :]
        flags = dict(zip(tokens[0::2], tokens[1::2], strict=True))
        output = Path(flags["--output"])
        gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        if not (self.repo / CUSTODY).is_file():
            self.violations.append(f"{output.name} launched before custody")
        if output.name.startswith("modules-") and not (output.parent / "04-schedule.json").is_file():
            self.violations.append(f"{output.name} launched before 04-schedule.json")
        self.launched.append((output.name, action, gpu, flags, module))
        output.mkdir()
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
                "epoch_state_sha256": [f"epoch-{seed}"],
                "forward_backward": {"output_sha256": f"fb-{seed}"},
                "parameters": PARAMETERS[scale],
                "training_seconds": 1.0,
            }
        if action == "acceptance":
            items = ("resume_bitwise", "checkpoint_bitwise", "static_step0_bitwise", "evaluation_consistent")
            return {**base, **dict.fromkeys(items, True), "straight_epoch_state_sha256": ["s"], "checkpoint_rate": 0.3}
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
                "arm_losses": {"A1": 0.8, "A2": 0.82, "A3": 0.805, "B3": 0.81},
                "oracle_gain_fraction_of_A3": 0.95,
            }
            rungs = {"0.1": 0.2, "0.3": 0.3, "0.03": 0.06}
            return {**base, "materiality": {"seed": seed, "rungs": {"P1": rungs}}, "posthoc": posthoc}
        if action == "noop":
            return {**base, "noop_choice": "raw", "noop_sha256": f"noop-{seed}"}
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
            return {**base, **written, "static_sha256": f"static-{seed}"}
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
                records[arm], vectors[arm] = arm_record(arm, seed, n0, r_aff=r_aff, certified=certified)
            torch.save(vectors, output / "vectors.pt")
            return {
                **base,
                **schedule,
                "noop_sha256": f"noop-{seed}",
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

    def launch(self, root_name: str = "run", **scenario: object) -> tuple[int, Path, FakeCluster, list[str]]:
        cluster = FakeCluster(self.repo, **scenario)
        published: list[str] = []
        real_publish = m0_pilot.publish

        def record(path: Path, value: Json) -> None:
            published.append(path.name)
            real_publish(path, value)

        root = self.base / root_name
        with (
            patch("sultai_screen.pilot.subprocess.Popen", side_effect=cluster),
            patch("sultai_screen.m0.pilot.REPO", self.repo),
            patch("sultai_screen.m0.pilot.check_certificate", return_value=IDENTITY),
            patch("sultai_screen.m0.pilot.environment_identity", return_value={"torch": "fake"}),
            patch("sultai_screen.m0.pilot.check_cards", return_value={"gpu_query": "fake"}),
            patch("sultai_screen.m0.pilot.publish", side_effect=record),
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
    )


class AdmissionTests(unittest.TestCase):
    def test_projection_is_charged_plus_remaining_work_after_two_stages_have_run(self) -> None:
        w = workloads()
        charged = stage_seconds("V0", w) + stage_seconds("hosts", w)
        done = already_run(w, raised=False)
        self.assertEqual(done["N0"]["seconds"], 6 * w.fold_and_calibrations)
        self.assertEqual(done["M-aff"]["seconds"], 6 * w.grid["aff"])
        record = admission(charged, w, M0_PILOT_SECONDS, done["seconds"])
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
        self.assertLessEqual(expected, M0_PILOT_SECONDS)
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
        }
        w, sources = derive_workloads(timing, charges, fixed_v0=1200.0, ratio=1.0)
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


class ReceiptTests(unittest.TestCase):
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
        report = m0_stop_report(FeasibilityHold("over"), "admission", 12.5)
        self.assertEqual(report["verdict"], "hold-feasibility")
        self.assertEqual(report["charged_gpu_process_seconds"], 12.5)
        for key, value in protocol_identity().items():
            self.assertEqual(report[key], value)
        self.assertEqual(m0_stop_report(InstrumentHold("x"), "core", 0.0)["verdict"], "instrument-failed")
        self.assertEqual(m0_stop_report(PilotHold("x"), "core", 0.0)["verdict"], "execution-invalid")

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
