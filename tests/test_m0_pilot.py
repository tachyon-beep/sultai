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
from sultai_screen.m0.confirmatory import CUSTODY as CONFIRMATORY_CUSTODY
from sultai_screen.m0.confirmatory import RESTART_CUSTODY, Guarded, confirmatory, whitelist
from sultai_screen.m0.costs import (
    EXTENDED_ORDER,
    M0_CONFIRMATORY_STATIC_SECONDS,
    STATIC_ARMS,
    Workloads,
    extended_cost,
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
    CUSTODY,
    admission,
    already_run,
    check_cards,
    check_certificate,
    job,
    m0_stop_report,
    pilot,
    slicing_known_answer,
)
from sultai_screen.m0.receipts import Source, anchor, arm_vectors, derive_workloads, workloads_record
from sultai_screen.m0.stats import CONFIRMATORY_ARMS, KNOWN_ARMS, PILOT_ARMS
from sultai_screen.m0.worker import ATTEMPT4_FINAL_STATE
from sultai_screen.pilot import GPU_UUIDS, Supervisor

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
        self.acceptance_failed = scenario.pop("acceptance_failed", frozenset())
        self.boost = scenario.pop("boost", {})
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
                boost = self.boost[arm] if arm in self.boost else 0.0
                records[arm], vectors[arm] = arm_record(arm, seed, n0, r_aff=r_aff, certified=certified, boost=boost)
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
        "confirmatory_count": 32,
        "static_admitted": False,
        "charged_gpu_process_seconds": 9000.0,
        **protocol_identity(),
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

    def certificate(self, hours: int = 12, *, disposition: bool = False) -> Path:
        paths = [self.pilot_result, self.results / "04-admission.json", *([self.disposition] if disposition else [])]
        evidence = [{"path": str(path.relative_to(self.repo)), "sha256": file_sha(path)} for path in paths]
        path = self.base / "certificate.json"
        path.write_text(json.dumps({**IDENTITY, "evidence": evidence, "confirmatory_ceiling_hours": hours}))
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
        run = Guarded(root, Supervisor(root, self.base / "data", ceiling=M0_CONFIRMATORY_SECONDS, per_card=1), allowed)
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
