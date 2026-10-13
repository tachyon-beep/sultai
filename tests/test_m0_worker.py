"""CPU controls for the M0 worker: seed-admitting CLI, receipts, and every child action on a 64-image fixture."""

from __future__ import annotations

import io
import math
import re
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch
from torch import Tensor

from sultai_screen.contracts import Json, PilotHold, Record, file_sha, mapping, publish, read_record, sequence
from sultai_screen.data import Dataset, Seal
from sultai_screen.host import REFERENCE_WIDTHS, state_hash
from sultai_screen.m0 import worker
from sultai_screen.m0.calibrate import load_noop
from sultai_screen.m0.identity import protocol_identity
from sultai_screen.m0.stats import CORE_ARMS, PILOT_ARMS, AnchorResult, ArmResult, quantities

REPO = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
IDENTITY: Record = {"commit": "fixture", "source_sha256": "f" * 64, **protocol_identity()}
SEEN = torch.arange(48, 64)
HALF = (32, 32, 64, 64)
RETUNE_ONLY = frozenset(("R-272s", "R-aff"))
CUT_ARMS = (
    "M-wide→8:rank:sliced",
    "M-wide→8:random:sliced",
    "M-wide→8:ablation",
    "M-wide→16:sliced",
    "M-wide→4:sliced",
    "M-wide→2:sliced",
)
LABEL = re.compile(r"^[0-9.e+-]+@[0-9]+$")


def fixture() -> Dataset:
    rng = torch.Generator().manual_seed(16)
    images = torch.randint(0, 256, (64, 3, 32, 32), generator=rng, dtype=torch.uint8)
    labels = torch.randint(0, 10, (64,), generator=rng)
    ids = torch.arange(64)
    return Dataset(images, labels, ids, ids[0:16].clone(), ids[16:32].clone(), ids[32:48].clone())


@contextmanager
def patched() -> Iterator[None]:
    data = fixture()
    with ExitStack() as stack:
        for target, value in (
            ("configure", CPU),
            ("load_dataset", data),
            ("install_seal", Seal()),
            ("provenance", IDENTITY),
            ("seen_fit", SEEN),
        ):
            stack.enter_context(patch(f"sultai_screen.m0.worker.{target}", return_value=value))
        for target, constant in (
            ("CHECKPOINTS", (1, 2)),
            ("CHECKPOINTS_B", (2, 4)),
            ("MODULE_RATES", (0.1,)),
            ("RETUNE_RATES", (0.1,)),
            ("RALL_CHECKPOINTS", (1, 2)),
            ("SURVIVAL_REFIT_STEPS", 3),
        ):
            stack.enter_context(patch(f"sultai_screen.m0.worker.{target}", constant))
        stack.enter_context(patch("sultai_screen.fit.PLATEAU_WINDOW", 1))
        yield


def run(action: str, output: Path, seed: int = 3000, lineage: str = "anchor", *extra: str) -> Record:
    scale = {"anchor": "1.0", "frontier-0.5": "0.5", "cooled-20": "1.0"}[lineage]
    argv = [action, "--output", str(output), "--data-root", "/nonexistent", "--seed", str(seed)]
    argv += ["--width", "0", "--lineage", lineage, "--scale", scale, *extra]
    with patched():
        if worker.main(argv) != 0:
            raise AssertionError(f"{action} child failed")
    return read_record(output / "result.json")


def payload_of(record: Record) -> Record:
    return mapping(record["payload"])


def keys(value: Json) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys(item)


def vectors(root: Path) -> dict[str, dict[str, Tensor]]:
    loaded: dict[str, dict[str, Tensor]] = torch.load(root / "vectors.pt", weights_only=True)
    return loaded


def optional_bool(value: Json) -> bool | None:
    return value if isinstance(value, bool) else None


def optional_float(value: Json) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class WorkerTests(unittest.TestCase):
    temporary: tempfile.TemporaryDirectory[str]
    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        train = cls.root / "train"
        run("train", train)
        run("noop", cls.root / "noop", 3000, "anchor", "--training-root", str(train))
        run("tcool", cls.root / "tcool", 3000, "anchor", "--training-root", str(train))
        run("static", cls.root / "static")
        run("train", cls.root / "half", 6000, "frontier-0.5")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def roots(self) -> tuple[str, ...]:
        return (
            "--noop-root",
            str(self.root / "noop"),
            "--tcool-root",
            str(self.root / "tcool"),
            "--static-root",
            str(self.root / "static"),
        )

    def test_cli_admits_seeds_by_lineage_scale_width_and_epochs(self) -> None:
        base = ["train", "--output", "/nonexistent/out", "--data-root", "/nonexistent", "--width", "0"]
        refused = (
            ["--seed", "3000", "--lineage", "frontier-0.5", "--scale", "0.5"],
            ["--seed", "6000", "--lineage", "frontier-0.5", "--scale", "1.0"],
            ["--seed", "6150", "--lineage", "frontier-0.5", "--scale", "0.5"],
            ["--seed", "6000", "--lineage", "frontier-0.5", "--scale", "0.5", "--epochs", "20"],
            ["--seed", "6018", "--lineage", "cooled-20", "--scale", "1.0"],
            ["--seed", "3000", "--lineage", "anchor", "--scale", "1.0", "--width", "4"],
            ["--seed", "3000", "--lineage", "nonsense", "--scale", "1.0"],
        )
        for extra in refused:
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                worker.main(base + extra)
            self.assertEqual(raised.exception.code, 2)
        with patch("sultai_screen.m0.worker.worker", return_value=0) as child:
            code = worker.main([*base, "--seed", "6000", "--lineage", "frontier-0.5", "--scale", "0.5"])
        self.assertEqual(code, 0)
        request = child.call_args.args[0]
        self.assertEqual((request.seed, request.lineage, request.widths), (6000, "frontier-0.5", HALF))
        self.assertEqual(request.arms, None)
        with patch("sultai_screen.m0.worker.worker", return_value=0) as child:
            worker.main(
                [
                    *("cooled20", "--output", "/x", "--data-root", "/y", "--seed", "6018"),
                    *("--lineage", "cooled-20", "--scale", "1", "--epochs", "20", "--arms", "M-sq,M-aff"),
                    *("--magnitudes", "/ignored"),
                ]
            )
        request = child.call_args.args[0]
        self.assertEqual((request.epochs, request.arms, request.widths), (20, ("M-sq", "M-aff"), REFERENCE_WIDTHS))
        with patch("sultai_screen.m0.worker.worker", return_value=0) as child:
            worker.main(
                [
                    "noop",
                    "--output",
                    "/x",
                    "--data-root",
                    "/y",
                    "--seed",
                    "6018",
                    "--lineage",
                    "cooled-20",
                    "--scale",
                    "1",
                ]
            )
        self.assertEqual(child.call_args.args[0].lineage, "cooled-20")

    def test_attempt4_hashes_are_the_published_receipts(self) -> None:
        self.assertEqual(worker.DEV_ARTIFACT, REPO / "docs/results/gate3-split-2026-10-11/dev-indices-int32-le.bin")
        self.assertTrue(worker.DEV_ARTIFACT.is_file())
        self.assertEqual(
            (worker.SURVIVAL_REFIT_STEPS, worker.RALL_CHECKPOINTS[0], worker.RALL_CHECKPOINTS[-1]), (800, 40, 800)
        )
        self.assertEqual(len(worker.RALL_CHECKPOINTS), 20)
        self.assertEqual(set(worker.ATTEMPT4_FINAL_STATE), set(range(3000, 3006)))
        for seed, pinned in worker.ATTEMPT4_FINAL_STATE.items():
            receipt = REPO / f"docs/results/gate3-pilot-2026-10-13-linked-3/intact-{seed}/result.json"
            self.assertEqual(pinned, payload_of(read_record(receipt))["final_state_sha256"])
            self.assertEqual(len(pinned), 64)

    def test_worker_imports_neither_the_gate3_screen_nor_its_pilot(self) -> None:
        probe = (
            "import sys, sultai_screen.m0.worker\n"
            "banned = ('sultai_screen.screen', 'sultai_screen.pilot', 'sultai_screen.former_bridge',"
            " 'sultai_screen.alignment')\n"
            "print([name for name in banned if name in sys.modules])\n"
        )
        out = subprocess.check_output(
            [sys.executable, "-B", "-c", probe], env={"PYTHONPATH": str(REPO / "src")}, text=True
        )
        self.assertEqual(out.strip(), "[]")

    def test_train_binds_host_optimiser_and_stream_and_load_training_reads_the_width_tuple(self) -> None:
        record = run("train", self.root / "train-check")
        self.assertEqual(mapping(record["identity"])["protocol_version"], "m0-v4")
        self.assertEqual(record["peak_cuda_bytes"], 0)
        payload = payload_of(record)
        out = self.root / "train-check"
        self.assertEqual(payload["checkpoint_sha256"], file_sha(out / "host.pt"))
        self.assertEqual(payload["optimizer_sha256"], file_sha(out / "optimizer.pt"))
        self.assertEqual(payload["loss_stream_sha256"], file_sha(out / "loss-stream.bin"))
        self.assertEqual(
            (payload["epochs"], payload["schedule"], payload["steps"], payload["lineage"]),
            (10, "constant-0.05", 20, "anchor"),
        )
        self.assertEqual(payload["widths"], list(REFERENCE_WIDTHS))
        self.assertEqual(
            payload["final_state_sha256"],
            payload_of(read_record(self.root / "train/result.json"))["final_state_sha256"],
        )
        with patched():
            host = worker.load_training(self.root / "half", 6000, "frontier-0.5", HALF, CPU, frozen=True)
            self.assertEqual(
                state_hash(host), payload_of(read_record(self.root / "half/result.json"))["final_state_sha256"]
            )
            with self.assertRaisesRegex(PilotHold, "width"):
                worker.load_training(self.root / "half", 6000, "frontier-0.5", REFERENCE_WIDTHS, CPU, frozen=True)
            with self.assertRaisesRegex(PilotHold, "lineage"):
                worker.load_training(self.root / "half", 6000, "anchor", HALF, CPU, frozen=True)
            with patch("sultai_screen.m0.worker.provenance", return_value={**IDENTITY, "source_sha256": "0" * 64}):
                with self.assertRaisesRegex(PilotHold, "source"):
                    worker.load_training(self.root / "half", 6000, "frontier-0.5", HALF, CPU, frozen=True)
        half = payload_of(read_record(self.root / "half/result.json"))
        self.assertEqual(half["widths"], list(HALF))
        self.assertEqual(half["scale"], 0.5)

    def test_cooled20_replay_and_tcool_children(self) -> None:
        cooled = payload_of(run("cooled20", self.root / "cooled", 6018, "cooled-20", "--epochs", "20"))
        self.assertEqual((cooled["epochs"], cooled["schedule"], cooled["steps"]), (20, "cooled-20", 40))
        replay = payload_of(run("replay", self.root / "replay"))
        self.assertIs(replay["matches_attempt4"], False)
        self.assertEqual(replay["attempt4_final_state_sha256"], worker.ATTEMPT4_FINAL_STATE[3000])
        self.assertEqual(
            replay["final_state_sha256"], payload_of(read_record(self.root / "train/result.json"))["final_state_sha256"]
        )
        tcool = payload_of(read_record(self.root / "tcool/result.json"))
        self.assertEqual((tcool["epochs"], tcool["schedule"], tcool["steps"]), (5, "tcool", 10))
        self.assertIn(tcool["noop_choice"], ("R-bn", "raw"))
        self.assertEqual(tcool["checkpoint_sha256"], file_sha(self.root / "tcool/host.pt"))
        self.assertNotEqual(tcool["final_state_sha256"], tcool["source_final_state_sha256"])
        noop = load_noop(self.root / "tcool", 3000, REFERENCE_WIDTHS, str(tcool["noop_sha256"]), CPU)
        self.assertEqual(state_hash(noop), tcool["noop_sha256"])
        static = payload_of(read_record(self.root / "static/result.json"))
        self.assertEqual(static["module_sha256"], file_sha(self.root / "static/static-module.pt"))
        self.assertEqual(static["checkpoint_sha256"], file_sha(self.root / "static/host.pt"))
        self.assertIn("static_sha256", static)

    def test_noop_publishes_the_choice_and_never_a_development_number(self) -> None:
        payload = payload_of(read_record(self.root / "noop/result.json"))
        self.assertIn(payload["noop_choice"], ("R-bn", "raw"))
        noop = load_noop(self.root / "noop", 3000, REFERENCE_WIDTHS, str(payload["noop_sha256"]), CPU)
        self.assertEqual(state_hash(noop), payload["noop_sha256"])
        chosen = payload["recalibrated_sha256"] if payload["noop_choice"] == "R-bn" else payload["raw_sha256"]
        self.assertEqual(chosen, payload["noop_sha256"])
        bias = payload["class_bias"]
        assert isinstance(bias, list)
        self.assertEqual(len(bias), 11)
        temperature = optional_float(payload["temperature"])
        assert temperature is not None
        self.assertGreater(temperature, 0)
        for name in ("raw", "rbn", "rtemp", "rcal"):
            self.assertTrue(math.isfinite(float(str(payload[f"{name}_selection_loss"]))))
        self.assertFalse([key for key in keys(payload) if "development" in key])

    def test_modules_runs_every_module_arm_with_vectors_harm_cuts_and_single_survival_refits(self) -> None:
        out = self.root / "modules"
        record = run("modules", out, 3000, "anchor", *self.roots())
        payload = payload_of(record)
        arms = mapping(payload["arms"])
        expected = (set(CORE_ARMS) - RETUNE_ONLY) | {"M-exp", "lottery", "S-static", "S-static:refit"}
        self.assertEqual(set(arms), expected)
        self.assertEqual({str(name) for name in sequence(payload["arms_run"])}, expected)
        self.assertEqual(payload["vectors_sha256"], file_sha(out / "vectors.pt"))
        saved = vectors(out)
        self.assertEqual(set(saved), expected | {"N0"})
        for name, entry in saved.items():
            self.assertEqual(entry["loss"].shape, (16,), name)
            self.assertTrue(bool(((entry["correct"] == 0) | (entry["correct"] == 1)).all()), name)
        for name in expected:
            arm = mapping(arms[name])
            self.assertAlmostEqual(float(str(arm["development_loss"])), float(saved[name]["loss"].double().mean()))
            for label in mapping(arm["harm_inventory"]):
                self.assertRegex(label, LABEL)
        self.assertEqual(len(mapping(mapping(arms["M-sq"])["harm_inventory"])), 2)
        self.assertEqual(len(mapping(mapping(arms["M-small-B"])["harm_inventory"])), 2)
        for name in CUT_ARMS:
            arm = mapping(arms[name])
            h = int(name.split("→")[1].split(":")[0])
            survivors = arm["survivors"]
            assert isinstance(survivors, list)
            self.assertEqual(len(survivors), h, name)
            self.assertIn("unfolded_development_loss", arm)
            self.assertEqual(saved[name]["unfolded_loss"].shape, (16,))
        wide_rate = mapping(arms["M-wide"])["rate"]
        for h in (16, 4, 2):
            arm = mapping(arms[f"M-wide→{h}:refit"])
            report = mapping(arm["report"])
            candidates = report["candidates"]
            assert isinstance(candidates, list)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(mapping(candidates[0])["steps"], 3)
            self.assertEqual(arm["rate"], wide_rate)
            self.assertEqual(len(mapping(arm["harm_inventory"])), 1)
        refit = mapping(arms["M-wide→8:rank:refit"])
        self.assertIsNotNone(refit["step0_selection_loss"])
        self.assertTrue(math.isfinite(float(str(payload["r_aff"]))))
        self.assertEqual(payload["noop_sha256"], payload_of(read_record(self.root / "noop/result.json"))["noop_sha256"])
        retune = self.root / "retune-for-quantities"
        retuned = payload_of(run("retune", retune, 3000, "anchor", "--noop-root", str(self.root / "noop")))
        merged = {**saved, **{k: v for k, v in vectors(retune).items() if k != "N0"}}
        results = {**arms, **mapping(retuned["arms"])}
        anchor = AnchorResult(
            arms={
                name: ArmResult(
                    merged[name]["loss"],
                    merged[name]["correct"],
                    optional_bool(mapping(results[name])["certified"]),
                    optional_float(mapping(results[name])["rate"]),
                    optional_bool(mapping(results[name])["edge_of_grid"]),
                )
                for name in results
            },
            noop_loss=saved["N0"]["loss"],
            noop_correct=saved["N0"]["correct"],
            tcool_noop_loss=saved["T-cool"]["loss"],
            fold_residual=float(str(payload["r_aff"])),
            survivors={
                name: tuple(int(str(u)) for u in sequence(mapping(arms[name])["survivors"])) for name in CUT_ARMS
            },
            unfolded={name: saved[name]["unfolded_loss"] for name in CUT_ARMS},
            raw_candidates={},
        )
        computed = quantities(anchor, PILOT_ARMS)
        self.assertIsNotNone(computed["delta_acc"])
        self.assertIsNotNone(computed["D3"])

    def test_modules_arm_list_restricts_the_child(self) -> None:
        out = self.root / "modules-two"
        payload = payload_of(run("modules", out, 3000, "anchor", "--arms", "M-sq,M-aff", *self.roots()))
        self.assertEqual(set(mapping(payload["arms"])), {"M-sq", "M-aff"})
        self.assertEqual(set(vectors(out)), {"M-sq", "M-aff", "N0"})
        self.assertNotIn("M-exp", mapping(payload["arms"]))

    def test_retune_keeps_the_complement_and_r_all_reports_both_recalibration_sides(self) -> None:
        payload = payload_of(
            run("retune", self.root / "retune", 3000, "anchor", "--noop-root", str(self.root / "noop"))
        )
        arms = mapping(payload["arms"])
        self.assertEqual(set(arms), {"R-272s", "R-aff"})
        for name in ("R-272s", "R-aff"):
            arm = mapping(arms[name])
            self.assertIs(arm["complement_unchanged"], True)
            self.assertGreater(int(str(arm["parameter_steps"])), 0)
            self.assertGreater(int(str(arm["example_passes"])), 0)
            for label in mapping(arm["harm_inventory"]):
                self.assertRegex(label, LABEL)
        out = self.root / "retune-all"
        extended = payload_of(
            run("retune", out, 3000, "anchor", "--arms", "R-all", "--noop-root", str(self.root / "noop"))
        )
        everything = mapping(mapping(extended["arms"])["R-all"])
        self.assertIs(everything["complement_unchanged"], True)
        self.assertIn("pre_recalibration_development_loss", everything)
        self.assertIn("development_loss", everything)
        saved = vectors(out)
        self.assertEqual(saved["R-all"]["pre_recalibration_loss"].shape, (16,))

    def test_acceptance_matrix_holds_on_cpu(self) -> None:
        payload = payload_of(
            run("acceptance", self.root / "acceptance", 3000, "anchor", "--training-root", str(self.root / "train"))
        )
        for name in (
            "resume_bitwise",
            "checkpoint_bitwise",
            "static_step0_bitwise",
            "evaluation_consistent",
            "fold_residual_reproducible",
        ):
            self.assertIs(payload[name], True, name)
        self.assertTrue(math.isfinite(float(str(payload["fold_residual"]))))
        alone = payload_of(run("acceptance", self.root / "acceptance-alone"))
        self.assertIs(alone["training_root_used"], False)
        self.assertEqual(alone["resumed_epoch_state_sha256"], payload["resumed_epoch_state_sha256"])
        for name in ("resume_bitwise", "checkpoint_bitwise", "evaluation_consistent", "fold_residual_reproducible"):
            self.assertIs(alone[name], True, name)

    def test_timing_and_instrument_children(self) -> None:
        timing = payload_of(run("timing", self.root / "timing", 3000, "anchor", "--noop-root", str(self.root / "noop")))
        for name in (
            "module_grid_seconds",
            "front_rerun_grid_seconds",
            "front_rerun_seconds_per_1000_steps",
            "development_evaluation_seconds",
        ):
            self.assertGreaterEqual(float(str(timing[name])), 0.0, name)
        self.assertEqual((timing["front_rerun_steps"], timing["host"]), (2, "N0"))
        raw = payload_of(
            run("timing", self.root / "timing-raw", 3000, "anchor", "--training-root", str(self.root / "train"))
        )
        self.assertEqual(raw["host"], "raw")
        with (
            patch("sultai_screen.m0.worker.materiality", return_value={"seed": 3000}) as materiality,
            patch("sultai_screen.m0.worker.posthoc", return_value={"alpha": 0.1}) as posthoc,
        ):
            instrument = payload_of(
                run("instrument", self.root / "instrument", 3000, "anchor", "--training-root", str(self.root / "train"))
            )
        self.assertEqual(instrument, {"seed": 3000, "materiality": {"seed": 3000}, "posthoc": {"alpha": 0.1}})
        self.assertEqual(materiality.call_count, 1)
        self.assertEqual(posthoc.call_args.args[3], self.root / "instrument")

    def test_failing_action_publishes_an_m0_failure_receipt_and_reraises(self) -> None:
        out = self.root / "modules-bad"
        with self.assertRaisesRegex(PilotHold, "M-bogus"):
            run("modules", out, 3000, "anchor", "--arms", "M-sq,M-bogus", *self.roots())
        failure = read_record(out / "failure.json")
        self.assertEqual(mapping(failure["identity"])["protocol_version"], "m0-v4")
        self.assertEqual((failure["error_type"], failure["software_valid"]), ("PilotHold", False))
        self.assertFalse((out / "result.json").exists())
        bad = self.root / "bad-train"
        bad.mkdir()
        publish(bad / "result.json", {"identity": IDENTITY, "action": "train", "payload": {"seed": 3000}})
        with patched(), self.assertRaises(PilotHold):
            worker.load_training(bad, 3000, "anchor", REFERENCE_WIDTHS, CPU, frozen=True)


if __name__ == "__main__":
    unittest.main()
