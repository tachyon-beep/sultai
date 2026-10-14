"""CPU controls for the M0 worker: seed-admitting CLI, receipts, and every child action on a 64-image fixture."""

from __future__ import annotations

import dataclasses
import io
import math
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch
from torch import Tensor
from torch.nn import functional as F

from sultai_screen import fit as fit_module
from sultai_screen.contracts import (
    InstrumentHold,
    Json,
    OracleUnavailable,
    PilotHold,
    Record,
    encode,
    file_sha,
    mapping,
    number,
    publish,
    read_record,
    sequence,
)
from sultai_screen.data import Dataset, Seal
from sultai_screen.fit import Adapter, Candidate, Grid, fit_sgd, grid
from sultai_screen.host import REFERENCE_WIDTHS, Features, Host, make_host, recipe_optimizer, state_hash
from sultai_screen.m0 import receipts, worker
from sultai_screen.m0.calibrate import load_noop
from sultai_screen.m0.identity import protocol_identity
from sultai_screen.m0.retune import RetuneGrid, Subset, retune_grid
from sultai_screen.m0.static import StaticChoice, StaticHost, load_static, static_hash
from sultai_screen.m0.stats import CORE_ARMS, PILOT_ARMS, SURVIVAL_WIDTHS, AnchorResult, ArmResult, quantities

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


OVERFLOW = (1e300,)


def overflowing(kind: str) -> Callable[..., Grid]:
    """fit.grid with the rate list replaced by an overflowing rate for one kind of grid ("expanded" or "wide")."""

    def wrapped(
        host: Host,
        conditioning: Features,
        selection: Features,
        seed: int,
        *,
        rates: tuple[float, ...],
        checkpoints: tuple[int, ...],
        start: Adapter | None = None,
        hidden: int | None = None,
        nonlinear: bool = True,
        expanded: bool = False,
    ) -> Grid:
        hit = expanded if kind == "expanded" else hidden == 64
        return grid(
            host,
            conditioning,
            selection,
            seed,
            rates=OVERFLOW if hit else rates,
            checkpoints=checkpoints,
            start=start,
            hidden=hidden,
            nonlinear=nonlinear,
            expanded=expanded,
        )

    return wrapped


def overflowing_refit(
    host: Host,
    conditioning: Features,
    selection: Features,
    *,
    seed: int,
    rate: float,
    steps: int,
    minibatch: bool,
    start: Adapter | None = None,
    hidden: int | None = None,
) -> Candidate:
    return fit_sgd(
        host,
        conditioning,
        selection,
        seed=seed,
        rate=OVERFLOW[0],
        steps=steps,
        minibatch=minibatch,
        start=start,
        hidden=hidden,
    )


def overflowing_retune(
    host: Host,
    subset: Subset,
    *,
    data: Dataset,
    fit_indices: Tensor,
    selection_indices: Tensor,
    seed: int,
    rates: tuple[float, ...],
    checkpoints: tuple[int, ...],
    from_images: bool,
    certify: bool,
) -> RetuneGrid:
    return retune_grid(
        host,
        subset,
        data=data,
        fit_indices=fit_indices,
        selection_indices=selection_indices,
        seed=seed,
        rates=OVERFLOW if from_images else rates,
        checkpoints=checkpoints,
        from_images=from_images,
        certify=certify,
    )


def fixture() -> Dataset:
    rng = torch.Generator().manual_seed(16)
    images = torch.randint(0, 256, (64, 3, 32, 32), generator=rng, dtype=torch.uint8)
    labels = torch.randint(0, 10, (64,), generator=rng)
    ids = torch.arange(64)
    return Dataset(images, labels, ids, ids[0:16].clone(), ids[16:32].clone(), ids[32:48].clone())


@contextmanager
def patched(rates: tuple[tuple[float, ...], tuple[float, ...]] = ((0.1,), (0.1,))) -> Iterator[None]:
    """The CPU fixture; `rates` are the (MODULE_RATES, RETUNE_RATES) the child sees."""
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
            ("MODULE_RATES", rates[0]),
            ("RETUNE_RATES", rates[1]),
            ("RALL_CHECKPOINTS", (1, 2)),
        ):
            stack.enter_context(patch(f"sultai_screen.m0.worker.{target}", constant))
        stack.enter_context(patch("sultai_screen.fit.PLATEAU_WINDOW", 1))
        yield


def run(
    action: str,
    output: Path,
    seed: int = 3000,
    lineage: str = "anchor",
    *extra: str,
    rates: tuple[tuple[float, ...], tuple[float, ...]] = ((0.1,), (0.1,)),
) -> Record:
    scale = {"anchor": "1.0", "frontier-0.5": "0.5", "cooled-20": "1.0"}[lineage]
    argv = [action, "--output", str(output), "--data-root", "/nonexistent", "--seed", str(seed)]
    argv += ["--width", "0", "--lineage", lineage, "--scale", scale, *extra]
    with patched(rates):
        if worker.main(argv) != 0:
            raise AssertionError(f"{action} child failed")
    return read_record(output / "result.json")


def provenance_calls(action: str, output: Path, *extra: str) -> int:
    """Run an anchor-3000 child in process and count how often it computed the M0 provenance."""
    argv = [action, "--output", str(output), "--data-root", "/nonexistent", "--seed", "3000"]
    argv += ["--lineage", "anchor", "--scale", "1.0", *extra]
    with patched(), patch("sultai_screen.m0.worker.provenance", return_value=IDENTITY) as identity:
        if worker.main(argv) != 0:
            raise AssertionError(f"{action} child failed")
    return identity.call_count


def rewrite(root: Path, change: Callable[[Record], None]) -> None:
    """Edit a published receipt's payload in place (tamper fixtures only)."""
    record = read_record(root / "result.json")
    change(mapping(record["payload"]))
    (root / "result.json").write_text(encode(record) + "\n")


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

    def test_corner_classifies_the_selected_rate_and_step_on_their_own_axes(self) -> None:
        # Amendment 5 (A5.2 item 7): lowest/highest/interior by the grid's rates, first/last/interior by its
        # checkpoints, "single" on a one-value axis; the grid is read from the candidates, so M-small-B's doubled
        # axis classifies on 400/1,600/6,400.
        rates, steps = (1.0, 0.3, 0.1), (200, 800, 3200)
        self.assertEqual(worker.corner(0.1, 200, rates, steps), {"rate": "lowest", "step": "first"})
        self.assertEqual(worker.corner(1.0, 3200, rates, steps), {"rate": "highest", "step": "last"})
        self.assertEqual(worker.corner(0.3, 800, rates, steps), {"rate": "interior", "step": "interior"})
        self.assertEqual(worker.corner(0.3, 400, rates, (400, 1600, 6400)), {"rate": "interior", "step": "first"})
        self.assertEqual(worker.corner(0.1, 2, (0.1,), (1, 2)), {"rate": "single", "step": "last"})
        self.assertEqual(worker.corner(0.1, 1, (0.1, 0.1), (1,)), {"rate": "single", "step": "single"})

    def test_attempt4_hashes_are_the_published_receipts(self) -> None:
        self.assertEqual(worker.DEV_ARTIFACT, REPO / "docs/results/gate3-split-2026-10-11/dev-indices-int32-le.bin")
        self.assertTrue(worker.DEV_ARTIFACT.is_file())
        self.assertEqual((worker.RALL_CHECKPOINTS[0], worker.RALL_CHECKPOINTS[-1]), (40, 800))
        self.assertFalse(hasattr(worker, "SURVIVAL_REFIT_STEPS"))
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
        self.assertEqual(mapping(record["identity"])["protocol_version"], "m0-v4.3")
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
        # M8: the host's state at construction, before the probe and before any step.
        self.assertEqual(payload["initial_state_sha256"], state_hash(make_host(3000, 0, CPU)))
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
        # Amendment 5 (A5.2 item 2): T-cool's no-op rule publishes the two losses that decided it and the fit indices.
        self.assertEqual(tcool["noop_choice"] == "R-bn", tcool["rbn_selection_loss"] < tcool["raw_selection_loss"])
        self.assertEqual(tcool["fit_indices_sha256"], worker.tensor_sha(SEEN))
        static = payload_of(read_record(self.root / "static/result.json"))
        self.assertEqual(static["module_sha256"], file_sha(self.root / "static/static-module.pt"))
        self.assertEqual(static["checkpoint_sha256"], file_sha(self.root / "static/host.pt"))
        self.assertIn("static_sha256", static)
        # Amendment 4 (A4.1): the static child applies the no-op rule to the complete model and publishes the chosen one.
        self.assertIn(static["noop_choice"], ("R-bn", "raw"))
        chosen = (
            static["recalibrated_static_sha256"] if static["noop_choice"] == "R-bn" else static["raw_static_sha256"]
        )
        self.assertEqual(static["static_sha256"], chosen)
        self.assertNotEqual(static["raw_static_sha256"], static["recalibrated_static_sha256"])
        self.assertGreater(static["noop_seconds"], 0.0)
        self.assertEqual(static["noop_choice"] == "R-bn", static["rbn_selection_loss"] < static["raw_selection_loss"])
        # Amendment 5 (A5.2 item 2): the static host's no-op application names the fit indices it consumed.
        self.assertEqual(static["fit_indices_sha256"], worker.tensor_sha(SEEN))
        published = load_static(self.root / "static", 3000, REFERENCE_WIDTHS, CPU, str(static["static_sha256"]))
        self.assertEqual(static_hash(published), static["static_sha256"])
        self.assertEqual(static["host_sha256"], state_hash(published.host))
        # M8: S-static's inner host at step zero is the anchor's make_host state.
        train = payload_of(read_record(self.root / "train/result.json"))
        self.assertEqual(static["initial_state_sha256"], train["initial_state_sha256"])

    def test_static_action_publishes_the_raw_model_when_the_rule_keeps_it(self) -> None:
        # Amendment 4 (A4.1): the branch the 64-image fixture does not take (it picks R-bn): forced raw, the files are
        # the raw model, its hash is the recorded one, and the module bytes are those of the R-bn run.
        def keep_raw(raw: StaticHost, recalibrated: StaticHost, data: object) -> StaticChoice:
            return StaticChoice(raw, "raw", 1.0, 2.0)

        output = self.root / "static-raw"
        with patch("sultai_screen.m0.worker.static_no_op", keep_raw):
            run("static", output)
        static = payload_of(read_record(output / "result.json"))
        self.assertEqual(static["noop_choice"], "raw")
        self.assertEqual(static["static_sha256"], static["raw_static_sha256"])
        self.assertEqual((static["raw_selection_loss"], static["rbn_selection_loss"]), (1.0, 2.0))
        self.assertEqual(static["checkpoint_sha256"], file_sha(output / "host.pt"))
        self.assertEqual(static["module_sha256"], file_sha(output / "static-module.pt"))
        published = load_static(output, 3000, REFERENCE_WIDTHS, CPU, str(static["static_sha256"]))
        self.assertEqual(static_hash(published), static["raw_static_sha256"])
        rbn_run = payload_of(read_record(self.root / "static/result.json"))
        self.assertEqual(static["module_sha256"], rbn_run["module_sha256"])
        self.assertEqual(static["raw_static_sha256"], rbn_run["raw_static_sha256"])
        self.assertEqual(static["recalibrated_static_sha256"], rbn_run["recalibrated_static_sha256"])
        self.assertNotEqual(static["checkpoint_sha256"], rbn_run["checkpoint_sha256"])

    def test_noop_publishes_the_choice_and_never_a_development_number(self) -> None:
        payload = payload_of(read_record(self.root / "noop/result.json"))
        self.assertIn(payload["noop_choice"], ("R-bn", "raw"))
        noop = load_noop(self.root / "noop", 3000, REFERENCE_WIDTHS, str(payload["noop_sha256"]), CPU)
        self.assertEqual(state_hash(noop), payload["noop_sha256"])
        chosen = payload["recalibrated_sha256"] if payload["noop_choice"] == "R-bn" else payload["raw_sha256"]
        self.assertEqual(chosen, payload["noop_sha256"])
        # Amendment 5 (A5.2 item 2): the fit-index tensor the recalibration consumed, by hash, beside both losses.
        self.assertEqual(payload["fit_indices_sha256"], worker.tensor_sha(SEEN))
        self.assertEqual(
            payload["noop_choice"] == "R-bn", payload["rbn_selection_loss"] < payload["raw_selection_loss"]
        )
        bias = payload["class_bias"]
        assert isinstance(bias, list)
        self.assertEqual(len(bias), 11)
        temperature = optional_float(payload["temperature"])
        assert temperature is not None
        self.assertGreater(temperature, 0)
        for name in ("raw", "rbn", "rtemp", "rcal"):
            self.assertTrue(math.isfinite(float(str(payload[f"{name}_selection_loss"]))))
        self.assertFalse([key for key in keys(payload) if "development" in key])
        # R25: a zero module of every form is bitwise identity with N0 on development (flags only, no number).
        flags = mapping(payload["zero_module_identity"])
        self.assertEqual(set(flags), {"affine", "nonlinear", "expanded", "hidden-64", "hidden-8"})
        self.assertTrue(all(value is True for value in flags.values()), flags)
        data = fixture()
        host = load_noop(self.root / "noop", 3000, REFERENCE_WIDTHS, str(payload["noop_sha256"]), CPU)
        development = worker.features_from_images(host, data, data.development)
        with patch.object(Adapter, "apply", lambda adapter, read, base: base + 1e-3):
            shifted = worker.zero_module_identity(host, development)
        self.assertEqual(set(shifted), set(flags))
        self.assertTrue(all(value is False for value in shifted.values()), shifted)

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
        wide_steps = mapping(arms["M-wide"])["steps"]
        for h in (16, 4, 2):
            arm = mapping(arms[f"M-wide→{h}:refit"])
            report = mapping(arm["report"])
            candidates = report["candidates"]
            assert isinstance(candidates, list)
            self.assertEqual(len(candidates), 1)
            # Amendment 5 (A5.1): the refit carries M-wide's selected (rate, step) pair, never a fixed step count.
            self.assertEqual(mapping(candidates[0])["steps"], wide_steps)
            self.assertEqual(arm["steps"], wide_steps)
            self.assertEqual(arm["rate"], wide_rate)
            self.assertEqual(len(mapping(arm["harm_inventory"])), 1)
        refit = mapping(arms["M-wide→8:rank:refit"])
        self.assertIsNotNone(refit["step0_selection_loss"])
        # M5: every grid or single-run arm books its final-checkpoint candidates (whole runs), §4 units.
        # Amendment 5 (A5.2 item 7): every grid arm names its selected (rate, step) corner; a single run has none.
        for name in ("M-sq", "M-small-B", "M-wide", "M-wide→8:rank:refit", "M-exp"):
            corner = mapping(mapping(arms[name])["corner"])
            self.assertEqual(corner["rate"], "single", name)
            axis = (2, 4) if name == "M-small-B" else (1, 2)
            self.assertEqual(corner["step"], "first" if mapping(arms[name])["steps"] == axis[0] else "last", name)
        self.assertIsNone(mapping(arms["M-wide→4:refit"])["corner"])
        self.assertIsNone(mapping(arms["M-wide→4:sliced"])["corner"])
        for name in ("M-sq", "M-small-B", "M-wide", "M-wide→8:rank:refit", "M-wide→4:refit", "M-exp"):
            arm = mapping(arms[name])
            candidates = [mapping(c) for c in sequence(mapping(arm["report"])["candidates"])]
            last = max(int(str(c["steps"])) for c in candidates)
            final = [c for c in candidates if c["steps"] == last]
            booking = mapping(arm["booking"])
            for key in ("example_passes", "parameter_steps", "wall_seconds"):
                self.assertAlmostEqual(number(booking[key]), sum(number(c[key]) for c in final), msg=f"{name} {key}")
            self.assertGreater(number(booking["parameter_steps"]), 0)
        self.assertNotIn("booking", mapping(arms["M-wide→8:rank:sliced"]))
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

    def test_real_children_pass_through_the_supervisor_readers_and_stats(self) -> None:
        """I5: real noop, tcool, static, modules and retune receipts through receipts.source, anchor and quantities."""
        root = self.root / "end-to-end"
        root.mkdir()
        run("modules", root / "modules-3000", 3000, "anchor", *self.roots())
        run("retune", root / "retune-3000", 3000, "anchor", "--noop-root", str(self.root / "noop"))
        certified = str(IDENTITY["source_sha256"])
        sources = tuple(
            receipts.source(root, f"{kind}-3000", kind, 3000, "anchor", certified) for kind in ("modules", "retune")
        )
        joined = receipts.anchor(3000, sources, PILOT_ARMS)
        computed = quantities(joined.result, PILOT_ARMS)
        self.assertIs(joined.report["printed_counts"], True)
        self.assertIs(joined.report["complement_unchanged"], True)
        labelled = mapping(joined.report["harm_inventory"])
        self.assertTrue(labelled)
        for name, labels in labelled.items():
            for label in mapping(labels):
                self.assertIsInstance(receipts.candidate_rate(label), float, f"{name} {label}")
        ran = set(mapping(payload_of(read_record(root / "modules-3000/result.json"))["arms"]))
        self.assertEqual(set(joined.result.arms), ran | RETUNE_ONLY)
        for key in ("D1", "D1prime", "D2", "D3", "D5", "delta_acc"):
            self.assertIsNotNone(computed[key], key)
        # M4: a receipt of another implementation source is refused.
        with self.assertRaisesRegex(PilotHold, "implementation source"):
            receipts.source(root, "modules-3000", "modules", 3000, "anchor", "0" * 64)

    def test_lottery_reads_the_rank8_survivors_from_the_core_modules_receipt(self) -> None:
        # R31: the extended lottery child never re-fits M-wide; it reads the survivors of the anchor's core receipt
        # (its M-wide→8:rank:sliced record) and retrains those units from their initial weights on the module grid.
        noop = ("--noop-root", str(self.root / "noop"))
        core = self.root / "lottery-core"
        run("modules", core, 3000, "anchor", "--arms", "M-wide,M-wide→8:rank:sliced", *noop)
        sliced = mapping(mapping(payload_of(read_record(core / "result.json"))["arms"])["M-wide→8:rank:sliced"])
        planted = [8, 9, 10, 11, 12, 13, 14, 15] if sliced["survivors"] != list(range(8, 16)) else list(range(8))
        moved = self.root / "lottery-core-moved"
        shutil.copytree(core, moved)
        rewrite(moved, lambda p: mapping(mapping(p["arms"])["M-wide→8:rank:sliced"]).update({"survivors": planted}))
        widths: list[int | None] = []

        def spy(
            host: Host,
            conditioning: Features,
            selection: Features,
            seed: int,
            *,
            rates: tuple[float, ...],
            checkpoints: tuple[int, ...],
            start: Adapter | None = None,
            hidden: int | None = None,
            expanded: bool = False,
        ) -> Grid:
            widths.append(hidden)
            return grid(
                host, conditioning, selection, seed, rates=rates, checkpoints=checkpoints, start=start, hidden=hidden,
                expanded=expanded,
            )  # fmt: skip

        with patch("sultai_screen.m0.worker.grid", side_effect=spy):
            payload = payload_of(
                run("modules", self.root / "lottery-1", 3000, "anchor", "--arms", "lottery", *noop,
                    "--modules-root", str(moved))
            )  # fmt: skip
        record = mapping(mapping(payload["arms"])["lottery"])
        self.assertEqual(record["status"], "available")
        self.assertEqual(record["survivors"], planted)
        self.assertEqual(record["survivors_source"], "M-wide→8:rank:sliced")
        self.assertEqual(record["survivors_receipt_vectors_sha256"], file_sha(moved / "vectors.pt"))
        # One lottery grid at h = 8 and no M-wide (h = 64) grid.
        self.assertEqual(widths, [8])
        bare = self.root / "lottery-core-bare"
        shutil.copytree(core, bare)
        rewrite(bare, lambda p: mapping(p["arms"]).pop("M-wide→8:rank:sliced"))
        with self.assertRaisesRegex(PilotHold, "survivors"):
            run(
                "modules",
                self.root / "lottery-2",
                3000,
                "anchor",
                "--arms",
                "lottery",
                *noop,
                "--modules-root",
                str(bare),
            )

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
            # R32: R-272s and R-aff carry the Table B certification (a bool) and its paired SE.
            self.assertIsInstance(arm["certified"], bool, name)
            report = mapping(arm["report"])
            self.assertEqual(report["plateau_certified"], arm["certified"])
            self.assertEqual(report["plateau_evidence"], "both finite schedule comparators available")
            self.assertIsInstance(report["selection_paired_bootstrap_se"], float)
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
        self.assertIsNone(everything["certified"])
        self.assertEqual(mapping(everything["report"])["plateau_evidence"], "R-all is never plateau-certified (§4)")
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
        ):
            self.assertIs(payload[name], True, name)
        for removed in ("fold_residual_reproducible", "fold_residual", "fold_residual_repeat", "fold_rate"):
            self.assertNotIn(removed, payload)
        alone = payload_of(run("acceptance", self.root / "acceptance-alone"))
        self.assertIs(alone["training_root_used"], False)
        self.assertEqual(alone["resumed_epoch_state_sha256"], payload["resumed_epoch_state_sha256"])
        for name in ("resume_bitwise", "checkpoint_bitwise", "evaluation_consistent"):
            self.assertIs(alone[name], True, name)
        cooled = self.root / "acceptance-cooled-host"
        run("train", cooled, 6018, "cooled-20", "--epochs", "20")
        with self.assertRaisesRegex(PilotHold, "constant-0.05"):
            run("acceptance", self.root / "acceptance-cooled", 6018, "cooled-20", "--training-root", str(cooled))

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
        for always_true in ("module_grid_finite", "front_rerun_finite"):
            self.assertNotIn(always_true, timing)
        with (
            patch("sultai_screen.m0.worker.materiality", return_value={"seed": 3000}) as materiality,
            patch("sultai_screen.m0.worker.posthoc", return_value={"alpha": 0.1}) as posthoc,
        ):
            instrument = payload_of(
                run("instrument", self.root / "instrument", 3000, "anchor", "--training-root", str(self.root / "train"))
            )
        self.assertEqual(
            instrument,
            {
                "seed": 3000,
                "lineage": "anchor",
                "scale": 1.0,
                "widths": list(REFERENCE_WIDTHS),
                "materiality": {"seed": 3000},
                "posthoc": {"alpha": 0.1},
            },
        )
        self.assertEqual(materiality.call_count, 1)
        self.assertEqual(posthoc.call_args.args[3], self.root / "instrument")

    def test_schedule_flags_set_every_axis_and_refuse_a_non_increasing_list(self) -> None:
        base = ["modules", "--output", "/x", "--data-root", "/y", "--seed", "3000", "--lineage", "anchor"]
        base += ["--scale", "1.0"]
        for extra in (
            ["--checkpoints", "2,2"],
            ["--checkpoints", "3,1"],
            ["--checkpoints-b", "0,4"],
            ["--checkpoints", "a"],
        ):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                worker.main([*base, *extra])
            self.assertEqual(raised.exception.code, 2)
        with patch("sultai_screen.m0.worker.worker", return_value=0) as child:
            worker.main([*base, "--checkpoints", "1,2", "--checkpoints-b", "2,4"])
        request = child.call_args.args[0]
        self.assertEqual((request.checkpoints, request.checkpoints_b), ((1, 2), (2, 4)))
        with patch("sultai_screen.m0.worker.worker", return_value=0) as child:
            worker.main(base)
        request = child.call_args.args[0]
        self.assertEqual((request.checkpoints, request.checkpoints_b), ((200, 800, 3200), (400, 1600, 6400)))
        # The fixture patches the defaults to (1, 2) and (2, 4); the flags must override them.
        axes = ("--checkpoints", "1,3", "--checkpoints-b", "2,5")
        noop = ("--noop-root", str(self.root / "noop"))
        payload = payload_of(
            run("modules", self.root / "modules-axis", 3000, "anchor", "--arms", "M-sq,M-small-B", *axes, *noop)
        )
        self.assertEqual((payload["checkpoints"], payload["checkpoints_b"]), ([1, 3], [2, 5]))

        def steps(arm: Json) -> set[int]:
            return {int(str(mapping(c)["steps"])) for c in sequence(mapping(mapping(arm)["report"])["candidates"])}

        arms = mapping(payload["arms"])
        self.assertEqual(steps(arms["M-sq"]), {1, 3})
        self.assertEqual(steps(arms["M-small-B"]), {2, 5})
        retuned = payload_of(run("retune", self.root / "retune-axis", 3000, "anchor", *axes, *noop))
        self.assertEqual(steps(mapping(retuned["arms"])["R-272s"]), {1, 3})
        self.assertEqual(steps(mapping(retuned["arms"])["R-aff"]), {1, 3})
        timing = payload_of(run("timing", self.root / "timing-axis", 3000, "anchor", *axes, *noop))
        self.assertEqual((timing["front_rerun_steps"], timing["checkpoints"]), (3, [1, 3]))

    def test_unavailable_arms_are_recorded_and_the_child_continues(self) -> None:
        out = self.root / "modules-unavailable"
        arms = "M-sq,M-exp,M-wide→4:sliced,M-wide→4:refit"
        with (
            patch("sultai_screen.m0.worker.grid", side_effect=overflowing("expanded")),
            patch("sultai_screen.m0.worker.fit_sgd", side_effect=overflowing_refit),
        ):
            payload = payload_of(run("modules", out, 3000, "anchor", "--arms", arms, *self.roots()))
        records = mapping(payload["arms"])
        self.assertEqual(set(sequence(payload["unavailable_arms"])), {"M-exp", "M-wide→4:refit"})
        for name in ("M-exp", "M-wide→4:refit"):
            record = mapping(records[name])
            self.assertEqual(record["status"], "unavailable")
            candidates = sequence(record["candidates"])
            self.assertTrue(candidates)
            self.assertTrue(all(mapping(c)["finite"] is False for c in candidates))
            self.assertNotIn("development_loss", record)
        self.assertEqual(len(sequence(mapping(records["M-exp"])["candidates"])), 2)
        for name in ("M-exp", "M-wide→4:refit"):
            self.assertEqual(mapping(records[name])["cause"], name)
        for name in ("M-sq", "M-wide→4:sliced"):
            self.assertEqual(mapping(records[name])["status"], "available")
        self.assertEqual(set(vectors(out)), {"N0", "M-sq", "M-wide→4:sliced"})
        wide = self.root / "modules-wide-unavailable"
        with patch("sultai_screen.m0.worker.grid", side_effect=overflowing("wide")):
            payload = payload_of(
                run(
                    "modules", wide, 3000, "anchor", "--arms", "M-sq,M-wide,M-wide→8:rank:sliced,lottery", *self.roots()
                )
            )
        self.assertEqual(sequence(payload["unavailable_arms"]), ["M-wide", "M-wide→8:rank:sliced", "lottery"])
        for name in ("M-wide", "M-wide→8:rank:sliced", "lottery"):
            self.assertEqual(mapping(mapping(payload["arms"])[name])["cause"], "M-wide")
        self.assertEqual(mapping(mapping(payload["arms"])["M-sq"])["status"], "available")
        noop = ("--noop-root", str(self.root / "noop"))
        with patch("sultai_screen.m0.worker.retune_grid", side_effect=overflowing_retune):
            retuned = payload_of(run("retune", self.root / "retune-unavailable", 3000, "anchor", *noop))
        self.assertEqual(sequence(retuned["unavailable_arms"]), ["R-aff"])
        self.assertEqual(mapping(mapping(retuned["arms"])["R-aff"])["status"], "unavailable")
        self.assertIs(mapping(mapping(retuned["arms"])["R-272s"])["complement_unchanged"], True)

    def test_fold_check_recomputes_the_modules_residual_on_n0(self) -> None:
        modules = self.root / "fold-modules"
        noop = ("--noop-root", str(self.root / "noop"))
        receipt = payload_of(run("modules", modules, 3000, "anchor", "--arms", "M-aff", *noop))
        arm = mapping(mapping(receipt["arms"])["M-aff"])
        check = payload_of(
            run("fold-check", self.root / "fold-check", 3000, "anchor", *noop, "--modules-root", str(modules))
        )
        self.assertIs(check["fold_residual_matches_receipt"], True)
        self.assertLessEqual(abs(number(check["recomputed_r_aff"]) - number(check["receipt_r_aff"])), 1e-6)
        self.assertEqual(check["receipt_r_aff"], receipt["r_aff"])
        self.assertEqual((check["selected_rate"], check["selected_steps"]), (arm["rate"], arm["steps"]))
        self.assertEqual(check["modules_vectors_sha256"], receipt["vectors_sha256"])
        self.assertEqual((check["lineage"], check["widths"]), ("anchor", list(REFERENCE_WIDTHS)))

        tampered = self.root / "fold-tampered"
        shutil.copytree(modules, tampered)

        def shift(payload: Record) -> None:
            payload["r_aff"] = number(payload["r_aff"]) + 0.01
            mapping(mapping(payload["arms"])["M-aff"])["r_aff"] = payload["r_aff"]

        rewrite(tampered, shift)
        moved = payload_of(
            run("fold-check", self.root / "fold-moved", 3000, "anchor", *noop, "--modules-root", str(tampered))
        )
        self.assertIs(moved["fold_residual_matches_receipt"], False)

        unavailable = self.root / "fold-unavailable"
        shutil.copytree(modules, unavailable)

        def drop(payload: Record) -> None:
            payload["r_aff"] = None
            mapping(payload["arms"])["M-aff"] = {"status": "unavailable", "cause": "M-aff", "candidates": []}

        rewrite(unavailable, drop)
        with self.assertRaisesRegex(PilotHold, "M-aff"):
            run("fold-check", self.root / "fold-none", 3000, "anchor", *noop, "--modules-root", str(unavailable))
        with self.assertRaisesRegex(PilotHold, "modules"):
            run("fold-check", self.root / "fold-missing", 3000, "anchor", *noop)
        foreign = self.root / "fold-foreign"
        shutil.copytree(modules, foreign)
        rewrite(foreign, lambda payload: payload.update({"noop_sha256": "0" * 64}))
        with self.assertRaisesRegex(PilotHold, "N0"):
            run("fold-check", self.root / "fold-foreign-check", 3000, "anchor", *noop, "--modules-root", str(foreign))

    def test_a_child_computes_its_provenance_once(self) -> None:
        self.assertEqual(provenance_calls("modules", self.root / "modules-once", *self.roots()), 1)
        training = ("--training-root", str(self.root / "train"))
        self.assertEqual(provenance_calls("noop", self.root / "noop-once", *training), 1)

    def test_failing_action_publishes_an_m0_failure_receipt_and_reraises(self) -> None:
        out = self.root / "modules-bad"
        with self.assertRaisesRegex(PilotHold, "M-bogus"):
            run("modules", out, 3000, "anchor", "--arms", "M-sq,M-bogus", *self.roots())
        failure = read_record(out / "failure.json")
        self.assertEqual(mapping(failure["identity"])["protocol_version"], "m0-v4.3")
        self.assertEqual((failure["error_type"], failure["software_valid"]), ("PilotHold", False))
        self.assertFalse((out / "result.json").exists())
        bad = self.root / "bad-train"
        bad.mkdir()
        publish(bad / "result.json", {"identity": IDENTITY, "action": "train", "payload": {"seed": 3000}})
        with patched(), self.assertRaises(PilotHold):
            worker.load_training(bad, 3000, "anchor", REFERENCE_WIDTHS, CPU, frozen=True)

    def test_train_and_replay_children_are_the_gate3_training_recipe(self) -> None:
        # C2 (attempt-4 F1): the worker's inline recipe (kept to save the momentum buffers) is bitwise host.train_host
        # on the fixture, and its probe is attempt 4's Gate 3 training_worker probe.
        from sultai_screen import screen  # Gate 3, in this test process only (the worker never imports it)
        from sultai_screen.host import train_host

        host = make_host(3000, 0, CPU)
        hashes, stream, _ = train_host(host, fixture(), 3000)
        with tempfile.TemporaryDirectory() as temporary:
            gate3 = screen.training_worker(fixture(), 3000, 0, Path(temporary))
        for action in ("train", "replay"):
            payload = payload_of(run(action, self.root / f"recipe-{action}"))
            self.assertEqual(payload["final_state_sha256"], state_hash(host), action)
            self.assertEqual(payload["epoch_state_sha256"], list(hashes), action)
            self.assertEqual(payload["loss_stream_sha256"], worker.tensor_sha(stream), action)
            self.assertEqual(payload["forward_backward"], gate3["forward_backward"], action)
            for key in ("final_state_sha256", "epoch_state_sha256", "loss_stream_sha256", "checkpoint_sha256"):
                self.assertEqual(payload[key], gate3[key], f"{action} {key}")

    def test_replay_refuses_any_lineage_but_the_attempt4_anchors(self) -> None:
        # C3 (attempt-4 F2): the guard names itself, never a KeyError on the pin table.
        request = worker.Request(
            action="replay",
            output=self.root / "replay-frontier",
            data_root=Path("/nonexistent"),
            seed=6000,
            lineage="frontier-0.5",
            widths=HALF,
            epochs=10,
            checkpoints=(1, 2),
            checkpoints_b=(2, 4),
            arms=None,
            training_root=None,
            noop_root=None,
            tcool_root=None,
            static_root=None,
        )
        with self.assertRaisesRegex(PilotHold, "attempt-4 anchors"):
            worker.replay_action(request, fixture(), CPU)

    def test_the_worker_shell_seals_pins_and_never_reuses_an_output(self) -> None:
        # F1 (I-1): the seal is installed before the card is configured, the output created and the data loaded;
        # configure asks for CUDA; a second child into an existing output raises before any receipt.
        output = self.root / "shell"
        order: list[tuple[str, bool]] = []
        data = fixture()

        def seal() -> Seal:
            order.append(("seal", output.exists()))
            return Seal()

        def configure(*, cuda: bool) -> torch.device:
            order.append((f"configure cuda={cuda}", output.exists()))
            return CPU

        def load(root: Path, artifact: Path, device: torch.device) -> Dataset:
            order.append(("load", output.exists()))
            return data

        argv = ["train", "--output", str(output), "--data-root", "/nonexistent", "--seed", "3000"]
        argv += ["--lineage", "anchor", "--scale", "1.0"]
        with (
            patched(),
            patch("sultai_screen.m0.worker.install_seal", side_effect=seal),
            patch("sultai_screen.m0.worker.configure", side_effect=configure),
            patch("sultai_screen.m0.worker.load_dataset", side_effect=load),
        ):
            self.assertEqual(worker.main(argv), 0)
        self.assertEqual(order, [("seal", False), ("configure cuda=True", False), ("load", True)])
        reused = self.root / "shell-reused"
        reused.mkdir()
        with patched(), self.assertRaises(FileExistsError):
            worker.main([*argv[:2], str(reused), *argv[3:]])
        self.assertEqual(list(reused.iterdir()), [])

    def test_instrument_class_failures_publish_software_valid_receipts(self) -> None:
        # F1 (I-1) and D1: InstrumentHold and OracleUnavailable are instrument failures (software valid), so the
        # Supervisor reads instrument_valid: false; OracleUnavailable carries its candidates.
        def raising(error: Exception) -> Callable[[worker.Request, Dataset, torch.device], Record]:
            def action(request: worker.Request, data: Dataset, device: torch.device) -> Record:
                raise error

            return action

        candidates: tuple[Record, ...] = ({"rate": 1.0, "steps": 2, "finite": False},)
        cases = (
            (InstrumentHold("held"), None),
            (OracleUnavailable("no finite candidate", candidates=candidates), [dict(c) for c in candidates]),
        )
        for index, (error, unavailable) in enumerate(cases):
            out = self.root / f"instrument-class-{index}"
            with patch.dict(worker.ACTIONS, {"noop": raising(error)}), self.assertRaises(type(error)):
                run("noop", out, 3000, "anchor", "--training-root", str(self.root / "train"))
            failure = read_record(out / "failure.json")
            self.assertIs(failure["software_valid"], True)
            self.assertIs(failure["instrument_valid"], False)
            self.assertEqual(failure["unavailable_candidates"], unavailable)
            self.assertFalse((out / "result.json").exists())
        # D1: materiality's exact-inverse failure through the real instrument child (mean_loss order: intact, no-op,
        # inverse) publishes instrument_valid: false.
        out = self.root / "instrument-inverse"
        with (
            patch("sultai_screen.m0.instrument.mean_loss", side_effect=(1.0, 1.5, 1.2)),
            self.assertRaisesRegex(InstrumentHold, "exact inverse failed"),
        ):
            run("instrument", out, 3000, "anchor", "--training-root", str(self.root / "train"))
        failure = read_record(out / "failure.json")
        self.assertEqual((failure["software_valid"], failure["instrument_valid"]), (True, False))
        self.assertEqual(failure["error_type"], "InstrumentHold")

    def test_every_sha_binder_refuses_a_file_changed_or_missing_after_publication(self) -> None:
        # F2 (I-2): vectors.pt (receipts.source), host.pt and optimizer.pt (receipts.bound, worker.load_training).
        certified = str(IDENTITY["source_sha256"])
        root = self.root / "binders"
        root.mkdir()
        run("modules", root / "modules-3000", 3000, "anchor", "--arms", "M-sq", "--noop-root", str(self.root / "noop"))
        receipts.source(root, "modules-3000", "modules", 3000, "anchor", certified)
        for damage in ("rewrite", "remove"):
            tampered = root / f"modules-{damage}"
            shutil.copytree(root / "modules-3000", tampered)
            if damage == "rewrite":
                flip(tampered / "vectors.pt")
            else:
                (tampered / "vectors.pt").unlink()
            with self.assertRaisesRegex(PilotHold, "unbound or changed after publication"):
                receipts.source(root, tampered.name, "modules", 3000, "anchor", certified)
        payload = payload_of(read_record(self.root / "train/result.json"))
        receipts.bound(self.root / "train", payload, receipts.TRAINING_FILES)
        for name in ("host.pt", "optimizer.pt"):
            for damage in ("rewrite", "remove"):
                tampered = self.root / f"train-{damage}-{name}"
                shutil.copytree(self.root / "train", tampered)
                if damage == "rewrite":
                    flip(tampered / name)
                else:
                    (tampered / name).unlink()
                with self.assertRaisesRegex(PilotHold, "unbound or changed after publication"):
                    receipts.bound(tampered, payload, receipts.TRAINING_FILES)
                with patched(), self.assertRaisesRegex(PilotHold, "changed after publication or is unbound"):
                    worker.load_training(tampered, 3000, "anchor", REFERENCE_WIDTHS, CPU, frozen=True)

    def test_each_arm_fits_on_its_own_frozen_rate_set(self) -> None:
        # F3 (I-3): distinct module and re-tune grids; M-aff and every re-tune arm on the re-tune set, every other grid
        # arm on the module set; each h = 16/4/2 survival refit at M-wide's SELECTED rate, which the patched selection
        # makes the second module rate, not MODULE_RATES[0].
        module_rates, retune_rates = (0.1, 0.05), (0.2, 0.1, 0.03)

        def second_rate_wide(
            host: Host,
            conditioning: Features,
            selection: Features,
            seed: int,
            *,
            rates: tuple[float, ...],
            checkpoints: tuple[int, ...],
            start: Adapter | None = None,
            hidden: int | None = None,
            nonlinear: bool = True,
            expanded: bool = False,
        ) -> Grid:
            result = grid(
                host, conditioning, selection, seed, rates=rates, checkpoints=checkpoints, start=start, hidden=hidden,
                nonlinear=nonlinear, expanded=expanded,
            )  # fmt: skip
            if hidden != 64 or start is not None:
                return result
            chosen = [c for c in result.candidates if c.rate == rates[1] and c.selection is not None]
            return dataclasses.replace(result, best=min(chosen, key=lambda c: number(c.selection)))

        out = self.root / "modules-rates"
        with patch("sultai_screen.m0.worker.grid", side_effect=second_rate_wide):
            payload = payload_of(run("modules", out, 3000, "anchor", *self.roots(), rates=(module_rates, retune_rates)))
        arms = mapping(payload["arms"])
        noop = ("--noop-root", str(self.root / "noop"))
        retuned = mapping(
            payload_of(
                run("retune", self.root / "retune-rates", 3000, "anchor", "--arms", "R-272s,R-aff,R-all", *noop,
                    rates=(module_rates, retune_rates))
            )["arms"]
        )  # fmt: skip

        def rates_of(record: Json) -> set[float]:
            return {number(mapping(c)["rate"]) for c in sequence(mapping(mapping(record)["report"])["candidates"])}

        for name in ("R-272s", "R-aff", "R-all"):
            self.assertEqual(rates_of(retuned[name]), set(retune_rates), name)
        self.assertEqual(rates_of(arms["M-aff"]), set(retune_rates))
        grids = ("M-sq", "M-sq-fresh", "M-wide", "M-wide→8:rank:refit", "M-wide→8:random:refit", "M-small",
                 "M-small-B", "M-exp", "lottery", "M-sq|T-cool", "S-static:refit")  # fmt: skip
        for name in grids:
            self.assertEqual(rates_of(arms[name]), set(module_rates), name)
        wide = number(mapping(arms["M-wide"])["rate"])
        self.assertEqual(wide, module_rates[1])
        self.assertNotEqual(wide, module_rates[0])
        for h in (16, 4, 2):
            self.assertEqual(number(mapping(arms[f"M-wide→{h}:refit"])["rate"]), wide, h)

    def test_each_acceptance_flag_turns_false_when_its_comparison_breaks(self) -> None:
        # F4 (I-4): a negative control per flag (i)-(iv).
        training = ("--training-root", str(self.root / "train"))

        def accept(name: str) -> Record:
            return payload_of(run("acceptance", self.root / f"acceptance-{name}", 3000, "anchor", *training))

        def no_momentum(host: Host, path: Path, device: torch.device) -> torch.optim.SGD:
            return recipe_optimizer(host.parameters())

        def shifted_rate(*args: Host | Features, **kwargs: float | int | bool) -> Candidate:
            return fit_sgd(*args, **{**kwargs, "rate": number(kwargs["rate"]) * 0.5})  # (ii): another rate

        class Offset(StaticHost):
            def __init__(self, host: Host) -> None:
                super().__init__(host)
                with torch.no_grad():
                    self.b.fill_(0.5)

        real = worker.losses
        controls: tuple[tuple[str, str, object], ...] = (
            ("resume_bitwise", "_optimizer", no_momentum),
            ("checkpoint_bitwise", "fit_sgd", shifted_rate),
            ("static_step0_bitwise", "StaticHost", Offset),
            ("evaluation_consistent", "losses", lambda *a, **k: real(*a, **k) + 1e-3),
        )
        for flag, target, replacement in controls:
            with self.subTest(flag=flag), patch(f"sultai_screen.m0.worker.{target}", replacement):
                record = accept(flag)
                self.assertIs(record[flag], False, flag)
                for other in ("resume_bitwise", "checkpoint_bitwise", "static_step0_bitwise", "evaluation_consistent"):
                    if other != flag:
                        self.assertIs(record[other], True, f"{flag} broke {other}")

    def test_acceptance_accuracy_is_checked_against_an_unpadded_forward(self) -> None:
        # F6 (M-3): an evaluator that drifts coherently (its logits, the accuracy read from them and the losses all
        # agree with each other) passed the old self-comparison; (iv) now compares the padded evaluator's accuracy with
        # host.classify on the whole development site, an independent unpadded forward, and fails it.
        real = fit_module.logits

        def drifted(host: Host, data: Features, adapter: Adapter | None) -> Tensor:
            # Every example "correct": the drifted evaluator scores 100 % on the random-label fixture.
            return real(host, data, adapter) + 100.0 * F.one_hot(data.labels, 10).to(torch.float32)

        def drifted_losses(host: Host, data: Features, adapter: Adapter | None, *, check_finite: bool = True) -> Tensor:
            return F.cross_entropy(drifted(host, data, adapter), data.labels, reduction="none")

        with (
            patch("sultai_screen.fit.logits", drifted),
            patch("sultai_screen.m0.worker.logits", drifted),
            patch("sultai_screen.m0.worker.losses", drifted_losses),
        ):
            record = payload_of(
                run("acceptance", self.root / "acceptance-argmax", 3000, "anchor", "--training-root",
                    str(self.root / "train"))
            )  # fmt: skip
        self.assertEqual(number(record["loss_max_difference"]), 0.0)
        self.assertIs(record["evaluation_consistent"], False)

    def test_the_accuracy_reference_runs_in_eval_chunk_batches(self) -> None:
        # R34b/R34c: the independent reference forward sees the padded evaluator's batch shape (EVAL_CHUNK rows, the
        # last chunk zero-padded), never one whole-split forward, so a GPU batch-size bit difference cannot falsely
        # stop stage 1; the 0/1 comparison stays exact.
        host = make_host(3000, 0, CPU)
        host.eval()
        base = torch.randn((300, 16, 8, 8), generator=torch.Generator().manual_seed(9))
        data = Features(base, base, torch.arange(300) % 10)
        sizes: list[int] = []
        real = Host.classify

        def recording(self: Host, site: Tensor) -> Tensor:
            sizes.append(len(site))
            return real(self, site)

        with patch.object(Host, "classify", recording):
            reference = worker.chunked_accuracy(host, data)
        # R34c: every batch is exactly EVAL_CHUNK rows (the last zero-padded), the padding discarded.
        self.assertEqual(fit_module.EVAL_CHUNK, 128)
        self.assertEqual(sizes, [128, 128, 128])
        self.assertEqual(len(reference), 300)
        with torch.no_grad():
            last = torch.cat((base[256:], torch.zeros((84, 16, 8, 8))))
            logits = torch.cat([real(host, base[0:128]), real(host, base[128:256]), real(host, last)[:44]])
        self.assertTrue(torch.equal(reference, (logits.argmax(1) == data.labels).float()))

    def test_survival_widths_are_the_frozen_set(self) -> None:
        # G3: the production survival widths, 16/8/4/2 (§5): three single refits plus the M-wide→8 grid arms.
        self.assertEqual(worker.SURVIVAL_CUTS, (16, 4, 2))
        self.assertEqual((worker.SMALL, worker.WIDE), (8, 64))
        self.assertEqual(SURVIVAL_WIDTHS, (16, 8, 4, 2))
        self.assertEqual(tuple(sorted((*worker.SURVIVAL_CUTS, worker.SMALL), reverse=True)), SURVIVAL_WIDTHS)


def flip(path: Path) -> None:
    """Rewrite one byte of a published file in place."""
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


if __name__ == "__main__":
    unittest.main()
