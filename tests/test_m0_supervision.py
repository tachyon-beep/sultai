"""CPU controls for the M0 supervisor arguments (ceiling, one process per card, worker module) and M0 cost projection."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sultai.trust import ContractViolation
from sultai_screen.contracts import PILOT_SECONDS, PilotHold, read_record
from sultai_screen.m0.costs import (
    ARM_UNITS,
    CORE_CASE_ARMS,
    EXTENDED_ORDER,
    EXTENDED_RESERVE,
    R_ALL_EVALUATIONS,
    STAGE_WORKLOADS,
    Workloads,
    admit_extended,
    core_over_ceiling,
    extended_cost,
    extended_limit,
    measured_confirmatory,
    per_anchor,
    project_confirmatory,
    project_pilot,
    stage_seconds,
    with_static,
)
from sultai_screen.m0.identity import M0_CONFIRMATORY_SECONDS
from sultai_screen.m0.stats import CONFIRMATORY_ARMS, KNOWN_ARMS
from sultai_screen.pilot import Job, Supervisor

SRC = Path(__file__).resolve().parents[1] / "src"
CHILD = (
    "import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
    "(p/'result.json').write_text(json.dumps({'software_valid':True,'identity':{'physical_gpu':sys.argv[2]},"
    "'payload':{}}))"
)


def workloads() -> Workloads:
    # A literal transcription of the §8 bookings: the §10 per-anchor decomposition is host 45, T-cool 25, N0 and
    # calibrations 30, four nonlinear module grids 4 x 60, M-aff 100, two M-wide→8 refits 2 x 60, M-small-B 120,
    # R-272s 80 and R-aff 250 = 1,010 s; S-static adds its host 50 and its module-grid refit 60 = 110 s. The front re-run
    # unit is per 1,000 optimiser steps: R-aff is 5 rates x 3,200 steps = 16 units, so 15.625 s per unit gives 250 s.
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


class SupervisionTests(unittest.TestCase):
    def _children(self, commands: list[list[str]]):
        """A Popen side effect that records each command and starts a real child writing a valid result."""
        real_popen = subprocess.Popen

        def start_child(command, **kwargs):
            commands.append(list(command))
            output = command[command.index("--output") + 1]
            gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            return real_popen([sys.executable, "-B", "-c", CHILD, output, gpu], **kwargs)

        return start_child

    def test_m0_ceiling_and_one_process_per_card(self) -> None:
        commands: list[list[str]] = []
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("sultai_screen.pilot.subprocess.Popen", side_effect=self._children(commands)),
        ):
            root = Path(temporary)
            supervisor = Supervisor(root, root, ceiling=14_400.0, per_card=1)
            self.assertEqual(supervisor.ceiling, 14_400.0)
            with self.assertRaises(PilotHold):
                supervisor.run((Job("a", "train", 3000, 0, 0), Job("b", "train", 3001, 0, 0)))
            self.assertEqual(commands, [])
            self.assertFalse((root / "a.process.json").exists())
            # One per card is a per-card limit, not a global cap of one: one job on each card is admitted.
            supervisor.run((Job("c", "train", 3000, 0, 0), Job("d", "train", 3001, 0, 1)))
            self.assertEqual(len(commands), 2)
            ledger = read_record(root / "ledger-000.json")
            self.assertEqual(ledger["ceiling_gpu_process_seconds"], 14400.0)
            self.assertEqual(len(ledger["entries"]), 2)
        # The opening charge is checked against the M0 ceiling, not the Gate 3 one.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Supervisor(root, root, 12_000.0, ceiling=14_400.0)
            with self.assertRaises(PilotHold):
                Supervisor(root, root, 14_400.0, ceiling=14_400.0)

    def test_gate3_defaults_keep_the_ledger_ceiling_and_bind_the_patched_constant_at_call_time(self) -> None:
        commands: list[list[str]] = []
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("sultai_screen.pilot.subprocess.Popen", side_effect=self._children(commands)),
        ):
            root = Path(temporary)
            supervisor = Supervisor(root, root)
            self.assertEqual(supervisor.ceiling, PILOT_SECONDS)
            self.assertEqual(supervisor.per_card, 3)
            supervisor.run(tuple(Job(f"g-{i}", "train", 3000, 0, i % 2) for i in range(6)))
            ceiling = read_record(root / "ledger-000.json")["ceiling_gpu_process_seconds"]
            self.assertEqual(ceiling, 10_800)
            # C11: the Gate 3 ledger bytes keep the int 10800, never a coerced 10800.0.
            self.assertIs(type(ceiling), int)
            self.assertIn('"ceiling_gpu_process_seconds":10800,', (root / "ledger-000.json").read_text())
            with self.assertRaisesRegex(PilotHold, "reviewed maximum three processes/card required"):
                supervisor.run(tuple(Job(f"x-{i}", "train", 3000, 0, 0) for i in range(4)))
        with tempfile.TemporaryDirectory() as temporary, patch("sultai_screen.pilot.PILOT_SECONDS", 10.15):
            root = Path(temporary)
            supervisor = Supervisor(root, root)
            self.assertEqual(supervisor.ceiling, 10.15)
            with self.assertRaises(PilotHold):
                Supervisor(root, root, 10.15)

    def test_child_command_is_unchanged_by_default_and_m0_appends_its_module_and_extra_arguments(self) -> None:
        commands: list[list[str]] = []
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("sultai_screen.pilot.subprocess.Popen", side_effect=self._children(commands)),
        ):
            root = Path(temporary)
            data = root / "data"
            training = root / "training"
            magnitudes = root / "magnitudes.json"
            Supervisor(root, data).run(
                (
                    Job("plain", "train", 3000, 0, 0),
                    Job("full", "oracle", 3001, 32, 1, training=training, magnitudes=magnitudes),
                )
            )
            # pilot.py:163-184 before this change, token for token.
            self.assertEqual(
                commands[0],
                [
                    "nice",
                    "-n",
                    "19",
                    sys.executable,
                    "-B",
                    "-m",
                    "sultai_screen.screen",
                    "train",
                    "--output",
                    str(root / "plain"),
                    "--data-root",
                    str(data),
                    "--seed",
                    "3000",
                    "--width",
                    "0",
                ],
            )
            self.assertEqual(
                commands[1],
                [
                    "nice",
                    "-n",
                    "19",
                    sys.executable,
                    "-B",
                    "-m",
                    "sultai_screen.screen",
                    "oracle",
                    "--output",
                    str(root / "full"),
                    "--data-root",
                    str(data),
                    "--seed",
                    "3001",
                    "--width",
                    "32",
                    "--training-root",
                    str(training),
                    "--magnitudes",
                    str(magnitudes),
                ],
            )
            m0_root = root / "m0"
            m0_root.mkdir()
            m0 = Supervisor(m0_root, data, ceiling=14_400.0, per_card=1, module="sultai_screen.m0.worker")
            m0.run(
                (
                    Job("frontier", "host", 3000, 0, 0, extra=("--lineage", "frontier-0.5")),
                    Job("grid", "modules", 3001, 0, 1, training=training, extra=("--lineage", "frontier-0.5")),
                )
            )
            self.assertEqual(
                commands[2],
                [
                    "nice",
                    "-n",
                    "19",
                    sys.executable,
                    "-B",
                    "-m",
                    "sultai_screen.m0.worker",
                    "host",
                    "--output",
                    str(m0_root / "frontier"),
                    "--data-root",
                    str(data),
                    "--seed",
                    "3000",
                    "--width",
                    "0",
                    "--lineage",
                    "frontier-0.5",
                ],
            )
            self.assertEqual(commands[3][-4:], ["--training-root", str(training), "--lineage", "frontier-0.5"])

    def test_arm_units_cover_every_known_arm_and_unknown_arms_raise(self) -> None:
        self.assertEqual(set(ARM_UNITS), set(KNOWN_ARMS))
        w = workloads()
        with self.assertRaises(ContractViolation):
            per_anchor(w, ("M-sq", "M-unknown"))
        with self.assertRaises(ContractViolation):
            per_anchor(w, ("M-sq", "M-sq"))
        with self.assertRaises(ContractViolation):
            extended_cost("F(2.0)", w)
        with self.assertRaises(ContractViolation):
            extended_cost("F(0.25)", dataclasses.replace(w, host={1.0: 45.0}))
        with self.assertRaises(ContractViolation):
            dataclasses.replace(w, grid={"module": 60.0})
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ContractViolation):
                dataclasses.replace(w, tcool=bad)
            with self.assertRaises(ContractViolation):
                dataclasses.replace(w, host={1.0: bad})

    def test_confirmatory_projection_from_measured_units(self) -> None:
        w = workloads()
        self.assertEqual(per_anchor(w, CONFIRMATORY_ARMS), 1_010.0)
        record = project_confirmatory(32, w, CONFIRMATORY_ARMS, 1500)
        self.assertEqual(record["projected_gpu_process_seconds"], 33_820.0)
        self.assertEqual(record["ceiling_seconds"], M0_CONFIRMATORY_SECONDS)
        self.assertIs(record["within_ceiling"], True)
        # Measured, never a literal: doubling the measured front re-run unit moves the projection by 32 x 250.
        self.assertEqual(per_anchor(w, ("R-aff",)) - per_anchor(w, ()), 16 * 15.625)
        slower = project_confirmatory(32, dataclasses.replace(w, front_rerun=31.25), CONFIRMATORY_ARMS, 1500)
        self.assertEqual(slower["projected_gpu_process_seconds"], 33_820.0 + 32 * 250.0)
        over = project_confirmatory(42, w, CONFIRMATORY_ARMS, 1500)
        self.assertEqual(over["projected_gpu_process_seconds"], 1500 + 42 * 1_010.0)
        self.assertIs(over["within_ceiling"], False)
        with self.assertRaises(ContractViolation):
            project_confirmatory(32, w, CONFIRMATORY_ARMS | {"R-all"}, 1500)
        with self.assertRaises(ContractViolation):
            project_confirmatory(31, w, CONFIRMATORY_ARMS, 1500)

    def test_static_projection_requires_fourteen_hours(self) -> None:
        w = workloads()
        arms = CONFIRMATORY_ARMS | {"S-static", "S-static:refit"}
        # The measured static unit: S-static host training 50 plus its module-grid refit 60 = 110 s per anchor.
        self.assertEqual(per_anchor(w, arms) - per_anchor(w, CONFIRMATORY_ARMS), 110.0)
        record = with_static(32, w, arms, 1500)
        self.assertEqual(record["projected_gpu_process_seconds"], 37_340.0)
        self.assertEqual(record["ceiling_seconds"], 50_400)
        self.assertIs(record["requires_14h"], True)
        self.assertIs(record["within_ceiling"], True)
        with self.assertRaises(ContractViolation):
            with_static(32, w, CONFIRMATORY_ARMS, 1500)
        with self.assertRaises(ContractViolation):
            with_static(32, w, CONFIRMATORY_ARMS | {"S-static"}, 1500)

    def test_pilot_projection_is_charged_plus_remaining(self) -> None:
        # Two remaining stages: the core (6,540 s at the bookings) and the F(0.25) frontier item, 1,000 s (6 x (host +
        # its noop and evaluate children, measured at the 0.5 width)).
        w = dataclasses.replace(workloads(), host={0.25: 1000.0 / 6 - 40.0, 1.0: 45.0})
        record = project_pilot(3000.0, ("core", "F(0.25)"), w)
        self.assertAlmostEqual(record["projected_gpu_process_seconds"], 3000.0 + 6540.0 + 1000.0, places=9)
        self.assertEqual(record["charged"], 3000.0)
        self.assertEqual(set(record["remaining_lines"]), {"core", "F(0.25)"})
        self.assertEqual(project_pilot(3000.0, (), w)["projected_gpu_process_seconds"], 3000.0)
        # B6: stages 1-3 are charged before stage 4 projects (R19); only the core is a projected stage.
        self.assertEqual(set(STAGE_WORKLOADS), {"core"})
        for charged_before in ("V0", "hosts"):
            with self.assertRaises(ContractViolation):
                project_pilot(3000.0, (charged_before,), w)
        with self.assertRaises(ContractViolation):
            project_pilot(3000.0, ("core", "core"), w)
        with self.assertRaises(ContractViolation):
            project_pilot(3000.0, ("medians",), w)

    def test_core_stage_is_the_section_8_core_table(self) -> None:
        # C10: §8's stage-5 core lines at the bookings (DEVELOPMENT_ANCHORS = 6): module-grid cases M-sq, M-sq-fresh,
        # M-sq | T-cool, M-wide, M-small and the S-static refit 6 x 6 x 60 = 2,160; M-aff 6 x 100 = 600; M-small-B
        # 6 x 120 = 720; the two M-wide→8 refit grids 6 x 2 x 60 = 720; R-272s 6 x 80 = 480; R-aff 6 x 250 = 1,500;
        # cuts, folds, leave-one-out, calibrations and evaluations 6 x 30 = 180; together 6,360 s. §5 adds the three
        # single 800-step M-wide→{16, 4, 2} refits (R17: the refit-single unit, 10 s): 6 x 3 x 10 = 180. Total 6,540 s.
        self.assertEqual(stage_seconds("core", workloads()), 6_540.0)
        # The §8 core cases: every pilot arm but T-cool (its resume is a stage-3 host child) plus the S-static refit.
        self.assertEqual(
            CORE_CASE_ARMS,
            frozenset(
                (
                    "M-sq",
                    "M-sq-fresh",
                    "M-sq|T-cool",
                    "M-wide",
                    "M-small",
                    "S-static:refit",
                    "M-aff",
                    "M-small-B",
                    "M-wide→8:rank:refit",
                    "M-wide→8:random:refit",
                    "R-272s",
                    "R-aff",
                    "M-wide→16:refit",
                    "M-wide→4:refit",
                    "M-wide→2:refit",
                    "M-wide→8:rank:sliced",
                    "M-wide→8:random:sliced",
                    "M-wide→16:sliced",
                    "M-wide→4:sliced",
                    "M-wide→2:sliced",
                    "M-wide→8:ablation",
                    "R-temp",
                    "R-cal",
                )
            ),
        )
        module_grid_cases = ("M-sq", "M-sq-fresh", "M-sq|T-cool", "M-wide", "M-small", "S-static:refit")
        self.assertTrue(all(ARM_UNITS[name] == ("grid:module",) for name in module_grid_cases))
        # Every other core case is costed inside the 30 s cuts-and-calibrations line (no unit of its own).
        free = ("M-wide→8:rank:sliced", "M-wide→8:random:sliced", "M-wide→8:ablation", "R-temp", "R-cal")
        self.assertTrue(all(ARM_UNITS[name] == () for name in free))

    def test_extended_admission_stops_at_the_first_miss(self) -> None:
        w = workloads()
        self.assertEqual(
            EXTENDED_ORDER, ("starved-pair", "R-all", "M-exp", "lottery", "cooled-20", "F(0.25)", "F(0.75)")
        )
        # The starved pair runs on the six 0.5 hosts already trained in stage 3: no host term (R9).
        self.assertEqual(extended_cost("starved-pair", w), 6 * (25 + 60 + 100 + 60 + 30))
        self.assertEqual(extended_cost("starved-pair", dataclasses.replace(w, host={1.0: 45.0})), 1_650.0)
        # R-all by its own terms (R23): 5 rates x 800 steps = 4 front re-run units, plus its 100 from-images
        # evaluations (5 rates x 20 checkpoints) at the measured development-evaluation kernel, on six anchors.
        self.assertEqual(R_ALL_EVALUATIONS, 100)
        self.assertEqual(extended_cost("R-all", w), 6 * (4 * 15.625 + 100 * 0.5))
        # Host items are whole children (R23): the train child plus the noop and evaluate children of its kind.
        self.assertEqual(extended_cost("F(0.25)", w), 6 * (20.0 + 40.0))
        self.assertEqual(extended_cost("cooled-20", w), 6 * (2 * 45.0 + 45.0))
        # The 5 % reserve (R23): admission fills to 0.95 x ceiling, never to the ceiling.
        self.assertEqual(EXTENDED_RESERVE, 0.05)
        self.assertEqual(extended_limit(14_400.0), 13_680.0)
        # 2,180 s under the limit: the starved pair (1,650) fits; R-all (675) misses; F(0.25) (360) would still fit but
        # is never reached, because admission stops at the first item that does not fit.
        self.assertEqual(admit_extended(3_000.0, 8_500.0, w, 14_400.0), ["starved-pair"])
        # The starved pair fits under the ceiling (14,050) but not under the reserve: nothing is admitted.
        self.assertEqual(admit_extended(3_000.0, 9_400.0, w, 14_400.0), [])
        self.assertEqual(admit_extended(0.0, 0.0, w, 1e9), list(EXTENDED_ORDER))

    def test_unmeasured_frontier_widths_are_costed_at_the_measured_half_width_host(self) -> None:
        # By stage 4 only the 0.5, 1.0 and 1.5 hosts have been measured; F(0.25) and F(0.75) run only after admission.
        w = dataclasses.replace(workloads(), host={0.5: 35.0, 1.0: 45.0, 1.5: 85.0})
        self.assertEqual(extended_cost("F(0.75)", w), 6 * (35.0 + 40.0))
        self.assertEqual(extended_cost("F(0.25)", w), 6 * (35.0 + 40.0))
        self.assertEqual(admit_extended(0.0, 0.0, w, 1e9), list(EXTENDED_ORDER))
        with self.assertRaises(ContractViolation):
            extended_cost("cooled-20", dataclasses.replace(w, noop_evaluate={0.5: 40.0}))

    def test_measured_confirmatory_projection_is_fixed_plus_n_times_the_measured_anchor(self) -> None:
        record = measured_confirmatory(33, 1_100.0, 700.0, 43_200)
        self.assertEqual(record["projected_gpu_process_seconds"], 700.0 + 33 * 1_100.0)
        self.assertIs(record["within_ceiling"], True)
        self.assertIs(measured_confirmatory(40, 1_100.0, 700.0, 43_200)["within_ceiling"], False)
        with self.assertRaises(ContractViolation):
            measured_confirmatory(31, 1_100.0, 700.0, 43_200)
        with self.assertRaises(ContractViolation):
            measured_confirmatory(32, 0.0, 700.0, 43_200)

    def test_core_over_ceiling_stops_before_v1(self) -> None:
        self.assertFalse(core_over_ceiling(5_000.0, 9_400.0, 14_400.0))
        self.assertTrue(core_over_ceiling(5_000.0, 9_400.5, 14_400.0))
        self.assertEqual(admit_extended(5_000.0, 9_400.5, workloads(), 14_400.0), [])

    def test_costs_import_isolation(self) -> None:
        code = (
            "import sys; import sultai_screen.m0.costs; "
            "banned = ('screen', 'pilot', 'former_bridge', 'alignment'); "
            "sys.exit(1 if any(f'sultai_screen.{m}' in sys.modules for m in banned) else 0)"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], env={"PYTHONPATH": str(SRC)}, check=False)
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
