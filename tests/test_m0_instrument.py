from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sultai_screen.contracts import InstrumentHold, PilotHold, Record, mapping
from sultai_screen.fit import Adapter, zero_adapter
from sultai_screen.host import Cached, Features, Host, freeze
from sultai_screen.impair import Impairment
from sultai_screen.m0 import instrument
from sultai_screen.m0.instrument import (
    ALPHA,
    INVERSE_TOLERANCE,
    KNOWN_ANSWER_TOLERANCE,
    MATERIALITY_FLOOR,
    P1_LADDER,
    check_instrument,
    materiality,
    materiality_report,
    posthoc,
)

SRC = Path(__file__).resolve().parents[1] / "src"
SEEDS = (3000, 3001, 3002, 3003, 3004, 3005)


def fixture() -> tuple[Host, Cached]:
    # D2: seeded before the host is built, so its weights (and every drift below) are the same in every process.
    torch.manual_seed(0)
    host = Host()
    freeze(host)
    z = torch.rand(8, 16, 8, 8)
    data = Features(z, z, torch.arange(8))
    return host, Cached(data, data, data)


class FakeBest:
    def __init__(self) -> None:
        self.adapter = zero_adapter(torch.device("cpu"))


class FakeOracle:
    def __init__(self) -> None:
        self.best = FakeBest()

    def report(self) -> Record:
        return {"fake": True}


def good(seed: int, *, noop: float = 1.0, a1: float = 0.5, a3: float = 0.505, fraction: float | None = 0.9) -> Record:
    return {
        "seed": seed,
        "alpha": 0.1,
        "noop_loss": noop,
        "arm_losses": {"A1": a1, "A2": 0.6, "A3": a3, "B3": 0.55},
        "oracle_gain_fraction_of_A3": fraction,
    }


class InstrumentTests(unittest.TestCase):
    def test_materiality_is_p1_only_with_three_rungs(self) -> None:
        host, cache = fixture()
        record = materiality(host, cache, 3000)
        rungs = record["rungs"]
        assert isinstance(rungs, dict)
        self.assertEqual(list(rungs), ["P1"])
        gains = rungs["P1"]
        assert isinstance(gains, dict)
        self.assertEqual(list(gains), [str(m) for m in P1_LADDER])
        self.assertEqual(P1_LADDER, (0.1, 0.3, 0.03))

    def test_wrong_inverse_is_an_instrument_failure(self) -> None:
        # D1 (R21): materiality's exact-inverse clause at alpha = 0.1 is A1, an instrument clause: InstrumentHold on
        # every rung (the class itself, so the child publishes instrument_valid: false).
        host, cache = fixture()
        with patch.object(Impairment, "undo", lambda self, values: values + 0.1):
            with self.assertRaisesRegex(InstrumentHold, "exact inverse failed 1e-4 nats: P1/3000") as raised:
                materiality(host, cache, 3000)
        self.assertIs(type(raised.exception), InstrumentHold)
        # The first rung checked is alpha = 0.1 itself (P1_LADDER order): an inverse that fails only there holds.
        self.assertEqual(P1_LADDER[0], ALPHA)
        original = Impairment.undo
        calls: list[int] = []

        def first_rung_only(operator: Impairment, values: torch.Tensor) -> torch.Tensor:
            calls.append(len(calls))
            restored = original(operator, values)
            return restored + 0.1 if len(calls) == 1 else restored

        with patch.object(Impairment, "undo", first_rung_only), self.assertRaises(InstrumentHold) as raised:
            materiality(host, cache, 3000)
        self.assertIs(type(raised.exception), InstrumentHold)
        self.assertEqual(calls, [0])

    @staticmethod
    def clears(report: Record) -> dict[str, bool]:
        rungs = mapping(mapping(report["rungs"])["P1"])
        return {key: mapping(value)["clears"] is True for key, value in rungs.items()}

    def test_materiality_report_is_descriptive(self) -> None:
        def rec(seed: int, gain: float) -> Record:
            return {"seed": seed, "intact_loss": 0.5, "rungs": {"P1": {"0.1": gain, "0.3": 0.2, "0.03": 0.01}}}

        passing = materiality_report(tuple(rec(s, 0.06) for s in SEEDS))
        failing = materiality_report(tuple(rec(s, 0.06 if s != 3002 else 0.049) for s in SEEDS))
        self.assertEqual(self.clears(passing), {"0.1": True, "0.3": True, "0.03": False})
        self.assertEqual(self.clears(failing), {"0.1": False, "0.3": True, "0.03": False})
        self.assertNotIn("alpha", passing)
        self.assertNotIn("chosen", passing)
        with self.assertRaises(PilotHold):
            materiality_report(tuple(rec(3000, 0.06) for _ in SEEDS))

    def test_posthoc_is_fixed_alpha_with_four_arms(self) -> None:
        self.assertNotIn("alpha", inspect.signature(posthoc).parameters)
        host, cache = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(instrument, "oracle", lambda *a, **k: FakeOracle()):
                record = posthoc(host, cache, 3000, Path(tmp))
            self.assertTrue((Path(tmp) / "p1-oracle.pt").exists())
        self.assertEqual(record["alpha"], ALPHA)
        self.assertEqual(ALPHA, 0.1)
        arms = record["arm_losses"]
        assert isinstance(arms, dict)
        self.assertEqual(sorted(arms), ["A1", "A2", "A3", "B3"])
        self.assertEqual(record["rho_required"], 0.8)

    def test_posthoc_b3_fraction_of_a3_and_its_none_branch(self) -> None:
        # M10: fraction = (noop - B3) / (noop - A3); mean_loss order is intact, A1, noop, B3, A2, A3.
        host, cache = fixture()
        for losses, expected in (((0.5, 0.5, 1.0, 0.55, 0.6, 0.5), 0.9), ((0.5, 0.5, 0.5, 0.55, 0.6, 0.6), None)):
            with tempfile.TemporaryDirectory() as tmp, patch.object(instrument, "oracle", lambda *a, **k: FakeOracle()):
                with patch.object(instrument, "mean_loss", side_effect=losses):
                    record = posthoc(host, cache, 3000, Path(tmp))
            if expected is None:
                self.assertIsNone(record["oracle_gain_fraction_of_A3"])
            else:
                self.assertAlmostEqual(float(str(record["oracle_gain_fraction_of_A3"])), expected)
            self.assertEqual(record["a1_drift"], 0.0)

    def test_a1_drift_and_zero_module_identity_are_instrument_failures(self) -> None:
        # M1 (R21): these V0 clauses publish instrument-failed, so the M0 path raises InstrumentHold.
        host, cache = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(instrument, "oracle", lambda *a, **k: FakeOracle()):
            with patch.object(Impairment, "undo", lambda self, values: values + 0.1):
                with self.assertRaisesRegex(InstrumentHold, "exact inverse endpoint drift"):
                    posthoc(host, cache, 3000, Path(tmp))
        with patch.object(Adapter, "apply", lambda adapter, read, base: base + 1e-3):
            with self.assertRaisesRegex(InstrumentHold, "zero module is not bitwise identity"):
                materiality(host, cache, 3000)

    def test_a1_drift_clause_is_exactly_1e_4(self) -> None:
        # D2: mean_loss order in posthoc is intact, A1, noop, B3, A2, A3; drift = |A1 - intact|.
        self.assertEqual(INVERSE_TOLERANCE, 1e-4)
        host, cache = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(instrument, "oracle", lambda *a, **k: FakeOracle()):
            with patch.object(instrument, "mean_loss", side_effect=(0.0, 1e-4, 1.0, 0.55, 0.6, 0.5)):
                record = posthoc(host, cache, 3000, Path(tmp))
            self.assertEqual(record["a1_drift"], 1e-4)
            with patch.object(instrument, "mean_loss", side_effect=(0.0, 1e-4 + 1e-9)):
                with self.assertRaisesRegex(InstrumentHold, "exact inverse endpoint drift"):
                    posthoc(host, cache, 3000, Path(tmp))

    def test_instrument_literals_and_exact_boundaries(self) -> None:
        # D3: the frozen P1 literals, each at its exact boundary.
        self.assertEqual((MATERIALITY_FLOOR, KNOWN_ANSWER_TOLERANCE, ALPHA), (0.05, 0.01, 0.1))

        def rec(seed: int, gain: float) -> Record:
            return {"seed": seed, "intact_loss": 0.5, "rungs": {"P1": {"0.1": gain, "0.3": 0.2, "0.03": 0.01}}}

        self.assertTrue(self.clears(materiality_report(tuple(rec(s, 0.05) for s in SEEDS)))["0.1"])
        self.assertFalse(self.clears(materiality_report(tuple(rec(s, 0.0499) for s in SEEDS)))["0.1"])
        # noop - A1 = 0.05 exactly is material; A3 - A1 = 0.01 exactly passes; a fraction of exactly 0.8 passes.
        check_instrument(tuple(good(s, noop=0.05, a1=0.0, a3=0.01, fraction=0.8) for s in SEEDS))
        with self.assertRaisesRegex(InstrumentHold, "materiality"):
            check_instrument(tuple(good(s, noop=0.0499, a1=0.0) for s in SEEDS))
        with self.assertRaisesRegex(InstrumentHold, "known answer"):
            check_instrument(tuple(good(s, a1=0.0, a3=0.0101) for s in SEEDS))
        # check_instrument reads alpha = 0.1 records only.
        other = tuple({**good(s), "alpha": 0.3} if s == 3003 else good(s) for s in SEEDS)
        with self.assertRaisesRegex(PilotHold, "alpha"):
            check_instrument(other)

    def test_check_instrument(self) -> None:
        check_instrument(tuple(good(s) for s in SEEDS))
        low = tuple(good(s, fraction=0.79 if s == 3004 else 0.9) for s in SEEDS)
        with self.assertRaisesRegex(InstrumentHold, "B3 did not recover"):
            check_instrument(low)
        with self.assertRaisesRegex(InstrumentHold, "B3 did not recover"):
            check_instrument(tuple(good(s, fraction=None) for s in SEEDS))
        immaterial = tuple(good(s, noop=0.54 if s == 3001 else 1.0) for s in SEEDS)
        with self.assertRaisesRegex(InstrumentHold, "pilot materiality failed: P1 at alpha 0.1"):
            check_instrument(immaterial)
        with self.assertRaisesRegex(InstrumentHold, "supervised known answer failed"):
            check_instrument(tuple(good(s, a3=0.52) for s in SEEDS))
        with self.assertRaises(PilotHold):
            check_instrument(tuple(good(3000) for _ in SEEDS))

    def test_import_isolation(self) -> None:
        code = (
            "import sys; import sultai_screen.m0.instrument; "
            "banned = ('screen', 'pilot', 'former_bridge', 'alignment'); "
            "sys.exit(1 if any(f'sultai_screen.{m}' in sys.modules for m in banned) else 0)"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], env={"PYTHONPATH": str(SRC)}, check=False)
        self.assertEqual(result.returncode, 0)
