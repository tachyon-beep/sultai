"""Check runner refusal and reporting without rerunning the full experiment."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from sultai import hybrid


class StubGenerator:
    def form(self, examples):
        raise AssertionError("fake lifecycle must not query a model")


class HybridRunnerTests(unittest.TestCase):
    def run_stub(self, *, formation_gate=True, lifecycle_gate=True, extra=None):
        formation = {"acceptance": {"formation": formation_gate}, "summary": {"gain": 1.0}}
        lifecycle = {"acceptance": {"removal": lifecycle_gate}, "summary": {"final_parameters": 272}}
        if extra is not None:
            formation["extra"] = extra
        with patch.object(hybrid, "run_formation", return_value=(formation, StubGenerator())), \
             patch.object(hybrid, "run_lifecycle", return_value=lifecycle):
            return hybrid.run()

    def test_failed_gate_is_retained_and_source_bytes_bound(self):
        result = self.run_stub(lifecycle_gate=False)
        self.assertFalse(result["passed"])
        self.assertFalse(result["acceptance"]["lifecycle.removal"])
        self.assertEqual(len(result["source_identity"]["aggregate_sha256"]), 64)
        self.assertIn("src/sultai/lifecycle.py", result["source_identity"]["files"])
        self.assertFalse(result["claims"]["half_parameter_goal_demonstrated"])

    def test_nonboolean_gate_or_nonfinite_payload_refused(self):
        with self.assertRaises(ValueError):
            self.run_stub(formation_gate=1)
        with self.assertRaises(ValueError):
            self.run_stub(extra=float("nan"))

    def test_summary_exit_status_reports_failed_demonstration(self):
        result = self.run_stub(lifecycle_gate=False)
        stream = io.StringIO()
        with patch.object(hybrid, "run", return_value=result), contextlib.redirect_stdout(stream):
            code = hybrid.main(["--summary"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(stream.getvalue())["passed"])
