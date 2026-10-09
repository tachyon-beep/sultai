"""CLI refuses ambiguous execution and preserves valid negative evidence."""

import importlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@dataclass(frozen=True)
class NegativeReport:
    software_valid: bool = True
    diagnostic_specificity: bool = False
    training_ready: bool = False


class CorrectionCliTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("sultai.correction"), "correction CLI has not been implemented")
        return importlib.import_module("sultai.correction")

    def test_phase_is_required_before_any_execution(self):
        cli = self.module()
        with patch.object(cli, "run") as run, self.assertRaises(SystemExit) as stopped:
            cli.main([])
        self.assertEqual(stopped.exception.code, 2)
        run.assert_not_called()

    def test_dirty_source_refused_before_any_cohort(self):
        cli = self.module()
        with (
            patch.object(cli, "_git", return_value=" M src/sultai/correction.py"),
            patch.object(cli, "run_formation_controls") as formation,
        ):
            with self.assertRaisesRegex(cli.ContractViolation, "commit the reviewed"):
                cli.run("development")
            formation.assert_not_called()

    def test_source_change_invalidates_completed_assay(self):
        cli = self.module()

        def git(_root, *args):
            return "" if args[0] == "status" else "a" * 40

        generator = SimpleNamespace(form=lambda _examples: None)
        with (
            patch.object(cli, "_git", side_effect=git),
            patch.object(cli, "source_identity", side_effect=[{"identity": "before"}, {"identity": "after"}]),
            patch.object(cli, "run_formation_controls", return_value=(None, generator)),
            patch.object(cli, "run_trajectory", return_value=None),
            self.assertRaisesRegex(cli.ContractViolation, "source changed"),
        ):
            cli.run("development")

    def test_existing_output_refused_before_any_execution(self):
        cli = self.module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            output.write_text("preserved")
            with patch.object(cli, "run") as run, self.assertRaises(FileExistsError):
                cli.main(["--phase", "development", "--output", str(output)])
            run.assert_not_called()
            self.assertEqual(output.read_text(), "preserved")

    def test_valid_scientific_negative_is_complete_successful_execution(self):
        cli = self.module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            with patch.object(cli, "run", return_value=NegativeReport()) as run:
                status = cli.main(["--phase", "development", "--output", str(output)])
            run.assert_called_once_with("development")
            self.assertEqual(status, 0)
            result = json.loads(output.read_text())
            self.assertTrue(result["software_valid"])
            self.assertFalse(result["diagnostic_specificity"])
            self.assertFalse(result["training_ready"])
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_invalid_instrument_is_recorded_with_nonzero_exit(self):
        cli = self.module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            with patch.object(cli, "run", return_value=NegativeReport(software_valid=False)):
                status = cli.main(["--phase", "development", "--output", str(output)])
            self.assertEqual(status, 2)
            self.assertFalse(json.loads(output.read_text())["software_valid"])

    def test_execution_fault_propagates_without_publishing_evidence(self):
        cli = self.module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            with patch.object(cli, "run", side_effect=RuntimeError("trusted-code fault")):
                with self.assertRaisesRegex(RuntimeError, "trusted-code fault"):
                    cli.main(["--phase", "development", "--output", str(output)])
            self.assertFalse(output.exists())
