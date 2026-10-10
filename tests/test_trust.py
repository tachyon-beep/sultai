"""Operation-boundary checks and positive/negative controls for the local policy."""

import ast
import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from report_fixture import reference_report, validated_splits
from sultai import formation
from sultai import report_types as reports
from sultai.contracts import numeric_matrix, numeric_vector
from sultai.repair import Adapter, Example, finite, fit
from sultai.trust import ContractViolation, FitUnavailable, InputDataError, t3_boundary

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_policy", ROOT / "scripts/check_policy.py")
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


class BoundaryTests(unittest.TestCase):
    def test_primitive_boundaries(self):
        cases = (
            (reports.object_map, ({"a": 1},), {"a": 1}, ([],), ({1: 2},)),
            (reports.required, ({"a": 1}, ("a",)), {"a": 1}, ([], ("a",)), ({}, ("a",))),
            (reports.number, (2,), 2.0, ("2",), (float("inf"),)),
            (reports.loss, (2,), 2.0, (True,), (-1.0,)),
            (reports.count, (2,), 2, (True,), (-1,)),
            (reports.boolean, (False,), False, (0,), (None,)),
            (reports.string, ("x",), "x", (1,), ("",)),
            (reports.nullable_string, (None,), None, (1,), ("",)),
            (
                reports.sequence,
                ((1, 2), reports.number),
                [1.0, 2.0],
                ({}, reports.number),
                ([1, float("nan")], reports.number),
            ),
            (
                reports.named,
                ({"a": 1}, reports.loss, ("a",)),
                {"a": 1.0},
                ([], reports.loss),
                ({"a": -1}, reports.loss),
            ),
            (reports.float_tuple, ([1, 2],), (1.0, 2.0), (None,), ([float("nan")],)),
            (reports.sha256_digest, ("a" * 64,), "a" * 64, (4,), ("A" * 64,)),
        )
        for parse, good, expected, bad_type, bad_value in cases:
            with self.subTest(boundary=parse.__name__):
                result = parse(*good)
                self.assertEqual(result, expected)
                if parse in (reports.number, reports.loss):
                    self.assertIs(type(result), float)
                with self.assertRaises(InputDataError):
                    parse(*bad_type)
                with self.assertRaises(InputDataError):
                    parse(*bad_value)
        self.assertIsNone(reports.nullable_string(None))
        for bad in (True, float("nan"), 10**1000):
            with self.assertRaises(InputDataError):
                reports.number(bad)

    def test_report_boundaries(self):
        report = reference_report()
        formation, lifecycle = report["formation"], report["lifecycle"]
        phase = formation["test"]
        lineage = phase["lineages"][0]
        arm = lifecycle["arms"]["formed_taper"]
        cases = (
            (reports.parse_CostLedger, lineage["costs"], "selection_queries", -1),
            (reports.parse_MethodMetrics, lineage["methods"]["paired"], "raw_test_mse", -1.0),
            (reports.parse_MethodSummary, phase["summary"]["paired"], "raw_mean_mse", -1.0),
            (reports.parse_LineageReport, lineage, "host_lineage", ""),
            (reports.parse_PhaseReport, phase, "lineages", []),
            (reports.parse_PartitionIdentity, formation["isolation"]["train"], "sample_count", -1),
            (reports.parse_SplitIdentity, formation["isolation"], "train", {}),
            (reports.parse_LineageCounts, formation["config"]["lineages"], "train", 31),
            (reports.parse_EvidenceModeCounts, formation["config"]["evidence_dimensions"], "paired", -1),
            (reports.parse_FormationConfig, formation["config"], "input_bound", float("inf")),
            (reports.parse_FormationCosts, formation["costs"], "units", ""),
            (reports.parse_FormationSummary, formation["summary"], "no_op_test_mean_mse", -1.0),
            (reports.parse_FormationGates, formation["acceptance"], "finite_metrics", 1),
            (reports.parse_FormationReport, formation, "assay", ""),
            (reports.parse_StageState, arm["stage_state"]["pre_add"], "completed_host_updates", -1),
            (reports.parse_ParameterCounts, arm["counts"], "peak", -1),
            (reports.parse_ArmReport, arm, "host_updates", -1),
            (reports.parse_LifecycleConfig, lifecycle["config"], "initial_steps", -1),
            (reports.parse_LifecycleCost, lifecycle["cost"], "host_gradient_steps", -1),
            (reports.parse_LifecycleSummary, lifecycle["summary"], "formed_immediate_add_mse", -1.0),
            (reports.parse_LifecycleGates, lifecycle["acceptance"], "common_host_update_count", 1),
            (reports.parse_LifecycleReport, lifecycle, "assay", ""),
            (reports.parse_AffineDiagnostic, report["diagnostics"]["affine_folding"], "adapter_bias", []),
            (reports.parse_SignedDiagnostic, report["diagnostics"]["signed_probe"], "passive_evidence_a", []),
            (reports.parse_Diagnostics, report["diagnostics"], "affine_folding", {}),
            (reports.parse_SourceIdentity, report["source_identity"], "aggregate_sha256", "a" * 64),
            (reports.parse_RuntimeIdentity, report["runtime"], "python", ""),
            (reports.parse_Claims, report["claims"], "scope", ""),
            (reports.parse_HybridReport, report, "protocol", ""),
        )
        for parse, good, field, invalid in cases:
            with self.subTest(boundary=parse.__name__):
                result = parse(copy.deepcopy(good))
                self.assertEqual(json.loads(json.dumps(result)), json.loads(json.dumps(good)))
                self.assertEqual(parse(result), result)
                self.assertIsNot(result, good)
                with self.assertRaises(InputDataError):
                    parse([])
                changed = copy.deepcopy(good)
                changed[field] = invalid
                with self.assertRaises(InputDataError):
                    parse(changed)
                changed = copy.deepcopy(good)
                del changed[field]
                with self.assertRaises(InputDataError):
                    parse(changed)

    def test_adapter_json_boundary(self):
        adapter = Adapter.zero()
        parsed = Adapter.from_json(adapter.to_json())
        self.assertEqual(parsed, adapter)
        self.assertIsInstance(parsed.bias, tuple)
        for bad in (
            [],
            {},
            {"weights": [], "bias": []},
            {"weights": adapter.weights, "bias": [True] * 16},
            {"weights": adapter.weights, "bias": [float("inf")] * 16},
        ):
            with self.subTest(value=bad), self.assertRaises(InputDataError):
                Adapter.from_json(json.dumps(bad))

    def test_formation_object_boundaries(self):
        ledger = formation.costs(selection_queries=2)
        self.assertEqual(formation.add_costs(ledger, ledger)["selection_queries"], 4)
        for bad in (None, {**ledger, "selection_queries": -1}, {}):
            with self.assertRaises(InputDataError):
                formation.add_costs(bad)
        with self.assertRaises(InputDataError):
            formation.add_costs()
        report = reference_report()["formation"]
        arguments = (report["development"], report["test"], report["healthy"])
        result = formation.formation_acceptance(*arguments, isolation=validated_splits())
        self.assertEqual(result, report["acceptance"])
        with self.assertRaises(InputDataError):
            formation.formation_acceptance(None, arguments[1], arguments[2], isolation=validated_splits())
        changed = copy.deepcopy(arguments[0])
        changed["lineages"][0]["host_lineage"] = "wrong-provenance"
        with self.assertRaises(InputDataError):
            formation.formation_acceptance(changed, arguments[1], arguments[2], isolation=validated_splits())
        with self.assertRaises(InputDataError):
            formation.formation_acceptance(*arguments)

    def test_numeric_boundaries(self):
        cases = (
            (finite, (2,), 2.0, ("2",), (float("nan"),)),
            (numeric_vector, ([1, 2], 2), (1.0, 2.0), (None, 2), ([1], 2)),
            (numeric_matrix, ([[1, 2]], 1, 2), ((1.0, 2.0),), (None, 1, 2), ([[1]], 1, 2)),
        )
        for parse, good, expected, bad_type, bad_value in cases:
            with self.subTest(boundary=parse.__name__):
                self.assertEqual(parse(*good), expected)
                with self.assertRaises(InputDataError):
                    parse(*bad_type)
                with self.assertRaises(InputDataError):
                    parse(*bad_value)

    def test_t2_fit_unavailable_and_precondition_fault_are_distinct(self):
        example = Example("trust-solver", (1.0,) * 16, (0.0,) * 16)
        with self.assertRaises(FitUnavailable):
            fit((example,), ridge=5e-324)
        with self.assertRaises(FitUnavailable):
            formation._regress(((1.0, 1.0),), ((0.0,),), ridge=5e-324)
        for operation in (lambda: fit((), ridge=1.0), lambda: formation._regress((), (), ridge=1.0)):
            with self.assertRaises(ContractViolation):
                operation()
        self.assertFalse(issubclass(FitUnavailable, ContractViolation))
        self.assertFalse(issubclass(FitUnavailable, InputDataError))

    def test_owned_fit_preconditions_fail_as_contract_faults(self):
        example = Example("owned-fit", (1.0,) * 16, (0.0,) * 16)
        cases = (
            lambda: fit((example,), ridge=float("nan")),
            lambda: formation._regress(((1.0,),), ((0.0,),), ridge=float("nan")),
            lambda: formation._regress(((1.0,), (1.0, 2.0)), ((0.0,), (0.0,))),
            lambda: formation._regress(((1.0,), (2.0,)), ((0.0,), (0.0, 1.0))),
            lambda: formation._regress(((float("nan"),),), ((0.0,),)),
        )
        for operation in cases:
            with self.subTest(operation=operation), self.assertRaises(ContractViolation):
                operation()

    def test_valid_finite_solver_overflow_is_recoverable(self):
        example = Example("finite-overflow", (1e308,) * 16, (-1e308,) * 16)
        with self.assertRaises(FitUnavailable):
            fit((example,))
        with self.assertRaises(FitUnavailable):
            formation._regress(((1e308,),), ((-1e308,),))

    def test_solver_does_not_reclassify_unrelated_trusted_faults(self):
        example = Example("trusted-fault", (1.0,) * 16, (0.0,) * 16)
        with patch("sultai.repair.math.tanh", side_effect=RuntimeError("trusted math fault")):
            with self.assertRaisesRegex(RuntimeError, "trusted math fault"):
                fit((example,))
        with patch("sultai.formation.numeric_matrix", side_effect=RuntimeError("trusted shape fault")):
            with self.assertRaisesRegex(RuntimeError, "trusted shape fault"):
                formation._regress(((1.0,),), ((0.0,),))

    def test_faults_propagate_without_boundary_wrapping(self):
        @t3_boundary(
            test="tests/test_trust.py::BoundaryTests.test_faults_propagate_without_boundary_wrapping",
            fingerprint="a" * 64,
        )
        def faulty(value: object) -> object:
            raise ContractViolation("broken trusted operation")

        with self.assertRaises(ContractViolation):
            faulty(1)
        self.assertFalse(issubclass(ContractViolation, InputDataError))
        self.assertTrue(issubclass(InputDataError, ValueError))


class PolicyControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for relative in POLICY.SOURCE_ROOTS:
            (self.root / relative).mkdir(parents=True)
            if relative != "src/sultai":
                (self.root / relative / "__init__.py").write_text(
                    '"""Declared root placeholder for the policy fixture."""\n'
                )
        (self.root / "tests").mkdir()
        self.test_source = (
            "class Controls:\n    def test_parse(self):\n        assert parse(2) == 2.0\n"
            "        with self.assertRaises(InputDataError):\n            parse(None)\n"
            "        with self.assertRaises(InputDataError):\n            parse(float('inf'))\n"
        )
        (self.root / "tests/test_boundary.py").write_text(self.test_source)
        self.fingerprint = POLICY.test_fingerprint(self.root, "tests/test_boundary.py::Controls.test_parse")
        self.source = f'@t3_boundary(test="tests/test_boundary.py::Controls.test_parse", fingerprint="{self.fingerprint}")\ndef parse(value: object) -> float:\n    return float(value)\n'
        self.path = self.root / "src/sultai/example.py"
        self.path.write_text(self.source)

    def test_positive_and_negative_real_ast_controls(self):
        self.assertEqual(POLICY.check_policy(self.root), [])
        mutations = (
            (self.source.split("\n", 1)[1], "missing T3"),
            (self.source.replace(self.fingerprint, "0" * 64), "stale test fingerprint"),
            (self.source.replace("Controls.test_parse", "Controls.test_absent"), "paired test"),
            (self.source + '\ndef hidden(data: dict[str, int]) -> int:\n    return data.get("missing", 0)\n', ".get"),
            (
                self.source
                + '\ndef alias(data: dict[str, int]) -> int:\n    getter = data.get\n    return getter("missing", 0)\n',
                ".get",
            ),
            (self.source.split("\n", 1)[1].replace("value: object", 'value: "object"'), "missing T3"),
            (self.source + "\nfrom typing import Any\n", "Any"),
            (self.source + "\nfrom typing import cast\n", "cast"),
            (self.source + "\nvalue = 1  # type: ignore\n", "ignore"),
        )
        for source, diagnostic in mutations:
            self.path.write_text(source)
            with self.subTest(diagnostic=diagnostic):
                self.assertTrue(any(diagnostic in item for item in POLICY.check_policy(self.root)))

    def test_t2_requires_documented_specific_domain_failures(self):
        valid = '\n@t2_operation(invariants="Positive finite ridge and complete data", failures=(FitUnavailable,))\ndef solve(value: int) -> int:\n    return value\n'
        self.path.write_text(self.source + valid)
        self.assertEqual(POLICY.check_policy(self.root), [])
        for invalid in (
            valid.replace("Positive finite ridge and complete data", ""),
            valid.replace("(FitUnavailable,)", "()"),
            valid.replace("FitUnavailable", "ValueError"),
        ):
            self.path.write_text(self.source + invalid)
            with self.subTest(marker=invalid):
                self.assertTrue(any("T2" in item for item in POLICY.check_policy(self.root)))

    def test_test_ast_drift_fails_and_formatting_is_stable(self):
        test = self.root / "tests/test_boundary.py"
        test.write_text("# comment\n" + self.test_source)
        self.assertEqual(POLICY.check_policy(self.root), [])
        test.write_text(self.test_source.replace("2.0", "0.0"))
        self.assertTrue(any("stale test fingerprint" in item for item in POLICY.check_policy(self.root)))

    def test_disposable_real_source_decorator_and_test_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("src", "tests"):
                shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            self.assertEqual(POLICY.check_policy(root), [])
            source = root / "src/sultai/report_types.py"
            original = source.read_text()
            tree = ast.parse(original)
            function = next(
                item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "object_map"
            )
            start = function.decorator_list[0].lineno - 1
            lines = original.splitlines(keepends=True)
            source.write_text("".join(lines[:start] + lines[function.lineno - 1 :]))
            self.assertTrue(any("object_map missing T3" in item for item in POLICY.check_policy(root)))
            source.write_text(original)
            test = root / "tests/test_trust.py"
            test.write_text(test.read_text().replace("(reports.count, (2,), 2,", "(reports.count, (2,), 3,"))
            self.assertTrue(any("stale test fingerprint" in item for item in POLICY.check_policy(root)))

    def test_empty_production_surface_is_refused(self):
        self.path.unlink()
        self.assertTrue(any("empty" in item for item in POLICY.check_policy(self.root)))

    def test_missing_declared_root_is_refused_by_name(self):
        self.assertEqual(POLICY.check_policy(self.root), [])
        for relative in POLICY.SOURCE_ROOTS:
            if relative == "src/sultai":
                continue
            shutil.rmtree(self.root / relative)
            errors = POLICY.check_policy(self.root)
            self.assertTrue(any("empty" in item and relative in item for item in errors), errors)
            break
        else:
            self.fail("fixture expects at least one declared root beyond src/sultai")

    def test_normalization_ignores_locations_but_keeps_assertions(self):
        left = ast.parse("def test_a():\n    assert 2 == 2\n").body[0]
        right = ast.parse("\n\ndef test_a():\n    assert (2 == 2)  # spacing\n").body[0]
        self.assertEqual(POLICY.ast_fingerprint(left), POLICY.ast_fingerprint(right))
