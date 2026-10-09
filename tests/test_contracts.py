"""Reject invalid required state without inventing successful/no-op evidence."""

import contextlib
import copy
import io
import json
import math
import unittest
from unittest.mock import patch

from report_fixture import reference_report, validated_splits
from sultai import formation, hybrid, lifecycle
from sultai.repair import Adapter, Example, mse
from sultai.report_types import METHODS, SOURCE_FILES, parse_FormationReport, parse_HybridReport, parse_LifecycleReport


class StubGenerator:
    def form(self, examples):
        raise AssertionError("report boundary tests must not query a former")


def run_reports(formation_report, lifecycle_report):
    with (
        patch.object(hybrid, "run_formation", return_value=(formation_report, StubGenerator())),
        patch.object(hybrid, "run_lifecycle", return_value=lifecycle_report),
    ):
        return hybrid.run()


class StateContracts(unittest.TestCase):
    def test_affine_complete_shapes_and_finite_scalars_required(self):
        weights, bias = Adapter.zero().weights, Adapter.zero().bias
        for bad_weights, bad_bias in (
            ((), ()),
            (weights[:-1], bias),
            ((*weights, weights[0]), bias),
            (((0.0,) * 15,) * 16, bias),
            (((0.0,) * 17,) * 16, bias),
            (weights, bias[:-1]),
            (weights, (*bias, 0.0)),
            (((math.nan,) * 16,) * 16, bias),
            (weights, (math.inf,) * 16),
            (((True,) * 16,) * 16, bias),
            (weights, ("0",) * 16),
        ):
            with self.subTest(weights=bad_weights, bias=bad_bias), self.assertRaises(ValueError):
                formation.AffineControl(bad_weights, bad_bias)
        control = formation.AffineControl(weights, bias)
        self.assertEqual(control.apply((1.0,) * 16), (1.0,) * 16)
        for h in ((1.0,) * 15, (1.0,) * 17, (True,) * 16, (math.inf,) * 16):
            with self.subTest(h=h), self.assertRaises(ValueError):
                control.apply(h)

    def test_metrics_refuse_missing_extra_or_nonfinite_predictions(self):
        example = Example("metric", (0.0,) * 16, (1.0,) * 16)

        class MalformedReadout:
            def __init__(self, prediction):
                self.prediction = prediction

            def apply(self, h):
                return self.prediction

        for prediction in ((), (0.0,) * 15, (0.0,) * 17, (math.nan,) * 16, (True,) * 16):
            with self.subTest(prediction=prediction):
                with self.assertRaises(ValueError):
                    mse(MalformedReadout(prediction), (example,))
                with self.assertRaises(ValueError):
                    lifecycle._prediction_mse((prediction,), (example,))
                with self.assertRaises(ValueError):
                    formation.admit(MalformedReadout(prediction), (example,))
        self.assertEqual(mse(MalformedReadout((0.0,) * 16), (example,)), 1.0)
        with self.assertRaises(ValueError):
            formation.admit(Adapter.zero(), (example,), baseline_loss=-1.0)

    def test_generator_missing_extra_ragged_and_nonfinite_state_refused(self):
        for mode, rows in (("coarse", 7), ("paired", 11), ("paired_probes", 11)):
            for weights in (
                (),
                ((0.0,) * 4,) * (rows - 1),
                ((0.0,) * 4,) * (rows + 1),
                ((0.0,) * 3,) * rows,
                ((0.0,) * 5,) * rows,
                ((math.nan,) * 4,) * rows,
                ((True,) * 4,) * rows,
                ((0.0,) * 4,) * rows + ((math.inf,) * 4,),
            ):
                with self.subTest(mode=mode, weights=weights), self.assertRaises(ValueError):
                    formation.FrozenGenerator(mode, weights)
            mutable = [[0.0] * 4 for _ in range(rows)]
            former = formation.FrozenGenerator(mode, mutable)
            mutable[0][0] = math.nan
            self.assertIsInstance(former.weights, tuple)
            self.assertEqual(former.form(formation.make_episode(3000).conditioning), Adapter.zero())
        with self.assertRaises(ValueError):
            formation.FrozenGenerator("unknown", ((0.0,) * 4,) * 11)

    def test_teacher_and_regression_state_require_complete_shapes(self):
        for evidence, adapters in (((), ()), (((0.0,) * 10,), ()), (((0.0,) * 9,), (Adapter.zero(),))):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                formation.TeacherBank(evidence, adapters)
        for rows, targets in (
            ((), ()),
            (((0.0,),), ()),
            (((0.0,), (0.0, 1.0)), ((0.0,), (0.0,))),
            (((math.nan,),), ((0.0,),)),
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                formation._regress(rows, targets)

    def test_adapter_wire_refuses_invalid_object_arrays_and_exact_scalar_types(self):
        valid = json.loads(Adapter.zero().to_json())
        for value in (
            None,
            [],
            {},
            {**valid, "extra": 1},
            {**valid, "weights": None},
            {**valid, "bias": [True] * 16},
            {**valid, "bias": ["0"] * 16},
            {**valid, "bias": [math.nan] * 16},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Adapter.from_json(json.dumps(value))

    def test_sparse_cost_factory_and_complete_aggregate_are_distinct(self):
        empty = formation.costs()
        self.assertEqual(set(empty), set(formation.COST_KEYS))
        self.assertEqual(set(empty.values()), {0})
        self.assertEqual(formation.add_costs(empty, formation.costs(selection_queries=3))["selection_queries"], 3)
        for value in (True, False, 1.5, -1, math.nan, math.inf, "3", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                formation.costs(selection_queries=value)
        for value in (
            {},
            {"selection_queres": 99},
            {**empty, "unknown": 1},
            {key: val for key, val in empty.items() if key != "selection_queries"},
            {**empty, "selection_queries": -1},
            {**empty, "selection_queries": True},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                formation.add_costs(value)
        with self.assertRaises(ValueError):
            formation.add_costs()

    def test_explicit_absence_releases_frozen_and_trainable_weights(self):
        import weakref

        for trainable in (False, True):
            model = lifecycle._Model()
            model.add(Adapter.zero(), trainable=trainable)
            reference = weakref.ref(model.temporary)
            before = model.serialized()
            model.remove()
            self.assertIsNone(model.temporary)
            self.assertIsNone(reference())
            self.assertEqual(model.parameters, 272)
            self.assertNotIn("adapter", model.serialized())
            self.assertIn("adapter", before)


class EvidenceContracts(unittest.TestCase):
    def setUp(self):
        self.report = reference_report()

    def test_every_required_runner_gate_must_be_present_and_boolean(self):
        for phase in ("formation", "lifecycle"):
            for key in self.report[phase]["acceptance"]:
                for bad in ("missing", 1, None):
                    changed = copy.deepcopy(self.report)
                    if bad == "missing":
                        del changed[phase]["acceptance"][key]
                    else:
                        changed[phase]["acceptance"][key] = bad
                    with self.subTest(phase=phase, key=key, bad=bad), self.assertRaises(ValueError):
                        run_reports(changed["formation"], changed["lifecycle"])
        changed = copy.deepcopy(self.report)
        changed["lifecycle"]["acceptance"]["unknown"] = True
        with self.assertRaises(ValueError):
            run_reports(changed["formation"], changed["lifecycle"])

    def test_every_phase_method_summary_and_lineage_is_required(self):
        for phase in ("development", "test", "healthy"):
            for mutation in ("phase", "empty", "lineage", "summary"):
                changed = copy.deepcopy(self.report["formation"])
                if mutation == "phase":
                    del changed[phase]
                elif mutation == "empty":
                    changed[phase]["lineages"] = []
                elif mutation == "lineage":
                    changed[phase]["lineages"].pop()
                else:
                    changed[phase]["summary"] = {}
                with self.subTest(phase=phase, mutation=mutation), self.assertRaises(ValueError):
                    parse_FormationReport(changed)
            for method in METHODS:
                for location in ("summary", "lineage"):
                    changed = copy.deepcopy(self.report["formation"])
                    methods = (
                        changed[phase]["summary"] if location == "summary" else changed[phase]["lineages"][0]["methods"]
                    )
                    del methods[method]
                    with self.subTest(phase=phase, method=method, location=location), self.assertRaises(ValueError):
                        parse_FormationReport(changed)

    def test_independent_acceptance_requires_actual_split_provenance(self):
        report = self.report["formation"]
        with self.assertRaisesRegex(ValueError, "provenance"):
            formation.formation_acceptance(report["development"], report["test"], report["healthy"])
        for bad in ({"summary": {}}, {**report["healthy"], "lineages": []}):
            with self.assertRaises(ValueError):
                formation.formation_acceptance(report["development"], report["test"], bad, isolation=validated_splits())
        report["test"]["lineages"][0]["host_lineage"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "provenance"):
            formation.formation_acceptance(
                report["development"], report["test"], report["healthy"], isolation=validated_splits()
            )

    def test_summary_and_counts_cannot_erase_bad_healthy_evidence(self):
        report = self.report["formation"]
        report["healthy"]["lineages"][0]["methods"]["paired"]["acted"] = True
        with self.assertRaisesRegex(ValueError, "counts"):
            parse_FormationReport(report)

    def test_negative_and_nonfinite_loss_refused_but_signed_delta_preserved(self):
        for value in (-1.0, math.nan, math.inf, True, "0"):
            changed = copy.deepcopy(self.report["formation"])
            changed["test"]["lineages"][0]["methods"]["paired"]["raw_test_mse"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_FormationReport(changed)
        changed = copy.deepcopy(self.report["formation"])
        changed["test"]["lineages"][0]["methods"]["paired"]["harm_mse_delta"] = -1.0
        self.assertEqual(
            parse_FormationReport(changed)["test"]["lineages"][0]["methods"]["paired"]["harm_mse_delta"], -1.0
        )

    def test_conditional_formed_withdrawal_and_formation_provenance_required(self):
        for field in ("frozen_host_withdrawal_mse", "formation_conditioning_sha256"):
            changed = copy.deepcopy(self.report["lifecycle"])
            del changed["arms"]["formed_taper"][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_LifecycleReport(changed)
        changed = copy.deepcopy(self.report["lifecycle"])
        changed["arms"]["formed_taper"]["frozen_host_withdrawal_mse"] = None
        with self.assertRaises(ValueError):
            parse_LifecycleReport(changed)
        changed = copy.deepcopy(self.report["lifecycle"])
        changed["arms"]["no_growth"]["frozen_host_withdrawal_mse"] = {"with_adapter": 0.0, "without_adapter": 0.0}
        with self.assertRaises(ValueError):
            parse_LifecycleReport(changed)

    def test_main_refuses_missing_summary_or_diagnostic_gate(self):
        for phase in ("formation", "lifecycle"):
            for key in self.report[phase]["summary"]:
                changed = copy.deepcopy(self.report)
                del changed[phase]["summary"][key]
                with (
                    self.subTest(phase=phase, key=key),
                    patch.object(hybrid, "run", return_value=changed),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(ValueError),
                ):
                    hybrid.main(["--summary"])
        for key in tuple(self.report["acceptance"]):
            changed = copy.deepcopy(self.report)
            del changed["acceptance"][key]
            with (
                self.subTest(key=key),
                patch.object(hybrid, "run", return_value=changed),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(ValueError),
            ):
                hybrid.main(["--summary"])

    def test_all_closed_nested_maps_require_exact_keys_and_fixed_counts(self):
        for field in ("lineages", "generator_parameters", "evidence_dimensions"):
            original = self.report["formation"]["config"][field]
            for operation in ("empty", "missing", "unknown", "wrong_count"):
                changed = copy.deepcopy(self.report)
                target = changed["formation"]["config"][field]
                if operation == "empty":
                    target.clear()
                elif operation == "missing":
                    target.pop(next(iter(original)))
                elif operation == "unknown":
                    target["unknown"] = 0
                else:
                    target[next(iter(original))] += 1
                with self.subTest(field=field, operation=operation), self.assertRaises(ValueError):
                    parse_HybridReport(changed)
        # Method/arm maps and stage/withdrawal maps are also closed contracts.
        for path in (
            ("lifecycle", "summary", "post_learning_mse"),
            ("lifecycle", "arms", "formed_taper", "mse"),
            ("lifecycle", "arms", "formed_taper", "stage_state"),
            ("lifecycle", "arms", "formed_taper", "frozen_host_withdrawal_mse"),
        ):
            for operation in ("empty", "missing", "unknown"):
                changed = copy.deepcopy(self.report)
                target = changed
                for field in path:
                    target = target[field]
                if operation == "empty":
                    target.clear()
                elif operation == "missing":
                    target.pop(next(iter(target)))
                else:
                    target["unknown"] = 0
                with self.subTest(path=path, operation=operation), self.assertRaises(ValueError):
                    parse_HybridReport(changed)

    def test_source_identity_requires_all_runtime_files_and_coherent_digests(self):
        for operation in ("empty", "unknown", "digest_short", "digest_nonhex", "aggregate"):
            changed = copy.deepcopy(self.report)
            identity = changed["source_identity"]
            if operation == "empty":
                identity["files"] = {}
            elif operation == "unknown":
                identity["files"]["src/sultai/unknown.py"] = "0" * 64
            elif operation == "digest_short":
                identity["files"][SOURCE_FILES[0]] = "0"
            elif operation == "digest_nonhex":
                identity["files"][SOURCE_FILES[0]] = "x" * 64
            else:
                identity["aggregate_sha256"] = "0" * 64
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                parse_HybridReport(changed)
        for filename in SOURCE_FILES:
            changed = copy.deepcopy(self.report)
            del changed["source_identity"]["files"][filename]
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                parse_HybridReport(changed)
        package = hybrid.Path(hybrid.__file__).resolve().parent
        paths = [package / hybrid.Path(name).name for name in SOURCE_FILES]
        with (
            patch.object(hybrid.Path, "glob", return_value=[*paths, package / "unrecorded.py"]),
            self.assertRaisesRegex(ValueError, "inventory"),
        ):
            hybrid.source_identity()

    def test_actual_runner_numerical_report_matches_pinned_evidence(self):
        actual = json.loads(json.dumps(hybrid.run(), allow_nan=False))
        reference = json.loads(json.dumps(self.report, allow_nan=False))
        actual.pop("runtime")
        reference.pop("runtime")
        self.assertEqual(actual, reference)
        self.assertEqual(actual["protocol"], "sultai-hybrid-cpu-v2")
        self.assertEqual(len(actual["source_identity"]["files"]), 12)
        self.assertEqual(set(actual["source_identity"]["files"]), set(SOURCE_FILES))
