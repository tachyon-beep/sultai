"""Mechanics tests use a labeled conventional callback, not learned evidence."""

import json
import math
import unittest
import weakref

from sultai import lifecycle
from sultai.repair import CHANNELS, PARAMETERS, Adapter, Example, fit


def fixture():
    # A same-feature planted task; this conventional callback test is independent
    # of the learned generator's measured comparative outcomes.
    target = Adapter(
        tuple(
            tuple(
                0.25
                if i == j
                else -0.20
                if j == (i + 1) % CHANNELS
                else 0.15 * (-1 if i % 2 else 1)
                if j == (i + 2) % CHANNELS
                else 0.0
                for j in range(CHANNELS)
            )
            for i in range(CHANNELS)
        ),
        tuple(0.10 * (-1 if i % 2 else 1) for i in range(CHANNELS)),
    )
    return (
        lifecycle._examples(7001, "training", lifecycle.TOTAL_STEPS * lifecycle.BATCH_SIZE, target),
        lifecycle._examples(7002, "conditioning", lifecycle.CONDITIONING_COUNT, target),
        lifecycle._examples(7003, "audit", lifecycle.AUDIT_COUNT, target),
    )


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training, cls.conditioning, cls.audit = fixture()
        cls.received = []
        cls.references = []

        def conventional_callback(examples):
            cls.received.append(tuple(examples))
            result = fit(examples)
            cls.references.append(weakref.ref(result))
            return result

        cls.report = lifecycle.run_lifecycle(
            conventional_callback, formation_label="test-only conventional ridge callback"
        )

    def test_real_deletion_and_serialized_counts(self):
        self.assertEqual(self.report["arms"]["early_static_retained"]["counts"]["final"], 544)
        for name, arm in self.report["arms"].items():
            expected = 544 if name == "early_static_retained" else 272
            self.assertEqual(arm["counts"]["before"], PARAMETERS)
            self.assertEqual(arm["counts"]["final"], expected)
            self.assertEqual(arm["serialized_parameter_count"], expected)
            self.assertEqual("adapter" in arm["final_serialization_keys"], name == "early_static_retained")
            self.assertTrue(arm["optimizer_state_empty"])
            self.assertEqual(
                set(arm["final_serialization_keys"]),
                {"host", "optimizer_state", "adapter"}
                if name == "early_static_retained"
                else {"host", "optimizer_state"},
            )
        self.assertEqual(self.report["arms"]["formed_taper"]["counts"]["peak"], 544)
        self.assertTrue(all(reference() is None for reference in self.references))

    def test_sgd_is_task_gradient_not_teacher_copy(self):
        model = lifecycle._Model()
        teacher = Adapter(
            tuple(tuple(0.4 if i == j else 0.0 for j in range(CHANNELS)) for i in range(CHANNELS)), (0.1,) * CHANNELS
        )
        model.add(teacher, trainable=False)
        h = (0.2,) * CHANNELS
        example = Example("gradient-check", h, (0.8,) * CHANNELS)
        before = teacher.to_json()
        count = lifecycle._sgd_step(model, (example,), 0.5)
        error = model.host.weights[0][0]  # Check exact analytical first gradient.
        residual = 0.2 + 0.5 * (0.4 * math.tanh(0.2) + 0.1) - 0.8
        expected = -lifecycle.LEARNING_RATE * 2 / CHANNELS * residual * math.tanh(0.2)
        self.assertAlmostEqual(error, expected, places=15)
        self.assertEqual(count, 1)
        self.assertEqual(teacher.to_json(), before)
        self.assertNotEqual(model.host.weights[0][0], teacher.weights[0][0])
        self.assertNotEqual(model.host.weights[0][1], 0.0)
        self.assertIs(model.temporary, teacher)
        self.assertTrue(all(arm["host_unchanged_at_insertion"] for arm in self.report["arms"].values()))
        self.assertEqual(self.report["arms"]["formed_taper"]["auxiliary_gradient_updates"], 0)

    def test_current_host_residual_supplied_without_projection(self):
        self.assertEqual(len(self.received), 1)
        model = lifecycle._Model()
        for step in range(lifecycle.INITIAL_STEPS):
            batch = self.training[step * lifecycle.BATCH_SIZE : (step + 1) * lifecycle.BATCH_SIZE]
            lifecycle._sgd_step(model, batch, 0.0)
        for original, received in zip(self.conditioning, self.received[0], strict=True):
            prediction = model.predict(original.h, 0.0)
            expected = tuple(
                h + target - output for h, target, output in zip(original.h, original.target, prediction, strict=True)
            )
            self.assertEqual(received.sample_id, original.sample_id)
            self.assertEqual(received.h, original.h)
            self.assertEqual(received.target, expected)
        self.assertNotEqual(self.received[0][0].target, self.conditioning[0].target)

    def test_common_future_updates_and_truthful_costs(self):
        arms = self.report["arms"]
        self.assertEqual(len({arm["initial_host_sha256"] for arm in arms.values()}), 1)
        self.assertEqual(len({arm["training_pairs_sha256"] for arm in arms.values()}), 1)
        self.assertEqual(len({arm["training_example_ids_sha256"] for arm in arms.values()}), 1)
        self.assertEqual({arm["host_updates"] for arm in arms.values()}, {lifecycle.TOTAL_STEPS})
        self.assertEqual({arm["training_examples"] for arm in arms.values()}, {2176})
        self.assertEqual(self.report["cost"]["host_gradient_steps"], 6 * 136)
        self.assertEqual(self.report["cost"]["formation_calls"], 1)
        self.assertEqual(self.report["cost"]["formation_conditioning_examples"], 64)
        self.assertEqual(self.report["cost"]["selection_queries"], 0)
        self.assertEqual(self.report["cost"]["final_audit_queries"], (6 * 6 + 2) * 64)
        self.assertEqual(self.report["cost"]["auxiliary_gradient_steps"], 2 * (72 - 8) + 136 + 72)
        self.assertIn("raw candidate", self.report["insertion_policy"])

    def test_fixed_schedule_stage_metrics_and_budgets(self):
        for arm in self.report["arms"].values():
            self.assertEqual(
                set(arm["mse"]),
                {"pre_add", "immediate_add", "mid_taper", "pre_remove", "immediate_remove", "post_learning"},
            )
            self.assertEqual(arm["stage_state"]["mid_taper"]["completed_host_updates"], 48)
            self.assertEqual(arm["stage_state"]["immediate_remove"]["completed_host_updates"], 72)
            self.assertEqual(arm["stage_state"]["post_learning"]["completed_host_updates"], 136)
            self.assertTrue(all(math.isfinite(value) for value in arm["mse"].values()))
        formed = self.report["arms"]["formed_taper"]
        self.assertEqual(formed["stage_state"]["pre_remove"]["alpha"], 0.0)
        self.assertEqual(formed["stage_state"]["mid_taper"]["alpha"], 0.5)
        self.assertEqual(formed["mse"]["pre_remove"], formed["mse"]["immediate_remove"])
        self.assertEqual(formed["stage_state"]["pre_remove"]["allocated_parameters"], 544)
        self.assertEqual(formed["stage_state"]["immediate_remove"]["allocated_parameters"], 272)
        self.assertEqual(self.report["arms"]["early_static_retained"]["counts"]["at_step_zero"], 544)
        self.assertEqual(self.report["arms"]["no_growth"]["counts"]["peak"], 272)

    def test_held_out_targets_cannot_change_formation_or_training(self):
        audit = tuple(Example(e.sample_id, e.h, tuple(x + 10.0 for x in e.target)) for e in self.audit)
        supplied = []

        def callback(examples):
            supplied.append(tuple(examples))
            return fit(examples)

        changed = lifecycle._run_task(
            callback, self.training, self.conditioning, audit, "test-only conventional ridge callback"
        )
        self.assertEqual(supplied, self.received)
        for name, original in self.report["arms"].items():
            self.assertEqual(
                original["final_serialization_sha256"], changed["arms"][name]["final_serialization_sha256"]
            )
            self.assertEqual(
                original["formation_conditioning_sha256"], changed["arms"][name]["formation_conditioning_sha256"]
            )
            self.assertNotEqual(original["mse"], changed["arms"][name]["mse"])

    def test_disjoint_split_guard(self):
        conditioning = (self.training[0], *self.conditioning[1:])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            lifecycle._run_task(fit, self.training, conditioning, self.audit, "unused")

    def test_direct_model_remove_is_structural(self):
        model = lifecycle._Model()
        model.add(Adapter.zero(), trainable=True)
        reference = weakref.ref(model.temporary)
        self.assertEqual(model.parameters, 544)
        model.remove()
        self.assertIsNone(model.temporary)
        self.assertIsNone(reference())
        self.assertEqual(model.parameters, 272)
        self.assertNotIn("adapter", model.serialized())
        final = json.loads(json.dumps(model.serialized(), allow_nan=False))
        self.assertEqual(set(final), {"host", "optimizer_state"})
        self.assertEqual(sum(len(row) for row in final["host"]["weights"]) + len(final["host"]["bias"]), 272)
        self.assertEqual(final["optimizer_state"], {})

    def test_affine_folding_identity_and_nonlinear_signed_counterexample(self):
        diagnostics = lifecycle.run_diagnostics()
        self.assertTrue(diagnostics["affine_folding"]["identity_verified"])
        self.assertLess(diagnostics["affine_folding"]["max_absolute_error"], 1e-12)
        signed = diagnostics["signed_probe"]
        self.assertTrue(signed["passive_identical"])
        self.assertEqual(signed["a_plus"], signed["b_minus"])
        self.assertFalse(signed["finite_changes_are_opposite"])
        self.assertAlmostEqual(signed["sum_same_sign_changes"], 2 * signed["epsilon"] ** 2)
        with self.assertRaises(ValueError):
            lifecycle.signed_probe_diagnostic(0.0)

    def test_actual_failed_audit_reports_failure_without_retry_or_state_changes(self):
        altered = tuple(Example(e.sample_id, e.h, tuple(v + 10.0 for v in e.target)) for e in self.audit)
        report = lifecycle._run_task(fit, self.training, self.conditioning, altered, "negative audit control")
        self.assertFalse(report["acceptance"]["formed_post_removal_target_met"])
        self.assertFalse(report["arms"]["formed_taper"]["post_removal_target_met"])
        self.assertGreater(report["arms"]["formed_taper"]["mse"]["post_learning"], 0.01)
        self.assertEqual(report["cost"], self.report["cost"])
        for name in lifecycle.ARMS:
            self.assertEqual(report["arms"][name]["host_updates"], lifecycle.TOTAL_STEPS)
            self.assertEqual(
                report["arms"][name]["final_serialization_sha256"],
                self.report["arms"][name]["final_serialization_sha256"],
            )

    def test_failure_is_reported_without_target_driven_retry(self):
        formed = self.report["arms"]["formed_taper"]
        self.assertEqual(
            formed["post_removal_target_met"], formed["mse"]["post_learning"] <= 0.01 and formed["adapter_absent"]
        )
        self.assertEqual(self.report["config"]["post_removal_mse_target"], 0.01)
        self.assertEqual(self.report["formation_label"], "test-only conventional ridge callback")
        json.dumps(self.report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
