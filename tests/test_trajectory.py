"""Independent known-answer fixtures; never run corrected protocol cohorts."""

import dataclasses
import math
import random
import unittest
import weakref
from unittest.mock import patch

from sultai import lifecycle, trajectory
from sultai.formation import materialize
from sultai.repair import CHANNELS, PARAMETERS, Adapter, Example, fit
from sultai.trust import ContractViolation


class RateTests(unittest.TestCase):
    def test_changed_learning_rate_known_answer_and_default(self):
        h = (0.2,) * CHANNELS
        batch = (Example("lr-known-answer", h, (0.8,) * CHANNELS),)
        default, explicit, doubled = lifecycle._Model(), lifecycle._Model(), lifecycle._Model()
        lifecycle._sgd_step(default, batch, 0.0)
        lifecycle._sgd_step(explicit, batch, 0.0, learning_rate=0.8)
        lifecycle._sgd_step(doubled, batch, 0.0, learning_rate=1.6)
        self.assertEqual(default.serialized(), explicit.serialized())
        expected = -1.6 * 2 / CHANNELS * (0.2 - 0.8) * math.tanh(0.2)
        self.assertAlmostEqual(doubled.host.weights[0][0], expected, places=15)
        self.assertEqual(doubled.host.weights[0][0], 2 * default.host.weights[0][0])
        for invalid in (0.0, -1.0, float("nan")):
            with self.assertRaises(ContractViolation):
                lifecycle._sgd_step(doubled, batch, 0.0, learning_rate=invalid)


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = 61000  # Separate from every fixed development/confirmation root.
        cls.training, cls.conditioning, cls.audit = trajectory._fixture(cls.root)
        cls.received = []
        cls.references = []

        def conventional(examples):
            cls.received.append(tuple(examples))
            adapter = fit(examples)
            cls.references.append(weakref.ref(adapter))
            return adapter

        cls.report = trajectory._run_stream(conventional, cls.root, cls.training, cls.conditioning, cls.audit)
        cls.arms = {arm.name: arm for arm in cls.report.arms}

    def test_fixed_record_window_cannot_be_rescued_by_final_recovery(self):
        for arm in self.report.arms:
            self.assertEqual(tuple(point.completed_steps for point in arm.checkpoints), tuple(range(8, 73)))
            self.assertEqual(len(arm.checkpoints), 65)
            self.assertEqual(arm.trajectory_mean_mse, sum(point.mse for point in arm.checkpoints) / 65)
            self.assertEqual(arm.insertion_mse, arm.checkpoints[0].mse)
            self.assertEqual(arm.immediate_removal_mse, arm.checkpoints[-1].mse)
            self.assertTrue(math.isfinite(arm.final_recovery_mse))
        zero_arms = tuple(dataclasses.replace(arm, final_recovery_mse=0.0) for arm in self.report.arms)
        self.assertEqual(trajectory._contrasts(zero_arms), self.report.contrasts)
        self.assertEqual(trajectory._specificity(zero_arms), self.report.diagnostic_specificity)
        self.assertEqual(self.report.worthwhile_margin, None)
        self.assertFalse(self.report.meaningful_benefit_established)

    def test_zero_cannot_satisfy_specificity_even_if_endpoint_is_good(self):
        zero = self.arms["frozen_zero_taper"]
        baseline = self.arms["no_growth"]
        self.assertEqual(tuple(p.mse for p in zero.checkpoints), tuple(p.mse for p in baseline.checkpoints))
        self.assertEqual(zero.final_serialization_sha256, baseline.final_serialization_sha256)
        replaced = tuple(
            dataclasses.replace(zero, name="formed_taper", final_recovery_mse=0.0)
            if arm.name == "formed_taper"
            else arm
            for arm in self.report.arms
        )
        self.assertFalse(trajectory._specificity(replaced))

    def test_negated_candidate_fails_immediate_insertion_and_random_norm_matches(self):
        self.assertGreater(self.arms["formed_taper"].insertion_improvement, 1e-12)
        self.assertLess(self.arms["frozen_negated_taper"].insertion_improvement, 0.0)
        self.assertAlmostEqual(
            self.arms["formed_taper"].candidate_squared_norm,
            self.arms["frozen_random_taper"].candidate_squared_norm,
            places=12,
        )
        self.assertEqual(
            self.arms["formed_taper"].candidate_squared_norm,
            self.arms["frozen_negated_taper"].candidate_squared_norm,
        )
        self.assertEqual(self.arms["frozen_zero_taper"].candidate_squared_norm, 0.0)
        self.assertEqual(len(self.received), 1)
        model = lifecycle._Model()
        for step in range(8):
            lifecycle._sgd_step(model, self.training[step * 16 : (step + 1) * 16], 0.0)
        for original, supplied in zip(self.conditioning, self.received[0], strict=True):
            output = model.predict(original.h, 0.0)
            self.assertEqual(supplied.h, original.h)
            self.assertEqual(
                supplied.target,
                tuple(
                    h + target - actual for h, target, actual in zip(original.h, original.target, output, strict=True)
                ),
            )

    def test_lr_matched_retained_equivalence_in_function_space_at_all_boundaries(self):
        self.assertEqual(tuple(point.completed_steps for point in self.report.rate_equivalence), tuple(range(137)))
        self.assertTrue(all(point.max_absolute_error <= 1e-12 for point in self.report.rate_equivalence))
        self.assertTrue(self.report.rate_equivalence_verified)
        matched, retained = self.arms["no_growth_lr_matched"], self.arms["early_static_retained"]
        for left, right in zip(matched.checkpoints, retained.checkpoints, strict=True):
            self.assertAlmostEqual(left.mse, right.mse, delta=1e-12)
        self.assertNotEqual(matched.final_parameters, retained.final_parameters)

    def test_actual_deletion_frozen_gradient_accounting_and_allocated_parameter_steps(self):
        self.assertTrue(all(reference() is None for reference in self.references))
        self.assertEqual(
            set(self.arms),
            set(lifecycle.ARMS)
            | {"no_growth_lr_matched", "frozen_zero_taper", "frozen_negated_taper", "frozen_random_taper"},
        )
        for arm in self.report.arms:
            retained = arm.name == "early_static_retained"
            self.assertEqual(arm.final_parameters, 544 if retained else 272)
            self.assertEqual(arm.serialized_parameter_count, arm.final_parameters)
            self.assertEqual(arm.adapter_absent, not retained)
            self.assertEqual("adapter" in arm.final_serialization_keys, retained)
            self.assertTrue(arm.host_unchanged_at_insertion)
            self.assertEqual(arm.cost.host_gradient_steps, 136)
            self.assertEqual(arm.cost.training_example_uses, 2176)
            self.assertEqual(arm.cost.host_parameter_gradient_steps, 136 * PARAMETERS)
            self.assertEqual(
                arm.cost.auxiliary_parameter_gradient_steps, arm.cost.auxiliary_gradient_steps * PARAMETERS
            )
            if arm.name.startswith("frozen") or arm.name == "formed_taper":
                self.assertEqual(arm.cost.auxiliary_gradient_steps, 0)
                self.assertEqual(arm.cost.allocated_parameter_steps, (136 + 64) * PARAMETERS)
                self.assertEqual(arm.checkpoints[-1].allocated_parameters, 272)
            self.assertGreaterEqual(arm.cost.audit_queries, 67 * 64)
        self.assertEqual(self.report.cost.formation_calls, 1)
        self.assertEqual(self.report.cost.formation_conditioning_examples, 64)
        self.assertEqual(self.report.cost.candidate_count, 4)
        self.assertEqual(self.report.cost.selection_queries, 0)
        self.assertEqual(self.report.cost.audit_queries, sum(arm.cost.audit_queries for arm in self.report.arms))
        self.assertEqual(self.report.cost.host_gradient_steps, 10 * 136)
        self.assertEqual(self.report.cost.training_example_uses, 10 * 2176)
        self.assertEqual(self.report.cost.candidate_parameter_values, 4 * PARAMETERS)
        self.assertEqual(len({arm.training_pairs_sha256 for arm in self.report.arms}), 1)

    def test_held_out_targets_do_not_change_formation_or_training(self):
        changed_targets = tuple(
            Example(e.sample_id, e.h, tuple(target + 5.0 for target in e.target)) for e in self.audit
        )
        received = []

        def callback(examples):
            received.append(tuple(examples))
            return fit(examples)

        changed = trajectory._run_stream(callback, self.root, self.training, self.conditioning, changed_targets)
        self.assertEqual(received, self.received)
        self.assertEqual(changed.cost, self.report.cost)
        for old, new in zip(self.report.arms, changed.arms, strict=True):
            self.assertEqual(old.final_serialization_sha256, new.final_serialization_sha256)
            self.assertEqual(old.candidate_sha256, new.candidate_sha256)
            self.assertEqual(old.formation_conditioning_sha256, new.formation_conditioning_sha256)
            self.assertNotEqual(old.trajectory_mean_mse, new.trajectory_mean_mse)

    def test_fixed_phase_roots_without_executing_protocol_cohorts(self):
        self.assertEqual(trajectory.DEVELOPMENT_ROOTS, (8000, 8010))
        self.assertEqual(trajectory.CONFIRMATION_ROOTS, (9000, 9010, 9020, 9030))
        # Intercept construction, so this test never executes the named cohorts.
        with patch.object(trajectory, "_fixture", side_effect=RuntimeError("cohort execution blocked")) as fixture:
            with self.assertRaisesRegex(RuntimeError, "cohort execution blocked"):
                trajectory.run_trajectory(fit, "development")
            fixture.assert_called_once_with(8000)
        with self.assertRaisesRegex(ValueError, "phase"):
            trajectory.run_trajectory(fit, "invalid")
        self.assertEqual(self.report.partition_seeds, (61001, 61002, 61003, 61004))
        self.assertEqual(materialize(lifecycle.TASK_COEFFICIENTS), trajectory._target())

    def test_faults_propagate_and_split_validation_rejects_duplicates(self):
        def fault(examples):
            raise RuntimeError("trusted former fault")

        with self.assertRaisesRegex(RuntimeError, "trusted former fault"):
            trajectory._run_stream(fault, self.root, self.training, self.conditioning, self.audit)
        with self.assertRaisesRegex(ContractViolation, "Adapter"):
            trajectory._run_stream(lambda examples: None, self.root, self.training, self.conditioning, self.audit)
        duplicate = (self.training[0], *self.conditioning[1:])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            trajectory._run_stream(fit, self.root, self.training, duplicate, self.audit)
        with self.assertRaisesRegex(ValueError, "counts"):
            trajectory._run_stream(fit, self.root, self.training[:-1], self.conditioning, self.audit)

    def test_per_stream_paired_contrasts_keep_all_comparators_and_strict_epsilon(self):
        self.assertEqual(tuple(item.comparator for item in self.report.contrasts), trajectory.COMPARATORS)
        formed = self.arms["formed_taper"]
        for item in self.report.contrasts:
            self.assertEqual(
                item.improvement, self.arms[item.comparator].trajectory_mean_mse - formed.trajectory_mean_mse
            )
            self.assertEqual(item.strictly_improved, item.improvement > 1e-12)
        self.assertEqual(
            self.report.diagnostic_specificity,
            formed.insertion_improvement > 1e-12 and all(c.strictly_improved for c in self.report.contrasts),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.report.root = 2


class ValidityTests(unittest.TestCase):
    def test_mechanical_checks_allow_negative_scientific_result(self):
        original = TrajectoryTests.report
        self.assertTrue(original.software_valid)
        self.assertTrue(all(dataclasses.asdict(original.known_answers).values()))
        shortened = tuple(
            dataclasses.replace(arm, checkpoints=arm.checkpoints[:-1]) if arm.name == "formed_taper" else arm
            for arm in original.arms
        )
        checks = trajectory._known_answers(shortened, original.rate_equivalence)
        self.assertFalse(checks.complete_record_window)
        self.assertTrue(checks.rate_equivalence)
        bad_rates = tuple(
            dataclasses.replace(point, max_absolute_error=0.1) if point.completed_steps == 40 else point
            for point in original.rate_equivalence
        )
        self.assertFalse(trajectory._known_answers(original.arms, bad_rates).rate_equivalence)
        masked = tuple(
            dataclasses.replace(arm, adapter_absent=False) if arm.name == "formed_taper" else arm
            for arm in original.arms
        )
        self.assertFalse(trajectory._known_answers(masked, original.rate_equivalence).actual_removal)


class RandomDirectionTests(unittest.TestCase):
    def test_frozen_random_is_fixed_gaussian_four_template_direction(self):
        candidate = materialize((0.1, -0.2, 0.3, -0.4))
        rng = random.Random(61004)
        direction = materialize(tuple(rng.gauss(0.0, 1.0) for _ in range(4)))
        expected = trajectory._scaled(
            direction, (trajectory._norm_squared(candidate) / trajectory._norm_squared(direction)) ** 0.5
        )
        zero, negated, actual = trajectory._replacements(candidate, 61000)
        self.assertEqual(actual, expected)
        self.assertEqual(zero, Adapter.zero())
        self.assertEqual(negated, trajectory._scaled(candidate, -1.0))
        self.assertEqual(trajectory._replacements(Adapter.zero(), 61000)[2], Adapter.zero())


if __name__ == "__main__":
    unittest.main()
