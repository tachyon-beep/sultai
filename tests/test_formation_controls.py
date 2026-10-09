import dataclasses
import hashlib
import json
import math
import random
import unittest
from unittest.mock import patch

from sultai.formation import FIXED, conditioning_evidence, make_episode, materialize
from sultai.formation_controls import (
    CONTROL_METHODS,
    analytic_template_fit,
    evaluate_formation_controls,
    guard_control_cohorts,
    make_noisy_healthy_episode,
    marginal_evidence,
    norm_matched_random,
    run_formation_controls,
)
from sultai.repair import Adapter, Example, mse
from sultai.trust import FitUnavailable


def small_episode(seed, *, healthy=False):
    episode = make_episode(seed, healthy=healthy)
    return dataclasses.replace(
        episode, conditioning=episode.conditioning[:20], selection=episode.selection[:4], test=episode.test[:8]
    )


def fixtures():
    return (
        tuple(small_episode(seed) for seed in range(610, 614)),
        tuple(small_episode(seed) for seed in (710, 711)),
        tuple(small_episode(seed, healthy=True) for seed in (810, 811)),
        tuple(
            dataclasses.replace(
                episode, conditioning=episode.conditioning[:20], selection=episode.selection[:4], test=episode.test[:8]
            )
            for episode in (make_noisy_healthy_episode(910), make_noisy_healthy_episode(911))
        ),
    )


class FormationControlTests(unittest.TestCase):
    def test_marginal96_matches_independent_moments_and_arbitrary_residual_permutation(self):
        episode = small_episode(710)
        expected = []
        for part in (
            tuple(x.h for x in episode.conditioning),
            tuple(tuple(math.tanh(v) for v in x.h) for x in episode.conditioning),
            tuple(tuple(t - h for t, h in zip(x.target, x.h, strict=True)) for x in episode.conditioning),
        ):
            for power in (1, 2):
                expected.extend(math.fsum(row[channel] ** power for row in part) / len(part) for channel in range(16))
        observed = marginal_evidence(episode.conditioning)
        self.assertEqual(len(observed), 96)
        for left, right in zip(observed, expected, strict=True):
            self.assertAlmostEqual(left, right, places=14)
        permutation = list(range(len(episode.conditioning)))
        random.Random(77).shuffle(permutation)
        permuted = tuple(
            Example(
                example.sample_id,
                example.h,
                tuple(
                    h + target - donor_h
                    for h, target, donor_h in zip(
                        example.h,
                        episode.conditioning[permutation[index]].target,
                        episode.conditioning[permutation[index]].h,
                        strict=True,
                    )
                ),
            )
            for index, example in enumerate(episode.conditioning)
        )
        for left, right in zip(observed, marginal_evidence(permuted), strict=True):
            self.assertAlmostEqual(left, right, places=14)
        self.assertNotEqual(
            conditioning_evidence(episode.conditioning, "paired"), conditioning_evidence(permuted, "paired")
        )

    def test_analytic4x4_known_answer_independent_diagonal_system_without_intercept(self):
        # tanh(0)=0 isolates the alternating-bias template; coordinate inputs
        # then provide an independently identifiable, nondegenerate system.
        planted = materialize((0.25, -0.20, 0.15, 0.10))
        inputs = [(0.0,) * 16]
        inputs.extend(tuple(1.0 if i == channel else 0.0 for i in range(16)) for channel in range(16))
        pairs = tuple(Example(f"analytic:{i}", h, planted.apply(h)) for i, h in enumerate(inputs))
        with patch("sultai.formation_controls._regress", side_effect=AssertionError("meta/intercept fit")):
            observed = analytic_template_fit(pairs)
        self.assertLess(mse(observed, pairs), 1e-16)
        for observed_row, expected_row in zip(observed.weights, planted.weights, strict=True):
            for left, right in zip(observed_row, expected_row, strict=True):
                self.assertAlmostEqual(left, right, places=8)
        self.assertEqual(analytic_template_fit(small_episode(810, healthy=True).conditioning), Adapter.zero())
        with self.assertRaises(ValueError):
            analytic_template_fit(())

    def test_analytic_valid_finite_overflow_is_recoverable(self):
        for h, target in ((1e308, -1e308), (1.0, 1e308)):
            example = Example("analytic-finite-overflow", (h,) * 16, (target,) * 16)
            with self.subTest(h=h, target=target), self.assertRaises(FitUnavailable):
                analytic_template_fit((example,))
        with patch("sultai.formation_controls.math.fsum", side_effect=RuntimeError("trusted summation fault")):
            with self.assertRaisesRegex(RuntimeError, "trusted summation fault"):
                analytic_template_fit((Example("analytic-trusted-fault", (1.0,) * 16, (0.0,) * 16),))

    def test_noisy_healthy_independent_partition_streams_and_clean_audit(self):
        episode = make_noisy_healthy_episode(910)
        self.assertEqual(episode, make_noisy_healthy_episode(910))
        base = make_episode(910, healthy=True)
        for name in ("conditioning", "selection"):
            seed = int(
                hashlib.sha256(f"sultai-noisy-healthy-v1:{episode.host_lineage}:{name}".encode()).hexdigest(), 16
            )
            rng = random.Random(seed)
            actual = getattr(episode, name)
            original = getattr(base, name)
            for noisy, clean in zip(actual, original, strict=True):
                self.assertEqual(noisy.h, clean.h)
                self.assertNotEqual(noisy.sample_id, clean.sample_id)
                self.assertEqual(noisy.target, tuple(h + rng.gauss(0.0, 0.1) for h in clean.h))
        self.assertTrue(all(x.target == x.h for x in episode.test))
        self.assertNotEqual(episode.conditioning[0].target, episode.selection[0].target)

    def test_global_id_collision_in_healthy_prevents_any_fit(self):
        training, evaluation, healthy, noisy = fixtures()
        collision = dataclasses.replace(
            healthy[0],
            conditioning=(
                dataclasses.replace(healthy[0].conditioning[0], sample_id=training[0].conditioning[0].sample_id),
                *healthy[0].conditioning[1:],
            ),
        )
        with patch("sultai.formation_controls.train_generators", side_effect=AssertionError("fit before guard")):
            with self.assertRaisesRegex(ValueError, "sample ID"):
                evaluate_formation_controls(training, evaluation, (collision, healthy[1]), noisy)
        with self.assertRaisesRegex(ValueError, "lineage"):
            guard_control_cohorts({"train": training, "noisy_healthy": (training[0],)})

    def test_fixture_helper_does_not_claim_noise_seed_authenticity_and_bad_audit_prevents_fit(self):
        training, evaluation, clean, noisy = fixtures()
        changed = dataclasses.replace(
            noisy[0],
            conditioning=(
                dataclasses.replace(
                    noisy[0].conditioning[0], target=tuple(t + 0.01 for t in noisy[0].conditioning[0].target)
                ),
                *noisy[0].conditioning[1:],
            ),
        )
        corrupted_audit = dataclasses.replace(
            noisy[0], test=(dataclasses.replace(noisy[0].test[0], target=(1.0,) * 16), *noisy[0].test[1:])
        )
        report, _ = evaluate_formation_controls(training, evaluation, clean, (changed, noisy[1]))
        noisy_report = next(c for c in report.cohorts if c.name == "noisy_healthy")
        self.assertTrue(
            all(
                stream.role != "sultai-noisy-healthy-v1"
                for row in noisy_report.lineages
                for stream in row.random_streams
            )
        )
        with patch("sultai.formation_controls.train_generators", side_effect=AssertionError("fit before guard")):
            with self.assertRaises(ValueError):
                evaluate_formation_controls(training, evaluation, clean, (corrupted_audit, noisy[1]))

    def test_within_episode_duplicate_rejected_even_for_structural_source(self):
        episode = small_episode(710)

        class DuplicateSource:
            host_lineage = episode.host_lineage
            bottleneck_family = episode.bottleneck_family
            conditioning = (episode.conditioning[0], episode.conditioning[0])
            selection = episode.selection
            test = episode.test

        with self.assertRaisesRegex(ValueError, "sample ID"):
            guard_control_cohorts({"evaluation": (DuplicateSource(),)})

    def test_frozen_random_norm_seed_and_outcome_independence(self):
        adapter = materialize((0.25, -0.2, 0.15, 0.1))
        observed = norm_matched_random(adapter, "known-answer-lineage")

        def norm(value):
            return math.fsum(w * w for row in value.weights for w in row) + math.fsum(b * b for b in value.bias)

        self.assertAlmostEqual(norm(observed), norm(adapter), places=14)
        self.assertEqual(observed, norm_matched_random(adapter, "known-answer-lineage"))
        self.assertNotEqual(observed, norm_matched_random(adapter, "different-lineage"))
        self.assertEqual(norm_matched_random(Adapter.zero(), "known-answer-lineage"), Adapter.zero())

    def test_same_offline_teacher_supervision_and_one_additional_meta_fit(self):
        from sultai.formation import _regress, _teacher_coefficients, train_generators

        training, evaluation, clean, noisy = fixtures()
        _, teachers, _ = train_generators(training)
        expected_targets = tuple(_teacher_coefficients(a) for a in teachers.adapters)
        with (
            patch("sultai.formation_controls._regress", wraps=_regress) as meta,
            patch("sultai.formation_controls.train_generators", wraps=train_generators) as legacy,
        ):
            evaluate_formation_controls(training, evaluation, clean, noisy)
        self.assertEqual(legacy.call_count, 1)
        self.assertEqual(meta.call_count, 1)
        rows, targets, ridge = meta.call_args.args
        self.assertEqual(rows, tuple(marginal_evidence(e.conditioning) for e in training))
        self.assertEqual(targets, expected_targets)
        self.assertEqual(ridge, FIXED.meta_ridge)

    def test_helpful_rejections_harmful_and_neutral_admissions_retained(self):
        from sultai.formation_controls import ControlCandidate, ControlCost, _evaluate_frozen_cohort

        adapter = Adapter(Adapter.zero().weights, (0.1,) * 16)
        empty = ControlCost(
            tuple(
                (key, 0)
                for key in (
                    "oracle_conditioning_queries",
                    "identity_probe_queries",
                    "teacher_fits",
                    "teacher_fit_examples",
                    "meta_regression_fits",
                    "meta_training_rows",
                    "full_ridge_fits",
                    "full_ridge_fit_examples",
                    "sgd_steps",
                    "sgd_example_steps",
                    "generator_forwards",
                    "retrieval_distance_evaluations",
                    "selection_queries",
                    "final_audit_queries",
                    "affine_ridge_fits",
                    "affine_ridge_fit_examples",
                )
            ),
            0,
            0,
        )
        candidates = tuple(
            ControlCandidate(name, Adapter.zero() if name == "no_op" else adapter, empty) for name in CONTROL_METHODS
        )

        def episode(lineage, selection_target, audit_target):
            from sultai.repair import Episode

            def pair(part, target):
                return (Example(f"{lineage}:{part}", (0.0,) * 16, (target,) * 16),)

            return Episode(
                lineage,
                "known-answer",
                pair("conditioning", 0.1),
                pair("selection", selection_target),
                pair("audit", audit_target),
            )

        cases = (
            ("harmful", episode("harmful", 0.1, 0.0)),
            ("neutral", episode("neutral", 0.1, 0.05)),
            ("helpful", episode("helpful", -0.1, 0.1)),
        )
        report = _evaluate_frozen_cohort("fixture", tuple(e for _, e in cases), (candidates,) * 3)
        harmful, neutral, helpful = (next(m for m in row.methods if m.name == "paired") for row in report.lineages)
        self.assertTrue(
            harmful.acted and harmful.false_admission and harmful.harmful_admission and harmful.realized_harm
        )
        self.assertTrue(neutral.acted and neutral.false_admission and neutral.neutral_admission)
        self.assertFalse(neutral.harmful_admission)
        self.assertTrue(helpful.false_rejection and helpful.no_op)
        self.assertFalse(helpful.acted)
        self.assertEqual(
            (harmful.audit_class, neutral.audit_class, helpful.audit_class), ("harmful", "neutral", "helpful")
        )
        summary = next(s for s in report.summary if s.name == "paired")
        self.assertEqual(
            (
                summary.false_admissions,
                summary.harmful_admissions,
                summary.neutral_admissions,
                summary.false_rejections,
            ),
            (2, 1, 1, 1),
        )

    def test_exact_repair_negation_fails_noiseless_insertion_and_zero_is_exact_null(self):
        episode = small_episode(710)
        repair = analytic_template_fit(episode.conditioning)
        negated = Adapter(tuple(tuple(-w for w in row) for row in repair.weights), tuple(-b for b in repair.bias))
        self.assertLess(mse(repair, episode.test), 1e-12)
        self.assertGreater(mse(negated, episode.test), mse(Adapter.zero(), episode.test))
        report, _ = evaluate_formation_controls(*fixtures())
        for cohort in report.cohorts:
            for row in cohort.lineages:
                baseline = next(m for m in row.methods if m.name == "no_op")
                zero = next(m for m in row.methods if m.name == "frozen_zero")
                self.assertEqual(zero.raw_audit_mse, baseline.raw_audit_mse)
                self.assertEqual(zero.admitted_audit_mse, baseline.admitted_audit_mse)
                self.assertTrue(zero.exact_null and zero.no_op)
                self.assertEqual(zero.audit_class, "neutral")

    def test_full_report_retains_all_outcomes_accounting_and_healthy_provenance(self):
        report, generator = evaluate_formation_controls(*fixtures())
        again, again_generator = evaluate_formation_controls(*fixtures())
        self.assertEqual(report, again)
        self.assertEqual(generator, again_generator)
        json.dumps(dataclasses.asdict(report), allow_nan=False)
        self.assertEqual(dict(report.offline_cost.legacy)["teacher_fits"], 4)
        self.assertEqual(dict(report.offline_cost.legacy)["meta_regression_fits"], 4)
        self.assertEqual(dict(report.offline_cost.legacy)["meta_training_rows"], 16)
        self.assertEqual(dict(report.offline_cost.legacy)["oracle_conditioning_queries"], 5 * 4 * 20)
        self.assertEqual(report.assets.marginal_generator_parameters, 97 * 4)
        self.assertEqual(
            tuple(x.name for x in report.provenance), ("train", "evaluation", "clean_healthy", "noisy_healthy")
        )
        self.assertTrue(all(len(p.sample_ids_sha256) == len(p.pairs_sha256) == 64 for p in report.provenance))
        for cohort in report.cohorts:
            for row in cohort.lineages:
                self.assertEqual(tuple(m.name for m in row.methods), CONTROL_METHODS)
                self.assertEqual(dict(row.cost.legacy)["selection_queries"], (len(CONTROL_METHODS) + 1) * 4)
                self.assertEqual(dict(row.cost.legacy)["final_audit_queries"], len(CONTROL_METHODS) * 2 * 8)
                self.assertEqual(row.cost.analytic4x4_fits, 1)
                self.assertEqual(row.cost.analytic4x4_fit_examples, 20)
                self.assertEqual(row.cost.derived_candidates, 3)
                self.assertEqual(row.cost.negation_transforms, 1)
                paired = next(m for m in row.methods if m.name == "paired")
                self.assertEqual(row.cost.random_direction_draws, 0 if paired.exact_null else 4)
                self.assertEqual(row.cost.norm_rescalings, int(not paired.exact_null))
                for method in row.methods:
                    self.assertEqual(method.candidate_count, 0 if method.name == "no_op" else 1)
                    self.assertEqual(method.no_op, not method.acted)
                    self.assertEqual(method.false_admission, method.acted and method.audit_class != "helpful")
                    self.assertEqual(method.false_rejection, not method.acted and method.audit_class == "helpful")
            for summary in cohort.summary:
                values = [next(m for m in row.methods if m.name == summary.name) for row in cohort.lineages]
                self.assertEqual(summary.acted_lineages, sum(v.acted for v in values))
                self.assertEqual(summary.false_admissions, sum(v.false_admission for v in values))
                self.assertEqual(summary.false_rejections, sum(v.false_rejection for v in values))
                self.assertEqual(summary.no_op_lineages, sum(v.no_op for v in values))
        clean = next(c for c in report.cohorts if c.name == "clean_healthy")
        self.assertTrue(all(not m.acted and m.admitted_audit_mse == 0 for row in clean.lineages for m in row.methods))
        self.assertTrue(report.software_valid)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            report.phase = "confirmation"

    def test_all_cohorts_candidate_banks_precede_selection_and_audit_is_not_admission(self):
        from sultai.formation_controls import control_candidate_bank

        training, evaluation, clean, noisy = fixtures()
        observed = {"banks": 0, "selection": 0, "audit": 0}
        selection_ids = {id(e.selection) for cohort in (evaluation, clean, noisy) for e in cohort}
        audit_ids = {id(e.test) for cohort in (evaluation, clean, noisy) for e in cohort}

        def bank(*args, **kwargs):
            self.assertEqual(observed["selection"], 0)
            self.assertEqual(observed["audit"], 0)
            observed["banks"] += 1
            return control_candidate_bank(*args, **kwargs)

        def measured(adapter, examples):
            self.assertEqual(observed["banks"], 6)
            if id(examples) in selection_ids:
                observed["selection"] += 1
            elif id(examples) in audit_ids:
                observed["audit"] += 1
            else:
                self.fail("unexpected evaluation partition")
            return mse(adapter, examples)

        with (
            patch("sultai.formation_controls.control_candidate_bank", side_effect=bank),
            patch("sultai.formation_controls.mse", side_effect=measured),
        ):
            report, _ = evaluate_formation_controls(training, evaluation, clean, noisy)
        self.assertEqual(observed["selection"], 6 * (len(CONTROL_METHODS) + 1))
        self.assertEqual(observed["audit"], 6 * 2 * len(CONTROL_METHODS))
        poisoned = dataclasses.replace(
            evaluation[0],
            test=tuple(dataclasses.replace(x, target=tuple(t + 90 for t in x.target)) for x in evaluation[0].test),
        )
        changed, _ = evaluate_formation_controls(training, (poisoned, evaluation[1]), clean, noisy)
        for original, altered in zip(
            report.cohorts[0].lineages[0].methods, changed.cohorts[0].lineages[0].methods, strict=True
        ):
            self.assertEqual(original.acted, altered.acted)
            self.assertEqual(original.raw_selection_mse, altered.raw_selection_mse)
            self.assertNotEqual(original.raw_audit_mse, altered.raw_audit_mse)

    def test_public_phase_type_and_value_rejected_without_cohort_execution(self):
        with patch("sultai.formation_controls.make_episode", side_effect=AssertionError("assay ran")):
            for phase in (None, 1, "test", "fixture"):
                with self.subTest(phase=phase), self.assertRaises(ValueError):
                    run_formation_controls(phase)

    def test_phase_dispatch_constructs_only_requested_seed_set_without_running_assay(self):
        # Constructor spies only; no protocol cohort is built or evaluated.
        with (
            patch("sultai.formation_controls.make_episode", return_value="fixture") as maker,
            patch("sultai.formation_controls.make_noisy_healthy_episode", return_value="fixture") as noise,
            patch(
                "sultai.formation_controls.evaluate_formation_controls", return_value=("report", "generator")
            ) as evaluate,
        ):
            self.assertEqual(run_formation_controls("development"), ("report", "generator"))
        self.assertEqual([call.args[0] for call in maker.call_args_list[:32]], list(FIXED.train_seeds))
        self.assertEqual([call.args[0] for call in maker.call_args_list[32:40]], list(range(4000, 4008)))
        self.assertEqual([call.args[0] for call in maker.call_args_list[40:]], list(range(14000, 14008)))
        self.assertEqual([call.args[0] for call in noise.call_args_list], list(range(24000, 24008)))
        self.assertEqual(evaluate.call_args.kwargs["phase"], "development")


if __name__ == "__main__":
    unittest.main()
