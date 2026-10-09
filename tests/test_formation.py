import dataclasses
import json
import unittest
from unittest.mock import patch

from report_fixture import reference_report, validated_splits
from sultai.formation import (
    FIXED,
    _evaluate,
    admit,
    candidate_bank,
    conditioning_evidence,
    formation_acceptance,
    guard_lineage_splits,
    make_episode,
    materialize,
    mismatched_residuals,
    public_templates,
    run_formation,
    train_generators,
)
from sultai.repair import Adapter, Episode, Example, fit, mse


class FormationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = tuple(make_episode(seed) for seed in FIXED.train_seeds)
        cls.generators, cls.bank, cls.cost = train_generators(cls.training)

    def test_honest_shared_family_and_disjoint_lineages_samples(self):
        splits = {
            "train": self.training,
            "dev": tuple(make_episode(s) for s in FIXED.dev_seeds),
            "test": tuple(make_episode(s) for s in FIXED.test_seeds),
        }
        guard_lineage_splits(splits)
        self.assertEqual(len({e.bottleneck_family for es in splits.values() for e in es}), 1)
        with self.assertRaises(ValueError):
            guard_lineage_splits({"train": self.training, "test": (self.training[0],)})
        e = self.training[0]
        duplicate = Episode("another-lineage", e.bottleneck_family, e.conditioning, e.selection, e.test)
        with self.assertRaises(ValueError):
            guard_lineage_splits({"train": (e,), "test": (duplicate,)})

    def test_only_conditioning_and_numeric_values_are_evidence(self):
        e = make_episode(FIXED.test_seeds[0])
        g = self.generators["paired"]
        changed_ids = tuple(Example("arbitrary-" + str(i), x.h, x.target) for i, x in enumerate(e.conditioning))
        self.assertEqual(g.form(e.conditioning), g.form(changed_ids))
        poisoned = Episode(
            e.host_lineage,
            e.bottleneck_family,
            e.conditioning,
            tuple(Example(x.sample_id, x.h, tuple(v + 900 for v in x.target)) for x in e.selection),
            tuple(Example(x.sample_id, x.h, tuple(v - 900 for v in x.target)) for x in e.test),
        )
        self.assertEqual(g.form(e.conditioning), g.form(poisoned.conditioning))
        altered = tuple(Example(x.sample_id, x.h, Adapter.zero().apply(x.h)) for x in e.conditioning)
        self.assertNotEqual(g.form(e.conditioning), g.form(altered))

    def test_training_cannot_read_selection_or_test(self):
        class ConditioningOnly:
            def __init__(self, episode):
                self.conditioning = episode.conditioning

            @property
            def selection(self):
                raise AssertionError("selection leaked into training")

            @property
            def test(self):
                raise AssertionError("test leaked into training")

        generators, _, _ = train_generators(tuple(ConditioningOnly(e) for e in self.training))
        self.assertEqual(generators, self.generators)

    def test_frozen_generator_and_fixture_immutability(self):
        e = make_episode(445)
        original = repr(e)
        g = self.generators["paired"]
        before = repr(g)
        g.form(e.conditioning)
        self.assertEqual(repr(e), original)
        self.assertEqual(repr(g), before)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            g.weights = ()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            FIXED.meta_ridge = 1.0

    def test_templates_are_full_stored_adapters_and_nonlinear(self):
        templates = public_templates()
        self.assertEqual(len(templates), 4)
        self.assertTrue(all(sum(len(row) for row in a.weights) + len(a.bias) == 272 for a in templates))
        adapter = materialize((0.3, 0.1, -0.2, 0.05))
        x = (0.4,) * 16
        y = (1.2,) * 16
        mid = tuple((a + b) / 2 for a, b in zip(x, y, strict=True))
        self.assertGreater(
            sum(
                abs(p - (a + b) / 2)
                for p, a, b in zip(adapter.apply(mid), adapter.apply(x), adapter.apply(y), strict=True)
            ),
            0.01,
        )

    def test_admission_ties_and_healthy_prefer_noop(self):
        e = make_episode(445, healthy=True)
        for g in self.generators.values():
            decision = admit(g.form(e.conditioning), e.selection)
            self.assertFalse(decision.acted)
            self.assertEqual(decision.adapter, Adapter.zero())
        decision = admit(Adapter.zero(), e.selection)
        self.assertFalse(decision.acted)
        self.assertEqual(decision.selection_queries, 48)

    def test_identity_probes_execute_without_added_information(self):
        e = make_episode(445)
        with patch("sultai.formation.FrozenHost.output", autospec=True, side_effect=lambda host, h: tuple(h)) as output:
            probes = conditioning_evidence(e.conditioning, "paired_probes")
            self.assertEqual(output.call_count, 32)
        self.assertEqual(probes, conditioning_evidence(e.conditioning, "paired"))
        self.assertEqual(
            self.generators["paired"].form(e.conditioning), self.generators["paired_probes"].form(e.conditioning)
        )

    def test_known_positive_without_new_host_fitting(self):
        e = make_episode(445)
        with patch("sultai.formation.fit", side_effect=AssertionError("new host fit")):
            candidate = self.generators["paired"].form(e.conditioning)
        self.assertLess(mse(candidate, e.test), mse(Adapter.zero(), e.test))
        self.assertLess(mse(fit(e.conditioning), e.test), 1e-12)

    def test_failed_full_ridge_audit_rejects_gate_independently(self):
        report = reference_report()["formation"]
        test = report["test"]
        gates = formation_acceptance(report["development"], test, report["healthy"], isolation=validated_splits())
        self.assertTrue(gates["full_ridge_representability"])
        test["lineages"][1]["methods"]["full_ridge"]["raw_test_mse"] = 1e-4
        test["summary"]["full_ridge"]["raw_mean_mse"] = sum(
            row["methods"]["full_ridge"]["raw_test_mse"] for row in test["lineages"]
        ) / len(test["lineages"])
        gates = formation_acceptance(report["development"], test, report["healthy"], isolation=validated_splits())
        self.assertFalse(gates["full_ridge_representability"])
        self.assertTrue(gates["paired_reduces_test_mean_mse"])
        self.assertTrue(gates["healthy_all_no_op"])
        self.assertTrue(gates["finite_metrics"])

    def test_real_selection_test_order_and_executed_query_accounting(self):
        e = make_episode(446)
        other = make_episode(447)
        observed = {"selection": 0, "audit": 0, "queries": 0, "banks": 0}

        class ProtectedAudit:
            host_lineage = e.host_lineage
            bottleneck_family = e.bottleneck_family
            conditioning = e.conditioning
            selection = e.selection

            @property
            def test(self):
                if observed["selection"] != 14:
                    raise AssertionError("audit accessed before complete admission")
                return e.test

        def measured_mse(adapter, examples):
            self.assertEqual(observed["banks"], 2, "all phase banks must precede selection")
            if examples is e.selection or examples is other.selection:
                observed["selection"] += 1
            elif examples is e.test or examples is other.test:
                observed["audit"] += 1
            else:
                raise AssertionError("unexpected evaluation examples")
            observed["queries"] += len(examples)
            return mse(adapter, examples)

        def measured_bank(*args, **kwargs):
            self.assertEqual(observed["selection"], 0)
            self.assertEqual(observed["audit"], 0)
            observed["banks"] += 1
            return candidate_bank(*args, **kwargs)

        with (
            patch("sultai.formation.mse", side_effect=measured_mse),
            patch("sultai.formation.candidate_bank", side_effect=measured_bank),
        ):
            report = _evaluate((ProtectedAudit(), other), self.generators, self.bank)
        self.assertEqual(observed["selection"], 28)
        self.assertEqual(observed["audit"], 52)
        self.assertEqual(
            observed["queries"], report["costs"]["selection_queries"] + report["costs"]["final_audit_queries"]
        )
        poisoned = Episode(
            e.host_lineage,
            e.bottleneck_family,
            e.conditioning,
            e.selection,
            tuple(Example(x.sample_id, x.h, tuple(v + 900 for v in x.target)) for x in e.test),
        )
        poisoned_report = _evaluate((poisoned, other), self.generators, self.bank)
        for name, original in report["lineages"][0]["methods"].items():
            changed = poisoned_report["lineages"][0]["methods"][name]
            self.assertEqual(original["acted"], changed["acted"])
            self.assertEqual(original["raw_selection_mse"], changed["raw_selection_mse"])
            self.assertNotEqual(original["raw_test_mse"], changed["raw_test_mse"])

    def test_conditioning_ablation_deranges_lineages_and_mispair_preserves_marginals(self):
        e, donor = make_episode(448), make_episode(449)
        bank = candidate_bank(e.conditioning, self.generators, self.bank, shuffled_conditioning=donor.conditioning)
        shuffled = next(c.adapter for c in bank if c.name == "shuffled_conditioning")
        self.assertEqual(shuffled, self.generators["paired"].form(donor.conditioning))
        self.assertNotEqual(shuffled, self.generators["paired"].form(e.conditioning))
        mismatched = mismatched_residuals(e.conditioning)
        self.assertEqual(tuple(x.h for x in mismatched), tuple(x.h for x in e.conditioning))
        for a, b in zip(
            conditioning_evidence(e.conditioning, "coarse"), conditioning_evidence(mismatched, "coarse"), strict=True
        ):
            self.assertAlmostEqual(a, b, places=12)
        self.assertNotEqual(
            conditioning_evidence(e.conditioning, "paired"), conditioning_evidence(mismatched, "paired")
        )

    def test_report_reproducibility_accounting_and_all_outcomes(self):
        report, g = run_formation()
        again, g_again = run_formation()
        self.assertEqual(report, again)
        self.assertEqual(g, g_again)
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["config"]["lineages"], {"train": 32, "development": 8, "test": 8})
        self.assertEqual(report["costs"]["offline"]["teacher_fits"], 32)
        self.assertEqual(report["costs"]["offline"]["meta_regression_fits"], 3)
        self.assertEqual(report["costs"]["offline"]["identity_probe_queries"], 1024)
        for phase in ("development", "test", "healthy"):
            self.assertEqual(len(report[phase]["lineages"]), 8)
            for row in report[phase]["lineages"]:
                self.assertEqual(len(row["methods"]), 13)
                self.assertEqual(row["costs"]["selection_queries"], 14 * 24)
                self.assertEqual(row["costs"]["final_audit_queries"], 2 * 13 * 64)
                self.assertEqual(row["costs"]["sgd_example_steps"], 7 * 64)
                self.assertEqual(row["costs"]["retrieval_distance_evaluations"], 64)
                for name, values in row["methods"].items():
                    self.assertIn("raw_test_mse", values)
                    self.assertIn("admitted_test_mse", values)
                    self.assertIn("realized_harm", values)
                    self.assertIn("costs", values)
                    self.assertEqual(values["candidate_count"], 0 if name == "no_op" else 1)
        self.assertTrue(report["acceptance"]["paired_reduces_test_mean_mse"])
        self.assertTrue(report["acceptance"]["full_ridge_representability"])
        self.assertTrue(report["acceptance"]["healthy_all_no_op"])


if __name__ == "__main__":
    unittest.main()
