import math
import unittest
from dataclasses import FrozenInstanceError, replace

from sultai.repair import (
    CHANNELS,
    PARAMETERS,
    Adapter,
    Example,
    FrozenHost,
    ShadowProbe,
    feature_rank,
    fit,
    guard_group_splits,
    mse,
    select,
    synthetic_episode,
)
from sultai.smoke import RIDGES, run


class AdapterTests(unittest.TestCase):
    def test_shape_parameter_count_noop_and_blend_zero(self):
        zero = Adapter.zero()
        self.assertEqual(sum(map(len, zero.weights)) + len(zero.bias), PARAMETERS)
        self.assertEqual(PARAMETERS, 272)
        h = tuple(i / 10 for i in range(CHANNELS))
        nonzero = Adapter(zero.weights, (1.0,) * CHANNELS)
        self.assertEqual(zero.apply(h), h)
        self.assertEqual(nonzero.apply(h, alpha=0), h)
        self.assertEqual(len(nonzero.apply(h)), CHANNELS)

    def test_reject_malformed_nonfinite_and_output_overflow(self):
        zero = Adapter.zero()
        for h in [(0.0,) * 15, (0.0,) * 17, (math.nan,) * 16, (math.inf,) * 16, (True,) * 16]:
            with self.assertRaises(ValueError):
                zero.apply(h)
        for alpha in [math.nan, math.inf, True]:
            with self.assertRaises(ValueError):
                zero.apply((0.0,) * 16, alpha)
        for weights, bias in [
            (zero.weights[:-1], zero.bias),
            (((0.0,) * 15,) * 16, zero.bias),
            (zero.weights, (math.nan,) * 16),
            (((math.inf,) * 16,) * 16, zero.bias),
        ]:
            with self.assertRaises(ValueError):
                Adapter(weights, bias)
        huge = Adapter(zero.weights, (1e308,) * 16)
        with self.assertRaises(ValueError):
            huge.apply((0.0,) * 16, alpha=2)

    def test_serialization_determinism_and_nonaffinity(self):
        # f(2x)-2f(x)+f(0) is nonzero for tanh, even with residual h.
        adapter = Adapter(tuple(tuple(float(i == j) for j in range(16)) for i in range(16)), (0.0,) * 16)
        x = (0.6,) * 16
        result = adapter.apply(x)
        self.assertEqual(result, adapter.apply(x))
        self.assertEqual(adapter, Adapter.from_json(adapter.to_json()))
        self.assertEqual(result, Adapter.from_json(adapter.to_json()).apply(x))
        delta = adapter.apply((1.2,) * 16)[0] - 2 * result[0] + adapter.apply((0.0,) * 16)[0]
        self.assertGreater(abs(delta), 0.2)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.episode = synthetic_episode()

    def test_fixture_is_frozen_and_samples_are_disjoint(self):
        before = repr(self.episode)
        host = FrozenHost()
        h = self.episode.conditioning[0].h
        with self.assertRaises(FrozenInstanceError):
            host.site_id = "changed"
        with self.assertRaises(FrozenInstanceError):
            self.episode.conditioning[0].target = h
        fit(self.episode.conditioning)
        self.assertEqual(before, repr(self.episode))
        self.assertEqual(host.output(h), h)
        self.assertEqual(self.episode, synthetic_episode())
        with self.assertRaises(ValueError):
            replace(self.episode, selection=self.episode.conditioning[:1])
        with self.assertRaises(ValueError):
            replace(self.episode, conditioning=(self.episode.conditioning[0],) * 2)

    def test_outer_guard_rejects_either_group_and_transitive_overlap(self):
        a = self.episode
        b = replace(synthetic_episode(8), host_lineage=a.host_lineage)
        c = replace(synthetic_episode(9), bottleneck_family=b.bottleneck_family)
        for splits in [
            {"train": [a], "validation": [b]},
            {"train": [b], "test": [c]},
            {"train": [a], "validation": [b], "test": [c]},
        ]:
            with self.assertRaises(ValueError):
                guard_group_splits(splits)
        guard_group_splits({"train": [a], "validation": [synthetic_episode(8)], "test": [synthetic_episode(9)]})

    def test_selection_ignores_opposing_test_ranking_and_accounts_queries(self):
        zero = Adapter.zero()
        one = Adapter(zero.weights, (1.0,) * 16)
        h = (0.0,) * 16
        selection = (Example("s", h, h), Example("s2", h, h))
        test = (Example("t", h, (1.0,) * 16),)
        chosen = select((zero, one), selection)
        self.assertEqual(chosen.candidate_index, 0)
        self.assertLess(mse(one, test), mse(chosen.adapter, test))
        self.assertEqual(chosen.candidates, 2)
        self.assertEqual(chosen.selection_queries, 4)
        self.assertEqual(select((zero, zero), selection).candidate_index, 0)

    def test_changing_test_cannot_change_fit_or_selection(self):
        candidates = tuple(fit(self.episode.conditioning, ridge) for ridge in RIDGES)
        changed = replace(self.episode, test=tuple(replace(item, target=(123.0,) * 16) for item in self.episode.test))
        changed_candidates = tuple(fit(changed.conditioning, ridge) for ridge in RIDGES)
        self.assertEqual(candidates, changed_candidates)
        self.assertEqual(select(candidates, self.episode.selection), select(changed_candidates, changed.selection))
        self.assertNotEqual(mse(candidates[0], self.episode.test), mse(candidates[0], changed.test))

    def test_changing_selection_cannot_change_preselection_candidate_bank(self):
        changed = replace(
            self.episode, selection=tuple(replace(item, target=(-200.0,) * 16) for item in self.episode.selection)
        )
        self.assertEqual(
            tuple(fit(self.episode.conditioning, ridge) for ridge in RIDGES),
            tuple(fit(changed.conditioning, ridge) for ridge in RIDGES),
        )

    def test_shuffled_and_mismatched_pairs_change_fitting_evidence(self):
        examples = self.episode.conditioning
        shuffled = tuple(
            replace(item, target=examples[(i + 1) % len(examples)].target) for i, item in enumerate(examples)
        )
        other = synthetic_episode(8).conditioning
        mismatched = tuple(replace(item, target=other[i].target) for i, item in enumerate(examples))
        correct = fit(examples)
        for corrupted in (shuffled, mismatched):
            fitted = fit(corrupted)
            self.assertNotEqual(correct, fitted)
            self.assertGreater(mse(fitted, self.episode.test), mse(correct, self.episode.test) + 0.01)


class InstrumentTests(unittest.TestCase):
    def test_planted_repair_and_healthy_no_repair_controls(self):
        report = run()
        self.assertEqual(report, run())
        positive = report["positive_planted"]
        self.assertGreater(positive["test_mse"]["no_repair"], 0.01)
        self.assertLess(positive["test_mse"]["best_of_k"], 1e-12)
        self.assertEqual(positive["selection_queries"], 3 * 24)
        self.assertEqual(positive["test_queries"], 3 * 32)
        self.assertEqual(positive["conditioning_feature_rank"], 17)
        healthy = report["healthy_no_repair_control"]
        self.assertEqual(set(healthy["test_mse"].values()), {0.0})

    def test_rank_deficient_conditioning_is_finite_with_ridge(self):
        examples = (Example("a", (0.0,) * 16, (1.0,) * 16),)
        self.assertEqual(feature_rank(examples), 1)
        self.assertTrue(all(math.isfinite(v) for v in fit(examples).apply((0.0,) * 16)))
        for ridge in [0.0, -1.0, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                fit(examples, ridge)
        for call in [
            lambda: fit(()),
            lambda: mse(Adapter.zero(), ()),
            lambda: select((), examples),
            lambda: select((Adapter.zero(),), ()),
        ]:
            with self.assertRaises(ValueError):
                call()

    def test_fixed_shadow_probe_disambiguates_identical_passive_evidence(self):
        a, b = ShadowProbe(1), ShadowProbe(-1)
        self.assertEqual(a.observe(), b.observe())
        self.assertEqual(a.observe(), (0.0, 0.0, 1.0))
        self.assertEqual(a.observe(0.25)[1], 0.25)
        self.assertEqual(b.observe(0.25)[1], -0.25)
        self.assertEqual(a.observe(1)[1], a.target)
        self.assertEqual(b.observe(-1)[1], b.target)
        self.assertEqual(ShadowProbe(0).observe(0.25)[1], 0.0)
        healthy = ShadowProbe(1, offset=1)
        self.assertEqual(healthy.observe()[1], healthy.target)
        with self.assertRaises(ValueError):
            a.observe(math.nan)


if __name__ == "__main__":
    unittest.main()
