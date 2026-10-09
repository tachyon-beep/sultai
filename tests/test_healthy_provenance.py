"""Legacy entry point must inspect healthy controls before any fitting."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from sultai import formation


class HealthyProvenanceTests(unittest.TestCase):
    def test_healthy_training_sample_collision_refused_before_fit(self):
        original = formation.make_episode
        training_id = original(formation.FIXED.train_seeds[0]).conditioning[0].sample_id

        def fixture(seed, *, healthy=False):
            episode = original(seed, healthy=healthy)
            if healthy and seed == formation.FIXED.test_seeds[0]:
                collided = replace(episode.conditioning[0], sample_id=training_id)
                return replace(episode, conditioning=(collided, *episode.conditioning[1:]))
            return episode

        with (
            patch.object(formation, "make_episode", side_effect=fixture),
            patch.object(
                formation, "train_generators", side_effect=AssertionError("fitting reached before healthy isolation")
            ) as fitting,
        ):
            with self.assertRaisesRegex(ValueError, "sample ID overlap"):
                formation.run_formation()
            fitting.assert_not_called()

    def test_healthy_training_lineage_collision_refused_before_fit(self):
        original = formation.make_episode
        lineage = original(formation.FIXED.train_seeds[0]).host_lineage

        def fixture(seed, *, healthy=False):
            episode = original(seed, healthy=healthy)
            if healthy and seed == formation.FIXED.test_seeds[0]:
                return replace(episode, host_lineage=lineage)
            return episode

        with (
            patch.object(formation, "make_episode", side_effect=fixture),
            patch.object(
                formation, "train_generators", side_effect=AssertionError("fitting reached before healthy isolation")
            ) as fitting,
        ):
            with self.assertRaisesRegex(ValueError, "host lineage overlap"):
                formation.run_formation()
            fitting.assert_not_called()
