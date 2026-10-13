from __future__ import annotations

import unittest

import torch
from torch import Tensor
from torch.nn import functional as F

from sultai.trust import ContractViolation
from sultai_screen.data import Dataset, generator, normalized
from sultai_screen.host import Host, freeze, make_host, state_hash
from sultai_screen.m0.retune import (
    R272S,
    RAFF,
    RALL,
    RetuneCandidate,
    Subset,
    complement_hash,
    count,
    fit_retune,
    recalibrate_after,
    retune_grid,
)

CPU = torch.device("cpu")
FIT = torch.arange(32)
SELECTION = torch.arange(32, 64)
AFFINE = ("front.8.weight", "front.8.bias", "tail.1.weight", "tail.1.bias", "classifier.bias")


def dataset() -> Dataset:
    rng = torch.Generator().manual_seed(10)
    images = torch.randint(0, 256, (64, 3, 32, 32), generator=rng, dtype=torch.uint8)
    labels = torch.randint(0, 10, (64,), generator=rng)
    ids = torch.arange(64)
    return Dataset(images, labels, ids, ids.clone(), ids.clone(), ids.clone())


def frozen_host() -> Host:
    host = make_host(3000, 0, CPU)
    freeze(host)
    return host


def parameters(host: Host) -> dict[str, Tensor]:
    return {name: p.detach().clone() for name, p in host.named_parameters()}


def buffers(host: Host) -> dict[str, Tensor]:
    return {name: b.detach().clone() for name, b in host.named_buffers()}


def first_step_gradients(host: Host, data: Dataset, subset: Subset, seed: int) -> dict[str, Tensor]:
    """Gradients of fit_retune's first minibatch loss on an unfrozen eval-mode clone, for every subset tensor."""
    clone = make_host(3000, 0, CPU)
    clone.load_state_dict(host.state_dict())
    clone.eval()
    named = dict(clone.named_parameters())
    order = torch.randperm(len(FIT), generator=generator(seed, "oracle-order"))
    ids = FIT[order[:256]]
    loss = F.cross_entropy(clone(normalized(data.images[ids])), data.labels[ids])
    gradients = torch.autograd.grad(loss, [named[name] for name in subset])
    return dict(zip(subset, gradients, strict=True))


class RetuneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = dataset()
        self.host = frozen_host()

    def assert_movable_entries_moved(self, subset: Subset, before: dict[str, Tensor], after: dict[str, Tensor]) -> None:
        """Every declared entry moved unless its first step could not move it in float32.

        On the random fixture dead ReLUs give exact-zero gradients and some gradients are ~1e-8, so 0.1 * g is
        below half an ulp of the weight; any other unmoved entry fails.
        """
        gradients = first_step_gradients(self.host, self.data, subset, 11)
        moved = 0
        for name, mask in subset.items():
            changed = after[name] != before[name]
            stuck = mask & ~changed
            self.assertTrue(torch.equal((before[name] - 0.1 * gradients[name])[stuck], before[name][stuck]), name)
            moved += int((mask & changed).sum())
        self.assertGreater(moved, count(subset) // 2)

    def run_arm(self, subset: Subset, *, from_images: bool, steps: int = 3, rate: float = 0.1) -> RetuneCandidate:
        return fit_retune(
            self.host,
            subset,
            data=self.data,
            fit_indices=FIT,
            selection_indices=SELECTION,
            seed=11,
            rate=rate,
            steps=steps,
            checkpoints=(),
            from_images=from_images,
        )

    def test_declared_counts(self) -> None:
        self.assertEqual(count(R272S(self.host)), 272)
        self.assertEqual(count(RAFF(self.host)), 170)
        self.assertEqual(count(RALL(self.host)), 284_634)
        subset = R272S(self.host)
        self.assertEqual(set(subset), {"tail.0.weight", "tail.0.bias"})
        self.assertEqual(int(subset["tail.0.weight"][0:16, :, 1, 1].sum()), 256)
        self.assertEqual(int(subset["tail.0.bias"][0:16].sum()), 16)

    def test_empty_or_malformed_subset_raises(self) -> None:
        empty = {"tail.0.bias": torch.zeros(64, dtype=torch.bool)}
        with self.assertRaises(ContractViolation):
            count(empty)
        with self.assertRaises(ContractViolation):
            self.run_arm(empty, from_images=True)
        with self.assertRaises(ContractViolation):
            complement_hash(self.host, {"tail.0.bias": torch.ones(63, dtype=torch.bool)})
        with self.assertRaises(ContractViolation):
            complement_hash(self.host, {"nowhere": torch.ones(1, dtype=torch.bool)})
        with self.assertRaises(ContractViolation):
            complement_hash(self.host, {"tail.0.bias": torch.ones(64)})
        with self.assertRaises(ContractViolation):
            self.run_arm(RAFF(self.host), from_images=False)

    def test_complement_hash_sees_only_the_complement(self) -> None:
        subset = R272S(self.host)
        before = complement_hash(self.host, subset)
        inside, outside = make_host(3000, 0, CPU), make_host(3000, 0, CPU)
        with torch.no_grad():
            inside.tail[0].weight[3, 5, 1, 1] += 1.0
            outside.tail[0].weight[3, 5, 0, 1] += 1.0
        self.assertEqual(complement_hash(make_host(3000, 0, CPU), subset), before)
        self.assertEqual(complement_hash(inside, subset), before)
        self.assertNotEqual(complement_hash(outside, subset), before)
        self.assertNotEqual(complement_hash(self.host, subset), state_hash(self.host))

    def test_r272s_changes_declared_entries_only(self) -> None:
        subset = R272S(self.host)
        original = state_hash(self.host)
        before = parameters(self.host)
        candidate = self.run_arm(subset, from_images=False)
        self.assertEqual(state_hash(self.host), original)
        self.assertTrue(candidate.host.sealed)
        self.assertEqual(candidate.complement, complement_hash(self.host, subset))
        self.assertEqual(complement_hash(candidate.host, subset), complement_hash(self.host, subset))
        after = parameters(candidate.host)
        self.assert_movable_entries_moved(subset, before, after)
        self.assertEqual(candidate.parameter_steps, 272 * 3)
        self.assertEqual(candidate.example_passes, 32 * 3)
        self.assertEqual(len(candidate.training_losses), 3)
        self.assertIsNotNone(candidate.selection)
        self.assertEqual(buffers(candidate.host).keys(), buffers(self.host).keys())
        for name, value in buffers(self.host).items():
            self.assertTrue(torch.equal(buffers(candidate.host)[name], value), name)

    def test_raff_from_images_trains_affines_and_leaves_convolutions(self) -> None:
        subset = RAFF(self.host)
        before = parameters(self.host)
        before_buffers = buffers(self.host)
        candidate = self.run_arm(subset, from_images=True)
        self.assertEqual(complement_hash(candidate.host, subset), complement_hash(self.host, subset))
        after = parameters(candidate.host)
        for name in AFFINE:
            self.assertFalse(torch.equal(after[name], before[name]), name)
        self.assert_movable_entries_moved(subset, before, after)
        for name, value in before.items():
            if value.ndim == 4 or name == "classifier.weight":
                self.assertTrue(torch.equal(after[name], value), name)
        for name, value in before_buffers.items():
            self.assertTrue(torch.equal(buffers(candidate.host)[name], value), name)
        self.assertEqual(candidate.parameter_steps, 170 * 3)

    def test_one_step_is_independent_masked_sgd(self) -> None:
        for subset, from_images in ((R272S(self.host), False), (RAFF(self.host), True)):
            before = parameters(self.host)
            gradients = first_step_gradients(self.host, self.data, subset, 11)
            after = parameters(self.run_arm(subset, from_images=from_images, steps=1).host)
            for name, value in before.items():
                expected = value
                if name in subset:
                    expected = torch.where(subset[name], value - 0.1 * gradients[name], value)
                self.assertTrue(torch.equal(after[name], expected), name)

    def test_cached_site_equals_from_images_for_one_step(self) -> None:
        subset = R272S(self.host)
        cached = self.run_arm(subset, from_images=False, steps=1)
        images = self.run_arm(subset, from_images=True, steps=1)
        self.assertEqual(cached.training_losses, images.training_losses)
        self.assertEqual(state_hash(cached.host), state_hash(images.host))
        self.assertEqual(cached.selection, images.selection)

    def test_checkpoint_snapshots_match_separate_runs(self) -> None:
        subset = R272S(self.host)
        run = fit_retune(
            self.host,
            subset,
            data=self.data,
            fit_indices=FIT,
            selection_indices=SELECTION,
            seed=11,
            rate=0.1,
            steps=3,
            checkpoints=(1, 3),
            from_images=False,
        )
        short = self.run_arm(subset, from_images=False, steps=1)
        self.assertEqual([step for step, _, _ in run.snapshots], [1, 3])
        self.assertEqual(run.snapshots[0][1], state_hash(short.host))
        self.assertEqual(run.snapshots[0][2], short.selection)
        self.assertEqual(state_hash(run.snapshot_hosts[0]), run.snapshots[0][1])
        self.assertEqual(run.snapshots[1][1], state_hash(run.host))
        self.assertEqual(run.snapshots[1][2], run.selection)
        report = run.report()
        self.assertEqual(report["parameter_steps"], 272 * 3)
        self.assertEqual(report["complement_sha256"], run.complement)

    def test_nonfinite_run_is_retained_unselected(self) -> None:
        candidate = self.run_arm(R272S(self.host), from_images=False, rate=1e300)
        self.assertIsNone(candidate.selection)
        self.assertTrue(candidate.host.sealed)
        self.assertEqual(candidate.complement, complement_hash(self.host, R272S(self.host)))
        self.assertFalse(candidate.report()["finite"])

    def test_recalibrate_after_changes_only_batchnorm_buffers(self) -> None:
        candidate = self.run_arm(RALL(self.host), from_images=True, steps=2, rate=0.01)
        self.assertIsNotNone(candidate.selection)
        recalibrated = recalibrate_after(candidate, self.data, FIT, SELECTION)
        self.assertTrue(recalibrated.host.sealed)
        for name, value in parameters(candidate.host).items():
            self.assertTrue(torch.equal(parameters(recalibrated.host)[name], value), name)
        changed = [
            name
            for name, value in buffers(candidate.host).items()
            if not torch.equal(buffers(recalibrated.host)[name], value)
        ]
        self.assertTrue(changed)
        self.assertTrue(all("running_" in name or "num_batches_tracked" in name for name in changed))
        self.assertEqual(recalibrated.pre_recalibration_selection, candidate.selection)
        self.assertEqual(recalibrated.pre_recalibration_hash, state_hash(candidate.host))
        self.assertNotEqual(recalibrated.selection, candidate.selection)
        self.assertNotEqual(state_hash(recalibrated.host), state_hash(candidate.host))
        report = recalibrated.report()
        self.assertEqual(report["pre_recalibration_selection_loss"], candidate.selection)
        self.assertNotIn("pre_recalibration_selection_loss", candidate.report())

    def test_retune_grid_best_and_edge(self) -> None:
        subset = R272S(self.host)
        result = retune_grid(
            self.host,
            subset,
            data=self.data,
            fit_indices=FIT,
            selection_indices=SELECTION,
            seed=11,
            rates=(0.3, 0.1),
            checkpoints=(1, 2),
            from_images=False,
        )
        self.assertEqual([(c.rate, c.steps) for c in result.candidates], [(0.3, 1), (0.3, 2), (0.1, 1), (0.1, 2)])
        finite = [c for c in result.candidates if c.selection is not None]
        best = min(finite, key=lambda c: c.selection if c.selection is not None else float("inf"))
        self.assertIs(result.best, best)
        self.assertTrue(result.edge_of_grid)
        self.assertIsNone(result.plateau)
        self.assertEqual(result.plateau_evidence, "not applied to re-tune arms")
        for candidate in result.candidates:
            self.assertEqual(candidate.parameter_steps, 272 * candidate.steps)
            self.assertEqual(candidate.complement, complement_hash(self.host, subset))
            if candidate.selection is not None:
                self.assertEqual(len(candidate.training_losses), candidate.steps)
        separate = self.run_arm(subset, from_images=False, steps=1, rate=0.3)
        self.assertEqual(state_hash(result.candidates[0].host), state_hash(separate.host))
        self.assertEqual(result.candidates[0].selection, separate.selection)
        self.assertEqual(result.candidates[0].wall_seconds, 0.0)
        report = result.report()
        self.assertIsNone(report["plateau_certified"])
        self.assertEqual(len(report["candidates"]), 4)

    def test_retune_grid_retains_failed_run_unselected(self) -> None:
        result = retune_grid(
            self.host,
            R272S(self.host),
            data=self.data,
            fit_indices=FIT,
            selection_indices=SELECTION,
            seed=11,
            rates=(1e300, 0.1),
            checkpoints=(1, 2),
            from_images=False,
        )
        failed = [c for c in result.candidates if c.rate == 1e300]
        self.assertEqual([c.steps for c in failed], [1, 2])
        self.assertTrue(all(c.selection is None for c in failed))
        self.assertEqual([c.wall_seconds == 0.0 for c in failed], [True, False])
        self.assertTrue(all(c.example_passes == 32 and c.parameter_steps == 272 for c in failed))
        self.assertEqual(result.best.rate, 0.1)
        self.assertFalse(result.report()["candidates"][0]["finite"])


if __name__ == "__main__":
    unittest.main()
