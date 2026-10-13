from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import Tensor
from torch.nn import functional as F

from sultai.trust import ContractViolation, InputDataError
from sultai_screen.contracts import PilotHold
from sultai_screen.data import Dataset, generator, normalized
from sultai_screen.fit import fit_sgd, losses, mean_loss, paired_se
from sultai_screen.host import (
    Host,
    cached,
    constant,
    freeze,
    make_host,
    recipe_optimizer,
    state_hash,
    train_epochs,
    train_host,
)
from sultai_screen.m0.calibrate import recalibrate_bn
from sultai_screen.m0.data import features_from_images
from sultai_screen.m0.retune import (
    R272S,
    RAFF,
    RALL,
    RetuneCandidate,
    RetuneGrid,
    Subset,
    complement_hash,
    count,
    fit_retune,
    recalibrate_after,
    retune_grid,
)
from sultai_screen.m0.static import (
    StaticHost,
    load_static,
    save_static,
    static_adapter,
    static_features,
    static_hash,
    static_module,
    train_static,
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
        # C8: BatchNorm running statistics are in the complement: a leak during fitting changes the hash.
        drifted = make_host(3000, 0, CPU)
        with torch.no_grad():
            dict(drifted.named_buffers())["front.1.running_mean"][0] += 1e-3
        self.assertNotEqual(complement_hash(drifted, subset), before)
        # Even when every parameter is in the subset (R-all), the buffers stay in the complement.
        self.assertNotEqual(
            complement_hash(drifted, RALL(drifted)), complement_hash(make_host(3000, 0, CPU), RALL(drifted))
        )

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

        # C9: R-bn re-estimates on the FIT examples (§4), never on selection.
        def reference(indices: Tensor) -> dict[str, Tensor]:
            unsealed = make_host(3000, 0, CPU)
            unsealed.load_state_dict(candidate.host.state_dict())
            return buffers(recalibrate_bn(unsealed, self.data.images, indices))

        on_fit, on_selection = reference(FIT), reference(SELECTION)
        for name, value in buffers(recalibrated.host).items():
            self.assertTrue(torch.equal(value, on_fit[name]), name)
        self.assertTrue(
            any(not torch.equal(value, on_selection[name]) for name, value in buffers(recalibrated.host).items())
        )

    def reference_plateau(self, result: RetuneGrid, data: Dataset, selection: Tensor) -> tuple[bool, float]:
        """Table B recomputed from the grid's own candidates: the 800- vs 3,200-step analogue is checkpoints[-2] vs [-1]."""
        steps = sorted({c.steps for c in result.candidates})
        early, late = (
            min((c for c in result.candidates if c.steps == step and c.selection is not None),
                key=lambda c: c.selection if c.selection is not None else float("inf"))
            for step in steps[-2:]
        )  # fmt: skip
        assert early.selection is not None and late.selection is not None
        difference = losses(early.host, features_from_images(early.host, data, selection), None) - losses(
            late.host, features_from_images(late.host, data, selection), None
        )
        se = paired_se(difference, 11)
        tail = torch.tensor(late.training_losses[-80:], dtype=torch.float64)
        tail_se = float(tail.std(unbiased=True)) / math.sqrt(80)
        change = abs(float(tail[:40].mean()) - float(tail[40:].mean()))
        return early.selection - late.selection < 2 * se and change < 2 * tail_se, se

    def test_retune_plateau_is_certified_by_table_b_at_the_last_two_checkpoints(self) -> None:
        # R32: R-272s and R-aff are certified by the v4 Table B rule on the checkpoints[-2] and [-1] bests (800 vs
        # 3,200 steps in production; 40 vs 80 here, with the real 40-step windows). Uncertified: on the 32-example
        # fixture every step is one full batch, so the training loss still drifts down across the last two windows.
        subset = R272S(self.host)
        drifting = retune_grid(
            self.host, subset, data=self.data, fit_indices=FIT, selection_indices=SELECTION, seed=11,
            rates=(0.1,), checkpoints=(20, 40, 80), from_images=False,
        )  # fmt: skip
        expected, se = self.reference_plateau(drifting, self.data, SELECTION)
        self.assertIs(drifting.plateau, False)
        self.assertIs(expected, False)
        self.assertEqual(drifting.selection_se, se)
        self.assertEqual(drifting.plateau_evidence, "both finite schedule comparators available")
        # Certified: 520 random-label fit examples give 256/256/8 minibatches, whose noise dominates the late windows,
        # and the fit does not move held-out selection by 2 paired SEs.
        rng = torch.Generator().manual_seed(12)
        images = torch.randint(0, 256, (600, 3, 32, 32), generator=rng, dtype=torch.uint8)
        ids = torch.arange(600)
        noisy = Dataset(images, torch.randint(0, 10, (600,), generator=rng), ids, ids.clone(), ids.clone(), ids.clone())
        fit, selection = torch.arange(520), torch.arange(520, 600)
        flat = retune_grid(
            self.host, subset, data=noisy, fit_indices=fit, selection_indices=selection, seed=11, rates=(0.1,),
            checkpoints=(40, 80), from_images=False,
        )  # fmt: skip
        expected, se = self.reference_plateau(flat, noisy, selection)
        self.assertIs(flat.plateau, True)
        self.assertIs(expected, True)
        self.assertEqual(flat.selection_se, se)
        self.assertEqual(flat.report()["plateau_certified"], True)
        # R-all (certify=False, the 40..800 axis) is never certified, whatever its traces.
        never = retune_grid(
            self.host, subset, data=noisy, fit_indices=fit, selection_indices=selection, seed=11, rates=(0.1,),
            checkpoints=(40, 80), from_images=False, certify=False,
        )  # fmt: skip
        self.assertIsNone(never.plateau)
        self.assertEqual(never.plateau_evidence, "R-all is never plateau-certified (§4)")
        self.assertIsNone(never.report()["plateau_certified"])

    def test_retune_grid_best_and_edge(self) -> None:
        subset = R272S(self.host)
        # Two training losses at the last checkpoint: one-step windows let the Table B rule run on this fixture.
        with patch("sultai_screen.fit.PLATEAU_WINDOW", 1):
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
        self.assertIsInstance(result.plateau, bool)
        self.assertEqual(result.plateau_evidence, "both finite schedule comparators available")
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
        self.assertEqual(report["plateau_certified"], result.plateau)
        self.assertEqual(report["selection_paired_bootstrap_se"], result.selection_se)
        self.assertEqual(len(report["candidates"]), 4)

    def test_retune_grid_retains_failed_run_unselected(self) -> None:
        # One-step plateau windows (R32's Table B needs two windows of the late run's two step losses).
        with patch("sultai_screen.fit.PLATEAU_WINDOW", 1):
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


SMALL = (16, 16, 32, 32)


class StaticTests(unittest.TestCase):
    static: StaticHost
    hashes: list[str]
    stream: Tensor
    plain_hashes: list[str]
    plain_stream: Tensor

    @classmethod
    def setUpClass(cls) -> None:
        data = dataset()
        cls.static, cls.hashes, cls.stream, _, _ = train_static(3000, data, CPU, widths=SMALL)
        cls.plain_hashes, cls.plain_stream, _ = train_host(make_host(3000, 0, CPU, widths=SMALL), data, 3000)

    def setUp(self) -> None:
        self.data = dataset()

    def test_construction_is_the_anchor_host_and_the_zero_module_is_identity(self) -> None:
        host = make_host(3000, 0, CPU)
        before = torch.random.get_rng_state()
        static = StaticHost(host)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
        self.assertEqual(state_hash(static.host), state_hash(make_host(3000, 0, CPU)))
        self.assertEqual(tuple(static.w.shape), (16, 16))
        self.assertEqual(tuple(static.b.shape), (16,))
        self.assertFalse(bool(static.w.any()) or bool(static.b.any()))
        plain = make_host(3000, 0, CPU)
        static.eval()
        plain.eval()
        images = normalized(self.data.images[:8])
        with torch.no_grad():
            self.assertTrue(torch.equal(static(images), plain(images)))

    def test_one_step_moves_the_module(self) -> None:
        ids = torch.arange(32)
        data = Dataset(self.data.images, self.data.labels, ids, ids.clone(), ids.clone(), ids.clone())
        static = StaticHost(make_host(3000, 0, CPU, widths=SMALL))
        optimizer = recipe_optimizer(static.parameters())
        _, stream, _ = train_epochs(
            static, static.host, data, 3000, epochs=range(1), rates=constant(0.05), optimizer=optimizer
        )
        self.assertEqual(len(stream), 1)
        self.assertTrue(bool(static.w.detach().ne(0).any()))
        self.assertTrue(bool(static.b.detach().ne(0).any()))

    def test_module_joins_the_recipe_groups(self) -> None:
        static = StaticHost(make_host(3000, 0, CPU, widths=SMALL))
        decay, affine = recipe_optimizer(static.parameters()).param_groups
        self.assertEqual(decay["weight_decay"], 5e-4)
        self.assertEqual(affine["weight_decay"], 0.0)
        self.assertTrue(any(p is static.w for p in decay["params"]))
        self.assertFalse(any(p is static.w for p in affine["params"]))
        self.assertTrue(any(p is static.b for p in affine["params"]))
        self.assertFalse(any(p is static.b for p in decay["params"]))

    def test_training_stream_shares_order_and_differs_with_the_module_in_the_graph(self) -> None:
        self.assertEqual(len(self.stream), 20)
        self.assertEqual(len(self.hashes), 10)
        # Same order/crop/flip draws and an identical step-0 forward: the first loss is bitwise the anchor's.
        self.assertTrue(torch.equal(self.stream[0], self.plain_stream[0]))
        self.assertFalse(torch.equal(self.stream[1:], self.plain_stream[1:]))
        self.assertNotEqual(self.hashes[-1], self.plain_hashes[-1])
        self.assertEqual(self.hashes[-1], state_hash(self.static.host))
        self.assertTrue(self.static.host.sealed)
        self.assertFalse(self.static.training or self.static.w.requires_grad or self.static.b.requires_grad)
        self.assertIsNone(self.static.w.grad)
        self.assertTrue(bool(self.static.w.ne(0).any()))

    def test_save_load_round_trip_and_tampered_module_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = save_static(self.static, root)
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["host.pt", "static-module.pt"])
            self.assertEqual(record["static_sha256"], static_hash(self.static))
            self.assertEqual(record["host_sha256"], state_hash(self.static.host))
            loaded = load_static(root, 3000, SMALL, CPU, static_hash(self.static))
            self.assertEqual(static_hash(loaded), static_hash(self.static))
            self.assertTrue(loaded.host.sealed)
            self.assertFalse(loaded.w.requires_grad or loaded.b.requires_grad)
            module = torch.load(root / "static-module.pt", weights_only=True)
            module["w"] = module["w"].clone()
            module["w"][0, 0] += 1.0
            torch.save(module, root / "static-module.pt")
            with self.assertRaises(PilotHold):
                load_static(root, 3000, SMALL, CPU, static_hash(self.static))
        self.assertNotEqual(static_hash(self.static), state_hash(self.static.host))

    def test_static_module_boundary_rejects_missing_keys_wrong_shape_and_nonfinite(self) -> None:
        good = {"w": torch.ones(16, 16), "b": torch.ones(16)}
        checked = static_module(good)
        self.assertTrue(torch.equal(checked["w"], good["w"]) and torch.equal(checked["b"], good["b"]))
        nan = torch.ones(16)
        nan[3] = float("nan")
        inf = torch.ones(16, 16)
        inf[2, 2] = float("inf")
        for bad in (
            [torch.ones(16, 16), torch.ones(16)],
            {"w": torch.ones(16, 16)},
            {"w": torch.ones(16, 16), "b": torch.ones(16), "c": torch.ones(1)},
            {"w": torch.ones(16, 15), "b": torch.ones(16)},
            {"w": torch.ones(16, 16), "b": torch.ones(16, 1)},
            {"w": torch.ones(16, 16, dtype=torch.float64), "b": torch.ones(16)},
            {"w": torch.ones(16, 16), "b": nan},
            {"w": inf, "b": torch.ones(16)},
            {"w": torch.ones(16, 16), "b": [1.0] * 16},
        ):
            with self.assertRaises(InputDataError):
                static_module(bad)

    def test_no_refit_term_is_the_complete_model(self) -> None:
        development = cached(self.static.host, self.data).development
        adapter = static_adapter(self.static)
        self.assertTrue(adapter.nonlinear)
        self.assertTrue(
            torch.equal(adapter.weights[0], self.static.w) and torch.equal(adapter.weights[1], self.static.b)
        )
        complete = losses(self.static.host, development, adapter)
        self.assertFalse(torch.equal(complete, losses(self.static.host, development, None)))
        features = static_features(self.static, self.data, self.data.development)
        self.assertTrue(torch.equal(losses(self.static.host, features, adapter), complete))
        with torch.no_grad():
            direct = F.cross_entropy(
                self.static(normalized(self.data.images[self.data.development])),
                self.data.labels[self.data.development],
                reduction="none",
            )
        self.assertTrue(torch.allclose(direct, complete, atol=1e-5))

    def test_refit_warm_starts_from_the_learned_module(self) -> None:
        features = static_features(self.static, self.data, SELECTION)
        adapter = static_adapter(self.static)
        refit = fit_sgd(self.static.host, features, features, seed=11, rate=0.1, steps=1, minibatch=True, start=adapter)
        self.assertEqual(refit.step0_selection, mean_loss(self.static.host, features, adapter))
        self.assertNotEqual(refit.step0_selection, mean_loss(self.static.host, features, None))


if __name__ == "__main__":
    unittest.main()
