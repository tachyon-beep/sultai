from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sultai.trust import ContractViolation
from sultai_screen import fit as fit_module
from sultai_screen.contracts import PilotHold
from sultai_screen.data import Dataset, as_tensor, generator, normalized
from sultai_screen.fit import Adapter, initial_adapter, losses, mean_loss
from sultai_screen.host import Cached, Features, Host, cached, freeze, load_host, make_host, state_hash
from sultai_screen.m0.calibrate import (
    apply_calibration,
    class_bias,
    load_noop,
    no_op,
    recalibrate_bn,
    save_noop,
    temperature,
)
from sultai_screen.m0.data import features_from_images
from sultai_screen.m0.fold import fold_affine, fold_residual
from sultai_screen.m0.modules import (
    FRONTIER_WIDTHS,
    Cut,
    ablation_cut,
    activation_means,
    contract,
    contribution,
    cut,
    cut_unfolded,
    leave_one_out,
    lottery_start,
    random_cut,
    rank_cut,
)

CPU = torch.device("cpu")
NONZERO = (3, 9, 17, 22, 30, 41, 50, 63)


def random_module(hidden: int, seed: int, scale: float = 0.1) -> Adapter:
    rng = torch.Generator().manual_seed(seed)
    shapes = ((hidden, 16), (hidden,), (16, hidden), (16,))
    return Adapter(tuple(torch.randn(shape, generator=rng) * scale for shape in shapes))


def sparse_module() -> Adapter:
    """64 units whose V has exactly 56 zero columns; only NONZERO units write to the output."""
    inside, bias_in, outside, bias_out = random_module(64, 7).weights
    mask = torch.zeros(64)
    mask[list(NONZERO)] = 1.0
    return Adapter((inside, bias_in, outside * mask[None, :], bias_out))


def site(seed: int, n: int = 4, size: int = 16) -> tuple[Tensor, Tensor]:
    rng = torch.Generator().manual_seed(seed)
    return torch.randn((n, 16, size, size), generator=rng), torch.randn((n, 16, size, size), generator=rng)


def frozen_host() -> Host:
    torch.manual_seed(0)
    host = Host()
    freeze(host)
    return host


def max_diff(a: Tensor, b: Tensor) -> float:
    return float((a - b).abs().max())


class ModuleTests(unittest.TestCase):
    def test_frontier_widths_table(self) -> None:
        self.assertEqual(
            FRONTIER_WIDTHS,
            {
                0.25: (16, 16, 32, 32),
                0.5: (32, 32, 64, 64),
                0.75: (48, 48, 96, 96),
                1.0: (64, 64, 128, 128),
                1.5: (96, 96, 192, 192),
            },
        )

    def test_known_answer_rank_cut_preserves_output_random_and_wrong_cuts_do_not(self) -> None:
        adapter = sparse_module()
        self.assertEqual(int((adapter.weights[2].abs().sum(0) == 0).sum()), 56)
        read, base = site(1)
        original = adapter.apply(read, base)

        ranked = rank_cut(adapter, read, 8)
        self.assertEqual(ranked.survivors, NONZERO)
        self.assertLess(max_diff(ranked.folded.apply(read, base), original), 1e-6)

        seed = 0
        drawn = torch.randperm(64, generator=generator(seed, "random-cut"))[:8]
        self.assertNotEqual(set(drawn.tolist()), set(NONZERO))
        randomised = random_cut(adapter, read, 8, seed)
        self.assertEqual(randomised.survivors, tuple(sorted(drawn.tolist())))
        self.assertGreater(max_diff(randomised.folded.apply(read, base), original), 1e-4)

        zero_units = torch.tensor([i for i in range(64) if i not in NONZERO][:8])
        wrong = cut(adapter, zero_units, activation_means(adapter, read))
        self.assertGreater(max_diff(wrong.apply(read, base), original), 1e-4)

    def test_fold_adds_removed_unit_means_and_unfolded_omits_them(self) -> None:
        adapter = random_module(4, 3, scale=0.5)
        read, _ = site(2, n=3, size=8)
        result = rank_cut(adapter, read, 2)
        inside, bias_in, outside, bias_out = (w.double() for w in adapter.weights)
        hidden = torch.tanh(torch.einsum("ij,bjhw->bihw", inside, read.double()) + bias_in[None, :, None, None])
        mean = hidden.mean(dim=(0, 2, 3))
        removed = [i for i in range(4) if i not in result.survivors]
        term = sum((outside[:, i] * mean[i] for i in removed), torch.zeros(16, dtype=torch.float64))
        self.assertGreater(float(term.abs().max()), 1e-3)
        self.assertLess(max_diff(result.folded.weights[3].double(), bias_out + term), 1e-6)
        self.assertTrue(torch.equal(result.unfolded.weights[3], adapter.weights[3]))
        keep = list(result.survivors)
        for sliced in (result.folded, result.unfolded):
            self.assertTrue(torch.equal(sliced.weights[0], adapter.weights[0][keep]))
            self.assertTrue(torch.equal(sliced.weights[1], adapter.weights[1][keep]))
            self.assertTrue(torch.equal(sliced.weights[2], adapter.weights[2][:, keep]))
        self.assertLess(max_diff(activation_means(adapter, read), mean), 1e-6)
        rms = hidden.pow(2).mean(dim=(0, 2, 3)).sqrt()
        self.assertLess(max_diff(contribution(adapter, read), outside.norm(dim=0) * rms), 1e-6)
        explicit = cut(adapter, torch.tensor(keep), activation_means(adapter, read))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(explicit.weights, result.folded.weights, strict=True)))
        unfolded = cut_unfolded(adapter, torch.tensor(keep))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(unfolded.weights, result.unfolded.weights, strict=True)))

    def test_survivors_sorted_for_every_cut(self) -> None:
        adapter = random_module(64, 4)
        read, _ = site(5, n=2, size=8)
        ablation = torch.randn(64, generator=torch.Generator().manual_seed(6))
        cuts = (rank_cut(adapter, read, 8), random_cut(adapter, read, 8, 11), ablation_cut(adapter, read, 8, ablation))
        for result in cuts:
            self.assertIsInstance(result, Cut)
            self.assertEqual(len(result.survivors), 8)
            self.assertEqual(list(result.survivors), sorted(set(result.survivors)))
            self.assertTrue(torch.equal(result.folded.weights[0], adapter.weights[0][list(result.survivors)]))
        expected = torch.argsort(-ablation, stable=True)[:8]
        self.assertEqual(cuts[2].survivors, tuple(sorted(expected.tolist())))
        # G1: every cut (rank, random, ablation) folds the removed units' means into d; its unfolded twin keeps d.
        means = activation_means(adapter, read)
        outside, bias = adapter.weights[2].double(), adapter.weights[3]
        for result in cuts:
            removed = [unit for unit in range(64) if unit not in result.survivors]
            folded = bias.double() + (outside[:, removed] * means[removed][None, :]).sum(dim=1)
            self.assertGreater(float((folded - bias.double()).abs().max()), 1e-3)
            self.assertLessEqual(float((result.folded.weights[3].double() - folded).abs().max()), 1e-6)
            self.assertTrue(torch.equal(result.unfolded.weights[3], bias))

    def test_ties_resolve_to_lower_index(self) -> None:
        inside, bias_in, outside, bias_out = random_module(4, 8).weights
        inside, bias_in, outside = inside.clone(), bias_in.clone(), outside.clone()
        inside[2], bias_in[2], outside[:, 2] = inside[1], bias_in[1], outside[:, 1]
        outside[:, 3] = outside[:, 1] * 5.0
        outside[:, 0] = outside[:, 1] * 0.1
        inside[3], bias_in[3] = inside[1], bias_in[1]
        inside[0], bias_in[0] = inside[1], bias_in[1]
        adapter = Adapter((inside, bias_in, outside, bias_out))
        read, _ = site(9, n=2, size=8)
        scores = contribution(adapter, read)
        self.assertEqual(float(scores[1]), float(scores[2]))
        self.assertGreater(float(scores[3]), float(scores[1]))
        self.assertGreater(float(scores[1]), float(scores[0]))
        self.assertEqual(rank_cut(adapter, read, 2).survivors, (1, 3))
        tied = torch.tensor([0.1, 0.2, 0.2, 0.5])
        self.assertEqual(ablation_cut(adapter, read, 2, tied).survivors, (1, 3))

    def test_counts_are_33h_plus_16(self) -> None:
        adapter = random_module(64, 12)
        read, _ = site(13, n=2, size=8)
        counts = []
        for h in (16, 8, 4, 2):
            result = rank_cut(adapter, read, h)
            self.assertEqual(result.folded.count(), 33 * h + 16)
            self.assertEqual(result.unfolded.count(), 33 * h + 16)
            counts.append(result.folded.count())
        self.assertEqual(counts, [544, 280, 148, 82])

    def test_contract_reproduces_expanded_output(self) -> None:
        rng = torch.Generator().manual_seed(14)
        expanded = Adapter(
            (torch.randn((16, 64), generator=rng) * 0.1, torch.randn((64, 16), generator=rng) * 0.1, torch.randn(16))
        )
        read, base = site(15)
        contracted = contract(expanded)
        self.assertEqual(len(contracted.weights), 2)
        self.assertTrue(contracted.nonlinear)
        self.assertTrue(torch.equal(contracted.weights[1], expanded.weights[2]))
        self.assertLess(max_diff(contracted.apply(read, base), expanded.apply(read, base)), 1e-6)
        with self.assertRaises(ContractViolation):
            contract(contracted)

    def test_lottery_start_slices_regenerated_initial_tensors(self) -> None:
        keep = (2, 5, 17, 40, 63)
        start = lottery_start(21, keep, CPU)
        inside, bias_in, outside, bias_out = initial_adapter(
            21, hidden=64, nonlinear=True, expanded=False, device=CPU
        ).weights
        index = list(keep)
        expected = (inside[index], bias_in[index], outside[:, index], bias_out)
        self.assertEqual(len(start.weights), 4)
        self.assertTrue(start.nonlinear)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(start.weights, expected, strict=True)))
        with self.assertRaises(ContractViolation):
            lottery_start(21, (), CPU)
        with self.assertRaises(ContractViolation):
            lottery_start(21, (5, 2), CPU)
        with self.assertRaises(ContractViolation):
            lottery_start(21, (1, 64), CPU)

    def test_lottery_start_is_the_step_zero_adapter_of_the_wide_grid(self) -> None:
        # G2: the lottery's "initial weights" are bitwise the h = 64 adapter fit_sgd (and so grid) starts M-wide from.
        host = Host()
        freeze(host)
        read = torch.randn((32, 16, 4, 4), generator=torch.Generator().manual_seed(3))
        data = Features(read, read * 0.3, torch.arange(32) % 10)
        starts: list[Adapter] = []
        real = fit_module.gradient_step

        def spy(host: Host, data: Features, adapter: Adapter, rate: float, indices: Tensor) -> tuple[Adapter, Tensor]:
            starts.append(adapter.detached())
            return real(host, data, adapter, rate, indices)

        # One step each: the single gradient step of a fit_sgd run and of a one-checkpoint grid run.
        with patch("sultai_screen.fit.gradient_step", side_effect=spy):
            fit_module.fit_sgd(host, data, data, seed=21, rate=0.1, steps=1, minibatch=True, hidden=64)
            fit_module.grid(host, data, data, 21, rates=(0.1,), checkpoints=(1,), hidden=64)
        full = lottery_start(21, tuple(range(64)), CPU)
        self.assertEqual(len(starts), 2)
        for start in starts:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(start.weights, full.weights, strict=True)))
            self.assertEqual(start.nonlinear, full.nonlinear)

    def test_leave_one_out_shape_and_zero_for_silent_unit(self) -> None:
        inside, bias_in, outside, bias_out = random_module(4, 16, scale=1.0).weights
        outside = outside.clone()
        outside[:, 2] = 0.0
        adapter = Adapter((inside, bias_in, outside, bias_out))
        rng = torch.Generator().manual_seed(17)
        read = torch.randn((4, 16, 16, 16), generator=rng)
        data = Features(read, read * 0.3, torch.arange(4))
        ablation = leave_one_out(frozen_host(), data, adapter)
        self.assertEqual(tuple(ablation.shape), (4,))
        self.assertLess(abs(float(ablation[2])), 1e-5)
        self.assertGreater(float(ablation.abs().max()), 1e-4)
        means = activation_means(adapter, read)
        full = float(losses(frozen_host(), data, adapter).double().mean())
        removed = cut(adapter, torch.tensor([1, 2, 3]), means)
        self.assertAlmostEqual(
            float(ablation[0]), float(losses(frozen_host(), data, removed).double().mean()) - full, places=12
        )

    def test_contract_violations(self) -> None:
        adapter = random_module(4, 18)
        read, _ = site(19, n=1, size=8)
        two = Adapter((torch.zeros(16, 16), torch.zeros(16)))
        means = activation_means(adapter, read)
        for call in (
            lambda: rank_cut(two, read, 2),
            lambda: random_cut(two, read, 2, 0),
            lambda: ablation_cut(two, read, 2, torch.zeros(4)),
            lambda: activation_means(two, read),
            lambda: contribution(two, read),
            lambda: cut(two, torch.tensor([0]), means),
            lambda: cut_unfolded(two, torch.tensor([0])),
            lambda: rank_cut(adapter, read, 0),
            lambda: rank_cut(adapter, read, 5),
            lambda: random_cut(adapter, read, 5, 0),
            lambda: ablation_cut(adapter, read, 0, torch.zeros(4)),
            lambda: ablation_cut(adapter, read, 2, torch.zeros(3)),
            lambda: cut(adapter, torch.tensor([0]), means[:3]),
            lambda: cut(adapter, torch.tensor([], dtype=torch.long), means),
            lambda: cut(adapter, torch.tensor([0, 0]), means),
            lambda: cut(adapter, torch.tensor([4]), means),
        ):
            with self.assertRaises(ContractViolation):
                call()


def affine(seed: int, *, w_scale: float = 0.1, b_scale: float = 0.1) -> Adapter:
    rng = torch.Generator().manual_seed(seed)
    w = torch.randn((16, 16), generator=rng) * w_scale
    b = torch.randn(16, generator=rng) * b_scale
    return Adapter((w, b), nonlinear=False)


def tiny_dataset() -> Dataset:
    rng = torch.Generator().manual_seed(8)
    images = torch.randint(0, 256, (8, 3, 32, 32), generator=rng, dtype=torch.uint8)
    labels = torch.randint(0, 10, (8,), generator=rng)
    ids = torch.arange(8)
    return Dataset(images, labels, ids, ids.clone(), ids.clone(), ids.clone())


class FoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.host = make_host(3000, 0, CPU)
        freeze(self.host)
        self.z = torch.randn((4, 16, 16, 16), generator=torch.Generator().manual_seed(5))

    def conv_pair(self, adapter: Adapter) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """tail[0] of folded on z and of host on u: float32 modules, then float64 conv on the stored float32 params."""
        folded = fold_affine(self.host, adapter)
        with torch.no_grad():
            u = adapter.apply(self.z, self.z)
            z64 = self.z.double()
            u64 = Adapter(tuple(w.double() for w in adapter.weights), nonlinear=False).apply(z64, z64)
            folded_conv, host_conv = folded.tail[0], self.host.tail[0]
            folded64 = F.conv2d(z64, folded_conv.weight.double(), folded_conv.bias.double(), padding=1)
            reference64 = F.conv2d(u64, host_conv.weight.double(), host_conv.bias.double(), padding=1)
            return as_tensor(folded_conv(self.z)), as_tensor(host_conv(u)), folded64, reference64

    def test_interior_exact_and_border_over_adds_when_bias_nonzero(self) -> None:
        folded_out, reference, folded64, reference64 = self.conv_pair(affine(1))
        self.assertLessEqual(max_diff(folded64[..., 1:15, 1:15], reference64[..., 1:15, 1:15]), 1e-6)
        # The float32 kernel is already ~1.3e-6 from its float64 value before any fold: this bounds the kernel.
        self.assertLessEqual(max_diff(folded_out[..., 1:15, 1:15], reference[..., 1:15, 1:15]), 1e-5)
        border = torch.ones((16, 16), dtype=torch.bool)
        border[1:15, 1:15] = False
        self.assertEqual(int(border.sum()), 60)
        self.assertGreater(max_diff(folded_out[..., border], reference[..., border]), 1e-5)

    def test_zero_bias_fold_is_exact_everywhere(self) -> None:
        folded_out, reference, folded64, reference64 = self.conv_pair(affine(2, b_scale=0.0))
        self.assertLessEqual(max_diff(folded64, reference64), 1e-6)
        # The float32 kernel is already ~1.3e-6 from its float64 value before any fold: this bounds the kernel.
        self.assertLessEqual(max_diff(folded_out, reference), 1e-5)

    def test_only_tail0_weight_and_bias_change_and_original_untouched(self) -> None:
        before = state_hash(self.host)
        original = {name: tensor.clone() for name, tensor in self.host.state_dict().items()}
        folded = fold_affine(self.host, affine(3))
        self.assertEqual(state_hash(self.host), before)
        self.assertNotEqual(state_hash(folded), before)
        self.assertTrue(folded.sealed)
        self.assertFalse(folded.training)
        self.assertFalse(any(p.requires_grad for p in folded.parameters()))
        changed = folded.state_dict()
        self.assertEqual(set(changed), set(original))
        for name, tensor in original.items():
            self.assertEqual(changed[name].dtype, tensor.dtype, name)
            self.assertEqual(changed[name].shape, tensor.shape, name)
            if name in ("tail.0.weight", "tail.0.bias"):
                self.assertFalse(torch.equal(changed[name], tensor), name)
            else:
                self.assertTrue(torch.equal(changed[name], tensor), name)
        for name, tensor in self.host.state_dict().items():
            self.assertTrue(torch.equal(tensor, original[name]), name)

    def test_fold_does_not_consume_global_rng(self) -> None:
        torch.manual_seed(11)
        expected = torch.rand(4)
        torch.manual_seed(11)
        fold_affine(self.host, affine(4))
        self.assertTrue(torch.equal(torch.rand(4), expected))

    def residual(self, adapter: Adapter) -> float:
        data = tiny_dataset()
        development = features_from_images(self.host, data, torch.arange(8))
        folded = fold_affine(self.host, adapter)
        return fold_residual(self.host, adapter, folded, development, data, torch.arange(8))

    def test_residual_zero_bitwise_for_zero_adapter(self) -> None:
        value = self.residual(affine(5, w_scale=0.0, b_scale=0.0))
        self.assertIsInstance(value, float)
        self.assertEqual(value, 0.0)

    def test_residual_finite_and_nonzero_with_bias(self) -> None:
        adapter = affine(6, w_scale=0.0, b_scale=1.0)
        value = self.residual(adapter)
        self.assertIsInstance(value, float)
        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertNotEqual(value, 0.0)
        data = tiny_dataset()
        development = features_from_images(self.host, data, torch.arange(8))
        folded = fold_affine(self.host, adapter)
        expected = mean_loss(folded, features_from_images(folded, data, torch.arange(8)), None) - mean_loss(
            self.host, development, adapter
        )
        self.assertEqual(value, expected)

    def test_residual_finite_for_random_affine(self) -> None:
        value = self.residual(affine(7))
        self.assertIsInstance(value, float)
        self.assertTrue(torch.isfinite(torch.tensor(value)))

    def test_contract_violations(self) -> None:
        w, b = affine(8).weights
        bad = (
            Adapter((w, b)),
            Adapter((w[:8], b), nonlinear=False),
            Adapter((w, b[:8]), nonlinear=False),
            Adapter((w, w, b), nonlinear=False),
            random_module(4, 1),
        )
        for adapter in bad:
            with (
                self.subTest(count=len(adapter.weights), nonlinear=adapter.nonlinear),
                self.assertRaises(ContractViolation),
            ):
                fold_affine(self.host, adapter)
        for width in (2, 4):
            wide = make_host(3000, width, CPU)
            freeze(wide)
            with self.subTest(width=width), self.assertRaises(ContractViolation):
                fold_affine(wide, affine(8))

    def test_residual_rejects_mismatched_folded_host(self) -> None:
        adapter = affine(9)
        data = tiny_dataset()
        development = features_from_images(self.host, data, torch.arange(8))
        for folded in (self.host, fold_affine(self.host, affine(10))):
            with self.assertRaises(ContractViolation):
                fold_residual(self.host, adapter, folded, development, data, torch.arange(8))


BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def calibration_dataset(labels: Tensor | None = None) -> Dataset:
    rng = torch.Generator().manual_seed(9)
    images = torch.randint(0, 256, (64, 3, 32, 32), generator=rng, dtype=torch.uint8)
    drawn = torch.randint(0, 10, (64,), generator=rng)
    return Dataset(
        images,
        drawn if labels is None else labels,
        torch.arange(64),
        torch.arange(0, 16),
        torch.arange(16, 40),
        torch.arange(40, 64),
    )


def batch_norms(host: Host) -> list[nn.BatchNorm2d]:
    return [module for module in host.modules() if isinstance(module, nn.BatchNorm2d)]


def sampled_logits(seed: int, n: int = 2000) -> tuple[Tensor, Tensor]:
    """Logits with labels drawn from their own softmax, so the fitted temperature has a finite interior optimum."""
    rng = torch.Generator().manual_seed(seed)
    values = torch.randn((n, 10), generator=rng, dtype=torch.float64) * 3
    labels = torch.multinomial(torch.softmax(values, 1), 1, generator=rng).squeeze(1)
    return values, labels


class SelectionOnly:
    """Stands in for Cached and fails the test if anything other than the selection split is read."""

    def __init__(self, inner: Cached, seen: list[Features]) -> None:
        self._inner = inner
        self._seen = seen

    @property
    def selection(self) -> Features:
        self._seen.append(self._inner.selection)
        return self._inner.selection

    @property
    def conditioning(self) -> Features:
        raise AssertionError("no_op read the conditioning split")

    @property
    def development(self) -> Features:
        raise AssertionError("no_op read the development split")


class CalibrateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = calibration_dataset()
        self.checkpoint = self.save(make_host(3000, 0, CPU), "host.pt")

    def save(self, host: Host, name: str) -> Path:
        path = self.root / name
        torch.save(dict(host.state_dict()), path)
        return path

    def recalibrated(self, path: Path, data: Dataset | None = None) -> Host:
        source = self.data if data is None else data
        return recalibrate_bn(load_host(path, 3000, 0, CPU, frozen=False), source.images, source.train)

    def guarded_no_op(self, raw: Host, recalibrated: Host, data: Dataset) -> tuple[tuple[Host, str], list[Features]]:
        seen: list[Features] = []

        def wrapper(host: Host, source: Dataset) -> SelectionOnly:
            return SelectionOnly(cached(host, source), seen)

        with patch("sultai_screen.m0.calibrate.cached", side_effect=wrapper):
            return no_op(raw, recalibrated, data), seen

    def test_recalibrate_refuses_a_sealed_host_before_any_forward(self) -> None:
        host = load_host(self.checkpoint, 3000, 0, CPU, frozen=True)
        before = {name: tensor.clone() for name, tensor in host.state_dict().items()}
        with self.assertRaisesRegex(PilotHold, "sealed"):
            recalibrate_bn(host, self.data.images, self.data.train)
        for name, tensor in host.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)
        for module in batch_norms(host):
            self.assertEqual(module.momentum, 0.1)
            self.assertFalse(module.training)

    def test_recalibrate_changes_only_batchnorm_statistics_and_freezes(self) -> None:
        host = load_host(self.checkpoint, 3000, 0, CPU, frozen=False)
        self.assertFalse(host.sealed)
        before = {name: tensor.clone() for name, tensor in host.state_dict().items()}
        result = recalibrate_bn(host, self.data.images, self.data.train)
        self.assertIs(result, host)
        after = result.state_dict()
        self.assertEqual(set(after), set(before))
        changed = {name for name, tensor in after.items() if not torch.equal(tensor, before[name])}
        statistics = {name for name in before if name.rsplit(".", 1)[1] in BN_BUFFERS}
        self.assertEqual(changed, statistics)
        self.assertEqual(len(statistics), 3 * len(batch_norms(result)))
        for name, parameter in result.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
        self.assertTrue(result.sealed)
        self.assertFalse(result.training)
        result.train()
        with self.assertRaises(PilotHold):
            result(normalized(self.data.images[:2]))

    def test_recalibrate_is_cumulative_over_fixed_chunks_with_the_partial_chunk(self) -> None:
        indices = torch.cat((torch.arange(64), torch.arange(64), torch.arange(2)))
        host = recalibrate_bn(load_host(self.checkpoint, 3000, 0, CPU, frozen=False), self.data.images, indices)
        for module in batch_norms(host):
            self.assertIsNone(module.momentum)
            self.assertEqual(int(module.num_batches_tracked), 2)
        first = host.front[1]
        self.assertIsInstance(first, nn.BatchNorm2d)
        with torch.no_grad():
            means = [
                as_tensor(host.front[0](normalized(self.data.images[chunk]))).mean(dim=(0, 2, 3))
                for chunk in (indices[:128], indices[128:])
            ]
        self.assertLessEqual(max_diff(as_tensor(first.running_mean), (means[0] + means[1]) / 2), 1e-5)
        repeat = recalibrate_bn(load_host(self.checkpoint, 3000, 0, CPU, frozen=False), self.data.images, indices)
        self.assertEqual(state_hash(repeat), state_hash(host))

    def test_no_op_recomputes_selection_features_from_each_candidate(self) -> None:
        raw = load_host(self.checkpoint, 3000, 0, CPU, frozen=True)
        rbn = self.recalibrated(self.checkpoint)
        front = [as_tensor(module.running_mean) for module in batch_norms(raw)[:3]]
        moved = [as_tensor(module.running_mean) for module in batch_norms(rbn)[:3]]
        self.assertGreater(max(max_diff(a, b) for a, b in zip(front, moved, strict=True)), 1e-2)
        (_, label), seen = self.guarded_no_op(raw, rbn, self.data)
        self.assertIn(label, ("raw", "R-bn"))
        self.assertEqual(len(seen), 2)
        self.assertFalse(torch.equal(seen[0].read, seen[1].read))
        self.assertTrue(torch.equal(seen[0].labels, seen[1].labels))
        self.assertTrue(torch.equal(seen[0].read, cached(raw, self.data).selection.read))
        self.assertTrue(torch.equal(seen[1].read, cached(rbn, self.data).selection.read))

    def absurd(self, host: Host) -> int:
        """Pin the last tail BatchNorm far off its data: every example then gets one confident class."""
        last = batch_norms(host)[-1]
        with torch.no_grad():
            last.running_mean.fill_(-1e3)
            last.running_var.fill_(1.0)
            predicted = as_tensor(host(normalized(self.data.images[:1]))).argmax(1)
        return int(predicted[0])

    def test_no_op_picks_recalibration_when_it_lowers_selection_loss(self) -> None:
        stale = load_host(self.checkpoint, 3000, 0, CPU, frozen=False)
        freeze(stale)
        wrong = (self.absurd(stale) + 1) % 10
        stale_path = self.save(stale, "stale.pt")
        data = calibration_dataset(torch.full((64,), wrong))
        raw = load_host(stale_path, 3000, 0, CPU, frozen=True)
        rbn = self.recalibrated(stale_path, data)
        raw_loss = mean_loss(raw, cached(raw, data).selection, None)
        rbn_loss = mean_loss(rbn, cached(rbn, data).selection, None)
        self.assertGreater(raw_loss, rbn_loss + 1.0)
        (chosen, label), seen = self.guarded_no_op(raw, rbn, data)
        self.assertEqual(label, "R-bn")
        self.assertIs(chosen, rbn)
        self.assertEqual(len(seen), 2)

    def test_no_op_keeps_raw_when_recalibration_does_not_help(self) -> None:
        raw = self.recalibrated(self.checkpoint)
        worse = self.recalibrated(self.checkpoint)
        wrong = (self.absurd(worse) + 1) % 10
        data = calibration_dataset(torch.full((64,), wrong))
        raw_loss = mean_loss(raw, cached(raw, data).selection, None)
        worse_loss = mean_loss(worse, cached(worse, data).selection, None)
        self.assertGreater(worse_loss, raw_loss + 1.0)
        (chosen, label), seen = self.guarded_no_op(raw, worse, data)
        self.assertEqual(label, "raw")
        self.assertIs(chosen, raw)
        self.assertEqual(len(seen), 2)

    def test_no_op_tie_keeps_raw(self) -> None:
        raw = load_host(self.checkpoint, 3000, 0, CPU, frozen=True)
        twin = load_host(self.checkpoint, 3000, 0, CPU, frozen=True)
        (chosen, label), _ = self.guarded_no_op(raw, twin, self.data)
        self.assertEqual(label, "raw")
        self.assertIs(chosen, raw)

    def test_save_and_load_noop_bind_the_chosen_host(self) -> None:
        raw = load_host(self.checkpoint, 3000, 0, CPU, frozen=True)
        rbn = self.recalibrated(self.checkpoint)
        output = self.root / "anchor"
        output.mkdir()
        digest = save_noop(rbn, output)
        self.assertEqual(digest, state_hash(rbn))
        self.assertTrue((output / "noop.pt").is_file())
        self.assertNotEqual(state_hash(raw), digest)
        with self.assertRaises(PilotHold):
            load_noop(output, 3000, (64, 64, 128, 128), state_hash(raw), CPU)
        loaded = load_noop(output, 3000, (64, 64, 128, 128), digest, CPU)
        self.assertEqual(state_hash(loaded), digest)
        self.assertTrue(loaded.sealed)
        self.assertFalse(loaded.training)

    def test_temperature_scales_with_the_logits(self) -> None:
        values, labels = sampled_logits(21)
        once = temperature(values, labels)
        twice = temperature(values * 2, labels)
        self.assertIsInstance(once, float)
        self.assertLess(abs(once - 1.0), 0.2)
        self.assertLess(abs(twice / once - 2.0), 1e-3)
        self.assertEqual(temperature(values, labels), once)
        self.assertEqual(temperature(values.float(), labels), temperature(values.float(), labels))

    def test_class_bias_has_eleven_numbers_and_fits_no_worse_than_temperature(self) -> None:
        values, labels = sampled_logits(22)
        shifted = values + torch.linspace(-1.0, 1.0, 10, dtype=torch.float64)
        t, bias = class_bias(values, labels)
        self.assertIsInstance(t, float)
        self.assertEqual(tuple(bias.shape), (10,))
        self.assertEqual(1 + bias.numel(), 11)
        self.assertTrue(bool(torch.isfinite(bias).all()))
        t_shift, bias_shift = class_bias(shifted, labels)
        only = temperature(shifted, labels)
        joint_loss = float(F.cross_entropy(apply_calibration(shifted, t_shift, bias_shift), labels))
        temp_loss = float(F.cross_entropy(shifted / only, labels))
        self.assertLess(joint_loss, temp_loss)
        repeat = class_bias(values, labels)
        self.assertEqual(repeat[0], t)
        self.assertTrue(torch.equal(repeat[1], bias))

    def test_apply_calibration_is_scaled_logits_plus_bias(self) -> None:
        values, _ = sampled_logits(23, n=5)
        bias = torch.linspace(-0.5, 0.5, 10, dtype=torch.float64)
        self.assertTrue(torch.equal(apply_calibration(values, 1.7, bias), values / 1.7 + bias))
        narrow = values.float()
        result = apply_calibration(narrow, 1.7, bias)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result, narrow / 1.7 + bias.float()))

    def test_calibration_contract_violations(self) -> None:
        values, labels = sampled_logits(24, n=8)
        for bad_values, bad_labels in (
            (values[:, :9], labels),
            (values, labels[:7]),
            (values[:0], labels[:0]),
            (values.clone().fill_(float("nan")), labels),
            (values, labels.clone().fill_(10)),
        ):
            with self.subTest(shape=tuple(bad_values.shape)), self.assertRaises(ContractViolation):
                temperature(bad_values, bad_labels)
            with self.subTest(shape=tuple(bad_values.shape)), self.assertRaises(ContractViolation):
                class_bias(bad_values, bad_labels)


if __name__ == "__main__":
    unittest.main()
