from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from sultai.trust import ContractViolation
from sultai_screen.contracts import Record
from sultai_screen.data import Dataset, generator, partitions
from sultai_screen.fit import (
    Adapter,
    Candidate,
    accuracy,
    at_edge,
    choose,
    fit_sgd,
    grid,
    initial_adapter,
    logits,
    losses,
    mean_accuracy,
    mean_loss,
    oracle,
    paired_se,
    plateau,
    zero_adapter,
)
from sultai_screen.host import Features, Host, freeze
from sultai_screen.m0.data import SEEN_FIT_SHA, features_from_images, seen_fit

REPO = Path(__file__).resolve().parents[1]
DEV_ARTIFACT = REPO / "docs/results/gate3-split-2026-10-11/dev-indices-int32-le.bin"
# Pinned from the unedited losses() on the 129-row fixture below (controller ruling R3).
LOSSES_SHA = "343a6bf756878ae0000abeb5f4a72eaa35c2a843c5b5e7cfe6602a7e0c2b6067"
# Pinned from the unedited fit_sgd()/oracle() (before Task 6) on fit_features() below (controller ruling R3).
FIT_SHAS = (
    "d5e0cf902f0006f5a801535d73f6175d3226e80c647d267384d0f91025847b67",
    "e002060d21fcacad69a455955a8d862eba293ec6f5565f41351de81075128d1b",
)
WIDE_SHAS = (
    "a20a94ba662812eb770c044077a59d08a8429dccc170f86c5b14c0504876a8e0",
    "c291020582cbdf50d26ced482af27048a863bc399f69afb6bdfe2ebcdc600416",
    "75a57420375b3fdecc5d0de54f4ba47c694847e59744fd414bc3b9e51b460d5f",
    "8c0abb9bdd78d73773aea9e54243e29a8c0662fe8ca6be984e4719b4e3a90f7c",
)
FULL_BATCH_SHAS = (
    "518dc8546817894a652334cbfe94dcb9fd07bd2b06d1a2e68efa38d9818e3ede",
    "1908f34e462f74d217ef7bb517f601fb17d23d02afcaf6368e78026e2202c1c9",
)
ORACLE_BEST_SHAS = (
    "c1b26d05d259b9eb7354f47e632f57c8469c54d9505cd041618498fd973c850d",
    "c6e94e9cfb4607b01f9cab4ff6429649a654c765a0d32b79a7669996016f498b",
)
# SHA-256 of the full oracle report with only wall_seconds excluded (key set included).
ORACLE_REPORT_SHA = "baeb09439fb0e8176705f5031578b2c5358365ed1d722e4769b32c955b902393"


def fixture_features(n: int = 129) -> Features:
    rng = torch.Generator().manual_seed(5)
    z = torch.rand((n, 16, 8, 8), generator=rng)
    return Features(z, z * 0.3, torch.randint(0, 10, (n,), generator=rng))


def fit_features() -> Features:
    rng = torch.Generator().manual_seed(11)
    read = torch.randn((64, 16, 8, 8), generator=rng)
    return Features(read, read * 0.3, torch.arange(64) % 10)


def tensor_shas(adapter: Adapter) -> tuple[str, ...]:
    return tuple(hashlib.sha256(w.numpy().tobytes()).hexdigest() for w in adapter.weights)


def same_weights(a: Adapter, b: Adapter) -> bool:
    return len(a.weights) == len(b.weights) and all(
        torch.equal(x, y) for x, y in zip(a.weights, b.weights, strict=True)
    )


def without_wall(report: Record) -> Record:
    return {key: value for key, value in report.items() if key != "wall_seconds"}


def frozen_host() -> Host:
    torch.manual_seed(0)
    host = Host()
    freeze(host)
    return host


def real_partition_dataset() -> Dataset:
    train, conditioning, selection, development = partitions(DEV_ARTIFACT)
    images = torch.zeros((1, 3, 32, 32), dtype=torch.uint8)
    return Dataset(images, torch.zeros(1, dtype=torch.long), train, conditioning, selection, development)


class FitTests(unittest.TestCase):
    def test_seen_fit_properties_and_sha(self) -> None:
        data = real_partition_dataset()
        indices = seen_fit(data)
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(len(indices), 2000)
        self.assertEqual(len(set(indices.tolist())), 2000)
        train = set(data.train.tolist())
        self.assertTrue(set(indices.tolist()) <= train)
        for other in (data.conditioning, data.selection, data.development):
            self.assertFalse(set(indices.tolist()) & set(other.tolist()))
        # Production data.train is a permutation of real ids, so this SHA is the production value.
        self.assertEqual(hashlib.sha256(indices.numpy().astype("<i8").tobytes()).hexdigest(), SEEN_FIT_SHA)

    def test_seen_fit_rejects_wrong_train_size(self) -> None:
        data = real_partition_dataset()
        short = Dataset(
            data.images, data.labels, data.train[:39_999], data.conditioning, data.selection, data.development
        )
        with self.assertRaises(ContractViolation):
            seen_fit(short)

    def test_accuracy_matches_unpadded_forward(self) -> None:
        host = frozen_host()
        data = fixture_features(129)
        rng = torch.Generator().manual_seed(9)
        adapter = Adapter((torch.randn((16, 16), generator=rng) * 0.1, torch.randn(16, generator=rng) * 0.1))
        for candidate in (None, adapter):
            with torch.no_grad():
                values = data.base if candidate is None else candidate.apply(data.read, data.base)
                expected = (host.classify(values).argmax(1) == data.labels).float()
            got = accuracy(host, data, candidate)
            self.assertEqual(got.shape, (129,))
            self.assertTrue(torch.equal(got, expected))
            self.assertEqual(logits(host, data, candidate).shape, (129, 10))
            self.assertEqual(mean_accuracy(host, data, candidate), float(expected.double().mean()))

    def test_losses_bytes_unchanged(self) -> None:
        host = frozen_host()
        digest = hashlib.sha256(losses(host, fixture_features(129), None).numpy().tobytes()).hexdigest()
        self.assertEqual(digest, LOSSES_SHA)

    def test_features_from_images_matches_cached_part(self) -> None:
        host = frozen_host()
        rng = torch.Generator().manual_seed(3)
        images = torch.randint(0, 256, (150, 3, 32, 32), generator=rng, dtype=torch.uint8)
        labels = torch.randint(0, 10, (150,), generator=rng)
        empty = torch.zeros(0, dtype=torch.long)
        data = Dataset(images, labels, empty, empty, empty, empty)
        ids = torch.tensor([140, 3, 7, 99] * 33)
        got = features_from_images(host, data, ids)
        self.assertEqual(got.read.shape[0], 132)
        self.assertTrue(torch.equal(got.labels, labels[ids]))
        self.assertTrue(torch.equal(got.read[0], got.read[4]))

    def test_fit_sgd_golden_pins(self) -> None:
        host, data = frozen_host(), fit_features()
        fit = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=4, minibatch=True)
        self.assertEqual(tensor_shas(fit.adapter), FIT_SHAS)
        wide = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=4, minibatch=True, wide=True)
        self.assertEqual(tensor_shas(wide.adapter), WIDE_SHAS)
        full = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=4, minibatch=False)
        self.assertEqual(tensor_shas(full.adapter), FULL_BATCH_SHAS)

    def test_oracle_outputs_unchanged(self) -> None:
        host, data = frozen_host(), fit_features()
        result = oracle(host, data, data, 3000, seeds=1)
        self.assertEqual(tensor_shas(result.best.adapter), ORACLE_BEST_SHAS)
        report = result.report()
        best, candidates = report["best"], report["candidates"]
        assert isinstance(best, dict) and isinstance(candidates, list)
        self.assertTrue(all(isinstance(c, dict) for c in candidates))
        pinned: Record = {
            **report,
            "best": without_wall(best),
            "candidates": [without_wall(c) for c in candidates if isinstance(c, dict)],
        }
        digest = hashlib.sha256(json.dumps(pinned, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, ORACLE_REPORT_SHA)

    def test_oracle_passes_only_the_gate3_fit_sgd_keywords(self) -> None:
        # Mirrors tests/test_screen.py's fake: any new keyword passed by oracle() raises TypeError here.
        host, data = frozen_host(), fit_features()

        def candidate(
            host: Host,
            conditioning: Features,
            selection: Features,
            *,
            seed: int,
            rate: float,
            steps: int,
            minibatch: bool,
        ) -> Candidate:
            ok = steps == 800
            trace = tuple(1.0 for _ in range(steps)) if ok else ()
            return Candidate(zero_adapter(data.read.device), rate, seed, steps, 1.0 if ok else None, trace, 0.1, 1, 1)

        with patch("sultai_screen.fit.fit_sgd", side_effect=candidate):
            result = oracle(host, data, data, 3000, seeds=1)
        self.assertEqual(len(result.candidates), 8)
        self.assertEqual(result.best.steps, 800)
        self.assertIn("unavailable", result.plateau_evidence)

    def test_checkpointed_run_is_bitwise_three_separate_runs(self) -> None:
        host, data = frozen_host(), fit_features()
        one = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=12, minibatch=True, checkpoints=(4, 8, 12))
        self.assertEqual([step for step, _, _ in one.snapshots], [4, 8, 12])
        for step, adapter, selection in one.snapshots:
            separate = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=step, minibatch=True)
            self.assertTrue(same_weights(adapter, separate.adapter))
            self.assertEqual(selection, separate.selection)
            self.assertEqual(one.training_losses[:step], separate.training_losses)
        self.assertTrue(same_weights(one.snapshots[-1][1], one.adapter))
        reported = one.report()["snapshots"]
        assert isinstance(reported, list)
        self.assertEqual([r["step"] for r in reported if isinstance(r, dict)], [4, 8, 12])
        with self.assertRaises(ContractViolation):
            fit_sgd(host, data, data, seed=3000, rate=0.3, steps=12, minibatch=True, checkpoints=(8, 4))
        with self.assertRaises(ContractViolation):
            fit_sgd(host, data, data, seed=3000, rate=0.3, steps=12, minibatch=True, checkpoints=(4, 13))

    def test_affine_fit_updates_without_tanh(self) -> None:
        host, data = frozen_host(), fit_features()
        rng = torch.Generator().manual_seed(21)
        w0, b0 = torch.randn((16, 16), generator=rng) * 0.05, torch.randn(16, generator=rng) * 0.05
        start = Adapter((w0.clone(), b0.clone()), nonlinear=False)
        rate = 0.1
        got = fit_sgd(host, data, data, seed=3000, rate=rate, steps=1, minibatch=True, start=start, nonlinear=False)
        self.assertFalse(got.adapter.nonlinear)
        # Independent affine loss on the first minibatch (n=64 < 256, so one fresh permutation of all rows).
        ids = torch.randperm(64, generator=generator(3000, "oracle-order"))
        w, b = w0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
        site = data.base[ids] + torch.einsum("ij,bjhw->bihw", w, data.read[ids]) + b[None, :, None, None]
        loss = F.cross_entropy(host.classify(site), data.labels[ids], reduction="sum") / 64
        grad_w, grad_b = torch.autograd.grad(loss, (w, b))
        expected = (w0 - rate * grad_w, b0 - rate * grad_b)
        self.assertTrue(all(torch.equal(a, e) for a, e in zip(got.adapter.weights, expected, strict=True)))
        self.assertEqual(got.training_losses, (float(loss.detach()),))
        tanh = fit_sgd(host, data, data, seed=3000, rate=rate, steps=1, minibatch=True, start=Adapter((w0, b0)))
        self.assertFalse(same_weights(tanh.adapter, got.adapter))
        self.assertTrue(torch.equal(start.weights[0], w0))  # the warm start is cloned, never updated in place
        with self.assertRaises(ContractViolation):  # declared kind must match the start
            fit_sgd(host, data, data, seed=3000, rate=rate, steps=1, minibatch=True, start=start)

    def test_hidden_width_and_expanded_shapes_and_counts(self) -> None:
        cpu = torch.device("cpu")
        cases = (
            (None, False, ((16, 16), (16,)), 272),
            (8, False, ((8, 16), (8,), (16, 8), (16,)), 280),
            (64, False, ((64, 16), (64,), (16, 64), (16,)), 2128),
            (None, True, ((16, 64), (64, 16), (16,)), 16 * 64 + 64 * 16 + 16),
        )
        for hidden, expanded, shapes, count in cases:
            adapter = initial_adapter(3000, hidden=hidden, nonlinear=True, expanded=expanded, device=cpu)
            self.assertEqual(tuple(tuple(w.shape) for w in adapter.weights), shapes)
            self.assertEqual(adapter.count(), count)
        affine = initial_adapter(3000, hidden=None, nonlinear=False, expanded=False, device=cpu)
        self.assertFalse(affine.nonlinear)
        self.assertEqual(affine.count(), 272)
        with self.assertRaises(ContractViolation):
            initial_adapter(3000, hidden=8, nonlinear=True, expanded=True, device=cpu)
        with self.assertRaises(ContractViolation):
            initial_adapter(3000, hidden=8, nonlinear=False, expanded=False, device=cpu)
        host, data = frozen_host(), fit_features()
        wide = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=3, minibatch=True, wide=True)
        same = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=3, minibatch=True, hidden=64)
        self.assertTrue(same_weights(wide.adapter, same.adapter))
        small = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=3, minibatch=True, hidden=8)
        self.assertEqual(small.parameter_steps, 280 * 3)
        # M-exp applies base + (w1 @ w2) tanh(read) + b without forming the product.
        exp = initial_adapter(5, hidden=None, nonlinear=True, expanded=True, device=cpu)
        w1, w2, b = (w * 100 for w in exp.weights)
        scaled = Adapter((w1, w2, b))
        direct = data.base + torch.einsum("ij,bjhw->bihw", w1 @ w2, data.read.tanh()) + b[None, :, None, None]
        self.assertTrue(torch.allclose(scaled.apply(data.read, data.base), direct, rtol=1e-5, atol=1e-6))
        fitted = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=3, minibatch=True, expanded=True)
        self.assertEqual(len(fitted.adapter.weights), 3)
        self.assertEqual(fitted.parameter_steps, 2064 * 3)
        self.assertIsNotNone(fitted.selection)

    def test_warm_start_reports_step0_selection(self) -> None:
        host, data = frozen_host(), fit_features()
        start = initial_adapter(77, hidden=8, nonlinear=True, expanded=False, device=torch.device("cpu"))
        warm = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=2, minibatch=True, hidden=8, start=start)
        self.assertEqual(warm.step0_selection, mean_loss(host, data, start))
        self.assertEqual(warm.report()["step0_selection_loss"], warm.step0_selection)
        # The first update starts from the given tensors, not from a fresh oracle-init draw.
        cold = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=2, minibatch=True, hidden=8)
        self.assertIsNone(cold.step0_selection)
        # Gate 3-shaped candidates (no snapshots, no warm start) report exactly the pre-M0 key set.
        gate3_keys = (
            "rate",
            "seed",
            "steps",
            "selection_loss",
            "finite",
            "training_losses",
            "wall_seconds",
            "example_passes",
            "parameter_steps",
        )
        self.assertEqual(tuple(cold.report()), gate3_keys)
        self.assertNotIn("snapshots", warm.report())
        self.assertFalse(same_weights(warm.adapter, cold.adapter))
        self.assertTrue(
            same_weights(
                start, initial_adapter(77, hidden=8, nonlinear=True, expanded=False, device=torch.device("cpu"))
            )
        )

    def test_edge_of_grid_flag_and_plateau_rule(self) -> None:
        self.assertTrue(at_edge(1.0, (1.0, 0.3, 0.1)))
        self.assertTrue(at_edge(0.1, (1.0, 0.3, 0.1)))
        self.assertFalse(at_edge(0.3, (1.0, 0.3, 0.1)))
        host, data = frozen_host(), fit_features()
        rates = (1.0, 0.3, 0.1)
        result = grid(host, data, data, 3000, rates=rates, checkpoints=(40, 80))
        self.assertEqual(
            sorted((c.rate, c.steps) for c in result.candidates), sorted((r, s) for r in rates for s in (40, 80))
        )
        self.assertIs(result.best, choose(result.candidates))
        self.assertEqual(result.edge_of_grid, at_edge(result.best.rate, rates))
        early = choose(tuple(c for c in result.candidates if c.steps == 40 and c.selection is not None))
        late = choose(tuple(c for c in result.candidates if c.steps == 80 and c.selection is not None))
        self.assertEqual(len(late.training_losses), 80)
        self.assertEqual(
            late.training_losses[:40],
            next(c for c in result.candidates if c.steps == 40 and c.rate == late.rate).training_losses,
        )
        se = paired_se(losses(host, data, early.adapter) - losses(host, data, late.adapter), 3000)
        tail = torch.tensor(late.training_losses[-80:], dtype=torch.float64)
        tail_se = float(tail.std(unbiased=True) / math.sqrt(80))
        change = float((tail[:40].mean() - tail[40:].mean()).abs())
        assert early.selection is not None and late.selection is not None
        expected = early.selection - late.selection < 2 * se and change < 2 * tail_se
        self.assertEqual(plateau(early, late, data, 3000, host), (expected, se, tail_se))
        self.assertEqual((result.plateau, result.selection_se, result.loss_tail_se), (expected, se, tail_se))
        report = result.report()
        self.assertEqual(report["edge_of_grid"], result.edge_of_grid)
        self.assertEqual(report["plateau_certified"], expected)
        self.assertTrue(all(c.example_passes == 64 * c.steps for c in result.candidates))
        self.assertTrue(all(c.parameter_steps == 272 * c.steps for c in result.candidates))
        # Each run's wall time is booked once, on its final-checkpoint candidate.
        self.assertTrue(all(c.wall_seconds == 0.0 for c in result.candidates if c.steps == 40))

    def test_plateau_rule_branches(self) -> None:
        host, data = frozen_host(), fit_features()
        rng = torch.Generator().manual_seed(4)
        moved = Adapter((torch.randn((16, 16), generator=rng) * 0.05, torch.randn(16, generator=rng) * 0.05))
        flat = tuple(1.0 if i % 2 else 1.1 for i in range(80))
        falling = tuple(2.0 - i / 80 for i in range(80))

        def make(adapter: Adapter, selection: float, trace: tuple[float, ...]) -> Candidate:
            return Candidate(adapter, 0.3, 3000, len(trace), selection, trace, 0.0, 1, 1)

        early = make(zero_adapter(torch.device("cpu")), 1.0, flat[:40])
        self.assertTrue(plateau(early, make(moved, 1.0, flat), data, 3000, host)[0])
        self.assertFalse(plateau(early, make(moved, 1.0, falling), data, 3000, host)[0])
        self.assertFalse(plateau(make(early.adapter, 2.0, flat[:40]), make(moved, 1.0, flat), data, 3000, host)[0])
        with self.assertRaises(ContractViolation):  # the late schedule must hold two 40-step windows
            plateau(early, make(moved, 1.0, flat[:79]), data, 3000, host)

    def test_grid_comparator_unavailable_keeps_finite_winner(self) -> None:
        host, data = frozen_host(), fit_features()
        adapter = zero_adapter(torch.device("cpu"))

        def run(
            host: Host,
            conditioning: Features,
            selection: Features,
            *,
            seed: int,
            rate: float,
            steps: int,
            minibatch: bool,
            start: Adapter | None,
            hidden: int | None,
            nonlinear: bool,
            expanded: bool,
            checkpoints: tuple[int, ...],
        ) -> Candidate:
            snapshots = ((40, adapter, 1.0 + rate),)  # every run fails between step 40 and step 80
            return Candidate(adapter, rate, seed, steps, None, (), 0.5, 10, 10, snapshots)

        with patch("sultai_screen.fit.fit_sgd", side_effect=run):
            result = grid(host, data, data, 3000, rates=(0.3, 0.1), checkpoints=(40, 80))
        self.assertEqual(len(result.candidates), 4)
        self.assertEqual((result.best.rate, result.best.steps), (0.1, 40))
        self.assertTrue(result.edge_of_grid)
        self.assertFalse(result.plateau)
        self.assertIsNone(result.selection_se)
        self.assertIn("unavailable", result.plateau_evidence)


if __name__ == "__main__":
    unittest.main()
