from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch import Tensor

from sultai.trust import ContractViolation, InputDataError
from sultai_screen.contracts import PilotHold
from sultai_screen.data import Dataset
from sultai_screen.host import (
    COOLED_20,
    TCOOL,
    Host,
    constant,
    linear_cooldown,
    load_host,
    make_host,
    momentum_buffers,
    optimizer_state,
    parameter_count,
    recipe_optimizer,
    restore_optimizer,
    state_hash,
    train_epochs,
    train_host,
)

CPU = torch.device("cpu")
REFERENCE_NAMES = [
    "front.0.weight",
    "front.0.bias",
    "front.1.weight",
    "front.1.bias",
    "front.3.weight",
    "front.3.bias",
    "front.4.weight",
    "front.4.bias",
    "front.7.weight",
    "front.7.bias",
    "front.8.weight",
    "front.8.bias",
    "tail.0.weight",
    "tail.0.bias",
    "tail.1.weight",
    "tail.1.bias",
    "tail.3.weight",
    "tail.3.bias",
    "tail.4.weight",
    "tail.4.bias",
    "tail.7.weight",
    "tail.7.bias",
    "tail.8.weight",
    "tail.8.bias",
    "tail.10.weight",
    "tail.10.bias",
    "tail.11.weight",
    "tail.11.bias",
    "classifier.weight",
    "classifier.bias",
]
REFERENCE_HASH = "41d64c71e61d5310e26a7c33b624713b2047a0e1303de1a293e9213eccf26b12"
BOTTLENECK_HASH = "68c71ac0ba8970979469b8ed13c8fb5be8e85ea45ac32e06b82ac919d4dc083e"
# Pinned against the pre-refactor train_host (commit 25a2b25) on the fixture below.
GOLDEN_FINAL_HASH = "17325d1d563158daf85a23b57fccf7bd5456ca8847a27c3065d89fda3ac2eee4"
GOLDEN_STREAM_SHA = "56f5bff8e3e2d61923feafb9137742ebbe51b255b53a062fe90da4efbcf2ba1f"


def fixture() -> Dataset:
    source = torch.Generator(device="cpu").manual_seed(20261014)
    images = torch.randint(0, 256, (64, 3, 32, 32), generator=source, dtype=torch.uint8)
    labels = torch.randint(0, 10, (64,), generator=source)
    return Dataset(images, labels, torch.arange(64), torch.arange(8), torch.arange(8), torch.arange(8))


def stream_sha(stream: Tensor) -> str:
    return hashlib.sha256(stream.numpy().tobytes()).hexdigest()


class HostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_train_host_golden_pin_is_unchanged(self) -> None:
        hashes, stream, _ = train_host(make_host(3000, 0, CPU), fixture(), 3000)
        self.assertEqual(len(hashes), 10)
        self.assertEqual(stream.shape, (20,))
        self.assertEqual(hashes[-1], GOLDEN_FINAL_HASH)
        self.assertEqual(stream_sha(stream), GOLDEN_STREAM_SHA)

    def test_train_host_is_train_epochs_with_constant_rate(self) -> None:
        host = make_host(3000, 0, CPU)
        optimizer = recipe_optimizer(host.parameters())
        hashes, stream, _ = train_epochs(
            host, host, fixture(), 3000, epochs=range(10), rates=constant(0.05), optimizer=optimizer
        )
        self.assertEqual(hashes[-1], GOLDEN_FINAL_HASH)
        self.assertEqual(stream_sha(stream), GOLDEN_STREAM_SHA)
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [5e-4, 0.0])
        with self.assertRaises(ContractViolation):
            train_host(make_host(3000, 0, CPU), fixture(), 3000, epochs=9)

    def test_linear_cooldown_values(self) -> None:
        self.assertEqual(TCOOL(0), 0.05)
        self.assertEqual(TCOOL(12_499), 0.05)
        self.assertEqual(TCOOL(12_500), 0.05)
        self.assertAlmostEqual(TCOOL(12_500 + 3_125), 0.025)
        self.assertAlmostEqual(TCOOL(18_749), 0.05 / 6_250)
        self.assertEqual(COOLED_20(18_749), 0.05)
        self.assertAlmostEqual(COOLED_20(25_000 - 1), 0.05 / 6_250)
        self.assertEqual(constant(0.05)(99_999), 0.05)
        with self.assertRaises(ContractViolation):
            linear_cooldown(10, 10)

    def test_linear_cooldown_is_zero_at_and_after_stop(self) -> None:
        for schedule, stop in ((TCOOL, 18_750), (COOLED_20, 25_000)):
            self.assertEqual(schedule(stop), 0.0)
            self.assertEqual(schedule(stop + 1), 0.0)
            self.assertEqual(schedule(stop + 10_000), 0.0)
            self.assertGreater(schedule(stop - 1), 0.0)

    def test_resume_from_saved_momentum_continues_the_stream_bitwise(self) -> None:
        # Two steps per epoch; the cooldown starts mid-epoch 0, so a resume that restarted
        # the global step counter at 0 rather than 2 would train epoch 1 at the wrong rates.
        data, rates = fixture(), linear_cooldown(1, 4)
        whole = make_host(3000, 0, CPU)
        hashes, stream, _ = train_epochs(
            whole, whole, data, 3000, epochs=range(2), rates=rates, optimizer=recipe_optimizer(whole.parameters())
        )
        first = make_host(3000, 0, CPU)
        optimizer = recipe_optimizer(first.parameters())
        with self.assertRaises(ContractViolation):
            momentum_buffers(optimizer, first)
        _, head, _ = train_epochs(first, first, data, 3000, epochs=range(1), rates=rates, optimizer=optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "optimizer.pt"
            torch.save(momentum_buffers(optimizer, first), path)
            loaded: object = torch.load(path, map_location=CPU, weights_only=True)
        buffers = optimizer_state(loaded, dict(first.named_parameters()))
        self.assertEqual(list(buffers), REFERENCE_NAMES)

        def resume(saved: dict[str, Tensor]) -> tuple[str, Tensor]:
            host = make_host(3000, 0, CPU)
            host.load_state_dict(first.state_dict(), strict=True)
            resumed = restore_optimizer(host, saved)
            tail_hashes, tail, _ = train_epochs(
                host, host, data, 3000, epochs=range(1, 2), rates=rates, optimizer=resumed
            )
            return tail_hashes[-1], tail

        final, tail = resume(buffers)
        self.assertEqual(final, hashes[-1])
        self.assertEqual(stream_sha(torch.cat([head, tail])), stream_sha(stream))

        dropped = dict(buffers)
        del dropped["tail.4.weight"]
        with self.assertRaises(InputDataError):
            optimizer_state(dropped, dict(first.named_parameters()))
        with self.assertRaises(ContractViolation):
            restore_optimizer(make_host(3000, 0, CPU), dropped)
        zeroed = dict(buffers)
        zeroed["tail.4.weight"] = torch.zeros_like(buffers["tail.4.weight"])
        self.assertNotEqual(resume(zeroed)[0], hashes[-1])

    def test_optimizer_state_boundary_rejects_missing_keys_wrong_shape_and_nonfinite(self) -> None:
        expected: dict[str, Tensor] = {"a": torch.zeros(2, 3), "b": torch.zeros(4)}
        good = {"a": torch.ones(2, 3), "b": torch.ones(4)}
        self.assertEqual(optimizer_state(good, expected), good)
        for bad in (
            [torch.ones(2, 3), torch.ones(4)],
            {"a": torch.ones(2, 3)},
            {"a": torch.ones(2, 3), "b": torch.ones(4), "c": torch.ones(1)},
            {"a": torch.ones(3, 2), "b": torch.ones(4)},
            {"a": torch.ones(2, 3, dtype=torch.float64), "b": torch.ones(4)},
            {"a": torch.ones(2, 3), "b": torch.tensor([1.0, float("nan"), 1.0, 1.0])},
            {"a": torch.ones(2, 3), "b": torch.tensor([1.0, float("inf"), 1.0, 1.0])},
            {"a": torch.ones(2, 3), "b": [1.0, 1.0, 1.0, 1.0]},
        ):
            with self.assertRaises(InputDataError):
                optimizer_state(bad, expected)

    def test_load_host_unfrozen_allows_a_train_mode_forward_and_frozen_raises_on_it(self) -> None:
        source = make_host(3000, 0, CPU, widths=(16, 16, 32, 32))
        images = torch.zeros(2, 3, 32, 32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "host.pt"
            torch.save(dict(source.state_dict()), path)
            thawed = load_host(path, 3000, 0, CPU, widths=(16, 16, 32, 32), frozen=False)
            sealed = load_host(path, 3000, 0, CPU, widths=(16, 16, 32, 32))
        self.assertFalse(source.sealed)
        self.assertNotIn("sealed", source.state_dict())
        self.assertFalse(thawed.sealed)
        self.assertEqual(state_hash(thawed), state_hash(source))
        thawed.train()
        self.assertEqual(thawed(images).shape, (2, 10))
        self.assertTrue(sealed.sealed)
        self.assertEqual(state_hash(sealed), state_hash(source))
        sealed.train()
        with self.assertRaises(PilotHold):
            sealed(images)

    def test_reference_host_hash_and_count_are_unchanged(self) -> None:
        host = make_host(3000, 0, CPU)
        self.assertEqual(state_hash(host), REFERENCE_HASH)
        self.assertEqual(sum(p.numel() for p in host.parameters()), 284_634)
        self.assertEqual([n for n, _ in host.named_parameters()], REFERENCE_NAMES)
        self.assertEqual(state_hash(make_host(3000, 4, CPU)), BOTTLENECK_HASH)

    def test_frontier_counts_match_the_frozen_table(self) -> None:
        for widths, count in (
            ((16, 16, 32, 32), 33_978),
            ((32, 32, 64, 64), 85_274),
            ((48, 48, 96, 96), 168_826),
            ((64, 64, 128, 128), 284_634),
            ((96, 96, 192, 192), 613_018),
        ):
            host = Host(0, widths=widths)
            self.assertEqual(parameter_count(host), count)
            self.assertEqual(sum(p.numel() for p in host.front.parameters()), 14_928)

    def test_explicit_default_tuple_is_bitwise_the_reference(self) -> None:
        a = make_host(3000, 0, CPU)
        b = make_host(3000, 0, CPU, widths=(64, 64, 128, 128))
        self.assertEqual(state_hash(a), state_hash(b))

    def test_bottleneck_and_tuple_compose(self) -> None:
        host = Host(4, widths=(32, 32, 64, 64))
        self.assertEqual(host(torch.zeros(2, 3, 32, 32)).shape, (2, 10))

    def test_invalid_widths_are_rejected(self) -> None:
        for bad in ((0, 16, 32, 32), (-1, 16, 32, 32), (16, 16, 32, 2.5), (True, 16, 32, 32)):
            with self.assertRaises(ContractViolation):
                Host(0, widths=bad)
        with self.assertRaises(ContractViolation):
            Host(3)


if __name__ == "__main__":
    unittest.main()
