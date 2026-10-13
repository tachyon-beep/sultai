from __future__ import annotations

import unittest

import torch

from sultai.trust import ContractViolation
from sultai_screen.host import Host, make_host, parameter_count, state_hash

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


class HostTests(unittest.TestCase):
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
