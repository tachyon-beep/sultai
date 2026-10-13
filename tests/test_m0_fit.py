from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import torch

from sultai.trust import ContractViolation
from sultai_screen.data import Dataset, partitions
from sultai_screen.fit import Adapter, accuracy, logits, losses, mean_accuracy
from sultai_screen.host import Features, Host, freeze
from sultai_screen.m0.data import SEEN_FIT_SHA, features_from_images, seen_fit

REPO = Path(__file__).resolve().parents[1]
DEV_ARTIFACT = REPO / "docs/results/gate3-split-2026-10-11/dev-indices-int32-le.bin"
# Pinned from the unedited losses() on the 129-row fixture below (controller ruling R3).
LOSSES_SHA = "343a6bf756878ae0000abeb5f4a72eaa35c2a843c5b5e7cfe6602a7e0c2b6067"


def fixture_features(n: int = 129) -> Features:
    rng = torch.Generator().manual_seed(5)
    z = torch.rand((n, 16, 8, 8), generator=rng)
    return Features(z, z * 0.3, torch.randint(0, 10, (n,), generator=rng))


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


if __name__ == "__main__":
    unittest.main()
