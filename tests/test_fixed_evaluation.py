"""CPU controls for the v4.1 fixed-shape frozen evaluator."""

import unittest

import torch
from torch import Tensor
from torch.nn import functional as F

from sultai.trust import ContractViolation
from sultai_screen.contracts import PilotHold
from sultai_screen.fit import EVAL_CHUNK, Adapter, identity_check, losses, mean_loss, zero_adapter
from sultai_screen.host import Features, Host, freeze

SEEN: list[tuple[int, int]] = []


class ShapeSensitiveHost(Host):
    """Logits depend on batch shape, standing in for shape-dependent GPU kernels."""

    def classify(self, site: Tensor) -> Tensor:
        logits = super().classify(site)
        if site.shape[0] != EVAL_CHUNK:
            logits = logits.clone()
            logits[:, 0] += 1e-3
        return logits


class RecordingAdapter(Adapter):
    def apply(self, read: Tensor, base: Tensor) -> Tensor:
        SEEN.append((read.shape[0], base.shape[0]))
        return super().apply(read, base)


def features(n: int, seed: int = 0) -> Features:
    rng = torch.Generator().manual_seed(seed)
    z = torch.rand((n, 16, 8, 8), generator=rng)
    return Features(z, z * 0.3, torch.randint(0, 10, (n,), generator=rng))


def reference(host: Host, data: Features, adapter: Adapter | None) -> Tensor:
    """Independent padded reference: last real row duplicated, module after padding, slice to real rows."""
    out = []
    with torch.no_grad():
        for start in range(0, len(data.labels), EVAL_CHUNK):
            read, base, labels = (t[start : start + EVAL_CHUNK] for t in (data.read, data.base, data.labels))
            real = len(labels)
            pad = EVAL_CHUNK - real
            if pad:
                read = torch.cat((read, read[-1:].expand(pad, *read.shape[1:])))
                base = torch.cat((base, base[-1:].expand(pad, *base.shape[1:])))
                labels = torch.cat((labels, labels[-1:].expand(pad)))
            values = base if adapter is None else adapter.apply(read, base)
            out.append(F.cross_entropy(host.classify(values), labels, reduction="none")[:real])
    return torch.cat(out)


class FixedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_every_chunk_seen_by_module_and_classifier_has_exactly_128_rows(self) -> None:
        host = Host(4)
        freeze(host)
        sizes: list[int] = []

        def record(_module: torch.nn.Module, inputs: tuple[Tensor, ...]) -> None:
            sizes.append(inputs[0].shape[0])

        handle = host.tail.register_forward_pre_hook(record)
        for n in (1, 127, 128, 129, 255, 256, 257):
            sizes.clear()
            SEEN.clear()
            values = losses(host, features(n), RecordingAdapter(zero_adapter(torch.device("cpu")).weights))
            self.assertEqual(len(values), n)
            self.assertEqual(len(sizes), -(-n // EVAL_CHUNK))
            self.assertEqual(set(sizes), {EVAL_CHUNK})
            self.assertEqual(set(SEEN), {(EVAL_CHUNK, EVAL_CHUNK)})
        handle.remove()

    def test_real_losses_match_independent_padded_reference_in_caller_order(self) -> None:
        host = Host(4)
        freeze(host)
        rng = torch.Generator().manual_seed(7)
        adapter = Adapter((torch.randn((16, 16), generator=rng) * 0.01, torch.randn(16, generator=rng) * 0.01))
        for n in (1, 129, 257):
            data = features(n, seed=n)
            self.assertTrue(torch.equal(losses(host, data, adapter), reference(host, data, adapter)))
            self.assertTrue(torch.equal(losses(host, data, None), reference(host, data, None)))
            self.assertEqual(mean_loss(host, data, None), float(reference(host, data, None).double().mean()))

    def test_shape_sensitive_classifier_is_bitwise_invariant_under_global_permutation(self) -> None:
        host = ShapeSensitiveHost(4)
        freeze(host)
        data = features(300)
        identity_check(host, data, 3000)
        order = torch.randperm(300, generator=torch.Generator().manual_seed(1))
        twin = Features(data.read[order], data.base[order], data.labels[order])
        self.assertTrue(torch.equal(losses(host, data, None), losses(host, twin, None)[torch.argsort(order)]))

    def test_shape_sensitive_control_differs_from_unpadded_whole_batch_evaluation(self) -> None:
        host = ShapeSensitiveHost(4)
        freeze(host)
        data = features(300)
        with torch.no_grad():
            direct = F.cross_entropy(host.classify(data.base), data.labels, reduction="none")
        self.assertFalse(torch.equal(direct, losses(host, data, None)))

    def test_real_host_is_bitwise_invariant_under_global_permutation(self) -> None:
        host = Host(4)
        freeze(host)
        identity_check(host, features(300), 3000)

    def test_labels_travel_with_their_examples(self) -> None:
        host = Host(4)
        freeze(host)
        data = features(130)
        rolled = Features(data.read, data.base, torch.roll(data.labels, 1))
        self.assertFalse(torch.equal(losses(host, data, None), losses(host, rolled, None)))

    def test_real_last_row_failure_is_not_hidden_by_padding(self) -> None:
        host = Host(4)
        freeze(host)
        data = features(129)
        base = data.base.clone()
        base[-1] = float("nan")
        bad = Features(data.read, base, data.labels)
        with self.assertRaises(PilotHold):
            losses(host, bad, None)
        values = losses(host, bad, None, check_finite=False)
        self.assertEqual(len(values), 129)
        self.assertFalse(bool(torch.isfinite(values[-1])))
        self.assertTrue(bool(torch.isfinite(values[:-1]).all()))

    def test_empty_or_misaligned_features_are_refused(self) -> None:
        host = Host(4)
        freeze(host)
        data = features(5)
        for broken in (
            Features(data.read[:0], data.base[:0], data.labels[:0]),
            Features(data.read, data.base[:-1], data.labels),
            Features(data.read, data.base, data.labels[:-1]),
        ):
            with self.assertRaises(ContractViolation):
                losses(host, broken, None)


if __name__ == "__main__":
    unittest.main()
