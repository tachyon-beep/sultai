"""CPU controls for the separately authorized no-op twin diagnostic."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sultai.trust import ContractViolation
from sultai_screen.contracts import PilotHold, publish, read_record
from sultai_screen.data import generator
from sultai_screen.fit import losses
from sultai_screen.host import Features, Host, freeze
from sultai_screen.twin_diagnostic import (
    AUTHORIZATION,
    CUSTODY,
    OPENING_CHARGE,
    diagnose,
    differences,
    evaluate,
    permutation_orders,
    run_child,
    supervise,
)


class TwinDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_orders_preserve_full_permutation_and_batch_size(self):
        orders = permutation_orders(5000, 3000, torch.device("cpu"))
        expected = torch.randperm(5000, generator=generator(3000, "noop-evaluation-twin"))
        self.assertTrue(torch.equal(orders["global"], expected))
        self.assertTrue(torch.equal(orders["preserve_batch_size"].sort().values, torch.arange(5000)))
        self.assertTrue(bool((orders["preserve_batch_size"][:4992] < 4992).all()))
        self.assertTrue(bool((orders["preserve_batch_size"][4992:] >= 4992).all()))
        for count in (0, 128, 256):
            with self.assertRaises(ContractViolation):
                permutation_orders(count, 3000, torch.device("cpu"))

    def test_evaluation_matches_production_slicing_and_inverse(self):
        host = Host()
        freeze(host)
        z = torch.randn(129, 16, 8, 8)
        site = Features(z, z, torch.arange(129) % 10)
        order = permutation_orders(129, 3000, torch.device("cpu"))["global"]
        baseline = evaluate(host, site, torch.arange(129), padded=False)
        self.assertTrue(torch.equal(baseline.loss, losses(host, site, None)))
        twin = Features(z[order], z[order], site.labels[order])
        actual = evaluate(host, site, order, padded=False)
        self.assertTrue(torch.equal(actual.loss, losses(host, twin, None)[torch.argsort(order)]))
        self.assertEqual(baseline.batch_sizes[-1].item(), 1)
        self.assertTrue(bool((baseline.batch_sizes[:-1] == 128).all()))
        with self.assertRaises(ContractViolation):
            evaluate(host, site, torch.zeros(129, dtype=torch.long), padded=False)

    def test_batch_shape_fault_localization_and_padded_control(self):
        host = Host()
        freeze(host)
        z = torch.zeros(129, 16, 8, 8)
        site = Features(z, z, torch.zeros(129, dtype=torch.long))

        def shape_sensitive(values):
            count = len(values)
            return torch.stack((torch.full((count,), count / 128), torch.zeros(count)), dim=1)

        with tempfile.TemporaryDirectory() as temporary, patch.object(host, "classify", side_effect=shape_sensitive):
            root = Path(temporary)
            result = diagnose(host, site, root)
            self.assertFalse(result["original_twin_bitwise_equal"])
            self.assertFalse(result["pilot_ready"])
            self.assertFalse(result["confirmation_ready"])
            cases = result["cases"]
            self.assertTrue(cases["identical_repeat"]["loss"]["bitwise_equal"])
            self.assertTrue(cases["preserve_batch_size"]["loss"]["bitwise_equal"])
            self.assertTrue(result["padded_pair"]["logits"]["bitwise_equal"])
            self.assertTrue(result["padded_pair"]["loss"]["bitwise_equal"])
            original = cases["original_global_permutation"]
            changed = [
                i
                for i, (a, b) in enumerate(
                    zip(original["baseline_batch_sizes"], original["actual_batch_sizes"], strict=True)
                )
                if a != b
            ]
            self.assertEqual(original["loss"]["mismatch_indices"], changed)
            self.assertEqual(original["logits"]["mismatch_indices"], changed)
            self.assertTrue(original["ce_of_baseline_logits_in_case_order"]["bitwise_equal"])
            self.assertFalse(original["case_logits_ce_in_identity_order"]["bitwise_equal"])
            self.assertEqual(len(list(root.glob("comparison-*.json"))), 5)
            self.assertTrue(all(v["bitwise_equal"] for v in result["round_trip"].values()))

    def test_difference_exactness_and_nonfinite_rejection(self):
        baseline = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        actual = baseline.clone()
        actual[1, 0] += 0.125
        report = differences(baseline, actual)
        self.assertEqual(report["mismatch_indices"], [1])
        self.assertEqual(report["mismatch_count"], 1)
        self.assertEqual(report["max_absolute_difference"], 0.125)
        self.assertNotEqual(report["baseline_sha256"], report["actual_sha256"])
        with self.assertRaises(PilotHold):
            differences(torch.tensor([float("nan")]), torch.tensor([0.0]))
        with self.assertRaises(ContractViolation):
            differences(baseline, actual[:1])

    def test_one_diagnostic_custody_across_output_paths_and_environment_failure(self):
        approved = {
            "diagnostic_approved": True,
            "diagnostic_max_seconds": 120,
            "opening_charge_gpu_process_seconds": OPENING_CHARGE,
            "evidence": [{"path": AUTHORIZATION}],
        }
        charge = {"child_returncode": 0, "timed_out": False, "diagnostic_charge_gpu_process_seconds": 1.0}

        def complete(output):
            publish(output / "worker-result.json", {"software_valid": True})
            return charge

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cert = root / "certificate.json"
            publish(cert, approved)
            (root / ".weft").mkdir()
            with (
                patch("sultai_screen.twin_diagnostic.REPO", root),
                patch("sultai_screen.twin_diagnostic.check_certificate", return_value={"source_sha256": "fixture"}),
                patch("sultai_screen.twin_diagnostic.verify_prior", return_value={}),
                patch("sultai_screen.twin_diagnostic.check_handover", return_value={}),
                patch("sultai_screen.twin_diagnostic.environment_identity", side_effect=PilotHold("mismatch")),
                patch("sultai_screen.twin_diagnostic.run_child", side_effect=complete) as child,
            ):
                with self.assertRaisesRegex(PilotHold, "mismatch"):
                    supervise(root / "one", cert)
                self.assertFalse((root / CUSTODY).exists())
                child.assert_not_called()
                with patch("sultai_screen.twin_diagnostic.environment_identity", return_value={}):
                    for key, value in (
                        ("diagnostic_approved", False),
                        ("diagnostic_max_seconds", 121),
                        ("opening_charge_gpu_process_seconds", 0.0),
                        ("evidence", []),
                    ):
                        bad_cert = root / (key + ".json")
                        publish(bad_cert, dict(approved, **{key: value}))
                        with self.assertRaises(PilotHold):
                            supervise(root / "bad-input", bad_cert)
                        self.assertFalse((root / CUSTODY).exists())
                        child.assert_not_called()
                    self.assertEqual(supervise(root / "one", cert), 0)
                    original = (root / CUSTODY).read_bytes()
                    with self.assertRaisesRegex(PilotHold, "already exists"):
                        supervise(root / "two", cert)
                    self.assertEqual((root / CUSTODY).read_bytes(), original)
                    self.assertEqual(child.call_count, 1)

    def test_real_cpu_timeout_failure_and_carried_charge(self):
        import subprocess

        real_popen = subprocess.Popen

        def child(command, **kwargs):
            return real_popen(["/usr/bin/python3", "-B", "-c", "import time;time.sleep(30)"], **kwargs)

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("sultai_screen.twin_diagnostic.subprocess.Popen", side_effect=child),
            patch("sultai_screen.twin_diagnostic.MAX_SECONDS", 5.15),
        ):
            charge = run_child(Path(temporary))
            self.assertTrue(charge["timed_out"])
            self.assertNotEqual(charge["child_returncode"], 0)
            self.assertGreater(charge["diagnostic_charge_gpu_process_seconds"], 0)
            self.assertLess(charge["diagnostic_charge_gpu_process_seconds"], 5.15)
            self.assertEqual(
                charge["total_charge_gpu_process_seconds"],
                OPENING_CHARGE + charge["diagnostic_charge_gpu_process_seconds"],
            )
            pid = read_record(Path(temporary) / "process.json")["pid"]
            self.assertFalse(Path(f"/proc/{pid}").exists())


if __name__ == "__main__":
    unittest.main()
