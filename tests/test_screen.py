"""Short CPU known answers and negative controls for the GPU screen's scientific contracts."""

import itertools
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sultai.repair import Example, fit
from sultai.trust import ContractViolation, InputDataError
from sultai_screen.alignment import hungarian, transport
from sultai_screen.contracts import (
    PREREG_SHA,
    FeasibilityHold,
    InstrumentHold,
    OracleUnavailable,
    PilotHold,
    anchor_count,
    derive,
    file_sha,
    json_value,
    number,
    publish,
    read_record,
)
from sultai_screen.costs import project_cost
from sultai_screen.data import Seal, as_tensor, cifar_batch, partitions
from sultai_screen.fit import Adapter, choose, fit_sgd, gradient_step, identity_check, losses, ridge, zero_adapter
from sultai_screen.host import Cached, Features, Host, assert_frozen, augment, checkpoint_state, freeze, state_hash
from sultai_screen.impair import make_impairment
from sultai_screen.pilot import (
    Job,
    Supervisor,
    check_certificate,
    check_timing,
    claim_attempt,
    magnitude_decision,
    stop_report,
    verify_replay,
)
from sultai_screen.screen import _site, environment_identity, materiality_worker, source_identity


class ScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_json_boundary(self) -> None:
        self.assertEqual(json_value({"required": [0, False, None, 1.5]}), {"required": [0, False, None, 1.5]})
        for value in ({1: "bad"}, float("nan"), float("inf"), object(), (1, 2)):
            with self.assertRaises(InputDataError):
                json_value(value)
        for value in (True, None, "0", float("nan")):
            with self.assertRaises(InputDataError):
                number(value)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            publish(path, {"required": 0})
            self.assertEqual(read_record(path), {"required": 0})
            with self.assertRaises(FileExistsError):
                publish(path, {"required": 1})
            path.write_text('{"required":1,"required":2}')
            with self.assertRaises(InputDataError):
                read_record(path)

    def test_tensor_boundary(self) -> None:
        tensor = torch.zeros(2, 3)
        self.assertIs(as_tensor(tensor), tensor)
        for value in (None, np.zeros((2, 3)), [1, 2]):
            with self.assertRaises(InputDataError):
                as_tensor(value)

    def test_pickle_boundary(self) -> None:
        data = np.zeros((10000, 3072), dtype=np.uint8)
        labels = [0] * 10000
        images, target = cifar_batch({b"data": data, b"labels": labels})
        self.assertEqual(images.shape, (10000, 3, 32, 32))
        self.assertEqual(target.dtype, torch.long)
        for value in (
            {},
            {b"data": data},
            {b"data": data, b"labels": [True] * 10000},
            {b"data": data[:1], b"labels": labels},
            {b"data": data.astype(np.float32), b"labels": labels},
            {b"data": data, b"labels": [10] * 10000},
        ):
            with self.assertRaises(InputDataError):
                cifar_batch(value)

    def test_checkpoint_boundary(self) -> None:
        expected = {"weight": torch.zeros(2, 3)}
        self.assertEqual(checkpoint_state({"weight": torch.ones(2, 3)}, expected)["weight"].shape, (2, 3))
        for value in (
            {},
            {"weight": None},
            {"weight": torch.zeros(3, 2)},
            {"weight": torch.ones(2, 3, dtype=torch.float64)},
            {"weight": torch.full((2, 3), float("nan"))},
            {"weight": torch.zeros(2, 3), "extra": torch.zeros(1)},
        ):
            with self.assertRaises(InputDataError):
                checkpoint_state(value, expected)

    def test_seal_and_split_negative_controls(self) -> None:
        seal = Seal()
        for value in ("/tmp/test_batch", b"/tmp/test_batch", "/tmp/test_batch.copy"):
            with self.assertRaises(PilotHold):
                seal.hook("open", (value,))
        seal.hook("open", ("/tmp/data_batch_1",))
        self.assertEqual(seal.report(), {"data_batch_1": 1})
        artifact = Path(__file__).resolve().parents[1] / "docs/results/gate3-split-2026-10-11/dev-indices-int32-le.bin"
        parts = partitions(artifact)
        self.assertEqual(tuple(len(p) for p in parts), (40000, 2000, 3000, 5000))
        self.assertEqual(len(torch.unique(torch.cat(parts))), 50000)
        self.assertEqual(derive(20261004, "cifar-fit-dev-v1"), 6798730733443669431)
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "indices.bin"
            altered.write_bytes(artifact.read_bytes()[:-4] + b"BAD!")
            with self.assertRaises(PilotHold):
                partitions(altered)

    def test_impairment_exact_inverse_and_identity(self) -> None:
        source = torch.rand((8, 16, 16, 16), generator=torch.Generator().manual_seed(2))
        for family, magnitude in (("P1", 0.03), ("P2", 0.03), ("P3", 1.0), ("P3prime", 0.5)):
            operator = make_impairment(family, magnitude, 3000, source)
            impaired = operator.apply(source)
            self.assertLess(float((operator.undo(impaired) - source).abs().max()), 1e-4)
            self.assertGreater(float((impaired - source).abs().max()), 0.01)
            self.assertTrue(torch.equal(zero_adapter(source.device).apply(source, impaired), impaired))
            self.assertFalse(torch.equal(operator.undo(impaired + 0.1), source))

    def test_ridge_matches_existing_solver(self) -> None:
        rng = torch.Generator().manual_seed(3)
        h = torch.randn((64, 16, 1, 1), generator=rng, dtype=torch.float64)
        w = torch.randn((16, 16), generator=rng, dtype=torch.float64) * 0.1
        b = torch.randn(16, generator=rng, dtype=torch.float64) * 0.1
        target = Adapter((w, b)).apply(h, h)
        pairs = tuple(
            Example(str(i), tuple(float(v) for v in h[i, :, 0, 0]), tuple(float(v) for v in target[i, :, 0, 0]))
            for i in range(64)
        )
        expected = fit(pairs)
        actual = ridge(h, h, target, nonlinear=True)
        self.assertLess(
            float((actual.weights[0] - torch.tensor(expected.weights, dtype=torch.float64)).abs().max()), 1e-8
        )
        self.assertLess(float((actual.weights[1] - torch.tensor(expected.bias, dtype=torch.float64)).abs().max()), 1e-8)
        affine_target = h + torch.einsum("ij,bjhw->bihw", w, h) + b[None, :, None, None]
        recovered = ridge(h, h, affine_target, nonlinear=False)
        self.assertLess(float((recovered.apply(h, h) - affine_target).abs().max()), 1e-7)
        with self.assertRaises(ContractViolation):
            ridge(h, h[:1], target, nonlinear=True)

    def test_frozen_fit_and_full_batch_gradient(self) -> None:
        torch.manual_seed(12)
        host = Host()
        freeze(host)
        original = state_hash(host)
        read = torch.rand((129, 16, 8, 8), generator=torch.Generator().manual_seed(4))
        data = Features(read, read * 0.3, torch.arange(129) % 10)
        adapter = zero_adapter(read.device)
        w, b = (p.requires_grad_(True) for p in adapter.weights)
        direct = torch.nn.functional.cross_entropy(host.classify(Adapter((w, b)).apply(read, data.base)), data.labels)
        gradients = torch.autograd.grad(direct, (w, b))
        actual, _ = gradient_step(host, data, adapter, 0.1, torch.arange(129))
        for found, initial, gradient in zip(actual.weights, (w, b), gradients, strict=True):
            self.assertTrue(torch.allclose(found, initial - 0.1 * gradient, atol=1e-7))
        fitted = fit_sgd(host, data, data, seed=3000, rate=0.3, steps=4, minibatch=False)
        self.assertIsNotNone(fitted.selection)
        assert_frozen(host, original)
        self.assertTrue(torch.equal(losses(host, data, None), losses(host, data, zero_adapter(read.device))))
        host.front[1].train()
        with self.assertRaises(PilotHold):
            losses(host, data, None)

    def test_nonfinite_candidate_retained_and_never_selected(self) -> None:
        host = Host()
        freeze(host)
        read = torch.ones(4, 16, 8, 8)
        data = Features(read, read, torch.arange(4))
        failed = fit_sgd(host, data, data, seed=3000, rate=1e300, steps=2, minibatch=False)
        self.assertIsNone(failed.selection)
        from sultai_screen.contracts import OracleUnavailable

        with self.assertRaises(OracleUnavailable):
            choose((failed,))
        good = fit_sgd(host, data, data, seed=3000, rate=0.1, steps=1, minibatch=False)
        self.assertIs(choose((failed, good)), good)
        with self.assertRaises(PilotHold):
            losses(host, data, Adapter((torch.full((16, 16), float("nan")), torch.zeros(16))))

    def test_hungarian_and_two_sided_transport(self) -> None:
        cost = ((3.0, 1.0, 2.0), (4.0, 8.0, 1.0), (1.0, 2.0, 7.0))
        assignment = hungarian(cost)
        score = sum(cost[i][j] for i, j in enumerate(assignment))
        self.assertEqual(score, min(sum(cost[i][j] for i, j in enumerate(p)) for p in itertools.permutations(range(3))))
        pin = torch.eye(16)[torch.roll(torch.arange(16), 1)]
        pout = torch.eye(16)[torch.roll(torch.arange(16), 3)]
        w = torch.arange(256, dtype=torch.float32).reshape(16, 16)
        b = torch.arange(16, dtype=torch.float32)
        found = transport(Adapter((w, b)), pin, pout)
        self.assertTrue(torch.equal(found.weights[0], pout.T @ w @ pin))
        self.assertTrue(torch.equal(found.weights[1], pout.T @ b))
        self.assertFalse(torch.equal(found.weights[0], pout.T @ w @ pout))

    def test_sizing_uses_sd_formula_and_never_caps(self) -> None:
        self.assertEqual(anchor_count(0.03), (39, 39))
        self.assertEqual(anchor_count(0), (0, 32))
        required, count = anchor_count(1.0)
        self.assertGreater(required, 200)
        self.assertEqual(required, count)
        for value in (float("nan"), -1.0, float("inf")):
            with self.assertRaises(ContractViolation):
                anchor_count(value)

    def test_negative_instrument_is_valid_execution(self) -> None:
        for error in (InstrumentHold("known answer"), OracleUnavailable("all failed")):
            report = stop_report(error, "known-answer", 123.0)
            self.assertTrue(report["software_valid"])
            self.assertFalse(report["instrument_valid"])
            self.assertFalse(report["pilot_clean"])
            self.assertEqual(report["verdict"], "instrument-failed")
        report = stop_report(FeasibilityHold("count too high"), "sizing", 321.0)
        self.assertTrue(report["software_valid"])
        self.assertIsNone(report["instrument_valid"])
        self.assertFalse(report["confirmatory_ready"])
        for error in (PilotHold("replay mismatch"), ContractViolation("owned state"), InputDataError("missing")):
            self.assertFalse(stop_report(error, "execution", 0.0)["software_valid"])

    def test_timing_admission_and_replay_negative_controls(self) -> None:
        good = {"fit_finite": True, "fit_state_sha256": ["a", "b"], "fit_trace_sha256": "c"}
        self.assertEqual(check_timing((good, good.copy(), good.copy())), good)
        for key, value, error in (("fit_finite", False, InstrumentHold), ("fit_trace_sha256", "bad", PilotHold)):
            bad = good.copy()
            bad[key] = value
            with self.assertRaises(error):
                check_timing((good, bad, good))
        replay = {
            "epoch_state_sha256": ["a"],
            "loss_stream_sha256": "b",
            "final_state_sha256": "c",
            "forward_backward": {"gradient": "d"},
        }
        verify_replay(replay, replay.copy())
        for key in replay:
            bad = replay.copy()
            bad[key] = "altered"
            with self.assertRaises(PilotHold):
                verify_replay(replay, bad)

    def test_complete_projection_counts_all_frozen_arms(self) -> None:
        timing = {
            "fit_finite": True,
            "wide_grid_finite": True,
            "full_batch_step_seconds": 1.0,
            "wide_grid_seconds": 2.0,
            "retrieval_pair_seconds": 0.01,
            "development_evaluation_seconds": 0.1,
        }
        kwargs = {
            "timing": (timing,),
            "intact_seconds": (3.0,),
            "bottleneck_seconds": (4.0,),
            "oracle_worker_seconds": (5.0,),
            "posthoc_seconds": (6.0,),
        }
        result = project_cost(32, **kwargs)
        lines = result["line_gpu_process_seconds"]
        self.assertEqual(result["headroom_host_cases"], 128)
        self.assertEqual(result["oracle_host_cases"], 192)
        self.assertEqual(result["full_batch_steps_per_headroom_case"], 446)
        self.assertEqual(result["ordered_retrieval_pairs"], 2 * 32 * 31 + 64 * 63)
        self.assertEqual(lines["B5_eight_P6_anchors"], 16.0)
        self.assertEqual(result["projected_gpu_process_seconds"], sum(lines.values()))
        self.assertGreater(
            project_cost(39, **kwargs)["projected_gpu_process_seconds"], result["projected_gpu_process_seconds"]
        )
        for key in timing:
            if key.endswith("seconds"):
                bad = timing.copy()
                del bad[key]
                with self.assertRaises(KeyError):
                    project_cost(32, **dict(kwargs, timing=(bad,)))
        with self.assertRaises(PilotHold):
            project_cost(32, **dict(kwargs, timing=(dict(timing, wide_grid_finite=False),)))

    def test_all_frozen_state_and_zero_site_controls(self) -> None:
        host = Host(4)
        freeze(host)
        baseline = state_hash(host)
        for module in host.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()
                with self.assertRaises(PilotHold):
                    assert_frozen(host, baseline)
                module.eval()
        parameter = next(host.parameters())
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaises(PilotHold):
            assert_frozen(host, baseline)
        parameter.grad = None
        parameter.requires_grad_(True)
        with self.assertRaises(PilotHold):
            assert_frozen(host, baseline)
        parameter.requires_grad_(False)
        buffer = next(host.buffers())
        saved = buffer.clone()
        buffer.add_(1)
        with self.assertRaises(PilotHold):
            assert_frozen(host, baseline)
        buffer.copy_(saved)
        saved = parameter.clone()
        parameter.add_(1)
        with self.assertRaises(PilotHold):
            assert_frozen(host, baseline)
        parameter.copy_(saved)
        assert_frozen(host, baseline)
        z = torch.rand(4, 16, 8, 8)
        ztilde = z * 0.3
        features = Features(z, ztilde, torch.arange(4))
        cache = Cached(features, features, features)
        for site in (_site(cache, False).development, _site(cache, True).development, Features(z, z, features.labels)):
            identity_check(host, site, 3000)
            self.assertTrue(torch.equal(zero_adapter(z.device).apply(site.read, site.base), site.base))
            self.assertTrue(torch.equal(losses(host, site, None), losses(host, site, zero_adapter(z.device))))
        self.assertTrue(torch.equal(_site(cache, False).conditioning.read, ztilde))
        self.assertTrue(torch.equal(_site(cache, True).conditioning.read, z))
        for family, magnitude in (("P1", 0.03), ("P2", 0.03), ("P3", 1.0), ("P3prime", 1.0)):
            operator = make_impairment(family, magnitude, 3000, z)
            impaired = operator.apply(z)
            restored = Features(z, operator.undo(impaired), features.labels)
            intact = Features(z, z, features.labels)
            self.assertLess(float((losses(host, restored, None) - losses(host, intact, None)).abs().max()), 1e-4)
            wrong = Features(z, torch.full_like(z, float("nan")), features.labels)
            with self.assertRaises(PilotHold):
                losses(host, wrong, None)

    def test_augmentation_matches_direct_crop_flip(self) -> None:
        images = torch.arange(3 * 3 * 32 * 32).reshape(3, 3, 32, 32).remainder(256).byte()
        y, x, flip = torch.tensor([0, 4, 8]), torch.tensor([8, 4, 0]), torch.tensor([True, False, True])
        padded = torch.nn.functional.pad(images, (4, 4, 4, 4), mode="reflect")
        expected = torch.stack(
            [
                padded[i, :, y[i] : y[i] + 32, x[i] : x[i] + 32].flip(2)
                if flip[i]
                else padded[i, :, y[i] : y[i] + 32, x[i] : x[i] + 32]
                for i in range(3)
            ]
        )
        actual = augment(images, y, x, flip)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.dtype, torch.uint8)

    def test_real_audit_hook_seals_test_batch(self) -> None:
        script = "from sultai_screen.data import install_seal; install_seal(); open('/tmp/test_batch','rb')"
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script], capture_output=True, text=True, env=env, timeout=20
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("sealed CIFAR test batch requested", completed.stderr)
        self.assertNotIn("FileNotFoundError", completed.stderr)

    def test_magnitude_ladder_exhaustion_never_uses_failed_rung(self) -> None:
        good = {
            "rungs": {
                "P1": {"0.1": 0.04, "0.3": 0.06, "0.03": 0.07},
                "P2": {"0.1": 0.06, "0.03": 0.09},
                "P3": {"0.25": 0.06, "0.5": 0.1, "1.0": 0.2},
            }
        }
        self.assertEqual(
            magnitude_decision(tuple(good for _ in range(6))),
            {"alpha_low": 0.1, "alpha_high": 0.3, "s_min": 0.1, "kappa": 0.25},
        )
        bad = {"rungs": dict(good["rungs"], P1={"0.1": 0.0, "0.3": 0.0, "0.03": 0.0})}
        with self.assertRaises(InstrumentHold):
            magnitude_decision((good, good, good, good, good, bad))

    def test_supervisor_real_concurrency_cleanup_and_budget(self) -> None:
        real_popen = subprocess.Popen

        def start_child(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            name = output.name
            gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            code = "import pathlib,json,time,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'result.json').write_text(json.dumps({'software_valid':True,'identity':{'physical_gpu':sys.argv[2]},'payload':{}})); time.sleep(float(sys.argv[3])); sys.exit(int(sys.argv[4]))"
            return real_popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    code,
                    str(output),
                    gpu,
                    "30" if name == "timeout" else "0.08",
                    "7" if name == "failure" else "0",
                ],
                **kwargs,
            )

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("sultai_screen.pilot.subprocess.Popen", side_effect=start_child),
        ):
            root = Path(temporary)
            supervisor = Supervisor(root, root, 2.370860553928651)
            supervisor.run(tuple(Job(f"parallel-{i}", "train", 3000, 0, i % 2) for i in range(6)))
            ledger = read_record(root / "ledger-000.json")
            self.assertEqual(len(ledger["entries"]), 6)
            self.assertGreaterEqual(supervisor.charged, 2.370860553928651 + 6 * 0.08)
            self.assertEqual(ledger["opening_charge_gpu_process_seconds"], 2.370860553928651)
            self.assertAlmostEqual(
                supervisor.charged, 2.370860553928651 + sum(e["charged_gpu_process_seconds"] for e in ledger["entries"])
            )
            self.assertTrue(all(entry["returncode"] == 0 for entry in ledger["entries"]))
            with self.assertRaises(PilotHold):
                supervisor.run(tuple(Job(f"excess-{i}", "train", 3000, 0, 0) for i in range(4)))
            started = time.monotonic()
            with self.assertRaisesRegex(PilotHold, "child failure exited 7"):
                supervisor.run((Job("timeout", "train", 3000, 0, 0), Job("failure", "train", 3000, 0, 1)))
            self.assertLess(time.monotonic() - started, 2.0)
            failed = read_record(root / "ledger-001.json")
            self.assertEqual(failed["entries"][-1]["returncode"], 7)
            self.assertIsNotNone(failed["entries"][-2]["returncode"])
            self.assertTrue(all(entry["charged_gpu_process_seconds"] > 0 for entry in failed["entries"]))
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("sultai_screen.pilot.subprocess.Popen", side_effect=start_child),
            patch("sultai_screen.pilot.PILOT_SECONDS", 10.15),
        ):
            root = Path(temporary)
            supervisor = Supervisor(root, root)
            with self.assertRaisesRegex(PilotHold, "timeout"):
                supervisor.run((Job("timeout", "train", 3000, 0, 0),))
            self.assertLess(supervisor.charged, 10.15)
            self.assertNotEqual(read_record(root / "ledger-000.json")["entries"][0]["returncode"], 0)
            supervisor.charged = 10.15
            with self.assertRaisesRegex(PilotHold, "exhausted"):
                supervisor.run((Job("never-started", "train", 3000, 0, 0),))
            self.assertFalse((root / "never-started.process.json").exists())

    def test_environment_distribution_runtime_and_build_controls(self) -> None:
        expected = {
            "torch_wheel": "2.9.1",
            "torch": "2.9.1+cu128",
            "cuda": "12.8",
            "numpy_wheel": "2.3.5",
            "numpy": "2.3.5",
        }
        for bad in (None, *expected):
            values = dict(expected)
            if bad is not None:
                values[bad] = "wrong"
            with (
                patch(
                    "sultai_screen.screen.importlib.metadata.version",
                    side_effect=lambda name, values=values: values[name + "_wheel"],
                ),
                patch("sultai_screen.screen.torch.__version__", values["torch"]),
                patch("sultai_screen.screen.torch.version.cuda", values["cuda"]),
                patch("sultai_screen.screen.np.__version__", values["numpy"]),
                patch("sultai_screen.screen.torch.cuda.init") as cuda_init,
            ):
                if bad is None:
                    self.assertEqual(environment_identity(), expected)
                else:
                    with self.assertRaisesRegex(PilotHold, bad):
                        environment_identity()
                cuda_init.assert_not_called()

    def test_environment_precedes_custody_and_child_launch(self) -> None:
        from sultai_screen.pilot import pilot

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            certificate = base / "certificate.json"
            publish(certificate, {})
            with (
                patch("sultai_screen.pilot.check_certificate", return_value={"commit": "fixture"}),
                patch("sultai_screen.pilot.environment_identity", side_effect=PilotHold("environment mismatch")),
                patch("sultai_screen.pilot.claim_attempt") as claim,
                patch("sultai_screen.pilot.check_handover") as handover,
                patch("sultai_screen.pilot.Supervisor") as supervisor,
            ):
                for retry in (False, True):
                    with self.assertRaisesRegex(PilotHold, "environment mismatch"):
                        pilot(base / "new-output", base, certificate, retry=retry)
                claim.assert_not_called()
                handover.assert_not_called()
                supervisor.assert_not_called()
                self.assertFalse((base / "new-output").exists())

    def test_linked_retry_binds_originals_and_cannot_reset_charge(self) -> None:
        import sultai_screen.pilot as module

        repo = Path(__file__).resolve().parents[1]
        approved = {
            "linked_retry_approved": True,
            "prior_charge_gpu_process_seconds": module.PRIOR_CHARGE,
            "evidence": [{"path": module.DISPOSITION}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for relative, _ in module.PRIOR_RECEIPTS:
                source = (
                    repo
                    / "docs/results/gate3-pilot-2026-10-11-01a125af"
                    / ("attempt-custody.json" if relative.startswith(".weft/") else Path(relative).name)
                )
                target = base / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            originals = {relative: (base / relative).read_bytes() for relative, _ in module.PRIOR_RECEIPTS}
            with patch("sultai_screen.pilot.REPO", base):
                linkage = module.linked_retry(approved)
                self.assertEqual(linkage["opening_charge_gpu_process_seconds"], 2.370860553928651)
                receipt = base / module.RETRY_RECEIPT
                module.claim_attempt(receipt, base / "retry-one", linkage)
                with self.assertRaises(FileExistsError):
                    module.claim_attempt(receipt, base / "retry-two", linkage)
                for key, bad in (
                    ("linked_retry_approved", False),
                    ("prior_charge_gpu_process_seconds", 0.0),
                    ("evidence", []),
                ):
                    with self.assertRaises(PilotHold):
                        module.linked_retry(dict(approved, **{key: bad}))
                for relative, _ in module.PRIOR_RECEIPTS:
                    path = base / relative
                    self.assertEqual(path.read_bytes(), originals[relative])
                    path.write_bytes(originals[relative] + b" ")
                    with self.assertRaisesRegex(PilotHold, "missing or changed"):
                        module.linked_retry(approved)
                    path.unlink()
                    with self.assertRaisesRegex(PilotHold, "missing or changed"):
                        module.linked_retry(approved)
                    path.write_bytes(originals[relative])
                with patch("sultai_screen.pilot.PILOT_SECONDS", module.PRIOR_CHARGE + 10):
                    supervisor = Supervisor(base, base, module.PRIOR_CHARGE)
                    with patch("sultai_screen.pilot.subprocess.Popen") as child:
                        with self.assertRaisesRegex(PilotHold, "exhausted"):
                            supervisor.run((Job("blocked", "train", 3000, 0, 0),))
                        child.assert_not_called()
                        self.assertEqual(supervisor.charged, module.PRIOR_CHARGE)
                for invalid in (-1.0, float("nan"), float("inf"), 10800.0):
                    with self.assertRaises(PilotHold):
                        Supervisor(base, base, invalid)

    def test_attempt_custody_cannot_be_reset_by_new_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "attempt.json"
            claim_attempt(path, root / "one", {"source": "a"})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                claim_attempt(path, root / "two", {"source": "a"})
            self.assertEqual(path.read_bytes(), original)

    def test_source_protocol_and_certificate_identity_negative_controls(self) -> None:
        actual_protocol = Path(__file__).resolve().parents[1] / "docs/plans/2026-10-10-gate3-headroom-screen.md"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.md"
            protocol.write_bytes(actual_protocol.read_bytes())
            for package in ("sultai", "sultai_screen"):
                (root / "src" / package).mkdir(parents=True)
                (root / "src" / package / "__init__.py").write_text('"""fixture"""\n')
            for name in ("quality", "runtime", "review"):
                (root / f"{name}.json").write_text("{}\n")

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()

            git("init", "-q")
            git("add", ".")
            git(
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=Screen Test",
                "-c",
                "user.email=screen@example.invalid",
                "commit",
                "-qm",
                "fixture",
            )
            with (
                patch("sultai_screen.screen.REPO", root),
                patch("sultai_screen.screen.PROTOCOL", protocol),
                patch("sultai_screen.pilot.REPO", root),
            ):
                identity = source_identity()
                certificate = {
                    "preregistration_sha256": PREREG_SHA,
                    "quality_passed": True,
                    "runtime_tests_passed": True,
                    "independent_panel_accepted": True,
                    "implementation_source_sha256": identity["source_sha256"],
                    "implementation_commit": identity["commit"],
                    "evidence": [
                        {"path": f"{name}.json", "sha256": file_sha(root / f"{name}.json")}
                        for name in ("quality", "runtime", "review")
                    ],
                }
                self.assertEqual(check_certificate(certificate), identity)
                for key, value in (
                    ("quality_passed", False),
                    ("runtime_tests_passed", False),
                    ("independent_panel_accepted", False),
                    ("implementation_source_sha256", "wrong"),
                    ("preregistration_sha256", "wrong"),
                ):
                    with self.assertRaises(PilotHold):
                        check_certificate(dict(certificate, **{key: value}))
                with self.assertRaises(KeyError):
                    check_certificate({})
                (root / "untracked.py").write_text("value=1\n")
                with self.assertRaises(PilotHold):
                    source_identity()
                (root / "untracked.py").unlink()
                protocol.write_text("altered")
                with self.assertRaises(PilotHold):
                    source_identity()
                protocol.write_bytes(actual_protocol.read_bytes())
                (root / "notes.md").write_text("doc only\n")
                git("add", "notes.md")
                git(
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "user.name=Screen Test",
                    "-c",
                    "user.email=screen@example.invalid",
                    "commit",
                    "-qm",
                    "docs",
                )
                self.assertEqual(check_certificate(certificate)["source_sha256"], identity["source_sha256"])
                (root / "review.json").write_text('{"changed":true}\n')
                git("add", "review.json")
                git(
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "user.name=Screen Test",
                    "-c",
                    "user.email=screen@example.invalid",
                    "commit",
                    "-qm",
                    "tamper",
                )
                with self.assertRaises(PilotHold):
                    check_certificate(certificate)
                (root / "src/sultai_screen/__init__.py").write_text("changed=1\n")
                git("add", "src")
                git(
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "user.name=Screen Test",
                    "-c",
                    "user.email=screen@example.invalid",
                    "commit",
                    "-qm",
                    "code",
                )
                with self.assertRaises(PilotHold):
                    check_certificate(certificate)

    def test_wrong_finite_inverse_fails_endpoint_control(self) -> None:
        host = Host()
        freeze(host)
        z = torch.rand(4, 16, 8, 8)
        data = Features(z, z, torch.arange(4))
        cache = Cached(data, data, data)
        materiality_worker(host, cache, 3000)
        operator = make_impairment("P1", 0.1, 3000, z)
        with patch.object(type(operator), "undo", lambda self, values: values * 100 + 10):
            with self.assertRaisesRegex(PilotHold, "exact inverse"):
                materiality_worker(host, cache, 3000)

    def test_derivation_label_tamper_fails_pinned_split(self) -> None:
        artifact = Path(__file__).resolve().parents[1] / "docs/results/gate3-split-2026-10-11/dev-indices-int32-le.bin"
        with patch(
            "sultai_screen.data.generator",
            side_effect=lambda seed, label: torch.Generator().manual_seed(derive(seed, label + "-tampered")),
        ):
            with self.assertRaisesRegex(PilotHold, "permutation mismatch"):
                partitions(artifact)

    def test_ordered_pilot_holds_before_median_and_on_nonfinite_timing(self) -> None:
        from sultai_screen.contracts import CHEAP_RATES
        from sultai_screen.pilot import pilot

        class FakeSupervisor:
            bad_timing = False

            def __init__(self, root, data):
                self.root = root
                self.charged = 0.0

            def run(self, jobs):
                for job in jobs:
                    output = self.root / job.name
                    output.mkdir()
                    value = {}
                    if job.action == "train":
                        value = {
                            "epoch_state_sha256": ["a"],
                            "loss_stream_sha256": "b",
                            "final_state_sha256": "c",
                            "forward_backward": {"gradient": "d"},
                        }
                    elif job.action == "timing":
                        value = {
                            "fit_finite": not self.bad_timing,
                            "fit_state_sha256": ["a", "b"],
                            "fit_trace_sha256": "c",
                            "wide_grid_finite": True,
                            "full_batch_step_seconds": 0.01,
                            "wide_grid_seconds": 0.02,
                            "retrieval_pair_seconds": 0.0001,
                            "development_evaluation_seconds": 0.001,
                        }
                    elif job.action == "materiality":
                        value = {
                            "rungs": {
                                "P1": {"0.1": 0.1, "0.3": 0.1, "0.03": 0.1},
                                "P2": {"0.1": 0.1, "0.03": 0.1},
                                "P3": {"0.25": 0.1, "0.5": 0.1, "1.0": 0.1},
                            }
                        }
                    elif job.action == "posthoc":
                        case = {
                            "noop_loss": 1.0,
                            "arm_losses": {"A1": 0.9, "A3": 0.9},
                            "oracle_gain_fraction_of_A3": 1.0,
                        }
                        value = {
                            "former": {"hash": "fixed"},
                            "families": {family: [case] for family in ("P1", "P2", "P3", "P3prime")},
                        }
                    elif job.action == "bottleneck":
                        value = {"paired_gain_difference": 0.1, "P5": {"oracle_loss": 1.0}, "P6": {"oracle_loss": 1.0}}
                    elif job.action == "cheap":
                        candidate = {"candidate": {"selection_loss": 1.0}, "development_loss": 1.0 + job.seed - 3000}
                        value = {family: {str(rate): candidate for rate in CHEAP_RATES} for family in ("P5", "P6")}
                        value["seed"] = job.seed
                    else:
                        raise AssertionError(job.action)
                    publish(output / "result.json", {"software_valid": True, "payload": value, "wall_seconds": 0.01})

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            certificate = base / "certificate.json"
            publish(certificate, {"fixture": True})
            with (
                patch("sultai_screen.pilot.Supervisor", FakeSupervisor),
                patch("sultai_screen.pilot.check_certificate", return_value={"commit": "fixture"}),
                patch("sultai_screen.pilot.check_handover", return_value={}),
                patch("sultai_screen.pilot.claim_attempt"),
            ):
                root = base / "large-sd"
                self.assertEqual(pilot(root, base, certificate), 2)
                for name in (
                    "01-timing-replay-decision",
                    "02-magnitudes",
                    "03-posthoc-known-answer",
                    "04-width",
                    "05-cheap-rate",
                    "06-spread-sizing",
                ):
                    self.assertTrue((root / f"{name}.json").is_file())
                self.assertFalse((root / "07-development-medians.json").exists())
                report = read_record(root / "pilot-result.json")
                self.assertTrue(report["software_valid"])
                self.assertEqual(report["verdict"], "hold-feasibility")
                self.assertGreater(read_record(root / "06-spread-sizing.json")["n_p6"], 200)
                FakeSupervisor.bad_timing = True
                root = base / "bad-timing"
                self.assertEqual(pilot(root, base, certificate), 2)
                self.assertFalse((root / "01-timing-replay-decision.json").exists())
                self.assertFalse((root / "materiality-3000").exists())
                self.assertEqual(read_record(root / "pilot-result.json")["verdict"], "instrument-failed")

    def test_oracle_retains_finite_schedule_and_all_failed_evidence(self) -> None:
        from sultai_screen.fit import Candidate, oracle
        from sultai_screen.screen import worker

        host = Host()
        freeze(host)
        z = torch.ones(4, 16, 8, 8)
        data = Features(z, z, torch.arange(4))
        cache = Cached(data, data, data)
        for available in (200, 800, None):

            def candidate(host, conditioning, selection, *, seed, rate, steps, minibatch, available=available):
                return Candidate(
                    zero_adapter(z.device),
                    rate,
                    seed,
                    steps,
                    1.0 if steps == available else None,
                    tuple(1.0 for _ in range(steps)) if steps == available else (),
                    0.125,
                    256 * steps,
                    272 * steps,
                )

            with patch("sultai_screen.fit.fit_sgd", side_effect=candidate):
                if available is not None:
                    result = oracle(host, data, data, 3000)
                    self.assertEqual(result.best.steps, available)
                    self.assertFalse(result.plateau)
                    self.assertIsNone(result.selection_se)
                    self.assertEqual(len(result.candidates), 24)
                    self.assertIn("unavailable", result.plateau_evidence)
                else:
                    with self.assertRaises(OracleUnavailable) as caught:
                        oracle(host, data, data, 3000)
                    error = caught.exception
                    self.assertEqual(len(error.candidates), 24)
                    self.assertEqual(sum(c["wall_seconds"] for c in error.candidates), 3.0)
                    with (
                        tempfile.TemporaryDirectory() as temporary,
                        patch("sultai_screen.screen.install_seal", return_value=Seal()),
                        patch("sultai_screen.screen.configure", return_value=torch.device("cpu")),
                        patch("sultai_screen.screen.provenance", return_value={}),
                        patch("sultai_screen.screen.torch.cuda.reset_peak_memory_stats"),
                        patch("sultai_screen.screen.load_dataset", return_value=None),
                        patch("sultai_screen.screen._training", return_value=host),
                        patch("sultai_screen.screen.cached", return_value=cache),
                        patch("sultai_screen.screen.bottleneck_worker", side_effect=error),
                    ):
                        output = Path(temporary) / "worker"
                        with self.assertRaises(OracleUnavailable):
                            worker("bottleneck", output, Path(temporary), 3000, 4, Path(temporary), None)
                        failure = read_record(output / "failure.json")
                        self.assertTrue(failure["software_valid"])
                        self.assertFalse(failure["instrument_valid"])
                        self.assertEqual(len(failure["unavailable_candidates"]), 24)
                        self.assertEqual(sum(c["wall_seconds"] for c in failure["unavailable_candidates"]), 3.0)

    def test_raw_cheap_nonfinite_dev_never_changes_selection(self) -> None:
        from sultai_screen.fit import Candidate
        from sultai_screen.pilot import selected_development
        from sultai_screen.screen import cheap_worker

        host = Host()
        freeze(host)
        z = torch.ones(4, 16, 8, 8)
        data = Features(z, z, torch.arange(4))
        cache = Cached(data, data, data)
        candidates = tuple(
            Candidate(zero_adapter(z.device), rate, 3000, 4, selection, (), 0.1, 16, 1088)
            for rate, selection in ((3.0, 2.0), (0.1, 1.0))
        )
        with (
            patch("sultai_screen.screen.cheap_bank", return_value=candidates),
            patch(
                "sultai_screen.screen.losses",
                side_effect=[
                    torch.full((4,), float("nan")),
                    torch.ones(4),
                    torch.full((4,), float("nan")),
                    torch.ones(4),
                ],
            ),
        ):
            result = cheap_worker(host, cache, 3000)
        for family in ("P5", "P6"):
            self.assertFalse(result[family]["3.0"]["development_finite"])
            self.assertIsNone(result[family]["3.0"]["development_loss"])
            self.assertEqual(choose(candidates).rate, 0.1)
            self.assertEqual(selected_development(result[family]["0.1"]), 1.0)
            with self.assertRaisesRegex(PilotHold, "selected cheap"):
                selected_development(result[family]["3.0"])
        # Lower selection loss makes the bad-development candidate win; no fallback
        # based on development finiteness is permitted.
        bad = Candidate(candidates[0].adapter, 3.0, 3000, 4, 0.0, (), 0.1, 16, 1088)
        self.assertEqual(choose((bad, candidates[1])).rate, 3.0)
        with self.assertRaises(PilotHold):
            selected_development(result["P6"][str(choose((bad, candidates[1])).rate)])

    def test_raw_output_staging_precedes_attempt_admission(self) -> None:
        from sultai_screen.pilot import check_output_root, pilot

        root = Path(__file__).resolve().parents[1]
        check_output_root(root / "runs/gate3-pilot-fixture")
        check_output_root(Path("/tmp/gate3-pilot-fixture"))
        with patch("sultai_screen.pilot.claim_attempt") as claim:
            with self.assertRaisesRegex(PilotHold, "stage raw pilot output"):
                pilot(root / "docs/results/gate3-pilot-fixture", root, Path("/missing/certificate"))
            claim.assert_not_called()

    def test_global_p6_rate_retains_unavailable_p5_fit_without_fallback(self) -> None:
        from sultai_screen.contracts import CHEAP_RATES
        from sultai_screen.pilot import cheap_rate_decision

        records = []
        for seed in range(3000, 3006):
            record = {"seed": seed}
            for family in ("P5", "P6"):
                bank = {}
                for rate in CHEAP_RATES:
                    failed = family == "P5" and rate == 3.0
                    bank[str(rate)] = {
                        "candidate": {"selection_loss": None if failed else 0.0 if rate == 3.0 else 1.0},
                        "development_evaluated": not failed,
                        "development_loss": None if failed else 1.0,
                    }
                record[family] = bank
            records.append(record)
        decision = cheap_rate_decision(tuple(records))
        self.assertEqual(decision["eta_B"], 3.0)
        self.assertTrue(all(r["P5"]["status"] == "fit-unavailable" for r in decision["fixed_rate_results"]))
        self.assertTrue(all(r["P6"]["status"] == "evaluated" for r in decision["fixed_rate_results"]))
        records[0]["P6"]["3.0"]["development_loss"] = None
        with self.assertRaisesRegex(PilotHold, "nonfinite development evaluation"):
            cheap_rate_decision(tuple(records))


if __name__ == "__main__":
    unittest.main()
