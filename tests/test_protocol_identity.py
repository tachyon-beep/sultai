"""CPU controls for the composed v4.1 protocol identity and the linked pilot 2 custody binding."""

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sultai_screen import pilot as pilot_module
from sultai_screen.contracts import (
    AMENDMENT_PATH,
    AMENDMENT_SHA,
    PREREG_SHA,
    PROTOCOL_SHA,
    PROTOCOL_VERSION,
    PilotHold,
    file_sha,
    protocol_identity,
)
from sultai_screen.pilot import Supervisor, check_certificate, stop_report
from sultai_screen.screen import _training, source_identity

REPO = Path(__file__).resolve().parents[1]


def git(root: Path, *args: str) -> str:
    command = ["git", "-c", "commit.gpgsign=false", "-c", "user.name=Fixture", "-c", "user.email=f@example.invalid"]
    return subprocess.check_output([*command, *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()


def protocol_fixture(root: Path) -> tuple[Path, Path]:
    protocol = root / "protocol.md"
    amendment = root / "amendment.md"
    protocol.write_bytes((REPO / "docs/plans/2026-10-10-gate3-headroom-screen.md").read_bytes())
    amendment.write_bytes((REPO / AMENDMENT_PATH).read_bytes())
    for package in ("sultai", "sultai_screen"):
        (root / "src" / package).mkdir(parents=True)
        (root / "src" / package / "__init__.py").write_text('"""fixture"""\n')
    for name in ("quality", "runtime", "review"):
        (root / f"{name}.json").write_text("{}\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return protocol, amendment


class ProtocolIdentityTests(unittest.TestCase):
    def test_amendment_bytes_and_composed_identity_are_pinned(self) -> None:
        self.assertEqual(file_sha(REPO / "docs/plans/2026-10-10-gate3-headroom-screen.md"), PREREG_SHA)
        self.assertEqual(file_sha(REPO / AMENDMENT_PATH), AMENDMENT_SHA)
        composed = hashlib.sha256(f"{PROTOCOL_VERSION}\n{PREREG_SHA}\n{AMENDMENT_SHA}\n".encode()).hexdigest()
        self.assertEqual(PROTOCOL_SHA, composed)
        self.assertEqual(PROTOCOL_VERSION, "gate3-v4.1")
        self.assertEqual(
            protocol_identity(),
            {
                "protocol_version": PROTOCOL_VERSION,
                "protocol_sha256": PROTOCOL_SHA,
                "preregistration_sha256": PREREG_SHA,
                "amendment_sha256": AMENDMENT_SHA,
            },
        )
        report = stop_report(PilotHold("fixture"), "stage", 0.0)
        self.assertEqual(report["protocol_sha256"], PROTOCOL_SHA)
        self.assertEqual(report["amendment_sha256"], AMENDMENT_SHA)

    def test_source_identity_requires_both_protocol_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, amendment = protocol_fixture(root)
            with (
                patch("sultai_screen.screen.REPO", root),
                patch("sultai_screen.screen.PROTOCOL", protocol),
                patch("sultai_screen.screen.AMENDMENT", amendment),
            ):
                identity = source_identity()
                for key, value in protocol_identity().items():
                    self.assertEqual(identity[key], value)
                original = amendment.read_bytes()
                amendment.write_bytes(original + b"\n")
                with self.assertRaisesRegex(PilotHold, "amendment"):
                    source_identity()
                amendment.write_bytes(original)
                protocol.write_bytes(protocol.read_bytes() + b"\n")
                with self.assertRaisesRegex(PilotHold, "preregistration"):
                    source_identity()

    def test_old_only_certificates_and_training_receipts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, amendment = protocol_fixture(root)
            with (
                patch("sultai_screen.screen.REPO", root),
                patch("sultai_screen.screen.PROTOCOL", protocol),
                patch("sultai_screen.screen.AMENDMENT", amendment),
                patch("sultai_screen.pilot.REPO", root),
            ):
                identity = source_identity()
                certificate = {
                    **protocol_identity(),
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
                for key in ("protocol_version", "protocol_sha256", "amendment_sha256"):
                    old_only = dict(certificate)
                    del old_only[key]
                    with self.assertRaises(PilotHold):
                        check_certificate(old_only)
                    with self.assertRaises(PilotHold):
                        check_certificate(dict(certificate, **{key: "wrong"}))
                training = root / "training"
                training.mkdir()
                old_receipt = {
                    "payload": {},
                    "identity": {
                        "source_sha256": identity["source_sha256"],
                        "preregistration_sha256": PREREG_SHA,
                    },
                }
                (training / "result.json").write_text(json.dumps(old_receipt) + "\n")
                with patch("sultai_screen.screen.provenance", return_value=dict(identity)):
                    with self.assertRaisesRegex(PilotHold, "protocol"):
                        _training(training, 3000, 0, torch.device("cpu"))
                    (training / "result.json").unlink()
                    mismatched = {
                        "payload": {},
                        "identity": {**dict(identity), "protocol_sha256": "wrong"},
                    }
                    (training / "result.json").write_text(json.dumps(mismatched) + "\n")
                    with self.assertRaisesRegex(PilotHold, "protocol"):
                        _training(training, 3000, 0, torch.device("cpu"))


PUBLISHED = {
    ".weft/gate3-pilot-attempt.json": "docs/results/gate3-pilot-2026-10-11-01a125af/attempt-custody.json",
    ".weft/gate3-pilot-linked-retry-1.json": "docs/results/gate3-pilot-2026-10-11-01a125af-retry1/attempt-custody.json",
    ".weft/gate3-twin-diagnostic-attempt.json": "docs/results/gate3-twin-diagnostic-2026-10-12-01a125af/attempt-custody.json",
    "runs/gate3-pilot-2026-10-11-01a125af-retry1/pilot-result.json": (
        "docs/results/gate3-pilot-2026-10-11-01a125af-retry1/pilot-result.json"
    ),
    "runs/gate3-pilot-2026-10-11-01a125af-retry1/ledger-015.json": (
        "docs/results/gate3-pilot-2026-10-11-01a125af-retry1/ledger-015.json"
    ),
    "runs/gate3-twin-diagnostic-2026-10-12-01a125af/charge.json": (
        "docs/results/gate3-twin-diagnostic-2026-10-12-01a125af/charge.json"
    ),
}


class LinkedPilotTwoTests(unittest.TestCase):
    def test_linked_pilot_two_binds_all_three_custodies_and_the_cumulative_charge(self) -> None:
        module = pilot_module
        self.assertEqual({relative for relative, _ in module.PRIOR_RECEIPTS_2}, set(PUBLISHED))
        self.assertEqual(module.CUMULATIVE_CHARGE, 1473.267264504917)
        approved = {
            "linked_pilot_2_approved": True,
            "prior_charge_gpu_process_seconds": module.CUMULATIVE_CHARGE,
            "evidence": [{"path": module.DISPOSITION_2}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for relative, expected in module.PRIOR_RECEIPTS_2:
                target = base / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((REPO / PUBLISHED[relative]).read_bytes())
                self.assertEqual(file_sha(target), expected)
            originals = {relative: (base / relative).read_bytes() for relative, _ in module.PRIOR_RECEIPTS_2}
            with patch("sultai_screen.pilot.REPO", base):
                linkage = module.linked_pilot_two(approved)
                self.assertEqual(linkage["opening_charge_gpu_process_seconds"], 1473.267264504917)
                self.assertEqual(linkage["attempt_number"], 3)
                self.assertIs(linkage["further_attempt_authorized"], False)
                self.assertEqual(len(linkage["prior_receipts"]), 6)
                receipt = base / module.PILOT_2_RECEIPT
                module.claim_attempt(receipt, base / "attempt-three", linkage)
                with self.assertRaises(FileExistsError):
                    module.claim_attempt(receipt, base / "attempt-four", linkage)
                for key, bad in (
                    ("linked_pilot_2_approved", False),
                    ("prior_charge_gpu_process_seconds", module.PRIOR_CHARGE),
                    ("evidence", [{"path": module.DISPOSITION}]),
                ):
                    with self.assertRaises(PilotHold):
                        module.linked_pilot_two(dict(approved, **{key: bad}))
                for relative, _ in module.PRIOR_RECEIPTS_2:
                    path = base / relative
                    path.write_bytes(originals[relative] + b" ")
                    with self.assertRaisesRegex(PilotHold, "missing or changed"):
                        module.linked_pilot_two(approved)
                    path.unlink()
                    with self.assertRaisesRegex(PilotHold, "missing or changed"):
                        module.linked_pilot_two(approved)
                    path.write_bytes(originals[relative])
                supervisor = Supervisor(base, base, module.CUMULATIVE_CHARGE)
                self.assertEqual(supervisor.charged, module.CUMULATIVE_CHARGE)

    def test_pilot_entry_refuses_second_linked_attempt_without_its_disposition(self) -> None:
        from sultai_screen.pilot import pilot

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            certificate = base / "certificate.json"
            certificate.write_text('{"linked_pilot_2_approved": false, "evidence": []}\n')
            with (
                patch("sultai_screen.pilot.check_certificate", return_value={"commit": "fixture"}),
                patch("sultai_screen.pilot.environment_identity", return_value={}),
                patch("sultai_screen.pilot.claim_attempt") as claim,
                patch("sultai_screen.pilot.check_handover") as handover,
                patch("sultai_screen.pilot.Supervisor") as supervisor,
            ):
                with self.assertRaisesRegex(PilotHold, "disposition"):
                    pilot(base / "new-output", base, certificate, second=True)
                claim.assert_not_called()
                handover.assert_not_called()
                supervisor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
