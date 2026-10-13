"""CPU controls for the M0 identity constants, the four-document source identity and import isolation."""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sultai_screen.contracts import PILOT_SECONDS, PilotHold, file_sha
from sultai_screen.m0.identity import (
    AMENDMENT_2_PATH,
    AMENDMENT_2_SHA,
    AMENDMENT_PATH,
    AMENDMENT_SHA,
    FOLD_TOLERANCE,
    HARM,
    M0_CONFIRMATORY_SECONDS,
    M0_PILOT_SECONDS,
    M_S,
    PREREG_PATH,
    PREREG_SHA,
    PROTOCOL_SHA,
    PROTOCOL_VERSION,
    RHO,
    STEERING_PATH,
    STEERING_SHA,
    X2,
    X,
    protocol_identity,
    source_identity,
)
from sultai_screen.m0.provenance import envelope

REPO = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    (PREREG_PATH, "preregistration"),
    (STEERING_PATH, "steering"),
    (AMENDMENT_PATH, "amendment 1"),
    (AMENDMENT_2_PATH, "amendment 2"),
)


def git(root: Path, *args: str) -> str:
    command = ["git", "-c", "commit.gpgsign=false", "-c", "user.name=Fixture", "-c", "user.email=f@example.invalid"]
    return subprocess.check_output([*command, *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()


def m0_fixture(root: Path) -> None:
    for relative, _ in DOCUMENTS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / relative).read_bytes())
    for package in ("sultai", "sultai_screen"):
        (root / "src" / package).mkdir(parents=True)
        (root / "src" / package / "__init__.py").write_text('"""fixture"""\n')
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")


class IdentityTests(unittest.TestCase):
    def test_four_documents_and_composed_identity_are_pinned(self) -> None:
        self.assertEqual(file_sha(REPO / PREREG_PATH), PREREG_SHA)
        self.assertEqual(file_sha(REPO / STEERING_PATH), STEERING_SHA)
        self.assertEqual(file_sha(REPO / AMENDMENT_PATH), AMENDMENT_SHA)
        self.assertEqual(file_sha(REPO / AMENDMENT_2_PATH), AMENDMENT_2_SHA)
        composed = hashlib.sha256(
            f"m0-v4\n{PREREG_SHA}\n{STEERING_SHA}\n{AMENDMENT_SHA}\n{AMENDMENT_2_SHA}\n".encode()
        ).hexdigest()
        self.assertEqual(PROTOCOL_VERSION, "m0-v4")
        self.assertEqual(PROTOCOL_SHA, composed)
        self.assertEqual(PROTOCOL_SHA, "3a76c6dbe31ca3f80ec301ed0523b1f382148f8114ee02a164ae4ae9557f6e5d")
        self.assertEqual(
            set(protocol_identity()),
            {
                "protocol_version",
                "protocol_sha256",
                "preregistration_sha256",
                "steering_sha256",
                "amendment_sha256",
                "amendment_2_sha256",
            },
        )

    def test_margins_and_ceilings_are_the_frozen_values(self) -> None:
        self.assertEqual((X, X2, M_S, RHO, HARM, FOLD_TOLERANCE), (0.02, 0.01, 0.02, 0.8, 0.01, 0.005))
        self.assertEqual((M0_PILOT_SECONDS, M0_CONFIRMATORY_SECONDS), (14_400, 43_200))
        self.assertEqual(PILOT_SECONDS, 3 * 3600)  # Gate 3 constant untouched

    def test_source_identity_names_the_changed_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m0_fixture(root)
            with patch("sultai_screen.m0.identity.REPO", root):
                identity = source_identity()
                for key, value in protocol_identity().items():
                    self.assertEqual(identity[key], value)
                self.assertEqual(identity["commit"], git(root, "rev-parse", "HEAD"))
                for relative, name in DOCUMENTS:
                    path = root / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n")
                    with self.assertRaisesRegex(PilotHold, name):
                        source_identity()
                    path.write_bytes(original)
                (root / "stray.txt").write_text("dirty\n")
                with self.assertRaisesRegex(PilotHold, "committed, unchanged source"):
                    source_identity()


class EnvelopeTests(unittest.TestCase):
    def test_envelope_reproduces_the_gate3_receipt_keys_and_imports_stay_isolated(self) -> None:
        receipt = envelope("train", {"commit": "c"}, {"seed": 1}, {"opens": 0}, 0.0, 7)
        self.assertEqual(
            set(receipt),
            {"action", "software_valid", "identity", "payload", "data_opens", "wall_seconds", "peak_cuda_bytes"},
        )
        self.assertIs(receipt["software_valid"], True)
        self.assertEqual(receipt["peak_cuda_bytes"], 7)
        probe = (
            "import sys, sultai_screen.m0.identity, sultai_screen.m0.provenance\n"
            "banned = ('sultai_screen.screen', 'sultai_screen.pilot', 'sultai_screen.former_bridge',"
            " 'sultai_screen.alignment')\n"
            "print([name for name in banned if name in sys.modules])\n"
        )
        out = subprocess.check_output(
            [sys.executable, "-B", "-c", probe], env={"PYTHONPATH": str(REPO / "src")}, text=True
        )
        self.assertEqual(out.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
