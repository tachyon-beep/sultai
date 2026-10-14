"""CPU controls for the M0 identity constants, the five-document source identity and import isolation."""

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
    AMENDMENT_3_PATH,
    AMENDMENT_3_SHA,
    AMENDMENT_PATH,
    AMENDMENT_SHA,
    CERTIFIED_FRACTION,
    CHECKPOINTS,
    CHECKPOINTS_B,
    FOLD_TOLERANCE,
    HARM,
    M0_CONFIRMATORY_SECONDS,
    M0_CONFIRMATORY_STATIC_SECONDS,
    M0_PILOT_SECONDS,
    M_S,
    MODULE_RATES,
    PREREG_PATH,
    PREREG_SHA,
    PROTOCOL_SHA,
    PROTOCOL_VERSION,
    RALL_CHECKPOINTS,
    RETUNE_RATES,
    RHO,
    S_STATIC_SD_LIMIT,
    STEERING_PATH,
    STEERING_SHA,
    X2,
    X,
    protocol_identity,
    source_identity,
)
from sultai_screen.m0.provenance import envelope
from sultai_screen.m0.seeds import BLOCKS, RESERVED_M2, admit, width_scale

REPO = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    (PREREG_PATH, "preregistration"),
    (STEERING_PATH, "steering"),
    (AMENDMENT_PATH, "amendment 1"),
    (AMENDMENT_2_PATH, "amendment 2"),
    (AMENDMENT_3_PATH, "amendment 3"),
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
    def test_five_documents_and_composed_identity_are_pinned(self) -> None:
        self.assertEqual(file_sha(REPO / PREREG_PATH), PREREG_SHA)
        self.assertEqual(file_sha(REPO / STEERING_PATH), STEERING_SHA)
        self.assertEqual(file_sha(REPO / AMENDMENT_PATH), AMENDMENT_SHA)
        self.assertEqual(file_sha(REPO / AMENDMENT_2_PATH), AMENDMENT_2_SHA)
        self.assertEqual(file_sha(REPO / AMENDMENT_3_PATH), AMENDMENT_3_SHA)
        # Amendment 3 (2026-10-14) is composed fifth under the Gate 3 v4.1 rule; the v4 identity 3a76c6db… stays on
        # record in freeze certificate 2 and the gate-B admission certificate for 978850a.
        composed = hashlib.sha256(
            f"m0-v4.1\n{PREREG_SHA}\n{STEERING_SHA}\n{AMENDMENT_SHA}\n{AMENDMENT_2_SHA}\n{AMENDMENT_3_SHA}\n".encode()
        ).hexdigest()
        self.assertEqual(PROTOCOL_VERSION, "m0-v4.1")
        self.assertEqual(AMENDMENT_3_SHA, "6dcfd9f2a781dc56d39c6e4fe548fc4728c2c743a8e5a978d0e4b9741459e862")
        self.assertEqual(PROTOCOL_SHA, composed)
        self.assertEqual(PROTOCOL_SHA, "dfd13cb45100d5da6bca0ac3348ce9d617d614132e9ceba07c74d57c45f33ca7")
        self.assertEqual(
            set(protocol_identity()),
            {
                "protocol_version",
                "protocol_sha256",
                "preregistration_sha256",
                "steering_sha256",
                "amendment_sha256",
                "amendment_2_sha256",
                "amendment_3_sha256",
            },
        )

    def test_margins_and_ceilings_are_the_frozen_values(self) -> None:
        self.assertEqual((X, X2, M_S, RHO, HARM, FOLD_TOLERANCE), (0.02, 0.01, 0.02, 0.8, 0.01, 0.005))
        self.assertEqual((M0_PILOT_SECONDS, M0_CONFIRMATORY_SECONDS), (14_400, 43_200))
        # §10: S-static joins the confirmatory only under John's 14 GPU-hour ceiling (B7: beside the other ceilings).
        self.assertEqual(M0_CONFIRMATORY_STATIC_SECONDS, 50_400)
        self.assertEqual(PILOT_SECONDS, 3 * 3600)  # Gate 3 constant untouched
        # C1: the certification rule, the S-static admission limit, both rate grids and every checkpoint axis (§4, §7).
        self.assertEqual((CERTIFIED_FRACTION, S_STATIC_SD_LIMIT), (0.75, 0.0275))
        self.assertEqual(MODULE_RATES, (1.0, 0.3, 0.1))
        self.assertEqual(RETUNE_RATES, (1.0, 0.3, 0.1, 0.03, 0.01))
        self.assertEqual(CHECKPOINTS, (200, 800, 3200))
        self.assertEqual(CHECKPOINTS_B, (400, 1600, 6400))
        self.assertEqual(RALL_CHECKPOINTS, tuple(range(40, 801, 40)))
        self.assertEqual((len(RALL_CHECKPOINTS), RALL_CHECKPOINTS[0], RALL_CHECKPOINTS[-1]), (20, 40, 800))

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


class SeedTests(unittest.TestCase):
    def test_blocks_match_appendix_a(self) -> None:
        self.assertEqual(
            BLOCKS,
            {
                "anchor": range(3000, 3006),
                "frontier-0.5": range(6000, 6006),
                "frontier-0.75": range(6006, 6012),
                "frontier-1.5": range(6012, 6018),
                "cooled-20": range(6018, 6024),
                "frontier-0.25": range(6024, 6030),
                "confirmatory": range(7500, 7600),
                "restart": range(7700, 7800),
            },
        )
        self.assertEqual(RESERVED_M2, range(6100, 6200))

    def test_blocks_are_disjoint_from_each_other_and_from_reserved_and_gate3(self) -> None:
        others = [RESERVED_M2, range(4000, 4364), range(5000, 5800)]
        seen: set[int] = set()
        for block in BLOCKS.values():
            members = set(block)
            self.assertFalse(members & seen)
            seen |= members
            for other in others:
                self.assertFalse(members & set(other))

    def test_admit(self) -> None:
        self.assertEqual(admit(7500, "confirmatory"), 7500)
        self.assertEqual(admit(3005, "anchor"), 3005)
        with self.assertRaises(PilotHold):
            admit(3000, "frontier-0.5")
        with self.assertRaises(PilotHold):
            admit(6100, "anchor")
        with self.assertRaises(PilotHold):
            admit(3006, "anchor")

    def test_width_scale(self) -> None:
        self.assertEqual(width_scale("anchor"), 1.0)
        self.assertEqual(width_scale("cooled-20"), 1.0)
        self.assertEqual(width_scale("frontier-0.25"), 0.25)
        self.assertEqual(width_scale("frontier-0.5"), 0.5)
        self.assertEqual(width_scale("frontier-0.75"), 0.75)
        self.assertEqual(width_scale("frontier-1.5"), 1.5)
        self.assertEqual(width_scale("confirmatory"), 1.0)
        self.assertEqual(width_scale("restart"), 1.0)
        for lineage in BLOCKS:
            self.assertIsInstance(width_scale(lineage), float)


if __name__ == "__main__":
    unittest.main()
