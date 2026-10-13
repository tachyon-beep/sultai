"""M0 protocol identity: the four hashed documents, frozen margins and source provenance (no torch)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from sultai_screen.contracts import PilotHold, Record, encode, file_sha

REPO = Path(__file__).resolve().parents[3]
PREREG_PATH = "docs/plans/2026-10-13-capacity-frontier-screen.md"
STEERING_PATH = "docs/STEERING.md"
AMENDMENT_PATH = "docs/plans/2026-10-13-gate-ladder-amendment.md"
AMENDMENT_2_PATH = "docs/plans/2026-10-13-gate-ladder-amendment-2.md"
# docs/certificates/2026-10-13-m0-freeze-2.md
PREREG_SHA = "c66599459adc898f96322ab11dfa3c6737cd3559e06c9432d79e377e00c03fd5"
STEERING_SHA = "08a9e1aaf79d3048a5d6f6f4d6138a9c7049f32b61c0250e1505773a27fa3ffd"
AMENDMENT_SHA = "af12d474a97aad83bc63405034fec3716170f7652f34364588750732c97b6a89"
AMENDMENT_2_SHA = "b4740b147a4a1ced52a6f0721215679c1567078260d5c11a92c35fa2065c2d77"
PROTOCOL_VERSION = "m0-v4"
PROTOCOL_SHA = hashlib.sha256(
    f"{PROTOCOL_VERSION}\n{PREREG_SHA}\n{STEERING_SHA}\n{AMENDMENT_SHA}\n{AMENDMENT_2_SHA}\n".encode()
).hexdigest()

X = 0.02
X2 = 0.01
M_S = 0.02
RHO = 0.8
HARM = 0.01
FOLD_TOLERANCE = 0.005
CERTIFIED_FRACTION = 0.75
S_STATIC_SD_LIMIT = 0.0275
M0_PILOT_SECONDS = 14_400
M0_CONFIRMATORY_SECONDS = 43_200
MODULE_RATES = (1.0, 0.3, 0.1)
RETUNE_RATES = (1.0, 0.3, 0.1, 0.03, 0.01)
CHECKPOINTS = (200, 800, 3200)
CHECKPOINTS_B = (400, 1600, 6400)


def protocol_identity() -> Record:
    """The identity fields every M0 receipt, report and certificate must carry."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA,
        "preregistration_sha256": PREREG_SHA,
        "steering_sha256": STEERING_SHA,
        "amendment_sha256": AMENDMENT_SHA,
        "amendment_2_sha256": AMENDMENT_2_SHA,
    }


def source_identity() -> Record:
    for relative, expected, name in (
        (PREREG_PATH, PREREG_SHA, "preregistration"),
        (STEERING_PATH, STEERING_SHA, "steering"),
        (AMENDMENT_PATH, AMENDMENT_SHA, "amendment 1"),
        (AMENDMENT_2_PATH, AMENDMENT_2_SHA, "amendment 2"),
    ):
        if file_sha(REPO / relative) != expected:
            raise PilotHold(f"committed {name} identity mismatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if dirty:
        raise PilotHold("run requires committed, unchanged source")
    files = subprocess.check_output(
        ["git", "ls-files", "src", "scripts", "tests", "pyproject.toml"], cwd=REPO, text=True
    ).splitlines()
    for root in (REPO / "src/sultai", REPO / "src/sultai_screen"):
        production = {str(path.relative_to(REPO)) for path in root.rglob("*.py")}
        if not production or not production.issubset(set(files)):
            raise PilotHold("uncommitted or empty production source coverage")
    hashes: Record = {name: file_sha(REPO / name) for name in files}
    aggregate = hashlib.sha256(encode(hashes).encode()).hexdigest()
    return {"commit": commit, "source_sha256": aggregate, "source_files": hashes, **protocol_identity()}
