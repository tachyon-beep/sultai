"""Versioned correction evidence; a scientific negative is a valid outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .formation_controls import FormationControlReport, run_formation_controls
from .hybrid import source_identity
from .report_types import SourceIdentity
from .trajectory import TrajectoryReport, run_trajectory
from .trust import ContractViolation

PROTOCOL = "sultai-hybrid-correction-v1"
PLAN = "docs/plans/2026-10-09-correction-protocol.md"
Phase = Literal["development", "confirmation"]


@dataclass(frozen=True)
class RunIdentity:
    source_commit: str
    source_tree: str
    plan_path: str
    plan_sha256: str
    canonical_command: tuple[str, ...]
    python: str
    implementation: str
    machine: str


@dataclass(frozen=True)
class GateVerdict:
    gate: int
    name: str
    verdict: str
    reason: str


@dataclass(frozen=True)
class CorrectionReport:
    protocol: str
    phase: Phase
    identity: RunIdentity
    source_identity: SourceIdentity
    formation: FormationControlReport
    trajectory: TrajectoryReport
    software_valid: bool
    diagnostic_specificity: bool
    training_ready: bool
    gates: tuple[GateVerdict, ...]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def run(phase: Phase) -> CorrectionReport:
    """T1: named fixed phase on committed source; no result-driven retries."""
    if phase not in ("development", "confirmation"):
        raise ContractViolation("unknown correction phase")
    root = Path(__file__).resolve().parents[2]
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ContractViolation("commit the reviewed inputs before cohort execution")
    source = source_identity()
    identity = RunIdentity(
        _git(root, "rev-parse", "HEAD"),
        _git(root, "rev-parse", "HEAD^{tree}"),
        PLAN,
        hashlib.sha256((root / PLAN).read_bytes()).hexdigest(),
        ("python3", "-m", "sultai.correction", "--phase", phase),
        platform.python_version(),
        platform.python_implementation(),
        platform.machine(),
    )
    formation, generator = run_formation_controls(phase)
    trajectory = run_trajectory(generator.form, phase)
    if source_identity() != source or _git(root, "rev-parse", "HEAD") != identity.source_commit:
        raise ContractViolation("source changed during execution; evidence is invalid")
    if hashlib.sha256((root / PLAN).read_bytes()).hexdigest() != identity.plan_sha256:
        raise ContractViolation("plan changed during execution; evidence is invalid")
    valid = formation.software_valid and trajectory.software_valid
    gates = (
        GateVerdict(
            1, "contract_hardening", "separate_verification", "See exact-source unit, type and policy evidence."
        ),
        GateVerdict(
            2,
            "instrument_truth",
            "bounded_checks_passed" if valid else "fail",
            "Single-run checks only; complete gate needs separate fault-injection, quality and deterministic replay evidence.",
        ),
        GateVerdict(3, "useful_repair_headroom", "not_established", "Shared four-template family; no transfer test."),
        GateVerdict(
            4, "honest_admission", "measured", "All clean/noisy outcomes retained; no population safety claim."
        ),
        GateVerdict(5, "benefit_after_removal", "not_established", "Pre-removal specificity is not retained benefit."),
        GateVerdict(
            6, "calibrated_safety_and_utility", "blocked", "Domain, utility/risk margins and budget are unset."
        ),
        GateVerdict(7, "training_readiness", "not_established", "Predecessor scientific requirements remain unproven."),
    )
    return CorrectionReport(
        PROTOCOL, phase, identity, source, formation, trajectory, valid, trajectory.diagnostic_specificity, False, gates
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("development", "confirmation"))
    parser.add_argument("--output", type=Path, help="New evidence file; an existing file is never overwritten.")
    arguments = parser.parse_args(argv)
    output: Path | None = arguments.output
    if output is not None and output.exists():
        raise FileExistsError(output)
    phase: Phase = "development" if arguments.phase == "development" else "confirmation"
    report = run(phase)
    serialized = json.dumps(asdict(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(serialized, end="")
    else:
        # Publish a complete file atomically without replacing another attempt.
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, output)
        finally:
            os.unlink(temporary)
    return 0 if report.software_valid else 2


if __name__ == "__main__":
    sys.exit(main())
