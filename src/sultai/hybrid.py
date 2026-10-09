"""Bounded hybrid TCD runner; no GPU, network, external data or dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from .formation import run_formation
from .lifecycle import run_diagnostics, run_lifecycle
from .report_types import (
    SOURCE_FILES,
    HybridReport,
    HybridSummary,
    SourceIdentity,
    combined_gates,
    parse_Diagnostics,
    parse_FormationReport,
    parse_HybridReport,
    parse_LifecycleReport,
)

PROTOCOL = "sultai-hybrid-cpu-v2"


def source_identity() -> SourceIdentity:
    """Bind results to executable bytes, independent of checkout location."""
    package = Path(__file__).resolve().parent
    actual_paths = {f"src/sultai/{path.name}" for path in package.glob("*.py")}
    if actual_paths != set(SOURCE_FILES):
        raise ValueError("runtime module inventory differs from the versioned source identity contract")
    digests = {name: hashlib.sha256((package / Path(name).name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    canonical = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    return {"files": digests, "aggregate_sha256": hashlib.sha256(canonical).hexdigest()}


def run() -> HybridReport:
    raw_formation, paired_generator = run_formation()
    formation = parse_FormationReport(raw_formation)
    lifecycle = parse_LifecycleReport(run_lifecycle(paired_generator.form, formation_label="frozen_paired_generator"))
    diagnostics = parse_Diagnostics(run_diagnostics())
    gates = combined_gates(formation["acceptance"], lifecycle["acceptance"], diagnostics)
    result: HybridReport = {
        "protocol": PROTOCOL,
        "source_identity": source_identity(),
        "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation()},
        "acceptance": gates,
        "passed": all(gates.values()),
        "formation": formation,
        "lifecycle": lifecycle,
        "diagnostics": diagnostics,
        "claims": {
            "scope": "procedural seen-family formation and separate planted same-feature handover mechanics",
            "diffusion": False,
            "unseen_family_transfer": False,
            "population_safety_guarantee": False,
            "learned_removal_policy": False,
            "half_parameter_goal_demonstrated": False,
            "near_empty_host": False,
        },
    }
    # Do not silently serialize a non-finite result as valid evidence.
    json.dumps(result, allow_nan=False)
    return parse_HybridReport(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="Print compact JSON with gates and assay summaries.")
    args = parser.parse_args(argv)
    report = parse_HybridReport(run())
    output: HybridReport | HybridSummary = report
    if args.summary:
        output = {
            "protocol": report["protocol"],
            "source_identity": report["source_identity"],
            "runtime": report["runtime"],
            "passed": report["passed"],
            "acceptance": report["acceptance"],
            "claims": report["claims"],
            "formation_summary": report["formation"]["summary"],
            "lifecycle_summary": report["lifecycle"]["summary"],
        }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
