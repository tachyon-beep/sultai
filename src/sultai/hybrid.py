"""Bounded hybrid TCD runner; no GPU, network, external data or dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

from .formation import run_formation
from .lifecycle import run_diagnostics, run_lifecycle

PROTOCOL = "sultai-hybrid-cpu-v1"


def source_identity() -> dict:
    """Bind results to executable bytes, independent of checkout location."""
    package = Path(__file__).resolve().parent
    paths = ("__init__.py", "repair.py", "formation.py", "lifecycle.py", "hybrid.py")
    digests = {f"src/sultai/{name}": hashlib.sha256((package / name).read_bytes()).hexdigest()
               for name in paths}
    canonical = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    return {"files": digests, "aggregate_sha256": hashlib.sha256(canonical).hexdigest()}


def run() -> dict:
    formation, paired_generator = run_formation()
    lifecycle = run_lifecycle(paired_generator.form, formation_label="frozen_paired_generator")
    diagnostics = run_diagnostics()
    gates = {
        "diagnostic.affine_folding": diagnostics["affine_folding"]["identity_verified"],
        "diagnostic.passive_equality": diagnostics["signed_probe"]["passive_identical"],
        "diagnostic.cross_host_probe_equality": diagnostics["signed_probe"]["signed_probe_identity_verified"],
        "diagnostic.nonlinear_changes_need_not_be_opposite": not diagnostics["signed_probe"]["finite_changes_are_opposite"],
    }
    for name, report in (("formation", formation), ("lifecycle", lifecycle)):
        if not report.get("acceptance") or not all(type(value) is bool for value in report["acceptance"].values()):
            raise ValueError(f"{name} must return explicit boolean acceptance gates")
        gates.update({f"{name}.{key}": value for key, value in report["acceptance"].items()})
    result = {
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
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="Print compact JSON with gates and assay summaries.")
    args = parser.parse_args(argv)
    report = run()
    if args.summary:
        report = {key: report[key] for key in ("protocol", "source_identity", "runtime", "passed", "acceptance", "claims")} | {
            "formation_summary": report["formation"].get("summary", {}),
            "lifecycle_summary": report["lifecycle"].get("summary", {}),
        }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
