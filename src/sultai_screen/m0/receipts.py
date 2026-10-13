"""M0 child receipts for the pilot and confirmatory supervisors: payloads, bound files, vectors, AnchorResult, Workloads.

Reads only published receipts and `vectors.pt`; never imports `sultai_screen.screen` or `sultai_screen.pilot`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from sultai.trust import InputDataError, t3_boundary
from sultai_screen.contracts import (
    InstrumentHold,
    Json,
    PilotHold,
    Record,
    file_sha,
    integer,
    mapping,
    number,
    read_record,
    sequence,
    text,
)
from sultai_screen.m0.costs import Workloads
from sultai_screen.m0.identity import CHECKPOINTS, CHECKPOINTS_B, MODULE_RATES, RETUNE_RATES, protocol_identity
from sultai_screen.m0.stats import AnchorResult, ArmResult
from sultai_screen.m0.worker import SURVIVAL_REFIT_STEPS, VECTORS

# The anchor-bootstrap and frontier-bootstrap seed of every pilot statistic (ruling R18 f): the first anchor.
BOOTSTRAP_SEED: Final = 3000
TRAINING_FILES: Final = {
    "host.pt": "checkpoint_sha256",
    "optimizer.pt": "optimizer_sha256",
    "loss-stream.bin": "loss_stream_sha256",
}
STATIC_FILES: Final = {
    "host.pt": "checkpoint_sha256",
    "static-module.pt": "module_sha256",
    "loss-stream.bin": "loss_stream_sha256",
}
# Ruling R17: optimiser steps of each grid kind on the default axis; seconds = measured per-step module kernel x steps.
GRID_STEPS: Final = {
    "module": len(MODULE_RATES) * CHECKPOINTS[-1],
    "module-B": len(MODULE_RATES) * CHECKPOINTS_B[-1],
    "refit": len(MODULE_RATES) * CHECKPOINTS[-1],
    "refit-single": SURVIVAL_REFIT_STEPS,
    "retune": len(RETUNE_RATES) * CHECKPOINTS[-1],
}
# Ledger charge kinds measured by whole children (rulings R17, R23): the stage-3 frontier hosts' noop and evaluate
# children are measured per width; the anchors' noop children are `fold_and_calibrations`.
CHARGE_KINDS: Final = (
    "host:0.5",
    "host:1.0",
    "host:1.5",
    "tcool",
    "static",
    "aff",
    "fold_and_calibrations",
    "noop:0.5",
    "noop:1.5",
    "evaluate:0.5",
    "evaluate:1.5",
)
# Final-review M9: the harm screen measures every raw candidate against the anchor's N0 (§6), M-sq | T-cool included.
HARM_BASELINE: Final = "anchor N0"

Vectors = dict[str, dict[str, Tensor]]


def receipt(root: Path, name: str, action: str, seed: int, lineage: str, certified_source: str) -> Record:
    """A child's payload, refused unless it is a valid M0 receipt of this action, seed and lineage, produced by the
    certified implementation source (final-review M4)."""
    record = read_record(root / name / "result.json")
    if record["software_valid"] is not True or "action" not in record or record["action"] != action:
        raise PilotHold(f"{name} is not a software-valid {action} receipt")
    identity = mapping(record["identity"])
    for key, value in protocol_identity().items():
        if key not in identity or identity[key] != value:
            raise PilotHold(f"{name} receipt carries another protocol identity ({key})")
    if "source_sha256" not in identity or identity["source_sha256"] != certified_source:
        raise PilotHold(f"{name} receipt was produced by another implementation source than the certified one")
    payload = mapping(record["payload"])
    if integer(payload["seed"]) != seed or payload["lineage"] != lineage:
        raise PilotHold(f"{name} receipt belongs to seed {payload['seed']} / {payload['lineage']}")
    return payload


def bound(root: Path, payload: Record, files: dict[str, str]) -> None:
    """Every named file in the child's output is the one its receipt hashed (screen.py:135-142)."""
    for name, key in files.items():
        if key not in payload or not (root / name).is_file() or file_sha(root / name) != text(payload[key]):
            raise PilotHold(f"{root / name} is unbound or changed after publication")


def schedule_matches(name: str, payload: Record, checkpoints: tuple[int, ...], checkpoints_b: tuple[int, ...]) -> None:
    if payload["checkpoints"] != list(checkpoints) or payload["checkpoints_b"] != list(checkpoints_b):
        raise PilotHold(f"{name} ran another schedule axis than 04-schedule.json")


@t3_boundary(
    test="tests/test_m0_pilot.py::ReceiptTests.test_vectors_boundary_requires_n0_and_aligned_floating_loss_and_correctness",
    fingerprint="3c9cb43ac7ec6e22e698e3ad787bebf018457f82f98dbed6a5cce018a1c6e982",
)
def arm_vectors(value: object) -> Vectors:
    """The `vectors.pt` mapping arm -> {"loss", "correct", optional extras}: aligned 1-D floating tensors, N0 present."""
    if not isinstance(value, dict) or "N0" not in value:
        raise InputDataError("vectors must map arm names to tensors and hold N0")
    result: Vectors = {}
    for name, entry in value.items():
        if not isinstance(name, str) or not isinstance(entry, dict) or not {"loss", "correct"} <= set(entry):
            raise InputDataError("each arm needs loss and correct vectors")
        loss: object = entry["loss"]
        if not isinstance(loss, Tensor):
            raise InputDataError("an arm's loss must be a tensor")
        tensors: dict[str, Tensor] = {}
        for key, tensor in entry.items():
            if not isinstance(key, str) or not isinstance(tensor, Tensor):
                raise InputDataError("vector entries must be named tensors")
            if tensor.ndim != 1 or not tensor.is_floating_point() or tensor.shape != loss.shape:
                raise InputDataError("vectors must be aligned 1-D floating tensors")
            tensors[key] = tensor
        result[name] = tensors
    return result


def load_vectors(root: Path, payload: Record) -> Vectors:
    path = root / VECTORS
    if not path.is_file() or file_sha(path) != text(payload["vectors_sha256"]):
        raise PilotHold(f"{path} is unbound or changed after publication")
    unknown: object = torch.load(path, map_location="cpu", weights_only=True)
    return arm_vectors(unknown)


@dataclass(frozen=True)
class Source:
    """One modules or retune child of an anchor: its payload and its per-example vectors."""

    name: str
    payload: Record
    vectors: Vectors


def source(root: Path, name: str, action: str, seed: int, lineage: str, certified_source: str) -> Source:
    payload = receipt(root, name, action, seed, lineage, certified_source)
    return Source(name, payload, load_vectors(root / name, payload))


def _flag(value: Json) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise InputDataError("flag must be a bool or null")


def _optional(value: Json) -> float | None:
    return None if value is None else number(value)


def candidate_rate(label: str) -> float:
    """The rate of a worker harm label "<rate>@<steps>" (ruling R12)."""
    rate, separator, steps = label.partition("@")
    if not separator or not steps.isdigit():
        raise InputDataError(f"harm label {label!r} is not <rate>@<steps>")
    return float(rate)


def inventory(record: Record) -> Record:
    """An arm's labelled raw candidates; an unavailable arm lists its nonfinite candidates (no inventory exists)."""
    if record["status"] == "available":
        return mapping(record["harm_inventory"])
    labelled: Record = {}
    for entry in sequence(record["candidates"]):
        candidate = mapping(entry)
        if candidate["finite"] is False:
            labelled[f"{number(candidate['rate'])}@{integer(candidate['steps'])}"] = None
    return labelled


@dataclass(frozen=True)
class Anchor:
    """One anchor's stats input and the pilot-side record published beside its quantities."""

    seed: int
    result: AnchorResult
    records: dict[str, Record]
    report: Record


def anchor(seed: int, sources: tuple[Source, ...], required: frozenset[str]) -> Anchor:
    """Merge one anchor's sources into an AnchorResult; R11: a required arm unavailable on the anchor holds."""
    if len({text(s.payload["noop_sha256"]) for s in sources}) != 1:
        raise PilotHold(f"anchor {seed}: sources were fitted on different N0 hosts")
    n0 = sources[0].vectors["N0"]
    records: dict[str, Record] = {}
    vectors: Vectors = {}
    for item in sources:
        same = item.vectors["N0"]
        if not torch.equal(same["loss"], n0["loss"]) or not torch.equal(same["correct"], n0["correct"]):
            raise PilotHold(f"anchor {seed}: {item.name} evaluated another N0")
        for name, value in mapping(item.payload["arms"]).items():
            if name in records:
                raise PilotHold(f"anchor {seed}: arm {name} appears in two receipts")
            records[name] = mapping(value)
            if name in item.vectors:
                vectors[name] = item.vectors[name]
    unavailable = sorted(name for name, record in records.items() if record["status"] != "available")
    held = [name for name in unavailable if name in required]
    if held:
        raise InstrumentHold(f"anchor {seed}: required arm(s) unavailable: {held}")
    missing = sorted(required - set(records))
    if missing:
        raise PilotHold(f"anchor {seed}: required arm(s) not run: {missing}")
    available = {name: record for name, record in records.items() if name not in unavailable}
    arms = {
        name: ArmResult(
            vectors[name]["loss"],
            vectors[name]["correct"],
            _flag(record["certified"]),
            _optional(record["rate"]),
            _flag(record["edge_of_grid"]),
        )
        for name, record in available.items()
    }
    survivors = {
        name: tuple(integer(unit) for unit in sequence(record["survivors"]))
        for name, record in available.items()
        if "survivors" in record
    }
    unfolded: dict[str, Tensor] = {}
    nonfinite: list[Json] = []
    for name, record in available.items():
        if "unfolded_development_loss" in record:
            loss = vectors[name]["unfolded_loss"]
            if bool(torch.isfinite(loss).all()):
                unfolded[name] = loss
            else:
                nonfinite.append(name)
    labelled = {name: inventory(record) for name, record in records.items()}
    raw = {
        name: tuple(
            (candidate_rate(label), math.nan if value is None else number(value)) for label, value in items.items()
        )
        for name, items in labelled.items()
        if items
    }
    result = AnchorResult(
        arms=arms,
        noop_loss=n0["loss"],
        noop_correct=n0["correct"],
        tcool_noop_loss=vectors["T-cool"]["loss"],
        fold_residual=number(records["M-aff"]["r_aff"]),
        survivors=survivors,
        unfolded=unfolded,
        raw_candidates=raw,
    )
    refit = available["M-wide→8:rank:refit"] if "M-wide→8:rank:refit" in available else None
    report: Record = {
        "seed": seed,
        "sources": [item.name for item in sources],
        "unavailable_arms": list(unavailable),
        "unfolded_nonfinite": nonfinite,
        "harm_inventory": dict(labelled),
        # M9: one baseline for every arm; M-sq | T-cool's candidates are not measured against the T-cool host's N0.
        "harm_baseline": dict.fromkeys(labelled, f"baseline: {HARM_BASELINE}"),
        # §6: the refit's step-0 (as-sliced) selection loss and its selected rate go beside D2.
        "refit_step0_selection_loss": None if refit is None else refit["step0_selection_loss"],
        "refit_rate": None if refit is None else refit["rate"],
        # §3: every sliced or refit module prints 33h + 16 retained parameters.
        "printed_counts": all(
            integer(available[name]["retained_parameters"]) == 33 * len(units) + 16 for name, units in survivors.items()
        ),
        # §7 V0: the complement hash is unchanged under every re-tune arm.
        "complement_unchanged": all(
            record["complement_unchanged"] is True for record in available.values() if "complement_unchanged" in record
        ),
    }
    return Anchor(seed, result, records, report)


def ledger_charges(ledger: list[Json], names: tuple[str, ...]) -> tuple[float, ...]:
    """The charged process-seconds of the named jobs, read from the supervisor's ledger entries."""
    charged = {text(mapping(entry)["job"]): number(mapping(entry)["charged_gpu_process_seconds"]) for entry in ledger}
    missing = [name for name in names if name not in charged]
    if missing or not names:
        raise PilotHold(f"no ledger charge for {missing or 'an empty job set'}")
    return tuple(charged[name] for name in names)


def derive_workloads(
    timing: tuple[Record, ...], charges: dict[str, tuple[float, ...]], *, fixed_v0: float, ratio: float
) -> tuple[Workloads, Record]:
    """Ruling R17: one-per-card maxima; grids from the per-step module kernel; a raised axis scales grids and front."""
    if set(charges) != set(CHARGE_KINDS) or ratio < 1:
        raise PilotHold("workloads need every measured charge kind and a raise-only axis ratio")
    per_step = max(number(t["module_grid_seconds"]) / integer(t["module_grid_steps"]) for t in timing)
    grid = {kind: per_step * steps * ratio for kind, steps in GRID_STEPS.items()}
    grid["aff"] = max(charges["aff"]) * ratio
    # R23: the whole noop and evaluate children after one trained host, per width. No evaluate child follows a 1.0
    # host (its N0 development loss comes from the M-aff child), so 1.0 takes the larger measured evaluate child.
    evaluate = {scale: max(charges[f"evaluate:{scale}"]) for scale in (0.5, 1.5)}
    noop_evaluate = {
        0.5: max(charges["noop:0.5"]) + evaluate[0.5],
        1.0: max(charges["fold_and_calibrations"]) + max(evaluate.values()),
        1.5: max(charges["noop:1.5"]) + evaluate[1.5],
    }
    workload = Workloads(
        host={
            float(kind.partition(":")[2]): max(values) for kind, values in charges.items() if kind.startswith("host:")
        },
        tcool=max(charges["tcool"]),
        static=max(charges["static"]),
        grid=grid,
        front_rerun=max(number(t["front_rerun_seconds_per_1000_steps"]) for t in timing) * ratio,
        fold_and_calibrations=max(charges["fold_and_calibrations"]),
        evaluation=max(number(t["development_evaluation_seconds"]) for t in timing),
        fixed_v0=fixed_v0,
        noop_evaluate=noop_evaluate,
    )
    sources: Record = {
        "method": "one-per-card maxima (ruling R17); grids = per-step module kernel x steps x axis ratio",
        "per_step_module_seconds": per_step,
        "grid_steps": dict(GRID_STEPS),
        "axis_ratio": ratio,
        "charges": {kind: list(values) for kind, values in charges.items()},
        "aff": "measured stage-2 M-aff child charge x axis ratio",
        "fixed_v0": "total charge at the end of stage 2",
        "noop_evaluate": "per width: the largest noop child plus the largest evaluate child (R23); width 1.0 uses the "
        "anchors' noop children and the larger frontier evaluate child (no evaluate child follows a 1.0 host)",
    }
    return workload, sources


def workloads_record(w: Workloads) -> Record:
    return {
        "host": {str(scale): seconds for scale, seconds in w.host.items()},
        "tcool": w.tcool,
        "static": w.static,
        "grid": dict(w.grid),
        "front_rerun": w.front_rerun,
        "fold_and_calibrations": w.fold_and_calibrations,
        "evaluation": w.evaluation,
        "fixed_v0": w.fixed_v0,
        "noop_evaluate": {str(scale): seconds for scale, seconds in w.noop_evaluate.items()},
    }
