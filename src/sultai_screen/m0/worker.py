"""M0 child worker: the seed-admitting CLI and every child action of the capacity-frontier pilot and confirmatory runs.

Run as `python -B -m sultai_screen.m0.worker <action> ...`. Never imports `sultai_screen.screen` or
`sultai_screen.pilot`; every receipt carries the M0 identity through `m0.provenance.envelope`.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor
from torch.nn import functional as F

from sultai.trust import FitUnavailable
from sultai_screen.contracts import (
    InstrumentHold,
    Json,
    OracleUnavailable,
    PilotHold,
    Record,
    file_sha,
    integer,
    json_value,
    mapping,
    number,
    publish,
    read_record,
    sequence,
    text,
)
from sultai_screen.data import Dataset, as_tensor, install_seal, load_dataset, normalized
from sultai_screen.fit import (
    EVAL_CHUNK,
    Adapter,
    Candidate,
    Grid,
    accuracy,
    adapter_shapes,
    fit_sgd,
    grid,
    logits,
    losses,
    mean_loss,
)
from sultai_screen.host import (
    COOLED_20,
    TCOOL,
    Features,
    Host,
    cached,
    configure,
    constant,
    freeze,
    load_host,
    make_host,
    momentum_buffers,
    optimizer_state,
    recipe_optimizer,
    restore_optimizer,
    state_hash,
    train_epochs,
)
from sultai_screen.m0.calibrate import (
    apply_calibration,
    class_bias,
    load_noop,
    no_op,
    recalibrate_bn,
    save_noop,
    temperature,
)
from sultai_screen.m0.data import features_from_images, seen_fit
from sultai_screen.m0.fold import fold_affine, fold_residual
from sultai_screen.m0.identity import CHECKPOINTS, CHECKPOINTS_B, MODULE_RATES, RALL_CHECKPOINTS, RETUNE_RATES
from sultai_screen.m0.instrument import materiality, posthoc
from sultai_screen.m0.modules import (
    FRONTIER_WIDTHS,
    Cut,
    ablation_cut,
    contract,
    leave_one_out,
    lottery_start,
    random_cut,
    rank_cut,
)
from sultai_screen.m0.provenance import envelope, provenance
from sultai_screen.m0.retune import (
    R272S,
    RAFF,
    RALL,
    RetuneCandidate,
    RetuneGrid,
    Subset,
    complement_hash,
    recalibrate_after,
    retune_grid,
)
from sultai_screen.m0.seeds import BLOCKS, Lineage, admit, width_scale
from sultai_screen.m0.static import (
    StaticHost,
    load_static,
    recalibrate_static,
    save_static,
    static_adapter,
    static_features,
    static_hash,
    static_no_op,
    train_static,
)

Widths = tuple[int, int, int, int]

REPO: Final = Path(__file__).resolve().parents[3]
DEV_ARTIFACT: Final = REPO / "docs/results/gate3-split-2026-10-11/dev-indices-int32-le.bin"
# Attempt 4's intact hosts: payload.final_state_sha256 of
# docs/results/gate3-pilot-2026-10-13-linked-3/intact-<seed>/result.json, replayed on GPU in V0 (pilot and confirmatory).
ATTEMPT4_FINAL_STATE: Final[dict[int, str]] = {
    3000: "f607728bff91b9cab9241c039addbc93d62f2edea70303e1f89e39dd30f1e607",
    3001: "b955026325e2c0fddfc4a1d677fb3795c77b5c200bc3b70ed175782f95750b91",
    3002: "e7f1a21f00651f9229acf285c0f372076b81eeb7949535f8a4ebf229aa31c960",
    3003: "ea01527424ec535a5747f85a95fde2c62d40d4bc7dffb51f88781dbc019f1e86",
    3004: "f9a39442fb96fd2667894d65c707ec0db070ad021644189e64b1d95c2a505777",
    3005: "aa5b5af44ceef02eb555cb62b0f492d34893b4d210b505b08484a6874abdedb9",
}
# §5 as amended by amendment 5 (A5.1): M-wide→{16, 4, 2} refit is one run at M-wide's selected (rate, step) pair.
SURVIVAL_CUTS: Final = (16, 4, 2)
WIDE: Final = 64
SMALL: Final = 8
ACCEPTANCE_TOLERANCE: Final = 1e-6
VECTORS: Final = "vectors.pt"
TRAINING_ACTIONS: Final = frozenset(("train", "cooled20", "replay"))
MODULE_ARMS: Final = (
    "M-sq",
    "M-aff",
    "M-sq-fresh",
    "M-wide",
    "M-wide→8:rank:sliced",
    "M-wide→8:rank:refit",
    "M-wide→8:random:sliced",
    "M-wide→8:random:refit",
    "M-wide→8:ablation",
    *(f"M-wide→{h}:{kind}" for h in SURVIVAL_CUTS for kind in ("sliced", "refit")),
    "M-small",
    "M-small-B",
    "M-exp",
    "lottery",
    "T-cool",
    "M-sq|T-cool",
    "R-temp",
    "R-cal",
    "S-static",
    "S-static:refit",
)
# The sliced record whose survivors the lottery retrains (R31: read from the core receipt by an extended child).
RANK8: Final = "M-wide→8:rank:sliced"
# Arms that need the M-wide grid, and the one that needs the rank-8 survivors.
WIDE_DEPENDENTS: Final = frozenset(name for name in MODULE_ARMS if name.startswith("M-wide") or name == "lottery")
TCOOL_ARMS: Final = frozenset(("T-cool", "M-sq|T-cool"))
STATIC_ARMS: Final = frozenset(("S-static", "S-static:refit"))
RETUNE_ARMS: Final = ("R-272s", "R-aff", "R-all")
RETUNE_DEFAULT: Final = ("R-272s", "R-aff")
# Ruling R25: every module form the arms fit, as (hidden, nonlinear, expanded) for fit.adapter_shapes.
ZERO_MODULE_KINDS: Final[dict[str, tuple[int | None, bool, bool]]] = {
    "affine": (None, False, False),
    "nonlinear": (None, True, False),
    "expanded": (None, True, True),
    "hidden-64": (WIDE, True, False),
    "hidden-8": (SMALL, True, False),
}


@dataclass(frozen=True)
class Request:
    """One admitted child: the parsed CLI with the seed admitted and the width tuple resolved from the lineage."""

    action: str
    output: Path
    data_root: Path
    seed: int
    lineage: Lineage
    widths: Widths
    epochs: int
    checkpoints: tuple[int, ...]
    checkpoints_b: tuple[int, ...]
    arms: tuple[str, ...] | None
    training_root: Path | None
    noop_root: Path | None
    tcool_root: Path | None
    static_root: Path | None
    modules_root: Path | None = None
    # The child's M0 provenance, computed once by worker() and used for every receipt check and the envelope.
    identity: Record | None = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def tensor_sha(values: Tensor) -> str:
    return hashlib.sha256(values.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _ints(values: tuple[int, ...]) -> list[Json]:
    return [int(value) for value in values]


def _texts(values: list[str]) -> list[Json]:
    return [str(value) for value in values]


def _identity(request: Request) -> Record:
    return {
        "seed": request.seed,
        "lineage": request.lineage,
        "scale": width_scale(request.lineage),
        "widths": _ints(request.widths),
    }


def _receipt(
    root: Path, actions: frozenset[str], seed: int, lineage: Lineage, widths: Widths, identity: Record | None
) -> Record:
    """The payload of a published M0 receipt, refused unless source, protocol, seed, lineage and widths all match.

    `identity` is the child's provenance (computed once by worker()); None computes it here (direct callers).
    """
    record = read_record(root / "result.json")
    for key in ("action", "identity", "payload"):
        if key not in record:
            raise PilotHold(f"{root} holds no published M0 receipt ({key} missing)")
    if text(record["action"]) not in actions:
        raise PilotHold(f"{root} is a {record['action']} receipt, not one of {sorted(actions)}")
    original = mapping(record["identity"])
    current = provenance() if identity is None else identity
    for key in ("source_sha256", "protocol_sha256"):
        if key not in original or original[key] != current[key]:
            raise PilotHold(f"receipt {root} belongs to another source/protocol ({key})")
    payload = mapping(record["payload"])
    for key in ("seed", "lineage", "widths"):
        if key not in payload:
            raise PilotHold(f"receipt {root} records no {key}")
    if integer(payload["seed"]) != seed:
        raise PilotHold(f"receipt {root} belongs to seed {payload['seed']}, not {seed}")
    if payload["lineage"] != lineage:
        raise PilotHold(f"receipt {root} belongs to lineage {payload['lineage']}, not {lineage}")
    if payload["widths"] != _ints(widths):
        raise PilotHold(f"receipt {root} records width tuple {payload['widths']}, not {list(widths)}")
    return payload


def _bound(root: Path, payload: Record, files: dict[str, str]) -> None:
    for name, key in files.items():
        # F2: a missing bound file holds like a changed one (receipts.bound's rule), never a FileNotFoundError.
        if key not in payload or not (root / name).is_file() or file_sha(root / name) != text(payload[key]):
            raise PilotHold(f"{root / name} changed after publication or is unbound")


def _training(
    root: Path,
    seed: int,
    lineage: Lineage,
    widths: Widths,
    device: torch.device,
    *,
    frozen: bool,
    identity: Record | None = None,
) -> tuple[Host, Record]:
    payload = _receipt(root, TRAINING_ACTIONS, seed, lineage, widths, identity)
    _bound(
        root,
        payload,
        {"host.pt": "checkpoint_sha256", "optimizer.pt": "optimizer_sha256", "loss-stream.bin": "loss_stream_sha256"},
    )
    host = load_host(root / "host.pt", seed, 0, device, widths=widths, frozen=frozen)
    if "final_state_sha256" not in payload or state_hash(host) != text(payload["final_state_sha256"]):
        raise PilotHold("loaded host state differs from training receipt")
    return host, payload


def load_training(
    root: Path,
    seed: int,
    lineage: Lineage,
    widths: Widths,
    device: torch.device,
    *,
    frozen: bool,
    identity: Record | None = None,
) -> Host:
    """The M0 analogue of screen._training, which also refuses a receipt of another lineage or width tuple (§2)."""
    return _training(root, seed, lineage, widths, device, frozen=frozen, identity=identity)[0]


def _request_training(request: Request, device: torch.device, *, frozen: bool) -> tuple[Host, Record]:
    root = _required(request.training_root, f"{request.action}")
    return _training(
        root, request.seed, request.lineage, request.widths, device, frozen=frozen, identity=request.identity
    )


def _request_receipt(request: Request, root: Path, actions: frozenset[str]) -> Record:
    return _receipt(root, actions, request.seed, request.lineage, request.widths, request.identity)


def _required(root: Path | None, purpose: str) -> Path:
    if root is None:
        raise PilotHold(f"{purpose} requires its published receipt root")
    return root


def _noop_host(request: Request, device: torch.device) -> tuple[Host, Record]:
    root = _required(request.noop_root, "fitting on N0")
    payload = _request_receipt(request, root, frozenset(("noop",)))
    return load_noop(root, request.seed, request.widths, text(payload["noop_sha256"]), device), payload


def _optimizer(host: Host, path: Path, device: torch.device) -> torch.optim.SGD:
    unknown: object = torch.load(path, map_location=device, weights_only=True)
    expected = {name: parameter.detach() for name, parameter in host.named_parameters()}
    return restore_optimizer(host, optimizer_state(unknown, expected))


def _stream(output: Path, stream: Tensor) -> str:
    with (output / "loss-stream.bin").open("xb") as target:
        target.write(stream.numpy().tobytes())
    return tensor_sha(stream)


# --- training children -------------------------------------------------------------------------------------------


def _probe(seed: int, data: Dataset, widths: Widths) -> Record:
    """A separate fresh host probes the 2-image forward/backward without changing training state."""
    probe = make_host(seed, 0, data.images.device, widths=widths)
    inputs = data.images[data.train[:2].to(data.images.device)].float() / 255
    output = probe(inputs)
    gradients = torch.autograd.grad(output.sum(), tuple(probe.parameters()))
    return {
        "output_sha256": tensor_sha(output),
        "gradient_sha256": hashlib.sha256("".join(tensor_sha(g) for g in gradients).encode()).hexdigest(),
    }


def train_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """A 10-epoch constant-rate host (anchor or frontier) or, for the cooled-20 lineage, 20 epochs under COOLED_20.

    The optimiser is built here (bitwise train_host for 10 epochs) so its momentum buffers can be saved for T-cool.
    """
    cooled = request.lineage == "cooled-20"
    epochs = 20 if cooled else 10
    if request.epochs != epochs:
        raise PilotHold(f"lineage {request.lineage} trains exactly {epochs} epochs")
    if request.action == "cooled20" and not cooled:
        raise PilotHold("cooled20 trains the cooled-20 lineage only")
    host = make_host(request.seed, 0, device, widths=request.widths)
    initial = state_hash(host)
    forward_backward = _probe(request.seed, data, request.widths)
    optimizer = recipe_optimizer(host.parameters())
    hashes, stream, elapsed = train_epochs(
        host,
        host,
        data,
        request.seed,
        epochs=range(epochs),
        rates=COOLED_20 if cooled else constant(0.05),
        optimizer=optimizer,
    )
    output = request.output
    torch.save(dict(host.state_dict()), output / "host.pt")
    torch.save(momentum_buffers(optimizer, host), output / "optimizer.pt")
    stream_sha = _stream(output, stream)
    freeze(host)
    return {
        **_identity(request),
        "epochs": epochs,
        "schedule": "cooled-20" if cooled else "constant-0.05",
        "steps": len(stream),
        "initial_state_sha256": initial,
        "epoch_state_sha256": _texts(hashes),
        "loss_stream_sha256": stream_sha,
        "checkpoint_sha256": file_sha(output / "host.pt"),
        "optimizer_sha256": file_sha(output / "optimizer.pt"),
        "final_state_sha256": state_hash(host),
        "training_seconds": elapsed,
        "parameters": sum(p.numel() for p in host.parameters()),
        "forward_backward": forward_backward,
    }


def replay_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """Retrain an attempt-4 anchor at s = 1.0 and compare its final state with the pinned attempt-4 string."""
    if request.lineage != "anchor" or request.seed not in ATTEMPT4_FINAL_STATE:
        raise PilotHold("replay covers the six attempt-4 anchors 3000-3005 of the anchor lineage only")
    payload = train_action(request, data, device)
    pinned = ATTEMPT4_FINAL_STATE[request.seed]
    payload["attempt4_final_state_sha256"] = pinned
    payload["matches_attempt4"] = payload["final_state_sha256"] == pinned
    return payload


def _noop_rule(host: Host, checkpoint: Path, request: Request, data: Dataset, device: torch.device) -> Record:
    """§4 N0 on a frozen host saved at checkpoint: R-bn on an unsealed copy versus raw, decided on selection."""
    unsealed = load_host(checkpoint, request.seed, 0, device, widths=request.widths, frozen=False)
    fit_ids = seen_fit(data)
    recalibrated = recalibrate_bn(unsealed, data.images, fit_ids)
    choice = no_op(host, recalibrated, data)
    return {
        "noop_choice": choice.label,
        "noop_sha256": save_noop(choice.model, request.output),
        "raw_sha256": state_hash(host),
        "recalibrated_sha256": state_hash(recalibrated),
        # Amendment 5 (A5.2 item 2): the fit indices the recalibration consumed and the two losses that decided.
        "fit_indices_sha256": tensor_sha(fit_ids),
        "raw_selection_loss": choice.raw_selection_loss,
        "rbn_selection_loss": choice.recalibrated_selection_loss,
    }


def tcool_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """Resume a 10-epoch host from its saved momentum for epochs 10-14 under TCOOL, then apply its own N0 rule."""
    root = _required(request.training_root, "T-cool")
    host, source = _request_training(request, device, frozen=False)
    if source["epochs"] != 10 or source["schedule"] != "constant-0.05":
        raise PilotHold("T-cool resumes a ten-epoch constant-rate host only")
    optimizer = _optimizer(host, root / "optimizer.pt", device)
    hashes, stream, elapsed = train_epochs(
        host, host, data, request.seed, epochs=range(10, 15), rates=TCOOL, optimizer=optimizer
    )
    output = request.output
    torch.save(dict(host.state_dict()), output / "host.pt")
    stream_sha = _stream(output, stream)
    freeze(host)
    started = time.monotonic()
    rule = _noop_rule(host, output / "host.pt", request, data, device)
    _sync(device)
    return {
        **_identity(request),
        "epochs": 5,
        "schedule": "tcool",
        "steps": len(stream),
        "epoch_state_sha256": _texts(hashes),
        "loss_stream_sha256": stream_sha,
        "checkpoint_sha256": file_sha(output / "host.pt"),
        "final_state_sha256": state_hash(host),
        "source_final_state_sha256": text(source["final_state_sha256"]),
        "training_seconds": elapsed,
        "noop_seconds": time.monotonic() - started,
        **rule,
    }


def static_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """S-static: the anchor's seed trained with the 16x16 module present from step zero (Task 11), then the no-op
    rule on the complete model (amendment 4, A4.1): the published files are the chosen model."""
    static, hashes, stream, elapsed, initial = train_static(request.seed, data, device, request.widths)
    raw = save_static(static, request.output)
    stream_sha = _stream(request.output, stream)
    _sync(device)
    started = time.monotonic()
    unsealed = load_static(
        request.output, request.seed, request.widths, device, text(raw["static_sha256"]), frozen=False
    )
    recalibrated = recalibrate_static(unsealed, data.images, seen_fit(data))
    choice = static_no_op(static, recalibrated, data)
    record = raw if choice.model is static else save_static(choice.model, request.output)
    _sync(device)
    noop_seconds = time.monotonic() - started
    return {
        **_identity(request),
        "epochs": 10,
        "schedule": "constant-0.05",
        "steps": len(stream),
        # M8: the inner host's step-zero state, for the pilot to compare with the anchor's make_host state.
        "initial_state_sha256": initial,
        "epoch_state_sha256": _texts(hashes),
        "loss_stream_sha256": stream_sha,
        "checkpoint_sha256": file_sha(request.output / "host.pt"),
        "training_seconds": elapsed,
        "noop_choice": choice.label,
        "raw_static_sha256": raw["static_sha256"],
        "recalibrated_static_sha256": static_hash(recalibrated),
        "raw_selection_loss": choice.raw_selection_loss,
        "rbn_selection_loss": choice.recalibrated_selection_loss,
        "noop_seconds": noop_seconds,
        **record,
    }


def _static(request: Request, device: torch.device) -> tuple[StaticHost, Record]:
    root = _required(request.static_root, "S-static")
    payload = _request_receipt(request, root, frozenset(("static",)))
    _bound(
        root,
        payload,
        {"host.pt": "checkpoint_sha256", "static-module.pt": "module_sha256", "loss-stream.bin": "loss_stream_sha256"},
    )
    return load_static(root, request.seed, request.widths, device, text(payload["static_sha256"])), payload


# --- instrument and N0 -------------------------------------------------------------------------------------------


def instrument_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """P1 materiality and the post-hoc known answer at alpha = 0.1, in one child on the raw trained host."""
    host, _ = _request_training(request, device, frozen=True)
    cache = cached(host, data)
    return {
        **_identity(request),
        "materiality": materiality(host, cache, request.seed),
        "posthoc": posthoc(host, cache, request.seed, request.output),
    }


def _calibrated_loss(values: Tensor, labels: Tensor) -> float:
    return float(F.cross_entropy(values, labels, reduction="none").double().mean())


def zero_module_identity(host: Host, development: Features) -> dict[str, bool]:
    """R25 (§3, §7 V0): at zero weights every module form is bitwise identity with the host on development."""
    device = development.read.device
    baseline = losses(host, development, None)
    flags: dict[str, bool] = {}
    for kind, (hidden, nonlinear, expanded) in ZERO_MODULE_KINDS.items():
        shapes = adapter_shapes(hidden=hidden, nonlinear=nonlinear, expanded=expanded)
        zero = Adapter(tuple(torch.zeros(shape, device=device) for shape in shapes), nonlinear)
        flags[kind] = torch.equal(baseline, losses(host, development, zero, check_finite=False))
    return flags


def noop_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """N0 (raw versus R-bn on selection), R-temp and R-cal fitted on the fit examples; never a development number."""
    raw, _ = _request_training(request, device, frozen=True)
    unsealed, _ = _request_training(request, device, frozen=False)
    fit_ids = seen_fit(data)
    recalibrated = recalibrate_bn(unsealed, data.images, fit_ids)
    choice = no_op(raw, recalibrated, data)
    chosen, label = choice.model, choice.label
    noop_sha = save_noop(chosen, request.output)
    fit = features_from_images(chosen, data, fit_ids)
    fit_logits = logits(chosen, fit, None)
    t = temperature(fit_logits, fit.labels)
    t_cal, bias = class_bias(fit_logits, fit.labels)
    selection = features_from_images(chosen, data, data.selection)
    selection_logits = logits(chosen, selection, None)
    identity = zero_module_identity(chosen, features_from_images(chosen, data, data.development))
    return {
        **_identity(request),
        "noop_choice": label,
        "noop_sha256": noop_sha,
        "raw_sha256": state_hash(raw),
        "recalibrated_sha256": state_hash(recalibrated),
        "fit_indices_sha256": tensor_sha(fit_ids),
        "temperature": t,
        "class_bias": [t_cal, *(float(value) for value in bias.double().cpu().unbind())],
        "raw_selection_loss": choice.raw_selection_loss,
        "rbn_selection_loss": choice.recalibrated_selection_loss,
        "rtemp_selection_loss": _calibrated_loss(selection_logits / t, selection.labels),
        "rcal_selection_loss": _calibrated_loss(apply_calibration(selection_logits, t_cal, bias), selection.labels),
        "fit_examples": len(fit_ids),
        # Flags only: the development losses compared are never published (R25).
        "zero_module_identity": dict(identity),
    }


# --- module arms -------------------------------------------------------------------------------------------------


def _arms(requested: tuple[str, ...] | None, known: tuple[str, ...], default: tuple[str, ...]) -> tuple[str, ...]:
    if requested is None:
        return default
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise PilotHold(f"unknown arm(s) for this action: {unknown}")
    if len(set(requested)) != len(requested):
        raise PilotHold("an arm is named twice")
    return tuple(name for name in known if name in requested)


def _label(rate: float, steps: int) -> str:
    return f"{rate}@{steps}"


def _mean(values: Tensor) -> float:
    return float(values.double().mean())


def _optional_mean(values: Tensor) -> float | None:
    return _mean(values) if bool(torch.isfinite(values).all()) else None


def _booking(candidates: tuple[Candidate, ...]) -> Record:
    """M5 (§4 booking): the arm's whole runs, i.e. its final-checkpoint candidates, which carry each run's cumulative
    example passes and parameter-steps and its wall (fit.grid books a run's wall once, there)."""
    last = max(candidate.steps for candidate in candidates)
    final = [candidate for candidate in candidates if candidate.steps == last]
    return {
        "example_passes": sum(candidate.example_passes for candidate in final),
        "parameter_steps": sum(candidate.parameter_steps for candidate in final),
        "wall_seconds": sum(candidate.wall_seconds for candidate in final),
    }


class Collector:
    """Arm records for the JSON payload and the aligned per-example development vectors for vectors.pt."""

    def __init__(self) -> None:
        self.records: Record = {}
        self.vectors: dict[str, dict[str, Tensor]] = {}
        self.unavailable: list[str] = []

    def attempt(self, names: tuple[str, ...], body: Callable[[], Json], *, cause: str | None = None) -> None:
        """Run one arm, or a block of arms sharing one fit; with no finite candidate, each unrecorded arm is unavailable.

        R11: the anchor is retained and the child continues; the arm keeps the failed fit's candidate reports and has
        no vectors. `cause` names the arm whose fit failed (default: the single arm itself), so a dependent's record
        never passes another fit's candidates off as its own.
        """
        try:
            body()
        except OracleUnavailable as error:
            candidates: list[Json] = []
            if error.candidates is not None:
                candidates.extend(error.candidates)
            for name in names:
                if name not in self.records:
                    self.records[name] = {
                        "status": "unavailable",
                        "cause": names[0] if cause is None else cause,
                        "error": str(error),
                        "candidates": candidates,
                    }
                    self.unavailable.append(name)

    def put(self, name: str, loss: Tensor, correct: Tensor, record: Record) -> Record:
        self.vectors[name] = {"loss": loss.detach().cpu(), "correct": correct.detach().cpu()}
        full: Record = {
            "status": "available",
            "development_loss": _mean(loss),
            "development_accuracy": _mean(correct),
            **record,
        }
        self.records[name] = full
        return full

    def fixed(self, name: str, host: Host, development: Features, adapter: Adapter | None, record: Record) -> Record:
        """An arm with no fit of its own (an evaluated host or a sliced module)."""
        base: Record = {
            "certified": None,
            "rate": None,
            "steps": None,
            "edge_of_grid": None,
            "step0_selection_loss": None,
            "retained_parameters": 0 if adapter is None else adapter.count(),
            "report": None,
            "harm_inventory": {},
        }
        base.update(record)
        loss = losses(host, development, adapter)
        return self.put(name, loss, accuracy(host, development, adapter), base)

    def fitted(
        self,
        name: str,
        host: Host,
        development: Features,
        fit: Grid,
        adapter: Adapter | None = None,
    ) -> Record:
        """A grid arm: the selected candidate (or its retained form) on development, and every raw candidate's harm."""
        chosen = fit.best.adapter if adapter is None else adapter
        record: Record = {
            "certified": fit.plateau,
            "rate": fit.best.rate,
            "steps": fit.best.steps,
            "edge_of_grid": fit.edge_of_grid,
            "step0_selection_loss": fit.best.step0_selection,
            "selection_loss": fit.best.selection,
            "retained_parameters": chosen.count(),
            "report": fit.report(),
            "harm_inventory": _harm(host, development, fit.candidates),
            "booking": _booking(fit.candidates),
        }
        return self.put(name, losses(host, development, chosen), accuracy(host, development, chosen), record)

    def single(self, name: str, host: Host, development: Features, run: Candidate) -> Record:
        """A single SGD run (the h = 16/4/2 survival refit): its one candidate, never a grid."""
        if run.selection is None:
            raise OracleUnavailable(f"{name}: the single refit run is nonfinite", candidates=(run.report(),))
        record: Record = {
            "certified": None,
            "rate": run.rate,
            "steps": run.steps,
            "edge_of_grid": None,
            "step0_selection_loss": run.step0_selection,
            "selection_loss": run.selection,
            "retained_parameters": run.adapter.count(),
            "report": {"best": run.report(), "candidates": [run.report()]},
            "harm_inventory": _harm(host, development, (run,)),
            "booking": _booking((run,)),
        }
        return self.put(name, losses(host, development, run.adapter), accuracy(host, development, run.adapter), record)

    def cut(self, name: str, host: Host, development: Features, cut: Cut) -> Record:
        """A sliced module with the mean fold; the unfolded slice's losses and the survivors are recorded beside."""
        unfolded = losses(host, development, cut.unfolded, check_finite=False)
        record = self.fixed(
            name,
            host,
            development,
            cut.folded,
            {"survivors": _ints(cut.survivors), "unfolded_development_loss": _optional_mean(unfolded)},
        )
        self.vectors[name]["unfolded_loss"] = unfolded.detach().cpu()
        return record


def _harm(host: Host, development: Features, candidates: tuple[Candidate, ...]) -> Record:
    """Every raw candidate's mean development loss, labelled rate@steps; selection is never re-done on it."""
    inventory: Record = {}
    for candidate in candidates:
        values = losses(host, development, candidate.adapter, check_finite=False)
        inventory[_label(candidate.rate, candidate.steps)] = _optional_mean(values)
    return inventory


@dataclass(frozen=True)
class Site:
    """One host's fit, selection and development features and the schedule axis its module grids run on."""

    host: Host
    fit: Features
    selection: Features
    development: Features
    seed: int
    checkpoints: tuple[int, ...]

    def grid(
        self,
        *,
        fit: Features | None = None,
        start: Adapter | None = None,
        hidden: int | None = None,
        expanded: bool = False,
        checkpoints: tuple[int, ...] | None = None,
    ) -> Grid:
        """The module grid (rates MODULE_RATES) on this site; fit defaults to the seen fit features."""
        return grid(
            self.host,
            self.fit if fit is None else fit,
            self.selection,
            self.seed,
            rates=MODULE_RATES,
            checkpoints=self.checkpoints if checkpoints is None else checkpoints,
            start=start,
            hidden=hidden,
            expanded=expanded,
        )

    def affine(self) -> Grid:
        """M-aff: the five-rate affine grid."""
        return grid(
            self.host,
            self.fit,
            self.selection,
            self.seed,
            rates=RETUNE_RATES,
            checkpoints=self.checkpoints,
            nonlinear=False,
        )


def _schedule(request: Request) -> Record:
    return {"checkpoints": _ints(request.checkpoints), "checkpoints_b": _ints(request.checkpoints_b)}


def modules_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """The module arms of one anchor on its published N0 (seen fit data, selection, development endpoint).

    An arm whose fit has no finite candidate is recorded unavailable with every candidate report and the child
    continues (ruling R11: anchor retained, arm unavailable).
    """
    arms = _arms(request.arms, MODULE_ARMS, MODULE_ARMS)
    wanted = set(arms)
    host, noop = _noop_host(request, device)
    cache = cached(host, data)
    fit_ids = seen_fit(data)
    site = Site(
        host,
        features_from_images(host, data, fit_ids),
        cache.selection,
        cache.development,
        request.seed,
        request.checkpoints,
    )
    development = site.development
    collect = Collector()
    collect.vectors["N0"] = {
        "loss": losses(host, development, None).cpu(),
        "correct": accuracy(host, development, None).cpu(),
    }
    payload: Record = {**_identity(request), **_schedule(request), "noop_sha256": text(noop["noop_sha256"])}
    payload["r_aff"] = None

    def affine() -> None:
        fitted = site.affine()
        folded = fold_affine(host, fitted.best.adapter)
        r_aff = fold_residual(host, fitted.best.adapter, folded, development, data, data.development)
        collect.fitted("M-aff", host, development, fitted)["r_aff"] = r_aff
        payload["r_aff"] = r_aff

    def expanded() -> None:
        fitted = site.grid(expanded=True)
        collect.fitted("M-exp", host, development, fitted, contract(fitted.best.adapter))

    plain: dict[str, Callable[[], Json]] = {
        "M-sq": lambda: collect.fitted("M-sq", host, development, site.grid()),
        "M-aff": affine,
        "M-sq-fresh": lambda: collect.fitted("M-sq-fresh", host, development, site.grid(fit=cache.conditioning)),
        "M-small": lambda: collect.fitted("M-small", host, development, site.grid(hidden=SMALL)),
        "M-small-B": lambda: collect.fitted(
            "M-small-B", host, development, site.grid(hidden=SMALL, checkpoints=request.checkpoints_b)
        ),
        "M-exp": expanded,
    }
    for name in ("M-sq", "M-aff", "M-sq-fresh"):
        if name in wanted:
            collect.attempt((name,), plain[name])
    # R31: with --modules-root the lottery is an extended child that reads the core receipt's rank-8 survivors; only
    # without it (every arm in one child) does it follow the in-process M-wide grid.
    from_receipt = "lottery" in wanted and request.modules_root is not None
    dependents = tuple(
        name for name in MODULE_ARMS if name in wanted & WIDE_DEPENDENTS and not (from_receipt and name == "lottery")
    )
    if dependents:
        collect.attempt(dependents, lambda: _wide_arms(site, wanted, collect, device), cause="M-wide")
    if from_receipt:
        survivors, vectors_sha = rank8_survivors(request, host)

        def lottery() -> None:
            fitted = site.grid(start=lottery_start(request.seed, survivors, device), hidden=SMALL)
            record = collect.fitted("lottery", host, development, fitted)
            record.update(
                {
                    "survivors": _ints(survivors),
                    "survivors_source": RANK8,
                    "survivors_receipt_vectors_sha256": vectors_sha,
                }
            )

        collect.attempt(("lottery",), lottery)
    for name in ("M-small", "M-small-B", "M-exp"):
        if name in wanted:
            collect.attempt((name,), plain[name])
    if wanted & TCOOL_ARMS:
        _tcool_arms(request, data, device, wanted, collect, payload)
    if wanted & {"R-temp", "R-cal"}:
        _calibration_arms(host, development, noop, wanted, collect)
    if wanted & STATIC_ARMS:
        _static_arms(request, data, device, wanted, collect, payload)
    torch.save(collect.vectors, request.output / VECTORS)
    payload.update(
        {
            "arms_run": list(arms),
            "arms": collect.records,
            "unavailable_arms": _texts(collect.unavailable),
            "n0": {
                "development_loss": _mean(collect.vectors["N0"]["loss"]),
                "development_accuracy": _mean(collect.vectors["N0"]["correct"]),
            },
            "fit_examples": len(fit_ids),
            "vectors_sha256": file_sha(request.output / VECTORS),
        }
    )
    return payload


def rank8_survivors(request: Request, host: Host) -> tuple[tuple[int, ...], str]:
    """R31: the rank-8 survivors of the anchor's core M-wide grid, read from its published modules receipt (bound by its
    vectors.pt and fitted on this N0), never re-fitted; a receipt without them holds."""
    root = _required(request.modules_root, "the lottery (the core modules child's output, --modules-root)")
    receipt = _request_receipt(request, root, frozenset(("modules",)))
    _bound(root, receipt, {VECTORS: "vectors_sha256"})
    if "noop_sha256" not in receipt or receipt["noop_sha256"] != state_hash(host):
        raise PilotHold("the core modules receipt was fitted on another N0 than the loaded noop.pt")
    arms = mapping(receipt["arms"]) if "arms" in receipt else {}
    record = mapping(arms[RANK8]) if RANK8 in arms else {}
    if "status" not in record or record["status"] != "available" or "survivors" not in record:
        raise PilotHold(f"the core modules receipt holds no available {RANK8} record with its survivors")
    survivors = tuple(integer(unit) for unit in sequence(record["survivors"]))
    if len(survivors) != SMALL:
        raise PilotHold(f"the core receipt's {RANK8} survivors are not {SMALL} units")
    return survivors, text(receipt["vectors_sha256"])


def _wide_arms(site: Site, wanted: set[str], collect: Collector, device: torch.device) -> None:
    """M-wide and everything cut from it; an unavailable M-wide makes every dependent arm unavailable (caller)."""
    host, fit, development, seed = site.host, site.fit, site.development, site.seed
    wide = site.grid(hidden=WIDE)
    module = wide.best.adapter
    if "M-wide" in wanted:
        collect.fitted("M-wide", host, development, wide)
    rank8 = rank_cut(module, fit.read, SMALL)
    for kind, cut in (("rank", rank8), ("random", random_cut(module, fit.read, SMALL, seed))):
        if f"M-wide→8:{kind}:sliced" in wanted:
            collect.cut(f"M-wide→8:{kind}:sliced", host, development, cut)
        name = f"M-wide→8:{kind}:refit"
        if name in wanted:

            def refit(name: str = name, cut: Cut = cut) -> None:
                fitted = site.grid(start=cut.folded, hidden=SMALL)
                collect.fitted(name, host, development, fitted)["survivors"] = _ints(cut.survivors)

            collect.attempt((name,), refit)
    if "M-wide→8:ablation" in wanted:
        ablation = leave_one_out(host, fit, module)
        collect.cut("M-wide→8:ablation", host, development, ablation_cut(module, fit.read, SMALL, ablation))
    for h in SURVIVAL_CUTS:
        if f"M-wide→{h}:sliced" not in wanted and f"M-wide→{h}:refit" not in wanted:
            continue
        cut = rank_cut(module, fit.read, h)
        if f"M-wide→{h}:sliced" in wanted:
            collect.cut(f"M-wide→{h}:sliced", host, development, cut)
        name = f"M-wide→{h}:refit"
        if name in wanted:

            def single(name: str = name, cut: Cut = cut, h: int = h) -> None:
                run = fit_sgd(
                    host,
                    fit,
                    site.selection,
                    seed=seed,
                    rate=wide.best.rate,
                    steps=wide.best.steps,
                    minibatch=True,
                    start=cut.folded,
                    hidden=h,
                )
                collect.single(name, host, development, run)["survivors"] = _ints(cut.survivors)

            collect.attempt((name,), single)
    if "lottery" in wanted:

        def lottery() -> None:
            fitted = site.grid(start=lottery_start(seed, rank8.survivors, device), hidden=SMALL)
            collect.fitted("lottery", host, development, fitted)["survivors"] = _ints(rank8.survivors)

        collect.attempt(("lottery",), lottery)


def _tcool_arms(
    request: Request, data: Dataset, device: torch.device, wanted: set[str], collect: Collector, payload: Record
) -> None:
    """T-cool (its host after its own N0 rule, no module) and M-sq | T-cool, each on the T-cool host's own site."""
    root = _required(request.tcool_root, "T-cool arms")
    receipt = _request_receipt(request, root, frozenset(("tcool",)))
    _bound(root, receipt, {"host.pt": "checkpoint_sha256", "loss-stream.bin": "loss_stream_sha256"})
    host = load_noop(root, request.seed, request.widths, text(receipt["noop_sha256"]), device)
    payload["tcool_noop_sha256"] = text(receipt["noop_sha256"])
    cache = cached(host, data)
    if "T-cool" in wanted:
        collect.fixed("T-cool", host, cache.development, None, {"noop_choice": receipt["noop_choice"]})
    if "M-sq|T-cool" in wanted:
        fit = features_from_images(host, data, seen_fit(data))
        site = Site(host, fit, cache.selection, cache.development, request.seed, request.checkpoints)
        collect.attempt(("M-sq|T-cool",), lambda: collect.fitted("M-sq|T-cool", host, cache.development, site.grid()))


def _calibration_arms(host: Host, development: Features, noop: Record, wanted: set[str], collect: Collector) -> None:
    """R-temp and R-cal: the noop child's fit-example calibrations applied to N0's development logits."""
    raw = logits(host, development, None)
    t = number(noop["temperature"])
    fitted = [number(value) for value in sequence(noop["class_bias"])]
    if len(fitted) != 11:
        raise PilotHold("R-cal records a temperature and ten class biases")
    calibrations: dict[str, tuple[Tensor, Record, int]] = {
        "R-temp": (raw / t, {"temperature": t}, 1),
        "R-cal": (
            apply_calibration(raw, fitted[0], torch.tensor(fitted[1:], dtype=torch.float64)),
            {"class_bias": json_value(fitted)},
            11,
        ),
    }
    for name, (values, extra, count) in calibrations.items():
        if name not in wanted:
            continue
        loss = F.cross_entropy(values, development.labels, reduction="none")
        if not bool(torch.isfinite(loss).all()):
            raise PilotHold(f"nonfinite {name} development evaluation")
        correct = (values.argmax(1) == development.labels).float()
        record: Record = {
            "certified": None,
            "rate": None,
            "steps": None,
            "edge_of_grid": None,
            "step0_selection_loss": None,
            "retained_parameters": count,
            "report": None,
            "harm_inventory": {},
        }
        record.update(extra)
        collect.put(name, loss, correct, record)


def _static_arms(
    request: Request, data: Dataset, device: torch.device, wanted: set[str], collect: Collector, payload: Record
) -> None:
    """S-static's complete model as trained (no refit) and the module grid warm-started from its own module."""
    static, receipt = _static(request, device)
    payload["static_sha256"] = text(receipt["static_sha256"])
    development = static_features(static, data, data.development)
    module = static_adapter(static)
    if "S-static" in wanted:
        collect.fixed("S-static", static.host, development, module, {})
    if "S-static:refit" in wanted:
        site = Site(
            static.host,
            static_features(static, data, seen_fit(data)),
            static_features(static, data, data.selection),
            development,
            request.seed,
            request.checkpoints,
        )
        collect.attempt(
            ("S-static:refit",),
            lambda: collect.fitted("S-static:refit", static.host, development, site.grid(start=module)),
        )


# --- re-tune arms ------------------------------------------------------------------------------------------------


def _retune_harm(candidates: tuple[RetuneCandidate, ...], evaluate: Callable[[Host], Features]) -> Record:
    inventory: Record = {}
    for candidate in candidates:
        values = losses(candidate.host, evaluate(candidate.host), None, check_finite=False)
        inventory[_label(candidate.rate, candidate.steps)] = _optional_mean(values)
    return inventory


def _retune_record(fit: RetuneGrid, subset: Subset, source: Host, evaluate: Callable[[Host], Features]) -> Record:
    best = fit.best
    last = max(candidate.steps for candidate in fit.candidates)
    runs = [candidate for candidate in fit.candidates if candidate.steps == last]
    return {
        "certified": fit.plateau,
        "rate": best.rate,
        "steps": best.steps,
        "edge_of_grid": fit.edge_of_grid,
        "step0_selection_loss": None,
        "selection_loss": best.selection,
        "retained_parameters": 0,
        "complement_sha256": best.complement,
        "complement_unchanged": best.complement == complement_hash(source, subset),
        "parameter_steps": sum(run.parameter_steps for run in runs),
        "example_passes": sum(run.example_passes for run in runs),
        "wall_seconds": sum(candidate.wall_seconds for candidate in fit.candidates),
        "booking": {
            "example_passes": sum(run.example_passes for run in runs),
            "parameter_steps": sum(run.parameter_steps for run in runs),
            "wall_seconds": sum(candidate.wall_seconds for candidate in fit.candidates),
        },
        "report": fit.report(),
        "harm_inventory": _retune_harm(fit.candidates, evaluate),
    }


def retune_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """R-272s (cached site), R-aff (re-runs the front) and, when named, R-all with its post-fit R-bn, on N0."""
    arms = _arms(request.arms, RETUNE_ARMS, RETUNE_DEFAULT)
    host, noop = _noop_host(request, device)
    cache = cached(host, data)
    fit_ids = seen_fit(data)
    collect = Collector()
    collect.vectors["N0"] = {
        "loss": losses(host, cache.development, None).cpu(),
        "correct": accuracy(host, cache.development, None).cpu(),
    }

    def cached_site(candidate: Host) -> Features:
        # R-272s moves tail[0] only, so the N0 front's cached development site is the candidate's own.
        return cache.development

    def from_images(candidate: Host) -> Features:
        return features_from_images(candidate, data, data.development)

    plans: dict[str, tuple[Subset, bool, tuple[int, ...]]] = {
        "R-272s": (R272S(host), False, request.checkpoints),
        "R-aff": (RAFF(host), True, request.checkpoints),
        "R-all": (RALL(host), True, RALL_CHECKPOINTS),
    }

    def arm(name: str) -> None:
        subset, images, checkpoints = plans[name]
        evaluate = from_images if images else cached_site
        fit = retune_grid(
            host,
            subset,
            data=data,
            fit_indices=fit_ids,
            selection_indices=data.selection,
            seed=request.seed,
            rates=RETUNE_RATES,
            checkpoints=checkpoints,
            from_images=images,
            certify=name != "R-all",
        )
        record = _retune_record(fit, subset, host, evaluate)
        record["checkpoints"] = _ints(checkpoints)
        chosen = fit.best
        if name != "R-all":
            features = evaluate(chosen.host)
            collect.put(name, losses(chosen.host, features, None), accuracy(chosen.host, features, None), record)
            return
        pre = chosen
        chosen = recalibrate_after(pre, data, fit_ids, data.selection)
        pre_features = evaluate(pre.host)
        pre_loss = losses(pre.host, pre_features, None)
        pre_correct = accuracy(pre.host, pre_features, None)
        record.update(
            {
                "selection_loss": chosen.selection,
                "pre_recalibration_selection_loss": pre.selection,
                "pre_recalibration_state_sha256": state_hash(pre.host),
                "pre_recalibration_development_loss": _mean(pre_loss),
                "pre_recalibration_development_accuracy": _mean(pre_correct),
                "recalibrated_state_sha256": state_hash(chosen.host),
            }
        )
        features = evaluate(chosen.host)
        collect.put(name, losses(chosen.host, features, None), accuracy(chosen.host, features, None), record)
        collect.vectors[name]["pre_recalibration_loss"] = pre_loss.cpu()
        collect.vectors[name]["pre_recalibration_correct"] = pre_correct.cpu()

    for name in arms:
        collect.attempt((name,), functools.partial(arm, name))
    torch.save(collect.vectors, request.output / VECTORS)
    return {
        **_identity(request),
        **_schedule(request),
        "noop_sha256": text(noop["noop_sha256"]),
        "arms_run": list(arms),
        "arms": collect.records,
        "unavailable_arms": _texts(collect.unavailable),
        "fit_examples": len(fit_ids),
        "vectors_sha256": file_sha(request.output / VECTORS),
    }


# --- timing and acceptance ---------------------------------------------------------------------------------------


def timing_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """One module grid and the first front-re-running arm (R-aff), one-per-card kernels for costs.Workloads.

    Uses N0 when --noop-root is given, else the raw trained host (stage 1 runs before N0 exists); no selection or
    development value is published. A grid with no finite candidate raises (choose), so completion means finite.
    """
    if request.noop_root is not None:
        host, _ = _noop_host(request, device)
        kind = "N0"
    else:
        host, _ = _request_training(request, device, frozen=True)
        kind = "raw"
    cache = cached(host, data)
    fit_ids = seen_fit(data)
    fit = features_from_images(host, data, fit_ids)
    _sync(device)
    started = time.monotonic()
    site = Site(host, fit, cache.selection, cache.development, request.seed, request.checkpoints)
    site.grid()
    _sync(device)
    module_seconds = time.monotonic() - started
    started = time.monotonic()
    retune_grid(
        host,
        RAFF(host),
        data=data,
        fit_indices=fit_ids,
        selection_indices=data.selection,
        seed=request.seed,
        rates=RETUNE_RATES,
        checkpoints=request.checkpoints,
        from_images=True,
    )
    _sync(device)
    front_seconds = time.monotonic() - started
    steps = len(RETUNE_RATES) * request.checkpoints[-1]
    # One development evaluation of a new host (R23): its features from images, then the shared evaluator.
    started = time.monotonic()
    _ = mean_loss(host, features_from_images(host, data, data.development), None)
    _sync(device)
    return {
        **_identity(request),
        "host": kind,
        "module_grid_seconds": module_seconds,
        **_schedule(request),
        "module_grid_steps": len(MODULE_RATES) * request.checkpoints[-1],
        "front_rerun_grid_seconds": front_seconds,
        "front_rerun_steps": steps,
        "front_rerun_seconds_per_1000_steps": front_seconds / (steps / 1000),
        "development_evaluation_seconds": time.monotonic() - started,
    }


def acceptance_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """The §9 stage-1 GPU acceptance matrix (rulings R10, R13), one child per card; four booleans and their evidence.

    (i) resume versus uninterrupted, (ii) checkpointed grid versus separate runs, (iii) static step zero versus the
    plain host, (iv) the padded evaluator's losses against its logits and its accuracy against an unpadded forward. The M-aff fold check is the separate `fold-check` child,
    which compares against the modules receipt's r_aff on the same N0.
    """
    seed, widths, output = request.seed, request.widths, request.output
    if request.training_root is not None:
        host, source = _request_training(request, device, frozen=True)
        if source["schedule"] != "constant-0.05":
            raise PilotHold("acceptance (i) resumes a constant-0.05 host; the training receipt has another schedule")
    # (i) resume versus uninterrupted under constant(0.05).
    straight = make_host(seed, 0, device, widths=widths)
    straight_hashes, _, _ = train_epochs(
        straight,
        straight,
        data,
        seed,
        epochs=range(12),
        rates=constant(0.05),
        optimizer=recipe_optimizer(straight.parameters()),
    )
    if request.training_root is not None:
        resumed, _ = _request_training(request, device, frozen=False)
        momentum = request.training_root / "optimizer.pt"
    else:
        resumed = make_host(seed, 0, device, widths=widths)
        optimizer = recipe_optimizer(resumed.parameters())
        train_epochs(resumed, resumed, data, seed, epochs=range(10), rates=constant(0.05), optimizer=optimizer)
        torch.save(dict(resumed.state_dict()), output / "acceptance-host.pt")
        momentum = output / "acceptance-optimizer.pt"
        torch.save(momentum_buffers(optimizer, resumed), momentum)
        host = load_host(output / "acceptance-host.pt", seed, 0, device, widths=widths, frozen=True)
    prefix = state_hash(resumed)
    resumed_hashes, _, _ = train_epochs(
        resumed,
        resumed,
        data,
        seed,
        epochs=range(10, 12),
        rates=constant(0.05),
        optimizer=_optimizer(resumed, momentum, device),
    )
    resume_bitwise = straight_hashes[9] == prefix and straight_hashes[10:] == resumed_hashes
    # (ii) one checkpointed grid against separate runs of each checkpoint length at the selected rate.
    cache = cached(host, data)
    fit = features_from_images(host, data, seen_fit(data))
    site = Site(host, fit, cache.selection, cache.development, seed, request.checkpoints)
    module = site.grid()
    rate = module.best.rate
    checkpoint_bitwise = True
    for steps in request.checkpoints:
        separate = fit_sgd(host, fit, cache.selection, seed=seed, rate=rate, steps=steps, minibatch=True)
        snapshot = next(c for c in module.candidates if c.rate == rate and c.steps == steps)
        same = all(torch.equal(a, b) for a, b in zip(snapshot.adapter.weights, separate.adapter.weights, strict=True))
        checkpoint_bitwise = checkpoint_bitwise and same and snapshot.selection == separate.selection
    # (iii) the complete static model at step zero forwards bitwise as the plain host.
    plain = make_host(seed, 0, device, widths=widths)
    static = StaticHost(make_host(seed, 0, device, widths=widths))
    plain.eval()
    static.eval()
    with torch.no_grad():
        images = normalized(data.images[data.train[:8].to(device)])
        static_step0_bitwise = torch.equal(as_tensor(static(images)), as_tensor(plain(images)))
    # (iv) the shared padded evaluator's losses agree with its logits, and its accuracy with an independent unpadded
    # forward of the development site (F6: never the same logits call on both sides; R34b: in EVAL_CHUNK batches).
    development = cache.development
    raw = logits(host, development, None)
    reference = F.cross_entropy(raw, development.labels, reduction="none")
    loss_difference = float((losses(host, development, None) - reference).abs().max())
    evaluation_consistent = loss_difference <= ACCEPTANCE_TOLERANCE and torch.equal(
        accuracy(host, development, None), chunked_accuracy(host, development)
    )
    _sync(device)
    return {
        **_identity(request),
        "training_root_used": request.training_root is not None,
        "resume_bitwise": resume_bitwise,
        "straight_epoch_state_sha256": _texts(straight_hashes),
        "resumed_prefix_state_sha256": prefix,
        "resumed_epoch_state_sha256": _texts(resumed_hashes),
        "checkpoint_bitwise": checkpoint_bitwise,
        "checkpoint_rate": rate,
        **_schedule(request),
        "static_step0_bitwise": static_step0_bitwise,
        "evaluation_consistent": evaluation_consistent,
        "loss_max_difference": loss_difference,
    }


def chunked_accuracy(host: Host, data: Features) -> Tensor:
    """R34b/R34c: 0/1 correctness from host.classify on the site in EVAL_CHUNK-row batches, each zero-padded to exactly
    EVAL_CHUNK rows (in practice only the last) with the padding rows discarded before the argmax, so the reference
    sees the padded evaluator's batch shape on card; independent of it by code path (classify, not logits)."""
    chunks: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(data.labels), EVAL_CHUNK):
            site = data.base[start : start + EVAL_CHUNK]
            padding = site.new_zeros((EVAL_CHUNK - len(site), *site.shape[1:]))
            chunks.append(host.classify(torch.cat((site, padding)))[: len(site)])
    return (torch.cat(chunks).argmax(1) == data.labels).float()


def fold_check_action(request: Request, data: Dataset, device: torch.device) -> Record:
    """R13: re-run the modules receipt's selected M-aff candidate on the same N0, fold it, recompute r_aff.

    One fit_sgd run at the receipt's selected (rate, steps) on the seen fit features: the grid's seed-derived
    stream makes it bitwise the grid's candidate (acceptance (ii)). Never re-grids. A residual that differs from
    the receipt by more than ACCEPTANCE_TOLERANCE is reported, not raised; the pilot decides.
    """
    host, _ = _noop_host(request, device)
    root = _required(request.modules_root, "fold-check (the modules child's output, --modules-root)")
    receipt = _request_receipt(request, root, frozenset(("modules",)))
    _bound(root, receipt, {VECTORS: "vectors_sha256"})
    if "noop_sha256" not in receipt or receipt["noop_sha256"] != state_hash(host):
        raise PilotHold("the modules receipt was fitted on another N0 than the loaded noop.pt")
    arms = mapping(receipt["arms"]) if "arms" in receipt else {}
    if "M-aff" not in arms:
        raise PilotHold("the modules receipt holds no M-aff arm")
    arm = mapping(arms["M-aff"])
    for key in ("rate", "steps", "r_aff"):
        if "status" not in arm or arm["status"] != "available" or key not in arm or arm[key] is None:
            raise PilotHold(f"the modules receipt's M-aff arm is unavailable or records no {key}")
    if "r_aff" not in receipt or receipt["r_aff"] is None:
        raise PilotHold("the modules receipt records no M-aff r_aff")
    rate, steps, receipt_r_aff = number(arm["rate"]), integer(arm["steps"]), number(receipt["r_aff"])
    cache = cached(host, data)
    fit = features_from_images(host, data, seen_fit(data))
    run = fit_sgd(
        host, fit, cache.selection, seed=request.seed, rate=rate, steps=steps, minibatch=True, nonlinear=False
    )
    if run.selection is None:
        raise PilotHold("the re-run M-aff candidate is nonfinite; the receipt selected a finite one")
    recomputed = fold_residual(
        host, run.adapter, fold_affine(host, run.adapter), cache.development, data, data.development
    )
    _sync(device)
    return {
        **_identity(request),
        "selected_rate": rate,
        "selected_steps": steps,
        "receipt_r_aff": receipt_r_aff,
        "recomputed_r_aff": recomputed,
        "fold_residual_matches_receipt": abs(recomputed - receipt_r_aff) <= ACCEPTANCE_TOLERANCE,
        "modules_vectors_sha256": text(receipt["vectors_sha256"]),
    }


# --- process entry -----------------------------------------------------------------------------------------------

Action = Callable[[Request, Dataset, torch.device], Record]
ACTIONS: Final[dict[str, Action]] = {
    "train": train_action,
    "replay": replay_action,
    "tcool": tcool_action,
    "cooled20": train_action,
    "static": static_action,
    "instrument": instrument_action,
    "noop": noop_action,
    "modules": modules_action,
    "retune": retune_action,
    "timing": timing_action,
    "acceptance": acceptance_action,
    "fold-check": fold_check_action,
}


def worker(request: Request) -> int:
    """One child: seal, pinned device, fresh output, M0 identity; publish result.json or failure.json."""
    seal = install_seal()
    device = configure(cuda=True)
    request.output.mkdir(parents=True, exist_ok=False)
    identity = provenance()
    request = dataclasses.replace(request, identity=identity)
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        data = load_dataset(request.data_root, DEV_ARTIFACT, device)
        payload = ACTIONS[request.action](request, data, device)
        _sync(device)
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        publish(
            request.output / "result.json",
            envelope(request.action, identity, payload, seal.report(), started, peak),
        )
        return 0
    except Exception as error:
        instrument_failure = isinstance(error, (InstrumentHold, OracleUnavailable, FitUnavailable))
        publish(
            request.output / "failure.json",
            {
                "action": request.action,
                "software_valid": instrument_failure,
                "instrument_valid": False if instrument_failure else None,
                "identity": identity,
                "error_type": type(error).__name__,
                "error": str(error),
                "unavailable_candidates": json_value(list(error.candidates))
                if isinstance(error, OracleUnavailable) and error.candidates is not None
                else None,
                "data_opens": seal.report(),
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise


def _checkpoints(value: str) -> tuple[int, ...]:
    """A schedule axis from the CLI: comma-separated positive integers, strictly increasing (04-schedule.json)."""
    try:
        steps = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"checkpoints must be comma-separated integers: {value}") from error
    if not steps or steps[0] <= 0 or any(a >= b for a, b in itertools.pairwise(steps)):
        raise argparse.ArgumentTypeError(f"checkpoints must be positive and strictly increasing: {value}")
    return steps


# argparse `choices` admits only these keys, so the lookup cannot miss.
LINEAGES: Final[dict[str, Lineage]] = {lineage: lineage for lineage in BLOCKS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=tuple(ACTIONS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, default=0, help="bottleneck width; always 0 (Supervisor compatibility)")
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--magnitudes", type=Path, help="accepted and ignored (Supervisor compatibility)")
    parser.add_argument("--lineage", choices=tuple(BLOCKS), required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--arms", help="comma-separated arm names; default: every arm of the action")
    parser.add_argument("--noop-root", type=Path)
    parser.add_argument("--tcool-root", type=Path)
    parser.add_argument("--static-root", type=Path)
    parser.add_argument(
        "--modules-root", type=Path, help="fold-check and the extended lottery: the core modules child's output"
    )
    parser.add_argument("--epochs", type=int, default=10, help="training actions: 20 for cooled-20, else 10")
    parser.add_argument("--checkpoints", type=_checkpoints, default=CHECKPOINTS, help="module and re-tune axis")
    parser.add_argument("--checkpoints-b", type=_checkpoints, default=CHECKPOINTS_B, help="M-small-B axis")
    args = parser.parse_args(argv)
    lineage = LINEAGES[args.lineage]
    if args.width != 0:
        parser.error("M0 hosts have no bottleneck; --width must be 0")
    if args.scale != width_scale(lineage):
        parser.error(f"--scale {args.scale} is not the {lineage} width scale {width_scale(lineage)}")
    try:
        seed = admit(args.seed, lineage)
    except PilotHold as error:
        parser.error(str(error))
    epochs = 20 if lineage == "cooled-20" else 10
    if args.action in TRAINING_ACTIONS and args.epochs != epochs:
        parser.error(f"lineage {lineage} trains exactly {epochs} epochs")
    if args.action == "cooled20" and lineage != "cooled-20":
        parser.error("cooled20 trains the cooled-20 lineage only")
    arms = None if args.arms is None else tuple(name.strip() for name in args.arms.split(","))
    return worker(
        Request(
            action=args.action,
            output=args.output,
            data_root=args.data_root,
            seed=seed,
            lineage=lineage,
            widths=FRONTIER_WIDTHS[width_scale(lineage)],
            epochs=args.epochs,
            checkpoints=args.checkpoints,
            checkpoints_b=args.checkpoints_b,
            arms=arms,
            training_root=args.training_root,
            noop_root=args.noop_root,
            tcool_root=args.tcool_root,
            static_root=args.static_root,
            modules_root=args.modules_root,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
