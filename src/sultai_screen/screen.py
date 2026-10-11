"""Sealed development pilot workers and source provenance. No confirmatory execution entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import subprocess
import time
from pathlib import Path

import torch
from torch import Tensor

from sultai.trust import FitUnavailable

from .alignment import align, moments, transport
from .contracts import (
    PREREG_SHA,
    RHO,
    InstrumentHold,
    Json,
    OracleUnavailable,
    PilotHold,
    Record,
    encode,
    file_sha,
    integer,
    mapping,
    number,
    publish,
    read_record,
    text,
)
from .data import Dataset, install_seal, load_dataset
from .fit import cheap_bank, fit_sgd, identity_check, mean_loss, oracle, ridge
from .former_bridge import derive_former, former_fits
from .host import Cached, Features, Host, cached, configure, flags, freeze, load_host, make_host, state_hash, train_host
from .impair import make_impairment

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "docs/plans/2026-10-10-gate3-headroom-screen.md"
DEV_ARTIFACT = REPO / "docs/results/gate3-split-2026-10-11/dev-indices-int32-le.bin"
INTACT_SEEDS = tuple(range(3000, 3006))
BOTTLENECK_SEEDS = INTACT_SEEDS


def source_identity(*, expected_prereg: str = PREREG_SHA) -> Record:
    if file_sha(PROTOCOL) != expected_prereg:
        raise PilotHold("committed preregistration identity mismatch")
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
    return {
        "commit": commit,
        "source_sha256": aggregate,
        "source_files": hashes,
        "preregistration_sha256": expected_prereg,
    }


def provenance(*, expected_prereg: str = PREREG_SHA) -> Record:
    result = source_identity(expected_prereg=expected_prereg)
    if importlib.metadata.version("torch") != "2.9.1+cu128" or importlib.metadata.version("numpy") != "2.3.5":
        raise PilotHold("installed screen wheels differ from the pinned environment")
    result.update(
        {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "torch_wheel": importlib.metadata.version("torch"),
            "numpy_wheel": importlib.metadata.version("numpy"),
            "determinism": flags(),
            "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        }
    )
    return result


def tensor_sha(values: Tensor) -> str:
    return hashlib.sha256(values.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _training(root: Path, seed: int, width: int, device: torch.device) -> Host:
    record = read_record(root / "result.json")
    payload = mapping(record["payload"])
    original_identity = mapping(record["identity"])
    current_identity = provenance()
    if (
        original_identity["source_sha256"] != current_identity["source_sha256"]
        or original_identity["preregistration_sha256"] != current_identity["preregistration_sha256"]
    ):
        raise PilotHold("training receipt belongs to another source/protocol")
    if integer(payload["seed"]) != seed or integer(payload["width"]) != width:
        raise PilotHold("host checkpoint belongs to another anchor")
    checkpoint = root / "host.pt"
    if file_sha(checkpoint) != text(payload["checkpoint_sha256"]):
        raise PilotHold("host checkpoint changed after publication")
    if file_sha(root / "loss-stream.bin") != text(payload["loss_stream_sha256"]):
        raise PilotHold("host loss stream changed after publication")
    host = load_host(checkpoint, seed, width, device)
    if state_hash(host) != text(payload["final_state_sha256"]):
        raise PilotHold("loaded host state differs from training receipt")
    return host


def training_worker(data: Dataset, seed: int, width: int, output: Path) -> Record:
    host = make_host(seed, width, data.images.device)
    # A separate fresh host probes the 2-image forward/backward without changing training state.
    probe = make_host(seed, width, data.images.device)
    inputs = data.images[data.train[:2].to(data.images.device)].float() / 255
    logits = probe(inputs)
    gradients = torch.autograd.grad(logits.sum(), tuple(probe.parameters()))
    forward_backward: Record = {
        "output_sha256": tensor_sha(logits),
        "gradient_sha256": hashlib.sha256("".join(tensor_sha(g) for g in gradients).encode()).hexdigest(),
    }
    del probe, gradients, logits
    hashes, stream, elapsed = train_host(host, data, seed)
    torch.save(dict(host.state_dict()), output / "host.pt")
    with (output / "loss-stream.bin").open("xb") as target:
        target.write(stream.numpy().tobytes())
    freeze(host)
    return {
        "seed": seed,
        "width": width,
        "epochs": 10,
        "steps": len(stream),
        "epoch_state_sha256": list(hashes),
        "loss_stream_sha256": tensor_sha(stream),
        "checkpoint_sha256": file_sha(output / "host.pt"),
        "final_state_sha256": state_hash(host),
        "training_seconds": elapsed,
        "parameters": sum(p.numel() for p in host.parameters()),
        "forward_backward": forward_backward,
    }


def _site(cache: Cached, bypass: bool) -> Cached:
    def part(data: Features) -> Features:
        return data if bypass else Features(data.base, data.base, data.labels)

    return Cached(part(cache.conditioning), part(cache.selection), part(cache.development))


def timing_worker(host: Host, cache: Cached, seed: int, output: Path) -> Record:
    trials = [
        fit_sgd(host, cache.conditioning, cache.selection, seed=seed, rate=0.1, steps=steps, minibatch=True)
        for steps in (200, 800)
    ]
    full = fit_sgd(host, cache.conditioning, cache.selection, seed=seed, rate=0.1, steps=1, minibatch=False)
    # B5 and B4 are timing probes only here: no development losses, gains or
    # retrieval decision are opened. They instantiate the frozen diagnostic kernels.
    wide = tuple(
        fit_sgd(host, cache.conditioning, cache.selection, seed=seed, rate=rate, steps=200, minibatch=True, wide=True)
        for rate in (0.1, 0.03)
    )
    torch.cuda.synchronize()
    started = time.monotonic()
    read, base = cache.conditioning.read[:1024], cache.conditioning.base[:1024]
    order = torch.roll(torch.arange(16, device=read.device), 1)
    p_in, p_out = align(read, read[:, order]), align(base, base[:, order])
    aligned = torch.einsum("ij,bjhw->bihw", p_in, read[:, order])
    distance = (moments(read) - moments(aligned)).square().sum()
    moved = transport(trials[1].adapter, p_in, p_out)
    # Force completion of all timing kernels; these values have no outcome role.
    _ = float(distance) + sum(float(w.sum()) for w in moved.weights)
    torch.cuda.synchronize()
    retrieval_seconds = time.monotonic() - started
    started = time.monotonic()
    _ = mean_loss(host, cache.development, None)
    torch.cuda.synchronize()
    evaluation_seconds = time.monotonic() - started
    # Timing stage exposes no selection/gain values; hashes supply a B3-fit replay check.
    best = trials[1]
    torch.save({"weight": best.adapter.weights[0], "bias": best.adapter.weights[1]}, output / "timing-adapter.pt")
    return {
        "fit_1x_seconds": trials[0].wall_seconds,
        "fit_4x_seconds": best.wall_seconds,
        "full_batch_step_seconds": full.wall_seconds,
        "fit_finite": all(c.selection is not None for c in (*trials, full)),
        "wide_grid_seconds": sum(c.wall_seconds for c in wide),
        "wide_grid_finite": any(c.selection is not None for c in wide),
        "retrieval_pair_seconds": retrieval_seconds,
        "development_evaluation_seconds": evaluation_seconds,
        "fit_state_sha256": [tensor_sha(w) for w in best.adapter.weights],
        "fit_trace_sha256": tensor_sha(torch.tensor(best.training_losses, dtype=torch.float64)),
        "full_batch_chunk_examples": 128,
    }


def materiality_worker(host: Host, cache: Cached, seed: int) -> Record:
    identity_check(host, cache.development, seed)
    original = mean_loss(host, cache.development, None)
    result: Record = {"seed": seed, "intact_loss": original, "rungs": {}}
    rungs: Record = {}
    for family, magnitudes in (("P1", (0.1, 0.3, 0.03)), ("P2", (0.1, 0.03)), ("P3", (0.25, 0.5, 1.0))):
        gains: Record = {}
        for magnitude in magnitudes:
            operator = make_impairment(family, magnitude, seed, cache.conditioning.read)
            data = operator.features(cache.development)
            no_op = mean_loss(host, data, None)
            restored = Features(data.read, operator.undo(data.base), data.labels)
            inverse_loss = mean_loss(host, restored, None)
            if abs(inverse_loss - original) > 1e-4:
                raise PilotHold(f"exact inverse failed 1e-4 nats: {family}/{seed}")
            gains[str(magnitude)] = no_op - inverse_loss
        rungs[family] = gains
    result["rungs"] = rungs
    return result


def posthoc_worker(
    host: Host, cache: Cached, seed: int, alpha: tuple[float, float], s_min: float, kappa: float, output: Path
) -> Record:
    former = derive_former()
    result: Record = {"seed": seed, "former": former.identity(), "families": {}}
    families: Record = {}
    intact_loss = mean_loss(host, cache.development, None)
    for family, magnitudes in (("P1", alpha), ("P2", (s_min,)), ("P3", (kappa,)), ("P3prime", (kappa,))):
        cases: list[Json] = []
        for magnitude in magnitudes:
            operator = make_impairment(family, magnitude, seed, cache.conditioning.read)
            conditioning = operator.features(cache.conditioning)
            selection = operator.features(cache.selection)
            development = operator.features(cache.development)
            a1 = mean_loss(host, Features(development.read, operator.undo(development.base), development.labels), None)
            if abs(a1 - intact_loss) > 1e-4:
                raise PilotHold("exact inverse endpoint drift")
            a2 = ridge(conditioning.read, conditioning.base, cache.conditioning.read, nonlinear=True)
            a3 = ridge(conditioning.read, conditioning.base, cache.conditioning.read, nonlinear=False)
            a7, a8 = former_fits(former, conditioning.read, cache.conditioning.read, conditioning.labels, seed)
            noop = mean_loss(host, development, None)
            arms: Record = {
                "A1": a1,
                "A2": mean_loss(host, development, a2),
                "A3": mean_loss(host, development, a3),
                "A7": mean_loss(host, development, a7),
                "A8": mean_loss(host, development, a8),
            }
            case: Record = {
                "magnitude": magnitude,
                "noop_loss": noop,
                "arm_losses": arms,
                "ridge_position_pairs": len(conditioning.labels) * 256,
                "former_position_pairs": 4096,
            }
            if family == "P1" and magnitude == alpha[0]:
                fit = oracle(host, conditioning, selection, seed)
                b3 = mean_loss(host, development, fit.best.adapter)
                arms["B3"] = b3
                case["oracle"] = fit.report()
                case["oracle_gain_fraction_of_A3"] = (
                    (noop - b3) / (noop - number(arms["A3"])) if noop > number(arms["A3"]) else None
                )
                torch.save(
                    {"weight": fit.best.adapter.weights[0], "bias": fit.best.adapter.weights[1]},
                    output / "p1-oracle.pt",
                )
            cases.append(case)
        families[family] = cases
    result["families"] = families
    result["rho_required"] = RHO
    return result


def bottleneck_worker(host: Host, cache: Cached, seed: int, output: Path) -> Record:
    result: Record = {"seed": seed, "width": host.width}
    for family, bypass in (("P5", False), ("P6", True)):
        features = _site(cache, bypass)
        fit = oracle(host, features.conditioning, features.selection, seed)
        noop = mean_loss(host, features.development, None)
        candidate_loss = mean_loss(host, features.development, fit.best.adapter)
        identity_check(host, features.development, seed)
        result[family] = {
            "noop_loss": noop,
            "oracle_loss": candidate_loss,
            "gain": noop - candidate_loss,
            "oracle": fit.report(),
        }
        torch.save(
            {"weight": fit.best.adapter.weights[0], "bias": fit.best.adapter.weights[1]}, output / f"{family}-oracle.pt"
        )
    result["paired_gain_difference"] = number(mapping(result["P6"])["gain"]) - number(mapping(result["P5"])["gain"])
    return result


def cheap_worker(host: Host, cache: Cached, seed: int) -> Record:
    result: Record = {"seed": seed}
    for family, bypass in (("P5", False), ("P6", True)):
        features = _site(cache, bypass)
        candidates = cheap_bank(host, features.conditioning, features.selection, seed)
        bank: Record = {}
        for candidate in candidates:
            # All candidates, including failed ones, remain visible and charged.
            bank[str(candidate.rate)] = {
                "candidate": candidate.report(),
                "development_loss": mean_loss(host, features.development, candidate.adapter)
                if candidate.selection is not None
                else None,
            }
        result[family] = bank
    return result


def worker(
    action: str,
    output: Path,
    data_root: Path,
    seed: int,
    width: int,
    training_root: Path | None,
    magnitudes: Path | None,
) -> int:
    seal = install_seal()
    device = configure(cuda=True)
    output.mkdir(parents=True, exist_ok=False)
    identity = provenance()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        data = load_dataset(data_root, DEV_ARTIFACT, device)
        if action == "train":
            payload = training_worker(data, seed, width, output)
        else:
            if training_root is None:
                raise PilotHold("fitting requires a published training receipt")
            host = _training(training_root, seed, width, device)
            cache = cached(host, data)
            if action == "timing":
                payload = timing_worker(host, cache, seed, output)
            elif action == "materiality":
                payload = materiality_worker(host, cache, seed)
            elif action == "posthoc":
                if magnitudes is None:
                    raise PilotHold("posthoc requires published magnitude decision")
                chosen = read_record(magnitudes)
                payload = posthoc_worker(
                    host,
                    cache,
                    seed,
                    (number(chosen["alpha_low"]), number(chosen["alpha_high"])),
                    number(chosen["s_min"]),
                    number(chosen["kappa"]),
                    output,
                )
            elif action == "bottleneck":
                payload = bottleneck_worker(host, cache, seed, output)
            elif action == "cheap":
                payload = cheap_worker(host, cache, seed)
            else:
                raise PilotHold("unknown worker action")
        torch.cuda.synchronize()
        publish(
            output / "result.json",
            {
                "action": action,
                "software_valid": True,
                "identity": identity,
                "payload": payload,
                "data_opens": seal.report(),
                "wall_seconds": time.monotonic() - started,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            },
        )
        return 0
    except Exception as error:
        instrument_failure = isinstance(error, (InstrumentHold, OracleUnavailable, FitUnavailable))
        publish(
            output / "failure.json",
            {
                "action": action,
                "software_valid": instrument_failure,
                "instrument_valid": False if instrument_failure else None,
                "identity": identity,
                "error_type": type(error).__name__,
                "error": str(error),
                "data_opens": seal.report(),
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "timing", "materiality", "posthoc", "bottleneck", "cheap"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--magnitudes", type=Path)
    args = parser.parse_args()
    if args.seed not in (*INTACT_SEEDS, *BOTTLENECK_SEEDS):
        parser.error("pilot development anchor outside the fixed six-plus-six set")
    return worker(args.action, args.output, args.data_root, args.seed, args.width, args.training_root, args.magnitudes)


if __name__ == "__main__":
    raise SystemExit(main())
