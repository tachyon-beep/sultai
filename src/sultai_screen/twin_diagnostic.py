"""One separately authorized no-op twin diagnostic; never a pilot or training entry point."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

import torch
from torch import Tensor
from torch.nn import functional as F

from sultai.trust import ContractViolation

from .contracts import (
    PILOT_SECONDS,
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
from .data import as_tensor, generator, install_seal, load_dataset, normalized
from .host import Features, Host, assert_frozen, configure, flags, load_host, state_hash
from .pilot import PRIOR_RECEIPTS, _signal_owned, check_certificate, check_handover, check_output_root, claim_attempt
from .screen import DEV_ARTIFACT, REPO, environment_identity, source_identity, tensor_sha

MAX_SECONDS = 120.0
OPENING_CHARGE = 1469.3526015179232
RUN = "runs/gate3-pilot-2026-10-11-01a125af-retry1"
TRAINING = f"{RUN}/intact-3000"
CUSTODY = ".weft/gate3-twin-diagnostic-attempt.json"
AUTHORIZATION = "docs/certificates/2026-10-12-gate3-twin-diagnostic-authorization.md"
RECEIPTS = (
    *PRIOR_RECEIPTS,
    (".weft/gate3-pilot-linked-retry-1.json", "d3726c4293655faf7c5630a12a235368a47bde6d8eccafa2d96780ddcdf01888"),
    (f"{RUN}/pilot-result.json", "943ace1533cde6950d0e57cc1ef64fd5ec4f318826a89b5f1d7cfce026cd3931"),
    (f"{RUN}/ledger-015.json", "42ee5feedcda3c7841064b0bd65c049cdcf84ed0432e480cf2ae6d89bef73ce1"),
    (f"{TRAINING}/result.json", "4b4f6f944a629bf2c32c14ec2c1c65aeecd06de47751dfb62f8c4bbd097b5890"),
)


@dataclass(frozen=True)
class Evaluation:
    logits: Tensor
    loss: Tensor
    batch_sizes: Tensor


def permutation_orders(count: int, seed: int, device: torch.device) -> dict[str, Tensor]:
    if count <= 128:
        raise ContractViolation("diagnostic requires a full batch and a final partial batch")
    tail = count % 128
    if tail == 0:
        raise ContractViolation("diagnostic requires a final partial batch")
    order = torch.randperm(count, generator=generator(seed, "noop-evaluation-twin"))
    full_count = count - tail
    preserved = torch.cat(
        (
            torch.randperm(full_count, generator=generator(seed, "diagnostic-full-batches")),
            full_count + torch.randperm(tail, generator=generator(seed, "diagnostic-tail")),
        )
    )
    return {
        "identity": torch.arange(count, device=device),
        "global": order.to(device),
        "preserve_batch_size": preserved.to(device),
    }


def evaluate(host: Host, data: Features, order: Tensor, *, padded: bool) -> Evaluation:
    count = len(data.labels)
    if (
        order.dtype != torch.long
        or order.shape != (count,)
        or not torch.equal(order.sort().values, torch.arange(count, device=order.device))
    ):
        raise ContractViolation("complete unique evaluation permutation required")
    expected = state_hash(host)
    assert_frozen(host, expected)
    identity_order = torch.equal(order, torch.arange(count, device=order.device))
    base = data.base if identity_order else data.base[order]
    labels = data.labels if identity_order else data.labels[order]
    logits, losses, sizes = [], [], []
    with torch.no_grad():
        for start in range(0, count, 128):
            values, targets = base[start : start + 128], labels[start : start + 128]
            real = len(targets)
            if padded and real < 128:
                values = torch.cat((values, values[-1:].expand(128 - real, *values.shape[1:])))
                targets = torch.cat((targets, targets[-1:].expand(128 - real)))
            predicted = host.classify(values)
            loss = F.cross_entropy(predicted, targets, reduction="none")
            logits.append(predicted[:real])
            losses.append(loss[:real])
            sizes.append(torch.full((real,), len(targets), device=order.device, dtype=torch.long))
    inverse = torch.argsort(order)
    result = Evaluation(torch.cat(logits)[inverse], torch.cat(losses)[inverse], torch.cat(sizes)[inverse])
    if not bool(torch.isfinite(result.logits).all()) or not bool(torch.isfinite(result.loss).all()):
        raise PilotHold("nonfinite diagnostic logits/loss; retain as execution failure")
    assert_frozen(host, expected)
    return result


def differences(baseline: Tensor, actual: Tensor) -> Record:
    if baseline.shape != actual.shape or baseline.ndim not in (1, 2):
        raise ContractViolation("matched per-example scalars or class logits required")
    if not bool(torch.isfinite(baseline).all()) or not bool(torch.isfinite(actual).all()):
        raise PilotHold("nonfinite comparison inputs")
    unequal = baseline != actual
    affected = unequal if unequal.ndim == 1 else unequal.any(dim=1)
    indices = torch.nonzero(affected).flatten()
    return {
        "bitwise_equal": torch.equal(baseline, actual),
        "mismatch_count": len(indices),
        "mismatch_indices": json_value(indices.cpu().tolist()),
        "max_absolute_difference": float((baseline.double() - actual.double()).abs().max()),
        "baseline_sha256": tensor_sha(baseline),
        "actual_sha256": tensor_sha(actual),
    }


def ce_in_order(logits: Tensor, labels: Tensor, order: Tensor, *, padded: bool) -> Tensor:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(labels), 128):
            ids = order[start : start + 128]
            real = len(ids)
            if padded and real < 128:
                ids = torch.cat((ids, ids[-1:].expand(128 - real)))
            chunks.append(F.cross_entropy(logits[ids], labels[ids], reduction="none")[:real])
    return torch.cat(chunks)[torch.argsort(order)]


def diagnose(host: Host, data: Features, output: Path) -> Record:
    orders = permutation_orders(len(data.labels), 3000, data.base.device)
    order = orders["global"]
    inverse = torch.argsort(order)
    round_trip: Record = {}
    for name, value in (("read", data.read), ("base", data.base), ("labels", data.labels)):
        restored = value[order][inverse]
        same = torch.equal(value, restored)
        round_trip[name] = {
            "bitwise_equal": same,
            "original_sha256": tensor_sha(value),
            "round_trip_sha256": tensor_sha(restored),
        }
        if not same:
            raise PilotHold("feature/label permutation round-trip failed")
    frozen = state_hash(host)
    baseline = evaluate(host, data, orders["identity"], padded=False)
    cases: Record = {}
    for name, case_order, padded in (
        ("identical_repeat", orders["identity"], False),
        ("original_global_permutation", order, False),
        ("preserve_batch_size", orders["preserve_batch_size"], False),
        ("padded_identity", orders["identity"], True),
        ("padded_global_permutation", order, True),
    ):
        actual = evaluate(host, data, case_order, padded=padded)
        report: Record = {
            "logits": differences(baseline.logits, actual.logits),
            "loss": differences(baseline.loss, actual.loss),
            "baseline_batch_sizes": json_value(baseline.batch_sizes.cpu().tolist()),
            "actual_batch_sizes": json_value(actual.batch_sizes.cpu().tolist()),
            "ce_of_baseline_logits_in_case_order": differences(
                baseline.loss, ce_in_order(baseline.logits, data.labels, case_order, padded=padded)
            ),
            "case_logits_ce_in_identity_order": differences(
                baseline.loss, ce_in_order(actual.logits, data.labels, orders["identity"], padded=False)
            ),
            "frozen_state_sha256": state_hash(host),
        }
        assert_frozen(host, frozen)
        publish(output / f"comparison-{name}.json", report)
        cases[name] = report
    padded_base = evaluate(host, data, orders["identity"], padded=True)
    padded_twin = evaluate(host, data, order, padded=True)
    padded_pair: Record = {
        "logits": differences(padded_base.logits, padded_twin.logits),
        "loss": differences(padded_base.loss, padded_twin.loss),
    }
    return {
        "round_trip": round_trip,
        "cases": cases,
        "padded_pair": padded_pair,
        "original_twin_bitwise_equal": mapping(mapping(cases["original_global_permutation"])["loss"])["bitwise_equal"],
        "frozen_state_sha256": frozen,
        "examples": len(data.labels),
        "diagnostic_only": True,
        "pilot_ready": False,
        "confirmation_ready": False,
    }


def verify_prior() -> Record:
    hashes: Record = {}
    for relative, expected in RECEIPTS:
        path = REPO / relative
        if not path.is_file() or file_sha(path) != expected:
            raise PilotHold(f"preserved diagnostic input missing or changed: {relative}")
        hashes[relative] = expected
    terminal = read_record(REPO / RUN / "pilot-result.json")
    if terminal["charged_gpu_process_seconds"] != OPENING_CHARGE or terminal["verdict"] != "execution-invalid":
        raise PilotHold("original cumulative failure charge required")
    training = read_record(REPO / TRAINING / "result.json")
    identity = mapping(training["identity"])
    if identity["source_sha256"] != "a9445331c6b92018fbe73316fe4b7eb3851fc157a4798185da51ce2fff17b595":
        raise PilotHold("checkpoint source identity mismatch")
    for relative, source_hash in mapping(identity["source_files"]).items():
        if file_sha(REPO / relative) != text(source_hash):
            raise PilotHold(f"prior production/test/config bytes changed: {relative}")
    trained = mapping(training["payload"])
    if integer(trained["seed"]) != 3000 or integer(trained["width"]) != 0:
        raise PilotHold("preserved intact-3000 host required")
    for name, key in (("host.pt", "checkpoint_sha256"), ("loss-stream.bin", "loss_stream_sha256")):
        if file_sha(REPO / TRAINING / name) != text(trained[key]):
            raise PilotHold(f"preserved training artifact changed: {name}")
    return hashes


def worker(output: Path) -> int:
    custody = read_record(REPO / CUSTODY)
    if integer(custody["pid"]) != os.getppid() or text(custody["output"]) != str(output):
        raise PilotHold("diagnostic worker requires its live authorized supervisor")
    identity = source_identity()
    if mapping(custody["identity"])["source_sha256"] != identity["source_sha256"]:
        raise PilotHold("worker source differs from admitted diagnostic")
    seal = install_seal()
    try:
        verify_prior()
        environment = environment_identity()
        device = configure(cuda=True)
        if os.environ["CUDA_VISIBLE_DEVICES"] != "0":
            raise PilotHold("diagnostic restricted to physical card 0")
        host = load_host(REPO / TRAINING / "host.pt", 3000, 0, device)
        trained = mapping(read_record(REPO / TRAINING / "result.json")["payload"])
        if state_hash(host) != trained["final_state_sha256"]:
            raise PilotHold("loaded frozen host identity mismatch")
        data = load_dataset(Path("/home/john/simic/runs/data/cifar-10-batches-py"), DEV_ARTIFACT, device)
        parts = []
        with torch.no_grad():
            for start in range(0, len(data.development), 128):
                ids = data.development[start : start + 128].to(device)
                parts.append(as_tensor(host.front(normalized(data.images[ids]))).detach())
        base = torch.cat(parts)
        site = Features(base, base, data.labels[data.development.to(device)])
        payload = diagnose(host, site, output)
        torch.cuda.synchronize()
        publish(
            output / "worker-result.json",
            {
                "software_valid": True,
                "identity": identity,
                "environment": environment,
                "determinism": flags(),
                "data_opens": seal.report(),
                "payload": payload,
            },
        )
        return 0
    except Exception as error:
        publish(
            output / "worker-failure.json",
            {
                "software_valid": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "identity": identity,
                "data_opens": seal.report(),
            },
        )
        raise


def run_child(output: Path) -> Record:
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONPATH=str(REPO / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    command = [
        "nice",
        "-n",
        "19",
        sys.executable,
        "-B",
        "-m",
        "sultai_screen.twin_diagnostic",
        "--worker",
        "--output",
        str(output),
    ]
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    started = time.monotonic()
    try:
        with (output / "worker.stdout").open("xb") as stdout, (output / "worker.stderr").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        publish(output / "process.json", {"pid": process.pid, "command": json_value(command), "gpu": 0})
        deadline = started + MAX_SECONDS - 5
        while process.poll() is None:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.02)
    finally:
        if process is not None:
            if process.poll() is None:
                _signal_owned(process, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    _signal_owned(process, signal.SIGKILL)
            process.wait()
            elapsed = time.monotonic() - started
            publish(
                output / "charge.json",
                {
                    "opening_charge_gpu_process_seconds": OPENING_CHARGE,
                    "diagnostic_charge_gpu_process_seconds": elapsed,
                    "total_charge_gpu_process_seconds": OPENING_CHARGE + elapsed,
                    "total_ceiling_gpu_process_seconds": PILOT_SECONDS,
                    "diagnostic_ceiling_seconds": MAX_SECONDS,
                    "child_returncode": process.returncode,
                    "timed_out": timed_out,
                },
            )
    if process is None:
        raise PilotHold("diagnostic process never started; no successful no-op")
    return read_record(output / "charge.json")


def supervise(output: Path, certificate: Path) -> int:
    check_output_root(output)
    if output.exists():
        raise PilotHold("diagnostic requires a new output path")
    approved = read_record(certificate)
    identity = check_certificate(approved)
    if approved["diagnostic_approved"] is not True or number(approved["diagnostic_max_seconds"]) != MAX_SECONDS:
        raise PilotHold("explicit bounded diagnostic certificate required")
    if (
        number(approved["opening_charge_gpu_process_seconds"]) != OPENING_CHARGE
        or OPENING_CHARGE + MAX_SECONDS > PILOT_SECONDS
    ):
        raise PilotHold("same cumulative ceiling and exact prior charge required")
    if not any(mapping(entry)["path"] == AUTHORIZATION for entry in sequence(approved["evidence"])):
        raise PilotHold("bound operator diagnostic authorization required")
    prior = verify_prior()
    environment = environment_identity()
    handover = check_handover()
    claim_attempt(REPO / CUSTODY, output, identity)
    output.mkdir(parents=True, exist_ok=False)
    publish(
        output / "admission.json",
        {
            "identity": identity,
            "prior_receipts": prior,
            "environment": environment,
            "handover": handover,
            "certificate_sha256": file_sha(certificate),
            "maximum_process_seconds": MAX_SECONDS,
        },
    )
    charge = run_child(output)
    complete = (
        charge["child_returncode"] == 0
        and charge["timed_out"] is False
        and number(charge["diagnostic_charge_gpu_process_seconds"]) <= MAX_SECONDS
    )
    if complete:
        complete = read_record(output / "worker-result.json")["software_valid"] is True
    publish(
        output / "diagnostic-result.json",
        {
            "diagnostic_complete": complete,
            "charge": charge,
            "pilot_ready": False,
            "confirmation_ready": False,
            "new_attempt_authorized": False,
        },
    )
    return 0 if complete else 2


def interrupted(signum: int, _frame: FrameType | None) -> None:
    raise PilotHold(f"diagnostic interrupted by signal {signum}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-certificate", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.output.resolve())
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.review_certificate is None:
        raise PilotHold("reviewed diagnostic certificate required")
    return supervise(args.output.resolve(), args.review_certificate.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
