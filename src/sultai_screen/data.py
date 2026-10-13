"""Sealed, hash-pinned CIFAR training-batch boundary and exact partition derivation."""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from sultai.trust import InputDataError, t3_boundary

from .contracts import DEV_SHA, PERM_SHA, PilotHold, Record, derive

HASHES = {
    "batches.meta": "f962466ef690d46b226450fb9aadc74ba4bc64a76aa526b5827fe4bc5c7125cb",
    "data_batch_1": "54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd",
    "data_batch_2": "766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8",
    "data_batch_3": "0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9",
    "data_batch_4": "3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b",
    "data_batch_5": "d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154",
}


@dataclass
class Seal:
    opens: dict[str, int] = field(default_factory=dict)

    @t3_boundary(
        test="tests/test_screen.py::ScreenTests.test_seal_and_split_negative_controls",
        fingerprint="95682bfadbf84165f80f41d604989e0bd5c578c3781669915f8420c92c7b17b9",
    )
    def hook(self, event: str, args: tuple[object, ...]) -> None:
        if event != "open" or not args:
            return
        raw = args[0]
        if not isinstance(raw, (str, bytes)):
            return
        path = os.fsdecode(raw)
        if "test_batch" in path:
            raise PilotHold("sealed CIFAR test batch requested")
        name = Path(path).name
        if name in HASHES:
            self.opens[name] = self.opens[name] + 1 if name in self.opens else 1

    def report(self) -> Record:
        result: Record = {}
        result.update(sorted(self.opens.items()))
        return result


def install_seal() -> Seal:
    seal = Seal()
    sys.addaudithook(seal.hook)
    return seal


@t3_boundary(
    test="tests/test_screen.py::ScreenTests.test_tensor_boundary",
    fingerprint="363b24c47d5d50a9ff013d1a630f491071889c662e5817e4188c5bb510ebb8e9",
)
def as_tensor(value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise InputDataError("torch Tensor required")
    return value


@t3_boundary(
    test="tests/test_screen.py::ScreenTests.test_pickle_boundary",
    fingerprint="8c31897b06de1da1a30fc41f2922e82a84f2a2b7b58708963578c2f64d2cc5b9",
)
def cifar_batch(value: object) -> tuple[Tensor, Tensor]:
    if not isinstance(value, dict) or b"data" not in value or b"labels" not in value:
        raise InputDataError("CIFAR data and labels required")
    data: object = value[b"data"]
    labels: object = value[b"labels"]
    if not isinstance(data, np.ndarray) or data.dtype != np.uint8 or data.shape != (10000, 3072):
        raise InputDataError("CIFAR batch must be uint8 [10000,3072]")
    if not isinstance(labels, list) or len(labels) != 10000:
        raise InputDataError("10000 CIFAR labels required")
    checked: list[int] = []
    for label in labels:
        if type(label) is not int or not 0 <= label < 10:
            raise InputDataError("CIFAR label must be integer in [0,10)")
        checked.append(label)
    return torch.from_numpy(data).reshape(10000, 3, 32, 32).contiguous(), torch.tensor(checked, dtype=torch.long)


@dataclass(frozen=True)
class Dataset:
    images: Tensor
    labels: Tensor
    train: Tensor
    conditioning: Tensor
    selection: Tensor
    development: Tensor


def generator(seed: int, label: str) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(derive(seed, label))


def partitions(artifact: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    raw = artifact.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DEV_SHA:
        raise PilotHold("development index artifact hash mismatch")
    perm = torch.randperm(50000, generator=generator(20261004, "cifar-fit-dev-v1"))
    if hashlib.sha256(perm.numpy().astype("<i8").tobytes()).hexdigest() != PERM_SHA:
        raise PilotHold("fit/dev permutation mismatch")
    dev = torch.from_numpy(np.frombuffer(raw, dtype="<i4").copy()).long()
    if not torch.equal(dev, perm[45000:]):
        raise PilotHold("development index artifact disagrees with derivation")
    fit = perm[:45000]
    order = torch.randperm(45000, generator=generator(20261004, "sultai-fit-partition-v1"))
    fit = fit[order]
    return fit[:40000], fit[40000:42000], fit[42000:], dev


def load_dataset(root: Path, artifact: Path, device: torch.device) -> Dataset:
    images, labels = [], []
    # Verify bytes before unpickling; never open the archive, test file, or an unpinned pickle.
    for name, expected in HASHES.items():
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise PilotHold(f"CIFAR source hash mismatch: {name}")
        if name == "batches.meta":
            continue
        unknown: object = pickle.loads(raw, encoding="bytes")
        batch, target = cifar_batch(unknown)
        images.append(batch)
        labels.append(target)
    train, conditioning, selection, development = partitions(artifact)
    return Dataset(
        torch.cat(images).to(device), torch.cat(labels).to(device), train, conditioning, selection, development
    )


def normalized(images: Tensor) -> Tensor:
    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=images.device).reshape(1, 3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616), device=images.device).reshape(1, 3, 1, 1)
    return (images.float().div(255) - mean) / std
