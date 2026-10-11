"""Existing frozen synthetic former and analytic span control on stratified real-site pairs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from torch import Tensor

from sultai.formation import FrozenGenerator, run_formation
from sultai.formation_controls import analytic_template_fit
from sultai.repair import Adapter as LegacyAdapter
from sultai.repair import Example
from sultai.trust import ContractViolation

from .contracts import Record
from .data import generator
from .fit import Adapter


@dataclass(frozen=True)
class Former:
    generator: FrozenGenerator
    weights_sha: str
    report_sha: str

    def identity(self) -> Record:
        return {
            "weights_sha256": self.weights_sha,
            "formation_report_sha256": self.report_sha,
            "mode": "paired",
            "train_seeds": list(range(1000, 1032)),
        }


def derive_former() -> Former:
    report, frozen = run_formation()
    if frozen.mode != "paired" or len(frozen.weights) != 11 or any(len(row) != 4 for row in frozen.weights):
        raise ContractViolation("existing paired 11x4 former required")

    # Canonical compact sorted JSON, tuples serialized as JSON arrays, no NaN.
    def _digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        ).hexdigest()

    return Former(frozen, _digest(frozen.weights), _digest(report))


def position_pairs(read: Tensor, target: Tensor, labels: Tensor, seed: int) -> tuple[Example, ...]:
    if read.shape != target.shape or read.shape[1:] != (16, 16, 16) or len(labels) != len(read):
        raise ContractViolation("matched 16x16 sites and labels required")
    rng = generator(seed, "former-position-pairs")
    selected: list[int] = []
    for label in range(10):
        indices = torch.nonzero(labels.cpu() == label).flatten()
        count = 103 if label < 4 else 102
        if len(indices) < count:
            raise ContractViolation("conditioning cannot supply 1024 class-balanced images")
        selected.extend(int(i) for i in indices[torch.randperm(len(indices), generator=rng)[:count]])
    source = read.detach().cpu()
    truth = target.detach().cpu()
    offsets = torch.randint(0, 8, (1024, 4, 2), generator=rng)
    pairs = []
    for i, image in enumerate(selected):
        for quadrant in range(4):
            y = int(offsets[i, quadrant, 0]) + (quadrant // 2) * 8
            x = int(offsets[i, quadrant, 1]) + (quadrant % 2) * 8
            pairs.append(
                Example(
                    f"{image}:{y}:{x}",
                    tuple(float(v) for v in source[image, :, y, x]),
                    tuple(float(v) for v in truth[image, :, y, x]),
                )
            )
    return tuple(pairs)


def tensor_adapter(adapter: LegacyAdapter, device: torch.device) -> Adapter:
    return Adapter((torch.tensor(adapter.weights, device=device), torch.tensor(adapter.bias, device=device)))


def former_fits(former: Former, read: Tensor, target: Tensor, labels: Tensor, seed: int) -> tuple[Adapter, Adapter]:
    pairs = position_pairs(read, target, labels, seed)
    return tensor_adapter(former.generator.form(pairs), read.device), tensor_adapter(
        analytic_template_fit(pairs), read.device
    )
