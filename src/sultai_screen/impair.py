"""Declared post-hoc operators, drawn once per anchor, and their exact affine inverses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from sultai.trust import ContractViolation

from .data import as_tensor, generator
from .host import Features


@dataclass(frozen=True)
class Impairment:
    family: str
    matrix: Tensor
    offset: Tensor
    inverse: Tensor

    def apply(self, values: Tensor) -> Tensor:
        return torch.einsum("ij,bjhw->bihw", self.matrix, values) + self.offset[None, :, None, None]

    def undo(self, values: Tensor) -> Tensor:
        return torch.einsum("ij,bjhw->bihw", self.inverse, values - self.offset[None, :, None, None])

    def features(self, original: Features) -> Features:
        values = self.apply(original.read).contiguous()
        return Features(values, values, original.labels)


def haar(seed: int, label: str) -> Tensor:
    q, r = torch.linalg.qr(torch.randn((16, 16), generator=generator(seed, label), dtype=torch.float64))
    return as_tensor(q * torch.diag(r).sign()[None, :])


def make_impairment(family: str, magnitude: float, seed: int, conditioning: Tensor) -> Impairment:
    if conditioning.ndim != 4 or conditioning.shape[1] != 16 or not math.isfinite(magnitude) or magnitude <= 0:
        raise ContractViolation("finite positive magnitude and [N,16,H,W] conditioning required")
    device, dtype = conditioning.device, conditioning.dtype
    matrix = torch.eye(16, dtype=torch.float64)
    offset = torch.zeros(16, dtype=torch.float64)
    if family == "P1":
        channels = torch.randperm(16, generator=generator(seed, "P1-channels"))[:4]
        matrix[channels, channels] = magnitude
    elif family == "P2":
        u, v = haar(seed, "P2-U"), haar(seed, "P2-V")
        spectrum = torch.logspace(0, math.log10(magnitude), 16, dtype=torch.float64)
        matrix = (u * spectrum[None, :]) @ v.T
    elif family in ("P3", "P3prime"):
        direction = torch.randn(16, generator=generator(seed, "P3-direction"), dtype=torch.float64)
        if family == "P3prime":
            direction = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(16)], dtype=torch.float64)
        rms = float(conditioning.double().square().mean().sqrt())
        offset = direction / direction.norm() * (magnitude * rms)
    else:
        raise ContractViolation("registered post-hoc family required")
    inverse = torch.linalg.inv(matrix)
    return Impairment(
        family,
        matrix.to(device=device, dtype=dtype),
        offset.to(device=device, dtype=dtype),
        inverse.to(device=device, dtype=dtype),
    )
