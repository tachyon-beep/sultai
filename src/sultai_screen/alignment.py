"""In-package Hungarian channel alignment and two-sided adapter transport (descriptive B4)."""

from __future__ import annotations

import torch
from torch import Tensor

from sultai.trust import ContractViolation

from .fit import Adapter


def hungarian(cost: tuple[tuple[float, ...], ...]) -> tuple[int, ...]:
    n = len(cost)
    if not n or any(len(row) != n for row in cost):
        raise ContractViolation("nonempty square assignment cost required")
    u, v = [0.0] * (n + 1), [0.0] * (n + 1)
    p, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return tuple(assignment)


def align(reference: Tensor, other: Tensor) -> Tensor:
    if reference.shape != other.shape or reference.ndim != 4 or reference.shape[1] != 16:
        raise ContractViolation("matched calibration sites required")
    a = reference.permute(1, 0, 2, 3).reshape(16, -1).double()
    b = other.permute(1, 0, 2, 3).reshape(16, -1).double()
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    norm = a.norm(dim=1)[:, None] * b.norm(dim=1)[None, :]
    correlation = torch.where(norm > 0, (a @ b.T) / norm.clamp_min(1e-30), torch.zeros_like(norm))
    cost = tuple(tuple(-float(v) for v in row) for row in correlation.cpu())
    assignment = hungarian(cost)
    return torch.eye(16, device=reference.device)[torch.tensor(assignment, device=reference.device)]


def transport(adapter: Adapter, p_in: Tensor, p_out: Tensor) -> Adapter:
    if len(adapter.weights) != 2:
        raise ContractViolation("B4 transports only a 272-parameter adapter")
    w, b = adapter.weights
    return Adapter((p_out.T @ w @ p_in, p_out.T @ b))


def moments(values: Tensor) -> Tensor:
    x = values.permute(1, 0, 2, 3).reshape(16, -1).double()
    mean = x.mean(1)
    centered = x - mean[:, None]
    covariance = centered @ centered.T / (x.shape[1] - 1)
    return torch.cat((mean, covariance.flatten()))
