"""M0 module arms: contribution slicing with the mean fold, leave-one-out ablation, contraction and lottery start."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

from sultai.trust import ContractViolation
from sultai_screen.data import generator
from sultai_screen.fit import Adapter, initial_adapter, losses
from sultai_screen.host import Features, Host

FRONTIER_WIDTHS: Final[dict[float, tuple[int, int, int, int]]] = {
    0.25: (16, 16, 32, 32),
    0.5: (32, 32, 64, 64),
    0.75: (48, 48, 96, 96),
    1.0: (64, 64, 128, 128),
    1.5: (96, 96, 192, 192),
}
LOTTERY_WIDTH: Final = 64
CHUNK: Final = 128


@dataclass(frozen=True)
class Cut:
    """A sliced module with and without the mean fold; survivors are the kept unit indices, ascending."""

    folded: Adapter
    unfolded: Adapter
    survivors: tuple[int, ...]


def _hidden(adapter: Adapter) -> int:
    if len(adapter.weights) != 4:
        raise ContractViolation("a four-tensor hidden module is required")
    return int(adapter.weights[0].shape[0])


def _width(adapter: Adapter, h: int) -> int:
    total = _hidden(adapter)
    if type(h) is not int or not 1 <= h <= total:
        raise ContractViolation("kept width must be an integer in 1..h_total")
    return total


def _moment(adapter: Adapter, read: Tensor, power: Callable[[Tensor], Tensor]) -> Tensor:
    """Float64 mean over examples and positions of power(tanh(U z + c)), in chunks of CHUNK examples."""
    _hidden(adapter)
    inside, bias_in = adapter.weights[0], adapter.weights[1]
    total = torch.zeros(len(bias_in), dtype=torch.float64, device=bias_in.device)
    with torch.no_grad():
        for start in range(0, len(read), CHUNK):
            chunk = read[start : start + CHUNK]
            hidden = torch.tanh(torch.einsum("ij,bjhw->bihw", inside, chunk) + bias_in[None, :, None, None])
            total = total + power(hidden.double()).sum(dim=(0, 2, 3))
    return total / (len(read) * read.shape[2] * read.shape[3])


def activation_means(adapter: Adapter, read: Tensor) -> Tensor:
    """[h] float64 mean of each hidden activation over the fit examples and positions."""
    return _moment(adapter, read, lambda hidden: hidden)


def contribution(adapter: Adapter, read: Tensor) -> Tensor:
    """[h] float64 ||V[:, i]||_2 times the RMS of activation i over the fit examples and positions."""
    rms = _moment(adapter, read, lambda hidden: hidden * hidden).sqrt()
    outside = adapter.weights[2].detach().double()
    return (outside * outside).sum(dim=0).sqrt() * rms


def _slice(adapter: Adapter, keep: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Validated ascending keep (possibly empty) and the sliced U, c, V with d unchanged."""
    total = _hidden(adapter)
    index = keep.to(dtype=torch.long, device=adapter.weights[0].device).reshape(-1)
    ordered = torch.sort(index).values
    if len(ordered) > total or (len(ordered) and (int(ordered[0]) < 0 or int(ordered[-1]) >= total)):
        raise ContractViolation("kept units must be indices in 0..h_total-1")
    if len(torch.unique(ordered)) != len(ordered):
        raise ContractViolation("kept units must be distinct")
    inside, bias_in, outside, bias_out = adapter.weights
    return ordered, inside[ordered], bias_in[ordered], outside[:, ordered], bias_out


def _folded(adapter: Adapter, keep: Tensor, means: Tensor) -> Adapter:
    total = _hidden(adapter)
    if means.shape != (total,):
        raise ContractViolation("means must have one entry per hidden unit")
    ordered, inside, bias_in, outside, bias_out = _slice(adapter, keep)
    removed = torch.ones(total, dtype=torch.bool, device=ordered.device)
    removed[ordered] = False
    full = adapter.weights[2]
    fold = full[:, removed].double() @ means.to(full.device, torch.float64)[removed]
    return Adapter((inside, bias_in, outside, (bias_out.double() + fold).to(bias_out.dtype)), adapter.nonlinear)


def cut(adapter: Adapter, keep: Tensor, means: Tensor) -> Adapter:
    """Slice U, c, V to keep (ascending) and fold d <- d + sum over removed i of V[:, i] * means[i]."""
    _width(adapter, len(keep.reshape(-1)))
    return _folded(adapter, keep, means)


def cut_unfolded(adapter: Adapter, keep: Tensor) -> Adapter:
    """Slice U, c, V to keep (ascending) without the mean fold."""
    _width(adapter, len(keep.reshape(-1)))
    _, inside, bias_in, outside, bias_out = _slice(adapter, keep)
    return Adapter((inside, bias_in, outside, bias_out), adapter.nonlinear)


def _cut(adapter: Adapter, read: Tensor, keep: Tensor) -> Cut:
    means = activation_means(adapter, read)
    folded = cut(adapter, keep, means)
    survivors = tuple(int(i) for i in torch.sort(keep).values.tolist())
    return Cut(folded, cut_unfolded(adapter, keep), survivors)


def rank_cut(adapter: Adapter, read: Tensor, h: int) -> Cut:
    """Keep the h largest contributions on read; ties go to the lower unit index."""
    _width(adapter, h)
    order = torch.argsort(-contribution(adapter, read), stable=True)
    return _cut(adapter, read, order[:h])


def random_cut(adapter: Adapter, read: Tensor, h: int, seed: int) -> Cut:
    """Keep h units drawn by randperm from generator(seed, "random-cut"); same mean fold as rank_cut."""
    total = _width(adapter, h)
    return _cut(adapter, read, torch.randperm(total, generator=generator(seed, "random-cut"))[:h])


def ablation_cut(adapter: Adapter, read: Tensor, h: int, ablation: Tensor) -> Cut:
    """Keep the h units whose removal costs most (largest ablation); ties go to the lower unit index."""
    total = _width(adapter, h)
    if ablation.shape != (total,):
        raise ContractViolation("ablation must have one entry per hidden unit")
    return _cut(adapter, read, torch.argsort(-ablation, stable=True)[:h])


def leave_one_out(host: Host, data: Features, adapter: Adapter) -> Tensor:
    """[h] float64 mean fit loss with unit i removed under the mean fold, minus the full module's mean fit loss."""
    total = _hidden(adapter)
    means = activation_means(adapter, data.read)
    full = losses(host, data, adapter).double().mean()
    result = []
    for unit in range(total):
        keep = torch.tensor([i for i in range(total) if i != unit], dtype=torch.long)
        removed = _folded(adapter, keep, means)
        result.append(losses(host, data, removed).double().mean() - full)
    return torch.stack(result).cpu()


def contract(expanded: Adapter) -> Adapter:
    """Exact two-tensor (w1 @ w2, b) form of an M-exp adapter."""
    if len(expanded.weights) != 3:
        raise ContractViolation("a three-tensor expanded adapter is required")
    w1, w2, bias = expanded.weights
    return Adapter((w1 @ w2, bias), nonlinear=True)


def lottery_start(seed: int, keep: tuple[int, ...], device: torch.device) -> Adapter:
    """The 64-wide initial module of seed, regenerated and sliced to the strictly ascending keep."""
    if not 1 <= len(keep) <= LOTTERY_WIDTH or any(type(i) is not int for i in keep):
        raise ContractViolation("lottery keep must hold 1..64 integer unit indices")
    if list(keep) != sorted(set(keep)) or keep[0] < 0 or keep[-1] >= LOTTERY_WIDTH:
        raise ContractViolation("lottery keep must be strictly ascending indices in 0..63")
    initial = initial_adapter(seed, hidden=LOTTERY_WIDTH, nonlinear=True, expanded=False, device=device)
    return cut_unfolded(initial, torch.tensor(keep, dtype=torch.long))
