"""Pinned host architecture, deterministic training, and frozen cached-site execution."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sultai.trust import ContractViolation, InputDataError, t3_boundary

from .contracts import PilotHold, Record, derive
from .data import Dataset, as_tensor, generator, normalized


def configure(*, cuda: bool) -> torch.device:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    if cuda:
        if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] not in ("0", "1"):
            raise PilotHold("one explicitly pinned physical card required")
        if torch.cuda.device_count() != 1:
            raise PilotHold("exactly one CUDA device must be visible")
    return torch.device("cuda:0" if cuda else "cpu")


def flags() -> Record:
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "threads": torch.get_num_threads(),
        "amp": False,
        "compile": False,
        "layout": "contiguous NCHW",
    }


def _block(inside: int, outside: int, kernel: int = 3) -> list[nn.Module]:
    return [nn.Conv2d(inside, outside, kernel, padding=kernel // 2), nn.BatchNorm2d(outside), nn.ReLU()]


REFERENCE_WIDTHS = (64, 64, 128, 128)


class Host(nn.Module):
    def __init__(self, width: int = 0, widths: tuple[int, int, int, int] = REFERENCE_WIDTHS) -> None:
        super().__init__()
        if width not in (0, 2, 4, 8):
            raise ContractViolation("registered host width required")
        if len(widths) != 4 or any(type(w) is not int or w <= 0 for w in widths):
            raise ContractViolation("tail widths must be four positive integers")
        self.front = nn.Sequential(*_block(3, 32), *_block(32, 32), nn.MaxPool2d(2), *_block(32, 16))
        self.bottleneck = nn.Sequential(*_block(16, width, 1), *_block(width, 16, 1)) if width else nn.Identity()
        self.tail = nn.Sequential(
            *_block(16, widths[0]),
            *_block(widths[0], widths[1]),
            nn.MaxPool2d(2),
            *_block(widths[1], widths[2]),
            *_block(widths[2], widths[3]),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(widths[3], 10)
        self.width = width
        self.widths = widths
        self.sealed = False

    def classify(self, site: Tensor) -> Tensor:
        return as_tensor(self.classifier(as_tensor(self.tail(site)).mean(dim=(2, 3))))

    def forward(self, images: Tensor) -> Tensor:
        return self.classify(as_tensor(self.bottleneck(as_tensor(self.front(images)))))


def make_host(
    seed: int, width: int, device: torch.device, *, widths: tuple[int, int, int, int] = REFERENCE_WIDTHS
) -> Host:
    # nn initialization uses the explicitly derived CPU global RNG only.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive(seed, "host-init"))
        host = Host(width, widths)
    return host.to(device)


def parameter_count(host: Host) -> int:
    return sum(p.numel() for p in host.parameters())


def state_hash(host: Host) -> str:
    digest = hashlib.sha256()
    for kind, pairs in (("parameter", host.named_parameters()), ("buffer", host.named_buffers())):
        for name, tensor in pairs:
            digest.update(f"{kind}:{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _bn_guard(module: nn.Module, _inputs: tuple[object, ...]) -> None:
    if module.training:
        raise PilotHold("BatchNorm entered train mode in frozen execution")


def freeze(host: Host) -> None:
    host.sealed = True
    host.eval()
    host.requires_grad_(False)
    for parameter in host.parameters():
        parameter.grad = None
    for module in host.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.register_forward_pre_hook(_bn_guard)


def assert_frozen(host: Host, expected: str) -> None:
    if host.training or any(module.training for module in host.modules() if isinstance(module, nn.BatchNorm2d)):
        raise PilotHold("frozen host/BatchNorm mode changed")
    if any(p.requires_grad or p.grad is not None for p in host.parameters()) or state_hash(host) != expected:
        raise PilotHold("frozen host parameters, buffers or gradients changed")


def augment(images: Tensor, y: Tensor, x: Tensor, flip: Tensor) -> Tensor:
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    rows = y.to(images.device)[:, None, None] + torch.arange(32, device=images.device)[None, :, None]
    cols = x.to(images.device)[:, None, None] + torch.arange(32, device=images.device)[None, None, :]
    batch = torch.arange(len(images), device=images.device)[:, None, None]
    cropped = padded.permute(0, 2, 3, 1)[batch, rows, cols].permute(0, 3, 1, 2).contiguous()
    return torch.where(flip.to(images.device)[:, None, None, None], cropped.flip(3), cropped).contiguous()


Schedule = Callable[[int], float]


def constant(rate: float) -> Schedule:
    def schedule(_step: int) -> float:
        return rate

    return schedule


def linear_cooldown(start_step: int, stop_step: int, rate: float = 0.05) -> Schedule:
    if not 0 <= start_step < stop_step:
        raise ContractViolation("cooldown requires 0 <= start_step < stop_step")

    def schedule(step: int) -> float:
        if step < start_step:
            return rate
        if step < stop_step:
            return rate * (1 - (step - start_step) / (stop_step - start_step))
        return 0.0

    return schedule


# Global optimiser steps at batch 32 on the 40,000-example partition: 1,250 per epoch.
TCOOL = linear_cooldown(12_500, 18_750)
COOLED_20 = linear_cooldown(18_750, 25_000)


def recipe_optimizer(parameters: Iterable[nn.Parameter]) -> torch.optim.SGD:
    materialised = tuple(parameters)
    decay = [p for p in materialised if p.ndim > 1]
    affine = [p for p in materialised if p.ndim <= 1]
    return torch.optim.SGD(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": affine, "weight_decay": 0.0},
        ],
        lr=0.05,
        momentum=0.9,
        nesterov=True,
    )


def train_epochs(
    model: nn.Module,
    hashed: Host,
    data: Dataset,
    seed: int,
    *,
    epochs: range,
    rates: Callable[[int], float],
    optimizer: torch.optim.SGD,
) -> tuple[list[str], Tensor, float]:
    if epochs.step != 1 or epochs.start < 0 or len(epochs) == 0:
        raise ContractViolation("training epochs must be a non-empty contiguous range from a non-negative start")
    model.train()
    parameters = tuple(as_tensor(p) for group in optimizer.param_groups for p in group["params"])
    step = epochs.start * ((len(data.train) + 31) // 32)
    device = data.images.device
    # No per-step float()/item() sync. Every loss remains on the device until the epoch ends.
    stream: list[Tensor] = []
    hashes: list[str] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.monotonic()
    for epoch in epochs:
        order = data.train[torch.randperm(len(data.train), generator=generator(seed, f"order:{epoch}"))]
        crop = torch.randint(0, 9, (len(order), 2), generator=generator(seed, f"crop:{epoch}"))
        flip = torch.rand(len(order), generator=generator(seed, f"flip:{epoch}")) < 0.5
        epoch_losses = []
        for index in range(0, len(order), 32):
            ids = order[index : index + 32].to(device)
            batch = augment(
                data.images[ids], crop[index : index + 32, 0], crop[index : index + 32, 1], flip[index : index + 32]
            )
            logits = as_tensor(model(normalized(batch)))
            loss = F.cross_entropy(logits, data.labels[ids])
            optimizer.zero_grad(set_to_none=True)
            gradients = torch.autograd.grad(loss, parameters)
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter.grad = gradient
            rate = rates(step)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            step += 1
            epoch_losses.append(loss.detach())
        values = torch.stack(epoch_losses)
        if not bool(torch.isfinite(values).all()):
            raise PilotHold("nonfinite host training loss")
        stream.append(values.cpu())
        hashes.append(state_hash(hashed))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return hashes, torch.cat(stream), time.monotonic() - start


def train_host(host: Host, data: Dataset, seed: int, *, epochs: int = 10) -> tuple[list[str], Tensor, float]:
    if epochs != 10:
        raise ContractViolation("registered host recipe is exactly ten epochs")
    return train_epochs(
        host, host, data, seed, epochs=range(10), rates=constant(0.05), optimizer=recipe_optimizer(host.parameters())
    )


def momentum_buffers(optimizer: torch.optim.SGD, host: nn.Module) -> dict[str, Tensor]:
    buffers: dict[str, Tensor] = {}
    for name, parameter in host.named_parameters():
        if parameter not in optimizer.state or "momentum_buffer" not in optimizer.state[parameter]:
            raise ContractViolation(f"no momentum buffer for {name}; optimiser has not stepped this parameter")
        buffers[name] = as_tensor(optimizer.state[parameter]["momentum_buffer"]).detach().clone()
    return buffers


@t3_boundary(
    test="tests/test_screen.py::ScreenTests.test_checkpoint_boundary",
    fingerprint="6971955b56153dd8bef4566e0c1de34b9d916d168df4360e89451f55f464aea2",
)
def checkpoint_state(value: object, expected: dict[str, Tensor]) -> dict[str, Tensor]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise InputDataError("checkpoint must contain exactly the registered state keys")
    result: dict[str, Tensor] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, Tensor) or not bool(torch.isfinite(item).all()):
            raise InputDataError("checkpoint requires finite tensors and string keys")
        if item.shape != expected[key].shape or item.dtype != expected[key].dtype:
            raise InputDataError("checkpoint tensor shape/dtype differs from registered host")
        result[key] = item
    return result


@t3_boundary(
    test="tests/test_m0_host.py::HostTests.test_optimizer_state_boundary_rejects_missing_keys_wrong_shape_and_nonfinite",
    fingerprint="150973c7f74bf3fa8c04d6dfc256e9128b08a797c5be0a51101087dc02099944",
)
def optimizer_state(value: object, expected: dict[str, Tensor]) -> dict[str, Tensor]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise InputDataError("optimiser state must contain exactly the host parameter names")
    result: dict[str, Tensor] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, Tensor) or not bool(torch.isfinite(item).all()):
            raise InputDataError("optimiser state requires finite tensors and string keys")
        if item.shape != expected[key].shape or item.dtype != expected[key].dtype:
            raise InputDataError("momentum buffer shape/dtype differs from its host parameter")
        result[key] = item
    return result


def restore_optimizer(host: nn.Module, buffers: dict[str, Tensor]) -> torch.optim.SGD:
    named = dict(host.named_parameters())
    if set(buffers) != set(named):
        raise ContractViolation("momentum buffers must cover exactly the host parameters")
    optimizer = recipe_optimizer(named.values())
    for name, parameter in named.items():
        optimizer.state[parameter]["momentum_buffer"] = buffers[name].detach().clone().to(parameter.device)
    return optimizer


def load_host(
    path: Path,
    seed: int,
    width: int,
    device: torch.device,
    *,
    widths: tuple[int, int, int, int] = REFERENCE_WIDTHS,
    frozen: bool = True,
) -> Host:
    host = make_host(seed, width, device, widths=widths)
    unknown: object = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint_state(unknown, dict(host.state_dict()))
    host.load_state_dict(state, strict=True)
    if frozen:
        freeze(host)
    return host


@dataclass(frozen=True)
class Features:
    read: Tensor
    base: Tensor
    labels: Tensor


@dataclass(frozen=True)
class Cached:
    conditioning: Features
    selection: Features
    development: Features


def cached(host: Host, data: Dataset) -> Cached:
    expected = state_hash(host)
    assert_frozen(host, expected)

    def part(indices: Tensor) -> Features:
        reads, bases = [], []
        with torch.no_grad():
            for index in range(0, len(indices), 128):
                z = as_tensor(host.front(normalized(data.images[indices[index : index + 128].to(data.images.device)])))
                base = as_tensor(host.bottleneck(z))
                reads.append(z.detach())
                bases.append(base.detach())
        return Features(torch.cat(reads), torch.cat(bases), data.labels[indices.to(data.labels.device)])

    result = Cached(part(data.conditioning), part(data.selection), part(data.development))
    assert_frozen(host, expected)
    return result
