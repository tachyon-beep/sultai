"""M0 M-aff fold: the affine module u = z + W z + b folded into tail[0], and the fold residual on development."""

from __future__ import annotations

import torch
from torch import Tensor

from sultai.trust import ContractViolation
from sultai_screen.data import Dataset, as_tensor
from sultai_screen.fit import Adapter, mean_loss
from sultai_screen.host import Features, Host, freeze, state_hash
from sultai_screen.m0.data import features_from_images


def _affine(adapter: Adapter) -> tuple[Tensor, Tensor]:
    if len(adapter.weights) != 2 or adapter.nonlinear:
        raise ContractViolation("fold requires the two-tensor affine adapter (nonlinear=False)")
    w, b = adapter.weights
    if tuple(w.shape) != (16, 16) or tuple(b.shape) != (16,):
        raise ContractViolation("affine adapter must be W [16, 16] and b [16]")
    return w, b


def fold_affine(host: Host, adapter: Adapter) -> Host:
    """Fresh frozen host with tail[0] absorbing u = z + W z + b; exact in the interior, over-adds b on the border ring.

    K'[o, i] = sum_j K[o, j] (I + W)[j, i] and c'[o] = c[o] + sum_j sum_hw K[o, j, h, w] b[j], in float64.
    """
    if host.width != 0:
        raise ContractViolation("affine fold requires an identity bottleneck (width 0), where read is base")
    w, b = _affine(adapter)
    state = dict(host.state_dict())
    kernel, bias = as_tensor(state["tail.0.weight"]), as_tensor(state["tail.0.bias"])
    if tuple(kernel.shape[1:]) != (16, 3, 3):
        raise ContractViolation("tail[0] must be a 3x3 convolution over the 16-channel site")
    k64 = kernel.detach().double()
    mixing = torch.eye(16, dtype=torch.float64, device=k64.device) + w.detach().double().to(k64.device)
    folded_kernel = torch.einsum("ojhw,ji->oihw", k64, mixing)
    folded_bias = bias.detach().double() + torch.einsum("ojhw,j->o", k64, b.detach().double().to(k64.device))
    # New tensors only: state_dict() entries share storage with the original host.
    state["tail.0.weight"] = folded_kernel.to(kernel.dtype).contiguous()
    state["tail.0.bias"] = folded_bias.to(bias.dtype).contiguous()
    # Host construction runs nn initialization; keep it off the caller's global RNG stream.
    with torch.random.fork_rng(devices=[]):
        result = Host(host.width, widths=host.widths)
    result = result.to(kernel.device)
    result.load_state_dict(state, strict=True)
    freeze(result)
    return result


def fold_residual(
    host: Host, adapter: Adapter, folded: Host, development: Features, data: Dataset, development_indices: Tensor
) -> float:
    """r_aff = L(folded host, from images) - L(host + adapter, cached site) on development, mean cross-entropy."""
    if state_hash(folded) != state_hash(fold_affine(host, adapter)):
        raise ContractViolation("folded host is not fold_affine(host, adapter)")
    from_images = features_from_images(folded, data, development_indices)
    return mean_loss(folded, from_images, None) - mean_loss(host, development, adapter)
