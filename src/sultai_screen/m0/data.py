"""M0 data: the seen-fit sub-partition of the 40,000 fit-train ids, and front/bottleneck features for any ids."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from sultai.trust import ContractViolation
from sultai_screen.data import Dataset, as_tensor, generator, normalized
from sultai_screen.host import Features, Host, assert_frozen, state_hash

SEEN_FIT_SEED: Final = 20261013
SEEN_FIT_LABEL: Final = "m0-seen-fit-v1"
SEEN_FIT_COUNT: Final = 2000
TRAIN_COUNT: Final = 40_000
# SHA-256 of the little-endian int64 bytes of the 2,000 production indices (derived from partitions()[0]).
SEEN_FIT_SHA: Final = "a4130e3c0eb483fbf0af0d975ddc341bed1781819a249f974baca07d73d21c9c"


def seen_fit(data: Dataset) -> Tensor:
    """Fixed 2,000-id subset of data.train, disjoint from conditioning, selection and development."""
    if len(data.train) != TRAIN_COUNT:
        raise ContractViolation(f"fit-train must have {TRAIN_COUNT} entries")
    order = torch.randperm(TRAIN_COUNT, generator=generator(SEEN_FIT_SEED, SEEN_FIT_LABEL))
    return data.train[order[:SEEN_FIT_COUNT]]


def features_from_images(host: Host, data: Dataset, indices: Tensor) -> Features:
    """Front and bottleneck outputs for arbitrary image ids, mirroring host.cached.part."""
    expected = state_hash(host)
    assert_frozen(host, expected)
    reads, bases = [], []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            ids = indices[start : start + 128].to(data.images.device)
            z = as_tensor(host.front(normalized(data.images[ids])))
            base = as_tensor(host.bottleneck(z))
            reads.append(z.detach())
            bases.append(base.detach())
    result = Features(torch.cat(reads), torch.cat(bases), data.labels[indices.to(data.labels.device)])
    assert_frozen(host, expected)
    return result
