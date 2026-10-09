"""Small, inspectable synthetic fixture and conventional adapter optimizer."""

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .contracts import Readout, numeric_matrix, numeric_vector
from .report_types import number, required
from .trust import ContractViolation, FitUnavailable, t2_operation, t3_boundary

CHANNELS = 16
PARAMETERS = CHANNELS * CHANNELS + CHANNELS


@t3_boundary(
    test="tests/test_trust.py::BoundaryTests.test_numeric_boundaries",
    fingerprint="0404ad9ffc3dd18aaa09deed859b847a4d88fec30950721c85377e5e0162265e",
)
def finite(value: object) -> float:
    return number(value)


def vector(values: Sequence[float]) -> tuple[float, ...]:
    return numeric_vector(values, CHANNELS)


@dataclass(frozen=True)
class Adapter:
    """W and b are the 272 learned values; alpha is external control."""

    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", numeric_matrix(self.weights, CHANNELS, CHANNELS))
        object.__setattr__(self, "bias", vector(self.bias))

    @classmethod
    def zero(cls) -> "Adapter":
        return cls(((0.0,) * CHANNELS,) * CHANNELS, (0.0,) * CHANNELS)

    def apply(self, h: Sequence[float], alpha: float = 1.0) -> tuple[float, ...]:
        h, alpha = vector(h), finite(alpha)
        if alpha == 0.0:
            return h
        features = tuple(math.tanh(value) for value in h)
        return vector(
            tuple(
                value + alpha * (sum(w * x for w, x in zip(row, features, strict=True)) + b)
                for value, row, b in zip(h, self.weights, self.bias, strict=True)
            )
        )

    def to_json(self) -> str:
        return json.dumps({"weights": self.weights, "bias": self.bias}, allow_nan=False)

    @classmethod
    @t3_boundary(
        test="tests/test_trust.py::BoundaryTests.test_adapter_json_boundary",
        fingerprint="694f143a3601fed06425ec1ced03364441b0ea1888f1221d9e2dd3dc2715708d",
    )
    def from_json(cls, payload: str) -> "Adapter":
        raw: object = json.loads(payload)
        data = required(raw, ("weights", "bias"))
        return cls(numeric_matrix(data["weights"], CHANNELS, CHANNELS), numeric_vector(data["bias"], CHANNELS))


@dataclass(frozen=True)
class Example:
    sample_id: str
    h: tuple[float, ...]
    target: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample ID must be a nonempty string")
        object.__setattr__(self, "h", vector(self.h))
        object.__setattr__(self, "target", vector(self.target))


@dataclass(frozen=True)
class Episode:
    host_lineage: str
    bottleneck_family: str
    conditioning: tuple[Example, ...]
    selection: tuple[Example, ...]
    test: tuple[Example, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.host_lineage, str)
            or not self.host_lineage
            or not isinstance(self.bottleneck_family, str)
            or not self.bottleneck_family
        ):
            raise ValueError("group identifiers must be nonempty")
        seen: set[str] = set()
        for field, split in (("conditioning", self.conditioning), ("selection", self.selection), ("test", self.test)):
            examples = tuple(split)
            if not examples or any(not isinstance(item, Example) for item in examples):
                raise ValueError("each episode split needs paired examples")
            ids = [item.sample_id for item in examples]
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError("sample IDs must be unique and disjoint across splits")
            seen.update(ids)
            object.__setattr__(self, field, examples)


def guard_group_splits(splits: Mapping[str, Sequence[Episode]]) -> None:
    """Reject shared lineage OR shared family across outer dataset partitions."""
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for attribute, a, b in (
                (
                    "host_lineage",
                    {item.host_lineage for item in splits[left]},
                    {item.host_lineage for item in splits[right]},
                ),
                (
                    "bottleneck_family",
                    {item.bottleneck_family for item in splits[left]},
                    {item.bottleneck_family for item in splits[right]},
                ),
            ):
                if a & b:
                    raise ValueError(f"{attribute} overlaps {left} and {right}")


@dataclass(frozen=True)
class FrozenHost:
    """Identity readout at a known 16-channel site; no trainable host state."""

    site_id: str = "synthetic-site-16"

    def output(self, h: Sequence[float]) -> tuple[float, ...]:
        return vector(h)


@dataclass(frozen=True)
class ShadowProbe:
    """Diagnostic scalar fixture, separate from the adapter benchmark."""

    slope: float
    offset: float = 0.0
    target: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "slope", finite(self.slope))
        object.__setattr__(self, "offset", finite(self.offset))
        object.__setattr__(self, "target", finite(self.target))

    def observe(self, z: float = 0.0) -> tuple[float, float, float]:
        z = finite(z)
        return z, finite(self.offset + self.slope * z), self.target


def synthetic_episode(seed: int = 7, *, healthy: bool = False) -> Episode:
    """Oracle target generation stays here; fit() receives paired examples only."""
    rng = random.Random(seed)
    host = FrozenHost()
    planted = (
        Adapter.zero()
        if healthy
        else Adapter(
            tuple(
                tuple((0.35 if i == j else 0.08 if j == (i + 1) % CHANNELS else 0.0) for j in range(CHANNELS))
                for i in range(CHANNELS)
            ),
            tuple(0.025 * (-1 if i % 2 else 1) for i in range(CHANNELS)),
        )
    )

    def examples(name: str, count: int) -> tuple[Example, ...]:
        result = []
        for i in range(count):
            h = host.output(tuple(rng.uniform(-2.0, 2.0) for _ in range(CHANNELS)))
            result.append(Example(f"{seed}:{name}:{i}", h, planted.apply(h)))
        return tuple(result)

    return Episode(
        f"synthetic-lineage-{seed}",
        f"synthetic-family-{seed}",
        examples("conditioning", 64),
        examples("selection", 24),
        examples("test", 32),
    )


@t2_operation(
    invariants="Finite complete examples, nonempty conditioning, and finite positive ridge; zero numeric pivot is recoverable.",
    failures=(FitUnavailable,),
)
def fit(conditioning: Sequence[Example], ridge: float = 1e-8) -> Adapter:
    """Ridge normal equations on tanh(h), fitting W and b only.

    This is conventional optimization on new-host conditioning pairs, not
    forward-only generated repair formation. No selection/test argument exists.
    """
    ridge = finite(ridge)
    if ridge <= 0 or not conditioning:
        raise ContractViolation("positive ridge and nonempty conditioning are required")
    width = CHANNELS + 1
    gram = [[ridge if i == j else 0.0 for j in range(width)] for i in range(width)]
    rhs = [[0.0] * CHANNELS for _ in range(width)]
    for example in conditioning:
        features = [math.tanh(value) for value in example.h] + [1.0]
        residual = [target - h for target, h in zip(example.target, example.h, strict=True)]
        for i in range(width):
            for j in range(width):
                gram[i][j] += features[i] * features[j]
            for j in range(CHANNELS):
                rhs[i][j] += features[i] * residual[j]
    # Pivoted elimination solves all sixteen outputs with one 17 x 17 system.
    augmented = [a + b for a, b in zip(gram, rhs, strict=True)]
    for column in range(width):
        pivot = max(range(column, width), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = finite(augmented[column][column])
        if divisor == 0.0:
            raise FitUnavailable("singular conditioning system at machine precision")
        augmented[column] = [finite(value / divisor) for value in augmented[column]]
        for row in range(width):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [
                    finite(a - factor * b) for a, b in zip(augmented[row], augmented[column], strict=True)
                ]
    solution = [row[width:] for row in augmented]
    return Adapter(tuple(tuple(solution[j][i] for j in range(CHANNELS)) for i in range(CHANNELS)), tuple(solution[-1]))


def feature_rank(conditioning: Sequence[Example], tolerance: float = 1e-10) -> int:
    """Numerical rank diagnostic of the 17-column fitting feature matrix."""
    tolerance = finite(tolerance)
    if tolerance <= 0 or not conditioning:
        raise ValueError("positive tolerance and conditioning required")
    rows = [[math.tanh(value) for value in example.h] + [1.0] for example in conditioning]
    rank = 0
    for column in range(CHANNELS + 1):
        if rank == len(rows):
            break
        pivot = max(range(rank, len(rows)), key=lambda i: abs(rows[i][column]))
        if abs(rows[pivot][column]) <= tolerance:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][column]
        rows[rank] = [value / divisor for value in rows[rank]]
        for i in range(rank + 1, len(rows)):
            factor = rows[i][column]
            rows[i] = [a - factor * b for a, b in zip(rows[i], rows[rank], strict=True)]
        rank += 1
    return rank


def mse(adapter: Readout, examples: Sequence[Example]) -> float:
    if not examples:
        raise ValueError("evaluation requires nonempty examples")
    total = sum(
        (prediction - target) ** 2
        for example in examples
        for prediction, target in zip(vector(adapter.apply(example.h)), example.target, strict=True)
    )
    return finite(total / (len(examples) * CHANNELS))


@dataclass(frozen=True)
class Selection:
    candidate_index: int
    adapter: Adapter
    losses: tuple[float, ...]
    candidates: int
    selection_queries: int


def select(candidates: Sequence[Adapter], selection: Sequence[Example]) -> Selection:
    """Only selection pairs determine the winner; ties prefer the first arm."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    losses = tuple(mse(candidate, selection) for candidate in candidates)
    index = min(range(len(losses)), key=losses.__getitem__)
    return Selection(index, candidates[index], losses, len(candidates), len(candidates) * len(selection))
