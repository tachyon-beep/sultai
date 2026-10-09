"""Small validated state contracts; unknown values are parsed explicitly."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .report_types import PartitionIdentity, SplitIdentity, count, number, sequence, string, validate_isolation
from .trust import InputDataError, t3_boundary


@t3_boundary(
    test="tests/test_trust.py::BoundaryTests.test_numeric_boundaries",
    fingerprint="0404ad9ffc3dd18aaa09deed859b847a4d88fec30950721c85377e5e0162265e",
)
def numeric_vector(value: object, width: int) -> tuple[float, ...]:
    values = tuple(sequence(value, number))
    if len(values) != width:
        raise InputDataError(f"expected exactly {width} channels")
    return values


@t3_boundary(
    test="tests/test_trust.py::BoundaryTests.test_numeric_boundaries",
    fingerprint="0404ad9ffc3dd18aaa09deed859b847a4d88fec30950721c85377e5e0162265e",
)
def numeric_matrix(value: object, rows: int, columns: int) -> tuple[tuple[float, ...], ...]:
    values = tuple(sequence(value, lambda row: numeric_vector(row, columns)))
    if len(values) != rows:
        raise InputDataError(f"expected exactly {rows} weight rows")
    return values


class Readout(Protocol):
    def apply(self, h: Sequence[float]) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class PartitionValidation:
    lineages: tuple[str, ...]
    sample_count: int
    sample_ids_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineages", tuple(sequence(self.lineages, string)))
        object.__setattr__(self, "sample_count", count(self.sample_count))
        digest = string(self.sample_ids_sha256)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("invalid sample identity digest")

    def report(self) -> PartitionIdentity:
        return {
            "lineages": self.lineages,
            "sample_count": self.sample_count,
            "sample_ids_sha256": self.sample_ids_sha256,
        }


@dataclass(frozen=True)
class SplitValidation:
    train: PartitionValidation
    development: PartitionValidation
    test: PartitionValidation

    def __post_init__(self) -> None:
        if not all(isinstance(part, PartitionValidation) for part in (self.train, self.development, self.test)):
            raise ValueError("validated split partitions are required")
        validate_isolation(self.report())

    def report(self) -> SplitIdentity:
        return {"train": self.train.report(), "development": self.development.report(), "test": self.test.report()}
