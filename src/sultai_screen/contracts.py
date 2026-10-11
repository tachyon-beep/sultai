"""Explicit screen contracts and deterministic serialization, independent of torch."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import TypeAlias

from sultai.trust import ContractViolation, InputDataError, t3_boundary

Json: TypeAlias = "None | bool | int | float | str | list[Json] | dict[str, Json]"
Record: TypeAlias = dict[str, Json]
PREREG_SHA = "7668d20ad55c3a62384975a7cf1f3989dbe69bf701ce02bc9c68ac2601d0d42a"
DEV_SHA = "27173d10eb5edba1d0fd331e7f85ea633753cfae11400c4e7ef81810c94eeeba"
PERM_SHA = "df1d87871f9afe3b15d7c5a434c31c5f68ff99564b4512410cc64fb1d042a24e"
X = 0.02
RHO = 0.8
HARM = 0.01
UNCERTIFIED = 0.25
PILOT_SECONDS = 3 * 3600
ORACLE_RATES = (0.3, 0.1, 0.03, 0.01)
CHEAP_RATES = (3.0, 1.0, 0.3, 0.1, 0.03)


class PilotHold(RuntimeError):
    """A published pilot deviation requires operator disposition; never a pass."""


class InstrumentHold(PilotHold):
    """Valid execution found that a declared scientific instrument requirement failed."""


class FeasibilityHold(PilotHold):
    """Valid measurements exceed the frozen compute/count envelope; operator decision required."""


class OracleUnavailable(RuntimeError):
    """All candidates failed numerically; retain this anchor and report the arm."""


def derive(seed: int, *labels: str) -> int:
    if seed < 0 or seed >= 2**64:
        raise ContractViolation("unsigned 64-bit seed required")
    digest = hashlib.sha256(seed.to_bytes(8, "big"))
    for label in labels:
        raw = label.encode()
        digest.update(len(raw).to_bytes(4, "big"))
        digest.update(raw)
    return int.from_bytes(digest.digest()[:8], "big")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value: Json) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def publish(path: Path, value: Json) -> None:
    """Exclusive publication: never replace an earlier receipt or result."""
    with path.open("x") as stream:
        stream.write(encode(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@t3_boundary(
    test="tests/test_screen.py::ScreenTests.test_json_boundary",
    fingerprint="26c80e123763fcfcc8a67d969e4aee2fed6fc19bfbb777bb1839ead4b88ed642",
)
def json_value(value: object) -> Json:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InputDataError("nonfinite JSON number")
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        result: Record = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InputDataError("JSON record keys must be strings")
            result[key] = json_value(item)
        return result
    raise InputDataError("unsupported JSON value")


def read_record(path: Path) -> Record:
    raw: object = json.loads(path.read_text(), object_pairs_hook=_unique_pairs)
    value = json_value(raw)
    return mapping(value)


def _unique_pairs(pairs: list[tuple[str, Json]]) -> Record:
    result: Record = {}
    for key, value in pairs:
        if key in result:
            raise InputDataError("duplicate JSON key")
        result[key] = value
    return result


def mapping(value: Json) -> Record:
    if not isinstance(value, dict):
        raise InputDataError("record required")
    return value


def number(value: Json) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise InputDataError("finite number required")
    return float(value)


def integer(value: Json) -> int:
    if type(value) is not int:
        raise InputDataError("integer required")
    return value


def text(value: Json) -> str:
    if not isinstance(value, str) or not value:
        raise InputDataError("nonempty string required")
    return value


def sequence(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise InputDataError("list required")
    return value


def anchor_count(sd: float) -> tuple[int, int]:
    if not math.isfinite(sd) or sd < 0:
        raise ContractViolation("finite nonnegative sample SD required")
    required = math.ceil((1.645 * 1.25 * sd / (X / 2)) ** 2)
    return required, max(32, required)
