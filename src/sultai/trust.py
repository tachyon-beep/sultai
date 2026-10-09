"""Operation-relative trust contracts; decorators preserve calls and exceptions.

T1 operations receive owned types satisfying their operation-specific value
preconditions. Violating those preconditions is a trusted-code fault. T2 has
the same owned type boundary, but declares recoverable domain failures. T3
receives unknown data and validates/coerces it before ownership. These markers
and the local gate record checked seams, not a proof of arbitrary invariants.
"""

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F")


class InputDataError(ValueError):
    """Untrusted data fails a boundary's type or value validation."""


class ContractViolation(ValueError):
    """Owned state or a trusted operation violates its required contract."""


class FitUnavailable(ValueError):
    """A valid conditioning problem cannot be solved at machine precision."""


def t3_boundary(*, test: str, fingerprint: str) -> Callable[[F], F]:
    """Bind an unknown-data seam to a reviewed test AST without wrapping it."""
    if "::" not in test or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ContractViolation("T3 requires a specific paired test and lowercase AST SHA-256")

    def decorate(function: F) -> F:
        if not callable(function):
            raise ContractViolation("trust decorators require a callable")
        return function

    return decorate


def t2_operation(*, invariants: str, failures: tuple[type[Exception], ...]) -> Callable[[F], F]:
    """Document owned-input preconditions and explicit recoverable failures."""
    if not invariants or not failures or any(not issubclass(failure, Exception) for failure in failures):
        raise ContractViolation("T2 requires operation invariants and explicit domain failures")

    def decorate(function: F) -> F:
        if not callable(function):
            raise ContractViolation("trust decorators require a callable")
        return function

    return decorate
