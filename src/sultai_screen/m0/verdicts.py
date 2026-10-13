"""M0 verdicts V0, V1, V2, labels, static reading and programme reading (preregistration v4 section 7).

Pure functions over already-computed numbers. No torch, no screen, no pilot.

V0 input records are boolean summaries the pilot builds, with these required keys:

* replays: ``both_cards_bitwise`` (bitwise replay on both cards against the pinned attempt-4 hashes)
* instrument: ``a1_within_1e-4``, ``b3_fraction_every_anchor`` (B3 >= RHO * A3 on every anchor)
* identities: ``zero_module_identity``, ``printed_counts`` (33h + 16 for every sliced module),
  ``complement_hash`` (unchanged under every re-tune arm), ``static_step0`` (host tensors bitwise the anchor's)
* slicing: ``known_answer``
* residuals: one |r_aff| per anchor, each at most FOLD_TOLERANCE
"""

from __future__ import annotations

import math

from sultai.trust import ContractViolation
from sultai_screen.contracts import Record
from sultai_screen.m0.identity import CERTIFIED_FRACTION, FOLD_TOLERANCE, M_S, X2, X

STATIC_MIN_N = 32


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ContractViolation(f"{name} must be finite")
    return value


def _fraction(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ContractViolation(f"{name} must be a finite fraction in [0, 1]")
    return value


def _flag(record: Record, record_name: str, key: str) -> bool:
    if key not in record:
        raise ContractViolation(f"{record_name} is missing {key!r}")
    value = record[key]
    if not isinstance(value, bool):
        raise ContractViolation(f"{record_name}[{key!r}] must be a bool")
    return value


def v0(
    replays: Record, instrument: Record, identities: Record, slicing: Record, residuals: tuple[float, ...]
) -> Record:
    checks = (
        ("replay", _flag(replays, "replays", "both_cards_bitwise")),
        ("A1", _flag(instrument, "instrument", "a1_within_1e-4")),
        ("B3", _flag(instrument, "instrument", "b3_fraction_every_anchor")),
        ("zero-module identity", _flag(identities, "identities", "zero_module_identity")),
        ("printed counts", _flag(identities, "identities", "printed_counts")),
        ("slicing known answer", _flag(slicing, "slicing", "known_answer")),
        ("complement hash", _flag(identities, "identities", "complement_hash")),
        ("S-static step 0", _flag(identities, "identities", "static_step0")),
    )
    if not residuals:
        raise ContractViolation("V0 needs at least one fold residual")
    fold_ok = all(abs(_finite("residual", r)) <= FOLD_TOLERANCE for r in residuals)
    failed = [name for name, ok in checks if not ok]
    if not fold_ok:
        failed.append("fold residual")
    return {"passed": not failed, "failed": list(failed)}


def v1(d1_lower: float, d1_upper: float, maff_certified_fraction: float, msq_certified_fraction: float) -> str:
    _finite("d1_lower", d1_lower)
    _finite("d1_upper", d1_upper)
    _fraction("maff_certified_fraction", maff_certified_fraction)
    _fraction("msq_certified_fraction", msq_certified_fraction)
    if d1_lower >= X and maff_certified_fraction >= CERTIFIED_FRACTION:
        return "present"
    if d1_upper < X and msq_certified_fraction >= CERTIFIED_FRACTION:
        return "re-tuning equivalent"
    return "indeterminate"


def labels(
    delta_acc_lower: float, d3_lower: float, d3_upper: float, d1prime_lower: float, d4_lower: float
) -> list[str]:
    for name, value in (
        ("delta_acc_lower", delta_acc_lower),
        ("d3_lower", d3_lower),
        ("d3_upper", d3_upper),
        ("d1prime_lower", d1prime_lower),
        ("d4_lower", d4_lower),
    ):
        _finite(name, value)
    result: list[str] = []
    if delta_acc_lower <= 0:
        result.append("loss gain without accuracy movement")
    if d3_lower >= X:
        result.append("survives finishing")
    if d3_upper < X:
        result.append("harvests undertraining")
    if d1prime_lower > 0:
        result.append("beats every named re-tune")
    if d4_lower > 0:
        result.append("fresh-data effect")
    return result


def v2(d2_lower: float, d2_upper: float, msmallb_certified: float, refit_certified: float) -> str:
    _finite("d2_lower", d2_lower)
    _finite("d2_upper", d2_upper)
    _fraction("msmallb_certified", msmallb_certified)
    _fraction("refit_certified", refit_certified)
    if d2_lower >= X2 and msmallb_certified >= CERTIFIED_FRACTION:
        return "helps"
    if d2_upper < 0 and refit_certified >= CERTIFIED_FRACTION:
        return "hurts"
    return "indeterminate"


def static_reading(d5_lower: float, in_confirmatory_set: bool, n: int) -> str | None:
    _finite("d5_lower", d5_lower)
    if in_confirmatory_set and n >= STATIC_MIN_N and d5_lower > M_S:
        return "design-in beats insertion"
    return None


def programme_reading(v1_verdict: str, label_list: list[str]) -> Record:
    beats = "beats every named re-tune" in label_list
    if v1_verdict == "present":
        return {
            "rank": 1,
            "outcome": "more capacity at a small parameter cost",
            "opens_m1": True,
            "beats_named_retunes": beats,
        }
    if v1_verdict == "re-tuning equivalent":
        return {
            "rank": 2,
            "outcome": "same loss with fewer retained parameters",
            "opens_m1": False,
            "beats_named_retunes": beats,
        }
    if v1_verdict == "indeterminate":
        return {"rank": 3, "outcome": "no substantial change", "opens_m1": False, "beats_named_retunes": beats}
    raise ContractViolation(f"unknown V1 verdict {v1_verdict!r}")
