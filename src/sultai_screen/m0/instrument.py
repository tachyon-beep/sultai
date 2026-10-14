"""P1-only carve-out of the Gate 3 instrument: materiality, post-hoc known answer and the hold check.

The mathematics mirrors screen.materiality_worker, screen.posthoc_worker and pilot.check_posthoc for P1 alone.
alpha is fixed at 0.1 (spec section 2); no ladder rung ever replaces it and it is not a pilot-fill field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import torch

from sultai_screen.contracts import InstrumentHold, PilotHold, Record, mapping, number
from sultai_screen.fit import identity_check, mean_loss, oracle, ridge
from sultai_screen.host import Cached, Features, Host
from sultai_screen.impair import make_impairment
from sultai_screen.m0.identity import RHO

P1_LADDER: Final = (0.1, 0.3, 0.03)
ALPHA: Final = 0.1
MATERIALITY_FLOOR: Final = 0.05
KNOWN_ANSWER_TOLERANCE: Final = 0.01
INVERSE_TOLERANCE: Final = 1e-4
ANCHORS: Final = 6


def materiality(host: Host, cache: Cached, seed: int) -> Record:
    try:
        identity_check(host, cache.development, seed)
    except PilotHold as error:
        # M1 (R21): an identity clause is an instrument failure in M0; Gate 3's identity_check is unchanged.
        raise InstrumentHold(str(error)) from error
    original = mean_loss(host, cache.development, None)
    gains: Record = {}
    for magnitude in P1_LADDER:
        operator = make_impairment("P1", magnitude, seed, cache.conditioning.read)
        data = operator.features(cache.development)
        no_op = mean_loss(host, data, None)
        restored = Features(data.read, operator.undo(data.base), data.labels)
        inverse_loss = mean_loss(host, restored, None)
        if abs(inverse_loss - original) > INVERSE_TOLERANCE:
            # D1 (R21): at alpha = 0.1 this is A1 itself, and on every rung an instrument clause, never execution.
            raise InstrumentHold(f"exact inverse failed 1e-4 nats: P1/{seed}")
        gains[str(magnitude)] = no_op - inverse_loss
    return {"seed": seed, "intact_loss": original, "rungs": {"P1": gains}}


def _distinct_seeds(records: tuple[Record, ...]) -> None:
    if len({number(record["seed"]) for record in records}) != ANCHORS or len(records) != ANCHORS:
        raise PilotHold("six distinct development anchors required")


def materiality_report(records: tuple[Record, ...]) -> Record:
    """Descriptive only: per-rung mean gain and whether every anchor clears the floor. Never chooses an alpha."""
    _distinct_seeds(records)
    rungs: Record = {}
    for magnitude in P1_LADDER:
        gains = [number(mapping(mapping(record["rungs"])["P1"])[str(magnitude)]) for record in records]
        rungs[str(magnitude)] = {
            "mean_gain": sum(gains) / len(gains),
            "clears": all(gain >= MATERIALITY_FLOOR for gain in gains),
        }
    return {"seeds": [number(record["seed"]) for record in records], "floor": MATERIALITY_FLOOR, "rungs": {"P1": rungs}}


def posthoc(host: Host, cache: Cached, seed: int, output: Path) -> Record:
    operator = make_impairment("P1", ALPHA, seed, cache.conditioning.read)
    conditioning = operator.features(cache.conditioning)
    selection = operator.features(cache.selection)
    development = operator.features(cache.development)
    intact_loss = mean_loss(host, cache.development, None)
    a1 = mean_loss(host, Features(development.read, operator.undo(development.base), development.labels), None)
    a1_drift = abs(a1 - intact_loss)
    if a1_drift > INVERSE_TOLERANCE:
        # M1 (R21): A1 drift is the V0 A1 clause, an instrument failure.
        raise InstrumentHold("exact inverse endpoint drift")
    a2 = ridge(conditioning.read, conditioning.base, cache.conditioning.read, nonlinear=True)
    a3 = ridge(conditioning.read, conditioning.base, cache.conditioning.read, nonlinear=False)
    noop = mean_loss(host, development, None)
    fit = oracle(host, conditioning, selection, seed)
    b3 = mean_loss(host, development, fit.best.adapter)
    arms: Record = {
        "A1": a1,
        "A2": mean_loss(host, development, a2),
        "A3": mean_loss(host, development, a3),
        "B3": b3,
    }
    a3_loss = number(arms["A3"])
    torch.save(
        {"weight": fit.best.adapter.weights[0], "bias": fit.best.adapter.weights[1]},
        output / "p1-oracle.pt",
    )
    return {
        "seed": seed,
        "alpha": ALPHA,
        "noop_loss": noop,
        "a1_drift": a1_drift,
        "arm_losses": arms,
        "oracle": fit.report(),
        "oracle_gain_fraction_of_A3": (noop - b3) / (noop - a3_loss) if noop > a3_loss else None,
        "rho_required": RHO,
    }


def check_instrument(records: tuple[Record, ...]) -> None:
    _distinct_seeds(records)
    for record in records:
        if "alpha" not in record or record["alpha"] != ALPHA:
            raise PilotHold(f"the P1 known answer is read at alpha {ALPHA} only (§2)")
        arms = mapping(record["arm_losses"])
        a1 = number(arms["A1"])
        if number(record["noop_loss"]) - a1 < MATERIALITY_FLOOR:
            raise InstrumentHold("pilot materiality failed: P1 at alpha 0.1")
        if number(arms["A3"]) - a1 > KNOWN_ANSWER_TOLERANCE:
            raise InstrumentHold("supervised known answer failed")
        fraction = record["oracle_gain_fraction_of_A3"]
        if fraction is None or number(fraction) < RHO:
            raise InstrumentHold("B3 did not recover rho of A3 on every development anchor")
