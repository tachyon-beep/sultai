"""M0 seed registry (preregistration v4 Appendix A): one disjoint block per lineage, admission fails closed."""

from __future__ import annotations

from typing import Final, Literal

from sultai_screen.contracts import PilotHold

Lineage = Literal[
    "anchor",
    "frontier-0.25",
    "frontier-0.5",
    "frontier-0.75",
    "frontier-1.5",
    "cooled-20",
    "confirmatory",
    "restart",
]

BLOCKS: Final[dict[Lineage, range]] = {
    "anchor": range(3000, 3006),
    "frontier-0.5": range(6000, 6006),
    "frontier-0.75": range(6006, 6012),
    "frontier-1.5": range(6012, 6018),
    "cooled-20": range(6018, 6024),
    "frontier-0.25": range(6024, 6030),
    "confirmatory": range(7500, 7600),
    "restart": range(7700, 7800),
}
RESERVED_M2: Final = range(6100, 6200)

_WIDTH_SCALES: Final[dict[Lineage, float]] = {
    "anchor": 1.0,
    "cooled-20": 1.0,
    "confirmatory": 1.0,
    "restart": 1.0,
    "frontier-0.25": 0.25,
    "frontier-0.5": 0.5,
    "frontier-0.75": 0.75,
    "frontier-1.5": 1.5,
}


def admit(seed: int, lineage: Lineage) -> int:
    """Return the seed if it lies in the lineage's block; otherwise hold."""
    if seed in RESERVED_M2:
        raise PilotHold(f"seed {seed} is reserved for M2")
    if seed not in BLOCKS[lineage]:
        raise PilotHold(f"seed {seed} is outside the {lineage} block {BLOCKS[lineage]}")
    return seed


def width_scale(lineage: Lineage) -> float:
    """The width multiplier s of a lineage; confirmatory and restart draw reference-host (s = 1.0) anchors."""
    return _WIDTH_SCALES[lineage]
