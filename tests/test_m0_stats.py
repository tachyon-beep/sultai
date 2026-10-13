from __future__ import annotations

import math
import statistics
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from sultai.trust import ContractViolation
from sultai_screen.contracts import mapping
from sultai_screen.data import generator
from sultai_screen.fit import paired_se
from sultai_screen.host import Host, parameter_count
from sultai_screen.m0.stats import (
    CORE_ARMS,
    CORE_PARAMETERS,
    REFERENCE_PARAMETERS,
    AnchorResult,
    ArmResult,
    achieved_half_width,
    anchor_bound,
    confirmatory_count,
    frontier_slope,
    frontier_slope_descriptive,
    gain_fraction,
    median,
    paired_bound,
    quantities,
    sd,
    sign_test_bound,
)

SRC = Path(__file__).resolve().parents[1] / "src"
SPREAD = torch.tensor([-0.1, 0.1, -0.05, 0.05], dtype=torch.float64)
SAMPLE = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
# Pinned once from the implementation (seed 0) and re-derived below by an independent loop.
SAMPLE_LOWER = 0.02
SAMPLE_UPPER = 0.05
OPTIONAL_GAINS = {"R-all": 0.08, "M-exp": 0.045, "lottery": 0.035, "S-static": 0.07, "S-static:refit": 0.075}


def correct(hits: int) -> torch.Tensor:
    return torch.tensor([1.0] * hits + [0.0] * (4 - hits))


def arm(noop: float, gain: float, hits: int = 2) -> ArmResult:
    return ArmResult(noop - gain + SPREAD, correct(hits), True, 0.3, False)


def default_gains() -> dict[str, float]:
    gains = dict.fromkeys(CORE_ARMS, 0.01)
    gains.update(OPTIONAL_GAINS)
    return gains


def anchor(
    gains: dict[str, float],
    *,
    noop: float = 2.0,
    tcool_noop: float = 1.9,
    tcool_module: float = 1.86,
    hits: dict[str, int] | None = None,
    raw: dict[str, tuple[tuple[float, float], ...]] | None = None,
) -> AnchorResult:
    accuracy = hits or {}
    arms = {name: arm(noop, gain, accuracy[name] if name in accuracy else 2) for name, gain in gains.items()}
    if "M-sq|T-cool" in arms:
        arms["M-sq|T-cool"] = ArmResult(tcool_module + SPREAD, correct(2), True, 0.3, False)
    return AnchorResult(
        arms=arms,
        noop_loss=noop + SPREAD,
        noop_correct=correct(2),
        tcool_noop_loss=tcool_noop + SPREAD,
        fold_residual=0.0012,
        survivors={"M-wide→8:rank": (1, 5, 9, 12, 20, 33, 40, 63)},
        unfolded={"M-wide→8:rank": noop - 0.004 + SPREAD},
        raw_candidates=raw or {},
    )


class StatsTests(unittest.TestCase):
    def test_d1_on_constructed_record(self) -> None:
        gains = default_gains()
        gains.update({"M-sq": 0.05, "M-aff": 0.02})
        self.assertAlmostEqual(float(quantities(anchor(gains))["D1"]), 0.03, places=12)

    def test_d1prime_is_per_anchor_maximum(self) -> None:
        first, second = default_gains(), default_gains()
        first.update({"M-sq": 0.05, "R-272s": 0.01, "R-aff": 0.03})
        second.update({"M-sq": 0.05, "R-272s": 0.04, "R-aff": 0.02})
        values = [float(quantities(anchor(g))["D1prime"]) for g in (first, second)]
        self.assertAlmostEqual(values[0], 0.02, places=12)
        self.assertAlmostEqual(values[1], 0.01, places=12)
        # The aggregate-first reading would give 0.05 - max(0.025, 0.025) = 0.025 for both anchors.
        self.assertNotAlmostEqual(statistics.median(values), 0.025, places=6)

    def test_d3_against_finished_host_noop_not_anchor_noop(self) -> None:
        gains = default_gains()
        gains["T-cool"] = 0.07
        record = quantities(anchor(gains, noop=2.0, tcool_noop=1.9, tcool_module=1.86))
        self.assertAlmostEqual(float(record["D3"]), 0.04, places=12)
        self.assertAlmostEqual(float(record["G_tcool"]), 0.07, places=12)

    def test_every_quantity_on_constructed_record(self) -> None:
        gains = default_gains()
        gains.update(
            {
                "M-sq": 0.05,
                "M-aff": 0.02,
                "R-272s": 0.01,
                "R-aff": 0.03,
                "M-sq-fresh": 0.065,
                "M-wide": 0.06,
                "M-wide→8:rank:refit": 0.045,
                "M-wide→8:rank:sliced": 0.03,
                "M-wide→8:random:refit": 0.04,
                "M-wide→8:random:sliced": 0.015,
                "M-wide→16:sliced": 0.048,
                "M-wide→4:sliced": 0.012,
                "M-wide→2:sliced": 0.006,
                "M-wide→16:refit": 0.054,
                "M-wide→4:refit": 0.036,
                "M-wide→2:refit": 0.018,
                "M-wide→8:ablation": 0.042,
                "M-small": 0.025,
                "M-small-B": 0.035,
                "R-temp": 0.004,
                "R-cal": 0.006,
                "R-all": 0.1,
                "M-exp": 0.045,
                "lottery": 0.035,
                "S-static": 0.07,
                "S-static:refit": 0.075,
            }
        )
        record = quantities(anchor(gains, hits={"M-sq": 3}))
        expected = {
            "D2": 0.045 - 0.035,
            "D2s": 0.03 - 0.025,
            "D2_random": 0.04 - 0.035,
            "D2s_random": 0.015 - 0.025,
            "D4": 0.065 - 0.05,
            "D5": (2.0 - 0.05) - (2.0 - 0.075),
            "D5_norefit": (2.0 - 0.05) - (2.0 - 0.07),
            "r_aff": 0.0012,
            "delta_acc": 0.25,
            "G_rtemp": 0.004,
            "G_rcal": 0.006,
            "G_mexp": 0.045,
            "G_lottery": 0.035,
            "retune_fraction": 0.5,
            "leave_one_out": 0.042,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(float(record[key]), value, places=12, msg=key)
            self.assertIsNone(record[f"{key}_reason"] if f"{key}_reason" in record else None, msg=key)
        survival = mapping(record["survival"])
        sliced, refit = mapping(survival["sliced"]), mapping(survival["refit"])
        for h, value in {"16": 0.8, "8": 0.5, "4": 0.2, "2": 0.1}.items():
            self.assertAlmostEqual(float(sliced[h]), value, places=12, msg=h)
        for h, value in {"16": 0.9, "8": 0.75, "4": 0.6, "2": 0.3}.items():
            self.assertAlmostEqual(float(refit[h]), value, places=12, msg=h)
        unfolded = mapping(mapping(record["unfolded"])["M-wide→8:rank"])
        self.assertAlmostEqual(float(unfolded["loss"]), 1.996, places=12)
        self.assertAlmostEqual(float(unfolded["gain"]), 0.004, places=12)
        self.assertEqual(mapping(record["survivors"])["M-wide→8:rank"], [1, 5, 9, 12, 20, 33, 40, 63])
        self.assertEqual(record["harm"], [])

    def test_retune_fraction_and_survival_none_on_zero_denominator(self) -> None:
        gains = default_gains()
        gains.update({"R-all": 0.0, "M-wide": 0.0})
        record = quantities(anchor(gains))
        self.assertIsNone(record["retune_fraction"])
        self.assertEqual(record["retune_fraction_reason"], "denominator ≤ 0")
        self.assertIsNone(record["survival"])
        self.assertEqual(record["survival_reason"], "denominator ≤ 0")

    def test_missing_optional_arms_are_none_with_reason(self) -> None:
        gains = dict.fromkeys(CORE_ARMS, 0.01)
        record = quantities(anchor(gains))
        for key, arm_name in (
            ("retune_fraction", "R-all"),
            ("D5", "S-static:refit"),
            ("D5_norefit", "S-static"),
            ("G_mexp", "M-exp"),
            ("G_lottery", "lottery"),
        ):
            self.assertIsNone(record[key], msg=key)
            self.assertEqual(record[f"{key}_reason"], f"{arm_name} not run", msg=key)

    def test_missing_core_arm_and_misaligned_vector_violate(self) -> None:
        gains = default_gains()
        del gains["M-aff"]
        with self.assertRaises(ContractViolation):
            quantities(anchor(gains))
        with self.assertRaises(ContractViolation):
            AnchorResult(
                arms={"M-sq": ArmResult(torch.zeros(3), torch.zeros(3), None, None, None)},
                noop_loss=torch.zeros(4),
                noop_correct=torch.zeros(4),
                tcool_noop_loss=torch.zeros(4),
                fold_residual=0.0,
                survivors={},
                unfolded={},
                raw_candidates={},
            )

    def test_harm_lists_candidates_beyond_harm_margin(self) -> None:
        raw = {
            "M-sq": ((1.0, 2.011), (0.3, 2.009), (0.1, 1.95)),
            "R-aff": ((0.01, 2.011), (0.1, 1.99)),
        }
        harm = quantities(anchor(default_gains(), raw=raw))["harm"]
        assert isinstance(harm, list)
        self.assertEqual([[entry[0], entry[1]] for entry in map(list, harm)], [["M-sq", 1.0], ["R-aff", 0.01]])
        for entry in harm:
            assert isinstance(entry, list)
            self.assertAlmostEqual(float(entry[2]), 0.011, places=9)

    def test_paired_bound_is_normal_bound_on_paired_se(self) -> None:
        difference = torch.randn(200, generator=torch.Generator().manual_seed(11), dtype=torch.float64) * 0.1 + 0.02
        se = paired_se(difference, 7)
        mean = float(difference.mean())
        self.assertAlmostEqual(paired_bound(difference, 7, lower=True), mean - 1.645 * se, delta=1e-9)
        self.assertAlmostEqual(paired_bound(difference, 7, lower=False), mean + 1.645 * se, delta=1e-9)

    def test_median_and_sd_conventions(self) -> None:
        self.assertEqual(median(SAMPLE), statistics.median(SAMPLE))
        self.assertAlmostEqual(median(SAMPLE), 0.035, places=15)
        self.assertEqual(sd(SAMPLE), statistics.stdev(SAMPLE))

    def test_anchor_bound_identical_values(self) -> None:
        values = (0.03,) * 6
        self.assertEqual(anchor_bound(values, 0, lower=True), 0.03)
        self.assertEqual(anchor_bound(values, 0, lower=False), 0.03)

    def test_anchor_bound_pinned_and_rederived(self) -> None:
        lower, upper = anchor_bound(SAMPLE, 0, lower=True), anchor_bound(SAMPLE, 0, lower=False)
        self.assertEqual(lower, SAMPLE_LOWER)
        self.assertEqual(upper, SAMPLE_UPPER)
        # Independent derivation: one (10000, 6) draw, an explicit loop of statistics.median, 1-based 500th / 9500th.
        rng = generator(0, "anchor-bootstrap")
        rows = torch.randint(len(SAMPLE), (10_000, len(SAMPLE)), generator=rng).tolist()
        medians = []
        for row in rows:
            medians.append(statistics.median([SAMPLE[i] for i in row]))
        medians.sort()
        self.assertEqual(medians[499], SAMPLE_LOWER)
        self.assertEqual(medians[9499], SAMPLE_UPPER)

    def test_achieved_half_width(self) -> None:
        expected = max(0.035 - SAMPLE_LOWER, SAMPLE_UPPER - 0.035)
        self.assertAlmostEqual(achieved_half_width(SAMPLE, 0), expected, places=15)

    def test_sign_test_bound_order_statistics(self) -> None:
        six = (0.05, 0.01, 0.06, 0.03, 0.02, 0.04)
        self.assertEqual(sign_test_bound(six, lower=True), (0.01, None))
        self.assertEqual(sign_test_bound(six, lower=False), (0.06, None))
        ordered = [(i + 1) / 100 for i in range(32)]
        shuffled = tuple(ordered[(7 * i) % 32] for i in range(32))
        self.assertEqual(sign_test_bound(shuffled, lower=True), (ordered[10], None))
        self.assertEqual(sign_test_bound(shuffled, lower=False), (ordered[21], None))
        self.assertEqual(sign_test_bound(six[:5], lower=True), (None, "n < 6"))

    def test_sign_test_zero_counts_against(self) -> None:
        # Dropping the zero would leave n = 6 and a lower bound of 0.01 > 0; counted, it bounds at 0.
        self.assertEqual(sign_test_bound((0.03, 0.0, 0.01, 0.06, 0.02, 0.05, 0.04), lower=True), (0.0, None))
        self.assertEqual(sign_test_bound((-0.03, 0.0, -0.01, -0.06, -0.02, -0.05, -0.04), lower=False), (0.0, None))

    def test_confirmatory_count(self) -> None:
        self.assertEqual(confirmatory_count(0.0275), (32, 32))
        self.assertEqual(confirmatory_count(0.03), (39, 39))

    def test_reference_parameters(self) -> None:
        self.assertEqual(REFERENCE_PARAMETERS, parameter_count(Host()))
        self.assertEqual(CORE_PARAMETERS, (85_274, 284_634, 613_018))

    def test_frontier_slope_on_exact_line(self) -> None:
        slope = frontier_slope(line_points(), 0)
        self.assertAlmostEqual(slope.per_doubling, 0.1, places=12)
        self.assertEqual(slope.lower, slope.upper)
        self.assertAlmostEqual(slope.lower, 0.1, places=12)
        self.assertEqual(slope.report()["label"], "same-recipe frontier, 10 epochs")

    def test_gain_fraction_on_exact_line(self) -> None:
        fraction = gain_fraction((0.02,) * 6, line_points(), 0)
        assert fraction.value is not None and fraction.lower is not None and fraction.upper is not None
        self.assertAlmostEqual(fraction.value, 0.2, places=12)
        self.assertAlmostEqual(fraction.lower, 0.2, places=12)
        self.assertAlmostEqual(fraction.upper, 0.2, places=12)
        self.assertIsNone(fraction.reason)

    def test_gain_fraction_draws_reference_seeds_jointly(self) -> None:
        # Gains built so that mean(gain) = c * slope(mean reference loss) for every resample of the reference seeds:
        # under a joint draw every replicate fraction is c; under independent draws the interval would open.
        x = [math.log2(p) for p in CORE_PARAMETERS]
        centre = sum(x) / 3
        sxx = sum((v - centre) ** 2 for v in x)
        weights = [(v - centre) / sxx for v in x]
        losses = [2.0 - 0.1 * (v - x[0]) for v in x]
        reference = tuple(losses[1] + d for d in (-0.03, 0.02, 0.05, -0.01, 0.0, -0.04))
        constant = -(weights[0] * losses[0] + weights[2] * losses[2])
        c = 0.3
        gains = tuple(c * (constant - weights[1] * value) for value in reference)
        points = {
            CORE_PARAMETERS[0]: (losses[0],) * 6,
            CORE_PARAMETERS[1]: reference,
            CORE_PARAMETERS[2]: (losses[2],) * 6,
        }
        fraction = gain_fraction(gains, points, 4)
        assert fraction.value is not None and fraction.lower is not None and fraction.upper is not None
        self.assertAlmostEqual(fraction.lower, c, places=9)
        self.assertAlmostEqual(fraction.upper, c, places=9)
        self.assertAlmostEqual(fraction.value, c, places=9)

    def test_gain_fraction_none_when_slope_not_positive(self) -> None:
        flat = dict.fromkeys(CORE_PARAMETERS, (2.0,) * 6)
        fraction = gain_fraction((0.02,) * 6, flat, 0)
        self.assertIsNone(fraction.value)
        self.assertEqual(fraction.reason, "slope ≤ 0")
        self.assertIsNone(fraction.report()["value"])

    def test_frontier_slope_requires_core_widths(self) -> None:
        points = line_points()
        del points[CORE_PARAMETERS[2]]
        with self.assertRaises(ContractViolation):
            frontier_slope(points, 0)

    def test_frontier_slope_descriptive(self) -> None:
        extra = 40_000.0
        points = line_points()
        points[extra] = (2.0 - 0.1 * (math.log2(extra) - math.log2(CORE_PARAMETERS[0])),) * 6
        slope = frontier_slope_descriptive(points, 0)
        self.assertAlmostEqual(slope.per_doubling, 0.1, places=12)
        self.assertEqual(slope.report()["label"], "same-recipe frontier, 10 epochs, descriptive")
        self.assertEqual(slope.report()["widths"], sorted(points))
        two = {p: v for p, v in line_points().items() if p != CORE_PARAMETERS[1]}
        self.assertAlmostEqual(frontier_slope_descriptive(two, 0).per_doubling, 0.1, places=12)
        with self.assertRaises(ContractViolation):
            frontier_slope_descriptive({CORE_PARAMETERS[0]: (2.0,) * 6}, 0)

    def test_import_isolation(self) -> None:
        code = (
            "import sys, sultai_screen.m0.stats; "
            "banned = ('screen', 'pilot', 'former_bridge', 'alignment'); "
            "sys.exit(1 if any(f'sultai_screen.{m}' in sys.modules for m in banned) else 0)"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], env={"PYTHONPATH": str(SRC)}, check=False)
        self.assertEqual(result.returncode, 0)


def line_points() -> dict[float, tuple[float, ...]]:
    base = math.log2(CORE_PARAMETERS[0])
    return {float(p): (2.0 - 0.1 * (math.log2(p) - base),) * 6 for p in CORE_PARAMETERS}


if __name__ == "__main__":
    unittest.main()
