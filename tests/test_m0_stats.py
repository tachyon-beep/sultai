from __future__ import annotations

import math
import statistics
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sultai.trust import ContractViolation
from sultai_screen import contracts
from sultai_screen.contracts import mapping, number
from sultai_screen.data import generator
from sultai_screen.fit import paired_se
from sultai_screen.host import Host, parameter_count
from sultai_screen.m0 import identity
from sultai_screen.m0.stats import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CONFIRMATORY_ARMS,
    CORE_ARMS,
    CORE_PARAMETERS,
    PAIRED_SE_KEYS,
    PILOT_ARMS,
    REFERENCE_PARAMETERS,
    STARVED_PAIR_ARMS,
    SURVIVAL_WIDTHS,
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
    survival_value,
)
from sultai_screen.m0.verdicts import labels, programme_reading, static_reading, v0, v1, v2

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

    def test_per_anchor_paired_bootstrap_se_for_every_two_arm_quantity(self) -> None:
        # A4 (§6): every two-arm quantity carries its per-example paired-bootstrap SE for loss (`<key>_se`) and, as a
        # secondary, for 0/1 correctness (`<key>_acc_se`), each fit.paired_se under the fixed seed BOOTSTRAP_SEED.
        gains = default_gains()
        gains.update({"M-sq": 0.05, "M-aff": 0.02, "R-272s": 0.01, "R-aff": 0.03})
        result = anchor(gains)
        sq_loss = torch.tensor([1.90, 2.05, 1.93, 2.01], dtype=torch.float64)
        aff_loss = torch.tensor([1.97, 2.02, 1.99, 1.98], dtype=torch.float64)
        sq_correct, aff_correct = torch.tensor([1.0, 1.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 0.0, 1.0])
        result.arms["M-sq"] = ArmResult(sq_loss, sq_correct, True, 0.3, False)
        result.arms["M-aff"] = ArmResult(aff_loss, aff_correct, True, 0.3, False)
        record = quantities(result)
        self.assertEqual(BOOTSTRAP_SEED, 3000)
        # D1 = G(M-sq) - G(M-aff): per example, L(M-aff) - L(M-sq); accuracy, acc(M-sq) - acc(M-aff).
        self.assertEqual(record["D1_se"], paired_se(aff_loss - sq_loss, 3000))
        self.assertEqual(record["D1_acc_se"], paired_se(sq_correct - aff_correct, 3000))
        self.assertGreater(float(record["D1_se"]), 0.0)
        self.assertGreater(float(record["D1_acc_se"]), 0.0)
        # D1': against the per-anchor maximum re-tune arm (R-aff here, gain 0.03 > R-272s 0.01).
        self.assertEqual(record["D1prime_se"], paired_se(result.arms["R-aff"].loss - sq_loss, 3000))
        # D3: against the T-cool host's own N0 (accuracy: the T-cool arm, which is that N0 evaluated).
        tcool = result.arms["M-sq|T-cool"]
        self.assertEqual(record["D3_se"], paired_se(result.tcool_noop_loss - tcool.loss, 3000))
        self.assertEqual(record["D3_acc_se"], paired_se(tcool.correct - result.arms["T-cool"].correct, 3000))
        self.assertEqual(
            PAIRED_SE_KEYS, ("D1", "D1prime", "D2", "D2s", "D2_random", "D2s_random", "D3", "D4", "D5", "D5_norefit")
        )
        for key in PAIRED_SE_KEYS:
            self.assertIsInstance(record[f"{key}_se"], float, key)
            self.assertIsInstance(record[f"{key}_acc_se"], float, key)
        # An absent arm: the quantity, its SE and its accuracy SE are all None.
        reduced = quantities(anchor(dict.fromkeys(CORE_ARMS, 0.01)))
        for key in ("D5", "D5_norefit"):
            self.assertIsNone(reduced[key])
            self.assertIsNone(reduced[f"{key}_se"])
            self.assertIsNone(reduced[f"{key}_acc_se"])

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

    def test_confirmatory_record_reports_absent_pilot_arms_with_reason(self) -> None:
        gains = dict.fromkeys(CONFIRMATORY_ARMS, 0.01)
        gains.update({"M-sq": 0.05, "M-aff": 0.02, "M-wide→8:rank:refit": 0.045, "M-small-B": 0.035})
        record = quantities(anchor(gains), CONFIRMATORY_ARMS)
        self.assertAlmostEqual(float(record["D1"]), 0.03, places=12)
        self.assertAlmostEqual(float(record["D1prime"]), 0.04, places=12)
        self.assertAlmostEqual(float(record["D2"]), 0.01, places=12)
        self.assertAlmostEqual(float(record["D3"]), 0.04, places=12)
        for key in ("D1", "D1prime", "D2", "D2s", "D2_random", "D2s_random", "D3", "G_tcool", "r_aff", "G_rcal"):
            self.assertIsNone(record[f"{key}_reason"], msg=key)
        # Amendment 3 (A3.1): the h = 16/4/2 arms are in the §10 set, so the whole curve is defined at confirmatory n.
        curve = mapping(record["survival"])
        for kind in ("sliced", "refit"):
            self.assertEqual(set(mapping(curve[kind])), {"16", "8", "4", "2"}, kind)
        self.assertEqual(record["survival_absent"], [])
        for key, arm_name in (
            ("D4", "M-sq-fresh"),
            ("leave_one_out", "M-wide→8:ablation"),
            ("retune_fraction", "R-all"),
            ("D5", "S-static:refit"),
            ("G_mexp", "M-exp"),
        ):
            self.assertIsNone(record[key], msg=key)
            self.assertEqual(record[f"{key}_reason"], f"{arm_name} not run", msg=key)
        with self.assertRaises(ContractViolation):
            quantities(anchor(gains), PILOT_ARMS)
        del gains["M-wide"]
        with self.assertRaises(ContractViolation):
            quantities(anchor(gains), CONFIRMATORY_ARMS)

    def test_survival_curve_reports_every_width_that_ran_and_names_the_absent(self) -> None:
        # Amendment 3 (A3.2): a partial curve is not undefined; the absent arms are named, in order.
        gains = default_gains()
        gains.update({"M-wide": 0.05, "M-wide→4:sliced": 0.01})
        for name in ("M-wide→16:sliced", "M-wide→16:refit", "M-wide→4:refit"):
            del gains[name]
        required = PILOT_ARMS - {"M-wide→16:sliced", "M-wide→16:refit", "M-wide→4:refit"}
        record = quantities(anchor(gains), required)
        curve = mapping(record["survival"])
        self.assertEqual(set(mapping(curve["sliced"])), {"8", "4", "2"})
        self.assertEqual(set(mapping(curve["refit"])), {"8", "2"})
        self.assertAlmostEqual(float(mapping(curve["sliced"])["4"]), 0.2, places=12)
        self.assertIsNone(record["survival_reason"])
        self.assertEqual(record["survival_absent"], ["M-wide→16:sliced", "M-wide→16:refit", "M-wide→4:refit"])
        self.assertAlmostEqual(number(survival_value(record, "sliced", 4)), 0.2, places=12)
        self.assertIsNone(survival_value(record, "refit", 4))
        undefined = quantities(anchor({**default_gains(), "M-wide": 0.0}))
        self.assertIsNone(survival_value(undefined, "sliced", 8))
        full = quantities(anchor(default_gains()))
        self.assertEqual(full["survival_absent"], [])

    def test_starved_pair_record_reports_d1_and_d3(self) -> None:
        gains = dict.fromkeys(STARVED_PAIR_ARMS, 0.01)
        gains.update({"M-sq": 0.05, "M-aff": 0.02, "T-cool": 0.07})
        record = quantities(anchor(gains, hits={"M-sq": 3}), STARVED_PAIR_ARMS)
        self.assertAlmostEqual(float(record["D1"]), 0.03, places=12)
        self.assertAlmostEqual(float(record["D3"]), 0.04, places=12)
        self.assertAlmostEqual(float(record["G_tcool"]), 0.07, places=12)
        self.assertAlmostEqual(float(record["delta_acc"]), 0.25, places=12)
        self.assertAlmostEqual(float(record["r_aff"]), 0.0012, places=12)
        for key, arm_name in (
            ("D1prime", "R-272s"),
            ("D2", "M-wide→8:rank:refit"),
            ("D2s", "M-wide→8:rank:sliced"),
            ("D4", "M-sq-fresh"),
            ("survival", "M-wide"),
            ("leave_one_out", "M-wide→8:ablation"),
        ):
            self.assertIsNone(record[key], msg=key)
            self.assertEqual(record[f"{key}_reason"], f"{arm_name} not run", msg=key)

    def test_stage_arm_sets(self) -> None:
        self.assertEqual(PILOT_ARMS, frozenset(CORE_ARMS))
        self.assertLess(CONFIRMATORY_ARMS, PILOT_ARMS)
        self.assertLess(STARVED_PAIR_ARMS, CONFIRMATORY_ARMS)
        # Amendment 3 (A3.1): only the fresh-data arm and the leave-one-out ceiling stay pilot-only.
        self.assertEqual(PILOT_ARMS - CONFIRMATORY_ARMS, {"M-sq-fresh", "M-wide→8:ablation"})
        self.assertLessEqual({f"M-wide→{h}:{k}" for h in (16, 4, 2) for k in ("sliced", "refit")}, CONFIRMATORY_ARMS)
        with self.assertRaises(ContractViolation):
            quantities(anchor(default_gains()), frozenset({"M-sq", "not-an-arm"}))
        # G3: the survival curve's widths are §5's 16/8/4/2.
        self.assertEqual(SURVIVAL_WIDTHS, (16, 8, 4, 2))

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

    def test_anchor_bounds_are_the_500th_and_9500th_of_one_10000_by_n_draw(self) -> None:
        # C4: the plan (g) convention, 10,000 resamples in one (samples, n) draw under generator(seed,
        # "anchor-bootstrap") and the 1-based 500th / 9,500th order statistics. A bootstrap MEDIAN cannot have distinct
        # neighbours at those ranks (an odd-n median is a sample element; an even-n one is mostly the mean of two
        # adjacent order statistics, so each value repeats far more than once in 10,000), so the ranks and the draw
        # are pinned with the statistic replaced by the resample mean, whose replicates are all distinct.
        self.assertEqual(BOOTSTRAP_SAMPLES, 10_000)
        values = [math.sqrt(p) for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)]
        data = torch.tensor(values, dtype=torch.float64)
        draw = torch.randint(len(values), (10_000, len(values)), generator=generator(7, "anchor-bootstrap"))
        replicates = sorted(float(v) for v in data[draw].mean(dim=1))
        with patch("sultai_screen.m0.stats._row_medians", lambda rows: rows.mean(dim=1)):
            lower = anchor_bound(values, 7, lower=True)
            upper = anchor_bound(values, 7, lower=False)
            self.assertEqual(anchor_bound(values, 7, lower=True, samples=10_000), lower)
        self.assertEqual(lower, replicates[499])  # the 500th smallest, 1-based
        self.assertEqual(upper, replicates[9499])  # the 9,500th smallest
        self.assertEqual(len({replicates[498], replicates[499], replicates[500]}), 3)
        self.assertEqual(len({replicates[9498], replicates[9499], replicates[9500]}), 3)
        # And with the real median: the default sample count is 10,000.
        self.assertEqual(anchor_bound(SAMPLE, 0, lower=True), anchor_bound(SAMPLE, 0, lower=True, samples=10_000))

    def test_frontier_bounds_are_the_500th_and_9500th_replicate_slopes(self) -> None:
        # C4: the frontier bootstrap with seed spread at every core width: one (10,000, m) draw per width in ascending
        # parameter order from generator(seed, "frontier-bootstrap"); the 500th / 9,500th replicate slopes.
        spread = (0.0, 0.0071, -0.0043, 0.0122, -0.0098, 0.0031)
        points = {p: tuple(1.5 - 0.1 * k + d * (1 + k) for d in spread) for k, p in enumerate(CORE_PARAMETERS)}
        result = frontier_slope(points, 11)
        rng = generator(11, "frontier-bootstrap")
        x = [math.log2(p) for p in CORE_PARAMETERS]
        centre = sum(x) / 3
        means = []
        for p in CORE_PARAMETERS:
            seeds = torch.tensor(points[p], dtype=torch.float64)
            means.append(seeds[torch.randint(6, (10_000, 6), generator=rng)].mean(dim=1).tolist())
        slopes = sorted(
            -sum((xi - centre) * m[r] for xi, m in zip(x, means, strict=True)) / sum((xi - centre) ** 2 for xi in x)
            for r in range(10_000)
        )
        self.assertAlmostEqual(result.lower, slopes[499], places=12)
        self.assertAlmostEqual(result.upper, slopes[9499], places=12)
        for rank in (499, 9499):
            self.assertGreater(slopes[rank] - slopes[rank - 1], 1e-10)
            self.assertGreater(slopes[rank + 1] - slopes[rank], 1e-10)
        self.assertEqual(result.samples, 10_000)

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
        # M11: the count is contracts.anchor_count, which reads Gate 3's X; it must be M0's frozen X.
        self.assertEqual(contracts.X, identity.X)

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


class VerdictTests(unittest.TestCase):
    def v0_inputs(self) -> tuple[dict[str, bool], dict[str, bool], dict[str, bool], dict[str, bool]]:
        return (
            {"both_cards_bitwise": True},
            {"a1_within_1e-4": True, "b3_fraction_every_anchor": True},
            {"zero_module_identity": True, "printed_counts": True, "complement_hash": True, "static_step0": True},
            {"known_answer": True},
        )

    def test_v0_passes_and_fold_residual_boundary(self) -> None:
        r, i, d, s = self.v0_inputs()
        self.assertEqual(v0(r, i, d, s, (0.005, -0.005)), {"passed": True, "failed": []})
        failed = v0(r, i, d, s, (0.001, 0.006))
        self.assertEqual(failed, {"passed": False, "failed": ["fold residual"]})

    def test_v0_each_clause_named(self) -> None:
        cases = (
            ("replays", "both_cards_bitwise", "replay"),
            ("instrument", "a1_within_1e-4", "A1"),
            ("instrument", "b3_fraction_every_anchor", "B3"),
            ("identities", "zero_module_identity", "zero-module identity"),
            ("identities", "printed_counts", "printed counts"),
            ("slicing", "known_answer", "slicing known answer"),
            ("identities", "complement_hash", "complement hash"),
            ("identities", "static_step0", "S-static step 0"),
        )
        for group, key, name in cases:
            with self.subTest(name=name):
                parts = dict(zip(("replays", "instrument", "identities", "slicing"), self.v0_inputs(), strict=True))
                parts[group][key] = False
                result = v0(parts["replays"], parts["instrument"], parts["identities"], parts["slicing"], (0.0,))
                self.assertEqual(result, {"passed": False, "failed": [name]})

    def test_v0_contract(self) -> None:
        r, i, d, s = self.v0_inputs()
        del d["complement_hash"]
        with self.assertRaises(ContractViolation):
            v0(r, i, d, s, (0.0,))
        r, i, d, s = self.v0_inputs()
        with self.assertRaises(ContractViolation):
            v0(r, i, d, s, (math.nan,))
        with self.assertRaises(ContractViolation):
            v0(r, i, d, s, ())

    def test_v1_branches(self) -> None:
        self.assertEqual(v1(0.02, 0.05, 0.75, 0.0), "present")
        self.assertEqual(v1(0.019, 0.05, 1.0, 0.0), "indeterminate")
        self.assertEqual(v1(0.02, 0.05, 0.74, 0.0), "indeterminate")
        self.assertEqual(v1(-0.01, 0.0199, 0.0, 0.75), "re-tuning equivalent")
        self.assertEqual(v1(-0.01, 0.02, 0.0, 0.75), "indeterminate")
        self.assertEqual(v1(-0.01, 0.0199, 0.0, 0.74), "indeterminate")

    def test_v2_branches(self) -> None:
        self.assertEqual(v2(0.01, 0.05, 0.75, 0.0), "helps")
        self.assertEqual(v2(0.0099, 0.05, 1.0, 0.0), "indeterminate")
        self.assertEqual(v2(0.01, 0.05, 0.74, 0.0), "indeterminate")
        self.assertEqual(v2(-0.05, -0.0001, 0.0, 0.75), "hurts")
        self.assertEqual(v2(-0.05, 0.0, 0.0, 0.75), "indeterminate")
        self.assertEqual(v2(-0.05, -0.0001, 0.0, 0.74), "indeterminate")

    def test_labels_each_branch_and_v1_unchanged(self) -> None:
        self.assertEqual(labels(0.1, 0.0, 0.5, 0.0, 0.0), [])
        self.assertIn("loss gain without accuracy movement", labels(0.0, 0.0, 0.5, 0.0, 0.0))
        self.assertNotIn("loss gain without accuracy movement", labels(0.001, 0.0, 0.5, 0.0, 0.0))
        self.assertIn("survives finishing", labels(0.1, 0.02, 0.5, 0.0, 0.0))
        self.assertNotIn("survives finishing", labels(0.1, 0.019, 0.5, 0.0, 0.0))
        self.assertIn("harvests undertraining", labels(0.1, -0.1, 0.0199, 0.0, 0.0))
        self.assertNotIn("harvests undertraining", labels(0.1, -0.1, 0.02, 0.0, 0.0))
        self.assertIn("beats every named re-tune", labels(0.1, 0.0, 0.5, 0.001, 0.0))
        self.assertNotIn("beats every named re-tune", labels(0.1, 0.0, 0.5, 0.0, 0.0))
        self.assertIn("fresh-data effect", labels(0.1, 0.0, 0.5, 0.0, 0.001))
        self.assertNotIn("fresh-data effect", labels(0.1, 0.0, 0.5, 0.0, 0.0))
        self.assertEqual(labels(0.0, 0.02, 0.5, 0.001, None), labels(0.0, 0.02, 0.5, 0.001, 0.0))
        with self.assertRaises(ContractViolation):
            labels(0.1, 0.0, 0.5, 0.0, math.nan)
        before = v1(0.02, 0.05, 0.75, 0.75)
        labels(0.0, 0.03, 0.05, 0.1, 0.1)
        self.assertEqual(v1(0.02, 0.05, 0.75, 0.75), before)

    def test_bounds_must_be_ordered(self) -> None:
        # B2: a swapped lower and upper bound is a contract violation, never a reading from the wrong side.
        with self.assertRaisesRegex(ContractViolation, "d1_lower"):
            v1(0.05, 0.02, 0.75, 0.75)
        with self.assertRaisesRegex(ContractViolation, "d2_lower"):
            v2(0.05, 0.01, 0.75, 0.75)
        with self.assertRaisesRegex(ContractViolation, "d3_lower"):
            labels(0.1, 0.03, 0.02, 0.0, None)
        self.assertEqual(v1(0.02, 0.02, 0.75, 0.75), "present")
        self.assertEqual(v2(0.01, 0.01, 0.75, 0.75), "helps")
        self.assertIn("survives finishing", labels(0.1, 0.02, 0.02, 0.0, None))

    def test_static_reading(self) -> None:
        self.assertIsNone(static_reading(0.5, True, 6))
        self.assertIsNone(static_reading(0.5, False, 32))
        self.assertIsNone(static_reading(0.02, True, 32))
        self.assertIsNone(static_reading(0.5, True, 31))
        self.assertEqual(static_reading(0.021, True, 32), "design-in beats insertion")

    def test_programme_reading(self) -> None:
        present = programme_reading("present", [])
        self.assertEqual(present["rank"], 1)
        self.assertEqual(present["outcome"], "more capacity at a small parameter cost")
        self.assertTrue(present["opens_m1"])
        equivalent = programme_reading("re-tuning equivalent", [])
        self.assertEqual((equivalent["rank"], equivalent["opens_m1"]), (2, False))
        self.assertEqual(equivalent["outcome"], "same loss with fewer retained parameters")
        flat = programme_reading("indeterminate", [])
        self.assertEqual((flat["rank"], flat["opens_m1"]), (3, False))
        self.assertEqual(flat["outcome"], "no substantial change")
        beating = ["beats every named re-tune"]
        for verdict in ("present", "re-tuning equivalent", "indeterminate"):
            plain = programme_reading(verdict, [])
            graded = programme_reading(verdict, beating)
            self.assertFalse(plain["beats_named_retunes"])
            self.assertTrue(graded["beats_named_retunes"])
            self.assertEqual(plain["opens_m1"], graded["opens_m1"])
            self.assertEqual(plain["rank"], graded["rank"])
        with self.assertRaises(ContractViolation):
            programme_reading("maybe", [])

    def test_programme_reading_routes_the_next_screen_to_the_starved_host(self) -> None:
        # B1 (§7): V1 re-tuning-equivalent or "harvests undertraining" sends the next screen to the starved 0.5 host.
        starved = "starved 0.5 host"
        self.assertEqual(programme_reading("re-tuning equivalent", [])["next_screen"], starved)
        self.assertEqual(programme_reading("indeterminate", ["harvests undertraining"])["next_screen"], starved)
        self.assertEqual(programme_reading("present", ["harvests undertraining"])["next_screen"], starved)
        self.assertIsNone(programme_reading("present", [])["next_screen"])
        self.assertIsNone(programme_reading("present", ["survives finishing"])["next_screen"])
        self.assertIsNone(programme_reading("indeterminate", ["beats every named re-tune"])["next_screen"])

    def test_non_finite_and_fraction_range(self) -> None:
        with self.assertRaises(ContractViolation):
            v1(math.nan, 0.0, 0.5, 0.5)
        with self.assertRaises(ContractViolation):
            v1(0.0, math.inf, 0.5, 0.5)
        with self.assertRaises(ContractViolation):
            v1(0.0, 0.0, 1.01, 0.5)
        with self.assertRaises(ContractViolation):
            v1(0.0, 0.0, 0.5, -0.01)
        with self.assertRaises(ContractViolation):
            v2(0.0, math.nan, 0.5, 0.5)
        with self.assertRaises(ContractViolation):
            v2(0.0, 0.0, 0.5, 1.5)
        with self.assertRaises(ContractViolation):
            labels(math.nan, 0.0, 0.0, 0.0, 0.0)
        with self.assertRaises(ContractViolation):
            labels(0.0, 0.0, 0.0, 0.0, math.inf)
        with self.assertRaises(ContractViolation):
            static_reading(math.nan, True, 32)


def line_points() -> dict[float, tuple[float, ...]]:
    base = math.log2(CORE_PARAMETERS[0])
    return {float(p): (2.0 - 0.1 * (math.log2(p) - base),) * 6 for p in CORE_PARAMETERS}


if __name__ == "__main__":
    unittest.main()
