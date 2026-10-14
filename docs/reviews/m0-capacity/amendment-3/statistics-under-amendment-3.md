# Exact-source review: statistics-under-amendment-3 (commit 3021ff3)

**Verdict:** ACCEPTED WITH FINDINGS

Scope: `src/sultai_screen/m0/stats.py`, `src/sultai_screen/m0/confirmatory.py` (`read_once`, `VERDICT_KEYS`,
`BESIDE_KEYS`), `src/sultai_screen/m0/pilot.py` (`survival_medians`, `development_medians`) and the tests named in the
brief, at commit 3021ff3, against amendment 3 (`docs/plans/2026-10-14-m0-amendment-3.md`, as committed at 3021ff3) and
preregistration §6, §7, §9 and §10. The pre-rendered `statistics.diff` was read in full, and the source was read where
its context ran out. Line numbers are those at 3021ff3. In the review environment the three test files pass: 111 tests
and 17 subtests, with `PYTHONPATH=src`.

Mutation probes were run on scratch copies made with `git archive 3021ff3` in the session scratchpad; the repository
was not touched. Each probe made one production change and then ran StatsTests, ConfirmatoryTests,
MeasuredProjectionTests, PilotRunTests and SupervisionTests. A probe is "killed" when a test fails and "survived" when
every test still passes:

| Probe | Production change | Result | Killing test |
|---|---|---|---|
| M1 | `beside` drops the three sign-test keys | killed | ConfirmatoryTests.test_clean_confirmatory_… |
| M2 | survival entries carry `sign_test_lower/upper = None` | **survived** | none |
| M3 | `read_once` drops width h = 2 from the survival intervals | killed | test_clean_confirmatory_… |
| M4 | `beside` raises InstrumentHold on an undefined anchor (instead of publishing undefined) | **survived** | none |
| M5 | `column` stops raising InstrumentHold on an undefined verdict quantity | **survived** | none |
| M6 | CONFIRMATORY_ARMS loses `M-wide→2:refit` | killed | StatsTests.test_confirmatory_record_reports_absent_pilot_arms_with_reason |
| M7 | survival curve reverts to undefined whenever any width is absent (pre-amendment behaviour) | killed | StatsTests.test_survival_curve_reports_every_width_that_ran_and_names_the_absent |
| M8 | `development_medians` omits `survival_medians` | killed | PilotRunTests.test_clean_pilot_publishes_every_receipt_… |
| M9 | V1 reads `sign_test_lower` in place of the bootstrap lower bound | **survived** | none |
| M10 | BESIDE_KEYS loses `G_rtemp` | killed | test_clean_confirmatory_… |
| M11 | `D5_norefit` is not bounded under the static flag | killed | ConfirmatoryTests.test_static_runs_s_static_under_the_14_hour_ceiling |
| M12 | survival `beside` reasons replaced by a constant (anchor reason lost) | **survived** | none |
| M13 | `sign_test_lower` replaced by the bootstrap lower bound | **survived** | none |
| M14 | `survival_value` returns None for every width of a partial curve | killed | test_survival_curve_reports_every_width_… |

## Claims and evidence

| Claim (quoted) | Source file:line | Pinning test | Verdict |
|---|---|---|---|
| A3.1 "The confirmatory arm set is extended by M-wide→16, M-wide→4 and M-wide→2, each as sliced with fold and as refit" | stats.py:74-93 (line 76 adds the six arms); costs.py:106-111 (sliced arms add nothing, refits book `grid:refit-single`; unchanged in this diff) | StatsTests.test_stage_arm_sets pins `PILOT_ARMS − CONFIRMATORY_ARMS == {M-sq-fresh, M-wide→8:ablation}`, with `PILOT_ARMS == CORE_ARMS` and `CONFIRMATORY_ARMS < PILOT_ARMS`, so the set is fixed exactly. The clean confirmatory test asserts `ran == set(CONFIRMATORY_ARMS)` per anchor. M6 killed. | Satisfied, pinned |
| A3.1 "so that the survival curves … h ∈ {16, 8, 4, 2}, sliced and refit, are reported at confirmatory n with medians and bounds … They have no verdict role." | confirmatory.py:431-441 (one `beside` per kind × width); no survival key reaches `verdicts.*` (lines 448-462) | test_clean_confirmatory_… asserts the curve's kinds and widths, a finite median and lower ≤ upper for each. M3 killed. | Satisfied, pinned |
| A3.1 "The stage 6 projection (§9) and the confirmatory launcher carry the arms automatically because both read the arm set." | confirmatory.py:354-357 (`stage_core` filters MODULE_ARMS and RETUNE_ARMS by `plan.arms`), 371-373 and 387 (`aggregate` requires `plan.arms` through `stats.quantities`); costs `per_anchor(w, CONFIRMATORY_ARMS)` | SupervisionTests.test_confirmatory_projection_from_measured_units (1,040 s; 34,780 s); MeasuredProjectionTests (1,040 s, 1,150 s, scale 840/960, 38,935 s); ConfirmatoryTests projection 600 + 32 × 1,040 | Satisfied, pinned |
| A3.2 s1 "every §6 quantity the confirmatory arm set defines on every anchor is published as its median with one-sided 95 percent percentile anchor-bootstrap bounds and the exact sign-test bound beside it: D1, D1′, D2, D2s, the random-rank D2 and D2s, D3, G(T-cool), r_aff, ΔAcc, G(R-temp), G(R-cal) and every survival ratio (each width, sliced and refit); D5 with and without the refit when S-static is in the set." | confirmatory.py:95-96 (VERDICT_KEYS ∪ BESIDE_KEYS is exactly the twelve named scalars), 403-413 (`interval` = median, bootstrap lower and upper, sign-test lower and upper), 422-423 (D5 and D5_norefit only when `plan.static`), 431-441 (survival 2 × 4) | test_clean_confirmatory_… asserts the interval key set exactly (12 scalars + `survival`, so no D5), the six-field shape of each scalar, and `sign_test_reason is None`. test_static_runs_… asserts D5 and D5_norefit. M1, M10 and M11 killed. The **values** of the sign test are not pinned (M13 survived), and on survival entries only the key's presence is pinned (M2 survived): finding F3. | Satisfied; partly pinned |
| A3.2 s2 "The verdict logic is unchanged: only D1 (V1), D2 (V2), D3, ΔAcc and D1′ (labels) and D5 (the static reading) condition anything; the rest are reported beside with no verdict role." | confirmatory.py:445-462: the V1, labels, V2 and static_reading calls and `bound_of` are unchanged context in the diff, and they read the bootstrap `lower` and `upper` only. VERDICT_KEYS (D1, D1prime, D2, D3, delta_acc, + D5 under static) equals the former BOUNDED tuple. `verdicts.py` is byte-unchanged (`git diff --stat 03a4a7a..3021ff3 -- src/sultai_screen/m0/verdicts.py` is empty). RNG: `stats.anchor_bound` (stats.py:393-398) draws from a fresh `generator(BOOTSTRAP_SEED, "anchor-bootstrap")` on every call (data.py:104-105), and `sign_test_bound` is pure, so the added calls cannot move any verdict bound. The D1/D2/D3/ΔAcc/D1′/D5 bounds are bit-identical to 03a4a7a for the same inputs. | `result["verdict_keys"]` is asserted in both tests. The byte evidence is the pin: no test fixes the V1 inputs exactly (M9 survived), and `v1 in (…)` would not catch a changed threshold. Finding F4. | Satisfied (byte evidence); not test-pinned |
| A3.2 s3 "A quantity undefined on an anchor that no verdict reads is published as undefined with the anchor named, and does not stop the reading" | confirmatory.py:415-420 (`beside` returns `{"median": None, "reason": "<name> undefined on anchor <seed> (<reason>)"}`); 425-430 and 431-441 send every no-verdict quantity through it | **None.** No test reaches the `value is None` branch of `beside` (M4 and M12 survived). Finding F1. | Satisfied in source; **unpinned** |
| A3.2 s3 "a verdict quantity undefined on an anchor holds the reading as before" | confirmatory.py:397-401 (`column` raises InstrumentHold), used for every verdict key at 424 and for half-widths at 455 | **None.** No test reaches it (M5 survived). At confirmatory n `aggregate` (lines 373 and 387 → `stats.quantities`, stats.py:195-197) raises on any absent required arm first, so the branch is defensive. test_an_unavailable_required_arm_holds_at_aggregation_… pins the upstream hold. Finding F1. | Satisfied in source; **unpinned** |
| A3.2 s4 "A survival curve reports every width that ran and names the absent widths; it is undefined only when M-wide is absent or G(M-wide) ≤ 0." | stats.py:269 (`survival_absent` lists the absent arms, width then kind), 270-273 (undefined only on `gain(M-wide) ≤ 0` → "denominator ≤ 0", or M-wide absent → "M-wide not run"), 318-323 (`_survival` keeps only the present widths), 326-334 (`survival_value`); `survival_absent` reaches the published 04-anchors.json via confirmatory.py:546-547 | test_survival_curve_reports_every_width_… (partial curve, absent names in order, M-wide gain 0 → None); test_starved_pair_record_… (M-wide absent → "M-wide not run"). M7 and M14 killed. At the confirmatory read, the G(M-wide) ≤ 0 path is unpinned (F1). | Satisfied, pinned at the record level |
| A3.2 s5 "The pilot's development medians (§9 stage 7) include the survival ratios, for the record only." | pilot.py:1115-1124 (`survival_medians`, None where any anchor leaves a width undefined, as `median_or_none` does), 1134 (included in `development_medians`); no verdict reads it | PilotRunTests.test_clean_pilot_publishes_every_receipt_… asserts the kinds, the widths and finite values. M8 killed. | Satisfied, pinned |
| §7 line 429 "D2s, the random-rank contrasts, the survival curves and the extended arms are reported beside with bounds and no verdict role." | As A3.2 s1 and s2. The extended arms stay pilot-only (§10); D4 is published as None with a reason (confirmatory.py:470-471). | As above | Satisfied |
| §6 line 346 "the refit's step-0 selection loss and selected rate beside D2" | Not bounded across anchors. The per-anchor rate is in `quantities["arms"][…]["rate"]`. | — | Outside A3.2's enumeration; observation F5 |

## Frozen values checked

| Value | Document | Source | Match |
|---|---|---|---|
| Confirmatory arm set | §10 (lines 558-561): N0 and calibrations (R-temp, R-cal), M-sq, M-aff, R-272s, R-aff, T-cool, M-sq\|T-cool, M-wide, M-wide→8 rank and random × sliced and refit, M-small, M-small-B (15 arms); A3.1 adds M-wide→{16,4,2} × {sliced, refit} (6) | stats.py:74-93; recomputed: `M-aff, M-small, M-small-B, M-sq, M-sq\|T-cool, M-wide, M-wide→16:refit, M-wide→16:sliced, M-wide→2:refit, M-wide→2:sliced, M-wide→4:refit, M-wide→4:sliced, M-wide→8:random:refit, M-wide→8:random:sliced, M-wide→8:rank:refit, M-wide→8:rank:sliced, R-272s, R-aff, R-cal, R-temp, T-cool` (21). PILOT − CONFIRMATORY = {M-sq-fresh, M-wide→8:ablation}. S-static is in STATIC_ARMS only, under the 14 h flag. N0 is the anchor's no-op, not an arm. | Yes, nothing extra |
| SURVIVAL_WIDTHS | §6 "for every h", §5 / A3.1 h ∈ {16, 8, 4, 2} | stats.py:38 `(16, 8, 4, 2)`; SURVIVAL_KINDS stats.py:39 `("sliced", "refit")`; h = 8 maps to the `rank` arms (stats.py:180-181, 310-315) | Yes |
| Bootstrap resamples | v4 §7 / Task 13: 10,000 | stats.py:27 `BOOTSTRAP_SAMPLES = 10_000` | Yes |
| Order statistics | 500th / 9,500th, 1-based, no interpolation | stats.py:380-385: tail = 10,000 // 20 = 500; lower `ordered[499]` (500th), upper `ordered[10000 − 500 − 1] = ordered[9499]` (9,500th) | Yes |
| Bootstrap seed | 3000 (R18 f) | stats.py:31; confirmatory imports it via receipts and passes it to every `anchor_bound`; label "anchor-bootstrap" | Yes |
| Sign-test rank | exact one-sided 95 %: lower = k-th smallest, upper = (n − k + 1)-th, k the largest with P(Bin(n, ½) < k) ≤ 0.05; zeros kept; n < 6 → None | stats.py:406-426. Recomputed: n = 32 → k = 11 (P(<11) = 0.0251, P(<12) = 0.0551); n = 33 → 12; n = 37 → 14; n = 6 → 1. Unchanged by this diff. | Yes |
| Per-anchor §10 cost in the §8 fixture | 1,010 s + 3 × `refit-single` 10 s = 1,040 s | Recomputed with `costs.per_anchor(workloads(), CONFIRMATORY_ARMS)`: 45 + 30 + 4 × 60 + 100 + 2 × 60 + 120 + 80 + 250 + 25 + 3 × 10 = **1,040.0** | Yes |
| Static per anchor | 1,040 + 50 + 60 = 1,150 s | recomputed **1,150.0** | Yes |
| 32 × 1,040 + 600 | 33,880 | recomputed **33,880** (carried_projection; clean-confirmatory kernel projection) | Yes |
| 32 × 1,040 + 1,500 | 34,780 | recomputed **34,780** (SupervisionTests) | Yes |
| 42 × 1,040 + 1,500 over 43,200 | 45,180 > 43,200 | recomputed **45,180**; `within_ceiling` False is asserted | Yes |
| 455 + 37 × 1,040 | 38,935 ≤ 43,200 | recomputed **38,935**; `within_ceiling` True is asserted | Yes |
| Static projection (also touched) | 32 × 1,150 + 1,500 | recomputed **38,300** ≤ 50,400 | Yes |
| Scale term (also touched) | (1,040 − 200) / 960 | `whole` = 45 + 30 + 100 + 25 = 200, so 840/960 | Yes |
| `refit-single` nominal vs fixture | A3.1: "3.2 s per 800 steps plus one development evaluation, about 10 s per anchor, about 320 s at n = 32" | fixture books 10 s **per refit**, 30 s per anchor, 960 s at n = 32 | Not frozen (the kernel is pilot-measured). The amendment's wording is ambiguous: finding F2. |
| Ceilings | 43,200 s; 50,400 s static | unchanged and asserted | Yes |

Gate 3 and verdict isolation: `git diff --stat 03a4a7a..3021ff3 -- src/sultai_screen/m0/verdicts.py
src/sultai_screen/screen.py src/sultai_screen/contracts.py src/sultai_screen/pilot.py tests/test_screen.py
tests/test_v42_ladder.py` is empty. The commit touches only `m0/` sources, `m0` tests and the two amendment and
certificate documents. The only output-shape changes are additive keys in M0 receipts (per-anchor `survival_absent`,
`intervals.*` sign-test fields and the beside keys, `verdict_keys`, `survival_medians`), and none is a Gate 3 receipt.
No new randomness is introduced: every new bound reuses the fixed per-call generator, and the sign test is
deterministic.

## Findings

**F1 — Important.** confirmatory.py:397-401 and 415-420; no test.
- What: neither undefined-quantity sentence of A3.2 is pinned at the confirmatory read.
  - Mutation M4 makes `beside` raise InstrumentHold on an undefined no-verdict quantity, which stops the reading the
    amendment says must continue. It passes every test.
  - Mutation M12 drops the anchor reason. It passes.
  - Mutation M5 makes `column` stop holding on an undefined verdict quantity. It passes.
- Why: the brief asks exactly whether the tests would fail if an undefined no-verdict quantity raised, or an undefined
  verdict quantity did not. They would not. At confirmatory n, `beside`'s None branch has one reachable path: G(M-wide) ≤
  0 on some anchor, which makes every survival width undefined. No fixture drives that path.
- Fix (tests only, no source change):
  - Add a ConfirmatoryTests launch with a `boost` that makes G(M-wide) ≤ 0 on one anchor. Assert that the verdict
    is still read (`v1` and `v2` present, code 0), that every `intervals.survival.*.*` entry has `median: None` and a
    reason naming that seed and "denominator ≤ 0", and that the verdict keys are intact.
  - Add a direct `read_once` unit call with None in one anchor's D1 (or D2), asserting InstrumentHold "no verdict is
    read".
  - Add one with None in a BESIDE key (e.g. `G_rtemp`), asserting the reading completes and the reason names the
    anchor.

**F2 — Minor.** amendment-3 A3.1 wording vs tests/test_m0_supervision.py:46-60.
- What: A3.1 says "about 10 s per anchor, about 320 s at n = 32", and the sentence can be read as the total for the
  three refits. The fixture books 10 s per refit: 30 s per anchor, 960 s at n = 32.
- Why: the code is consistent with the per-refit reading, and with the 1,040 s figure in the brief. The amendment's
  prose may understate the added charge threefold. No frozen value is involved, because the kernel is pilot-measured
  and both ceilings are unchanged.
- Fix: no code change. If the controller wants agreement between the documents, record in the disposition that the
  nominal is per refit (about 960 s at n = 32 for the three); changing the amendment's bytes would need amendment 4.

**F3 — Minor.** tests/test_m0_pilot.py (test_clean_confirmatory_…, survival block).
- What: the sign test's values are unpinned.
  - Mutation M13 replaces `sign_test_lower` with the bootstrap lower bound, and it survives.
  - Mutation M2 sets the survival entries' sign-test bounds to None, and it survives: the survival block asserts only
    `assertIn("sign_test_lower", each)`, not `sign_test_reason is None` or numeric bounds.
- Why: A3.2 requires "the exact sign-test bound beside it" for every survival ratio. The scalar beside keys get the
  stronger shape check; the survival entries do not, and no test ties the published sign-test value to
  `stats.sign_test_bound`. `stats.sign_test_bound` itself is pinned in StatsTests.
- Fix:
  - In the survival loop, assert the same six-field key set, `sign_test_reason is None` and
    `sign_test_lower <= sign_test_upper`.
  - For one key, assert `intervals["D1"]["sign_test_lower"] == stats.sign_test_bound(column, lower=True)[0]`,
    recomputed from the published 04-anchors.json quantities.

**F4 — Minor (pre-existing, recorded because the brief asks about it).** confirmatory.py:445-462.
- What: no test pins which bound feeds V1, the labels or V2 (mutation M9, V1 reading `sign_test_lower`, survives). The
  clean test asserts only `v1 in (…)`.
- Why: under this amendment the verdict path is protected by byte evidence (verdicts.py unchanged; the call lines are
  unchanged diff context), not by a test. A later edit could change it silently.
- Fix (optional): assert in the clean test that `result["v1"] == verdicts.v1(intervals.D1.lower, intervals.D1.upper,
  fractions…)`, and the same for V2.

**F5 — Observation (no severity).** stats.py / confirmatory.py.
- What, first: §6 line 346 also lists "the refit's step-0 selection loss and selected rate beside D2". Neither is
  bounded across anchors, and the step-0 selection loss is not in the `quantities` record at all. A3.2's colon list,
  which the code follows exactly, does not name them, so this matches the amendment as written.
- What, second: `verdict_keys` is a result field the amendment did not ask for. It is harmless, additive reporting on an
  M0 receipt.
- Fix: none required.

Note on the working tree: during this review, uncommitted edits by another party appeared in
`docs/plans/2026-10-14-m0-amendment-3.md`, `docs/certificates/2026-10-14-m0-freeze-3.md`,
`src/sultai_screen/m0/identity.py` and `tests/test_m0_identity.py`. They do not touch the A3.1 or A3.2 wording, and
they do not touch the files under review. This review is of the committed bytes at 3021ff3. The reviewer made no
changes to the repository.

Nothing found requires changing a frozen value. No BLOCKING finding.

## Dispositions (controller, commit d2e30b3)

| Finding | Disposition |
|---|---|
| F1 (Important) | Adopted: `tests/test_m0_pilot.py::ReadOnceTests` builds constructed confirmatory anchors and pins both A3.2 rules at `read_once` (an undefined no-verdict quantity and an undefined survival ratio are published naming the anchor while V1/V2 are still read; each of D1, D1′, D2, D3, ΔAcc undefined on an anchor raises InstrumentHold). Probes M4, M5, M12 re-run by the scoped re-review. |
| F2 (Minor) | Adopted: the amendment distinguishes about 3.3 s per refit from about 10 s per anchor and says the measured kernel replaces the nominals. |
| F3 (Minor) | Adopted: `ReadOnceTests.test_sign_test_bounds_are_the_order_statistics_of_each_column` pins the sign-test values on D1, D2s, G_rtemp, r_aff and the survival refit h = 4 entry (2nd and 7th order statistics at n = 8). Probes M2, M13 re-run by the re-review. |
| F4 (Minor, pre-existing) | Parked: which bound feeds V1/V2 is pinned by the unchanged `verdicts.py` and its own tests (`tests/test_m0_stats.py::VerdictTests`); a `read_once`-level pin is a later hardening, no verdict logic changed here. |
| F5 (observation) | Noted. `verdict_keys` is reporting only; the step-0 selection loss and selected rate stay per-anchor descriptives beside D2 as the amendment lists. |
