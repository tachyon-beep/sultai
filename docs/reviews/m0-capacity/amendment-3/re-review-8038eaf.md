# Scoped re-review: aggregation hold and refit cost (commit 8038eaf)

**Verdict:** FINDINGS REMAIN

## Items checked

| # | Item | Result |
|---|------|--------|
| 1a | `SURVIVAL_CURVE_ARMS` = exactly the six `M-wide→{16,4,2}:{sliced,refit}` arms (stats.py); `CONFIRMATORY_ARMS` unpacks it | OK; h = 8 arms, M-wide, M-small, M-small-B and verdict arms are not in it |
| 1b | `held` expression (receipts.py `anchor`): `held = [name for name in unavailable if name in required and name not in SURVIVAL_CURVE_ARMS]` | OK: every other unavailable required arm still raises InstrumentHold |
| 1c | Exempt arm never run: `missing = sorted(required - set(records))` then `if missing: raise PilotHold(f"anchor {seed}: required arm(s) not run: {missing}")` is unchanged and unexempted | OK |
| 1d | `quantities`: `missing = sorted(required - set(result.arms) - SURVIVAL_CURVE_ARMS)`; any other missing required arm raises ContractViolation (stats test deletes M-small and asserts it) | OK |
| 1e | `curve_reason` (confirmatory.py): curve's own reason first, else "<arm> unavailable" if named in the anchor's `unavailable_arms`, else "<arm> not run" | OK |
| 2 | `unittest test_m0_pilot.ReceiptTests test_m0_stats`: 51 tests OK. Both scenario tests: OK | OK |
| 3 | costs.py `ARM_UNITS`: three refit arms `("grid:refit-single", "evaluation")`, three sliced `()` | OK |
| 3 | `test_m0_supervision` + `MeasuredProjectionTests`: 14 tests OK | OK |
| 3 | Amendment line 41 ("3.2 s per 800 steps plus one development evaluation") vs code: refit = refit-single unit + one evaluation | Agree in structure (see finding 2 on the 3.2 s vs 10 s unit) |
| 4 | `verdicts.py`, pinned documents (nothing under docs/plans), Gate 3 paths | Untouched |
| 4 | Diff is NOT only m0 source and tests (see finding 1) | Deviation from the stated scope |

## Probes

| Probe | Failing test or assertion | Result |
|-------|---------------------------|--------|
| P1 drop `and name not in SURVIVAL_CURVE_ARMS` from `held` | `test_m0_pilot.ReceiptTests.test_an_unavailable_survival_curve_arm_never_holds_and_is_named_absent` (ERROR, InstrumentHold) | Pinned; 1 of 9 ReceiptTests fails |
| P2 widen exemption to `SURVIVAL_CURVE_ARMS \| {"M-small"}` | `ReceiptTests.test_an_unavailable_required_arm_holds_naming_anchor_and_arm` PASSED (unpinned); `ConfirmatoryTests.test_an_unavailable_required_arm_holds_at_aggregation_and_drops_no_anchor` FAILED: `('execution-invalid','aggregation') != ('instrument-failed','aggregation')` | Upper bound pinned by the confirmatory test only; the receipt-level test does not pin it (finding 3) |
| P3 `curve_reason` returns constant "not run" | `ConfirmatoryTests.test_an_unavailable_survival_curve_arm_is_published_undefined_and_the_verdict_is_read`: reason `'... anchor 7510 (not run)'` != `'... anchor 7510 (M-wide→4:refit unavailable)'` | Pinned |

All files restored byte-exactly (cmp against scratch copies); working tree clean, `git diff --stat` empty.

## Cost arithmetic

Fixture: refit-single 10, evaluation 0.5.

| Quantity | Recomputed | Literal in tests | Match |
|----------|-----------|------------------|-------|
| per anchor, confirmatory set | 1,040 + 3 x 0.5 = 1,041.5 | 1_041.5 (supervision, MeasuredProjection) | yes |
| with S-static | 1,041.5 + 110 = 1,151.5 | 1_151.5 | yes |
| raw / scale | 1,161.5 ; 841.5/961.5 | 1_161.5 ; 841.5 / 961.5 | yes |
| stage-5 core, six anchors | 6,360 + 6 x 3 x 10.5 = 6,549 | 6_549.0 and `3000+6549+1000` | yes |
| 1,500 + 32 x 1,041.5 | 34,828 | 34_828.0 (and +32 x 250 variant) | yes |
| with_static 1,500 + 32 x 1,151.5 | 38,348 | 38_348.0 | yes |
| 600 + 32 x 1,041.5 | 33,928 | 33_928.0 in `carried_projection`; `600 + 32 * 1_041.5` in ConfirmatoryTests | yes |
| 455 + 37 x 1,041.5 | 38,990.5 | 38_990.5 | yes |
| 455 + 37 x 1,161.5 | 43,430.5 (> 43,200) | 43_430.5, within_ceiling False | yes |
| 1,500 + 42 x 1,041.5 | 45,243 | `1500 + 42 * 1_041.5` | yes |

## Remaining findings

1. Scope: `git diff --stat d2e30b3..8038eaf` also carries README.md, docs/DECISIONS.md, docs/RESUME.md, docs/certificates/2026-10-14-m0-pilot-admission-2.json, docs/results/m0-amendment-3-2026-10-14/{quality,runtime}.json and runtime.txt, and four docs/reviews/m0-capacity/amendment-3 memos. No verdicts.py, pinned-document or Gate 3 path is touched, but the diff is not "only m0 source and tests". The admission certificate and result receipts sit in the same round as the source fix (confirm they were regenerated for 8038eaf and not left bound to d2e30b3).
2. Amendment line 41 says "3.2 s per 800 steps plus one development evaluation" while the code books the refit-single unit (10 s in the fixture) plus the evaluation unit. Structure agrees; the 3.2 s nominal is not what the fixture unit encodes. Wording only, if the measured unit is meant to stand in for 3.2 s.
3. P2: widening the exemption to M-small fails only the confirmatory test; `ReceiptTests.test_an_unavailable_required_arm_holds_naming_anchor_and_arm` (which uses M-small-B) does not pin M-small or M-wide at the receipt level. Minor; the bound is pinned elsewhere.
4. Cosmetic: the `quantities` docstring in stats.py now has a very long line (the added sentence is appended to the existing line); may trip the line-length lint.

## Dispositions (controller, commit c54fb7c)

| Finding | Disposition |
|---|---|
| 1 scope | Expected: the range crosses the evidence and certificate commits for d2e30b3; both are regenerated and reissued for c54fb7c in the certificate commit. |
| 2 wording | Noted, no change: 3.2 s per 800 steps is the §8 nominal and the amendment defers to the measured kernel; the 10 s is a test fixture. |
| 3 probe P2 | Adopted (c54fb7c): the receipt-level test holds on unavailable M-wide, both h = 8 cut forms, M-small and R-temp. |
| 4 docstring | Adopted (c54fb7c): wrapped. |
