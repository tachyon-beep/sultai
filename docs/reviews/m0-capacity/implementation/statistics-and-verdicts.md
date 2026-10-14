# Exact-source review: statistics-and-verdicts (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Scope: `src/sultai_screen/m0/stats.py`, `src/sultai_screen/m0/verdicts.py`, tests `tests/test_m0_stats.py::StatsTests, ::VerdictTests`. The pre-rendered diff is byte-identical to `git diff b79f145..e1df6e1` for these paths (all three files are new). The preregistration at e1df6e1 hashes to `c66599459adc…3fd5`, equal to `identity.PREREG_SHA`. The 36 tests in `tests/test_m0_stats.py` pass (`/usr/bin/python3 -B -m unittest test_m0_stats`, torch 2.9.1+cu128). Interval conventions are those of plan decision (g) (Task 13 of `docs/plans/2026-10-13-m0-implementation.md`).

## Claims and evidence
| claim | source file:line | pinning test | verdict |
|---|---|---|---|
| §6 "D1 = G(M-sq) − G(M-aff)", G = mean N0 loss − mean arm loss | stats.py:179-185, 220 | test_d1_on_constructed_record (0.03); swapping arms or the sign fails | met |
| §6 "Fold residual r_aff … reported beside D1" | stats.py:136, 156-157, 239 | test_every_quantity_on_constructed_record (r_aff 0.0012) | met |
| §6 "ΔAcc = acc(M-sq) − acc(N0)" | stats.py:240 | test_every_quantity… / test_starved_pair… (0.25); using another arm or N0 accuracy fails | met |
| §6 "D1′ = G(M-sq) − max(G(R-272s), G(R-aff))", per anchor (plan) | stats.py:221 | test_d1prime_is_per_anchor_maximum (0.02, 0.01; aggregate-first would give 0.025) | met |
| §6 "D3 = L(T-cool host, after its own N0 rule) − L(M-sq \| T-cool) … not over the anchor's N0; and G(T-cool) relative to the anchor's N0" | stats.py:234-235 | test_d3_against_finished_host_noop_not_anchor_noop (D3 0.04 ≠ 0.14 against anchor N0; G_tcool 0.07) | met |
| §6 "D4 = G(M-sq-fresh) − G(M-sq)" | stats.py:236 | test_every_quantity… (0.015) | met |
| §6 "D5 = L(host + M-sq) − L(S-static + refit) … also without the refit" | stats.py:237-238 | test_every_quantity… (D5 −0.025, D5_norefit −0.02); a reversed sign fails | met |
| §6 "D2 = G(M-wide→8, contribution rank, refit) − G(M-small-B)"; "D2s = G(M-wide→8, as sliced with fold) − G(M-small)"; "the random-rank versions of both" | stats.py:222-233 | test_every_quantity… (all four values) | met |
| §6 "the refit's step-0 selection loss and selected rate beside D2" | rate: stats.py:197; step-0 loss: not in this item (fit.py:207-208 via worker.py:660/678 receipts) | rate: none in this item | met outside item (not reviewed here) |
| §6 "survival curves G(M-wide→h)/G(M-wide) for every h, sliced and refit"; "the leave-one-out ceiling" | stats.py:34, 160-161, 245, 247-260 | test_every_quantity… (8 ratios, leave_one_out 0.042); test_retune_fraction_and_survival_none_on_zero_denominator | met |
| §6 "Calibration: G(R-temp), G(R-cal); re-tune fraction G(M-sq)/G(R-all) when R-all runs" | stats.py:241-242, 246, 214-218 | test_every_quantity… (0.004, 0.006, 0.5); test_missing_optional_arms_are_none_with_reason | met |
| §2 inherited "harm screen (Hm = 0.01 nats)" | stats.py:252, 263-272 | test_harm_lists_candidates_beyond_harm_margin (2.011 listed, 2.009 not) | met |
| §6 "Per anchor, with per-example paired-bootstrap SE over the 5,000 dev examples (v4 §7), for loss and, as a secondary, for accuracy" | stats.py:275-279 defines `paired_bound`; nothing computes it per anchor (no production caller in src/) | test_paired_bound_is_normal_bound_on_paired_se pins the helper only | **not met** (Important 1) |
| §6 "Frontier: slope in nats per doubling … three core widths only … seed-bootstrap interval" | stats.py:403-447 (`_core` rejects any other width set; within-width resampling `frontier-bootstrap`; negated OLS on log2 params) | test_frontier_slope_on_exact_line; test_frontier_slope_requires_core_widths | met (interval ranks not pinned: Important 2) |
| §6 "G(M-sq) expressed as a fraction of one doubling with its interval" (plan: joint draw of reference seeds) | stats.py:459-492 | test_gain_fraction_on_exact_line; test_gain_fraction_draws_reference_seeds_jointly (independent draws would open the interval); test_gain_fraction_none_when_slope_not_positive | met |
| §6 "a second, descriptive fit over every 10-epoch width … labelled as such; the 20-epoch point never enters" | stats.py:450-456 (label; 2-5 widths; exclusion of the 20-epoch point is the caller's) | test_frontier_slope_descriptive | met |
| §6 "no ratio to 272"; "label 'same-recipe frontier, 10 epochs'" | stats.py:32; no 272 ratio anywhere | test_frontier_slope_on_exact_line (label) | met |
| §6 "Across anchors: median with one-sided 95 percent percentile anchor-bootstrap bounds" (plan: 10,000 resamples under generator(seed, "anchor-bootstrap"), 500th/9,500th order statistics, statistics.median) | stats.py:27, 289-291, 299-324 | test_anchor_bound_pinned_and_rederived; test_anchor_bound_identical_values; test_median_and_sd_conventions; test_achieved_half_width | met; ranks and sample count not discriminated by the fixture (Important 2) |
| §6 "an exact sign-test bound beside it" (plan: k largest with P(Bin(n,½) < k) ≤ 0.05; k-th smallest / (n−k+1)-th; zeros count against; None when n < 6) | stats.py:28, 332-352 | test_sign_test_bound_order_statistics (n = 32 → ordered[10] and ordered[21]; changing 0.05 or the comparison moves k); test_sign_test_zero_counts_against | met (exhaustively checked k against the exact binomial for n = 6..59) |
| §9.6 "n = max(32, n*) where n* gives a half-width ≤ X/2 for the median D1 under a normal approximation with the v4 inflation (the 32 floor and the inflation are the existing code)" | stats.py:355-357 → contracts.py:182-186 (ceil((1.645·1.25·sd/(X/2))²), max(32, ·)) | test_confirmatory_count ((0.0275)→(32,32), (0.03)→(39,39); contracts.X == identity.X); floor pinned by test_screen.py:219 (anchor_count(0) == (0, 32)) | met |
| §7 V0: replay bitwise both cards; A1 within 1e-4; B3 ≥ ρ·A3 every anchor; zero-module identity; printed counts 33h + 16; slicing known answer; complement hash; S-static step 0 | verdicts.py:47-59, 63 (aggregates upstream booleans; the tolerances themselves are applied in other items) | test_v0_each_clause_named (each flag false names exactly its clause); test_v0_contract (missing key raises) | met |
| §7 V0 "fold residual \|r_aff\| ≤ X/4 = 0.005 nats on every development anchor" | verdicts.py:21, 60-65 | test_v0_passes_and_fold_residual_boundary (±0.005 passes; 0.006 fails naming "fold residual"; `<` or a different tolerance fails) | met |
| §7 V1 "present if the lower bound of the median D1 ≥ X … and M-aff is plateau-certified on ≥ 75 percent of anchors" | verdicts.py:74-75 | test_v1_branches (0.02/0.75 present; 0.019 and 0.74 indeterminate; using the M-sq fraction fails) | met |
| §7 V1 "re-tuning equivalent if the upper bound of the median D1 < X and M-sq is certified on ≥ 75 percent" | verdicts.py:76-77 | test_v1_branches (0.0199 equivalent; 0.02 and 0.74 indeterminate) | met |
| §7 V1 "otherwise indeterminate, reported with the certification fractions" | verdicts.py:78 (fractions reported by caller, confirmatory.py:430) | test_v1_branches | met |
| §7 label "loss gain without accuracy movement" if lower bound of median ΔAcc ≤ 0, moving no threshold | verdicts.py:94-95; v1 takes no label input | test_labels_each_branch_and_v1_unchanged (0.0 in, 0.001 out; v1 unchanged) | met |
| §7 "survives finishing" lower D3 ≥ X; "harvests undertraining" upper D3 < X | verdicts.py:96-99 | test_labels… (0.02/0.019; 0.0199/0.02) | met |
| §7 "beats every named re-tune" lower D1′ > 0; "fresh-data effect" lower D4 > 0 | verdicts.py:100-103 | test_labels… (0.001 in, 0.0 out for each) | met |
| §7 static: "descriptive at the pilot … only at confirmatory n, only if S-static is in the confirmatory set, and only if the lower bound of the median D5 > M_s" | verdicts.py:23, 119-123 | test_static_reading (n 6/31 None; not in set None; 0.02 None; 0.021 reads) | met (n proxy: Minor 4) |
| §7 "S-static is admitted … only if the pilot's recorded SD(D5) ≤ 0.0275" | not in this item: identity.py:33 `S_STATIC_SD_LIMIT`, pilot.py:767-780 | none in this item | met outside item (not reviewed here) |
| §7 V2 "helps if the lower bound of the median D2 ≥ X₂ … and M-small-B is certified on ≥ 75 percent" | verdicts.py:112-113 | test_v2_branches (0.01/0.75 helps; 0.0099, 0.74 indeterminate) | met |
| §7 V2 "hurts if the upper bound of the median D2 < 0 and the refit is certified on ≥ 75 percent" | verdicts.py:114-115 | test_v2_branches (−0.0001 hurts; 0.0 and 0.74 indeterminate) | met |
| §7 programme reading: V1 outcomes ranked, all valid (present > re-tuning equivalent > no substantial change); D1′ grades value, not a gate; M1 opens only on V1 present | verdicts.py:126-144 | test_programme_reading (ranks, outcomes, opens_m1 only on present, "beats" never changes rank or opens_m1) | met |
| §7 "V1 re-tuning-equivalent or harvests-undertraining sends the next screen to the starved 0.5 host" | absent from programme_reading | none | not met (Minor 1) |
| §10 confirmatory arm set (N0 and calibrations, M-sq, M-aff, R-272s, R-aff, T-cool, M-sq \| T-cool, M-wide, M-wide→8 both rankings sliced and refit, M-small, M-small-B; S-static optional) | stats.py:68-86 | test_stage_arm_sets (exact set difference from PILOT_ARMS); test_confirmatory_record_reports_absent_pilot_arms_with_reason | met |
| §5 extended item 1: the 0.5-host pair reports D1 and D3 | stats.py:88 | test_starved_pair_record_reports_d1_and_d3 | met |
| §5 core pilot arms include S-static | stats.py:61, 65 (S-static is OPTIONAL, not in PILOT_ARMS) | test_stage_arm_sets pins PILOT_ARMS = CORE_ARMS | partly (Minor 2) |

## Frozen values checked
| value | prereg | source | match |
|---|---|---|---|
| X | 0.02 nats (§7 J1) | identity.py:26, used verdicts.py:74,76,96,98; contracts.X 0.02 (anchor_count) | yes (pinned test_m0_identity.py:89 and test_confirmatory_count) |
| X₂ | 0.01 nats (§7 J2) | identity.py:27, verdicts.py:112 | yes |
| M_s | 0.02 nats (§7 J3) | identity.py:28, verdicts.py:121 | yes |
| fold tolerance | X/4 = 0.005 (§7 V0) | identity.py:31, verdicts.py:62 (inclusive) | yes |
| certification rule | ≥ 75 percent (§7) | identity.py:32 CERTIFIED_FRACTION 0.75, inclusive at verdicts.py:74,76,112,114 | yes (pinned by 0.75/0.74 boundaries) |
| Hm | 0.01 nats (§2) | identity.py:30 HARM, stats.py:270 (strict >) | yes |
| SD(D5) admission | ≤ 0.0275 (§7) | identity.py:33 (outside this item) | yes |
| confirmatory n | max(32, n*), half-width ≤ X/2, v4 inflation | contracts.py:185-186 (1.645, 1.25, X/2, floor 32) | yes |
| one-sided z | 1.645 (v4 §7 normal bound, plan (g)) | stats.py:26 | yes (test_paired_bound…) |
| bootstrap resamples / ranks | 10,000; 500th / 9,500th (plan (g)) | stats.py:27, 306-311 (tail = samples // 20; ordered[499], ordered[9499]) | yes in source; not discriminated by tests (Important 2) |
| sign-test level / minimum | one-sided 95 %; None when n < 6 (plan (g)) | stats.py:28, 335 (20·P ≤ 1 exact integer) | yes |
| RNG streams | generator(seed, "anchor-bootstrap"), generator(seed, "frontier-bootstrap"), fit.paired_se "paired-bootstrap" | stats.py:323, 419, 278 | yes |
| core frontier widths | 85,274; 284,634; 613,018 (§3 table) | stats.py:30-31 | yes (test_reference_parameters, parameter_count(Host())) |
| frontier label | "same-recipe frontier, 10 epochs" | stats.py:32 | yes |
| sliced counts | 33h + 16 | not computed here (V0 boolean) | n/a |

No frozen value needs changing to make the code and the document agree; nothing here is amendment territory.

## Findings

**Important 1. stats.py:164-253, 275-279: the per-anchor per-example paired-bootstrap SE of §6 is never produced.**
What: §6 opens "Per anchor, with per-example paired-bootstrap SE over the 5,000 dev examples (v4 §7), for loss and, as a secondary, for accuracy". `quantities()` returns scalar D-values only; `paired_bound` (the plan's helper for this) has no caller anywhere under `src/` (a `git grep` for `paired_bound|paired_se` in `src/sultai_screen/m0` hits only stats.py). Accuracy differences other than ΔAcc are derivable from the per-arm accuracies in the record, but they have no SE either.
Why: a §6 reported quantity is missing from the per-anchor record. No verdict reads it (V1/V2 read anchor-bootstrap bounds on the medians), so verdict logic is unaffected, but the published per-anchor results would not match §6.
Fix: in `quantities`, for each two-arm quantity form the aligned per-example difference vectors (loss, and 0/1 correctness for the accuracy secondary; D3 against `tcool_noop_loss`; D1′ against the per-anchor max arm) and emit `<key>_se` and `<key>_acc_se` via `fit.paired_se` under a fixed seed convention. Pin it with a test that compares one SE to a direct `paired_se` of the hand-built difference vector. No frozen value changes.

**Important 2. tests/test_m0_stats.py:43-47, 303-319, 346-382: the 500th / 9,500th ranks and the 10,000 sample count are not pinned.**
What: with the SAMPLE fixture at seed 0, the sorted bootstrap medians equal 0.02 at every 1-based rank from 345 to 1,107 and 0.05 at every rank from 8,895 to 9,662. `samples=2000`, `4000` and `20000` also return (0.02, 0.05). The "independent derivation" uses the same draw call and the same ranks, so it re-checks the implementation's arithmetic but not the convention. Plan (g) does not fix the draw shape. A row-by-row draw consumes the RNG differently from the single `(samples, n)` draw and would still pass, given those plateaus. The frontier and gain-fraction tests use exact lines or constructions where every replicate is equal, so they cannot see ranks either.
Why: plan decision (g) fixes these conventions "so two implementers compute the same numbers". The source is correct today (stats.py:306-311), but an off-by-one (`ordered[tail]`), a 4 % tail, or a different `BOOTSTRAP_SAMPLES` would pass the suite.
Fix: add a fixture whose bootstrap medians are distinct around ranks 499/500/501 and 9,499/9,500/9,501, for example seven irrational-spaced values, and assert the 500th and 9,500th exactly and that the neighbours differ. This also pins the draw shape. Assert `BOOTSTRAP_SAMPLES == 10_000` and that the default call equals an explicit `samples=10_000` call. Add one frontier fixture with seed spread that pins `lower`/`upper` the same way.

**Minor 1. verdicts.py:126-144: `programme_reading` omits the §7 routing rule.**
What: §7 says "V1 re-tuning-equivalent or harvests-undertraining sends the next screen to the starved 0.5 host [taken, J-host]". The function reports rank, outcome, opens_m1 and beats_named_retunes only. It never reads "harvests undertraining" from `label_list`.
Why: the reported programme reading would leave out a stated consequence of the verdict. It decides nothing inside M0.
Fix: add `"next_screen": "starved 0.5 host" if v1_verdict == "re-tuning equivalent" or "harvests undertraining" in label_list else None`, with a test for both triggers and for "present" without the label.

**Minor 2. stats.py:61, 65: S-static is a core pilot case (§5 core table; §9 steps 3 and 5) but is not in `PILOT_ARMS`.**
What: `quantities(…, PILOT_ARMS)` and pilot.py's `anchor(seed, sources, stats.PILOT_ARMS)` (pilot.py:701) do not hold when S-static or S-static:refit is missing or unavailable. D5 becomes None ("S-static:refit not run"), and `spreads` (pilot.py:762-780, read at e1df6e1) then records `sd_d5: None` with reason "D5 undefined on an anchor" and `static_admitted: False` rather than a hold.
Why: the §9 step 5 SD(D5) record could be absent, with only a reason string recorded. The failure direction is conservative: S-static is only excluded, and it stays out under the 12 h ceiling anyway.
Fix: require `S-static` and `S-static:refit` in the pilot's required set (for example `PILOT_ARMS | {"S-static", "S-static:refit"}` at pilot.py:701, or a `PILOT_REQUIRED` constant in stats.py), and test that a pilot record without them raises.

**Minor 3. verdicts.py:69-116: no check that lower ≤ upper.**
What: `v1`, `v2` and `labels` accept `d1_lower > d1_upper` (likewise D2, D3). A caller that swaps the arguments could get "present" from an upper bound.
Why: these are pure functions whose callers are elsewhere. A two-line contract check removes the possibility.
Fix: raise `ContractViolation` when lower > upper in `v1`, `v2` and for the D3 pair in `labels`, and add one test.

**Minor 4. verdicts.py:23, 119-123: "only at confirmatory n" is encoded as `n >= 32`.**
What: `static_reading` treats any n ≥ 32 as confirmatory rather than comparing n with the sized confirmatory count or taking a stage flag. Its only caller (confirmatory.py:427) passes the sized n with `in_confirmatory_set=True`, so behaviour is correct today.
Why: the "at confirmatory n" clause depends on the caller's discipline.
Fix (optional): take `stage: Literal["pilot", "confirmatory"]`, or have the caller assert that n equals the frozen confirmatory count.

**Behaviour not requested by the preregistration (item 4).** There is no new randomness beyond the plan's three named streams, and no RNG-order change. Gate 3 code is only read (`contracts.anchor_count`, `fit.paired_se`), not modified, and no Gate 3 receipt byte is touched (new files only; import isolation is tested). These descriptive extras do not affect any verdict: `G_mexp`, `G_lottery` (extended arms), `unfolded` losses, `noop_accuracy`, `tcool_noop_loss`, `Fraction.point`, and `programme_reading` (outside the plan's Task 14 interface but within prereg §7).

## Dispositions (controller, 2026-10-14)

- Important 1: FIXED in the post-review commit (item A4: per-anchor paired-bootstrap SEs for every two-arm quantity, loss and accuracy).
- Important 2: FIXED (item C4: rank/sample-count pins).
- Minor 1: FIXED (item B1, `next_screen`). Minor 3: FIXED (item B2). Minor 2: RULED R29 — S-static stays optional at pilot aggregation (unavailable → static_admitted False with reason, never a hold). Minor 4: deferred, recorded.

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
