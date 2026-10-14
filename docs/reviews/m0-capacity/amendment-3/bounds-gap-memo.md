# M0: missing uncertainty bounds and survival-curve widths (2026-10-14, pre-pilot)

Raised by John after the gate B merge (03a4a7a). Nothing has run on GPU; no seed block is burnt.
Preregistration lines cited are in docs/plans/2026-10-13-capacity-frontier-screen.md (frozen, c6659945…).

## Promises

- §6 lines 366-368: "Across anchors: median with one-sided 95 percent percentile anchor-bootstrap bounds (v4 §7),
  and an exact sign-test bound beside it at confirmatory n." Generic: every §6 quantity, not D1 alone.
- §6 lines 350-354: "survival curves G(M-wide→h)/G(M-wide) for every h, sliced and refit".
- §7 lines 429-430: "D2s, the random-rank contrasts, the survival curves and the extended arms are reported beside
  with bounds and no verdict role."
- §10 lines 558-561 (J7, taken): confirmatory arm set = N0 and calibrations, M-sq, M-aff, R-272s, R-aff, T-cool,
  M-sq | T-cool, M-wide, M-wide→8 (both rankings, sliced and refit), M-small, M-small-B. M-wide→{16, 4, 2} absent.

## Defects

| # | Where | What is published | What is promised | Owner |
|---|---|---|---|---|
| 1 | confirmatory.py:93 `BOUNDED = (D1, D1prime, D2, D3, delta_acc)` (+D5 when static) | medians + bounds for five keys; sign test (:409-412) for D1 only | bounds for D2s, D2_random, D2s_random, the survival ratios, G_tcool, r_aff, G_rtemp, G_rcal, D5_norefit (when static); sign test beside every bounded median at confirmatory n | code |
| 2 | stats.py:264-268 `put("survival", curve_arms, …)` | `None` with reason "M-wide→16:sliced not run" whenever any width is absent; at confirmatory n the whole curve vanishes, the h = 8 point included | every width that ran, absent widths named | code |
| 3 | §10 arm set; stats.py:72 `CONFIRMATORY_ARMS`; worker `--arms`; `test_stage_arm_sets` | no h = 16/4/2 arm at confirmatory; survival curve "with bounds" exists only at the n = 6 pilot, where the 500th/9,500th order statistics of the median bootstrap are near the sample min/max | §7's survival curves with bounds at confirmatory n | **document: amendment 3** |
| 4 | pilot.py:110 `MEDIAN_KEYS` | no survival key; stage 7 publishes no survival medians (per-anchor ratios sit only inside 05-spreads anchors[].quantities) | §9 stage 7 "development medians, for the record" over every §6 quantity | code |

Not a defect: per-anchor paired SEs (§6 head) are for per-example loss/accuracy differences; a survival ratio has
no per-example pairing, so across-anchor bounds (defect 1) are its interval. Single-arm gains (G_tcool, r_aff,
G_rtemp, G_rcal, delta_acc) do have per-example pairings against N0 and could carry a per-anchor SE cheaply; noted,
not required by §7.

## Cost of amendment 3 (option A below), §8 nominal kernels

- sliced h = 16/4/2: inside the per-anchor "cuts, folds, evaluations" unit already booked; 0 extra.
- refit h = 16/4/2: one 800-step single-rate run each, §8 "3.2 s per 800 steps" + 0.04 s evaluation ≈ 3.3 s each,
  ≈ 10 s per anchor (costs.py `grid:refit-single`, measured in the pilot's timing child).

| n | extra seconds | share of the 43,200 s ceiling |
|---|---:|---:|
| 32 | ≈ 320 | 0.7 % |
| 100 (block cap 7599) | ≈ 1,000 | 2.3 % |

The pilot's stage 6 projection would carry the measured kernel automatically once the arms are in CONFIRMATORY_ARMS.

## Options for defect 3

A. **Amendment 3 adds M-wide→{16, 4, 2}, sliced and refit, to the §10 confirmatory set** (recommended): keeps §7
   literally; ≈ 1 % of the ceiling; no verdict logic touched.
B. Amendment 3 rewrites §7 line 429: survival curves reported with development-only bounds at the pilot, and the
   h = 8 point alone with bounds at confirmatory n. Cheaper, weaker, and the n = 6 bounds say almost nothing.

Either option is an amendment: §9 "What it may not change" lists the arm definitions, and §10's set is "taken, J7".

## Consequences either way

- Defects 1, 2, 4 are code; the fix changes `source_sha256` (src scripts tests pyproject.toml), so the admission
  certificate (978850a, 47169fc0…) is void for the pilot and must be reissued: new implementation commit, quality and
  runtime evidence re-run, per-item reviews for stats / confirmatory / pilot, gate B again (PR, John's merge word).
- Amendment 3 also changes the protocol identity: identity.py composes prereg + STEERING + amendments 1 and 2 by
  name; adding amendment 3 means a new `protocol_sha256` and, under the Gate 3 v4.1 rule, a new version string
  (m0-v4 → m0-v4.1). Pinned documents are never edited; the amendment is a new file with its own hash.
- Pending gate-B decision 2 (S-static has no N0 rule of its own) can ride in the same amendment: one document,
  one hash, one re-certification.
- Pilot go moves after the reissued certificate. No GPU time is lost.

## Recommended path

1. John: amendment 3 = option A (+ decision 2's S-static clause if wanted). Draft: amendment-3-draft.md beside this memo.
2. Branch off main: amendment file; identity.py composition (+ test); stats survival partial curve (defect 2);
   confirmatory bounds + sign test for every §6 quantity defined at n (defect 1); pilot survival medians (defect 4);
   CONFIRMATORY_ARMS + arm-set tests + cost tests; TDD, trust policy, fingerprints.
3. Evidence, reviews, certificate reissue; PR; gate B merge on John's word; then the pilot go.
