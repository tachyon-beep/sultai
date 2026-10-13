# J0: Primary unit of the M0 endpoint
Agent: axiom-product-management:product-decision-critic   Date: 2026-10-13

## Options considered
- **A. Loss-only primary.** Mean dev cross-entropy in nats; ΔAcc reported beside with its own bound as a labelled secondary, not a verdict condition.
- **B. Loss with the accuracy guard** (the §12 default): "present" also requires the lower bound of median ΔAcc > 0, else "calibration, not capacity".

Dropped: *accuracy-only* (no arm optimises it, no code computes it, discontinuous, and John has withdrawn the mandate); *nats per retained parameter / fraction of a doubling as primary* (D1 and D2 are already matched-count contrasts, so normalising changes nothing, and the ratio to 272 was removed for cause in S10; the frontier fraction stays descriptive); *Brier or other proper score* (cross-entropy is already strictly proper and is what the fitter minimises; a second proper rule adds code without separating anything the arms separate).

## Recommendation
Option A: mean development cross-entropy in nats is the sole primary unit for V1 and V2; ΔAcc is reported per arm with a one-sided bound and the label "loss gain without accuracy movement" when its bound is ≤ 0, but it conditions no verdict.

## Strongest case for
1. **It is what the instrument measures and the arms optimise.** `losses()` returns per-example cross-entropy only (src/sultai_screen/fit.py:55–69); `paired_se` bootstraps those differences (fit.py:247). No accuracy or argmax code exists anywhere under `src/` (grep: zero hits), and none of the attempt-4 receipts carries accuracy (zero hits under docs/results/gate3-pilot-2026-10-13-linked-3/). The guard is therefore new code, and it is absent from the §2 new-code list (plan lines 73–87), which is a freeze defect in the default whichever way J0 goes.
2. **The guard is an unsized verdict condition.** §9 step 6 sizes the confirmatory n on D1 only (plan:366–368). The default makes "present" depend on a second bound nobody has powered. Without measured accuracy I can only estimate (assumption, binomial): a paired ΔAcc on 5,000 dev examples has SE ≈ √(f/N) with discordant fraction f; at f = 4 % that is ≈ 0.3 pp per anchor, against an accuracy effect for a 0.02-nat gain that is plausibly of the same order. A real capacity gain could then read "calibration, not capacity" by noise. Across ≥ 32 anchors the median bound tightens, but the condition remains unsized and unfalsifiable as written.
3. **Calibration is already separated structurally.** N0 is the BN-recalibrated host, every gain is relative to N0, R-temp/R-cal are reported, and D1 is netted against M-aff, an equal-count affine module (plan §4–§6). The guard's residual job is only to deny credit for example-dependent confidence shaping, and under a capability-per-parameter objective that is a genuine probabilistic improvement the parameters bought.
4. **Ladder consistency.** M1 (recovered fraction of the SGD-fit gain), M2 and M3 (frontier excess in nats) are all in loss (docs/STEERING.md:105–107); V2 in the default has no guard (plan:292). Option A makes V1 match the rest.
5. **User evidence.** John's objective sentence says "accuracy" once; his clarification says accuracy "was just a value we used in esper, its not 'the value we have to use'". Esper-lite itself offered `reward_family = "loss"` beside the accuracy-minus-rent form (/home/john/archive/esper-lite/README.md:242,255; docs/product/vision.md:27). Loss is not a departure from the baseline theory.

## Strongest case against
A publication reader who hears "accuracy" will ask whether argmax moved. A loss-only "present" can be earned by a nonlinear module that reshapes confidence on the same hard examples without changing a single decision; that is a weaker claim than "the host classifies better". Keeping the guard also keeps the reviewers' S1/A6 condition in verdict form rather than as a label, and the pilot pays nothing for it (dev eval is 0.038 s). If John wants an accuracy claim in the first publication, B is the honest choice and then ΔAcc must be sized (§9 step 6) and added to the §2 new-code list.

## Dependencies on other decisions
- **J1 (X)**: under A, X is the only V1 margin; its 0.02-nat "detectability floor" label stands (plan:268–270; sci review item 18). Under B, J1 needs a companion accuracy margin or the ΔAcc bound stays an unpowered > 0 test.
- **J-route**: A relies on M-aff being the D1 comparator so that affine calibration is netted; if J-route moves D1 to R-272s/R-aff, the case for some calibration label strengthens.
- **J7**: if B, n must be sized on ΔAcc too, which can raise n and the 12 GPU-hour ceiling.
- **J2/J3** unaffected (already loss-only).

## Evidence
- src/sultai_screen/fit.py:55–69 (`losses`, `reduction="none"`), :78 (`mean_loss`), :247 (`paired_se`).
- `grep -rn -i 'accuracy\|argmax' src/` → no output; `grep -rl -i accur docs/results/gate3-pilot-2026-10-13-linked-3/` → 0 files.
- docs/results/gate3-pilot-2026-10-13-linked-3/oracle*/result.json: `selection_paired_bootstrap_se` 0.0007–0.0030 nats on the 3,000 selection examples (24 fits); gains 0.037–0.260 nats; no accuracy field.
- docs/plans/2026-10-13-capacity-frontier-screen.md:73–87 (§2 new code; no accuracy evaluator), :219–226 (§6 ΔAcc), :266–275 (V1 conditions), :292 (V2, no guard), :364–368 (sizing on D1 only), :414 (J0 default).
- docs/STEERING.md:75–82 (rule 1), :104–107 (ladder units).
- docs/CONCEPT.md:11–16 (formalisation attributed to author, pending J0).
- docs/reviews/m0-capacity/v1-dispositions-2026-10-13.md:19 (A6), :32 (S1); scientific-review-v1:29, :63; alignment-review-v1:30.
- Esper-lite README.md:242, :255; docs/product/vision.md:27.

## Versus the recorded §12 default
Differs: the default keeps loss as primary but makes ΔAcc a "present" condition. After John's clarification the accuracy guard has lost its user-evidence basis, is unsized, is new code not listed in §2, and is inconsistent with V2 and rungs M1–M3. I keep the ΔAcc co-read as a reported, labelled secondary.

## What would change this recommendation
John stating that the published claim must be in accuracy; or a pilot ΔAcc per-anchor SD small enough (≤ 0.1 pp) that the guard is cheap to power at n ≥ 32; or J-route moving D1 off M-aff, which would re-open the calibration confound the guard was meant to close.

## Confidence
Medium-high: the instrument and receipt facts are verified; the accuracy-noise figures are a binomial estimate, not a measurement, because no accuracy exists in any receipt.
