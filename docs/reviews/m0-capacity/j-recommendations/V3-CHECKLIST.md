# v3 recut checklist (SYNTHESIS §1 decisions and §5 defects)

## Decisions (SYNTHESIS §1)
- [x] J0: loss sole primary; ΔAcc labelled secondary, conditions no verdict (author's resolution of the split under John's clarification; flagged in §12)
- [x] J1: X = 0.02, relabelled resolvable-at-32 / equivalence bound; D3 at achieved half-width
- [x] J2: X₂ = 0.01 ("helps" bar only; "hurts" at 0)
- [x] J3: M_s = 0.02; S-static enters confirmatory only if pilot SD(D5) ≤ 0.0275
- [x] J-route: M-aff kept; fold residual measured, tolerance X/4 = 0.005; M-aff on five-rate grid; R-272s = channels 0–15 of tail[0] (256 taps + 16 biases)
- [x] J-D2: matched-budget pair kept; unit = optimiser steps + example passes; 4.3× parameter-step ratio disclosed; M-small-B checkpoints 1,600 vs 6,400; refit step-0 selection loss and rate flag beside D2
- [x] J-host: reference host primary; 0.5-host extended pair enlarged to N0/calibrations, M-sq, M-aff, T-cool, M-sq|T-cool; first in extended order; provenance flag (new arms)
- [x] J4: {0.5, 1.0, 1.5} core; 20-epoch point ahead of F(0.75); decay rule named; keeps out of the same-recipe slope
- [x] J6: B, 14,400 s as a new M0 constant; PILOT_SECONDS untouched; stop-at-first-miss
- [x] J7: §10 set; n = max(32, n*); 12 h new hard-stop constant + supervisor instance; S-static out unless 14 h; fixed V0 term; projection from measured one-per-card maxima
- [x] J-retire: retire as §2; mechanism = code carve-out (P1-only workers/checks/cost projection/tests; former_bridge/alignment off import path)
- [x] J-D02: principle recorded (0.5 is not near-empty; near-empty = s ≤ 0.25); five recordings; F(0.25) in extended; M2 seed block reserved
- [x] J9: separate dated SHA-pinned amendment drafted (gate mapping pending John's acceptance at freeze); STEERING §6 → link; identity carries amendment sha; README pointer
- [x] J10: pending John (unchanged; §12 row)

## Draft defects (SYNTHESIS §5)
- [x] fold exact only in interior; 16-dof border bias; "folds to 0" reworded in §1, §3, §5
- [x] R-272s redefined to a subset that exists
- [x] (33h+16 = 272 was the caller's conflation; no change)
- [x] M-small-B plateau checkpoints stated
- [x] accuracy evaluator added to §2 new code
- [x] confirmatory ceiling: new constant + supervisor instance in §2; §10 hard stop
- [x] one-process-per-card assumption stated for every §8 kernel and §9 step 4 projection
- [x] main() seed refusal → seed admission in §2 new code
- [x] attempt-4 host hash pins in src/ → §2 new code
- [x] §2 "P1 carried unchanged" corrected; P1 line note in §8
- [x] 20-epoch decay rule; _training width check; 0.5 host launch-bound (35 s) and 1.5 (85 s) reshaped, 720 s total; §6 says which widths enter the slope
- [x] affine SGD path in §2 new code
- [x] confirmatory projection fixed V0 term (§8, §9 step 6)
- [x] identity carries amendment_sha256 (§11)
- [x] Appendix A reserves an M2 starting-host block; 6024–6029 for F(0.25)
- [x] "matched budget" unit named (§4, §6, §7)

## Outside the draft
- [x] docs/plans/2026-10-13-gate-ladder-amendment.md written (status: drafted, pending acceptance)
- [x] STEERING §4 rule 1 (J0 taken, author's resolution), §5 M0 row (no guard), §6 → link
- [x] DECISIONS D27 row
- [x] README pointer to the amendment
- [ ] tracker comment, memory (J0) — after the re-review
- [x] RESUME
- [ ] fresh targeted re-review of v3 against this checklist, preserved and dispositioned

## Targeted re-review of v3 (5c664a1) and dispositions
Report: `v3-targeted-re-review.md` (plan-review-reality, fresh reader). All 14 decisions and 16 defects found implemented; links, seeds, identity documents and J0 consistency clean. Eight findings, all adopted in the commit that follows: confirmatory itemisation (four nonlinear grids, 240 s, so the 1,010 s per anchor holds); overrun multipliers recomputed on the v3 ledger (1.19×, 1.09×, 1.59×) and "part of the rest" replaced by the stop at R-all; the confirmatory-ceiling source claim corrected (a pre-run `FeasibilityHold` at 16 h / 200 anchors exists; no supervisor stops the run); amendment row 2 now cites the Nyx final certificate ("complete with retained execution qualification") and notes the development certificate's "partial"; F(0.25) booked launch-bound at 210 s (extended ≈ 4,260 s, combined ≈ 13,300 s); S-static margin 1.16×; "reaches" rather than "reaches below"; V2 names its budget unit. Verdict after the edits: ready to freeze pending John's acceptance of the amendment and J10.
- [x] fresh targeted re-review of v3 against this checklist, preserved and dispositioned
