# J1–J3 and J0 (statistical half): margins X, X₂, M_s and the unit
Agent: statistician (general-purpose)   Date: 2026-10-13
Derivations: `J1-J3-calc.py` beside this file (`python3 -I J1-J3-calc.py`).

## Honesty clause
No fitted M-aff and no seen-data fit exist. The receipts do hold a
**measured within-anchor difference of two fitted 272-parameter arms on the
same host** (P6 − P5: read site differs, not nonlinearity; impaired hosts;
held-out fit data), used here as the empirical analogue of ρ beside assumed
ρ = 0 / 0.5 / 0.9. No accuracy field exists and per-example losses are
hashed, so accuracy noise is binomial at an assumed p = 0.75 and
nats→points is an outside prior.

## Derived numbers

| Quantity | Value | Receipt field |
|---|---|---|
| SD of single-arm gain, w2 P5 / w4 P5 | 0.088 / 0.034 | `execution-summary.json .bottleneck_known_answer.w2|w4.*.P5.gain` |
| SD of intact − B3 (module below intact) | 0.061, range 0.048–0.207 | `.posthoc_P1_low_alpha.*.arms.A1 − .B3` |
| SD of intact dev loss | 0.061, span 0.164 | `materiality-*/result.json .payload.intact_loss` |
| SD of B3 / w4 P5 endpoint | 0.013 / 0.009 | same arms; the module compresses hosts to a floor |
| **SD(P6 − P5)** w2 / w4; implied ρ | **0.0097 / 0.0059**; **0.996 / 0.992** | `.bottleneck_known_answer.*.P6_minus_P5` |
| Fit-seed SD at selected rate (3 seeds, 800 steps) | median 0.0016, max 0.0085; two one-seed arms floor 0.0022 | `oracle*/result.json .payload.P5|P6.oracle.candidates[].selection_loss` |
| `selection_paired_bootstrap_se` = SE of best-200 − best-800 on the 3,000 selection examples (not arm-vs-noop, not dev) | median 0.0025, max 0.0038; ×√(3/5) → 0.0019 on 5,000 dev | `.oracle.selection_paired_bootstrap_se`; fit.py:296–297 |
| Sizing rule | n\* = ⌈(1.645·1.25·SD/(X/2))²⌉, floor 32, X = 0.02 hard-coded | `contracts.py:43,182–186`; ceiling 16 h at `pilot.py:353` |
| Sign test, n = 32 | ≥ 22/32 above X; 80 % power at a shift of 0.63·SD(D) | exact binomial |
| Accuracy SE, assumed p = 0.75 | single arm 0.79 pts (3,000) / 0.61 pts (5,000); paired diff at 5–10 % discordant 0.32–0.45 pts (5,000) | not in receipts |

hw = 1.645·1.25·SD/√n (indeterminate band X ± hw); MDE = extra margin above
X for 80 % one-sided power; σ_arm = 0.061, SD(D) = σ_arm·√(2(1−ρ)):

| ρ | SD(D) | hw n=6 | hw n=32 | MDE n=32 | n\*(.02) | n\*(.015) | n\*(.01) |
|---|---|---|---|---|---|---|---|
| 0 | 0.086 | 0.073 | 0.031 | 0.047 | 316 | 561 | 1261 |
| 0.5 | 0.061 | 0.051 | 0.022 | 0.034 | 158 | 281 | 631 |
| 0.9 | 0.027 | 0.023 | 0.010 | 0.015 | 32 | 57 | 127 |
| 0.996 (w2 measured) | 0.0097 | 0.008 | 0.0035 | 0.0053 | 4 | 8 | 16 |
| 0.992 (w4 measured) | 0.0059 | 0.005 | 0.0021 | 0.0032 | 2 | 3 | 6 |

n\* ≤ 32 iff SD(D) ≤ 0.0275 (X = 0.02), 0.0206 (0.015), 0.0138 (0.01); the
12 GPU-h ceiling at 965 s/anchor allows n ≤ 44 (SD ≤ 0.032 at X = 0.02). At
n = 6 the percentile bound of a median is degenerate; that column is a
normal illustration, and §9 rightly reads only SDs.

## J1: X for D1 (and D3)
**Options.** (a) keep 0.02; (b) 0.01, the floor under measured ρ. Deriving X
from the pilot is dropped: STEERING §4 rule 6 forbids it and (a) covers the
plausible range.
**Recommendation.** X = 0.02 nats, relabelled "smallest margin resolvable at
n = 32 for ρ ≥ 0.9, and the equivalence bound: premiums below it are
published as re-tuning equivalent; about 0.5–0.7 accuracy points by outside
prior". D3 keeps X; §7 should say it is read at its achieved half-width and
that "harvests undertraining" is likewise an equivalence call at X.
**For.** Under the measured analogue (SD(D) 0.006–0.010) the band is X ±
0.004 and both "present" and "re-tuning equivalent" are reachable at n = 32
with room; under ρ = 0.9 the design sits exactly at n\* = 32, so 0.02 is the
lowest value that does not bet the confirmatory run on ρ ≈ 0.99
transferring from impaired, held-out-fit hosts to the unimpaired, seen-data
host.
**Against.** X is not only a floor: §7 reads "re-tuning equivalent" when
UB < X, so under the measured ρ (hw32 ≈ 0.0035) a true premium of 0.015
nats gives LB ≈ 0.012, UB ≈ 0.019 and is **published as equivalent to
re-tuning**, as is anything below ≈ 0.016 (at ρ = 0.9 it reads
indeterminate instead). 0.02 is 200× what 272 frontier parameters buy, so
John is agreeing that a 0.3-point premium from 272 parameters does not open
M1. D3 is a single-arm gain, not a two-arm difference: its SD on
attempt-4 hosts is 0.034–0.088 (hw32 0.012–0.032), resolvable at X only if
T-cool compresses the host spread as B3 does (then SD ≈ 0.018, hw32 0.007).

## J2: X₂ for D2
**Options.** (a) 0.01; (b) 0.02 = X for parsimony.
**Recommendation.** X₂ = 0.01. It sets only the "helps" bar; "hurts" uses 0.
**For.** D2 is two one-seed 280-parameter arms on the same host and data,
the closest M0 analogue of P6 − P5: SD ≈ 0.006–0.010 plus the 0.002
one-seed floor, hw32 ≈ 0.0035, so 0.01 is about three half-widths. 0.02
would ask the overgrow route to beat built-small by a third of a typical
whole-module gain, which the question does not require.
**Against.** D2 is not sized (§9 step 6); at ρ = 0.9 (SD 0.027) 0.01 is one
half-width and reads indeterminate; the refit arm (wide fit → cut → refit)
carries variance P6 − P5 does not. One margin (0.02) costs only power.

## J3: M_s for D5 (confirmatory only)
**Options.** (a) M_s = X = 0.02 plus a pilot-gated admission rule; (b) no
margin, D5 descriptive forever.
**Recommendation.** M_s = 0.02, and S-static enters the confirmatory set
(J7) only if the pilot's recorded SD(D5) ≤ 0.0275 (n\* ≤ 32). That is an
admission the pilot may fill under §9; the margin itself stays fixed.
**For.** D5 has two regimes: endpoints keeping the host's seed spread give
SD(D5) = √2·0.061 = 0.086, hw32 0.031, n\* = 315, hopeless at any ceiling;
endpoints compressed as B3 give SD ≈ 0.018, hw32 0.007, n\* = 14, readable.
Only the pilot can tell which, and the rule spends 110 s/anchor only when
it can pay.
**Against.** A conditional arm invites a "dropped because it looked bad"
reading; the pre-stated SD threshold, read before any median, is the
defence. A six-anchor SD carries about ±40 % sampling error (5 df), so this
threshold and the n\* ≤ 32 boundary are soft: a pilot SD near either is a
John-decides case, not a mechanical pass.

## J0, statistical half
**Recommendation.** Mean dev cross-entropy primary; accuracy as guard.
**For.** Per-anchor paired loss SE on 5,000 dev ≈ 0.002–0.004 nats (the
B3 contrasts gave 0.0034–0.0038 on 3,000); paired accuracy SE ≈ 0.3–0.45
pts. At an effect of 0.02 nats ≈ 0.5–0.7 pts the loss SNR is 5–10, the
accuracy SNR 1.1–2.2: a variance ratio of 5–80× (script), so an
accuracy-primary confirmatory needs n in the hundreds. Between-anchor spread (0.006–0.010) exceeds per-example SE, so anchors, not
dev examples, buy precision. The guard (acc(M-sq) − acc(N0), whole
module vs no-op, hw32 ≈ 0.18 pts at a 0.4-pt anchor SD) binds only when the
loss gain is pure calibration, which is its job.
**Against.** Accuracy is John's stated currency; the guard, not the unit,
carries that defence, and the guard's SD is assumed, not measured.

## Dependencies
J0 accuracy-primary re-expresses all margins in points and multiplies n.
J7's 12 h ceiling bounds SD(D1) ≤ 0.032 at X = 0.02. X ≠ 0.02 or a 12 h
ceiling is a code change (`contracts.py:43`, `pilot.py:353`); X₂ and M_s do
not exist in code.

## Versus §12 defaults
Match on all three values; differ on labelling (X is the ρ ≥ 0.9 resolvable
floor; D3 at achieved width) and on J3's admission rule.

## What would change this
The real fork: if John would open M1 on a 0.01-nat (≈ 0.3-point) premium,
X must drop to 0.01, which bets the run on ρ ≈ 0.99 transferring (n\* ≤ 32
iff SD(D1) ≤ 0.0138); if he would not, 0.02 stands with the equivalence
consequence named. Pilot SD(D1) > 0.0275 (ρ < 0.9): X = 0.02 is unreachable
at n ≤ 44 and John chooses ceiling or n = 32 at stated width.

## Confidence
Medium: the ρ transfer from impaired, held-out-fit hosts to the unimpaired,
seen-data host is the one assumption not in the receipts.
