# J6: Pilot ceiling, A (3 GPU-hours, existing constant) or B (4 GPU-hours)
Agent: program-design-architect   Date: 2026-10-13

## Options considered
A, 10,800 s (`PILOT_SECONDS`, contracts.py:47). B, 14,400 s, a new constant.
Dropped: carrying the Gate 3 remainder (4,593 s; execution-summary.json
`remaining_gpu_process_seconds`): pilot.py:135 only requires the opening charge
to be below the ceiling, so a carried pilot starts and dies mid-core, and any
carry variant needs the same code change as B. Dropped: 3.5 h, which costs the
same code change as B for 1,800 s less cover.

## Recommendation
B: a fresh 4 GPU-hour pilot ceiling (14,400 process-seconds), added as a new
M0 constant, with the §8 overrun rule unchanged and read strictly (stop at the
first extended arm that does not project to fit).

## Strongest case for
The cost of a stop is asymmetric. The supervisor is a hard stop (pilot.py:150-155
derives the batch deadline from `PILOT_SECONDS - charged`; :278 raises on
overrun) and an attempt never resumes (pilot.py:162). An early stop at the
§9 step 4 projection costs ~2,900 s (steps 1-3 plus the first R-aff anchor);
a late overrun found during step 5 burns up to 10,800 s, because step 4
measures only the front-re-running kernel while 4,440 s of core (module
grids, M-small-B, M-wide->8 refits, R-272s) still runs on the new, unmeasured
oracle-grid fitter (P1 uses the unchanged Gate 3 fitter and is only a proxy).
Either way the core costs ~8,800 s again under a raised ceiling; unspent
ceiling costs nothing. Break points on booked §8 lines
(core 8,800; extended cumulative 900, 1,260, 1,620, 1,830, 2,400, 3,120):
- A: core alone survives to 1.227x; core + R-all to 1.113x; core + R-all +
  M-exp + lottery + F(0.75) only to 1.016x (10,630 of 10,800). At the 30
  percent stress the pilot stops before V1 and John chooses, with nothing
  resumable. Realistic A delivery: core + R-all, perhaps M-exp.
- B: core survives to 1.636x; the full extended set to 1.208x; at 1.3x the
  pilot still delivers core + R-all + M-exp + lottery + F(0.75) and drops the
  cooled F(1.0) and the 0.5-host Q1 pair by the rule.
The variance is structural, not kernel noise (hosts 44.2-44.9 s, oracle jobs
101-105 s in ledger-044.json): the unmeasured front re-run factor, a new
fitter, process count and scheduling. A 30 percent stress is the right test.
Wall cost of B over A is ~30 minutes on two cards.

## Strongest case against
B is a code change and the ceiling then binds nothing at nominal (11,900 of
14,400), so the priority order never exercises and the pilot reports no
"what was worth the money" signal. The "A is existing code" advantage is
thinner than it reads: §2 (plan :60-88) makes every M0 worker action new code,
so the driver is new either way and imports one constant.

## Assumptions verified
- Front re-run 3x: host training is batch 32, 12,500 steps in ~37 s
  (host.py:151-156; execution-summary `host_costs`) = ~3 ms per full
  forward/backward step; a cached-site minibatch step is 3.209 s / 800 =
  4.0 ms (screen.py:187-189, :218; 01-timing-replay-decision.json). A batch-256
  front re-run adds about one latency-bound host step: 3x is plausible, 4x is
  the defensible worst case (R-aff 1,500 -> 2,000 s). §9 step 4 measures it
  before admission, so the exposure is the projection code, not the number.
- Replay 900 s: attempt 4 charged 883 s = replay 3 x 38.5 + timing 3 x 8 +
  co-tenancy fit 6 x 21.8 + co-tenancy hosts 6 x 102 (ledger-044.json). About
  613 s re-proves a host trainer §2 carries unchanged. That is a ~600 s lever
  for V0 scope (John's call, not a delivery call); it would make A less
  marginal but not safe.
- Scheduling is a hidden ledger assumption: the supervisor charges wall per
  process (pilot.py:254) and allows 3 per card (:146); at 3 per card a host
  charged ~102 s against ~44.5 s solo. Every §8 kernel is a 1-per-card number,
  so the M0 pilot must run at most one process per card and the §9 step 4
  projection must use 1-per-card kernels. Wall is then about half the charge.

## Dependencies on other decisions
J4: moving F(0.75) or the cooled point into core (+780 s) makes A fail at
nominal. J-host: a 0.5-host primary roughly halves host lines but not grids.
J1/J2 do not touch the pilot charge.

## Evidence
docs/plans/2026-10-13-capacity-frontier-screen.md:60-88, :211-217, :307-352,
:354-377, :422; src/sultai_screen/pilot.py:135, :146, :150-155, :162, :254,
:278, :514, :517, :569; src/sultai_screen/contracts.py:47;
src/sultai_screen/host.py:129-131, :151-156; src/sultai_screen/screen.py:185-227;
docs/results/gate3-pilot-2026-10-13-linked-3/{ledger-044.json,
01-timing-replay-decision.json, execution-summary.json}.

## Versus the recorded §12 default
Differs (default A). The §8 lines reproduce, but A leaves 2,000 s of headroom
against an unmeasured factor on the largest core line, and a stop is not
resumable; the expected cost of A exceeds B's worst case.

## What would change this recommendation
John trims the co-tenancy replay from V0 (-600 s) and accepts realistic
delivery of core + R-all only; or the implementation plan adds resume-from-
ledger so a stop no longer re-charges the core.

## Implementation obligations
Add the constant; do not edit `PILOT_SECONDS`, which pilot.py:514, :517, :569
check Gate 3 ledgers against exactly. §8 wording: declare stop-at-first-miss.
Measure one module grid on the new fitter before the step 4 projection, or at
least before fanning out the 42 grid cases. The §9 step 4 projection is new
code; it needs its exact-source review before the pilot reads it (the one
acceptance gate §8 does not name).

## Confidence
high: every break point is arithmetic on measured kernels the earlier re-review
also reproduced; the only judgement is how much tail to buy, and the no-resume
rule makes that judgement one-sided.

---

# J7: Confirmatory arm set, floor and ceiling
Agent: program-design-architect   Date: 2026-10-13

## Options considered
(i) §10 set without S-static, n = max(32, n*), 12 GPU-hours. (ii) §10 set with
S-static, 14 GPU-hours. Dropped: 12 h with S-static, which does not survive a
30 percent overrun at n = 32 (below); 16 h (pilot.py:352, :825) is Gate 3's
`sizing_receipt` constant and over-provisions by a third.

## Recommendation
(i): the §10 confirmatory arm set, S-static pilot-descriptive only, floor
n = 32 (the existing `anchor_count` floor, contracts.py:186), ceiling 12
GPU-hours (43,200 s) as a hard stop [proposed: nothing enforces a confirmatory
ceiling today; `sizing_receipt` only emits `within_ceiling` (pilot.py:352-353)
and the supervisor is bound to `PILOT_SECONDS` (:150, :268), so this needs a
new additive constant and supervisor instance]; the pilot receipt projects the charge
with and without S-static from measured 1-per-card maxima, and S-static enters
at the confirmatory go only if John raises the ceiling to 14 h.

## Strongest case for
Per anchor, summing §8 lines for the §10 set (host 45, T-cool 25, N0/
calibrations/evaluations 30, five module grids 300, two M-wide->8 refits 120,
M-small-B 120, R-272s 80, R-aff 250) gives 970 s, matching the stated ~965;
S-static adds host 50 + refit 60 = 110. §8's confirmatory line has no fixed
term for V0; add ~300 s (replay on both cards 115 s, grid replay and known
answers), ~900-1,500 s if V0's B3 >= rho A3 check needs the six development
hosts (270 s) and the P1 instrument (340 s) again or the co-tenancy replay is
repeated. At 1,500 s fixed, n = 32 survives to ~1.33x without S-static and
~1.20x with; the conclusions below hold.
- n = 32 without S-static: ~31,340 s (8.7 h); survives to 1.38x. Covers the
  30 percent stress. Largest nominal n under 12 h: 44.
- n = 32 with S-static: ~34,860 s (9.7 h); survives only to 1.24x. At 1.3x it
  needs ~45,300 s, over 12 h. Largest nominal n: 39. At 14 h it survives to
  1.45x and covers n up to 46.
The floor binds whenever the pilot's per-anchor D1 SD is <= 0.0275 nats
(n* = ceil((1.645 x 1.25 x sd / 0.01)^2), contracts.py:183-186); n = 44 needs
sd <= 0.0323. The only analogue in the receipts, attempt 4's P6 - P5 spread
(SD under 0.01 at w4 and w2, execution-summary.json), suggests the floor
binds, but D1 on an unimpaired host is a different quantity: the SD is the
unknown the pilot fills (§9 step 5).

## Strongest case against
D5 (design-in versus insertion) is on-theme for Sultai and reads only at
confirmatory n (§7, plan :284-289); leaving S-static out means the pilot's six
static hosts never become a verdict. The reply is cost and power: +11 percent
per anchor for a reading §7 itself labels underpowered against seed-to-seed
spread, and the ceiling stops covering the stress case.

## Dependencies on other decisions
J1: n* scales with 1/X^2; X = 0.01 roughly quadruples n* and no ceiling here
covers it. J3: M_s and S-static inclusion go together. J-host: a 0.5-host
confirmatory changes every host line. J6: both ceilings rely on the same
1-per-card scheduling constraint.

## Evidence
docs/plans/2026-10-13-capacity-frontier-screen.md:284-289, :307-352, :364-371,
:379-388, :423; src/sultai_screen/contracts.py:183-186; src/sultai_screen/
costs.py:1-55 (projection by observed maxima, `is_confirmatory_measurement:
False`); src/sultai_screen/pilot.py:146, :254, :341-355, :352, :825;
docs/results/gate3-pilot-2026-10-13-linked-3/{ledger-044.json,
execution-summary.json}.

## Versus the recorded §12 default
Matches the default (§10 set, n >= 32, 12 h) and adds what the default leaves
implicit: S-static out by default because 12 h does not cover it under
stress, a fixed V0 term, and the projection method (pilot maxima, costs.py
style, not §8 bookings).

## What would change this recommendation
Pilot per-anchor D1 SD above ~0.032 nats (n* > 44): then 12 h fails without
S-static and John must choose between a higher ceiling and reading at the
achieved half-width. A measured per-anchor kernel above ~1,250 s. John
weighting D5 highly: then (ii) at 14 h.

## Confidence
medium-high: the per-anchor figure reproduces from measured lines; the
coverage arithmetic is exact; the uncertainty is the D1 SD, which nothing in
the receipts measures directly.
