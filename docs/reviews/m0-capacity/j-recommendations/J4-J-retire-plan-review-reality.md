# J4: Frontier widths and the 20-epoch cooled point
Agent: axiom-planning:plan-review-reality   Date: 2026-10-13

## Options considered
(a) {0.5, 1.0, 1.5} core; 0.75 and 20-epoch extended (author's default).
(b) {0.5, 0.75, 1.0, 1.5} all core.
(c) {0.5, 1.0} only.
(d) 20-epoch cooled 1.0 in core in place of a width.
Dropped nothing; (c) and (d) are viable but dominated on the numbers below.

## Recommendation
Keep (a), with one ordering change: move the 20-epoch cooled F(1.0) ahead of F(0.75) in the extended priority list (§5, §8), and name its schedule in §2/§5.

## Strongest case for
Source facts verified. `Host(width)` accepts only bottleneck widths (host.py:59-60); tail widths are hard-coded (host.py:63-70); `train_host` raises on `epochs != 10` (host.py:129-131); the train action saves only `host.state_dict()` (screen.py:159), no optimiser state, so T-cool and every frontier host are new code as §2 says. Parameter counts reproduced on CPU by re-building the same block sequence with scaled tail widths: 85,274 / 168,826 / 284,634 / 613,018, and `Host(0)` gives 284,634 (matches intact-3000/result.json `parameters`).

Slope precision. The six 1.0 hosts' `intact_loss` (materiality-300x/result.json) have mean 0.681, SD 0.061, so each width mean has SE 0.025 (assumed constant across widths). OLS slope SE on log2(parameters): {0.5,1.0,1.5} 0.0123 nats/doubling (95% half-width 0.024); adding 0.75: 0.0120 (0.023, a 3% gain, because 0.75 sits near the centre of the range); {0.5,1.0}: 0.020 (0.040, 65% wider, no point above the reference, so the module's "fraction of a doubling" at 284,634 would be read from the starved side only); {1.0,1.5}: 0.032. The 1.5 point is the leverage point; 0.5 is required anyway by J-host (fallback screen, Q1 pair). The 0.75 point buys one residual degree of freedom (a linearity check) for 210 s: an extended diagnostic. The 20-epoch point is a different recipe and cannot enter the same-recipe slope; its unique content (parameter doubling versus compute doubling) feeds a reading §7 already labels descriptive, and T-cool (core, 150 s, paired by host) already answers "is the host unfinished". Under J6 = A the post-core headroom is about 2,000 s and R-all + M-exp + lottery + F(0.75) = 1,830 s would starve it, hence the reorder.

Cost. Bottleneck hosts add two tiny 1x1 blocks and cost 41-42 s against 35.8 s intact (bottle2-/bottle4-/intact-300x result.json `training_seconds`): training is launch-bound at 2.9 ms/step, so [inference] F(0.5) will cost near 35 s, not 25, and F(1.5) nearer 50 s than 95; the 720 s line is mis-shaped but not under-booked in total.

## Strongest case against
The frontier is "same-recipe, 10 epochs": a 1.5 host at 10 epochs may be more under-trained than the 1.0 host, so the slope partly measures recipe sufficiency, and only a longer-trained point would show it. T-cool answers that on the anchors, but unpaired from-scratch evidence is cleaner for a frontier claim. If John weights STEERING rule 2's "at matched compute" heavily, (d) is the honest reading; it costs the bracket.

## Dependencies on other decisions
J6: under B the 20-epoch point fits in either set; under A, adding it to core (about 9,370 s of 10,800 with R-aff's front kernel unmeasured, §8) raises the stop-before-V1 risk. J-host: 0.5 is not optional. J0: the precision argument is in loss units; no per-seed accuracy exists in the receipts.

## Evidence
host.py:59-71, 81-84, 89-94, 129-140, 184, 197; screen.py:121-176, 434-445; pilot.py:125-128, 684-697; contracts.py:44, 47, 82-88; docs/results/gate3-pilot-2026-10-13-linked-3/{intact,bottle2,bottle4}-300x/result.json, materiality-300x/result.json, 01-timing-replay-decision.json (`host_bitwise_replay: true`, read; pilot.py publishes it only after `verify_replay` passed on both cards); plan §3 lines 92-111, §5 lines 207-218, §6 lines 242-246, §8 lines 318-345, §11 lines 392-396; STEERING.md §4 rule 2 (lines 83-87); docs/research/claude-handoff/sultai-concept-demonstrator-design.md line 284 (starved widths 32, 32, 64, 64, so the §3 claim holds).

Hash reproduction at s = 1.0. Possible in principle: `make_host` seeds the forked CPU RNG with `derive(seed, "host-init")` and builds modules in order; Conv2d/Linear draw from that RNG in construction order, BatchNorm draws nothing. Breakers: (1) any change to module construction order or any extra parameterised module in `__init__`; (2) any change to parameter or buffer names, including nesting the tail in another container, because `state_hash` hashes `kind:name:dtype:shape` before the bytes (host.py:89-94); the sequential digest is also order-sensitive, so re-ordering registration breaks it with identical values; (3) a width tuple computed by float arithmetic that misses an integer (use explicit tuples); (4) outside the constructor: torch 2.9.1+cu128, the pinned dev split (DEV_SHA/PERM_SHA), `CUBLAS_WORKSPACE_CONFIG`, two threads, one of the two pinned cards. Also: no attempt-4 state hash is pinned in `src/` today, `verify_replay` compares two fresh receipts, so "compare to the published string" is new identity-pin code; `main()` refuses seeds outside INTACT/BOTTLENECK_SEEDS (screen.py:443), so 6000+ need the seed admission §2 lists; `_training` checks only `width` (screen.py:133), so a frontier receipt needs a tail-width field or an s = 0.5 checkpoint hits `checkpoint_state` with a hard InputDataError.

## Versus the recorded §12 default
Match on the sets. Differ on extended order (20-epoch before F(0.75)) and on a §2/§5 gap: `train_host` has no schedule at all (constant lr 0.05, host.py:135-143), so a "cooled" 20-epoch run needs its decay rule and start epoch named, not just "an epoch count other than 10"; §6 should also say whether an extended 0.75, if run, enters the slope fit.

The §9 "schedule axis (raise only)" latitude is the fit schedule, not host epochs; any host-epoch variant is an arm definition and must be decided under J4 now.

## What would change this recommendation
Redefining the 20-epoch point as the anchors resumed for 10 cooled epochs (seeds 3000-3005, the T-cool resume code): paired, roughly half the cost, bitwise-checkable at epoch 10, cheap enough for core. Or a measured F(1.5) kernel above 95 s under J6 = A.

## Confidence
medium: parameter counts and code facts are verified; the slope SEs assume the 1.0 host's seed SD holds at other widths.

# J-retire reality check
Agent: axiom-planning:plan-review-reality   Date: 2026-10-13

## Finding
The P1 *mathematics* has no retired dependency: the `P1` branch of `make_impairment` (impair.py:46-48), `ridge`, `oracle`, `mean_loss`, `identity_check` (fit.py) and host.py/data.py/alignment-free paths reference no retired family (grep of fit.py, host.py, data.py: none). But the P1 *instrument as executed* is interleaved with retired families at four places, so J-retire is a subtractive code carve-out plus a test rewrite, not a documentation change:

1. `materiality_worker` iterates `MATERIALITY_LADDERS` = P1, P2, P3 (screen.py:54, 236-247); `magnitude_decision` holds on `P3_RETIRED` replay and raises `InstrumentHold` if P2 or P3 exhausts (pilot.py:288-319). Alpha cannot be chosen today without P2/P3 passing.
2. `posthoc_worker` runs P1, P2, P3, P3prime and computes A7/A8 via `derive_former()` -> `sultai.formation.run_formation()` in the same loop as A1/A2/A3/B3 (screen.py:254-304, 271; former_bridge.py:37-50); `check_posthoc` requires former agreement and all four families before it reads B3 >= rho*A3 (pilot.py:322-338).
3. `timing_worker` runs the B5 wide grid and B4 retrieval as timing probes (screen.py:191-209); `costs.project_cost` lines are the Gate 3 arm set (costs.py:44-57); `sizing_receipt` is P6-specific (pilot.py:341-352); `bottleneck_worker`/`cheap_worker` are P5/P6 only and dormant unless called (`wide` defaults False, fit.py:168; `cheap_bank` only from screen.py:331).
4. Tests pinning retired families: six in tests/test_screen.py (lines 133, 262, 298, 376, 709, 922, e.g. `test_all_frozen_state_and_zero_site_controls`, which uses `Host(4)` and P2/P3) and all of tests/test_v42_ladder.py; tests/test_fixed_evaluation.py uses `Host(4)` as a fixture only, so keep the bottleneck argument in the new constructor.

## Consequence
If the existing workers run unchanged, retired families are computed, published and can *hold* the M0 pilot (InstrumentHold on P2/P3, PilotHold on P3 replay), contradicting "retired from this study". The ledger's "P1 instrument 6 x 56 s" (§8) inherits the whole posthoc wall time (posthoc-300x/result.json 55 s, including former fits and three retired families); P1-only costs less.

## Correction
State in §2/J-retire that P1-only materiality/posthoc/check functions, a replacement cost projection and the test rewrite are new code (already implied by "new worker actions ... and tests", not stated); keep `former_bridge.py` and `alignment.py` off the import path of the new workers. Risk is low: nothing retired feeds the P1 numbers.
