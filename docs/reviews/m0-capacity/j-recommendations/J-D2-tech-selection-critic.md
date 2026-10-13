# J-D2: Which pair of arms defines D2 and the V2 "add-and-remove helps" verdict
Agent: axiom-solution-architect:tech-selection-critic   Date: 2026-10-13

## Options considered
1. **Matched budget (author's default):** D2 = G(M-wide→8, contribution rank, refit on the full module grid) − G(M-small-B at {400, 1,600, 6,400}); as-sliced-with-fold versus M-small (D2s) as the strict secondary.
2. **Strict as-sliced versus M-small as primary:** steps-matched (9,600 each) but biased toward "hurts" by construction, since nothing repairs the cut.
3. **Matched parameter-steps:** M-small-B at about 8.6× the schedule (27,520 steps per rate). Retired on evidence: wall per step is flat in module width, so this equalises a quantity that costs no GPU time, at about 2,320 s extra.
Dropped: refit versus M-small unmatched (2× steps to the overgrow route; already dispositioned S7/A3). Lottery cut as primary (asks "inherited weights or lucky selection", a mechanism question, not the route contrast; stays extended).

## Recommendation
Keep the default pair, and add one sentence to §4/§6/§7 naming the matched unit: "matched budget" means matched optimiser steps and example passes (19,200 steps, 4,915,200 passes per anchor each side; measured wall equal), with the parameter-step ratio (23.1M versus 5.4M, 4.3×) disclosed in the ledger and not matched. Also name M-small-B's plateau comparator as 1,600 versus 6,400 steps.

## Strongest case for
- STEERING §5 asks the V2 question "at matched budget"; this candidate matches budget in the unit that costs GPU time. Attempt-4 timing: 200 steps of the 272-parameter module 0.9369 s; 400 steps of the h = 64 module 1.8545 s, i.e. 4.68 versus 4.64 ms per step. Cost is the host forward/backward through `classify`, not the module einsum, so matched steps = matched wall = matched FLOPs, which is Liu et al.'s Scratch-B convention.
- "Twice the schedule" checks: M-wide grid 3 × 3,200 = 9,600 steps plus refit 9,600 = 19,200 steps and 4,915,200 example passes; M-small-B 3 × 6,400 = 19,200 and 4,915,200. Parameter-steps: 20,428,800 + 2,688,000 = 23,116,800 against 5,376,000 (4.3×); D2s is 7.6× in that unit. Both units are booked, so the ratio is disclosed, not hidden.
- Refit is faithful to the route: Sultai overgrows, cuts and keeps training; the plug-in route is built small. A verdict on the as-sliced module would mostly grade the cheap ‖V‖·RMS criterion, which the survival curves, random cut and leave-one-out ceiling already do descriptively.
- Zero cost, fully reversible before freeze: every arm in every candidate pair is already in the core (§5) and confirmatory (§10) sets; the two D2 ledger lines (720 + 720 s) plus M-wide and M-small inside the module-grid line, and the §2 new-code list, are identical under any choice. It is a labelling decision; pick the honest contrast. The 4-tensor `Adapter.apply` is already h-agnostic, so the as-sliced secondary evaluates through existing code.

## Strongest case against
- Search asymmetry, not budget: the overgrow route draws 64 units and keeps the best 8, then selects twice (two grids of 9 checkpoint candidates = 18 selection events on the 3,000 held-out examples versus 9 for M-small-B). A longer single-seed schedule cannot buy that lottery, and the arm that separates it is extended, so a pilot "helps" can be a selection effect.
- A full refit grid at rates up to 1 for 3,200 steps may wash out the inherited weights, so D2 ≈ 0 reads "indeterminate" trivially. Guard without new code: the refit's step-0 selection loss is the as-sliced loss, and the selected rate/step with the edge-of-grid flag is already printed; report them beside D2.
- Plateau asymmetry: a warm-started refit near its floor certifies easily; M-small-B from N(0, 1e-3) may not (attempt 4 certified 2 of 30 at 800 steps). Correct direction under the control-must-certify rule, but it pushes V2 toward indeterminate.

## Dependencies on other decisions
- J2: X₂ = 0.01 is sized for this pair; the strict pair is expected negative by construction and would need a re-derived margin and sign logic.
- J6: unaffected by the default; option 3 adds about 2,320 s and pushes the core past ceiling A (8,800 + 2,320 > 10,800).
- J7: the confirmatory set already carries both refits, M-small and M-small-B; no change.

## Evidence
- docs/STEERING.md:104 ("at matched budget", unit unnamed), :91 (rule 4 books example passes and parameter-steps), :113 (terminal-truncation precedent).
- docs/plans/2026-10-13-capacity-frontier-screen.md:173–175 (warm start; M-small-B "twice the schedule ... total optimiser steps"), :160–166 (3-rate grid, one 3,200-step run per rate, comparator 800 vs 3,200), :76 and :166 (the comparator is stated only as 800 vs 3,200; which checkpoints certify M-small-B is never stated), :202 (M-wide→8), :204–205 (M-small, M-small-B), :235 (D2, D2s), :290 (V2), :325–326 (ledger lines), :419 (J-D2 default).
- docs/results/gate3-pilot-2026-10-13-linked-3/01-timing-replay-decision.json: fit_1x_seconds 0.9369 (200 steps, 272 params), wide_grid_seconds 1.8545 (2 rates × 200 steps, h = 64); replicated in timing-0-b/result.json and timing-1/result.json.
- src/sultai_screen/screen.py:185–196 (timing worker: wide grid is 2 rates × 200 steps); src/sultai_screen/fit.py:176–177 (`wide=True` hard-codes (64,16)/(16,64), no warm start), :25–30 (4-tensor apply is h-agnostic), :280 (`oracle(seeds=1|2|3)`), :227 (parameter-steps = numel × steps), :198 (example passes per 256-example minibatch step).
- src/sultai_screen/contracts.py:47 (PILOT_SECONDS = 10,800).
- docs/reviews/m0-capacity/scientific-review-v1-2026-10-13.md:41 (S7), alignment-review-v1-2026-10-13.md:24 (A3), v1-dispositions-2026-10-13.md:16,38.
- Arithmetic recomputed in a scratch script from the numbers above; matched-parameter-step schedule 27,520 steps per rate ≈ 387 s per anchor at 4.68 ms per step.

## Versus the recorded §12 default
Match. The default is the only candidate that matches the unit that costs GPU time while keeping the refit the Sultai route implies. Differ only in wording: §6/§7 and STEERING §5 say "matched budget" without the unit, and the 4.3× parameter-step ratio is booked but not named; state both, and name M-small-B's comparator checkpoints.

## What would change this recommendation
- John reads "budget" in rule 4 as parameter-steps: then option 3, with ceiling B or a dropped extended arm.
- John prefers to equalise search rather than schedule: 2 seeds × {200, 800, 3,200} for M-small-B spends the same 19,200 steps (`oracle(seeds=2)` exists); I would still not switch, because the single-seed convention runs through every arm.
- If the extended lottery cut matches the contribution-rank refit at the pilot, "helps" is a selection effect and the confirmatory pair should add the lottery arm.

## Confidence
medium-high: the budget arithmetic and the flat wall-per-step are measured; the residual risk is the search asymmetry, which no budget definition removes.
