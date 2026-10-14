# M0 pilot: training-dynamics review of the module rate grid and plateau comparator

Training diagnostician, SME protocol; read-only over `docs/results/m0-pilot-2026-10-14-0d4179a/` (commit
0d4179a, m0-v4.2); nothing run on GPU. Grids recomputed from `{modules,maff,retune}-<seed>/result.json`
(`payload.arms.<arm>.report.candidates[].{rate,steps,selection_loss,training_losses}`) and cross-checked against
`05-spreads.json` (`quantities.arms`, `plateau_fractions`, `harm_inventory`, `quantities.survival`) and `04-schedule.json`.

## Summary

The instrument is not clipping on the module arms: the loss surface is a ridge in rate×steps (1.0@200 ≈ 0.3@800 ≈
0.1@3200), flat within 1–3 paired SEs on most anchors, so "edge" selections are the ridge's two ends, not a
monotone run-off. The plateau comparator behaves as §4 specifies; its training-tail clause never binds, and
"certified" means only that 3,200 steps buys less than 2 SE over 800 at each schedule's own best rate.
The one consequential defect is downstream: the survival refit (§5, amendment 3) carries M-wide's selected rate
but fixes 800 steps, so when M-wide selects 1.0@200 (anchors 3003–3005) the h = 16 refit overtrains and the
survival ratio collapses (0.26, −0.02, 0.44 against 0.97–1.32 elsewhere). M-sq|T-cool is genuinely clipped at the
(min rate, min step) corner on 6/6; its bias is a few millinats on D3, below any threshold. Verdict: within the
preregistered design and reportable, except the refit-steps item, which needs John's decision.

## Findings

**F1. Major. Survival refit overtrains when M-wide selects the high ridge corner.**
Evidence: `modules-{3003,3004,3005}/result.json` arm `M-wide→16:refit`: `selection_loss` 0.5527 / 0.5826 / 0.5960
against `step0_selection_loss` 0.5284 / 0.5379 / 0.5553 (the refit ends worse than its sliced start); on
3000–3002 (M-wide at 0.1/0.3) it improves (0.5603 vs 0.5621, 0.5462 vs 0.5508, 0.5436 vs 0.5563).
`05-spreads.json` `quantities.survival.refit["16"]`: 1.07, 0.97, 1.32 on 3000–3002; 0.26, −0.02, 0.44 on 3003–3005.
Cause: `worker.py:135-137, 932-938` runs the refit at `wide.best.rate` for `SURVIVAL_REFIT_STEPS = 800`; M-wide's own
grid shows 1.0@800 is 0.03–0.04 worse than 1.0@200 on those anchors (3003: 0.5553 vs 0.5253). Binding text: §5
"one refit at the M-wide rate, 800 steps"; amendment 3 A3.1 restates it and promotes the survival widths to the
confirmatory set. Reasoning: on the ridge a rate is only meaningful with its step count; carrying one without the
other lands the refit off-ridge. h = 4 and h = 2 at 1.0@800 mostly still improve, so the published refit curve is
non-monotone in h on half the anchors, an instrument artefact rather than a property of the cut. The within-design
control is M-wide→8:rank:refit, which runs the full grid: on 3003/3004 its 1.0@800 candidate (0.5381, 0.5657) is
likewise worse than its step-0 start (0.5347, 0.5452), and the grid avoided it by choosing 1.0@200. Confidence:
high. Action: amendment to carry the selected (rate, step) pair, or run the refit on the module grid; cost nil
(each refit is 3.6 s). Not blocking: no verdict reads the survival widths, but the confirmatory would publish them
with this artefact on the anchors where M-wide picks 1.0@200 (3 of 6 here).

**F2. Minor. M-sq|T-cool is clipped at the (0.1, 200) corner on 6/6 anchors.**
Evidence: every candidate is worse than 0.1@200 on every anchor; at fixed steps loss rises with rate, at fixed rate
with steps (3000: 0.1 column 0.3688, 0.3755, 0.3944; 200-step row 0.4107, 0.3738, 0.3688). The best off-corner
candidate is 2.0–2.4 SE worse on four anchors. "Certified" 6/6 is vacuous: the late comparator is worse by 4–8 SE.
Reasoning: on the cooled host the module has nothing to add and only overfits the 2,000 fit examples; the
least-trained candidate wins and the grid cannot look below it. Bias: D3 (`quantities.D3`) is already negative on
all six anchors (−0.0003 to −0.006); a shorter or colder candidate could only pull it toward zero by a few
millinats, against X = 0.02. Confidence: high on the shape, medium on the bias magnitude (unobservable).
Action: report; no amendment.

**F3. Minor. `edge_of_grid` is nearly uninformative on a 3-rate grid and ignores the step axis.**
Evidence: `fit.at_edge` flags rate ∈ {max, min}, i.e. 2 of 3 rates; the flag was true on 45 of 54 module-grid
selections, interior only when 0.3 was chosen; 21 of 54 selections sit at the minimum checkpoint, which is never
flagged. Reasoning: on a ridge the informative classification is corner versus interior and whether the ridge is
flat; the flag distinguishes neither. Confidence: high. Action: report the corner classification (table under A)
beside the preregistered flag.

**F4. Observation. The plateau comparator is the selection clause alone; the training-tail clause has no power.**
Evidence: across all 72 grids, `change / (2 × training_tail_se)` maxes at 0.19 (M-small-B 3005); `training_tail_se`
is 0.003–0.006 while the two 40-step window means differ by 0.00005–0.002. Every uncertification is the selection
clause: late best beats early best by 2.4–4.7 SE (M-sq-fresh 3000 3.0, R-aff 3002 2.9, M-small 3004 2.7, M-wide
3004 2.5, M-aff 3004 2.4, R-272s 3004 4.7, R-aff 3004 3.0, M-sq 3005 2.9). `plateau_evidence` is an availability string on every grid and carries no
diagnostic content. Confidence: high. Action: none; this is §4 Table B as written. "Certified" speaks to the
schedule axis at each schedule's best rate, not to convergence of the selected run.

**F5. Observation. M-aff's certification bar is 2–7× tighter in nats than the module arms'.**
Evidence: `selection_paired_bootstrap_se` 0.00063–0.00076 for M-aff and 0.0003–0.0023 for R-272s/R-aff, against
0.001–0.0055 for the nonlinear module arms (S-static:refit 0.0006–0.0024); the 3004 M-aff miss is 0.0016 nats. Reasoning: M-aff re-tunes five
BatchNorm/bias vectors, so the two comparator hosts differ by very little per example and the 2-SE rule demands
near-exact equality. Confidence: high. Risk: the V1 "present" gate requires M-aff certified on ≥ 0.75 of 32; the
pilot's 1/6 failure rate has a 95% interval of about 0.004–0.64, so the pilot cannot bound that risk (see B).

**F6. Observation. Rate 1.0 is finite but harmful on M-aff on every anchor, and on M-wide at 3,200 steps.**
Evidence: M-aff 1.0 column selection loss 0.65–2.4 on all anchors (3005: 1.24, 1.04, 2.41), 0.04–1.8 nats above
N0 on dev (`quantities.harm`); never selected. M-wide 1.0@3200 0.66–0.75 on every anchor (harm 0.054–0.156) while
1.0@200 is on-ridge. Reasoning: the hot end of the ridge run too long, not divergence (no nonfinite candidate
anywhere; every `finite` true). The five-rate grid did what §4 said it was for: M-aff selects interior on 5/6
(0.03, 0.03, 0.03, 0.1, 0.1); on 3005 it selects 0.01@3200 with the 0.01 column still descending (0.5725, 0.5610,
0.5545) but the low ridge (0.1@200, 0.03@800, 0.01@3200) flat within 1.8 SE. Confidence: high.

**F7. Minor. R-272s sits at the top of the re-tune grid; the two lowest rates are never useful to it.**
Evidence: selections 0.3, 1.0, 1.0, 0.3, 0.3, 1.0 (top two of five on 6/6); the 1.0 column is still descending at
3,200 on 3002 (0.5482, 0.5421, 0.5406) and 3005 (0.5739, 0.5594, 0.5573), by 0.8 and 1.1 SE over 800; under-fit
≤ 0.002 nats on the evidence. Reasoning: R-272s enters D1′ ("beats every named re-tune"), so an under-fit favours
the module, but at this magnitude it cannot move a label read at 0. Confidence: medium. Action: report; a grid
shift would be an arm-definition amendment and is not warranted.

## Question A: the loss-versus-rate-versus-steps surface

The surface is a ridge along constant rate×steps (≈ 200–320 for module arms, ≈ 20–30 for M-aff, consistent with
§4's note on M-aff's unbounded features). Off-ridge candidates are undertrained (0.1@200, 0.1@800) or overtrained
(1.0@800, 1.0@3200, 0.3@3200) and lose by 3–50 SE. The two cold
ridge candidates (0.3@800, 0.1@3200) are within 3 SE on 49 of 54 module grids; the hot end (1.0@200) is never
better than them by more than 2 SE and is worse by more than 3 SE on 27 (M-sq-fresh 3002–3005 at 0.57–0.89,
S-static:refit 3000/3005). Dev losses (`harm_inventory`) agree: the selected candidate is within 0.003 nats of the
dev-best candidate on 43 of 54 and within 0.005 on 48 (3001 M-small ridge on dev: 0.5258, 0.5258, 0.5259). So the
edge selections are noise among near-equal candidates. The low corner is tested within the design by M-small-B:
0.1@6400 is worse than 0.1@1600 on 5/6 and better only on 3004 (by 0.0050 nats, 2.1 SE), the one slow anchor,
which also holds five of the eight uncertifications; extending steps at 0.1 does not help elsewhere. The high
corner is unknowable from these receipts (whether 1.0@100 would beat 1.0@200); on dev 1.0@200 is not better than
0.3@800 on 3003–3005, so no evidence of a missed optimum there.

Per arm × anchor, selection (rate@steps) and corner class: L = low corner (0.1 at the longest or middle
schedule), H = high corner (1.0@200 / 1.0@400), I = interior (0.3), C = monotone corner (clipped).

| Arm | 3000 | 3001 | 3002 | 3003 | 3004 | 3005 | edge / cert |
|---|---|---|---|---|---|---|---|
| M-small | 0.1@3200 L | 0.3@800 I | 0.1@3200 L | 0.1@3200 L | 0.1@3200 L* | 1.0@200 H | 5/6, 5/6 |
| M-small-B | 0.1@1600 L | 0.1@1600 L | 0.1@1600 L | 1.0@400 H | 1.0@400 H | 0.1@1600 L | 6/6, 6/6 |
| M-sq | 0.1@3200 L | 0.1@800 L | 0.1@3200 L | 0.3@200 I | 0.3@200 I | 0.1@3200 L* | 4/6, 5/6 |
| M-sq-fresh | 0.1@3200 L* | 0.3@200 I | 0.1@800 L | 0.3@200 I | 0.1@800 L | 0.3@800 I | 3/6, 5/6 |
| M-sq\|T-cool | 0.1@200 C | 0.1@200 C | 0.1@200 C | 0.1@200 C | 0.1@200 C | 0.1@200 C | 6/6, 6/6 |
| M-wide | 0.1@3200 L | 0.3@800 I | 0.1@800 L | 1.0@200 H | 1.0@200 H* | 1.0@200 H | 5/6, 5/6 |
| M-wide→8:rank:refit | 0.1@800 L | 0.1@200 L | 0.1@800 L | 1.0@200 H | 1.0@200 H | 0.1@800 L | 6/6, 6/6 |
| M-wide→8:random:refit | 0.1@800 L | 0.3@800 I | 0.1@800 L | 1.0@200 H | 1.0@200 H | 0.1@800 L | 5/6, 6/6 |
| S-static:refit | 0.1@3200 L | 0.3@800 I | 0.1@3200 L | 0.1@800 L | 0.1@3200 L | 0.1@3200 L | 5/6, 6/6 |
| M-aff (5 rates) | 0.03@800 I | 0.03@800 I | 0.03@800 I | 0.1@200 I | 0.1@200 I* | 0.01@3200 L | 1/6, 5/6 |
| R-272s (5 rates) | 0.3@3200 I | 1.0@800 H | 1.0@3200 H | 0.3@3200 I | 0.3@3200 I* | 1.0@3200 H | 3/6, 5/6 |
| R-aff (5 rates) | 0.1@800 I | 0.03@3200 I | 0.03@3200 I* | 1.0@200 H | 1.0@200 H* | 0.3@3200 I | 2/6, 4/6 |

`*` uncertified. The stated facts verify (M-wide, M-aff 3004, M-sq-fresh 3000, `plateau_fractions` 0.67–1.0).

## Question B: is the comparator discriminating as designed?

Yes, in the sense §4 defines. The early comparator is the best 800-step candidate and the late one is almost always
0.1@3200; the rule asks whether 4× steps at a re-chosen rate buys more than 2 paired SE. It passes trivially
whenever the late comparator is worse (42 of 72 grids, every M-sq|T-cool and most refits), which is why arms
selecting at 200 steps certify. The tail clause never binds (F4).

Uncertified cases, from the grids: M-aff 3004 is slow, real improvement at the low rates (0.01 column 0.5425,
0.5332, 0.5266) exceeding a 0.0013-nat bar; its selected 0.1@200 beats both comparators, so the arm is not
under-fitted. M-sq-fresh 3000 is the 0.1 column still descending (0.5616, 0.5465, 0.5390, 3.0 SE) with a flat
tail: a genuinely still-improving cold run. M-small 3004, M-wide 3004 and M-sq 3005 are the same pattern at
2.5–2.9 SE; R-272s 3004 (4.7 SE) and R-aff 3002/3004 are tight-SE re-tune arms with 0.0008–0.0034 nat
improvements. None is noisy or diverging: all late tails are flat (change ≤ 0.0021 nats). Eight failures of 72 at
a one-sided 2-SE bar exceeds the ≈ 2 expected under a flat null, so the long schedule buys a little on the cold
rate; M-small-B says doubling would not recover it (its late comparator is worse than its early one on 6/6).

Threshold sensitivity. With six anchors the raise rule fires iff ≥ 2 M-aff grids fail (5/6 = 0.833 passes,
4/6 = 0.667 raises), so it is insensitive to one anchor and sensitive to the second. At the observed per-anchor
failure rate 1/6, P(raise) ≈ 0.26; at the pooled 0.11, 0.13. The confirmatory does not re-run the raise rule
(`confirmatory.py:244-246` takes the pilot axis through `raise_only`); there `CERTIFIED_FRACTION` is the V1 gate,
needing ≤ 8 failures of 32: P(gate fails) ≈ 0.07 at p = 1/6, 0.006 at 0.11, 0.41 at 0.25. The pilot cannot say
which; that is the risk to carry forward, not the schedule.

## Question C: harm inventory

No divergence; every candidate on every grid is finite. M-aff at 1.0 is 0.04–1.8 nats above N0 on dev at every
checkpoint and anchor (F6) and never competes. M-wide at 1.0@3200 is 0.05–0.16 above N0 on every anchor: the hot
rate overtrains 2,000 examples over 400 passes, while 1.0@200 is on-ridge and matched on dev by 0.3@800 (3003:
0.5143 vs 0.5139). The other 1.0 harm entries are the same pattern or an early transient (M-sq-fresh 1.0@200 at
0.89 on 3002) the run recovers from. Edge selections at 1.0 are not suspicious as fits; their one consequence is F1. The 0.1 floor is still descending at 3,200 on most module
grids, by 0.001–0.013 nats over 800 (1.0–3.0 SE); it is the ridge's cold end, and M-small-B's 6,400-step point
shows it turns over by 2× steps on every anchor but 3004. M-aff 3005's 0.01 floor is the analogue (0.0065 nats
over 800, ridge-flat).

## Question D: recommendation and cost

Within the preregistered design and to be reported: F2–F7. `MODULE_RATES`, `RETUNE_RATES` and `CHECKPOINTS` sit in
`identity.py` under `PROTOCOL_SHA` and §9 forbids the pilot changing arm definitions or the 0.75 rule, so any
grid, comparator or rule change is an amendment. None is justified by the evidence: the cold end turns over by 2×
steps (M-small-B), the hot end is matched on dev by the interior, M-aff's grid is interior on 5/6, and a schedule
raise would have made certification vacuous rather than fits better. The one amendment worth John's word is F1
(carry the selected step with the rate for the survival refits; cost nil, no verdict role).

Cost (`01-replay-timing.json`: `module_grid_seconds` 41.3–42.1 s for 9,600 steps, 4.3–4.4 ms/step;
`front_rerun_seconds_per_1000_steps` 7.5–7.6): one added rate costs 3,200 steps ≈ 13.8 s per three-checkpoint
grid, 27.6 s for M-small-B, 13.8 s for M-aff and R-272s (`booking.wall_seconds` 69.1 and 79.6 for five rates),
24 s for R-aff. The confirmatory set runs six module grids, M-small-B and three five-rate grids per anchor, so one
rate added everywhere is about 160 s per anchor, 5,100 s at n = 32 on the projected 24,261 s against 43,200 s
(`06-confirmatory-count.json`); on the module grids only, about 110 s per anchor, 3,500 s. Doubling the schedule
axis doubles those ten grids' bookings, about 615 s per anchor, 19,700 s at n = 32, landing near 43,900 s: over
the ceiling by about 700 s.

## What I did not check

- Whether a candidate between grid points (0.5@400) or below the step floor (0.1@50) improves M-sq|T-cool or the
  high-corner selections; not inferable from the receipts.
- The paired bootstrap itself (`fit.paired_se`); `selection_paired_bootstrap_se` taken as published.
- `05-extended.json` arms: out of scope. CIFAR batch files and `vectors.pt`: not opened.
