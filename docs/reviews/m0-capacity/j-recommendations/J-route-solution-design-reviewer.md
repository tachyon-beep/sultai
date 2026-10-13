# J-route: which control defines D1 and the V1 "present" verdict
Agent: axiom-solution-architect:solution-design-reviewer   Date: 2026-10-13

## Options considered
- (a) **M-aff**, the foldable affine 272 module: the same `Adapter` form as M-sq with `nonlinear=False`.
- (b) **R-272s**. As drafted it does not exist: `tail[0]` is `Conv2d(16, 64, 3)`, so centre taps number 1,024 and biases 64, not "16×16 and 16"; the draft silently picks 16 of 64 output channels.
- (c) R-aff dropped as primary: 170 parameters, not count-matched, re-runs the front at an unmeasured 3× kernel, does not isolate the nonlinearity.
- (d) Per-anchor max dropped: a max over dev gains is dev-side selection on the control only; it biases D1 downward and makes "re-tuning equivalent" cheap (disposition A2 holds).
- (e) An exact-fold variant (re-tune `tail[0]` directly as K' = K(I+W), c' = c + ΣK b) exists; dropped because it is a new fit path bought for 16 border degrees of freedom measurable at 0.04 s.

## Recommendation
Keep **M-aff as the D1 comparator**, with three corrections pre-registered before freeze: (1) a measured **fold residual** per anchor, L(host with `tail[0]` replaced by the folded weights) − L(host + fitted M-aff) on dev, reported beside D1 with a tolerance tied to X [John; proposed X/4 = 0.005 nats]; (2) M-aff fitted on the **five-rate re-tune grid** {1, 0.3, 0.1, 0.03, 0.01}, selection-chosen, so a hot grid cannot under-fit the control; (3) **R-272s redefined** to a subset that exists (declared: centre taps of output channels 0–15 of `tail[0]`, 256, plus those channels' 16 biases) so D1′ is well-formed for the confirmatory set.

## Strongest case for
Checked from source. M-sq and M-aff are the same two-tensor `Adapter` (W 16×16 + b 16 = 272) differing in one flag, so the count match is by construction and D1 is a single-factor contrast: tanh versus identity at the same site, init, seed derivation and 3,000-example selection. The fold is exact at inference for the whole W part (K' = K(I+W), measured 1e-6) and for b in the interior; nothing sits between site S1 and `tail[0]`, and frozen execution keeps BatchNorm in eval mode, so the fold is a counting fact for every evaluation the study makes. Folded, M-aff is a 272-dof right-multiplication of the host's own first conv shared across all 64 outputs and 9 taps: an equal-count re-tune of host parameters in a module-aligned basis, which meets STEERING rule 2 in substance. The straw-man claim stands independently: 272 random scalars of a 64×16×3×3 kernel cannot express a channel mixing, so D1 against random R-272 reads "present" on any undertrained host. R-272s is the same conv in another basis; the pair tests basis-dependence rather than duplicating.

## Strongest case against
"Folds to 0" is overstated in §1 and the §5 table. On the one-pixel border ring (60 of 256 positions, 23.4 percent) the folded conv over-adds ΣK·b for the padded taps, so M-aff retains a 16-dof border-dependent bias the host cannot express. My CPU run (random weights, max border difference 0.05) demonstrates the mechanism, not the trained magnitude, which depends on the fitted b; hence correction (1). Matched procedure is not matched adequacy: M-aff's features are z (post-ReLU, unbounded) where M-sq's are tanh(z), so gradient scale differs and the module grid {1, 0.3, 0.1} may be off-scale for M-aff. The §7 rule that M-aff must be plateau-certified for "present" sends a badly fitted control to indeterminate, not to a false present, so the guard holds; correction (2) removes the failure mode at about 40 s per anchor. M-aff's SGD path is new code: `fit_sgd` hard-defaults `nonlinear=True`; only the closed-form `ridge(nonlinear=False)` exists.

## Dependencies on other decisions
J1: the fold-residual tolerance scales with X. J6: correction (2) adds about 6 × 40 s to the core ledger. J7: R-272s must be redefined before it enters the confirmatory set. Reversibility: §9 forbids changing arm definitions during the pilot and §10 makes any post-freeze change a new version with a fresh seed block; pilot step 7 opens dev medians on anchors 3000–3005, so a post-pilot swap is data-informed on development anchors (their purpose) and must be recorded as v3, while confirmatory seeds 7500+ stay unread, so the confirmatory verdict is not unblinded. The version-bump path is the mechanism; no fallback rule needed.

## Evidence
- `src/sultai_screen/host.py:61-70` (front ends Conv→BN→ReLU; `tail[0]` = Conv2d(16, 64, 3, pad 1) reads the site directly), `:62` Identity bottleneck at width 0, `:98-110` BN guard.
- `src/sultai_screen/fit.py:21-29` (two-tensor Adapter, `nonlinear` flag), `:43-44` 16×16 + 16, `:177-180` inits, `:180` no affine SGD path, `:96-111` affine ridge only. For the caller: "33h + 16 with h = 8" is 280, the M-small / M-wide→8 count, not M-sq's 272.
- Measured on CPU: conv weight (64, 16, 3, 3), bias (64,); centre taps 1,024; M-aff count 272; fold max |diff| interior 1.3e-6, border 0.050; W-only fold 1.1e-6 everywhere.
- `docs/plans/2026-10-13-capacity-frontier-screen.md:33-36`, `:118`, `:160-165`, `:196`, `:224-227`, `:267-277`, `:375-377`, `:386-388`.
- `docs/reviews/m0-capacity/scientific-review-v1-2026-10-13.md:31`; dispositions `:15`, `:33`; `docs/DECISIONS.md:47` (D21); `docs/STEERING.md` §4 rule 2.

## Versus the recorded §12 default
Match on the comparator; differ on its contract. The default asserts the fold; this measures it, widens the control's grid so a hot grid cannot under-fit it, and repairs an R-272s definition that cannot be instantiated on the host as built.

## What would change this recommendation
A pilot fold residual above tolerance on any anchor shows M-aff carrying border capacity; the exact-fold variant (e) should then replace it in a v3. A redefined R-272s that M-aff cannot beat on development anchors argues for the pair as co-primary rather than D1′.

## Confidence
High on mechanism and counts (checked numerically against source); medium on the border effect's trained magnitude, which only the pilot can measure.
