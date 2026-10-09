# Test plan

## Short CPU verification in the initial scope

- Exactly 272 adapter weights; 16-dimensional inputs/outputs; reject malformed
  or non-finite values; deterministic result for a fixed seed/input.
- No-op/blend-zero matches host output; nonzero adapter has a measurable
  non-affine response; parameter serialization round trips without drift.
- Synthetic repair reduces held-out error in a planted, representable case
  after fitting conditioning examples only; no repair remains unchanged.
- Frozen host and target fixture unchanged; condition, selection, and test
  sample IDs disjoint; independent host lineage and bottleneck split guards.
- Candidate selection uses selection loss, with an adversarial fixture where
  test ranking differs; best-of-K accounting includes all candidates/queries.
- Changing selection labels cannot alter the pre-selection candidate bank;
  transitive lineage/family overlap must fail the outer split guard.
- Opposite frozen downstream maps share passive observations at zero, but one
  fixed positive probe exposes opposite local influence. This is a diagnostic
  fixture, not learned intervention discovery.
- Shuffling paired targets changes fitting evidence; changing test targets
  cannot change learned weights or selected candidate.

These are software and instrument checks, not research evidence for diffusion,
real host repairability, or the final parameter-efficiency target.

## Controlled experiment after a budget is approved

Pre-register host/task, independent lineage/bottleneck partitions, sample
counts, capability/error margin, failure threshold, horizon, units, seeds,
query budgets, and K before collecting confirmatory results. Compare every
applicable method/condition combinations in DESIGN.md and shuffle controls;
do not secretly give coarse methods paired targets or gradients. Add the
early matched static-capacity comparator described there, separately from the
same-snapshot conventional fit. Declare oracle-site/target privileges and
practical allocation costs explicitly. Any future Simic interface must verify
pinned host/provider/encoder/probe source identities and preserve pairing,
plan hashes and untouched selection/test boundaries before a campaign.
Use paired uncertainty estimates over independent groups, not individual
examples or sibling branches. If bottleneck families induce cross-lineage
dependence, group by independent lineage–family components or use a suitable
two-way grouped analysis. Retain failures and no-op wins.

If a teacher or optimizer fails to repair, inspect optimization capacity,
data coverage and horizon; failure is not proof of unrepairability. If all
methods tie no-op, report an uninformative or null probe rather than a claimed
benefit. Evaluate oracle telemetry separately from implementable telemetry.

## Physical removal and efficiency gate, later

Measure all physically retained model parameters before addition, at peak
growth and after actual removal. Report trainable and frozen subsets,
retained optimizer-state bytes and inference cost separately. Include any
builder/controller/retrieval assets still required at inference.
Use a predeclared task-capability tolerance and final retained parameter ratio
near 0.5 relative to a properly tuned reference. Report search/training cost
separately. Compare growth-only, pruning-only where applicable, joint
development, and the final architecture trained from scratch. Temporary
structure must be gone at the final measurement; masks are insufficient.
Compare developmental removals with growth-only and matched late-removal or
pruning schedules, measuring subsequent learning as well as final capability.
Shuffling telemetry and disabling a mechanism are experimental ablations;
they must not be counted as the model's developmental removal policy.
