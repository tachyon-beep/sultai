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
- Shuffling paired targets changes fitting evidence; changing test targets
  cannot change learned weights or selected candidate.

These are software and instrument checks, not research evidence for diffusion,
real host repairability, or the final parameter-efficiency target.

## Controlled experiment after a budget is approved

Pre-register host/task, independent lineage/bottleneck partitions, sample
counts, capability/error margin, failure threshold, horizon, units, seeds,
query budgets, and K before collecting confirmatory results. Compare every
method in DESIGN.md under all three telemetry conditions and shuffle controls.
Use paired uncertainty estimates grouped by independent lineage, not by
individual example or sibling branch. Retain failures and no-op wins.

If a teacher or optimizer fails to repair, inspect optimization capacity,
data coverage and horizon; failure is not proof of unrepairability. If all
methods tie no-op, report an uninformative or null probe rather than a claimed
benefit. Evaluate oracle telemetry separately from implementable telemetry.

## Physical removal and efficiency gate, later

Measure allocated trainable parameters before addition, at peak growth and
after actual removal, including retained optimizer state and inference cost.
Use a predeclared task-capability tolerance and final retained parameter ratio
near 0.5 relative to a properly tuned reference. Report search/training cost
separately. Compare growth-only, pruning-only where applicable, joint
development, and the final architecture trained from scratch. Temporary
structure must be gone at the final measurement; masks are insufficient.
