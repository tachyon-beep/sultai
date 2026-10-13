# Proposed bounded no-op twin diagnostic — awaiting John

No execution is authorized by this document. Frozen protocol §9 / D-P requires
hold on a new deviation; the one authorized linked pilot has stopped. The
[execution hold](../certificates/2026-10-11-gate3-linked-retry-hold.md) preserves
source, failure, completed admission checks and exact accumulated budget.

Proposed allowance: **one process, one currently idle pinned GPU (card 0),
maximum 120 process-seconds**, charged against the existing 10,800-second total
ceiling (current cumulative 1469.3526015179232 seconds). No training, fitting,
new hosts, test-batch access, environment change or pilot restart. Use the
preserved retry `intact-3000/host.pt`, verify its receipt hash/source identity,
and load only the already sealed development partition. Preserve both existing
custody receipts; publish a distinct exclusive diagnostic receipt, source hash,
raw/GNU/outer exits, exact elapsed charge and GPU idle proof. Timeout must stop
only the diagnostic's owned process group. No automatic repeat on failure.

Keep the existing environment, two CPU threads, CUDA flags and frozen host.
Compare identical-order repetition and permutation round-trip; the original
global permutation; a permutation preserving each example's original batch
size; and diagnostic fixed-size padded batches with padding outputs discarded.
Record mismatch indices/counts, original/twin batch sizes, loss and logit hashes,
maximum absolute differences and frozen-state hashes. Localize discrepancies
to logits versus cross-entropy. Deltas are diagnostics, never tolerances.
Retain every failure; do not change the production bitwise assertion.

The independent PyTorch/operations reviewer recommended this bounded allowance
after read-only source/evidence analysis. The saved failure has no mismatch
count or delta, so neither small rounding error nor batch-shape causation has
been established. Passing replay used identical workloads; permutation changed
which examples were in the last 8-example batch versus the 128-example batches.
The earlier CPU fixture had four examples in one chunk.

John's disposition would authorize only implementing/reviewing this diagnostic
and its one bounded execution. Any ensuing implementation/protocol change,
further pilot or confirmation requires a separate evidence-supported decision.
