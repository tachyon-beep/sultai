# Bounded no-op twin diagnostic authorization — 2026-10-12

John replied **APPROVED** in this thread to the concrete
[120-second diagnostic proposal](../plans/2026-10-11-gate3-twin-diagnostic-proposal.md).
This is an authorization transcript, not a cryptographic operator signature.
The proposal and both pilot failure records remain preserved unchanged.

Root remains sole writer. Authorization covers implementing, CPU-validating,
independently reviewing and running exactly one diagnostic GPU worker on idle
physical card 0. A CPU-only parent supervises its process group, reserves five
seconds for cleanup inside a hard 120-process-second diagnostic allowance,
and records actual creation-through-reap charge (including any overrun as a
failure). A fixed exclusive `.weft/gate3-twin-diagnostic-attempt.json` prevents
a repeat through a new output path. Both existing pilot custody receipts stay
untouched. No training, optimizer, repair fit, new host, environment change or
CIFAR test-batch access is permitted.

The diagnostic verifies the exact existing `intact-3000` checkpoint, original
source-file hashes and both pilot receipt chains. It decodes the pinned CIFAR
training batches to extract the sealed 5,000-example development partition;
only that partition is forwarded. Conditioning/selection/training partitions
are not evaluated. Frozen state, seed, flags and two CPU threads are preserved.
Compare original/repeated/permuted evaluation, preservation of per-example batch
size, and diagnostic padding. Record exact loss/logit differences and hashes,
batch sizes, permutation round-trip and frozen-state hashes. All numerical
mismatches are reported, without tolerances or a changed production assertion.

Opening cumulative charge is **1469.3526015179232 process-seconds**, including
both prior attempts, under the same **10800-second total ceiling**. Add the exact
diagnostic charge; never reset custody or budget. Stop/reap only the diagnostic
worker on timeout/interruption. Preserve incomplete/failure evidence rather
than manufacturing successful completion. Publish raw/GNU/outer exits and
post-stop idle proof. No automatic retry follows any diagnostic outcome.

This permission does not authorize changing the pilot implementation/protocol,
a third pilot, confirmation, merge, schedule or unrelated workload.
