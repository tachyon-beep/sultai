# Proposed fixed-size evaluation repair — decision required

Status: recommendation only; no source change or execution authorized.
The [approved diagnostic result](../certificates/2026-10-12-gate3-twin-diagnostic-result.md)
localizes the observed original twin mismatch to examples changing between
128-example and 8-example classification batches. Both identical replay and
fixed-batch-shape controls pass. Do not replace the bitwise assertion with a
tolerance or erase either failed attempt.

Proposed source/protocol package:

1. Give frozen inference loss evaluation a canonical 128-example batch shape.
   Pad read/base/labels together before adapter application in the final partial
   batch, using an existing example, and discard padding logits/losses before
   per-example alignment, averaging or bootstrap. Apply
   the same evaluator to baseline, twin and every comparable arm/candidate
   loss path; keep real example counts/data splits unchanged. No training,
   gradients, optimizer, checkpoint, impairment, seed or acceptance margin changes.
2. Preserve the full global-permutation bitwise twin. Do not silently narrow it
   to within-batch-size permutation: that diagnostic passed, but it changes the
   registered assertion. Do not use padding only in the assertion while other
   losses retain a different numerical path.
3. Independently inspect the exact scientific/protocol disposition before
   changing frozen definitions. Keep the original protocol and source snapshot
   preserved; record a separately approved extension naming numerical-path
   changes and their scope. Floating-point changes may affect close candidate
   rankings, so old runtime outcomes cannot be relabeled as new-source evidence.
4. Add meaningful controls at full/partial batch boundaries for both roots,
   per-example alignment and counts, retained nonfinite checks, frozen BN/state,
   zero adapters, nonzero adapter evaluation and baseline/twin parity. Run strict gates and affected runtime
   tests, then independent exact-source review and a bound certificate.
5. Any new GPU pilot is a separate explicit decision after the repair package.
   Bind all three prior custody receipts and cumulative **1473.267264504917**
   process-seconds; maintain the same **10800-second** aggregate ceiling, with
   at most **9326.732735495083 seconds** remaining. Establish new exclusive
   linked custody without deletion/reset and preserve every historical artifact.
   Re-run required admission measurements for changed source; do not silently
   resume the old failed pilot or reuse old timing as new-source acceptance.

A source/protocol-repair approval would not alone clear a pilot, confirmation,
merge, environment change or schedule. Frozen §9/D-P still governs new deviations.
