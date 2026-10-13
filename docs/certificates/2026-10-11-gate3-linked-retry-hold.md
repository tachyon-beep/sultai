# Gate 3 linked retry execution hold — 2026-10-11

**HOLD. Execution invalid. No further retry or confirmation is authorized.**

The explicitly authorized linked retry used accepted repair commit
`bbf9811854f23e40b2700dab9e457852c09ba7da`, aggregate
`a9445331c6b92018fbe73316fe4b7eb3851fc157a4798185da51ce2fff17b595`,
and bound admission commit `9b97fb1ef4d170936dbd2757231428d03e70a107`.
All 20 strict quality checks and 140 CPU tests passed, and both independent
specialists accepted the exact provenance/custody repair. The frozen numerical
protocol and original research/results were unchanged. Draft PR #3 remains
unmerged: https://github.com/tachyon-beep/sultai/pull/3.

## Measured execution

Fresh admission at 10:04:41 UTC / 21:04:41 AEDT verified the distinct pinned
package/runtime/CUDA fields, idle exact GPU UUIDs, no screen processes, released
Simic terminals, unchanged original failure receipts and absent retry custody.
Parent admission consumed one separate exclusive linked receipt. The earlier
failure receipt and its exact charge remained unchanged.

Twelve development hosts completed training. Three serial host replays and
three-process/card host replay on both physical cards passed bitwise state and
loss-stream checks. All nine serial/co-tenant fit timing receipts passed finite
and bitwise replay checks; the ordered timing admission decision was published.
No seed reduction occurred. These are execution measurements, not headroom or
repair-effect findings.

The first materiality worker then stopped at `fit.identity_check`, fit.py:71:
**resampled no-op evaluation twin is not bitwise**. The zero-adapter equality
check preceding it passed. The worker published structured failure evidence,
including source/environment, sealed data opens and the exact error. No
materiality rung or subsequent repair/sizing decision was reached. Scientific
headroom, specificity, developmental advantage and confirmation remain unknown.

Supervisor, GNU time and outer tool exits are **2**. GNU time reports
**662.93 seconds wall (11:02.93)** and **1,564,996 KiB peak RSS**. The cumulative
ledger contains 30 successful children and one failed child and charges
**1,469.3526015179232 process-seconds**, including the original
**2.370860553928651 seconds**. The same 10,800-second total cap leaves
**9,330.647398482077 seconds**; this unused budget grants no new attempt.

Post-stop host verification at 10:16:20 UTC / 21:16:20 AEDT found both GPUs
0% / 15 MiB, no compute process and no screen process. No unrelated workload
was stopped. Both original and linked retry custody remain intact.

## Preserved evidence and next decision

[Publication manifest](../results/gate3-pilot-2026-10-11-01a125af-retry1/manifest.json)
binds 85 byte-identical copied receipts: all runtime JSON records, custody,
preflight, post-stop and GNU time. The execution summary binds raw-log hashes.
Private model weights, loss streams and raw logs remain in ignored local run
storage and are excluded from Git. Prior failed-attempt artifacts remain
byte-preserved and linked, not replaced.

Static inspection identifies a diagnostic gap: `losses` evaluates chunks of
128, whereas `identity_check` permutes all 5,000 examples before chunking. This
changes batch membership and which examples occupy the last 8-example chunk.
Deterministic execution for an identical input does not establish bitwise
invariance to those changes. The passing CPU fixture used four examples in one
chunk. CUDA batch-shape/composition effects are therefore a **hypothesis**, not
a measured cause. The failure receipt contains no mismatch count or maximum
delta. No post-stop numerical experiment was run and no repair is applied.

The concrete next disposition is the [proposed one-process, 120-second diagnostic](../plans/2026-10-11-gate3-twin-diagnostic-proposal.md) using the
preserved host: capture baseline/twin per-example mismatches and hashes, separate
batch membership/order from final-chunk shape, and determine whether the frozen
bitwise requirement is attainable under the prescribed execution. This needs
John's explicit clearance under protocol §9 / D-P before any new GPU execution
or source repair. Keep the bitwise stop, scientific thresholds and all existing
charges/custody unchanged. Do not silently reinterpret the twin, introduce a
tolerance, reset budget or launch a third attempt. A diagnostic allowance would
not itself authorize another pilot or confirmation.


## Independent post-stop evidence assessment

The independent PyTorch/operations specialist returned **HOLD / execution-invalid**
for source `9b97fb1`, after read-only static/source/evidence inspection. No
imports, tests, jobs or edits were performed. Both twin evaluations passed
finite-value and frozen-host checks; zero-adapter bitwise equality passed.
The reviewer confirmed that the failure receipt lacks mismatch counts,
affected examples, deltas and intermediate tensors; it cannot establish a
small-rounding characterization. Changing final-chunk batch size is the leading
unproven hypothesis, compatible with successful identical-workload replay.
See [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
for batching distinctions. The proposed diagnostic keeps all acceptance
thresholds unchanged and needs John's separate disposition. No new attempt
was authorized or executed by this assessment.
