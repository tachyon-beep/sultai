# Bounded no-op twin diagnostic — measured result, 2026-10-12

**Diagnostic complete. Original twin assertion fails; pilot and confirmation remain blocked.**

John approved the [one-run diagnostic](2026-10-12-gate3-twin-diagnostic-authorization.md).
Accepted source `9e19e5e9fee3be709b54f810c6b3e4954271416b`, aggregate
`c0e99d856b54c599b854492ea3b5fd20a57c4c77aa477d5d23aafad6068c0f36`, passed
20 quality checks across 24 production modules and 146 CPU tests and independent
scientific/PyTorch-operations review. The [bound certificate](2026-10-12-gate3-twin-diagnostic-admission.json)
was committed/pushed at `340bff2efe6f4ce732dccd5f2bf0f7d017907434` before launch.
No existing production module, frozen protocol, original research or saved
failure artifact was modified.

## Execution and custody

Fresh admission at 08:28:46 UTC / 19:28:46 AEDT verified both exact cards idle,
no screen/diagnostic process, exact source/environment and all prior receipt,
checkpoint and binary loss-stream hashes. The CPU guardian launched one GPU0
worker. The worker loaded the saved intact-3000 host and forwarded only the
sealed 5,000-example development partition, with no training or repair fits.
All five pinned CIFAR source batches and metadata were opened once each;
no CIFAR test batch was opened. Flags/environment and frozen state were retained.

Raw child, supervisor, GNU time and outer tool exits are **0**. GNU time reports
**5.06 seconds wall**, **1,316,628 KiB peak RSS**. Actual conservative diagnostic
charge is **3.9146629869937897 process-seconds**. Opening charge
1469.3526015179232 plus this diagnostic is **1473.267264504917 seconds** under
the unchanged 10800-second total ceiling; remaining **9326.732735495083 seconds**
grants no new attempt. Timeout was false. Both original pilot custody receipts
and the new exclusive diagnostic custody remain intact. Post-stop idle/resource
proof is preserved in the [evidence manifest](../results/gate3-twin-diagnostic-2026-10-12-01a125af/manifest.json).
All 14 copied receipts were verified byte-identical to their local originals.
Raw logs remain local and hash-bound; model weights/loss streams remain in
unchanged ignored storage, excluded from Git.

## Measured comparisons

| Comparison | Logit mismatches | Loss mismatches | Maximum logit delta | Maximum loss delta |
|---|---:|---:|---:|---:|
| Identical-order repeat | 0 | 0 | 0 | 0 |
| Original global permutation versus original baseline | 16 | 6 | 1.9073486328125e-6 | 5.960464477539062e-7 |
| Permutation preserving each example's batch size | 0 | 0 | 0 | 0 |
| Padded baseline versus padded global permutation | 0 | 0 | 0 | 0 |

The original baseline has 39 batches of 128 and one batch of 8. The original
permutation moves eight examples out of the final batch and eight into it.
**Exactly those 16 examples**, and no others, have changed logits. The six
changed losses are a subset. Permutation round-trip of features/labels is exact;
all evaluated frozen-state hashes are unchanged. Computing CE from fixed
baseline logits in each case's order/batching remains bitwise exact. Computing
CE from the changed logits in identity order reproduces the six loss changes.
The discrepancy therefore arises upstream of CE, in host classification under
changed batch size, in this observed checkpoint/environment.

Padding the original baseline alone changes its eight final-batch logits
(maximum 1.1920928955078125e-6) and three losses (maximum
5.960464477539062e-7). Thus padding is a changed evaluation implementation,
not permission to relabel the old failed assertion as a pass. The padded
baseline/twin pair is bitwise identical. Neither padding nor preserving batch
size has been installed in the production evaluator or frozen protocol.

## Interpretation and remaining boundary

This single diagnostic localizes the observed discrepancy to changed evaluation
batch size. It establishes neither a universal cross-host/backend guarantee
nor scientific headroom, harmlessness or a successful pilot. Exact measurements
are diagnostics, never replacement acceptance tolerances. The original failure,
bitwise rule and all prior charges remain valid and preserved.

The recommended next decision is a separately authorized, independently reviewed
fixed-size evaluation repair, keeping baseline/twin/all comparable loss paths on
the same batch shape with padding outputs excluded. A concrete
[repair proposal](../plans/2026-10-12-gate3-fixed-evaluation-proposal.md) separates
that source/protocol decision from any further pilot allowance. No repair,
new pilot, diagnostic repeat or confirmation is authorized by this result.


## Independent saved-evidence interpretation

After a read-only saved-source/evidence review (no imports, tests, jobs or edits),
`/root/gate3_torch_ops_panel` agreed that the observed discrepancy is localized
to batch-size-sensitive host-to-logit computation, upstream of cross-entropy.
It does not identify a particular convolution, pooling or linear kernel and
covers one checkpoint, card, environment and permutation. The reviewer confirmed
the exact mismatch counts/indices/deltas, all unchanged round-trips/states and
cumulative charge. Readiness remains false. The recommended separately authorized
repair pads read/base/labels together before adapter application in the shared
frozen-loss path, discards padding outputs and retains the full global-permutation
exact assertion. Adapter evaluation still needs validation because the diagnostic
used no adapter. A future pilot requires a separate linked-custody disposition.
