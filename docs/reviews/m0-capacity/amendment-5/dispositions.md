# M0 amendment 5: review dispositions (2026-10-14)

Two independent exact-source reviews of `nyx/m0-amendment-5` (diff against
`main` 0d4179a), dispatched at 712b09a and extended to 4a2a005 at the
controller's request; the conditions they set were applied in 411314c and the
source-change disposition repointed in 13989d9, the certified
implementation commit. Both reviews are preserved verbatim beside this file.

| Review | Verdict at 4a2a005 | Counts (blocking / major / minor / observation) |
|---|---|---|
| statistics-and-protocol | accept | 0 / 0 / 2 / 8 |
| determinism-and-receipts | accept with conditions | 0 / 0 / 5 / 6 |

## Dispositions

| Finding | Disposition | Where |
|---|---|---|
| statistics F1 (projected half-width is a point projection from six values, no interval on itself) | accepted as a narrative caveat; the receipt labels it `projected_half_width_development_only` | no code change |
| statistics F2 (no direct unit test of the corner classification) | taken: `worker.corner` unit test on every axis position | 411314c, `tests/test_m0_worker.py` |
| determinism F1 (stop-path terminal fields not pinned) | taken: pinned on the pilot and confirmatory stop paths, including an attempt that stops before its opening record | 411314c, `tests/test_m0_pilot.py` |
| determinism F2 (exit records pinned on the clean path only) | taken: the failing child's exit record pinned with its non-zero return code | 411314c |
| determinism F3 (replay discards the fit-index hash) | recommendation taken: `fit_indices_sha256` joins the compared keys | 411314c, `confirmatory.py` |
| determinism F4 (cuDNN call unguarded; timeout and malformed rows untested) | taken: `RuntimeError` recorded as absent; timeout, malformed-row and empty-query cases pinned | 411314c, `provenance.py`, `tests/test_m0_provenance.py` |
| determinism F5 (static host's no-op application publishes no fit-index hash) | taken | 411314c, `worker.py` |
| determinism F6 (no pilot baseline for the four new environment fields) | accepted: the first confirmatory publishes them against the pilot's five recorded fields; `environment_vs_pilot` shows the four as absent on the pilot side | no code change |
| determinism F7 (the replay's charge is outside the carried fixed V0 term) | accepted: about 10 s against 43,200, charged in the confirmatory's own ledger | no code change |
| determinism F8 (every worker child spawns nvidia-smi) | accepted: one short query per child, absent on failure | no code change |

The remaining observations (statistics F3 to F10; determinism F9 to F11) record
clean recomputations and need no action.
