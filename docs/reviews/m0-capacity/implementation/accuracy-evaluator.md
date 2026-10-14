# Exact-source review: accuracy-evaluator (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

## Claims and evidence
| claim | source file:line | pinning test | verdict |
|---|---|---|---|
| "dev-accuracy evaluator ... for the ΔAcc secondary" (§2 l.108-110; §6 ΔAcc) exists | src/sultai_screen/fit.py:133 `accuracy`, :138 `mean_accuracy` | FitTests.test_accuracy_matches_unpadded_forward (tests/test_m0_fit.py:125) | met |
| padded fixed-shape evaluation (EVAL_CHUNK rows, last-row padding, trimmed) | fit.py:114-130 `logits` (same `_padded` loop as `losses`, `chunk[: stop-start]` trim) | same test only indirectly: it compares to an UNPADDED full-batch forward, so removing `_padded` from `logits` would almost surely still pass (argmax rarely differs). Not pinned. | met in source; weakly pinned (F1) |
| argmax correctness aligned with `losses` (caller order, same data, same frozen-host assert) | fit.py:134 `argmax(1) == data.labels`; `logits` repeats the contract check, state_hash/assert_frozen, order | test checks shape (129,), exact equality to unpadded argmax, for adapter None and an adapter; wrong dim/label misalignment would fail | met; pinned |
| `losses` itself untouched, golden 343a6bf7… | fit.py:91-111 identical to b79f145 (diff shows no hunk in `losses`; only call sites elsewhere changed) | test_losses_bytes_unchanged (:140) | met |

## Frozen values checked
| value | prereg | source | match |
|---|---|---|---|
| losses golden SHA | 343a6bf7… | tests/test_m0_fit.py:40 `343a6bf756878ae0...6067` | match (prefix) |
| EVAL_CHUNK | 128 (v4.1 §3, unchanged) | fit.py:51 = 128, unchanged | match |
| padding rule | last real row repeated | `_padded` unchanged | match |
No frozen value needs changing; nothing amendment-territory.

## Findings
- Important? No. BLOCKING: none.
- F1 (Minor) tests/test_m0_fit.py:125-138: the test does not pin padding. It compares against an unpadded full-batch forward on a 129-row fixture, so a `logits` that skipped `_padded` (batch of 1 in the last chunk) would still pass unless the host's logits differ enough to flip an argmax. Fix: also assert `torch.equal(logits(...), <losses-style padded reference>)` or compare to a manual EVAL_CHUNK-padded forward, bitwise.
- F2 (Minor) fit.py:114-130: `logits`/`accuracy` have no finite check (losses has `check_finite`); NaN logits yield argmax 0 and a silent 0/1 verdict. Prereg does not ask for one, and ΔAcc conditions no verdict (J0); noting only. Not tested either way.
- F3 (Minor) tests: no test exercises the empty/misaligned `ContractViolation` or the frozen-host assertion for `logits`/`accuracy`; same checks as `losses` are copied, not shared. Duplication risk only.
- Extra behaviour: `logits` is a public extra function beyond "accuracy"; no extra randomness, no RNG order change, no Gate 3 receipt byte change (losses bytes pinned).

## Dispositions (controller, 2026-10-14)

- F1–F3 (Minor): deferred, recorded. ΔAcc is a labelled secondary that conditions no verdict (§6); the padded-path pin and the finite check are noted for a later test pass.

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
