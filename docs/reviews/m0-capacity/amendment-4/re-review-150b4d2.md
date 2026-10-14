# Scoped re-review: amendment 4 round 1 (commit 150b4d2)

**Verdict:** FINDINGS REMAIN (one minor: probe P6 survives; no blocking or important issue)

## Items checked

| Item | Result |
|---|---|
| I-1 pilot `stage_hosts` publishes per-anchor `static_noop` {noop_choice, raw_static_sha256, recalibrated_static_sha256} and `tcool_noop` {noop_choice, noop_sha256}; test asserts the mapping with seed-dependent fixture (seed%3==0 -> R-bn) | Fixed. Test passes (about 4 s, not 2 min). |
| I-2 `test_recalibrate_static_changes_only_batchnorm_buffers_with_the_module_in_the_forward` now uses a 200-index tensor (chunk 128 + partial 72), compares front buffers bitwise to `recalibrate_bn`, and requires a downstream difference | Fixed for P2-P5; order-insensitivity (P6) not caught. |
| M-1 `static_no_op` returns `StaticChoice` with both selection losses; test compares them to an independent computation | Fixed; P7 caught. |
| M-2 `WorkerTests.test_static_action_publishes_the_raw_model_when_the_rule_keeps_it` | Passes. It patches `worker.static_no_op` to return the raw model, so the raw branch runs. It asserts `static_sha256 == raw_static_sha256`, that `load_static` of the output files with that hash reloads, and that `static_hash(published) == raw_static_sha256`. It also asserts `checkpoint_sha256` differs from the R-bn run's. A `static_action` that published recalibrated files while recording the raw hash would fail the reload with the recorded hash and the `static_hash(published)` equality. |
| M-3/M-4 (confirmatory `static_noop_choice` record; worker publishes `raw_selection_loss`/`rbn_selection_loss`, checked in the R-bn test against the choice) | Present in diff, consistent. |
| Scope: diff touches only 4 m0 source files (confirmatory, pilot, static, worker) and 3 m0 tests | Confirmed. No docs, `calibrate.py`, `verdicts.py` or Gate 3 path touched. |

## Probes

| Probe | Failing test or assertion | Result |
|---|---|---|
| P1 pilot publishes `"noop_choice": "raw"` for all anchors | `PilotRunTests.test_clean_pilot_publishes_every_receipt_in_stage_order_under_the_m0_ceiling`, the `assertEqual(hosts["static_noop"][str(seed)], {...})` (got 'raw', expected 'R-bn' for seed 3000) | Caught |
| P2 no `module.momentum = None` | `StaticTests.test_recalibrate_static_changes_only_batchnorm_buffers_with_the_module_in_the_forward`, `front.1.running_mean` bitwise-equality assertion | Caught |
| P3 no `reset_running_stats()` | same test, same assertion | Caught |
| P4 `static.eval()` | same test, same assertion | Caught |
| P5 chunk 7 (range stride only; slice only; both) | same test, same assertion (all three variants) | Caught |
| P6 `indices = indices.flip(0)` before the pass | none; all 10 `StaticTests` pass. A debug print confirmed the flip ran, so the test is order-insensitive (likely the fixture's front statistics do not change under this reorder of LONG) | NOT caught |
| P7 `static_no_op` computes the recalibrated loss on RAW features | `StaticTests.test_static_no_op_picks_the_lower_selection_loss_and_ties_keep_raw`, `assertEqual(choice.recalibrated_selection_loss, selection(recalibrated))` (6.67 != 1.74) | Caught |

All sources restored byte-exactly (cmp against scratch copies); `git diff --stat` empty.

## Hashes

| Document | sha256 recomputed | Matches certificate | Matches identity.py |
|---|---|---|---|
| capacity-frontier-screen.md | c66599459adc898f...03fd5 | yes | yes |
| STEERING.md | 08a9e1aaf79d3048...3ffd | yes | yes |
| gate-ladder-amendment.md | af12d474a97aad83...6b89 | yes | yes |
| gate-ladder-amendment-2.md | b4740b147a4a1ced...2d77 | yes | yes |
| m0-amendment-3.md | 6dcfd9f2a781dc56...e862 | yes | yes (also in test_m0_identity.py) |
| m0-amendment-4.md | d8dd571bb39718946...dcdcc34 | yes | yes (also in test_m0_identity.py) |
| composition `m0-v4.2` | a2c1e55deffb144c...6007 | yes | not a literal in identity.py (computed there); pinned in tests/test_m0_identity.py |

The first four documents are not literal in test_m0_identity.py, which re-hashes the files. The six document hashes and the composition match the certificate, and the documents are unchanged this round.

## Remaining findings

- m-1 (minor, test strength): probe P6 (reversing the fit index order in `recalibrate_static`) is not caught by any `StaticTests` test. The I-2 test would need an index tensor whose front-BN statistics change when reordered, or an assertion that the pass uses the given order. Wrong order would diverge from `recalibrate_bn`, which the amendment says is "the same pass", so it is worth a one-line guard, but it is not a pilot-blocking defect.

## Dispositions (controller, commit c49a4de)

| Finding | Disposition |
|---|---|
| m-1 (minor) | Adopted (c49a4de, test only): the fixture is a seeded shuffle of 200 indices and the test first shows the plain rule differs on the reversed order; the reversed-order probe now fails the test (verified by the controller). |

## Dispositions (controller, commit c49a4de)

| Finding | Disposition |
|---|---|
| m-1 (minor) | Adopted (c49a4de, test only): the fixture is a seeded shuffle of 200 indices and the test first shows the plain rule differs on the reversed order; the reversed-order probe now fails the test (verified by the controller). |
