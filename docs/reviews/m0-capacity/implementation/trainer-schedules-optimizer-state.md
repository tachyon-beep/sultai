# Exact-source review: trainer-schedules-optimizer-state (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Tests run read-only: `PYTHONPATH=src python -m pytest tests/test_m0_host.py -p no:cacheprovider` gives 12 passed.
Note: the brief cites TCOOL/COOLED_20 literals as prereg §5 text; they are not in the frozen draft. They are in the implementation plan (Task 4, marked [plan], gate A decision (a)). The draft's §3 (lines 170-173) gives the 20-epoch schedule in words. Source is checked against both.

## Claims and evidence
| claim | source file:line | pinning test | verdict |
|---|---|---|---|
| "saving optimiser state" (§2 l.95) | host.py `momentum_buffers` (name -> momentum_buffer, raises if any param unstepped) | test_resume_from_saved_momentum_continues_the_stream_bitwise (round-trips through torch.save/load, key order == REFERENCE_NAMES, unstepped optimiser raises) | satisfied |
| "a resume path" | host.py `optimizer_state` (t3 boundary: exact keys, finite, shape and dtype), `restore_optimizer`, `train_epochs` (`step = epochs.start * ceil(n/32)`) | same test: head + resumed tail is bitwise equal to the whole run (hashes and stream); zeroed buffer changes the hash; dropped key rejected by both functions. Boundary test covers non-dict, missing, extra, shape, dtype, NaN, inf, non-tensor | satisfied |
| "with a decayed-rate schedule (T-cool)"; §5 "rate decayed linearly to 0", 5 epochs | host.py `linear_cooldown`, `TCOOL = linear_cooldown(12_500, 18_750)` | test_linear_cooldown_values, test_linear_cooldown_is_zero_at_and_after_stop | satisfied |
| "an epoch count other than 10 with a named decay for the extended 20-epoch point"; §3 "unchanged recipe for 15 epochs, then decayed linearly to zero over epochs 16-20" | host.py `COOLED_20 = linear_cooldown(18_750, 25_000)`; `train_epochs(range(20))` accepts the count | test_linear_cooldown_values, test_linear_cooldown_is_zero_at_and_after_stop. Nothing runs 20 epochs through `train_epochs` in tests | satisfied; see Minor 1 |
| `train_host` bitwise unchanged, `epochs != 10` still refused | host.py `train_host` delegates with `constant(0.05)` and a fresh `recipe_optimizer` | test_train_host_golden_pin_is_unchanged (17325d1d.../56f5bff8...), test_train_host_is_train_epochs_with_constant_rate (also epochs=9 raises, weight_decay groups [5e-4, 0.0]) | satisfied |
| decay per optimiser step [plan (a)] | host.py `rates(step)` set on every group before each `optimizer.step()`; global step counter | test_resume (cooldown starts mid-epoch 0 with 2 steps per epoch, and the resumed run must match bitwise) | satisfied |
| published `final_state_sha256` of frontier hosts | Not this item (worker/receipts). `train_epochs` returns per-epoch hashes of `hashed` | n/a | out of scope |

## Frozen values checked
| value | prereg | source | match |
|---|---|---|---|
| T-cool start/stop step | plan: 12,500 / 18,750 (epochs 10-14, 1,250 steps per epoch) | host.py TCOOL(12_500, 18_750) | yes |
| cooled-20 start/stop | plan: 18,750 / 25,000 (epochs 15-19 decayed) | host.py COOLED_20(18_750, 25_000) | yes (§3 words agree) |
| base rate | 0.05 recipe | `linear_cooldown(rate=0.05)`, `constant(0.05)` | yes |
| last executed update / boundary | plan: 0.05/6,250 at step 18,749, 0.0 at the boundary | schedule formula | yes (asserted) |
| momentum 0.9, nesterov, wd 5e-4 on ndim>1, 0 on affine | unchanged recipe | `recipe_optimizer` | yes (golden pin) |
| batch 32, steps/epoch | 1,250 at 40,000 | `(len+31)//32` | yes |
| golden hashes | 17325d1d.../56f5bff8... | test constants full 64 hex | yes (test passes) |
| epochs for `train_host` | 10 | `range(10)`, guard `!= 10` | yes |

## Findings
- BLOCKING: none. No frozen value needs changing.
- Important: none.
- Minor 1 (tests/test_m0_host.py, test_linear_cooldown_values): COOLED_20 is pinned at 18_749 (=0.05), 24_999 and 25_000 (=0), but not at its start or midpoint. A mutation of its start to e.g. 19_000 or its stop kept consistent with a different shape would pass. Fix: add `COOLED_20(18_750) == 0.05` and `COOLED_20(21_875)` approx 0.025.
- Minor 2 (host.py `train_epochs`): the step offset assumes a resume begins at `epochs.start` with optimiser state from the preceding epoch. Nothing checks that the restored buffers belong to the loaded host. A mismatched pair would resume silently. The preregistration does not ask for a check (provenance sits with the worker), so informational only.
- Minor 3 (extras, all plan-sanctioned and none changing a Gate 3 byte): `Host.widths`/`widths=` constructor, `parameter_count`, `Host.sealed` flag, `load_host(frozen=)`. `sealed` is a plain attribute, not in `state_dict` (asserted), so `state_hash` and receipts are unchanged. RNG order is unchanged (per-epoch labelled generators; golden pin confirms). `train_epochs` now takes its gradient set from optimiser groups (decay group first), which only reorders independent per-parameter updates; the golden pin confirms bitwise identity.

## Dispositions (controller, 2026-10-14)

- Minor 1–3: deferred, recorded. The cooled-20 start/midpoint pin and the optimiser-state/host provenance check are noted; provenance sits with the worker (receipt-bound optimizer.pt, R10/F2).

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
