# Exact-source review: static-host (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

## Claims and evidence
| claim | source file:line | pinning test | verdict |
|---|---|---|---|
| Module zero-initialised after the host (§5 "W = 0, b = 0"; §7 V0) | static.py `StaticHost.__init__` (torch.zeros after make_host in train_static) | test_construction_is_the_anchor_host_and_the_zero_module_is_identity (w,b all zero; host hash equals make_host) | met |
| No RNG draw (host init and data stream bitwise the anchor's) | static.py `StaticHost.__init__` uses torch.zeros only; train_static calls make_host(seed,0,...) then train_epochs with the anchor's generators | same test compares torch RNG state before/after; test_training_stream_shares_order... asserts stream[0] bitwise equals anchor's first loss | met |
| Step-0 host tensors bitwise the anchor's (§7 V0 S-static step 0) | train_static `initial = state_hash(host)` before StaticHost and any step; static_action emits initial_state_sha256 | construction test (hash equality); tests/test_m0_worker.py:350,393 (worker receipt) | met |
| Zero module is the identity at step 0 | StaticHost.forward z + W tanh z + b | construction test torch.equal(static(images), plain(images)) | met |
| w in the weight-decay group, b in the affine group | recipe_optimizer splits by ndim (w ndim 2 -> decay 5e-4; b ndim 1 -> 0) | test_module_joins_the_recipe_groups | met |
| Registered ten-epoch recipe (constant 0.05, epochs range(10), SGD nesterov 0.9) | train_static: epochs=range(10), constant(0.05), recipe_optimizer; same as train_host | test_training_stream... (10 hashes, 20 losses) ; moving module: test_one_step_moves_the_module | met |
| Module is M-sq (w tanh(z) + b, 16x16, 272) and equals Adapter nonlinear | forward einsum identical to Adapter.apply 2-weight branch; MODULE_SHAPES | test_no_refit_term_is_the_complete_model (direct forward vs losses via adapter, atol 1e-5) | met |
| Then module grid refitted on the frozen static host | _seal, static_adapter, static_features, load_static | test_refit_warm_starts_from_the_learned_module; test_save_load_round_trip | met |
| Ruling R8: no optimiser state saved | save_static writes only host.pt and static-module.pt; train_static returns no optimiser | test_save_load_round_trip asserts directory listing is exactly those two files | met |

## Frozen values checked
| value | prereg | source | match |
|---|---|---|---|
| epochs | 10 | range(10) | yes |
| learning rate | constant 0.05 (recipe) | constant(0.05); recipe_optimizer lr 0.05 | yes |
| weight decay on w | recipe's 5e-4 | host.py recipe_optimizer 5e-4 (unchanged by diff) | yes |
| module shapes / retained count | 16x16 + 16 = 272 | MODULE_SHAPES, zeros(16,16), zeros(16) | yes |
| seed | anchor's own seed, width 0 host | make_host(seed, 0, ...) | yes |
| t3_boundary fingerprint | n/a (implementation) | not a prereg value | n/a |

No frozen value needs changing; nothing is amendment territory.

## Findings
- Minor, tests/test_m0_retune.py:336: StaticTests discards the 5th return of train_static (`initial`); the step-zero capture is pinned only in the worker tests (test_m0_worker.py:350,393). Moving `initial = state_hash(host)` to after training would not fail StaticTests, but would fail the worker tests. Fix (optional): assert `initial == state_hash(make_host(3000, 0, CPU, widths=SMALL))` in StaticTests.
- Minor, static.py `StaticHost.forward`: duplicates Adapter.apply arithmetic by hand rather than calling it; equality is pinned only by an allclose (1e-5) test, so a drift in tolerance-level arithmetic would pass. Acceptable (comment documents it).
- Minor (no prereg conflict): train_static trains with `static.parameters()` including inner host parameters in the same two groups as the anchor; momentum and decay thus match the anchor's recipe exactly. No extra arms, extra randomness, RNG-order change or Gate 3 receipt byte change found.

## Dispositions (controller, 2026-10-14)

- Minor 1–3: deferred, recorded. The step-0 capture is pinned by the worker tests and compared by the pilot at stage 3 (final review M8).

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
