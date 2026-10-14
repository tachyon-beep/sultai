# Exact-source review: width-tuple-host (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Scope: `src/sultai_screen/host.py` and `tests/test_m0_host.py::HostTests` at e1df6e1 (HEAD verified), diff `b79f145..e1df6e1`. The reviewer reran HostTests (12 passed, CPU, `PYTHONPATH=src`), built an independent copy of `b79f145:src` in a scratchpad, and ran mutations against scratch copies only. The working tree was not modified.

## Claims and evidence

| # | Claim (prereg) | Source file:line | Pinning test | Verdict |
|---|---|---|---|---|
| 1 | "A tail-width constructor parameter that accepts an explicit width tuple (not only s ∈ {0.25, 0.5, 0.75, 1.0, 1.5})" (§2 l.86-88) | host.py:61 `widths: tuple[int,int,int,int] = REFERENCE_WIDTHS`; :65-66 validation (4 positive ints, any values); :69-77 tail and classifier built from `widths`; :89-96 `make_host(..., widths=)`; :303-312 `load_host(..., widths=)` | `test_frontier_counts_match_the_frozen_table` (all five tuples to exact counts), `test_bottleneck_and_tuple_compose`, `test_invalid_widths_are_rejected`, `test_load_host_unfrozen_...` (non-reference tuple through `make_host`/`load_host`). Fails if: tail ignores `widths` (counts), validation removed (invalid test), `load_host` drops `widths` (mutation confirmed: checkpoint key/shape mismatch fails the load test). Gap: `make_host` dropping `widths` is not caught by HostTests (see Minor 1) | Satisfied |
| 2 | "keeps the bottleneck argument" (l.88-89) | host.py:61 `width` stays first positional; :63-64 registry (0, 2, 4, 8) unchanged; :68 bottleneck 16→width→16 unchanged | `test_reference_host_hash_and_count_are_unchanged` (width 4 hash `68c71ac0…`), `test_bottleneck_and_tuple_compose`, `test_invalid_widths_are_rejected` (`Host(3)`). Fails if the bottleneck is changed or dropped, or the registry is loosened | Satisfied |
| 3 | "leaves the s = 1.0 layer order, parameter and buffer names and RNG consumption bitwise identical" (l.89-91) | host.py:57 `REFERENCE_WIDTHS = (64, 64, 128, 128)` as the default; :67-77 same module order and indices as b79f145; validation (:63-66) draws no RNG; `sealed` (:80, :118) is a plain attribute, not in the state dict or buffers | `test_reference_host_hash_and_count_are_unchanged` (`state_hash` covers kind/name/dtype/shape/bytes of params and buffers, plus an explicit name list); `test_explicit_default_tuple_is_bitwise_the_reference`; `test_train_host_golden_pin_is_unchanged` (training RNG path). Mutation: one extra `torch.rand(1)` in `Host.__init__` fails 3 tests. Independent check: old (b79f145) and new code give identical state hashes, parameter and buffer name/shape lists, counts, and post-construction global-RNG draws for seeds 3000 and 3005 × widths 0/2/4/8 | Satisfied |
| 4 | "the state hash covers kind, name, dtype and shape" (l.91-92) | host.py:103-109 (unchanged) | indirectly, through the pinned hashes above | Satisfied |
| 5 | "the attempt-4 `final_state_sha256` strings pinned in `src/`" (l.92-93) | Not in this item's source. They are in `src/sultai_screen/m0/worker.py:121-130` `ATTEMPT4_FINAL_STATE` | not in HostTests (outside this item) | Out of item scope. The reviewer checked all six against `docs/results/gate3-pilot-2026-10-13-linked-3/intact-<seed>/result.json` `payload.final_state_sha256`: all match |
| 6 | §3: front "a fixed 14,928 parameters at every s" | host.py:67 (unchanged) | `test_frontier_counts_match_the_frozen_table` asserts front = 14,928 for each tuple | Satisfied |
| 7 | §3: s = 1.0 host is bitwise `Host(0)` | host.py:57, :61 default | `test_explicit_default_tuple_is_bitwise_the_reference`, `test_reference_host_hash_and_count_are_unchanged` | Satisfied |

## Frozen values checked

| Value | Prereg | Source | Match |
|---|---|---|---|
| Reference tail widths | (64, 64, 128, 128) | host.py:57 | Yes |
| s=0.25 tuple / count | (16,16,32,32) / 33,978 | test table; constructor gives 33,978; analytic recount 33,978 | Yes |
| s=0.5 | (32,32,64,64) / 85,274 | 85,274 | Yes |
| s=0.75 | (48,48,96,96) / 168,826 | 168,826 | Yes |
| s=1.0 | (64,64,128,128) / 284,634 | 284,634 | Yes |
| s=1.5 | (96,96,192,192) / 613,018 | 613,018 | Yes |
| Front | 14,928 | 14,928 (constructor and analytic) | Yes |
| Golden seed 3000 width 0 | 41d64c71… | test `REFERENCE_HASH` 41d64c71e61d…6b12; old and new code both produce it | Yes |
| Golden seed 3000 width 4 | 68c71ac0… | test `BOTTLENECK_HASH` 68c71ac0ba89…083e; old and new code both produce it | Yes |
| Registered bottleneck widths | 0, 2, 4, 8 (v4) | host.py:63 | Yes (unchanged) |
| Training golden pin (fixture) | unchanged recipe | `GOLDEN_FINAL_HASH` 17325d1d…, stream 56f5bff8…; the reviewer reproduced both with b79f145 `train_host` and with HEAD | Yes |
| Frontier mapping s to tuple (outside item) | §3 table | `m0/modules.py:17-23` `FRONTIER_WIDTHS` | Yes |

## Findings

**BLOCKING:** none. No frozen value has to change for the code and the document to agree.

**Important:** none.

**Minor 1: HostTests do not pin `make_host` forwarding `widths`.** host.py:95. Mutating `Host(width, widths)` to `Host(width)` leaves all 12 HostTests passing. In `test_load_host_unfrozen_...` both the source and the loaded host go through `make_host`, so both fall back to the reference shape consistently. Only `tests/test_m0_modules.py::FoldTests::test_interior_exact_and_border_over_adds_when_bias_nonzero` catches it, and only incidentally. Why it matters: every frontier host is built through `make_host`; a regression would silently train reference hosts under frontier lineages. Fix: in `test_frontier_counts_match_the_frozen_table`, also assert `parameter_count(make_host(3000, 0, CPU, widths=widths)) == count`.

**Minor 2: index transpositions within a tuple are unpinned.** host.py:70-77. Every registered tuple has w0 = w1 and w2 = w3. So using `widths[0]` for `widths[1]` (:71) or `widths[2]` for the classifier (:77) passes all tests; the reviewer confirmed both by mutation. There is no effect on any preregistered tuple. Fix (optional): add one asymmetric tuple, e.g. (8, 12, 20, 24), and assert the per-layer weight shapes.

**Minor 3: a list passes validation.** host.py:65 and :79. `Host(0, widths=[64, 64, 128, 128])` is accepted and stores a list in `self.widths`, despite the tuple annotation. mypy strict catches this statically, and no current consumer compares `host.widths` by equality (fold.py:45 and retune.py:93 only forward it). Fix (optional): add `type(widths) is not tuple` to the guard.

**Behaviour beyond this item (question 4).** The host.py diff also adds `constant`, `linear_cooldown`, `TCOOL`, `COOLED_20`, `recipe_optimizer`, `train_epochs`, `momentum_buffers`, `optimizer_state`, `restore_optimizer`, `load_host(frozen=)` and `sealed`. These serve other §2 clauses (T-cool resume, saved optimiser state, the 20-epoch cooled point) and belong to their own item reviews. Within this item's concern they do not alter the s = 1.0 path: `train_host` remains exactly 10 epochs at a constant 0.05; the golden training hash and the loss-stream hash are bitwise equal to b79f145; `load_host` defaults to `frozen=True`; `sealed` stays out of the state dict (asserted in the test). This item adds no extra arms, no extra randomness, no changed RNG order and no changed receipt byte.

## Dispositions (controller, 2026-10-14)

- Minor 1–3: deferred, recorded. `make_host` forwarding is pinned transitively by FoldTests and the worker's width-tuple receipt check; the registered tuples are symmetric by construction; a list argument is caught by mypy strict.

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
