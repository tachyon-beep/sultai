# Exact-source review: attempt4-pins-and-replay (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Scope: `src/sultai_screen/m0/worker.py` (`ATTEMPT4_FINAL_STATE`, `train_action`, `replay_action`, `main`) and `tests/test_m0_worker.py::WorkerTests`, against preregistration §2 lines 91-93, §9 stage 1 (520-530), §11 (572-599) and the V0 definition in §7 (line 373). HEAD verified at e1df6e1. Read-only. Evidence gathered: the six receipts read directly, the targeted WorkerTests run (4 passed, 7 subtests), a CPU equivalence run of M0 `replay` against the attempt-4 `screen.training_worker` on the test fixture, and one in-memory mutation run. Scripts were kept in the session scratchpad; the tree was not touched.

## Claims and evidence

| Claim (prereg) | Source file:line | Pinning test | Verdict |
|---|---|---|---|
| §2 l.91: "the attempt-4 `final_state_sha256` strings pinned in `src/`" | worker.py:121-130 `ATTEMPT4_FINAL_STATE`, six literal 64-hex strings keyed 3000-3005 | `test_attempt4_hashes_are_the_published_receipts` (test_m0_worker.py:310-321). It checks the key set is exactly 3000-3005 and that each literal equals `payload.final_state_sha256` of `docs/results/gate3-pilot-2026-10-13-linked-3/intact-<seed>/result.json`. Changing any literal, or adding or dropping a key, fails it. | Satisfied, pinned |
| §2 l.91-92: "so V0 compares a fresh receipt to a pinned string rather than to another fresh receipt" | worker.py:395-403 `replay_action` retrains through `train_action` (fresh receipt), then records `attempt4_final_state_sha256 = ATTEMPT4_FINAL_STATE[seed]` and `matches_attempt4`. The comparison that actually decides V0 is pilot.py:380 (`payload["final_state_sha256"] != ATTEMPT4_FINAL_STATE[item.seed]` raises PilotHold). The confirmatory reuses it at confirmatory.py:499-500. | `test_cooled20_replay_and_tcool_children` (test_m0_worker.py:371-381) asserts the pinned string is echoed, `matches_attempt4 is False` on fixture data, and that replay's final hash equals M0 `train`'s. Inverting the comparison fails it. A constant `False` passes it (F3). | Satisfied in source; partly pinned |
| Implied by §2 l.91-92 and V0 §7 l.373 ("Replay bitwise ... against the pinned attempt-4 hashes"): the fresh receipt is produced by the attempt-4 recipe, so a match is possible | worker.py:349-392 `train_action`, which runs make_host, then the probe, then `recipe_optimizer`, then `train_epochs(range(10), constant(0.05))`, and hashes after `freeze`. I verified it on CPU: the fixture replay is bitwise equal to `screen.training_worker` (attempt 4's code path) for the final state, every epoch-state hash, the loss stream, the forward_backward probe and the checkpoint file sha. | **None in WorkerTests.** Replay is compared only to M0 `train`, which shares the same code. The mutation `constant(0.05)` to `constant(0.04)` in the worker survives all three relevant WorkerTests (F1). | Satisfied in source (verified); not pinned |
| §9 stage 1 l.524: "replay of attempt 4's pinned state hashes" covers the six development anchors 3000-3005 | worker.py:397-398 refuses any lineage other than `anchor` and any seed outside the pin keys. main() at worker.py:1384 admits through `seeds.admit` (the anchor block is `range(3000, 3006)`, seeds.py:21). | The anchor block is pinned by the seed-admission item. The worker-side refusal at :397-398 is not exercised (F2). | Satisfied; guard unpinned |
| §2 l.93: "seed admission beyond 3000–3005 (`main()` refuses other seeds today)" | worker.py:1358-1394 `main()` admits through `seeds.admit(args.seed, lineage)` with `--lineage` and `--scale` checks. | `test_cli_admits_seeds_by_lineage_scale_width_and_epochs` (test_m0_worker.py:261-308). Frontier 6000 is admitted. 3000 under frontier, 6150 (M2) and a scale mismatch are refused. The block contents themselves belong to the seed-admission-and-identity item. | Satisfied (cross-item) |
| §11 l.572-575: anchors stay 3000-3005, and no other lineage reuses them | seeds.py BLOCKS (anchor `range(3000, 3006)`). Replay is keyed to the same six seeds. | As above | Satisfied |

## Frozen values checked

| Value | Prereg | Source | Match |
|---|---|---|---|
| Pin 3000 | intact-3000 `payload.final_state_sha256` = f607728b...e1f607 | worker.py:124 f607728bff91b9cab9241c039addbc93d62f2edea70303e1f89e39dd30f1e607 | Yes (read from file) |
| Pin 3001 | b9550263...750b91 | worker.py:125 b955026325e2c0fddfc4a1d677fb3795c77b5c200bc3b70ed175782f95750b91 | Yes |
| Pin 3002 | e7f1a21f...31c960 | worker.py:126 e7f1a21f00651f9229acf285c0f372076b81eeb7949535f8a4ebf229aa31c960 | Yes |
| Pin 3003 | ea015274...1f86 | worker.py:127 ea01527424ec535a5747f85a95fde2c62d40d4bc7dffb51f88781dbc019f1e86 | Yes |
| Pin 3004 | f9a39442...505777 | worker.py:128 f9a39442fb96fd2667894d65c707ec0db070ad021644189e64b1d95c2a505777 | Yes |
| Pin 3005 | aa5b5af4...dabdedb9 | worker.py:129 aa5b5af44ceef02eb555cb62b0f492d34893b4d210b505b08484a6874abdedb9 | Yes |
| Count of pins | six (§9, §11: 3000-3005) | 6 keys, exactly range(3000, 3006) | Yes |
| Replay recipe: epochs | 10 (attempt-4 receipt `epochs: 10`, `steps: 12500`) | worker.py:356, 10 for non-cooled lineages, enforced at :357 and :1387-1389 | Yes |
| Replay recipe: rate | constant 0.05 (host.train_host) | worker.py:370 `constant(0.05)` | Yes |
| Replay recipe: widths and bottleneck | s = 1.0, REFERENCE_WIDTHS (64, 64, 128, 128), width 0 | `FRONTIER_WIDTHS[1.0]` = (64, 64, 128, 128) (modules.py:21). `make_host(seed, 0, ...)`. `--width` must be 0 (:1379-1380). | Yes |

## Findings

**F1 (Important): no test pins that the M0 replay recipe reproduces the attempt-4 training path.**
- **Where:** tests/test_m0_worker.py:371-381, against worker.py:349-403.
- **What:** WorkerTests only checks that replay's final hash equals M0 `train`'s final hash. Both come from the same `train_action`, so any change to that recipe moves both sides together. Examples are the rate, the epoch range, optimiser construction, probe ordering that touches global RNG, or the hashed module. In-memory mutation: patching `worker.constant` to produce 0.04 left `test_attempt4_hashes...`, `test_cooled20_replay...` and `test_train_binds...` all passing. tests/test_m0_host.py pins `host.train_host` against a pre-refactor golden, but `train_action` does not call `train_host`; it reassembles `recipe_optimizer` and `train_epochs` inline to keep the momentum buffers.
- **Why it matters:** this is the property the pins exist for. The source is correct today: I ran it on CPU and the fixture replay was bitwise equal to `screen.training_worker` on the final, epoch, stream, probe and checkpoint hashes. A regression would surface only as a GPU V0 PilotHold, which costs a pilot attempt. The science would not be corrupted.
- **Fix:** add a WorkerTests case that runs `replay` (or `train`) for anchor 3000 on the fixture and asserts `final_state_sha256`, `epoch_state_sha256`, `loss_stream_sha256` and `forward_backward` equal either `host.train_host(make_host(3000, 0, CPU), fixture(), 3000)` or the test_m0_host golden constants. Importing `sultai_screen.screen` in a test only would also work, since the import-isolation test runs in a subprocess.

**F2 (Minor): replay's lineage and seed guard is untested.**
- **Where:** worker.py:397-398.
- **What:** no test runs `replay` under a non-anchor lineage, such as `confirmatory` 7500 or `frontier-0.5` 6000, to check the PilotHold. If the guard were deleted, an out-of-range seed would still fail, but with a KeyError at :400 rather than the stated PilotHold, and no test would notice.
- **Why it matters:** this is defence in depth; `main()` admission already confines anchor seeds.
- **Fix:** add one `assertRaisesRegex(PilotHold, "attempt-4 anchors")` call on `replay_action` with a `frontier-0.5` Request.

**F3 (Minor): the `matches_attempt4 is True` branch is never exercised.**
- **Where:** worker.py:402; test_m0_worker.py:375.
- **What:** the fixture can only produce `False`, so a mutant that hard-codes `False` passes. The field is informational: pilot.py:380 recomputes the comparison against `ATTEMPT4_FINAL_STATE` and does not read the flag, so V0 is unaffected.
- **Fix:** either patch `ATTEMPT4_FINAL_STATE` to `{3000: <fixture train final hash>}` for one run and assert `True`, or drop the flag and let pilot.py's comparison stand alone.

**F4 (Minor): a stale source comment.**
- **Where:** worker.py:121-122.
- **What:** the comment says the pins are "replayed on GPU in pilot V0 only". The confirmatory also runs `stage_replay` (confirmatory.py:499-500), which is correct per §7 V0 and §10 ("V0 then V1 and V2"). The code is right and the comment is wrong.
- **Fix:** change the comment to "replayed on GPU in V0 (pilot and confirmatory)".

**No BLOCKING findings.** No frozen value has to change for the code and the document to agree.

Behaviour beyond the prereg:
- Replay also writes `optimizer.pt`, `initial_state_sha256`, `lineage`, `scale`, `widths` and `schedule`, because it serves as the anchor host for T-cool and later stages (pilot.py:573-580).
- These are M0-only receipt fields; `worker.py` does not import `sultai_screen.screen` or `sultai_screen.pilot` (test at :323).
- The fixture equivalence run shows no change in RNG order: init runs under `fork_rng`, and the probe uses a separate host.
- Cross-item dependency, not reviewed here: bitwise equality with the attempt-4 pins on GPU also requires `host.py`'s s = 1.0 parameter and buffer names, order and init to be unchanged since commit 027a258. That belongs to the width-tuple-host item and the test_m0_host golden pin.

## Dispositions (controller, 2026-10-14)

- F1 (Important): FIXED in the post-review commit (item C2: the train/replay child's hashes pinned to `host.train_host` on the fixture).
- F2: FIXED (item C3). F4: FIXED (item B8, comment). F3: deferred, recorded (the pilot recomputes the comparison; the flag is informational).

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
