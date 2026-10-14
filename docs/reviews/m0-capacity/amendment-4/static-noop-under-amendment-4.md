# Exact-source review: static-noop-under-amendment-4 (commit d2e47a9)

**Verdict:** ACCEPTED WITH FINDINGS

Reviewer: independent exact-source reviewer (read-only; no working-tree, index or branch change; `git status --short`
before and after shows only the untracked `docs/results/m0-amendment-4-2026-10-14/`). No CIFAR file was opened: every
test and mutation run used the suites' synthetic fixtures (`tests/test_m0_worker.py::fixture`, random uint8 images).

Scope read in full: `docs/plans/2026-10-14-m0-amendment-4.md`, `docs/certificates/2026-10-14-m0-freeze-4.md`,
`src/sultai_screen/m0/calibrate.py` (`recalibrate_bn`, `no_op`), `src/sultai_screen/m0/static.py`,
`src/sultai_screen/m0/worker.py` (`_noop_rule`, `tcool_action`, `static_action`, `_static`, `_static_arms`,
`_tcool_arms`), `src/sultai_screen/m0/pilot.py::stage_hosts`, `src/sultai_screen/m0/identity.py`, the supporting
`host.py` (`make_host`, `state_hash`, `freeze`, `load_host`, `cached`), `m0/data.py` (`seen_fit`,
`features_from_images`), `fit.py` (`Adapter.apply`, `losses`, `mean_loss`), `m0/stats.py` D5 lines,
`m0/confirmatory.py::stage_hosts`; prereg §4 lines 216-223, §5 line 295, §6 line 345, §7 line 380.

Test evidence: the four affected suites (`test_m0_identity`, `test_m0_retune::StaticTests`, `test_m0_worker`,
`test_m0_pilot`) run at HEAD: 108 tests, 0 failures, 0 errors. Mutation checks were run out of tree by monkeypatching
the imported production names from scratchpad scripts (no file in the repository was edited); results in the table
under claim 3.

## Claims and evidence

| # | A4.1 claim (quoted) | Source file:line | Pinning test | Verdict |
|---|---|---|---|---|
| 1 | "After the ten-epoch training with the module present from step zero" (training recipe unchanged) | `static.py:59-79` (`train_static`, unchanged in the diff); `worker.py:462` calls it first, before any no-op work | `test_m0_retune.py::StaticTests` training tests (unchanged); `test_m0_worker.py:387` checks `steps`, stream binding | Satisfied |
| 2 | "the complete static model (inner host with its designed-in module) is given the §4 no-op rule" | `worker.py:467-471`: unsealed copy loaded from the just-published raw files by the raw hash (`load_static(..., frozen=False)`, `static.py:169-190`), recalibrated, then `static_no_op` | `test_m0_worker.py:387` (choice is one of R-bn/raw, chosen hash = `static_sha256`) | Satisfied |
| 3 | "its BatchNorm statistics are re-estimated over the same 2,000 fit examples" | `worker.py:470` passes `seen_fit(data)`, the same call `_noop_rule` (`worker.py:416`) and `noop_action` (`worker.py:539`) use; `seen_fit` is `m0/data.py:22-27` (own generator, 2,000 ids) | none specific (worker test patches `seen_fit` to the fixture's ids for every action alike) | Satisfied by source; same-indices not separately pinned (shared call) |
| 4 | "by the same cumulative pass" | `static.py:98-116` is line-for-line `calibrate.py:31-47`: sealed refusal (98-101 / 31-32), nonempty 1-D index check (102-103 / 33-34), BN list (104-106 / 35-37), `momentum = None` + `reset_running_stats()` (107-109 / 38-40), train mode (110 / 41), `torch.no_grad()` (111 / 42), same `RECALIBRATION_CHUNK` imported from `calibrate` (`static.py:27`, 112-114 / 43-45), `indices` order with partial last chunk, `normalized(images[ids])`, then seal (`_seal`, which calls `freeze(static.host)` and seals `w`, `b`; 115 / 46). Unsealed load returns a fresh `make_host` with no `_bn_guard` hooks, so the train-mode pass cannot trip the frozen guard | `test_m0_retune.py:484` pins only: buffers change, only `running_*`/`num_batches_tracked` change, parameters and module unchanged, sealed in/out. Mutations momentum-default, no-reset, eval-mode pass, chunk 7, reversed order all PASS the tests (Finding I-2) | Source satisfied; tests do not pin "same pass" |
| 5 | "with the module in the forward so that every statistic downstream of the site is that of the designed-in network" | `static.py:114` calls `static(...)`, i.e. `StaticHost.forward` (`static.py:43-47`: front → bottleneck → `z + w·tanh(z) + b` → `classify`) | `test_m0_retune.py:484`, last block: recalibrated buffers must not all equal plain `recalibrate_bn`'s. Mutation "forward through `static.host`" FAILS this test (`True is not false`) | Satisfied and pinned (pin relies on trained `w ≠ 0`, asserted by `test_one_step_moves_the_module` and the training test) |
| 6 | "the raw and recalibrated static models are compared by mean selection loss, each on features recomputed from itself" | `static.py:122-123`: `mean_loss(static.host, static_features(static, data, data.selection), static_adapter(static))` evaluated separately for each model; `static_features` = `features_from_images` (`m0/data.py:30-44`, same 128-chunk computation as `host.cached.part` used by `no_op`, `calibrate.py:115-116`); `static_adapter` applies exactly `StaticHost.forward`'s site (`fit.py:28-32`) | `test_m0_retune.py:513` second half mirrors the production comparison. Mutations "raw model's features for both" and "compare without the module (adapter None)" PASS the tests (Finding M-1) | Source satisfied; not discriminated by tests |
| 7 | "the lower wins and a tie keeps the raw model" | `static.py:125-127`: strict `<`, else `(raw, "raw")`, identical to `calibrate.py:117-119` | `test_m0_retune.py:513` first half (`static_no_op(x, x)` → "raw", `is` raw). Mutations "tie to recalibrated (`<=`)" and "always raw" FAIL it | Satisfied and pinned |
| 8 | "The chosen model is the frozen static host every later reading uses: the no-refit term L(S-static), the module grid refit warm-started from its learned module (S-static:refit), and D5 with and without the refit" | `worker.py:472` publishes the chosen model's `host.pt` and `static-module.pt` (raw save at 463 kept if raw wins; `save_static(chosen)` overwrites both if R-bn wins); `worker.py:494-502` `_static` binds `host.pt`/`static-module.pt`/`loss-stream.bin` and loads by `static_sha256` (= chosen, via `**record` at 490); `_static_arms` (`worker.py:1007-1028`) uses that model for `S-static` and `S-static:refit` (fit, selection, development features and warm start all from it); `stats.py:111-112, 262-263` D5/D5_norefit read those two arms; pilot binds `STATIC_FILES` (`pilot.py:598`, `receipts.py:43-47`) | `test_m0_worker.py:387`: `static_sha256` = the hash named by `noop_choice`, `load_static` of the published files by that hash succeeds, `host_sha256` = published host's state hash, `checkpoint_sha256`/`module_sha256` = file hashes. Fixture takes the R-bn branch (verified empirically), so mutation "raw files published, recalibrated hash recorded" FAILS (`PilotHold: static host or module differs from its recorded identity`). Raw branch untested in the worker path (Finding M-2); verified out of tree by forcing it: consistent | Satisfied; R-bn branch pinned |
| 9 | Module tensors unchanged by recalibration (prompt 1c) | `static.py:107-115` touches only BN modules' momentum/buffers; `w`, `b` copied bitwise from the file at `load_static` (`static.py:183-185`) | `test_m0_retune.py:484` `torch.equal(recalibrated.w, self.static.w) and ...b`. Mutation "module touched (`w += 1e-3`)" FAILS it (the worker test alone would not catch it, since the hash is recomputed on the touched model) | Satisfied and pinned |
| 10 | `checkpoint_sha256` taken after the final save (prompt 1c) | `worker.py:484` evaluated in the return dict, after line 472's possible overwrite | `test_m0_worker.py:387` (`checkpoint_sha256 == file_sha(host.pt)`), R-bn branch | Satisfied and pinned |
| 11 | "The choice ("R-bn" or "raw") and both state hashes are published in the static host's receipt" | `worker.py:486-488`: `noop_choice`, `raw_static_sha256`, `recalibrated_static_sha256`; chosen as `static_sha256` via `**record` | `test_m0_worker.py:387` (all three keys used, raw ≠ recalibrated) | Satisfied and pinned |
| 12 | "...and in the pilot's host receipt beside T-cool's" | `pilot.py:590, 601, 617`: `static_noop_choice` per anchor in `03-hosts.json`. The two state hashes are NOT carried into `03-hosts.json` (only `static_sha256`, the chosen one, at 600/616); T-cool's own no-op choice is not in `03-hosts.json` either (only `tcool_final_state_sha256`, the pre-N0 host) | `test_m0_pilot.py:539` pins `static_noop_choice`; mutation "pilot drops the key" FAILS (`KeyError`). FakeCluster always returns "raw" (Finding M-3) | Choice satisfied and pinned; both hashes missing from the pilot's host receipt (Finding I-1) |
| 13 | Unchanged: "seed, initialisation and data stream (§5)" | `train_static` untouched; the no-op work happens after training (`worker.py:465-472`); no global RNG drawn: `make_host` is under `torch.random.fork_rng` (`host.py:93-96`), `seen_fit` uses its own generator, `StaticHost.__init__` uses zeros | existing training/stream tests (unchanged) | Satisfied |
| 14 | Unchanged: "the V0 clause that its tensors are bitwise the anchor's at step 0 (§7, acceptance item (iii))" (prompt 1d) | `static.py:66-67`: `initial = state_hash(host)` before `StaticHost` and training; returned at `worker.py:462`, published at 481; pilot comparison unchanged at `pilot.py:603-610` | `test_m0_worker.py:387` last assertion (= train's initial hash); `test_m0_pilot.py` static_step0 assertions (unchanged) | Satisfied and pinned |
| 15 | Unchanged: "the warm-start refit, S-static's descriptive role at the pilot, its SD(D5) admission rule and the 14-hour ceiling" | `_static_arms` unchanged except that it now receives the chosen model; `stats.py`, `verdicts.py`, `receipts.py`, `confirmatory.py` not in `git diff --stat 086f123..d2e47a9` | n/a | Satisfied |
| 16 | "The pass costs one recalibration and two selection evaluations inside the static child, measured by the pilot as the whole child (§8, ruling R23)" | `worker.py:470-471` (one recalibration, two `selection` calls in `static_no_op`); the charge stays the whole `static-{s}` child (`pilot.py:662`, `receipts.py:321`); `noop_seconds` (`worker.py:466, 474, 489`) is an extra disclosure field that mirrors T-cool's (`worker.py:440, 454`) and enters no charge | `test_m0_worker.py:387` (`noop_seconds > 0`) | Satisfied |
| 17 | Identity: `PROTOCOL_VERSION = "m0-v4.2"`, six documents composed under the Gate 3 v4.1 rule | `identity.py` (`AMENDMENT_4_PATH`, `AMENDMENT_4_SHA`, `PROTOCOL_VERSION`, composed `PROTOCOL_SHA`, `protocol_identity()` adds `amendment_4_sha256`, `source_identity()` checks the sixth file) | `test_m0_identity.py:79` re-hashes all six files and pins the literals; pilot/worker tests pin `m0-v4.2` | Satisfied and pinned |

Mutation results (out of tree; each row runs `StaticTests.test_recalibrate_static_...`, `StaticTests.test_static_no_op_...` and `WorkerTests.test_cooled20_replay_and_tcool_children` unless noted):

| Production change simulated | Result |
|---|---|
| none (baseline) | 3 run, 0 fail |
| `recalibrate_static` forwards through the plain host (`static.host(...)`) | FAILS `test_recalibrate_static_...` |
| tie goes to the recalibrated model (`<=`) | FAILS `test_static_no_op_...` |
| always "raw" (no-op rule skipped) | FAILS `test_static_no_op_...` |
| `static_action` publishes raw files but records the recalibrated hash (worker test only) | FAILS `test_cooled20_...` (`PilotHold` at `load_static`) |
| module tensors changed during recalibration | FAILS `test_recalibrate_static_...` |
| pilot drops `static_noop_choice` (`test_clean_pilot_...` only) | FAILS (`KeyError`) |
| BN momentum left at default 0.1 | passes all (not caught) |
| running statistics not reset | passes all (not caught) |
| pass run in eval mode (statistics only reset, never re-estimated) | passes all (not caught) |
| chunk size 7 instead of `RECALIBRATION_CHUNK` | passes all (not caught) |
| fit examples in reversed order | passes all (not caught) |
| comparison uses the raw model's selection features for both models | passes all (not caught) |
| comparison without the module (adapter `None`) | passes all (not caught) |

## Hashes recomputed

`sha256sum` on the working tree at d2e47a9 and on `git show d2e47a9:<path> | sha256sum` gave identical values.

| Document | Pinned (identity.py, test, certificate 4, amendment 4 where cited) | Recomputed | Match |
|---|---|---|---|
| `docs/plans/2026-10-13-capacity-frontier-screen.md` | `c66599459adc898f96322ab11dfa3c6737cd3559e06c9432d79e377e00c03fd5` | `c66599459adc898f96322ab11dfa3c6737cd3559e06c9432d79e377e00c03fd5` | yes |
| `docs/STEERING.md` | `08a9e1aaf79d3048a5d6f6f4d6138a9c7049f32b61c0250e1505773a27fa3ffd` | `08a9e1aaf79d3048a5d6f6f4d6138a9c7049f32b61c0250e1505773a27fa3ffd` | yes |
| `docs/plans/2026-10-13-gate-ladder-amendment.md` | `af12d474a97aad83bc63405034fec3716170f7652f34364588750732c97b6a89` | `af12d474a97aad83bc63405034fec3716170f7652f34364588750732c97b6a89` | yes |
| `docs/plans/2026-10-13-gate-ladder-amendment-2.md` | `b4740b147a4a1ced52a6f0721215679c1567078260d5c11a92c35fa2065c2d77` | `b4740b147a4a1ced52a6f0721215679c1567078260d5c11a92c35fa2065c2d77` | yes |
| `docs/plans/2026-10-14-m0-amendment-3.md` | `6dcfd9f2a781dc56d39c6e4fe548fc4728c2c743a8e5a978d0e4b9741459e862` | `6dcfd9f2a781dc56d39c6e4fe548fc4728c2c743a8e5a978d0e4b9741459e862` | yes |
| `docs/plans/2026-10-14-m0-amendment-4.md` | `d8dd571bb39718946af06a442ec74302b704c1bede090c83eeaed9d59dcdcc34` | `d8dd571bb39718946af06a442ec74302b704c1bede090c83eeaed9d59dcdcc34` | yes |
| Composed `printf 'm0-v4.2\n%s\n%s\n%s\n%s\n%s\n%s\n' ... \| sha256sum` | `a2c1e55deffb144c9ab596ed7817bc6546f88ac045a02fc0447ae25896db6007` | `a2c1e55deffb144c9ab596ed7817bc6546f88ac045a02fc0447ae25896db6007` | yes |

Every 64-hex literal in `identity.py`, `tests/test_m0_identity.py`, the amendment and certificate 4 is one of the seven
values above (grep of all `[0-9a-f]{64}` tokens in the four files: no other value). No stale `m0-v4.1` or `dfd13cb4`
literal remains in `src/` or `tests/` (certificate 4's prose reference to the superseded v4.1 identity is intentional).
`git log --oneline 086f123..d2e47a9` is the single commit d2e47a9; `git diff --stat 086f123..d2e47a9` lists exactly
ten files (certificate 4, amendment 4, `identity.py`, `pilot.py`, `static.py`, `worker.py`, four test files), none of
the five previously pinned documents, `calibrate.py`, `verdicts.py`, `stats.py`, `receipts.py`, `confirmatory.py` or
any Gate 3 module.

## Findings

### BLOCKING

None. No frozen value would have to change: every finding is fixed by additive code or tests.

### Important

**I-1. The pilot's host receipt carries the static choice but not "both state hashes"; "beside T-cool's" has no
referent there.** `src/sultai_screen/m0/pilot.py:597-601, 614-617`.
What: A4.1 says "The choice ("R-bn" or "raw") and both state hashes are published in the static host's receipt and in
the pilot's host receipt beside T-cool's." The static child's receipt has all three (`worker.py:486-488`).
`stage_hosts` copies only `noop_choice` into `03-hosts.json` (`static_noop_choice`); `raw_static_sha256` and
`recalibrated_static_sha256` never reach it (it carries `static_sha256`, the chosen hash, only). Separately,
`03-hosts.json` holds no T-cool no-op choice to sit beside: it records `tcool_final_state_sha256` (the pre-N0 T-cool
host), and T-cool's `noop_choice`/`noop_sha256` live in the tcool child receipt and the T-cool arm record
(`worker.py:960-964`).
Why: The amendment is frozen and hashed, so the source must match its text. The information is recoverable from the
bound child receipt, so no result is at risk, but the host receipt is under-populated against the literal text.
Fix (code only, no document change): in `stage_hosts`, add per-anchor `static_raw_sha256` and
`static_recalibrated_sha256` from the child payload, and, so that "beside T-cool's" has its referent, T-cool's
`noop_choice` (and `noop_sha256`) from the tcool payload; extend the `03-hosts` assertions in
`test_m0_pilot.py::test_clean_pilot_...` (FakeCluster must then return those keys for both actions). Alternatively the
controller records a disposition that "beside T-cool's" means "in the same stage-3 receipt as T-cool's host hashes";
the two missing static hashes still need adding. A third code-only option for the "beside" clause: `_static_arms` (`worker.py:1015-1016`) adds `{"noop_choice": receipt["noop_choice"]}` to the S-static arm record, mirroring the T-cool arm record (`worker.py:964`), so the two choices sit side by side in the modules receipt; this does not replace adding the two hashes to `03-hosts.json`.

**I-2. No test pins that `recalibrate_static` is "the same cumulative pass" as `recalibrate_bn`.**
`tests/test_m0_retune.py:484-511` (source `src/sultai_screen/m0/static.py:98-116` is correct by line-for-line reading).
What: The only behavioural pins are "some buffer changed, only BN buffers changed, parameters/module unchanged, result
differs from plain R-bn". Simulated production changes that leave momentum at 0.1, skip `reset_running_stats()`, run
the pass in eval mode (so statistics are reset to 0/1 and never re-estimated), use a chunk of 7, or reverse the fit
order all pass every affected test.
Why: Claim (a) of the review brief and the amendment's "by the same cumulative pass" is a central equivalence. A future
edit to `recalibrate_static` (or a divergence if `recalibrate_bn` is changed under a later amendment) would go
unnoticed; the eval-mode case would silently turn R-bn into a statistics wipe.
Fix: in `test_recalibrate_static_...`, assert that every BN buffer upstream of the site (`front.*`; the bottleneck is
`Identity` at width 0) is bitwise equal to plain `recalibrate_bn`'s on the same host and indices, and that at least one
tail buffer differs. Use an index tensor longer than `RECALIBRATION_CHUNK` and not a multiple of it (e.g. 200 of the
fixture's images repeated, or a larger fixture) so the chunking and partial last chunk are pinned too. The front
equality catches all five surviving mutations; the existing "not all equal" assertion keeps the module-in-forward pin.

### Minor

**M-1. "Each on features recomputed from itself" and "decided on the complete model" are not discriminated by tests.**
`tests/test_m0_retune.py:513-531`; source `static.py:122-127`.
What: the test's second half recomputes the same quantity as production and checks the same decision; on the fixture,
a comparison that reuses the raw model's selection features for both, or that drops the module (adapter `None`),
reaches the same winner, so both mutations pass.
Why: These are explicit A4.1 clauses; parity note: the static receipt, like T-cool's `_noop_rule` receipt, does not
publish the two selection losses (`noop_action` does, `worker.py:558-559`), so no receipt-level check exists either.
Fix: a unit test with a spy on `static_features`/`mean_loss` asserting two feature computations, one per model, each
with that model's own adapter; or publish `raw_selection_loss`/`rbn_selection_loss` in the static receipt and pin them
against an independent computation.

**M-2. The worker test exercises only the R-bn branch of `static_action`.** `tests/test_m0_worker.py:404-417`;
source `worker.py:472`.
What: the assertions are branch-conditional; on the 64-image fixture the static child takes R-bn (verified
empirically), so `record = raw` (raw chosen, no second save) never runs in the worker suite.
Why: half of the publication logic is unexercised. I forced the raw branch out of tree (patched `static_no_op`):
`static_sha256 == raw_static_sha256`, `checkpoint_sha256`/`module_sha256` equal the file hashes, `load_static` by the
recorded hash succeeds, module bytes identical across branches, `initial_state_sha256`/raw/recalibrated hashes
identical across branches. Production also has an independent guard (`load_static` raises on hash/file mismatch;
`_bound` and `bound(STATIC_FILES)` pin the file hashes).
Fix: add a worker test that patches `sultai_screen.m0.worker.static_no_op` to force the branch the fixture does not take
and repeats the publication assertions.

**M-3. The pilot fixture always reports "raw".** `tests/test_m0_pilot.py:317-323, 539`.
What: FakeCluster returns `"noop_choice": "raw"` for every anchor, so a pilot that hard-coded "raw" (instead of reading
the payload) would pass.
Fix: make FakeCluster return a seed-dependent choice (e.g. "R-bn" for one anchor) and assert the per-anchor mapping.

**M-4. Confirmatory host receipt does not surface the static choice.** `src/sultai_screen/m0/confirmatory.py:333-351`.
What: under the 14-hour flag the confirmatory runs the same `static_action` and binds `STATIC_FILES`, but its
`03-hosts.json` records only `static_sha256`. A4.1 names only "the pilot's host receipt", so this is not a deviation;
the bound child receipts carry the choice and both hashes.
Fix: optional, for symmetry with I-1; otherwise record "not required by A4.1".

### Observations (no action required)

- Nothing beyond the amendment was added to behaviour: the extra receipt field `noop_seconds` mirrors T-cool's and
  enters no charge; `load_static` gained a keyword-only `frozen=True` default, so every existing caller is unchanged;
  `ContractViolation`, `normalized`, `mean_loss`, `RECALIBRATION_CHUNK` imports only. No new randomness and no change
  to RNG order (the pass runs after training; `make_host` under `fork_rng`; `seen_fit` on its own generator).
  `calibrate.py`, `no_op`, the T-cool rule, `verdicts.py` and every Gate 3 path are untouched.
- When R-bn wins, the raw static files are overwritten (only the raw hash survives), whereas T-cool keeps its raw
  `host.pt` beside `noop.pt`. A4.1 asks only for the hashes, so this conforms; the raw model is reproducible by
  deterministic retraining and the training stream is still bound.

## Dispositions (controller, commit 150b4d2)

| Finding | Disposition |
|---|---|
| I-1 (Important) | Adopted: `03-hosts.json` carries per anchor `static_noop` (choice, raw and recalibrated static hashes) and `tcool_noop` (choice, noop hash); the fixture's choice is seed-dependent and the mapping asserted. The confirmatory host receipt records `static_noop_choice` (M-4, symmetry). |
| I-2 (Important) | Adopted: on a 200-index tensor every front BatchNorm buffer of `recalibrate_static` is bitwise `recalibrate_bn`'s on the same host, and a downstream buffer differs (module in the forward). |
| M-1 | Adopted: `StaticChoice` carries both selection losses; published in the static receipt and pinned against an independent computation (features recomputed from each model with its own module). |
| M-2 | Adopted: `test_static_action_publishes_the_raw_model_when_the_rule_keeps_it` patches the rule to force the raw branch. |
| M-3 | Adopted: seed-dependent fixture choice. |
| M-4 | Adopted (optional): the confirmatory stage-3 receipt records the choice; hashes stay in the bound child receipt. |
