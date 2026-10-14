# M0 amendment 5: determinism, replay and receipt review

Reviewer: determinism reviewer (Claude), 2026-10-14. Scope: `git diff main..13989d9 -- src tests`
in `/home/john/sultai-a5` (branch `nyx/m0-amendment-5`, base `main` 0d4179a). The target moved from 712b09a
to 13989d9 mid-review; line numbers below are at 13989d9. Read-only; no GPU; no subagents; CIFAR batch files
not opened. Checks executed: the m0 test modules on a `git archive` export of 13989d9 in the scratchpad with
`CUDA_VISIBLE_DEVICES=""` (`test_m0_provenance`, `_identity`, `_stats`, `_modules`, `_supervision`, `_pilot`:
165 tests OK in 223 s; `test_m0_worker`: 29 OK in 73 s; `test_m0_fit`, `_instrument`, `_host`, `_retune`:
66 OK in 52 s); `sha256sum docs/plans/2026-10-14-m0-amendment-5.md` = `0f6c2fa3…` as pinned in
`identity.py:30` and freeze certificate 5; the installed torch's `torch.backends.cudnn._init`/`version`
source read; the pilot of record's receipts on `nyx/m0-pilot-evidence` read for field shapes.

Equivalence predicate under review: bit-exact identity of `state_hash` (`host.py:103-109`) for the raw,
recalibrated and chosen no-op hosts across the two cards, from one shared serialized checkpoint; everything
else in this amendment is a receipt (recorded, never compared) or a CPU bootstrap.

## Summary

Accept with conditions.
The gate-B conditions are F1 and F2 (stop-path test pins) and F4's two test cases; F3, F5 and F4's cuDNN guard are recommendations the controller may bundle or defer.
At 13989d9 every A5.2 item is implemented as stated: the pins still refuse before custody, the four new fields are recorded and compared field by field with the pilot's opening record without refusing, the no-op receipts carry the fit-index hash and both deciding losses, the attempt-4 anchor's no-op is replayed on both cards before any fresh anchor trains and a mismatch is an instrument failure, terminal records carry certificate and custody on clean and stop paths, exit records come from the ledger in a `finally`, and the Gate 3 boundary is untouched.
Most important finding: the stop paths of items 4 and 5 are pinned by no test, so the `finally` and the stop-path fields could regress silently (F1, F2).
0 blocking, 0 major, 5 minor, 6 observations.

## Findings

**F1. Minor. The terminal-record fields on the stop path are not pinned.** Evidence:
`src/sultai_screen/m0/pilot.py:1318-1327` and `confirmatory.py:692-699` add `certificate_sha256` and
`custody` to the `except` branch; `tests/test_m0_pilot.py:457-463` and `:1710-1716` assert them on the clean
path only; `assert_burnt_stop` (`:1367-1376`) checks verdict, seed block and protocol sha, not these two keys;
the pilot stop helper (`:440-447`) likewise. Reasoning: deleting the two keys from either `except` branch
passes every test. Confidence: high (ran the suites). Action: add the two assertions to `assert_burnt_stop`
and the pilot stop helper.

**F2. Minor. Exit records are pinned on the clean path only; the `finally` semantics are untested.**
Evidence: `pilot.py:360-381` (`Run.run`: `before = len(ledger)`, `try: supervisor.run … finally: publish
<job>.exit.json` from `ledger[before:]`); `tests/test_m0_pilot.py:464-471` reads every `.exit.json` after a
clean pilot with `returncode == 0`. No stop-path test (child exit 1, ceiling timeout, signal) asserts the
failed job's exit record. Reasoning: the Gate 3 `Supervisor.run` ledgers every launched process in its own
`finally` (`src/sultai_screen/pilot.py:254-288`), signals raise `PilotHold` (`m0/pilot.py:1341`,
`confirmatory.py:716`), so the production path does write on supervisor failure, ceiling stop and signal; but
rewriting the `try/finally` as straight-line code would pass the suite. Job-name collision: the Supervisor
refuses a duplicate output directory (`pilot.py:176`) and duplicate names in a batch (`:161`), and `publish`
is exclusive (`contracts.py:101-104`); a collision needs a same-named job ledgered earlier, which needs a prior
child failure, which already stops the run, so the `FileExistsError`-inside-`finally` path is unreachable on a
live attempt. Confidence: high. Action: in an existing failing-child test, assert `<job>.exit.json` exists with
the non-zero `returncode`.

**F3. Minor. The cross-card no-op replay discards free evidence and never checks the fit-index hash.**
Evidence: `confirmatory.py:348` `NOOP_REPLAY_KEYS = ("raw_sha256", "recalibrated_sha256", "noop_sha256")`;
both children also publish `fit_indices_sha256`, `raw_selection_loss`, `rbn_selection_loss`
(`worker.py:570-590`), which `replay_noop` (`:351-369`) neither compares nor copies; nothing in src compares
`fit_indices_sha256` with `SEEN_FIT_SHA` (`m0/data.py:19`). Reasoning: `raw_sha256` equality is guaranteed by
the loader (`_training`, `worker.py:287-306`, refuses unless `state_hash == final_state_sha256` of the shared
receipt), so the real cross-card evidence is the recalibrated and chosen hashes. The two selection losses are
the frozen forward path on each card, which no state hash covers; comparing them bitwise costs nothing. Neither the fit-index tensor nor the data is single-sourced: each child derives the permutation in-process from a CPU generator (`data.py:22-27`) and loads the dataset independently from the sha-pinned batch files; what is shared by construction is the serialized `host.pt`, bound by `checkpoint_sha256`. The
fit-index hash is stable across devices and dtype (`tensor_sha`, `worker.py:210-211`: `.cpu().contiguous()
.numpy().tobytes()`; `partitions()` makes `train` int64 via `.long()`, `data.py:115`; native little-endian
equals the `<i8` pin at `tests/test_m0_fit.py:124`), but it is a record, not a check. Confidence: high.
Action: add `fit_indices_sha256`, `raw_selection_loss`, `rbn_selection_loss` to `NOOP_REPLAY_KEYS` and refuse
`fit_indices_sha256 != SEEN_FIT_SHA` in `replay_noop`.

**F4. Minor. The environment identity's failure modes are only partly tested, and one is not handled.**
Evidence: `provenance.py:41-52` catches `OSError` and `subprocess.SubprocessError` (so `TimeoutExpired` and
`CalledProcessError`) and returns absent on a row that does not split into exactly two fields; `:69-71`
`torch.backends.cudnn.version()` is unguarded. Installed torch source: `version()` calls `_init()`, which
raises `RuntimeError` on a runtime/compile cuDNN mismatch; `_cudnn.getVersionInt()` wraps `cudnnGetVersion`,
a library query that creates no CUDA context. `tests/test_m0_provenance.py:62-75` pins `OSError` and
`cudnn → None` only; the parse path and the timeout are untested, and `torch.cuda.init` is patched inert
(`:41`) so the test cannot see CUDA initialisation. Reasoning: narrowing the `except` to `OSError` passes the
suite; a cuDNN incompatibility would hold the launch on a field the amendment says never refuses. The live
guard behind "no CUDA context" is `check_cards` (`m0/pilot.py:172-186`), which runs right after
`environment_identity()` in both supervisors (`pilot.py:1228-1229`, `confirmatory.py:610-611`) and refuses
any active compute process on the pinned cards before custody; a context created by the supervisor's own call
would show there. Confidence: high on the handled paths, medium on "no context" (source, not a live probe).
Action: wrap the cuDNN call (`RuntimeError → None`); add a `TimeoutExpired` and a malformed-row case to the
provenance test.

**F5. Minor. The static host's no-op application publishes no fit-index hash.** Evidence: `worker.py:492-493`
`recalibrate_static(unsealed, data.images, seen_fit(data))`, record at `:498-514` carries `noop_choice`,
`raw_static_sha256`, `recalibrated_static_sha256`, both losses, no `fit_indices_sha256`; the anchor N0
(`:581`) and T-cool (`:442`) do. Reasoning: A5.2 item 2 names "every no-op rule application (the anchor's N0
and T-cool's)"; the static one is inside the purpose if outside the parenthesis. Confidence: high. Action:
one field in `static_action`, one assertion in the static worker test.

**F6. Observation. For the first confirmatory, `environment_vs_pilot` has no pilot baseline for the four new
fields.** Evidence: the pilot of record's `00-opening.json` (`nyx/m0-pilot-evidence`,
`docs/results/m0-pilot-2026-10-14-0d4179a/`) carries exactly the five pinned fields;
`environment_vs_pilot` (`confirmatory.py:333-342`) reports absent-on-one-side as `pilot: null, equal: false`.
Reasoning: correct and never refusing, as the amendment says, but the gate C narrative must not read
`equal: false` on python, driver, cudnn and gpu_names as a difference. The comparison also adds the pilot's
`00-opening.json` as a hash-listed admission requirement (`:268-270`, via `evidenced`); a certificate that
omits it refuses before custody, pinned by
`test_the_pilots_opening_record_must_be_hash_listed_for_the_environment_comparison`. Confidence: high. Action:
list `00-opening.json` in the confirmatory admission certificate; optionally split `equal` into
`pilot_recorded`/`equal`.

**F7. Observation. The no-op replay's charge is not inside the carried fixed V0 term.** Evidence: the
confirmatory's fixed term is the pilot's `fixed_v0_gpu_process_seconds` (552.58 s, `pilot-result.json`),
measured without a second noop child; `replay_noop` adds two children (the pilot's `noop-3000` charged 4.71 s,
`ledger-087.json`) to the ledger that no projection books, unlike the survival-refit uplift
(`confirmatory.py:186-200`). Reasoning: under 10 s against a 43,200 s ceiling; the amendment's "charged inside
the fixed V0 term" is true of the ledger, not of the admitted projection. Confidence: high. Action: say so in
the opening record or add a two-child uplift.

**F8. Observation. Every worker child now spawns `nvidia-smi`.** Evidence: `provenance()` (`provenance.py:83-
90`) calls `environment_identity()`, so each of the roughly 900 confirmatory children runs the 20 s-timeout
query. Reasoning: ordinarily about 0.1 s, charged to the child's wall; worst case 20 s recorded as absent. The
confirmatory's kernels are the pilot's measurement, which predates this call. Confidence: medium (not timed).
Action: none required; note it beside F7.

**F9. Observation. The published N0 losses are now the ones that decided.** Evidence: `calibrate.py:110-131`
`NoOpChoice` carries the two `cached()` losses; `noop_action` publishes them (`worker.py:583-584`) instead of
recomputing through `features_from_images`. Both paths are the same 128-row no-grad loop
(`host.py:335-352`, `m0/data.py:30-46`), so the values are the pilot's method bitwise; the decision itself is
unchanged. Pinned by `test_m0_modules.py:497-504` (label equals the strict `<`) and the worker tests. Confidence:
high.

**F10. Observation. Items 6 and 7 are receipts only.** Evidence: `read_once` (`confirmatory.py:490+`) reads
neither `corner` nor `projected_half_width`; `stats.ArmResult.corner` defaults to `None` (`stats.py:144-145`);
`projected_half_width` (`stats.py:419-425`) builds a fresh generator per call so `achieved_half_width` is
unchanged (pinned at `tests/test_m0_stats.py:447-455`, n = len equality); every `edge_of_grid` line in the diff
is context, not a change. Confidence: high.

**F11. Observation. Identity and admission.** `PROTOCOL_SHA` recomposes to `bde550df…` with the seventh
document (`identity.py:31-34`, pinned at `tests/test_m0_identity.py:93-100`); `pilot_protocol`
(`confirmatory.py:212-231`) admits exactly one prior (version, sha) pair, `m0-v4.2`/`a2c1e55d…`, and refuses
any other (pinned by `test_admits_the_v42_pilot_of_record_and_publishes_its_identity`); the pilot's opening and
admission files are read through `evidenced` (hash-listed, git-tracked), not `receipt()`, so no protocol check
applies to them, which is the intended asymmetry. `whitelist` (`confirmatory.py:131-143`) admits `noop` and
does not constrain lineage; `receipt()` (`receipts.py:78-93`) validates action, protocol, source, seed and
lineage, and the happy-path test pins `--lineage anchor` for `noop-3000-gpu*`. Confidence: high.

## Coverage table

| Item | Implemented as stated | Evidence | Test pinning it |
|---|---|---|---|
| 1 Environment identity | Yes at 13989d9 (partly at 712b09a: no comparison) | `provenance.py:21-27` pins unchanged, refused at `:65-67` before `claim_attempt`; `:69-77` python/driver/cudnn/gpu_names; `confirmatory.py:268-270, 333-342, 626` | `test_pinned_fields_refuse_a_mismatch_and_recorded_fields_never_do` (catches a pin change; `set(PINNED_ENVIRONMENT) == set(PINNED)` in the second test catches a recorded field entering the refusal loop); `test_an_unavailable_driver_query_or_cudnn_is_recorded_as_absent` (catches raising on OSError/None); `test_the_pilots_opening_record_must_be_hash_listed…`; happy-path `environment_vs_pilot` equality (`test_m0_pilot.py:1716-1724`; catches a refusal or a dropped field). Gaps: F4 |
| 2 No-op receipts | Partly (static lacks the hash, F5) | `worker.py:430-445` (`_noop_rule`, T-cool via `:463`), `:559-590` | `test_m0_worker.py:398-403` (tcool: hash equals `tensor_sha(SEEN)`, label equals `<`), `:451-458` (noop: same); `test_m0_modules.py:497-504`. Catches a dropped field, an inconsistent choice; not a hash of the wrong tensor on production indices (F3) |
| 3 Cross-card no-op replay | Yes | `confirmatory.py:348-369` both children from `run.path(anchor_host(3000))` = `replay-3000-gpu0`, gpu 0 and 1, `InstrumentHold` on any of three keys; `stage_v0:376` before `train-*`; `m0_stop_report` maps `InstrumentHold` to `instrument-failed` (`pilot.py:236-245`) | `test_a_no_op_replay_that_differs_across_cards_is_an_instrument_failure` (catches: dropped hold, wrong error class, replay moved after training, `noop_sha256` left out of the keys); happy path `:1686-1700` (catches: wrong card, wrong training root, `compared` set shrunk, flag missing). Residual single sources: each fresh anchor's own N0, T-cool and static recalibrations, as the amendment scopes |
| 4 Terminal records | Yes | `pilot.py:1310-1327`, `confirmatory.py:660-699` | Clean path only (`:457-463`, `:1710-1716`); stop path unpinned (F1) |
| 5 Child environment and exit | Yes; `src/sultai_screen/pilot.py` untouched (`git diff --stat main..13989d9` over the seven Gate 3 files plus `data.py`, `contracts.py` is empty) | `provenance.py:29-38, 83-90`; `m0/pilot.py:360-381` | `ChildEnvironmentTests` (catches a changed whitelist, absent not null); exit records clean path only (F2) |
| 6 Projected half-widths | Yes, receipt only | `pilot.py:1135-1157`; `stats.py:400-425` | `test_projected_half_width_resamples_n_anchors_and_narrows_with_n`; pilot happy path `:535-542` |
| 7 Grid corners | Yes, receipt only; `edge_of_grid` untouched | `worker.py:218-235, 706-708, 1083`; `receipts.py:166-171`; `pilot.py:837-854` | `test_m0_worker.py:525-531`; `test_m0_pilot.py:604-610` |

## What I did not check

- No GPU work, no live `cudnn.version()` or `nvidia-smi` probe, no child launched; the "no CUDA context" claim
  rests on the torch Python source and the documented `cudnnGetVersion` binding, plus the `check_cards` guard.
- CIFAR batch files not opened; the worktree's uncommitted state was not used (tests ran on a `git archive`
  export of 13989d9; at the time of review the worktree showed `confirmatory.py` and `test_m0_pilot.py`
  modified, matching 9680888).
- Test modules outside the m0 set (`test_screen`, `test_v42_ladder`, `test_protocol_identity`, and the
  non-screen suites) were not run; the A5.1 booking arithmetic (`workloads_of` uplift, `refit-bound`) and the
  source-change disposition were read only for receipt consistency, not for correctness of the sums.
- The publication/manifest step that lists result files (the predecessor's 449-file check) is not in this diff;
  whether it will include `*.exit.json` was not verified.
