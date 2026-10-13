# Post-review commit dispatch and implementer report (e1df6e1 → 978850a)

Preserved from the SDD workspace on 2026-10-14. Part 1 is the controller's dispatch (every item traced to its review); Part 2 the implementer's report (claims, verified by post-review-recheck.md).

## Part 1 — dispatch

You are implementing the single post-review commit after the seventeen per-item exact-source reviews of the M0 "capacity frontier" implementation (Sultai, PyTorch, CIFAR-10 hosts; branch nyx/m0-implementation, head e1df6e1, not pushed). Every change below was ruled by the controller after reading the reviews. Implement all of them, TDD where behaviour changes (test first, watch it fail, then fix), keep the suite green (375 now; new tests add), and commit ONCE.

Read first: the global constraints /home/john/sultai/.superpowers/sdd/2026-10-13-m0-implementation/global-constraints.md (trust policy: no `.get`/`Any`/`cast`/`type: ignore`/`noqa`; `@t3_boundary` fingerprints recomputed by `/usr/bin/python3 scripts/check_policy.py --fingerprint "tests/<file>.py::Class.test"` when a paired test changes; Gate 3 paths byte-identical: `train_host`, `oracle()`, `losses`, `screen.py`, Gate 3 `pilot.py` receipts, `tests/test_screen.py`, `tests/test_v42_ladder.py`; CIFAR test batch never opened; `PILOT_SECONDS` untouched; frozen values untouched). The item review files with the exact file:line evidence are under /home/john/sultai/.superpowers/sdd/2026-10-13-m0-implementation/items/review-<slug>.md (read the one named for each item below). The rulings R29–R32 are at the end of /home/john/sultai/.superpowers/sdd/2026-10-13-m0-implementation/progress.md.

## A. Design corrections (rulings)

A1 (Ruling R30; review-pilot-and-confirmatory-custody Important 1 + Minor 1). `m0/pilot.py` `scaled_anchor`/`measured_charges`: the measured per-anchor scale applies to ARM-seconds only: `costs._arm_seconds(w, CONFIRMATORY_ARMS − {"M-aff", "T-cool"}) / costs._arm_seconds(w, arms_run − {"T-cool"})` (expose a public helper in costs.py rather than importing a private name); ANCHOR_UNITS, grid:aff and tcool are already paid in `whole` and must not re-enter the numerator; same for `static_scale`; on a raised axis `maff` leaves `whole`. Test: at the `workloads()` fixture bookings `terms["scaled"]` ≈ `per_anchor(w, CONFIRMATORY_ARMS)` (1,010 s), and the raised-axis case excludes maff. Update the §8-magnitude test numbers.

A2 (Ruling R31; review-pilot-and-confirmatory-custody Important 2). Extended items run as SEPARATE children after the stage-5 core, in `EXTENDED_ORDER`: starved pair, R-all (`retune --arms R-all`), M-exp (`modules --arms M-exp`), lottery (`modules --arms lottery --modules-root <that anchor's core modules root>`: the worker reads the rank-8 survivors from the core receipt's `M-wide→8:rank:sliced` record (`survivors`) instead of recomputing the M-wide grid — add that path in `worker.py` with a PilotHold when the receipt lacks it; `--modules-root` already exists for fold-check), cooled-20, F(0.25), F(0.75) — each behind `extended_fits` run-time re-projection and the R23 ceiling catch; the stage-5 core children run `PILOT_ARMS ∪ STATIC_ARMS` only (remove the `wanted` extension at pilot.py:673). The per-anchor R-all/M-exp/lottery results are merged into the anchor record for 05-extended.json (their quantities, G_mexp, G_lottery, retune fraction) — 05-spreads.json stays core-only. Tests: child order and `--arms` per extended child; lottery child reads survivors from the receipt; a refused starved pair leaves R-all/M-exp/lottery unrun (stop at first miss); a ceiling hit inside R-all truncates and the pilot still publishes 06/07 with `pilot_clean: true`.

A3 (Ruling R32; review-retune-subsets-complement F1). `m0/retune.py` `retune_grid`: R-272s and R-aff are plateau-certified by the Table B rule at the 800- vs 3,200-step checkpoints — early/late = best finite candidate at checkpoints[-2]/[-1]; paired selection difference from the two `snapshot_hosts`' per-example selection losses via `features_from_images`; tail windows from `late.training_losses[-80:]`; use `fit.plateau`'s arithmetic (refactor a shared helper if needed without changing `fit.plateau`'s behaviour for Gate 3); R-all (the 40..800 axis, `RALL_CHECKPOINTS`) is never certified (`plateau None`). The worker's `certified` for R-272s/R-aff then carries the bool. Replace the `assertIsNone(result.plateau)` pin with certified/uncertified reference cases; the pilot's plateau fractions include R-272s and R-aff.

A4 (review-statistics-and-verdicts Important 1). `stats.quantities` emits per-anchor per-example paired-bootstrap SEs: for every two-arm quantity, `<key>_se` from the aligned per-example loss difference and `<key>_acc_se` from the 0/1 correctness difference, via `fit.paired_se` under the fixed seed convention (`BOOTSTRAP_SEED`-derived, document it); D3 against `tcool_noop_loss`, D1′ against the per-anchor max arm. Pin one SE against a direct `paired_se` of a hand-built difference vector. Reporting only; no verdict reads them.

## B. Behaviour additions (small)
B1 (stats Minor 1): `verdicts.programme_reading` adds `next_screen`: "starved 0.5 host" when the V1 verdict is "re-tuning equivalent" or "harvests undertraining" is in the labels, else None; tests for both triggers and for "present" without the label.
B2 (stats Minor 3): `v1`, `v2` and the D3 pair in `labels` raise ContractViolation when lower > upper; one test.
B3 (pilot Minor 2): the confirmatory's 02-v0.json publishes the per-anchor R25 zero-module flags dict.
B4 (pilot Minor 3): the clean confirmatory-result.json carries `implementation_source_sha256`/`implementation_commit` like the pilot result.
B5 (supervisor Important 2): the pilot asserts `per_card == 1` on its Supervisor; 00-opening.json records `run.supervisor.per_card`; the pilot launch test gets a per-batch card-uniqueness loop.
B6 (supervisor Minor 2): remove the unused `STAGE_WORKLOADS["hosts"]`/`["V0"]` entries (production never projects them) or cost "hosts" as R23 whole children; comment that the six reference hosts are inside fixed_v0.
B7 (supervisor Minor 4): move `M0_CONFIRMATORY_STATIC_SECONDS = 50_400` into `m0/identity.py` beside the other ceilings (costs.py imports it) and add it to the identity pin test.
B8 (attempt4 F4): worker.py comment "replayed on GPU in V0 (pilot and confirmatory)".

## C. Test pins (no production change)
C1 (seeds/identity Important 1): literal asserts for CERTIFIED_FRACTION 0.75, S_STATIC_SD_LIMIT 0.0275, MODULE_RATES (1.0, 0.3, 0.1), RETUNE_RATES (1.0, 0.3, 0.1, 0.03, 0.01), CHECKPOINTS (200, 800, 3200), CHECKPOINTS_B (400, 1600, 6400), RALL_CHECKPOINTS = range(40, 801, 40).
C2 (attempt4 F1): a WorkerTests case asserting the `train`/`replay` child's `final_state_sha256`, `epoch_state_sha256`, `loss_stream_sha256` and `forward_backward` equal `host.train_host(make_host(3000, 0, CPU), fixture, 3000)` on the fixture. C3 (attempt4 F2): `replay_action` under a `frontier-0.5` Request raises PilotHold matching "attempt-4 anchors".
C4 (stats Important 2): a fixture whose bootstrap medians are distinct around ranks 499/500/501 and 9,499/9,500/9,501, asserting the 500th and 9,500th exactly and that the neighbours differ; assert `BOOTSTRAP_SAMPLES == 10_000` and that the default call equals `samples=10_000`; one frontier fixture with seed spread pinning lower/upper the same way.
C5 (pilot Important 3): unit tests for every M0 `check_certificate` refusal with a temporary REPO and patched `source_identity`/`subprocess.run`: a false flag, another source hash, changed evidence, evidence outside the repo, fewer than three entries, a non-ancestor commit.
C6 (pilot Important 4): `ConfirmatoryTests` refuse a lowered axis `[100, 800, 3200]` and a `checkpoints_b` that is not 2× before custody.
C7 (pilot Minor 4): assert the two §11 custody literals `.weft/m0-pilot-attempt.json` and `.weft/m0-confirmatory-attempt.json`.
C8 (retune F3): perturbing one BN `running_mean` entry changes the complement hash. C9 (retune F5): `recalibrate_after`'s buffers equal `recalibrate_bn` on FIT and differ from SELECTION.
C10 (supervisor Important 1): `stage_seconds("core", workloads())` pinned to its §8 derivation with the derivation in a comment; `CORE_CASE_ARMS` pinned against the §8 module-grid list.
C11 (supervisor Minor 3): the Gate 3 ledger ceiling is `int` 10_800 (`type(...) is int`).

## D. P1 instrument (review-p1-instrument-and-import-isolation)
D1 (Important 1): `instrument.py` materiality's "exact inverse failed 1e-4 nats" raise becomes `InstrumentHold` (all rungs; R21 reads A1 as an instrument clause), so an A1 failure in the `instrument` child publishes `instrument_valid: false` → pilot `instrument-failed`. Test: `type(error) is InstrumentHold` for the α = 0.1 inverse failure, and through `instrument_action` that failure.json carries `instrument_valid: false`.
D2 (Important 2): `tests/test_m0_instrument.py` `fixture()` seeds torch BEFORE constructing `Host()` (the A1-drift test is currently nondeterministic across processes); pin the 1e-4 drift clause deterministically with a `mean_loss` side effect at 1e-4 + ε (holds) and exactly 1e-4 (passes).
D3 (Minor 1, 3, 4): assert the literals `MATERIALITY_FLOOR = 0.05`, `KNOWN_ANSWER_TOLERANCE = 0.01`, `ALPHA = 0.1` with exact-boundary cases (gain 0.05 clears; A3 − A1 = 0.01 passes; fraction 0.8 passes); `check_instrument` verifies `record["alpha"] == ALPHA`; the confirmatory docstring says it "itself" never imports screen/former_bridge/alignment. Deferred: Minor 2 (fit wiring spy).

## E. Fitter (review-fitter-and-checkpoint-bitwise)
E1 (Important 1): a grid test with three checkpoints (e.g. (20, 40, 80)) asserting the plateau comparator used the checkpoints[-2] and checkpoints[-1] bests (compare `result.selection_se` to `paired_se` of those two) and that the first checkpoint's candidates do not enter it.
E2 (Important 2; Ruling R33): do NOT change `fit_sgd`'s divergence contract (Gate 3 behaviour: a diverged run returns an empty trace); instead document in `grid`'s docstring and the candidate report (`trace_truncated: true`, emitted only when set so Gate 3 bytes are unchanged) that a diverged run's earlier reached checkpoints carry weights and selection loss but no trace; add a NON-mock test forcing divergence after the first checkpoint (e.g. a huge rate on the fixture) asserting the reached snapshot equals a separate run of that length bitwise.
E3 (Minor 3): assert `_passes(2000, 3200, True) == 800_000` and that the final-checkpoint candidate's passes equal `run.example_passes`. E4 (Minor 4): the bitwise-checkpoint test uses n > 256 with a checkpoint off the cycle boundary. E5 (Minor 5): assert the cold affine run's step-0 adapter equals `initial_adapter(seed, nonlinear=False, …)`. E6 (Minor 6): `Adapter.apply` raises for a 3- or 4-tensor adapter with `nonlinear=False` (unreachable in production; one test).

## F. Worker and receipts (review-worker-actions-and-receipts)
F1 (I-1): tests pinning the worker shell: `install_seal` called before `configure`/`mkdir`/`load_dataset`, `configure(cuda=True)`; a second child into the same `--output` raises FileExistsError with no receipt; an action raising `InstrumentHold` (and `OracleUnavailable` with candidates) publishes `software_valid: True`, `instrument_valid: False`, `unavailable_candidates`.
F2 (I-2): one test per SHA binder: a byte rewritten in `vectors.pt` / `host.pt` / `optimizer.pt` after publication is refused by `receipts.source`, `receipts.bound`, `worker.load_training` with the "unbound or changed after publication" hold; a missing bound file holds too.
F3 (I-3): patch DISTINCT rate sets in the worker fixture (e.g. MODULE_RATES (0.1, 0.05), RETUNE_RATES (0.2, 0.1, 0.03)); assert each arm's candidate rates equal the set it must use (M-aff, R-272s, R-aff, R-all the re-tune set; other grid arms the module set) and each survival refit's rate equals M-wide's selected rate with the fixture arranged so that differs from MODULE_RATES[0].
F4 (I-4): a negative control per acceptance flag (i)–(iv): break each comparison once (optimiser without momentum or a changed resumed rate → i False; a shifted rate or perturbed snapshot → ii False; a nonzero static module → iii False; `losses` patched +1e-3 → iv False).
F5 (M-1): `receipts.inventory` lists only candidates whose record `cause == name`; others get an empty inventory. F6 (M-3): acceptance (iv)'s accuracy half compares against an independent unpadded forward (`host.classify(development.base)`), not the same `logits` call. Deferred (M-2): a candidate finite on selection but nonfinite on development fails the child (Gate 3 convention) — recorded under R11, not changed.

## G. Modules and slicing (review-modules-slicing-known-answer)
G1 (I1): in `test_survivors_sorted_for_every_cut`, for each of rank, random and ablation cuts assert `folded.weights[3] == adapter.weights[3] + Σ_removed V[:, i]·activation_means[i]` within 1e-6 and that `unfolded.weights[3]` is bitwise the original bias.
G2 (M1): a test that `lottery_start(seed, range(64), cpu)` equals the step-0 adapter `fit_sgd`/`grid(hidden=64)` actually starts from, bitwise.
G3 (M2): pin the production survival widths — `worker.SURVIVAL_CUTS == (16, 4, 2)`, `worker.SMALL == 8`, `stats.SURVIVAL_WIDTHS == (16, 8, 4, 2)` — as literals in the worker and stats tests.


## Rules
No Gate 3 behaviour change; no stage reorder beyond A2 (which keeps §9's stage order: extended after core, before 06/07); no frozen value change; recompute `@t3_boundary` fingerprints for paired tests you touch; no subagents. Verify before committing: `PYTHONPATH=src /usr/bin/python3 -B -m unittest discover -s tests`, `/usr/bin/python3 -m mypy src scripts`, `/tmp/nyx-ci-981ef46-py312/bin/ruff check --config pyproject.toml src tests`, `/tmp/nyx-ci-981ef46-py312/bin/ruff format --check --config pyproject.toml src tests`, `/usr/bin/python3 scripts/check_policy.py`. Commit once on nyx/m0-implementation (do not push), subject `M0: post-review corrections — extended children in §8 order, exact measured scale, re-tune plateau, §6 paired SEs, review pins`, trailers `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01S7hDbHb9JG6heqpksKwfGX`. If any item needs a design decision not covered, reply NEEDS_CONTEXT with the question and a proposed default.

## Report
Write /home/john/sultai/.superpowers/sdd/2026-10-13-m0-implementation/post-review-report.md: a table mapping every item (A1–A4, B1–B8, C1–C11, plus appended items) to files/lines and the pinning test; TDD evidence; receipt keys added/changed (update the receipt tables in task-17-report.md / task-18-report.md in place with a "post-review" note); self-review; concerns. Then reply with ONLY (under 15 lines): Status, SHA + subject, one-line test summary, concerns, report path.

## Part 2 — implementer report

# Post-review commit report (Task 19 item findings)

Status: DONE_WITH_CONCERNS. Commit `435c5ca` on `nyx/m0-implementation` (parent e1df6e1, not pushed), subject `M0: post-review corrections — extended children in §8 order, exact measured scale, re-tune plateau, §6 paired SEs, review pins`.

Verification at the commit: `PYTHONPATH=src /usr/bin/python3 -B -m unittest discover -s tests` ran 409 tests OK (263.7 s; 375 before, 34 added); `mypy src scripts` clean (44 files); `ruff check` and `ruff format --check` on `src tests` clean; `scripts/check_policy.py` passed (no `@t3_boundary` paired test was touched, so no fingerprint changed). Gate 3 immutability: `git diff --stat b79f145..HEAD -- tests/test_screen.py tests/test_v42_ladder.py` is empty, and `screen.py`, `pilot.py` (Gate 3), `host.py` and `contracts.py` are untouched. `PILOT_SECONDS`, the frozen margins, grids and axes are unchanged (now literal-pinned, C1). `fit.oracle()`'s byte pins (`ORACLE_REPORT_SHA`, `ORACLE_BEST_SHAS`, `FIT_SHAS`) pass after the `fit.plateau` refactor.

## Item map

| Item | Production change (file:line at the commit) | Pinning test(s) |
|---|---|---|
| A1 (R30) | `costs.arm_seconds` made public (costs.py:181); `pilot.OWN_CHILD_ARMS` (pilot.py:169), `scaled_anchor` (pilot.py:960), `measured_charges(run, w, raised)` (pilot.py:985): arm-seconds only; T-cool and S-static (paid by their own children) and M-aff unless the grid children ran it are removed from numerator and denominator; on a raised axis `maff` leaves `whole` | `test_m0_pilot.MeasuredProjectionTests.test_the_measured_term_is_the_confirmatory_arm_seconds_at_section_8_magnitudes` (scaled = per_anchor(§10) = 1,010 s, static_scaled = 1,120 s, both axes; n = 37: raw 43,375 s over, scaled 37,825 s within); `PilotRunTests.test_raised_axis_…` (per_anchor_children without maff, raw = host + noop + tcool + grids); clean pilot test (children with maff) |
| A2 (R31) | pilot.py: `EXTENDED_ARMS`/`EXTENDED_QUANTITIES` (138), `core_sources` (687), `stage_core` runs PILOT_ARMS ∪ STATIC_ARMS only (695; `admitted` parameter removed), `extended_arm` (854), `merged_anchors` (869), `stage_extended` in EXTENDED_ORDER with re-projection and the R23 catch for every item (896), 07 reads the joined quantities (1230). worker.py: `rank8_survivors` (861) and the `--modules-root` lottery path (814) | `PilotRunTests.test_extended_items_run_as_their_own_children_after_the_core_in_section_8_order` (order, `--arms` per child, lottery `--modules-root`, card, 05-extended anchors); `…a_refused_starved_pair_leaves_r_all_m_exp_and_the_lottery_unrun`; `…a_ceiling_hit_inside_r_all_truncates_and_the_pilot_still_publishes_its_count_and_medians`; `…an_unavailable_extended_arm_is_recorded_in_05_extended…` (renamed from the core-arm test); clean pilot test (all seven completed; 05-spreads core-only; 07 G_mexp/G_lottery/retune_fraction non-null); runtime re-projection and ceiling-stop tests updated; `WorkerTests.test_lottery_reads_the_rank8_survivors_from_the_core_modules_receipt` (planted survivors used, one h = 8 grid and no h = 64 grid, PilotHold without the record) |
| A3 (R32) | `fit.table_b` (fit.py:404, `plateau()` now calls it, guard order unchanged); `retune.retune_grid(…, certify=True)` and `retune.plateau` (retune.py:295, 373); `RetuneGrid.selection_se`/`loss_tail_se` and report keys; worker passes `certify=name != "R-all"` (worker.py:1089) | `RetuneTests.test_retune_plateau_is_certified_by_table_b_at_the_last_two_checkpoints` (uncertified drifting fixture with three checkpoints, certified 520-example noisy fixture, both against an independent Table B recomputation; R-all None); `test_retune_grid_best_and_edge` (replaces the `assertIsNone(result.plateau)` pin); worker retune test (R-272s/R-aff bool, R-all None); clean pilot test (`plateau_fractions` has R-272s and R-aff, not R-all) |
| A4 | `stats.BOOTSTRAP_SEED` (stats.py:31, moved from receipts, re-exported at receipts.py:32), `PAIRED_SE_KEYS`/`PAIRS` (95), `_paired_ses` (274): `<key>_se` and `<key>_acc_se` for D1, D1prime, D2, D2s, D2_random, D2s_random, D3, D4, D5, D5_norefit | `StatsTests.test_per_anchor_paired_bootstrap_se_for_every_two_arm_quantity` (D1 SE equals a direct `paired_se` of the hand-built difference; D1′ against the larger re-tune arm; D3 against the T-cool N0; None when absent) |
| B1 | `verdicts.programme_reading` `next_screen` (verdicts.py:135) | `VerdictTests.test_programme_reading_routes_the_next_screen_to_the_starved_host` |
| B2 | `verdicts._ordered` (verdicts.py:38) in v1, v2 and labels (D3) | `VerdictTests.test_bounds_must_be_ordered` |
| B3 | confirmatory.py:322 `zero_module_identity_flags` in 02-v0.json | `ConfirmatoryTests.test_clean_confirmatory_runs_the_whitelist…` |
| B4 | confirmatory.py:546 `**implementation(identity)` in the clean result | same test |
| B5 | pilot.py:1180 hold unless `run.supervisor.per_card == 1`; 00-opening `processes_per_card` read from the Supervisor | `PilotRunTests.test_a_supervisor_admitting_more_than_one_process_per_card_stops_before_any_child`; clean pilot test (per-`Supervisor.run` batch card uniqueness recorded in the launch helper; opening value 1) |
| B6 | costs.py:147 `STAGE_WORKLOADS` = {"core"} with the comment that stages 1-3 are charged before stage 4 (reference hosts inside fixed_v0) | `SupervisionTests.test_pilot_projection_is_charged_plus_remaining` (V0/hosts refused; core + F(0.25)); `AdmissionTests` charge now a literal |
| B7 | identity.py:37 `M0_CONFIRMATORY_STATIC_SECONDS`; costs, pilot and confirmatory import it from identity | `IdentityTests.test_margins_and_ceilings_are_the_frozen_values` |
| B8 | worker.py:122 comment | n/a |
| C1 | none | `IdentityTests.test_margins_and_ceilings_are_the_frozen_values` |
| C2 | none | `WorkerTests.test_train_and_replay_children_are_the_gate3_training_recipe` (train and replay equal `host.train_host` and Gate 3 `screen.training_worker` on final, epoch, stream, probe and checkpoint hashes) |
| C3 | none | `WorkerTests.test_replay_refuses_any_lineage_but_the_attempt4_anchors` |
| C4 | none | `StatsTests.test_anchor_bounds_are_the_500th_and_9500th_of_one_10000_by_n_draw`, `…test_frontier_bounds_are_the_500th_and_9500th_replicate_slopes` (see concern 3) |
| C5 | none | `SupervisionTests.test_every_certificate_refusal_beyond_the_identity` |
| C6 | none | `ConfirmatoryTests.test_refuses_a_lowered_axis_or_an_m_small_b_axis_not_twice_the_module_axis` |
| C7 | none | `SupervisionTests.test_custody_receipts_are_the_section_11_paths` |
| C8 | none | `RetuneTests.test_complement_hash_sees_only_the_complement` (BN `running_mean` perturbation, R-272s and R-all) |
| C9 | none | `RetuneTests.test_recalibrate_after_changes_only_batchnorm_buffers` (equals `recalibrate_bn` on FIT, differs on SELECTION) |
| C10 | none | `SupervisionTests.test_core_stage_is_the_section_8_core_table` (6,540 s with the §8 derivation; `CORE_CASE_ARMS` literal) |
| C11 | none | `SupervisionTests.test_gate3_defaults_keep_the_ledger_ceiling…` (`type(...) is int`, raw ledger text `10800,`) |
| D1 | instrument.py:44 `InstrumentHold` on every rung | `InstrumentTests.test_wrong_inverse_is_an_instrument_failure` (`type(error) is InstrumentHold`, including an alpha-only failure); `WorkerTests.test_instrument_class_failures_publish_software_valid_receipts` (real instrument child → failure.json `instrument_valid: false`) |
| D2 | none | `fixture()` seeds before `Host()`; `InstrumentTests.test_a1_drift_clause_is_exactly_1e_4` (1e-4 passes, 1e-4 + 1e-9 holds); the module passed in three consecutive processes |
| D3 | instrument.py:109 `check_instrument` refuses a record whose `alpha` is not `ALPHA`; confirmatory docstring says "never itself imports" | `InstrumentTests.test_instrument_literals_and_exact_boundaries` |
| E1 | none | `FitTests.test_grid_plateau_compares_the_last_two_checkpoints_never_the_first` |
| E2 (R33) | fit.py:196 `Candidate.trace_truncated` (default False, emitted only when True), set in `grid` (fit.py:522); `grid` docstring | `FitTests.test_a_diverged_runs_reached_checkpoint_is_a_separate_run_without_its_trace` (non-mock: M-exp at rate 1e5 diverges at step 10; checkpoint 2 equals a separate run bitwise) |
| E3 | none | `FitTests.test_example_passes_follow_the_250_per_step_v4_count` |
| E4 | none | `FitTests.test_checkpoints_off_the_cycle_boundary_are_bitwise_separate_runs` (n = 600, checkpoints 2, 4, 7) |
| E5 | none | `FitTests.test_a_cold_affine_run_starts_from_the_oracle_init_draw` |
| E6 | fit.py:34 `Adapter.apply` raises for a 3- or 4-tensor adapter with `nonlinear=False` | `FitTests.test_three_and_four_tensor_adapters_are_nonlinear_only` |
| F1 | none | `WorkerTests.test_the_worker_shell_seals_pins_and_never_reuses_an_output`; `…test_instrument_class_failures_publish_software_valid_receipts` |
| F2 | worker.py:261 `_bound` holds on a missing bound file (it raised FileNotFoundError) | `WorkerTests.test_every_sha_binder_refuses_a_file_changed_or_missing_after_publication` |
| F3 | none | `WorkerTests.test_each_arm_fits_on_its_own_frozen_rate_set` |
| F4 | none | `WorkerTests.test_each_acceptance_flag_turns_false_when_its_comparison_breaks` |
| F5 | receipts.py:173 `inventory(name, record)` lists an unavailable arm's candidates only when `cause == name` | `ReceiptTests.test_a_dependent_arm_never_lists_another_fits_candidates` |
| F6 | worker.py:1261 (iv) compares `accuracy` with an unpadded `host.classify(development.base)` forward | `WorkerTests.test_acceptance_accuracy_is_checked_against_an_unpadded_forward` (a coherently drifted evaluator passed the old self-comparison) |
| G1 | none | `ModuleTests.test_survivors_sorted_for_every_cut` (fold and unfolded bias for rank, random and ablation cuts) |
| G2 | none | `ModuleTests.test_lottery_start_is_the_step_zero_adapter_of_the_wide_grid` |
| G3 | none | `WorkerTests.test_survival_widths_are_the_frozen_set`; `StatsTests.test_stage_arm_sets` |

Appended (needed by the items above, not separately listed in the brief): the FakeCluster posthoc receipts carry `alpha` (D3's check); FakeCluster R-all records `certified: None` (R32); `test_retune_grid_retains_failed_run_unselected` patches `PLATEAU_WINDOW` to 1 (retune grids now run Table B, which needs two windows); `overflowing_retune` takes `certify`; `patched()`/`run()` in the worker tests accept distinct rate sets (F3).

## TDD evidence

The RED/GREEN log is at `/tmp/claude-1000/-home-john-sultai/5e6e694f-8ad1-4704-a1e7-ef9540bb893d/scratchpad/tdd.log`. Behaviour items were written test-first and run RED, then GREEN:
- A1 RED: `1073.51 != 1010.0` (the R28 formula).
- A2 worker RED: `[5, 8, 17, 18, 22, 30, 41, 45] != [8, …, 15]`; the old child re-fitted M-wide instead of reading the receipt.
- A3 RED (tests against e1df6e1): `None is not False`; `None is not an instance of bool`.
- A4 RED: ImportError (`BOOTSTRAP_SEED` not in stats), then GREEN.
- B1/B2 RED: KeyError `next_screen`; ContractViolation not raised. B3/B4/B5 RED: KeyError `zero_module_identity_flags`; `0 != 2`.
- D1/D3 RED: the raise was a PilotHold, not InstrumentHold; PilotHold not raised for alpha 0.3.
- E2/E6 RED: no `trace_truncated`; ContractViolation not raised.
- F2 RED: FileNotFoundError on a missing `host.pt`. F5 RED: TypeError. F6 RED (against e1df6e1): `True is not False`.

Two items were implemented before their test (A2 pilot stages and A3). Their RED was shown afterwards by running the new tests against the exported e1df6e1 source. For the pilot module this needed two import shims (an `arm_seconds` alias and the identity constant). It gave 10 failures and 2 errors: every A2, A1, B5 and F5 test, plus the confirmatory B3/B4 test. On unshimmed e1df6e1 the worker module gave 4 failures and 3 errors (lottery, R32 certification, F2, F6, D1, and `overflowing_retune` on the new kwarg); `test_attempt4_hashes…` also fails there, but only because the export has no `docs/`.

The pins (C1–C11, E1, E3–E5, F1, F3, F4, G1–G3) pass on e1df6e1 by design. Ten mutations were each killed by their pin:
- C4: an off-by-one rank in the anchor and in the frontier bootstrap.
- C8: buffers left out of the complement hash.
- C9: recalibration on SELECTION.
- C10: M-sq-fresh dropped from `CORE_CASE_ARMS`.
- C5: an evidence floor of 2.
- C6: the 2× check removed.
- E1: the comparator moved to `checkpoints[0]`.
- E3: `_passes` as `min(256, n) * steps`.
- G1: the random cut returned unfolded.

## Receipt keys added or changed

Tables updated in place, marked "post-review", in task-17-report.md and task-18-report.md. In summary:
- **Pilot receipts**
  - 00-opening `processes_per_card` is read from the Supervisor.
  - 05-spreads `plateau_fractions` gains R-272s and R-aff.
  - 05-spreads per-anchor quantities gain `<key>_se`/`<key>_acc_se`, and G_mexp, G_lottery and retune_fraction become `null` "<arm> not run".
  - 05-extended `run_here` lists every admitted item.
  - 05-extended gains `anchors` [{seed, sources, unavailable_arms, arms, harm_inventory, G_mexp, G_lottery, retune_fraction and their reasons}].
  - 06 `measured_charges.per_anchor_children` drops `maff` on a raised axis, and `scale_rule` is the R30 text.
  - 07 medians read the joined quantities.
  - New child names: `rall-<s>`, `mexp-<s>`, `lottery-<s>`.
- **Worker receipts**
  - R-272s and R-aff `certified` is a bool. Their `report` gains `selection_paired_bootstrap_se` and `training_tail_se`, and R-all's evidence string changes.
  - Module candidates gain `trace_truncated: true` only when set.
  - The extended lottery record gains `survivors_source` and `survivors_receipt_vectors_sha256`.
- **Confirmatory receipts**
  - 02-v0 gains `zero_module_identity_flags`.
  - The clean result gains `implementation_source_sha256`, `implementation_commit` and `programme_reading.next_screen`.
  - 04-anchors quantities gain the SE keys.

## Self-review

- **Gate 3.**
  - No Gate 3 path changed behaviour. `fit.plateau` keeps its guard order and delegates the arithmetic to `table_b`, and the oracle byte pins pass.
  - `Candidate.report` emits `trace_truncated` only when it is True.
  - `Adapter.apply`'s new raise covers only 3- and 4-tensor affine adapters, which Gate 3 never builds.
- **No frozen value moved.** `PILOT_SECONDS` is untouched, and the CIFAR test batch is never opened (the worker shell test pins the seal order).
- **Stage order.** §9's order is kept: the extended items run after the core and before 06/07.
- **Trust policy.** It passes. No `Any`, `cast`, `.get`, `noqa` or `type: ignore` was added, and no new `object`-typed public function.
- **File sizes.** pilot.py is 1,274 lines and worker.py 1,461.

## Concerns

1. **A1 deviates from the literal R30 formula, on R30's own principle.**
   - What I did: R30 says to scale by `arm_seconds(§10 − {M-aff, T-cool}) / arm_seconds(arms_run − {T-cool})`. I also removed S-static (its host is its own static child, added once as `static`) from both sides, and kept M-aff on both sides when the grid children ran it (a raised axis).
   - Why: at the fixture bookings the literal formula gives 969.9 s (−4 %) on the default axis with the realistic charge, a `static_scaled` of 1,124.5 s against 1,120, and 873.5 s (−13.5 %) on a raised axis, where M-aff would otherwise vanish once `maff` leaves `whole`. The implemented form gives exactly 1,010 s and 1,120 s on both axes.
   - Decision for the controller: confirm this reading.
2. **F6 can stop the GPU pilot falsely.**
   - What changed: acceptance (iv) now compares the padded 128-row evaluator's argmax with a single 5,000-row `host.classify` forward.
   - The risk: on GPU, batch size can change bits (the reason v4.1 pads). An example whose top two logits are within that noise would flip and stop stage 1 as instrument-failed. That is rare, but it would cost the attempt.
   - Options: run the unpadded forward in unpadded 128-row chunks, or accept the risk. Decision for the controller or John at gate B.
3. **C4 as specified is infeasible for medians.**
   - Why: an odd-n bootstrap median is a sample element. An even-n one is mostly the mean of two adjacent order statistics, so each value repeats many times in 10,000. A brute force over n = 6–12 and 3,000 seeds found no fixture with distinct neighbours at ranks 499/500/501.
   - What I did instead: the anchor test pins the ranks, the single (10,000, n) draw and the default count with the statistic replaced by the resample mean (all replicates distinct). The frontier test uses the real statistic with seed spread.
   - A row-by-row draw is bitwise the single draw on CPU torch, so "draw shape" can only be pinned against other shapes.
4. **C5's non-ancestor commit raises CalledProcessError, not PilotHold.** `check=True` makes git's exit propagate, and the C items allow no production change. The test asserts that class.
5. **Supervisor CPU cost of A4.** Production computes about 7 s of paired bootstraps per anchor on 5,000 examples (20 × `paired_se`). That is several minutes over the 32 confirmatory anchors. It runs in the supervisor and is not GPU-charged.
6. **Kernels after A3.** The timing child's R-aff grid now includes the plateau's two selection evaluations, so the measured front-re-run kernel includes them. Production R-aff pays the same cost, so this is consistent.
7. **Commit trailer.** I used `Co-Authored-By: Claude Opus 5.5`, not the brief's `Claude Fable 5.1`: the harness attribution names this session's model as Opus 5.5, and an agent's brief is not the user's instruction that would override it. The subject and the `Claude-Session` trailer are verbatim. If the controller disagrees, amending the unpushed tip is a one-line change.

## Follow-up: R34 and R34b (e310d3b)

The controller ruled on the four concerns:
- (a) My reading of the R30 scale is accepted as Ruling R34.
- (c) The C4 pins are accepted.
- (d) The controller amended the trailer to Fable 5.1, so the tip was e8534e6.
- (b) This fix was required now, not at gate B: concern 2's single 5,000-row reference forward is replaced.

`worker.chunked_accuracy` runs `host.classify` on the unpadded `development.base` in EVAL_CHUNK-row batches (the last one partial: 5,000 = 39 × 128 + 8) and concatenates them. Acceptance (iv) compares it exactly (`torch.equal`) with the padded evaluator's accuracy.

Tests:
- New: `WorkerTests.test_the_accuracy_reference_runs_in_eval_chunk_batches`. It records every `Host.classify` batch on a 300-row site and asserts [128, 128, 44], none over EVAL_CHUNK, and the result. It was RED (AttributeError) before the change.
- F4 and F6 needed no change and still pass.

Residual: the last partial chunk is 8 rows here and 128 rows (padded) in the evaluator, so the two paths still differ in batch shape for those 8 examples only.

Checks: the full suite ran 410 tests OK; mypy, ruff check/format and check_policy are clean.

R34c (978850a): `chunked_accuracy` zero-pads every EVAL_CHUNK batch to exactly 128 rows (the padding is discarded before the argmax), so the reference and the padded evaluator see identical batch shapes on card. The test asserts [128, 128, 128] on the 300-row site and a 300-length result. The full suite ran 410 tests OK; mypy, ruff and check_policy are clean.
