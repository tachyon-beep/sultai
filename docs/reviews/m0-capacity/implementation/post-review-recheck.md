# Post-review recheck: e1df6e1..978850a

**Verdict:** ACCEPTED WITH FINDINGS

Counts: Critical 0, Important 0, Minor 3 (all informational or cosmetic; none blocks gate B).

Range: e8534e6 (post-review corrections), e310d3b (R34b), 978850a (R34c). There are 20 files: 11 production and 9 test.

Method:
- Read only, on `nyx/m0-implementation` at 978850a.
- The diff file `review-e1df6e1..978850a.diff` (4,356 lines) was read in eleven passes:
  1. header and fit.py
  2. confirmatory, costs, identity, instrument
  3. pilot.py
  4. receipts, retune, stats, verdicts
  5. worker.py
  6. test_m0_pilot, first half
  7. test_m0_pilot, second half
  8. fit, identity, instrument and modules tests
  9. retune and stats tests
  10. supervision tests and the first half of the worker tests
  11. the second half of the worker tests
- Production context outside the hunks was read where a check needed it:
  - `receipts.anchor`
  - `stats.quantities`
  - `pilot.spreads`
  - the `retune_grid` candidate bank
  - `worker.Collector.attempt` and the in-process lottery path
- Focused tests run, all OK (5 tests, 17 s):
  - `test_m0_pilot.MeasuredProjectionTests`
  - `PilotRunTests.test_extended_items_run_as_their_own_children_after_the_core_in_section_8_order`
  - `PilotRunTests.test_a_ceiling_hit_inside_r_all_truncates_and_the_pilot_still_publishes_its_count_and_medians`
  - `RetuneTests.test_retune_plateau_is_certified_by_table_b_at_the_last_two_checkpoints`
  - `WorkerTests.test_the_accuracy_reference_runs_in_eval_chunk_batches`
- Lint and policy:
  - `scripts/check_policy.py` reports "Sultai trust policy passed".
  - `ruff check` reports "All checks passed!".

## Gate 3 immutability evidence (check 1)

- `git diff --stat e1df6e1..978850a -- src/sultai_screen/screen.py src/sultai_screen/pilot.py src/sultai_screen/contracts.py tests/test_screen.py tests/test_v42_ladder.py` is **empty**.
- `git diff e1df6e1..978850a -- src/sultai_screen/contracts.py src/sultai_screen/host.py src/sultai_screen/data.py` is 0 lines, so host.py is untouched.
- `PILOT_SECONDS = 3 * 3600` is unchanged (contracts.py:47) and pinned again in the identity test.
- Frozen values: identity.py gains only the moved `M0_CONFIRMATORY_STATIC_SECONDS = 50_400` (identity.py:37). X, X2, M_S, RHO, HARM, FOLD_TOLERANCE, CERTIFIED_FRACTION, S_STATIC_SD_LIMIT, the rate grids, CHECKPOINTS, CHECKPOINTS_B and RALL_CHECKPOINTS are byte-unchanged and are now literal-pinned (C1).

The fit.py changes are additive and byte-neutral for Gate 3 callers:
- **E2 `trace_truncated`** (fit.py:196, 522):
  - The field defaults to False and `Candidate.report` emits the key only when it is True.
  - It is set only in `grid`, which no Gate 3 module calls: there is no `grid(` in screen.py or pilot.py, and Gate 3 fits through `oracle()` and `ridge`.
  - It is appended last with a default, so positional `Candidate(...)` construction is unaffected.
- **E6 raise** (fit.py:33-35): it fires only for `nonlinear=False` with 3 or 4 tensors. Gate 3 builds two-tensor adapters only (`ridge`, and the `oracle()` → `fit_sgd` default with hidden None and expanded False), so it cannot reach the raise.
- **`plateau` → `table_b`** (fit.py:391-424):
  - The guard order is unchanged: finite selections, then the two-window trace length, then the loss difference.
  - The arithmetic is line-for-line the old body, with `early.selection`/`late.selection` passed as floats.
  - The oracle byte pins were not re-run here; the report says they pass at the commit.

## Item table

| Item | Status | file:line (at 978850a) | Pinning test |
|---|---|---|---|
| A1 (R30/R34) | Addressed (per R34) | costs.py:181 `arm_seconds` (public); pilot.py:169 `OWN_CHILD_ARMS`, :960 `scaled_anchor`, :985 `measured_charges(run, w, raised)` | `MeasuredProjectionTests.test_the_measured_term_is_the_confirmatory_arm_seconds_at_section_8_magnitudes` (scaled = per_anchor(§10) = 1,010 s and static 1,120 s, both axes; run, OK); `PilotRunTests.test_raised_axis_…` (children without maff, M-aff once in arms_run) |
| A2 (R31) | Addressed | pilot.py:138 `EXTENDED_ARMS`, :687 `core_sources`, :695 `stage_core` (PILOT_ARMS ∪ STATIC_ARMS, `admitted` removed), :854 `extended_arm`, :869 `merged_anchors`, :896 `stage_extended` (EXTENDED_ORDER, R23 re-projection and catch per item), :1230 07 reads `joined`; worker.py:815 `from_receipt`, :862 `rank8_survivors` | `PilotRunTests.test_extended_items_run_as_their_own_children_after_the_core_in_section_8_order` (run, OK), `…a_refused_starved_pair_leaves_r_all_m_exp_and_the_lottery_unrun`, `…a_ceiling_hit_inside_r_all_truncates…` (run, OK), `…an_unavailable_extended_arm_is_recorded_in_05_extended…`; `WorkerTests.test_lottery_reads_the_rank8_survivors_from_the_core_modules_receipt` |
| A3 (R32) | Addressed | fit.py:404 `table_b`; retune.py:306 `certify`, :362 R-all None, :373 `plateau`; worker.py:1090 `certify=name != "R-all"` | `RetuneTests.test_retune_plateau_is_certified_by_table_b_at_the_last_two_checkpoints` (run, OK); `test_retune_grid_best_and_edge`; worker retune test (bool, R-all None); clean pilot test (`plateau_fractions` has R-272s and R-aff, not R-all) |
| A4 | Addressed | stats.py:31 `BOOTSTRAP_SEED`, :95 `PAIRED_SE_KEYS`/`PAIRS`, :274 `_paired_ses`; receipts.py:32 re-export | `StatsTests.test_per_anchor_paired_bootstrap_se_for_every_two_arm_quantity` |
| B1 | Addressed | verdicts.py:140 | `VerdictTests.test_programme_reading_routes_the_next_screen_to_the_starved_host` |
| B2 | Addressed | verdicts.py:38 `_ordered` (v1, v2, labels D3) | `VerdictTests.test_bounds_must_be_ordered` |
| B3 | Addressed | confirmatory.py:322 | `ConfirmatoryTests.test_clean_confirmatory_runs_the_whitelist…` |
| B4 | Addressed | confirmatory.py:546 | same test |
| B5 | Addressed | pilot.py:1180 hold; 00-opening reads `run.supervisor.per_card` | `PilotRunTests.test_a_supervisor_admitting_more_than_one_process_per_card_stops_before_any_child`; per-batch card-uniqueness loop in the clean test |
| B6 | Addressed | costs.py:147 `STAGE_WORKLOADS = {"core"}` with the stages 1-3 comment; no production caller projects "V0" or "hosts" (grep) | `SupervisionTests.test_pilot_projection_is_charged_plus_remaining`; `AdmissionTests` literal charge |
| B7 | Addressed | identity.py:37; costs, pilot and confirmatory import it | `IdentityTests.test_margins_and_ceilings_are_the_frozen_values` |
| B8 | Addressed | worker.py:123 | n/a (comment) |
| C1 | Addressed | none | `IdentityTests.test_margins_and_ceilings_are_the_frozen_values` |
| C2 | Addressed | none | `WorkerTests.test_train_and_replay_children_are_the_gate3_training_recipe` |
| C3 | Addressed | none | `WorkerTests.test_replay_refuses_any_lineage_but_the_attempt4_anchors` |
| C4 | Addressed (per R34 c) | none | `StatsTests.test_anchor_bounds_are_the_500th_and_9500th_of_one_10000_by_n_draw` (resample-mean statistic), `…frontier_bounds_are_the_500th_and_9500th_replicate_slopes` |
| C5 | Addressed (per R34 minor: non-ancestor commit raises CalledProcessError) | none | `SupervisionTests.test_every_certificate_refusal_beyond_the_identity` |
| C6 | Addressed | none | `ConfirmatoryTests.test_refuses_a_lowered_axis_or_an_m_small_b_axis_not_twice_the_module_axis` |
| C7 | Addressed (literal-only pin by design) | none | `SupervisionTests.test_custody_receipts_are_the_section_11_paths` |
| C8 | Addressed | none | `RetuneTests.test_complement_hash_sees_only_the_complement` (BN running_mean; R-272s and R-all) |
| C9 | Addressed | none | `RetuneTests.test_recalibrate_after_changes_only_batchnorm_buffers` |
| C10 | Addressed | none | `SupervisionTests.test_core_stage_is_the_section_8_core_table` (6,540 s with derivation; CORE_CASE_ARMS literal) |
| C11 | Addressed | none | `SupervisionTests.test_gate3_defaults_keep_the_ledger_ceiling…` (`type is int`, raw `10800,`) |
| D1 | Addressed | instrument.py:44 | `InstrumentTests.test_wrong_inverse_is_an_instrument_failure` (`type is InstrumentHold`, alpha-only rung); `WorkerTests.test_instrument_class_failures_publish_software_valid_receipts` (real instrument child, failure.json `instrument_valid: false`) |
| D2 | Addressed | none (tests/test_m0_instrument.py `fixture()` seeds before `Host()`) | `InstrumentTests.test_a1_drift_clause_is_exactly_1e_4` (1e-4 passes, 1e-4 + 1e-9 holds) |
| D3 | Addressed | instrument.py:109 alpha check; confirmatory.py:6 docstring | `InstrumentTests.test_instrument_literals_and_exact_boundaries` |
| E1 | Addressed | none | `FitTests.test_grid_plateau_compares_the_last_two_checkpoints_never_the_first` |
| E2 (R33) | Addressed | fit.py:196, :522; `grid` docstring | `FitTests.test_a_diverged_runs_reached_checkpoint_is_a_separate_run_without_its_trace` (non-mock) |
| E3 | Addressed | none | `FitTests.test_example_passes_follow_the_250_per_step_v4_count` |
| E4 | Addressed | none | `FitTests.test_checkpoints_off_the_cycle_boundary_are_bitwise_separate_runs` (n = 600; 2, 4, 7) |
| E5 | Addressed | none | `FitTests.test_a_cold_affine_run_starts_from_the_oracle_init_draw` |
| E6 | Addressed | fit.py:33-35 | `FitTests.test_three_and_four_tensor_adapters_are_nonlinear_only` |
| F1 | Addressed | none | `WorkerTests.test_the_worker_shell_seals_pins_and_never_reuses_an_output`; `…instrument_class_failures_publish_software_valid_receipts` |
| F2 | Addressed | worker.py:263 `_bound` holds on a missing file | `WorkerTests.test_every_sha_binder_refuses_a_file_changed_or_missing_after_publication` |
| F3 | Addressed | none | `WorkerTests.test_each_arm_fits_on_its_own_frozen_rate_set` |
| F4 | Addressed | none | `WorkerTests.test_each_acceptance_flag_turns_false_when_its_comparison_breaks` |
| F5 | Addressed | receipts.py:173 `inventory(name, record)` | `ReceiptTests.test_a_dependent_arm_never_lists_another_fits_candidates` |
| F6 (+R34b/c) | Addressed | worker.py:1262 (iv) against `chunked_accuracy`, :1281 | `WorkerTests.test_acceptance_accuracy_is_checked_against_an_unpadded_forward`; `…the_accuracy_reference_runs_in_eval_chunk_batches` (run, OK) |
| G1 | Addressed | none | `ModuleTests.test_survivors_sorted_for_every_cut` |
| G2 | Addressed | none | `ModuleTests.test_lottery_start_is_the_step_zero_adapter_of_the_wide_grid` |
| G3 | Addressed | none | `WorkerTests.test_survival_widths_are_the_frozen_set`; `StatsTests.test_stage_arm_sets` |

Nothing outside the dispatch changed. Every production hunk maps to an item above. The appended test-fixture changes the report lists are consistent with the items they serve: FakeCluster `alpha` and R-all `certified: None`, the `PLATEAU_WINDOW` patch, the `overflowing_retune` `certify` parameter, and the worker rate sets.

## Checks 2-9

**2. A2 (R31)**
- **Job order.** `stage_extended` iterates `[item for item in EXTENDED_ORDER if item in admitted]`, where EXTENDED_ORDER is starved-pair, R-all, M-exp, lottery, cooled-20, F(0.25), F(0.75). Each item is re-projected first through `extended_fits(run.supervisor.charged, extended_cost(item, w), …)` and stops at the first miss. The later items are recorded as "follows …".
- **Ceiling catch.** Every item, the arm items included, runs inside the R23 `try … except PilotHold` with the `ceiling_stop` filter.
- **`--arms` per child.**
  - Core modules children: the MODULE_ARMS order of (PILOT_ARMS ∪ STATIC_ARMS), less M-aff on the default axis.
  - Core retune children: `R-272s,R-aff`.
  - Extended children: `retune --arms R-all`, `modules --arms M-exp` and `modules --arms lottery`, all with `--noop-root` and both axis flags.
  - Only the lottery child gets `--modules-root <root>/modules-<s>`.
- **Lottery.** The worker drops the lottery from the M-wide dependents when `--modules-root` is given, so no h = 64 grid runs (the test spy sees `[8]`). It reads `survivors` from the core receipt's `M-wide→8:rank:sliced` record, which is bound by `vectors.pt` and must carry the same `noop_sha256`. If the record is missing, or holds other than 8 units, it raises PilotHold.
- **Merge into 05-extended.json.** `merged_anchors` joins the core sources with the completed extended children through `receipts.anchor`, which checks the shared N0 and refuses duplicated arms. Each anchor's record carries the extended arms, their unavailable and harm entries, and G_mexp, G_lottery and retune_fraction with their reasons.
- **05-spreads.json is core-only.** `spreads(anchors, quantities, …)` takes the stage-5 core quantities only. The clean test asserts `G_mexp` is None with "M-exp not run" there and non-null in 07.
- **Starved-pair refusal and ceiling hit.** A refused starved pair leaves every later item unrun and writes no `anchors` key. A ceiling hit inside R-all truncates with `not_run = EXTENDED_ORDER[2:]`, and 06 and 07 are still published with `pilot_clean: true`. Both behaviours are pinned.

**3. A1 (R30/R34)**
- **Scale.** `paid = {T-cool, S-static} ∪ ({M-aff} unless the grid children ran it)`. The scale is `arm_seconds(§10 − paid) / arm_seconds(arms_run − paid)`, and the static scale is the same with STATIC_ARMS added to the numerator. At the fixture this gives scaled = 1,010.0 = per_anchor(§10) and static_scaled = 1,120.0, on both axes; the test was run.
- **Raised axis.** `measured_charges` drops `maff` from the whole children, so `per_anchor_children` is `["noop", "tcool", "modules", "retune"]`, and M-aff stays on both sides of the scale.
- **`within_ceiling`.** `confirmatory_count` is absent from the diff, so `within_ceiling` is still `max(kernel, measured) <= M0_CONFIRMATORY_SECONDS`. The clean pilot test asserts `count["within_ceiling"] == (larger <= M0_CONFIRMATORY_SECONDS)` and that the carried `confirmatory_projection.projected_gpu_process_seconds == larger`, with `per_anchor_gpu_process_seconds == max(measured per_anchor)`. The confirmatory reads that same carried value (unchanged code).

**4. A3 (R32)**
- `retune.plateau` calls `fit.table_b(early.selection, late.selection, before − after, late.training_losses, seed)`.
  - `before` and `after` are `losses` on `features_from_images` of each snapshot host.
  - early and late are `_choose` over the finite candidates at checkpoints[-2] and [-1], mirroring `fit.grid`.
- The test recomputes Table B independently (paired_se, the 80-step tail, the 40-step windows) for one uncertified case (False) and one certified case (True), and asserts equal SEs.
- R-all passes `certify=False`, so `plateau` is None ("R-all is never plateau-certified (§4)").
- `pilot.spreads` collects every non-None `certified`. So R-272s and R-aff (now bools) enter `plateau_fractions` and R-all (None) does not, as pinned in the clean pilot test.

**5. A4**
- All ten two-arm keys get `<key>_se` and `<key>_acc_se`.
- Orientation matches each quantity's definition:
  - D1: L(M-aff) − L(M-sq).
  - D2 family: comparator − arm.
  - D4: L(M-sq) − L(fresh).
  - D5 and D5_norefit: L(M-sq) − L(S-static[:refit]).
  - D1′: L(max re-tune) − L(M-sq), with R-272s winning a tie as `max()` does.
  - D3: `tcool_noop_loss` − L(M-sq|T-cool), and for accuracy `correct(M-sq|T-cool) − correct(T-cool)`. `tcool_noop_loss` is `vectors["T-cool"]["loss"]`.
- The test pins D1 loss and accuracy, D1′ and D3 against direct `paired_se(·, 3000)`, plus None when an arm is absent.

**6. D1/D2**
- D1: `type(error) is InstrumentHold` on the first (α = 0.1) rung alone. Through the real `instrument` child, failure.json has `software_valid: true`, `instrument_valid: false` and `error_type: InstrumentHold`.
- D2: `torch.manual_seed(0)` now precedes `Host()` in `fixture()`.

**7. R34b/c**
- `chunked_accuracy` runs `host.classify` on `data.base` in EVAL_CHUNK = 128-row batches.
- Each batch is zero-padded to exactly 128 rows and sliced back before the argmax.
- (iv) compares it with the padded evaluator's `accuracy` by `torch.equal`.
- The test records [128, 128, 128] on a 300-row site and an exact reference.

**8. Pins and mutations**
- Judged by reading, every C/E/F/G pin exists and would fail on the mutation it targets.
- The implementer's ten recorded mutation kills (C4, C5, C6, C8, C9, C10, E1, E3, G1) are consistent with the assertions as written.
- Notable strengths:
  - E1 also asserts inequality with the checkpoints[0] comparator.
  - G1 asserts that the fold moves the bias by more than 1e-3, so it is non-vacuous.
  - F3 forces M-wide's selected rate off MODULE_RATES[0].
  - F6 drifts `logits` and `losses` coherently while `classify` stays independent.
- Pins that are weaker by design, all accepted:
  - C7 is literal-only.
  - C4 uses a resample-mean statistic (R34 c).
  - C5 asserts CalledProcessError for the non-ancestor commit (R34 minor).

**9. Trust policy**
- No added line contains `.get(`, `Any`, `cast(`, `type: ignore` or `noqa` (grep over the `+` lines of the range).
- `scripts/check_policy.py` passed, and it verifies the `@t3_boundary` paired-test fingerprints.
- New `.get`-free lookups use membership tests: `mapping(receipt["arms"]) if "arms" in receipt else {}` (worker.py:870-871).
- F5's new read of `record["cause"]` on every non-available record is safe. The only producer of `"status": "unavailable"` in the M0 code, `Collector.attempt` (worker.py:611-616), always sets `cause`. So there is no new KeyError path at aggregation.

## New issues

1. **Minor (informational): B5's check runs after custody** (pilot.py:1180, after `claim_attempt` at :1170).
   - The `per_card != 1` hold sits inside the attempt, so if it ever fired it would consume the single pilot custody attempt.
   - It is unreachable: the pilot constructs its own `Supervisor(…, per_card=1, …)` with the literal on the line before.
   - Placing it before `claim_attempt` would be cleaner. No action needed for gate B.
2. **Minor (informational): the lottery child holds, not "unavailable", when the core receipt lacks the survivors** (worker.py:822 calls `rank8_survivors` outside `collect.attempt`; :872-876 raise PilotHold).
   - Without an available `M-wide→8:rank:sliced` record, the extended lottery child fails, and the pilot stops `execution-invalid` (not a ceiling stop) instead of recording the lottery unavailable.
   - It is unreachable in production. `M-wide→8:rank:sliced` is in CONFIRMATORY_ARMS ⊂ PILOT_ARMS, and stage 5's `receipts.anchor(…, stats.PILOT_ARMS)` already holds (R11, InstrumentHold) if it is unavailable on any anchor, before any extended child runs.
   - This is recorded so the next reader need not re-derive it.
3. **Minor (cosmetic): one docstring line is 172 characters** (worker.py:1194, the acceptance_action docstring rewritten in e8534e6). That is over the project's 120 line length. E501 is ignored by config and ruff passes, and the file already had a few lines just over 120.

## Declined to judge

- Bitwise parity of padded and unpadded batch shapes on GPU (R34c's rationale): it cannot be exercised on CPU, so it was not judged.
- The oracle byte pins (`ORACLE_REPORT_SHA`, `ORACLE_BEST_SHAS`, `FIT_SHAS`) were not re-run. The `plateau` → `table_b` refactor was verified by reading only.
- Not run, per the brief: the full suite, mypy and `ruff format --check`. The implementer reports 410 tests OK with mypy, ruff and policy clean at 978850a.
- The extra GPU cost of A3 (two from-images selection evaluations per certified re-tune arm), and whether the timing child's kernels cover it for R-272s as well as R-aff: this is a cost-booking question, not a correctness one, and was not judged.
