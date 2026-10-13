# Exact-source review: pilot-and-confirmatory-custody (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Scope and method. The pre-rendered diff (`b79f145..e1df6e1`, 3,231 lines) adds three files and changes nothing else: `src/sultai_screen/m0/confirmatory.py` (574 lines), `src/sultai_screen/m0/pilot.py` (1,185 lines) and `tests/test_m0_pilot.py` (1,454 lines). Every line is an insertion, so I read the three files at HEAD = e1df6e1 in seven passes: pilot.py 1-330, 330-700, 700-1000 and 1000-1185; confirmatory.py whole; tests in four passes; then the supporting code. For the supporting code I read `m0/costs.py` (arm units, extended admission), `m0/identity.py`, `m0/seeds.py`, `m0/stats.py` (arm sets), `m0/receipts.py::receipt`, and Gate 3's `pilot.py::Supervisor` and `pilot.py::claim_attempt`. I checked them against prereg §2 (127-132), §7 (373-432), §8 (480-516), §9 (520-554), §10 (555-571), §11 (572-599) and rulings R15, R16 and R19-R28 (`progress.md`).

`PYTHONPATH=src:tests python3 -B -m unittest test_m0_pilot` gives 48 tests, all OK, in 70 s on CPU. The working tree stayed clean before and after.

## Claims and evidence

| Claim (prereg / ruling) | Source file:line | Pinning test (production change that would fail it) | Verdict |
|---|---|---|---|
| §2: "a confirmatory ceiling constant with its own supervisor instance" that charges and stops the run itself (§10: "12 GPU-hours as a new hard-stop constant") | confirmatory.py:251, 475 (`Supervisor(root, data, ceiling=plan.ceiling, per_card=1, …)`); identity.py:35 `M0_CONFIRMATORY_SECONDS = 43_200` | `ConfirmatoryTests.test_clean_confirmatory_runs_the_whitelist_on_fresh_anchors_under_its_own_ceiling`: every ledger's ceiling must be 43,200 and the constant must be the literal 43,200 (fails if `ceiling=` is dropped, since Gate 3 defaults to `PILOT_SECONDS`, or if the constant changes) | Met |
| §2/§8: the M0 pilot ceiling is a new constant, `PILOT_SECONDS` untouched | pilot.py:1090; identity.py:34 `M0_PILOT_SECONDS = 14_400` | `PilotRunTests.test_clean_pilot_publishes_every_receipt_in_stage_order_under_the_m0_ceiling`: every ledger ceiling must be 14,400, as a literal | Met |
| §2: one process per card | pilot.py:340-347 (`Run.run` splits by card and zips); pilot.py:1090 and confirmatory.py:475 `per_card=1`; R24 `card()` pilot.py:310-312 | Clean-pilot test and `ConfirmatoryTests.test_the_pilot_count_sets_the_anchors_exactly`: each ledger batch has unique cards, and every anchor's children share `(seed-start) % 2`. Fails if `Run.run` batches the whole list | Met |
| §11 custody: `.weft/m0-pilot-attempt.json`, claimed before any child, never reset | pilot.py:95, 1086 (`claim_attempt` after certificate and cards, before `root.mkdir` and any child) | `test_custody_is_claimed_before_any_child_and_a_new_root_cannot_reset_it` (FakeCluster records a violation for any child launched before custody); `test_an_existing_m0_custody_receipt_holds_before_any_child` | Met (literal path unpinned, Minor 4) |
| §11 custody: `.weft/m0-confirmatory-attempt.json`, separate from the pilot's | confirmatory.py:83, 473 | Clean-confirmatory test: the pilot `CUSTODY` must not exist afterwards (fails if the two constants are made equal); a second launch is refused | Met |
| "never the Gate 3 receipts" | pilot.py:94-95; confirmatory.py:82-85 | `test_a_gate3_custody_receipt_is_irrelevant_and_untouched` (bytes unchanged, no `gate3-pilot` prefix); clean confirmatory asserts the Gate 3 receipt text is unchanged | Met |
| §9 order: stages 1-7 in order | pilot.py:1110-1142 | Clean test: the published receipt sequence must equal `RECEIPTS`, in order | Met (with the R16/R17/R26 placements below) |
| §9 st.1: replay against attempt-4 hashes, both-card determinism, acceptance (i)-(iv) on both cards | pilot.py:365-413 | `test_replay_mismatch_holds_at_stage_one`; `test_an_acceptance_failure_is_instrument_failed_and_burns_the_block`, which reaches the same `stage_replay` | Met. "One grid replay" has no named comparator: gate-B point 3 |
| §9 st.1 (v) / R16: fold residual recomputed on card, after anchor 3000's M-aff | pilot.py:419-432, 471-473 | Clean test: maff-3000 < fold-check-3000 < maff-3002; `test_fold_check_mismatch_holds_at_stage_two_as_acceptance_item_five` | Met (as ruled) |
| §9 st.2: P1 known answer, zero-module identities, slicing known answer, M-aff fold residual on every anchor | pilot.py:450-508, 435-447, 237-264 | `test_a_false_zero_module_flag_holds_at_stage_two_before_any_maff_grid`; `test_fold_residual_above_tolerance_holds_at_stage_two_naming_the_fold_clause`; `SupervisionTests.test_slicing_known_answer_passes_on_the_cpu_fixture` | Met |
| §9 st.3: frontier, T-cool and S-static hosts; frontier slope; §7 S-static step 0 | pilot.py:561-600 | `test_a_static_step_zero_state_other_than_the_anchors_holds_at_stage_three` | Met |
| §9 st.4: projected charge from one-per-card maxima; extended admission by the overrun rule | pilot.py:606-661; costs.py:270-283 | `AdmissionTests.test_projection_is_charged_plus_remaining_work_after_two_stages_have_run`; `…workloads_take_one_per_card_maxima…` | Met at stage 4; run order deviates (Important 2) |
| §8/§9: core over the ceiling stops before V1 is opened, before any V1 field | pilot.py:281-283, 658-660 (`FeasibilityHold` after `04-admission.json`, before any `modules-`/`retune-` child and before 05) | `test_core_over_ceiling_stops_at_stage_four_before_any_core_child`: no modules/retune child and no 05-spreads; verdict hold-feasibility. Fails if the hold moves after `stage_core` | Met |
| §9 "schedule axis (raise only)" / R15 | pilot.py:610-632 (rule, published, read back, `raise_only`, 2× check); confirmatory.py:238-245 | `test_raised_axis_doubles_both_axes_in_one_receipt_and_reruns_maff` pins the raise. No test drives a lowered or mismatched axis into `raise_only` or the 2× check | Raise met; the guards are unpinned (Important 4) |
| R16: 04-schedule.json exists before any stage-5 module child | pilot.py:613-628 before 667-692 | FakeCluster `ordered` violation; clean test asserts no violations | Met |
| R19: exact remaining core | pilot.py:267-297 | `AdmissionTests…two_stages_have_run` (`remaining_core` literal arithmetic) | Met |
| §8 overrun rule / R23: re-project from the measured charge before each extended item; stop at the first miss; a ceiling stop truncates | pilot.py:828-880, 807-809 | `test_runtime_reprojection_stops_at_the_first_extended_item_that_no_longer_fits`; `test_a_ceiling_stop_inside_an_extended_item_truncates…`; `test_a_child_failure_inside_an_extended_item_still_stops_the_pilot`; `test_only_the_supervisors_ceiling_stops…` | Met for the host items and the starved pair. Not for R-all, M-exp or the lottery (Important 2) |
| §9 st.5: per-anchor SDs of D1, D2, D3, D5 before any median; S-static admission from SD(D5) | pilot.py:741-784 (`static_admitted` 780), 1130-1132 before 1142 | Clean test: the `sd` keys, `static_admitted == sd D5 <= 0.0275` as a literal, and 05 before 07; `test_confirmatory_hold_keeps_the_spreads_and_publishes_no_medians` | Met |
| §9 st.6: n = max(32, n*), sized on D1 only; fixed V0 term included; stop before freeze if over the ceiling; R20 half-widths | pilot.py:949-992, 1008-1024, 1139-1140 | Clean test (count ≥ 32, half-widths D2/D3/D5); `test_confirmatory_hold…` | Met. The measured term over-books (Important 1) |
| §9 st.7: development medians for the record only | pilot.py:1031-1069 | Clean test (`development_only`, `confirmatory_verdict` None) | Met |
| §10 whitelist = `stats.CONFIRMATORY_ARMS`; nothing else runs | confirmatory.py:88, 116-138, 250, 345-359 | `test_the_whitelist_refuses_an_extended_arm_and_an_unrestricted_arm_list` (refuses M-exp, M-sq-fresh, lottery, survival refits, R-all, missing or doubled `--arms`, instrument/timing/cooled20/static actions); clean test: arms run per anchor equal `CONFIRMATORY_ARMS` | Met |
| §10 S-static only if SD(D5) admitted it **and** the 14 h ceiling | confirmatory.py:224-227, 251; costs.py:22 `50_400` | `test_static_requires_the_pilot_admission_and_johns_14_hour_certificate`; `test_static_runs_s_static_under_the_14_hour_ceiling` (ledger ceilings 50,400, as a literal) | Met |
| §10 V0 then V1 and V2 read once; a V0 failure ends the attempt and burns its seeds | confirmatory.py:499-541, 443-455; failed_prior 162-169 | `test_a_v0_failure_publishes_instrument_failed_burns_the_block_and_reads_nothing`; `test_restart_refuses_after_a_prior_attempt_that_read_v1_and_v2` | Met (A1/B3 are §7 "development anchor" clauses, so not re-run, correctly) |
| §11 anchors 7500 onward, contiguous, ≤ 7599; count from the pilot | confirmatory.py:246-248; seeds.py:27, 48-49 | `test_the_pilot_count_sets_the_anchors_exactly` (33 gives 7500..7532); clean test (7500..7531) | Met |
| §11 restart: 7700 onward, only with a disposition, binding the prior | confirmatory.py:230-237, 85 | `test_restart_requires_the_listed_disposition_and_the_prior_custody_receipt` (7700..7731, bindings, prior bytes unchanged) | Met |
| R21: replay mismatch is execution-invalid; instrument clauses are instrument-failed | pilot.py:214-234, 381, 403 | `test_a_replay_mismatch_is_execution_invalid_and_still_burns_the_block`; `SupervisionTests.test_stop_report_has_the_gate3_shape_with_the_m0_identity` | Met |
| R22: `seed_block_burnt` on every post-custody stop | confirmatory.py:443-455 | `assert_burnt_stop` in three stop tests | Met |
| R27: pilot result carries the implementation source; the confirmatory requires the same source or a listed disposition naming the new hash | pilot.py:206-211, 1159; confirmatory.py:187-209 | `test_the_confirmatory_source_is_the_pilots_unless_a_listed_disposition_names_the_change` | Met |
| M0 identity on every receipt | custody (`source_identity()` carries `protocol_identity()`, identity.py:83); 00-opening; pilot-result; confirmatory-result; stop reports; child envelopes checked by `receipts.receipt` (receipts.py:83-87: full protocol identity plus certified source) | `assert_stop` / `assert_burnt_stop` (identity on stops); clean tests; `ReceiptTests.test_a_receipt_of_another_implementation_source_is_refused` | Met for custody, opening, terminal and child receipts. Stage files carry none (Minor 3) |
| §9 precondition: certificate (quality, runtime, independent panel, source hash, evidence) | pilot.py:175-203 | Only the protocol-identity loop is pinned (`test_certificate_must_carry_the_m0_identity_not_gate3`); every run test patches `check_certificate` out | Unpinned (Important 3) |
| R24 (anchor's card), R26 (two timing children), R25 (zero-module flags) | pilot.py:310-326, 393-397, 435-447 | Clean test (card per anchor, timing-0/1) | Met (ruled) |

Behaviour present but not asked for by the prereg. Each item is covered by a controller ruling, so none is a finding:
- `RAISE_FACTOR = 2` and the < 75 % M-aff trigger (R15).
- `EXTENDED_RESERVE = 0.05`, which admits extended work only under 0.95 × ceiling. That is stricter than §8's "projects to finish under the ceiling" (R23; gate-B point 4).
- Timing kernels are measured in stage 1, while §9 lists them under stage 4 (R17, R26).
- Fold-check (acceptance v) runs in stage 2 (R16).
- The CPU slicing known answer runs in the supervisor (R18 b).
- `R-temp` is the frontier N0 evaluator (R18 e).
- The confirmatory runs the fold residual on its fresh anchors too. This follows §10's "M-aff (with fold residual)".

No extra arm or extra randomness was found. `BOOTSTRAP_SEED` is a pinned constant (R18). No Gate 3 receipt byte is written: both supervisors use only `claim_attempt`, `Supervisor`, `Job`, `check_output_root` and `GPU_UUIDS`, and this diff does not touch `src/sultai_screen/pilot.py`.

## Frozen values checked

| Value | Prereg | Source | Match |
|---|---|---|---|
| Pilot ceiling | 14,400 s (§8 J6) | identity.py:34 `M0_PILOT_SECONDS = 14_400` | yes |
| Confirmatory ceiling | 12 GPU-h = 43,200 s (§10 J7) | identity.py:35 `43_200` | yes |
| S-static confirmatory ceiling | 14 GPU-h (§10, J3) | costs.py:22 `50_400`; confirmatory.py:226 requires `confirmatory_ceiling_hours == 14` | yes |
| Module checkpoints | 200, 800, 3,200 (§3/§4) | identity.py:38 | yes |
| M-small-B checkpoints | 400, 1,600, 6,400 | identity.py:39; 2× check pilot.py:631, confirmatory.py:244 | yes |
| Fold tolerance | X/4 = 0.005 (§7) | identity.py:31; pilot.py:484, confirmatory.py:311 | yes |
| Certification fraction | 75 % (§7) | identity.py:32 | yes |
| S-static SD limit | SD(D5) ≤ 0.0275 (§7 J3) | identity.py:33; pilot.py:780 (`<=`) | yes |
| ρ | 0.8 (§2) | identity.py:29; pilot.py:462 (`>=`) | yes |
| A1 tolerance | 1e-4 (§7) | instrument.py:24 `INVERSE_TOLERANCE = 1e-4`; pilot.py:460 | yes |
| Confirmatory minimum | n = max(32, n*) (§9 st.6) | `stats.confirmatory_count` (pilot.py:958), `measured_confirmatory` floor | yes |
| Development anchors | 3000-3005 (§9) | seeds.py:21 | yes |
| Frontier, pilot | 0.5: 6000-6005; 1.5: 6012-6017; 0.75: 6006-6011; cooled-20: 6018-6023; 0.25: 6024-6029 (§11) | seeds.py:22-26 | yes |
| Confirmatory anchors | 7500 onward, contiguous, ≤ 7599 | seeds.py:27 `range(7500, 7600)`; confirmatory.py:247-248 | yes |
| Restart | 7700 onward, ≤ 7799 | seeds.py:28 `range(7700, 7800)` | yes |
| M2 reserved | 6100-6199, never used by M0 | seeds.py:30 (not referenced by pilot or confirmatory) | yes |
| Extended priority | starved pair, R-all, M-exp, lottery, cooled-20, F(0.25), F(0.75) (§8) | costs.py:122 `EXTENDED_ORDER` | yes (the run order differs, Important 2) |
| §10 arm set | N0 and calibrations, M-sq, M-aff, R-272s, R-aff, T-cool, M-sq\|T-cool, M-wide, M-wide→8 (both rankings × sliced/refit), M-small, M-small-B | stats.py:68-84 (15 names: R-temp, R-cal, M-sq, M-aff, R-272s, R-aff, T-cool, M-sq\|T-cool, M-wide, four M-wide→8, M-small, M-small-B) | yes |
| V1/V2 certification arms | V1: M-aff (present), M-sq (equivalent); V2: M-small-B (helps), refit (hurts) (§7) | confirmatory.py:90, 413, 416-418 | yes |
| Protocol identity | `m0-v4`, the four-document composition (§11) | identity.py:21-23, 44-53; carried by custody, opening, results and stops | yes |

No frozen value needs to change for the code and the document to agree.

## Findings

**Important 1: the R28 measured per-anchor term over-books the §10 set by about 14 % at §8 magnitudes, which can cause a false stage-6 hold.**
- Where: pilot.py:886-901 (`scaled_anchor`), 909-919 (`measured_charges`); consumed at 961-990 and confirmatory.py:256-270.
- What: `whole` already pays the anchor's host:1.0, its N0/calibrations (noop child), its M-aff child and its T-cool child. The scale factor is `per_anchor(w, CONFIRMATORY_ARMS) / per_anchor(w, arms_run)`. Its numerator again contains `ANCHOR_UNITS` (host + fold_and_calibrations), M-aff's `grid:aff` and T-cool's `tcool`, so those costs are charged twice.
- Evidence: at the test fixture's §8 bookings (`workloads()`, where `per_anchor(CONFIRMATORY_ARMS) = 1,010`), `scaled_anchor` returns 1,153.6 s, while the exact §10 sum built from the same parts is 1,010 s, an over-book of 143.6 s (14.2 %). With §8's larger fixed term (1,500 s), n = 32 projects 38,414 s instead of 33,820 s. The surviving overrun factor under 43,200 s drops from §8's stated ≈ 1.28× to ≈ 1.12×. `projected_gpu_process_seconds = max(kernel, measured)`, so the inflated measured figure decides `within_ceiling`. A real overrun between about 1.12× and 1.28× therefore raises a stage-6 `FeasibilityHold`. The pilot is then not clean, and the confirmatory refuses it (confirmatory.py:216). That is a stop §9 stage 6 does not call for ("if the count exceeds the ceiling"), and it is the false-hold class R28 was ruled to remove.
- Pinning: `MeasuredProjectionTests` asserts that `scaled == whole + scale × grids`, i.e. it pins the formula, not its target. It passes because its n = 32 case still fits at nominal kernels.
- Fix (a correction to R28, no frozen value involved): scale only the arm-seconds the grid children ran to the §10 arms they stand for. For example, use `_arm_seconds(w, CONFIRMATORY_ARMS − {M-aff, T-cool}) / _arm_seconds(w, arms_run − {T-cool})`, or subtract `ANCHOR_UNITS`, `grid:aff` and `tcool` from the numerator and `ANCHOR_UNITS` from the denominator. Do the same for `static_scale`. Add a test asserting `terms["scaled"] ≈ per_anchor(w, CONFIRMATORY_ARMS)` at the fixture's bookings.

**Important 2: extended items 2-4 (R-all, M-exp, lottery) run inside the stage-5 core children, before item 1 and without the run-time re-projection.**
- Where: pilot.py:673 (`wanted` adds admitted R-all/M-exp/lottery to the core children), 837 (only the starved pair and host items reach `stage_extended`).
- What: §8 says extended arms "run in that order only while the measured charge projects to finish under the ceiling" and stop "at the **first** extended arm that does not project to fit". Here R-all, M-exp and the lottery always run before the starved pair (item 1). The starved pair is then the first item re-projected at run time. If it is refused, lower-priority items have already consumed charge, which is a stop-at-first-miss inversion. Also, an overrun inside those three arms surfaces as a ceiling stop in stage "core" and ends the attempt as execution-invalid. That contradicts "core arms are never dropped" and R23's "an extended overrun must never consume the attempt".
- Already on John's list: gate-B point 7 raises the core-exposure half. The priority-order inversion is not stated there.
- Pinning: no test exercises the starved pair being refused after R-all ran, or a ceiling stop inside a core child carrying extended arms.
- Fix: add the order inversion to gate-B point 7 for John's decision. Alternatively, run R-all, M-exp and the lottery as separate children after the starved pair, in `EXTENDED_ORDER`, each behind `extended_fits` and the R23 ceiling catch. The R28 scaling then no longer has to remove them.

**Important 3: M0's `check_certificate` beyond the identity loop is unpinned.**
- Where: pilot.py:180-202.
- What: the quality/runtime/independent-panel flags, the `implementation_source_sha256 == source_sha256` check, the `merge-base --is-ancestor` check, the evidence path/hash checks and the `len(evidence) < 3` floor are the certificate half of the §9 go precondition. All have no test. Every `PilotRunTests` and `ConfirmatoryTests` launch patches `check_certificate` out. `test_certificate_must_carry_the_m0_identity_not_gate3` exits at the first identity key (line 179). The Gate 3 tests in `test_screen.py` and `test_protocol_identity.py` exercise Gate 3's own function. Deleting any of lines 180-202 fails no test.
- Fix: add a unit test with a temporary `REPO` and patched `source_identity`/`subprocess.run`. It should drive each refusal: a false flag, another source hash, changed evidence, evidence outside the repo, fewer than three entries, and a non-ancestor commit.

**Important 4: the raise-only and 2× axis guards are unpinned.**
- Where: pilot.py:304-307, 629-632; confirmatory.py:238-245.
- What: §9 lets the pilot fill "the schedule axis (raise only)". The confirmatory reads the axis from an external committed file and applies `raise_only` and the M-small-B 2× check before custody. No test passes a lowered axis (e.g. `[100, 800, 3200]`) or a mismatched `checkpoints_b`. Removing either guard fails nothing. `test_refuses_a_pilot_result…` only covers a missing `checkpoints_b` key.
- Fix: add two `assert_refused` cases to `ConfirmatoryTests`: a lowered axis, and a `checkpoints_b` that is not 2×. Expect a refusal before custody.

**Minor 1: a raised axis double-counts M-aff in the measured term.**
- Where: pilot.py:909-918.
- What: on a raised axis, `whole` includes the default-axis `maff-{s}` child from stage 2, and `grids` includes the raised modules child that re-ran M-aff, which is then scaled. This is conservative and practically immaterial, because a raised axis roughly doubles the §10 per-anchor cost and projects over 43,200 s anyway. Still, 06-confirmatory-count.json would misstate the term.
- Fix: when `raised`, leave `maff` out of `whole`. Fold this into the Important 1 test.

**Minor 2: the confirmatory publishes literal V0 flags.**
- Where: confirmatory.py:316, 520-531.
- What: `02-v0.json` records `zero_module_identity: True`, and `confirmatory-result.json`'s `v0` block is all literal `True`. Each is true by control flow, because every clause raises first. Even so, the per-anchor R25 flags read at 288-295 are not published. The pilot was corrected for exactly this pattern (M3/R25, pilot.py:496-497, 705-721).
- Fix: publish the per-anchor flags dict in 02-v0.json.

**Minor 3: stage receipts carry no protocol identity, and the clean confirmatory result has no implementation source.**
- Where: pilot.py:1112-1142; confirmatory.py:500-511, 514-541.
- What: the stage files (01-07, 04-schedule, 04-admission, 05-extended, and confirmatory 01-04) carry no `protocol_identity()`. This follows the Gate 3 convention, and they sit under an output root bound by custody and 00-opening. However, the clean `confirmatory-result.json` omits `implementation_source_sha256` and `implementation_commit`. `pilot-result.json` and every stop report carry them (R27).
- Fix: add `**implementation(identity)` to the clean confirmatory result.

**Minor 4: the custody path literals are unpinned.**
- Where: pilot.py:95; confirmatory.py:83, 85.
- What: the tests import the constants, so renaming `.weft/m0-pilot-attempt.json` or `.weft/m0-confirmatory-attempt.json` to any other non-`gate3-pilot` name passes.
- Fix: assert the two §11 literals.

Not findings, because they are already ruled or already on John's gate-B list:
- The "one grid replay" comparator (gate-B 3).
- The stage-1 spend against the §8 600 s line (gate-B 4).
- The core exposure from extended arms inside core children (gate-B 7; see Important 2 for the order half).
- S-static step 0 in the confirmatory, which relies on acceptance (iii) (progress.md:120).

## Dispositions (controller, 2026-10-14)

- Important 1 + Minor 1: FIXED under Ruling R30 (item A1: arm-seconds-only scale; maff out of `whole` on a raised axis).
- Important 2: FIXED under Ruling R31 (item A2: R-all, M-exp and the lottery run as separate extended children after the core in §8 order).
- Important 3: FIXED (item C5). Important 4: FIXED (item C6). Minor 2: FIXED (item B3). Minor 3: FIXED (item B4). Minor 4: FIXED (item C7).

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
