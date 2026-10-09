# Independent semantic audit — Sultai hybrid TCD

Reviewed commit: `ccdfd5225ee5575f365aa6a4a0698bed0f652958` (tree `274d65d0b0b958a952cc4be4046b4fb077b50dfa`). Review date: 2026-10-09. Sources are the byte-exact allowlisted snapshot, not a mutable checkout.

## Decision

The fixed delivered experiment supports its bounded reported numerical results, separation of selection from final auditing, task-gradient handover, taper, and actual adapter deletion. It does **not** establish a general fail-closed contract for missing or invalid required state. Four actionable boundary gaps are below. None was triggered by the supplied fixed fixture; this distinction matters. The CLI takes no arbitrary telemetry or model input. The model-state counterexamples use existing callable constructors/functions; report-boundary examples inject incomplete producer reports at the existing runner seam. They establish observable failure to reject those states, not evidence of a corrupted historical run or a remote input vulnerability.

No previous review report or implementer assessment was read. No Esper, Simic, or ELSPETH repository/experiment was inspected. John's account of `.get` and Esper-lite telemetry is user-reported history, not a reproduced finding here. No training campaign, dependency installation, network access, source edit, commit, tracker action, or main-repository change was performed.

Applied the actual installed Nyx skill:
`/home/john/.codex/plugins/cache/foundryside-marketplace/yzmir-systems-thinking/2.0.0/skills/using-systems-thinking/SKILL.md`
and its `stocks-and-flows-modeling.md` sheet. No Windows skill path was substituted. Read snapshot `AGENTS.md`, `README.md`, all `src/` and `tests/`, active `HYBRID_CONTRACT.md`, `HYBRID_DEMO.md`, delivered result/runtime evidence, and the source manifest. `docs/RESUME.md`, requested by AGENTS, is absent from this allowlist snapshot; I did not escape the review boundary to retrieve it. This is a snapshot-context limitation, not a finding that the repository lacks that file.

## Confirmed findings

### R1 — P2: Missing affine output channels become perfect loss and an admitted action

Locations: `src/sultai/formation.py:207–215` (`AffineControl`), `src/sultai/repair.py:222–227` (`mse`); decision at `src/sultai/formation.py:332–337`.

`AffineControl` has no constructor shape/finite checks. Its nested `zip` truncates missing rows/bias. `mse` then compares only returned channels while dividing by the full 16-channel denominator. An entirely missing prediction is measured as zero error. This contradicts both the advertised 272-value affine control and the loss definition; finite scalar loss does not establish valid telemetry.

Concrete counterexample, using normal current constructors/functions:

```python
e = make_episode(3000)
a = AffineControl((), ())
d = admit(a, e.selection)
# len(a.apply(e.selection[0].h)) == 0
# d.raw_selection_mse == 0.0
# d.no_op_selection_mse == 0.04291646514408539
# d.acted is True
# mse(a, e.test) == 0.0
```

Impact: a malformed candidate is admitted as a perfect repair although it produces no outputs. Partially missing channels similarly understate error. The ordinary `_fit_affine` producer supplies valid dimensions today; no claim is made that the saved affine result is affected.

Fix: validate affine state as strictly as `Adapter` (16 rows of 16 finite values plus 16 finite biases), and require complete 16-channel predictions in the metric boundary. This prevents future producers from bypassing the loss contract. Apply the same shape rule to lifecycle `_prediction_mse` (`lifecycle.py:231–236`), whose nested zip has the same truncation behavior, though the normal `_Model.predict` currently validates its output.

Regression: `RequiredStateRegression.test_affine_empty_state_rejected_before_admission`. Expand with 15/17 rows, short/long bias, nonfinite values, and short predictions; assert refusal before an admission outcome is recorded.

### R2 — P2: Incomplete learned-generator state silently becomes no-op

Location: `src/sultai/formation.py:190–198` (`FrozenGenerator`).

The generator is a frozen dataclass but does not validate or normalize its learned arrays. `form` zips its required 7/11 input features with however many weight rows happen to exist. Missing rows are interpreted as absent contributions; extra rows, including nonfinite state, are ignored. This loses the distinction between a learned zero prediction and missing learned state.

Concrete counterexamples:

```python
FrozenGenerator('paired', ()).form(make_episode(3000).conditioning) == Adapter.zero()
# True, no error; an uninitialized generator looks like a valid no-op former.
FrozenGenerator('paired', ((0.0,) * 4,) * 11 + ((float('nan'),) * 4,)).form(
    make_episode(3000).conditioning) == Adapter.zero()
# True, the extra nonfinite learned row is never inspected.
```

Impact: a missing/truncated former can silently produce a valid-shaped no-op and erase the diagnostic signal that formation was unavailable. A future outcome consumer could learn that an opportunity merits no action rather than that the former state was invalid. This is not happening in the fixed `_regress` path: that producer supplies complete arrays. It is already observable at the public generator boundary, not an imagined new algorithm.

Fix: validate a known mode, exactly 7 rows for coarse or 11 for paired/probes, exactly 4 finite numeric entries per row, and convert arrays to immutable tuples at construction. Do not use an empty learned array as a zero model; construct a valid explicit zero model if that is the desired state.

Regressions: `test_frozen_generator_missing_rows_rejected`, `test_frozen_generator_unused_nonfinite_rows_rejected`. Also test ragged/extra columns and explicitly valid all-zero arrays, so zero remains distinct from missing.

### R3 — P2: Required evidence can disappear while the acceptance report stays successful

Locations: `src/sultai/hybrid.py:37–46`, `72–74`; `src/sultai/formation.py:393–401`. Supporting test contract: `tests/test_hybrid_cli.py:17–24` permits arbitrary one-key acceptance maps.

The runner checks only that each acceptance dictionary is nonempty and every supplied value is Boolean. It never checks the required gate names. Deleting `formed_post_removal_target_met` from a real lifecycle report still produces `passed=True`, with 15 rather than 16 gates. Separately, `formation_acceptance` treats an empty healthy summary as successful no-op evidence because `all([])` is true; its finite-metrics pass has the same missing-coverage weakness. Healthy per-lineage evidence/counts are not examined. In summary mode, absent assay `summary` is replaced with `{}` and still exits zero.

Reproducers in the supplied review script start from the real saved report and alter only the missing field. Observed results:

- Lifecycle removal-target gate omitted: `passed=True`, 15 gates.
- `formation_acceptance(real_dev, real_test, {'summary': {}})`: all five returned gates true, including `healthy_all_no_op` and `finite_metrics`.
- Missing `formation.summary` at the existing `main` report seam: exit code 0 and empty `formation_summary`.

Impact: the evidence validator cannot distinguish an omitted required check or absent healthy telemetry from evidence of success. The normal producers include these fields today; reproducing the CLI seam requires replacing a producer result, not a supported command-line flag. This is a validator-contract gap relevant to the requested missing-state audit, not proof of an ordinary fixed-run false pass.

Fix: define the exact required gate and report schema (including expected phases, methods, nonempty lineages and their expected counts); validate it before combining outcomes. Require summary keys rather than defaulting them. Report missing evidence as invalid/incomplete, distinct from a valid failed target and a valid no-op. Keep the ridge guard's existing explicit nonempty check and extend the principle to healthy/finite coverage. If `formation_acceptance` remains callable independently, do not unconditionally certify split isolation without validated provenance of the guard.

Regressions: `test_required_runner_gate_cannot_be_removed`, `test_empty_healthy_evidence_rejected`, `test_missing_summary_not_rendered_as_success`. Parameterize removal of every required gate/method/phase. Add a genuine bad-lifecycle-outcome fixture: current `test_failure_is_reported_without_target_driven_retry` checks consistency on a successful fixture rather than exercising failure.

### R4 — P2: Cost ledger aggregation erases missing/unknown state and accepts invalid counts

Locations: `src/sultai/formation.py:49–56`.

`costs` deliberately expands a sparse specification; defaulting unmentioned operations to zero there is reasonable. `add_costs`, however, uses `.get(key, 0)` on each ledger without validating its key set or values. Every normal caller passes complete factory-produced ledgers, so an absent key at that boundary represents missing state rather than a known zero. Unknown keys are silently discarded. The factory itself accepts booleans, fractional counts, NaN and positive infinity because its only numeric check is `< 0`.

Concrete outcomes:

```python
add_costs({'selection_queres': 99})['selection_queries']  # 0
add_costs({'selection_queries': -99})['selection_queries']  # -99
costs(selection_queries=True)['selection_queries']  # True
costs(selection_queries=1.5)['selection_queries']  # 1.5
costs(selection_queries=float('nan'))['selection_queries']  # nan
```

Impact: lost telemetry can become zero search/selection cost; a typo can disappear entirely. Negative/fractional counts survive ordinary JSON serialization. The final runner's `allow_nan=False` catches nonfinite numbers only if they survive into the final report, so it does not fix finite malformed counts or erased unknown keys. The delivered fixed ledgers match executed operations and use valid integers; this finding does not revise their recorded totals.

Fix: retain the sparse zero-filling factory, but require `type(value) is int` and `value >= 0` for all counts. Validate exact `COST_KEYS` at the aggregation boundary, or explicitly name/document a separate sparse-delta type and reject unknown keys even there. Tests must distinguish known zero from missing required ledger state.

Regressions: `test_malformed_cost_count_rejected` and `test_missing_or_unknown_ledger_key_rejected`.

## Successor hardening scope

R1–R4 are directly addressed by the subsequently authorized strict typing/required-contract work: owned validated model state, exact shapes/counts, complete report/acceptance schemas, and rejection of missing required evidence. No blanket prohibition on optional defaults is warranted. Preserve the pinned numerical settings and results; these fixes need not change algorithms, schedules, or claimed research scope. Broader real-host transfer, calibration, adaptive removal, or developmental-advantage experiments require a separate proposal and are not fixes requested by this audit. This review made no ELSPETH inspection; any such convention study belongs to the successor work, not evidence about this pinned commit.

## Causal/state model and checks

Boundary: the standard-library fixed procedural experiment, offline teacher/generator training, formation/admission, separate lifecycle assay, accounting, and evidence publication. Outside the boundary: Tamiyo learning, real-host transfer, real telemetry ingestion, a learned removal policy, calibration, near-empty growth, and a tuned half-size comparison. Time horizons are one formation episode and 136 lifecycle host updates; no forecast beyond those horizons is supported.

Stocks: host parameters (272 allocated from step zero), temporary adapter parameters (0 or 272), retained generator/teacher/template assets, host learned weights, and accumulated query/update counts. Allocation and deletion change parameter stocks discretely. Gradient updates change weight values but not allocated capacity. Influence `alpha` is a control coefficient, not a parameter count: zero influence does not remove a stock.

Information/flows: oracle conditioning pairs inform the frozen generator; candidate/no-op selection MSE determines the acted flag; test labels inform only final audit. In lifecycle, the frozen formed adapter contributes `alpha*A(x)`. Task residual `target - H(x) - alpha*A(x)` drives host SGD. Taper reduces `alpha`, increasing the host's unmet residual and thereby changing host gradients; this is a plausible balancing mechanism within the fixed same-feature task. The derivative direction is exact in the code, but convergence/generalization is not guaranteed for arbitrary tasks or step sizes. After insertion, the fixed schedule—not loss, audit, or an adaptive policy—controls withdrawal and deletion. No feedback from final-test loss into schedule, fitting, admission, or stopping was found.

Observed causal checks: all phase candidate banks form before phase selection; no-op baseline is shared with a separately charged query pass; admission ties prefer no-op; raw candidates and admitted candidates are both audited and retained. Rejected candidate harm delta is zero because its admitted adapter is no-op; acted/realized-harm counts use all eight lineages, with no population safety claim. Healthy examples intentionally reuse the evaluation input process with different zero-repair targets; this is a paired control, not independent additional host evidence. The seed/sample-ID guard protects the actual training/development/test split, and `Episode` independently rejects within-episode repeated IDs. No exception swallowing exists in the source.

Lifecycle adds a separate object, freezes the formed adapter, updates the host from task gradients, deletes the temporary attribute, serializes no adapter for removal arms, and continues training. Plain SGD has no optimizer buffers. Weak-reference and serialization tests support actual disposal; equality at alpha zero alone would not. The formed path's final error is higher than no growth and early-static taper, which is disclosed and reproduced. The simpler alternative explanation is sufficient: ordinary same-feature SGD learns the task; the temporary repair changes the transient residual. Nothing here identifies a developmental advantage.

Falsifiers/discriminating checks: poisoning final-test labels must change audit outcomes but not formed weights, selection decisions or final host state (existing counterfactual tests exercise this); insertion must leave host weights identical; deletion must remove object/serialization state even when predictions match alpha-zero state (existing tests exercise this). Claiming a useful learned former additionally requires an intact generator and meaningful outcome evidence; the R1–R4 counterexamples falsify a blanket claim that current boundaries reject missing/invalid state. A future real-task advantage claim needs independent controls/tasks and a declared margin; this audit did not run or propose tuning the held-out fixture.

Feasible intervention comparison: no change retains a reproducible fixed demonstrator but leaves silent malformed-state pathways. Local schema/shape/count validation closes those pathways with small runtime cost and makes caller errors explicit; it may break tests/callers intentionally using underspecified stubs, which should be made contract-complete. Broad controller or model redesign is unnecessary for these findings and would confound the already-pinned result.

## Evidence and reproduction

All 36 manifest files matched SHA-256 before and after the checks. Executable aggregate matches the saved result:
`96913b8d85d7d6dfc4feddc4844d0d61db6e51f5462ba483c98654e6a9035dd9`.

The live fixed report, normalized through JSON, matches the saved report exactly. The formatted serialized report SHA-256 is:
`adae7d8ab70e6bd13ddd44dd3127468e3b0937f0cca7fa05b0a7d615559e3326`.
All 16 fixed-run gates pass. Existing suite: 37 tests passed in 6.539 seconds. Reviewer regression expectations: 8 test methods, 13 assertion/subtest failures, 0 errors on this commit. These failures are deliberate demonstrations of absent refusal behavior; they are not tests added to or run from the source tree. Existing test success is therefore evidence for the fixed mechanics, not coverage of all negative boundaries.

Artifacts beside the snapshot:

- `test_semantic_review.py`: executable reproducers plus expected fixed-behavior regression tests.
- `semantic-review-results.json`: observed values, original-suite transcript, failing-regression transcript, manifest checks and report identity.
- `semantic-review.md`: this report.

Run expected-failure regression tests:

```sh
nice -n 10 timeout 120 env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/john/Documents/Codex/2026-10-09/task-6/sultai-review-ccdfd52/snapshot/src \
  python3 /home/john/Documents/Codex/2026-10-09/task-6/sultai-review-ccdfd52/test_semantic_review.py
```

Add `--observe` to regenerate the evidence JSON, rerun the fixed bounded assay and original suite, and print observations. Each run is CPU/stdlib only under the external 120-second limit. Source identity covers the five imported executable modules as documented; launcher/tests/docs are bound separately by the pinned manifest. Runtime wall-clock figures are historical measurements, not reproduced performance guarantees. No remaining blocker prevents these findings from being evaluated; the omitted resume brief limits only ancillary context.
