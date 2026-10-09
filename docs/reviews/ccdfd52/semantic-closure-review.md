# Targeted successor closure review

Date: 2026-10-09. Worktree: `/home/john/Documents/Codex/2026-10-09/task-6/sultai-hybrid`, successor branch `strict-contracts-ccdfd52`. This is one targeted closure review, not a replacement for the independent pinned-commit audit. Original `semantic-review.md`, its evidence/reproducers, the ccdfd52 snapshot and neutral Claude handoff were not changed.

Source aggregate observed at the end of this review:
`32bfe509c3e680514fbfcdf5e24e6a7643dc1c78363ad31a10433b5d23b1f1f2`.
The worktree was uncommitted and implementation resumed after the remaining finding was sent; this is not approval of the subsequently changing tree. The lead also reported small concurrent implementer self-review changes. The aggregate is a point-in-time observation, not an assertion that a frozen final revision underwent another full review. Conclusions refer to the boundary implementations examined here, not an invented commit identity. The aggregate covers all eight production modules. `report_types.py` hash at this point was `96695212679128d2f823ea686483fb88648470b66e6eb594435b1d35827feabb`.

## Disposition

R1, R2 and R4 are closed for the reviewed boundaries. The concrete original R3 counterexamples are also closed, but a closely related missing-required-state gap remains in nested count/provenance maps. I recommend fixing that small schema omission before describing the entire successor publication seam as complete. This is not a model/scientific redesign or evidence that ordinary fixed producers emit corrupted results.

| Finding | Closure evidence |
|---|---|
| R1: missing affine outputs score zero | `AffineControl.__post_init__` owns complete finite 16-by-16 weights and 16 biases; `apply`, repair `mse`, and lifecycle `_prediction_mse` validate 16-channel outputs. Empty/short/long/nonfinite/Boolean predictions now raise before admission. |
| R2: missing generator becomes no-op | `FrozenGenerator` validates mode and exact 7/11-by-4 state, normalizes to detached tuples, and uses strict pairing. Valid all-zero state remains a genuine zero model. Tests mutate the original input list after construction to establish ownership. Teacher-bank and regression dimensions also reject incomplete state. |
| R3: incomplete gates/evidence pass | Exact acceptance-key sets and Boolean values are required; phase/method/stage/summary coverage is checked; 32/8/8 lineage and 64+24+64 sample identity counts are required; phase identities and outcome summaries are cross-checked against detailed rows. Missing healthy evidence and missing split validation fail. `main` parses before rendering and directly indexes required summaries. Nested metadata omission below remains. |
| R4: invalid ledgers become zero | Sparse `costs()` remains deliberately allowed but accepts only exact nonnegative integers and known keys. `add_costs()` requires at least one complete schema-validated ledger; missing, extra, negative, fractional, Boolean and nonfinite values are rejected. No default converts missing aggregate fields to zero. |

## Remaining P2: nested required count/provenance maps can still be empty

Locations in the examined tree: `src/sultai/report_types.py:696–698` (`parse_FormationConfig`) and `:1117–1123` (`parse_SourceIdentity`). These use `named(value, parser)` without its supported required-key argument. The outer field is mandatory, but every required nested entry can disappear while `parse_HybridReport` still returns `passed=True`.

Confirmed with a complete reference report derived from the immutable numerical record. Each mutation was tested independently:

```python
from report_fixture import reference_report
from sultai.report_types import parse_HybridReport

report = reference_report()
report['formation']['config']['lineages'] = {}
assert parse_HybridReport(report)['passed'] is True
```

The same result occurs when emptying `formation.config.generator_parameters`, `formation.config.evidence_dimensions`, or `source_identity.files`. None needs a numerical assay run. This is the same producer-report boundary used to validate the original missing-summary/gate finding; it is not a supported external CLI input or a claim that current fixed producers omit the fields.

Impact: publication can retain a success flag after losing required per-partition counts, learned-asset parameter counts/evidence widths, or every executable file identity. In particular, empty `files` is accepted alongside the unchanged aggregate digest, so the provenance object is internally inconsistent. Validating only the outer mapping and value types leaves absent members unrepresented by the owned schema.

Bounded correction: require exactly `train/development/test` in lineage counts; exactly `coarse/paired/paired_probes` in generator parameter and evidence-width maps; validate expected fixed counts/dimensions or their defined relationships; require the eight supported production source paths with valid SHA-256 strings, and validate aggregate consistency if accepting a serialized source identity. Add mutation tests removing each key, supplying an unknown key, and emptying each mapping; retain complete positive fixtures. This is directly within authorized required-contract hardening. It does not require a training rerun or expansion of the experiment.

## Focused verification and test quality

Ran 15 selected tests once, under `nice -n 10 timeout 120`, standard library only, with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=src:tests`: 15 passed in 3.430 seconds, zero failures/errors. Selection was all seven `StateContracts` tests, the seven `EvidenceContracts` tests excluding the full numerical runner comparison, and `LifecycleTests.test_actual_failed_audit_reports_failure_without_retry_or_state_changes`. The lifecycle class setup supplies a normal control; no whole-suite or learned formation rerun was performed in this closure review.

The tests are meaningful refusal/counterfactual checks rather than assertions that mirror only successful implementation output:

- Shape, finite-value and ownership tests have valid positive controls, including legitimate zero arrays and sparse cost construction.
- Tests delete every required acceptance gate and every method at both summary and lineage levels, remove phases/rows and summary fields, and inject non-Boolean gate values.
- Altering a healthy row's acted flag without its aggregate causes a counts error.
- Formed withdrawal MSE must be present and non-null exactly for the formed arm; formation digest follows the same conditional rule. The optional report field is inspected explicitly rather than read with a default.
- `_Model.temporary` is now explicitly `_Trainable | Adapter | None`. `remove()` drops the reference by assigning `None`, and detached serialization omits the adapter. Weak-reference tests for both frozen and trainable modules prove object release and a 272-parameter final allocation; this is genuine disposal despite retaining an optional attribute.
- The new actual failure fixture changes only audit targets by +10, obtains loss above 0.01 and false post-removal acceptance, and checks unchanged costs, all arms' host update counts and final serialization hashes. It exercises real failure without target-driven retry, unlike the old successful-only consistency test.

The focused parser probes above were executed after these passing tests and returned `ACCEPTED True` for all four empty nested maps. Existing negative coverage therefore does not close this particular gap. A final same-class check independently emptied lifecycle `summary.post_learning_mse`, formed-arm `frozen_host_withdrawal_mse`, formed-arm `mse`, and formed-arm `stage_state`: all four were rejected. Those nested maps have exact key enforcement, so the confirmed omission is limited to the formation configuration/source-identity maps described above.

## Types and examined scope

Read the complete changed boundary implementations and their relevant call paths in `contracts.py`, `report_types.py`, `formation.py`, `hybrid.py`, `repair.py`, `lifecycle.py`, and `smoke.py`; inspected tests in `test_contracts.py`, `report_fixture.py`, `test_hybrid_cli.py`, the lifecycle changes, formation/repair changes, plus `pyproject.toml`, `AGENTS.md`, README, resume brief and `docs/TYPING_CONTRACTS.md`. No previous independent review was consulted to produce new findings.

Production reports now have explicit TypedDict schemas; runtime parsers accept `object`, narrow exact containers/scalars, build detached typed records and reject unexpected keys at most publication levels. Model arrays are validated frozen nominal dataclasses. No explicit `Any`, `cast`, type-ignore escape or exception swallowing was found in production source. Mypy configuration is strict with explicit-Any refusal and Python 3.10 targeting. Ruff exclusions are limited to documented wrapping/presence/deletion style rules, not suppressions masking these type defects. The lead's mypy/Ruff/full-suite and quality-runner results were not independently rerun here and remain separately attributable evidence.

Limits: this is closure of the scoped missing/invalid-state findings, not proof of every possible cross-field algebraic invariant or a malicious producer authenticity boundary. Full numerical preservation, the saved v2 artifact and quality-tool negative controls are owned by the lead's independent integration checks. Scientific scope stays the same bounded synthetic formation and same-feature handover demonstrator; no real-host transfer, adaptive removal, population safety, developmental advantage or half-size result is newly established. No code, commit, tracker, outside repository or pinned artifact was changed by this reviewer.

Handoff status: lead assigned the remaining nested-map correction to Sol and will verify its focused regressions and final source hashes after a new frozen-source signal. No second full closure review was performed or claimed.

## Final lead disposition after the targeted review

The remaining nested-map case was corrected after the reviewer stopped.
LineageCounts and EvidenceModeCounts are total TypedDicts with exact runtime
keys and fixed protocol values. Source identity requires all eight supported
module paths, valid SHA-256 values and aggregate consistency; adding a new
runtime module fails until the versioned identity contract includes it.

Lead verification on the final frozen source aggregate
`2431e053ddd5ddf10d0565923a382023ae3eeb881fd233da777eac28ee75ab1f`:
55 tests passed in 13.162 seconds, including empty/deleted/unknown nested maps,
all eight source-file deletions, malformed digests, inconsistent aggregate and
unexpected runtime-file inventory. The exact quality entry point passed all
16 checks. Two complete v2 reports were byte-identical and retained every v1
numerical value. The result SHA-256 is
`b4170d0b7da7909aea6744dfd7ea4ec63744c7b28b4eece0e7c6ec48ef5928e7`.
This records lead closure of the final scoped gap, not another independent
review or a guarantee against all possible future semantic defects.
