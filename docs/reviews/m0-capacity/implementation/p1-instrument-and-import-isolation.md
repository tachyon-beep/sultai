# Exact-source review: p1-instrument-and-import-isolation (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Scope: `src/sultai_screen/m0/instrument.py` (116 lines, new in b79f145..e1df6e1), `tests/test_m0_instrument.py::InstrumentTests`, `tests/test_m0_worker.py::WorkerTests::test_worker_imports_neither_the_gate3_screen_nor_its_pilot`, `tests/test_m0_pilot.py::SupervisionTests::test_import_isolation`. Compared against Gate 3 `screen.materiality_worker` (screen.py:231-249), `screen.posthoc_worker` (screen.py:252-303), `pilot.magnitude_decision` / `pilot.check_posthoc` (pilot.py:304-354), and the attempt-4 receipts in `docs/results/gate3-pilot-2026-10-13-linked-3/`. Tests were run (`PYTHONPATH=src`); mutations were applied to a scratch copy only. The working tree was not modified.

## Claims and evidence

| Claim (preregistration) | Source file:line | Pinning test | Verdict |
|---|---|---|---|
| §2 l.120: "**P1-only** materiality" | instrument.py:28-45: `materiality` loops `P1_LADDER` with family `"P1"` only and returns `{"rungs": {"P1": ...}}`. instrument.py:53-63: `materiality_report` is descriptive, gives per-rung mean gain and `clears`, and never chooses α. The P2/P3 ladders, the P3 replay and `magnitude_decision` are gone. | `test_materiality_is_p1_only_with_three_rungs` (fails if a family or rung is added); `test_materiality_report_is_descriptive` (fails if a chosen/alpha key appears, if the floor moves outside (0.049, 0.06], or if anchors are not distinct) | Satisfied |
| §2 l.120: P1-only "post-hoc" | instrument.py:66-101: one `make_impairment("P1", ALPHA, ...)`; arms A1, A2, A3, B3 only. P2/P3/P3′ and `derive_former`/`former_fits` (A7/A8) are removed. | `test_posthoc_is_fixed_alpha_with_four_arms` (fails on an extra arm, an `alpha` parameter, or α ≠ 0.1) | Satisfied |
| §2 l.120: P1-only "check" | instrument.py:105-116: `check_instrument` covers P1 at α = 0.1 only. There is no four-family loop and no former-agreement check. | `test_check_instrument` (each raise branch, plus distinct seeds) | Satisfied |
| §2 l.81-82 / §5 l.297: "A1, A2, A3, B3 at α = 0.1" | instrument.py:21 (`ALPHA = 0.1`) and :67, :72, :78-79, :84-90 | `test_posthoc_is_fixed_alpha_with_four_arms` asserts `ALPHA == 0.1` and the arm set | Satisfied. A2 is `nonlinear=True` and A3 `nonlinear=False`, as in Gate 3; this is not pinned (Minor 2). |
| §2 l.82 / §7 l.374: "B3 ≥ ρ · A3 gain, ρ = 0.8, on every development anchor" | instrument.py:99 computes the fraction (noop−B3)/(noop−A3), with `None` when noop ≤ A3, exactly as screen.py:286-288. instrument.py:114-116 raises `InstrumentHold` below `RHO`, with `RHO` from m0/identity.py:29 = 0.8. instrument.py:106 requires six distinct anchors. | `test_posthoc_b3_fraction_of_a3_and_its_none_branch` (formula and `None`); `test_check_instrument` (0.79 → hold, `None` → hold); `rho_required == 0.8` | Satisfied |
| §7 l.374: "A1 restores within 1e-4" | posthoc: instrument.py:73-76 (`InstrumentHold`). It is preceded in the same child by materiality's identical computation at instrument.py:41-43, which raises `PilotHold`. | `test_a1_drift_and_zero_module_identity_are_instrument_failures` (flaky, Important 2); `test_wrong_inverse_raises_pilot_hold` (cannot distinguish the class, Important 1) | Value satisfied. Classification defective (Important 1). |
| §7 V0: zero-module identity (the part in this item: `identity_check` inside materiality) | instrument.py:29-34 wraps `fit.identity_check` and re-raises `InstrumentHold` | Second half of `test_a1_drift_...` (`Adapter.apply` perturbed → `InstrumentHold`, "zero module is not bitwise identity") | Satisfied |
| §2 l.123: "Nothing retired feeds the P1 numbers" | instrument.py:14-18 imports only `contracts`, `fit`, `host`, `impair`, `m0.identity`. A fresh interpreter importing `m0.instrument` loads no `screen`, `pilot`, `former_bridge` or `alignment` (verified). All randomness goes through labelled `generator(seed, label)` draws (impair.py:47 "P1-channels", fit.py:79 "oracle-init", fit.py:283 "oracle-order"), so removing A7/A8 leaves every oracle draw unchanged. | `test_import_isolation` (InstrumentTests); it fails if instrument.py imports any of the four modules | Satisfied |
| §2 l.121-122: "`former_bridge.py` and `alignment.py` off the new workers' import path" | m0/worker.py imports are clean (verified: the fresh-interpreter module set contains none of the four). Children launch as `python -B -m sultai_screen.m0.worker` (pilot.py:178-186 with `module=WORKER`, m0/pilot.py:96), so the runtime path is the clean one. m0/receipts.py is also clean. | `test_worker_imports_neither_the_gate3_screen_nor_its_pilot` (fails if any of the four modules enters the worker's import closure); `SupervisionTests::test_import_isolation` (receipts closure; AST check that m0/pilot.py imports `sultai_screen.pilot` directly but not screen/former_bridge/alignment) | Satisfied for workers. m0/pilot.py and m0/confirmatory.py load all three transitively through `sultai_screen.pilot` → `screen` (pilot.py:45). That is permitted by the brief's exception; the docstring at Minor 4 overstates it. |
| Brief: "the P1 ladder" | instrument.py:20 `(0.1, 0.3, 0.03)` = screen.py:54 `MATERIALITY_LADDERS["P1"]` | `test_materiality_is_p1_only_with_three_rungs` (literal tuple) | Satisfied |

Test run at e1df6e1: all 8 InstrumentTests and both isolation tests pass on most runs. `test_a1_drift_and_zero_module_identity_are_instrument_failures` failed on unchanged code in 4 of about 9 process runs (Important 2).

## Frozen values checked

| Value | Prereg / origin | Source | Match |
|---|---|---|---|
| α (P1 magnitude for A1–B3) | 0.1 (§2 l.82, §5 l.297) | instrument.py:21 `ALPHA = 0.1` | Yes. Attempt-4 `02-magnitudes.json` also chose `alpha_low = 0.1`. |
| ρ | 0.8 (§2 l.82, §7 l.374) | m0/identity.py:29 `RHO = 0.8` (= contracts.py:44) | Yes |
| A1 inverse tolerance | 1e-4 (§7 l.374) | instrument.py:24 `INVERSE_TOLERANCE = 1e-4` (= screen.py:244, :267) | Yes. Not literal-pinned; the tests bracket it only at < 1e-3 (Minor 1). |
| P1 materiality ladder | Gate 3 (0.1, 0.3, 0.03) | instrument.py:20 | Yes; literal-pinned |
| Materiality floor | 0.05 nats. Not in the prereg text; inherited as Gate 3 "mathematics" (pilot.py:326, :345) | instrument.py:22 `MATERIALITY_FLOOR = 0.05` | Yes. Not literal-pinned (Minor 1). |
| Supervised known-answer tolerance (A3 − A1) | 0.01. Not in the prereg text; inherited (pilot.py:348) | instrument.py:23 `KNOWN_ANSWER_TOLERANCE = 0.01` | Yes. Not literal-pinned (Minor 1). |
| Development anchors | 6 (§8 l.470 "6 × 56 s") | instrument.py:25 `ANCHORS = 6` | Yes |

Gate 3 check_posthoc averaged the materiality and known-answer clauses over both α (0.1, 0.3). M0 applies them at α = 0.1 alone. This follows from the prereg's single α. Against attempt 4 the margins are wide: noop − A1 is 0.35–0.69, A3 − A1 ≈ 0, and the B3 fraction is 1.09–1.59 on all six anchors. No frozen value needs to move, so this is not amendment territory.

Unrequested behaviour (item 4):
- The descriptive 0.3 and 0.03 materiality rungs remain. Their inverse checks add hold paths that V0 does not name. Both are inherited from the Gate 3 P1 ladder and named in the brief, so they are not a finding.
- Removing A7/A8 does not change the RNG order, because every draw is label-seeded.
- instrument.py touches no Gate 3 path. Gate 3 `fit.py`, `host.py` and `pilot.py` did change in b79f145..e1df6e1, but those changes are outside this item and belong to other items' reviews.

## Findings

**Important 1: an A1 failure at α = 0.1 still publishes `execution-invalid`. The final-review M1 fix is incomplete.**
- *Where:* instrument.py:42-43 (`raise PilotHold(f"exact inverse failed 1e-4 nats: P1/{seed}")`); m0/worker.py:492-493; worker.py:1323; pilot.py:239.
- *What:* `materiality` at rung 0.1 computes the same quantity as posthoc's A1: same operator, same `undo(features(development).base)`, same intact loss. It runs first in `instrument_action`. A deterministic A1 failure therefore raises plain `PilotHold` in materiality, and posthoc's `InstrumentHold` at instrument.py:74-76 is unreachable for it. The child writes `software_valid: false`, the Supervisor reports "child exited", and the pilot verdict reads `execution-invalid` instead of `instrument-failed`.
- *Why:* Ruling R21 lists "A1/B3" among the instrument clauses that publish `instrument-failed`. Final-review M1 named this line (pre-fix `instrument.py:39`, confirmed at `4bb1c47^`). Commit 4bb1c47 wrapped `identity_check` and fixed posthoc but left this raise. The tests cannot catch it:
  - `test_wrong_inverse_raises_pilot_hold` asserts `PilotHold`, which `InstrumentHold` subclasses (contracts.py:56). Mutating the raise to `InstrumentHold` passes the test.
  - `test_a1_drift_...` exercises posthoc in isolation.
- *Fix:* Raise `InstrumentHold` at instrument.py:43, at least when `magnitude == ALPHA`; all rungs is simpler and matches R21's reading. Add a test that asserts the exact class, `type(error) is InstrumentHold`, for materiality's α = 0.1 inverse failure. Better still, drive the test through `instrument_action` and assert that `failure.json` has `instrument_valid: false`.

**Important 2: the A1-drift test is nondeterministic, so the 1e-4 `InstrumentHold` clause is not reliably pinned.**
- *Where:* tests/test_m0_instrument.py:24-30 (`fixture()`: `Host()` is constructed before `torch.manual_seed(0)`); :119-124.
- *What:* The fixture host's weights come from the process's random initial torch seed. Under the patched `undo` (+0.1), the drift ranged from 5.9e-6 to 4.3e-4 across four processes. 8 of 40 seeded host inits do not trip 1e-4. The test failed on unchanged e1df6e1 code in 4 of about 9 runs: "exact inverse endpoint drift" was not raised. The cause is process RNG, not cwd or test order.
- *Why:* A flaky test in the suite can mask or fake regressions. Mutation results on this test are confounded. The posthoc A1 clause's boundary (1e-4) is never tested deterministically. `test_wrong_inverse_raises_pilot_hold` uses the same fixture; the 0.3 rung makes it more robust there, and it did kill a 1e-3 mutation.
- *Fix:* Seed before `Host()` in `fixture()`. Pin the drift clause with a `mean_loss` side_effect, as the fraction test already does, at 1e-4 + ε (holds) and 1e-4 (passes).

**Minor 1: the tolerance literals are not pinned.** (instrument.py:22-24; test_m0_instrument.py)
- These mutations survive the whole file:
  - `MATERIALITY_FLOOR` 0.05 → 0.055
  - `KNOWN_ANSWER_TOLERANCE` 0.01 → 0.015
  - `>=` → `>` in `clears` (instrument.py:61)
- The tests only bracket the values: the floor lies in (0.049, 0.06] via the report and in (0.04, 0.5] via `check_instrument`; the known-answer tolerance lies in (0.005, 0.02]. `<` versus `<=` on ρ is unpinned at the exact boundary.
- *Fix:* Assert the three literals, and add exact-boundary cases (floor gain = 0.05 clears; A3 − A1 = 0.01 passes; fraction = 0.8 passes).

**Minor 2: the fit wiring is unpinned.** (instrument.py:78-80)
- Swapping A2/A3's `nonlinear` flags survives. So does swapping the oracle's conditioning and selection arguments.
- The fixture uses one dataset for all three splits, and `oracle` is mocked in every posthoc test.
- The source matches Gate 3 (screen.py:269-270, :283) by inspection.
- *Fix:* Build the fixture from distinct conditioning, selection and development tensors, and spy on `oracle`/`ridge` arguments.

**Minor 3: `check_instrument` does not verify `record["alpha"] == ALPHA`.** (instrument.py:105-116)
- It relies on the caller supplying α = 0.1 records. Today the only producer is `posthoc`, which hard-codes it, so the risk is low.
- *Fix:* Add a one-line equality check.

**Minor 4 (outside this item's source, about the isolation claim): the confirmatory docstring overstates isolation.** (m0/confirmatory.py:6-7)
- The docstring says the launcher "never imports `sultai_screen.screen`, `former_bridge` or `alignment`". Through `sultai_screen.pilot` (pilot.py:45 `from .screen import ...`), all three are in the supervisor's `sys.modules`. m0/pilot.py's docstring correctly says "itself".
- This is permitted by the brief's exception. Nothing from those modules reaches the P1 numbers, because `check_instrument` and `materiality_report` are pure over child records.
- *Fix:* Add "itself" to the docstring, and optionally a test asserting the supervisor's transitive set is exactly the permitted one.

No BLOCKING findings. No frozen value must change for the code and the document to agree.

## Dispositions (controller, 2026-10-14)

- Important 1: FIXED (item D1: materiality's inverse failure raises InstrumentHold, completing final-review M1 under R21).
- Important 2: FIXED (item D2: deterministic fixture seed; drift boundary pinned).
- Minor 1, 3, 4: FIXED (item D3). Minor 2: deferred, recorded.

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
