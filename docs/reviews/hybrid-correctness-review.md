# Independent hybrid correctness and evidence review

Reviewed 2026-10-09: the bounded shared-family formation and same-feature
lifecycle implementation, its contract/plan, legacy `repair.py`, tests, and
the initial `hybrid-result.json` / `hybrid-runtime.json` artifacts. No code,
fixtures, hyperparameters or legacy source were changed by this reviewer.

Verdict: one P2 acceptance/reporting defect; no demonstrated leakage,
invalid measured result, incorrect deletion, or broader blocking defect was
found. The measured results support the explicitly bounded claims. This is
not approval for larger research or evidence of developmental advantage.

## Finding: P2 — ridge representability target is not an acceptance gate

In the reviewed source, `src/sultai/formation.py:428–434` accepts paired
improvement, healthy no-op preference, split isolation and finite metrics, but
does not check the contract's separate full-ridge representability target.
`hybrid.py` aggregates the supplied gates, so a failed positive control can
still produce overall success.

Concrete reproduction: an in-memory `unittest.mock.patch` wrapped
`candidate_bank`, preserving every candidate and its ledger except replacing
the `full_ridge` adapter with `Adapter.zero()`. `run_formation()` then returned
full-ridge and no-op mean test MSE both **0.08715691674750341**, while all four
formation acceptance gates remained `True`. This took 1.95 seconds under
`nice -n 10 timeout 120`; no source or fixture was modified.

The actual supplied ridge result is **1.3025525995494743e-20**, so this defect
does not invalidate current representability evidence. It makes automated
acceptance incomplete. Add an explicit ridge gate and a focused failed-control
regression test. The pre-existing positive-control test already uses a
`1e-12` threshold; reusing that tolerance requires no fixture or training
retuning. Disclose that the reporting gate was added after the initial audit.

The lead acknowledged this finding and assigned the correction. This report
records the original defect; verification of the correction and refreshed
artifact belongs to final integration, rather than a second broad review.

## Supporting checks and limits

- **Isolation:** offline teachers/regression read conditioning only. The
  generator API cannot receive selection or audit pairs. All candidate banks
  within a phase are formed before selection begins; each lineage completes
  its admission decisions before its final audit. Earlier audit outcomes do
  not affect later formation, admission or stopping. Lifecycle training
  receives audit inputs but not audit targets, and finishes every arm before
  scoring the saved predictions.
- **Controls:** retrieval uses training teachers; cross-lineage conditioning
  rotates within the declared evaluation phase without training on it.
  Residual mismatching preserves residual/input marginals while breaking
  their pairing. It intentionally retains information such as the bias
  component, so its residual performance is not evidence of leakage.
  Identity probes execute but add no evidence features, as disclosed.
  Each method has one candidate plus the shared no-op comparison, not a
  best-of-many generated search.
- **Accounting:** formation selection costs charge the shared baseline once
  per lineage and every actual candidate evaluation; both raw and admitted
  final predictions are charged. Conditioning/SGD costs explicitly use
  repeated per-method reads, not unique examples. The report distinguishes
  272 stored adapter values from four effective output coefficients, frozen
  generator parameters, retained teacher-bank parameters and offline fitting.
  Lifecycle reports equal host-update counts, common training pairs and
  unequal auxiliary updates/retained static capacity.
- **Removal:** the formed adapter is frozen; host changes occur through task
  gradients rather than weight copying/folding. `remove()` deletes the module
  attribute. Final serialization excludes its weights and has empty optimizer
  state. The formed arm drops from 544 allocated values to 272; the early
  retained control remains at 544. Removal occurs at zero influence, so the
  identical immediate pre/post-delete losses demonstrate structural deletion,
  not robustness to abrupt withdrawal. A separate frozen-host withdrawal
  diagnostic is retained.
- **Outcomes:** paired raw/admitted test MSE is **0.00033316524303110713**
  versus no-op **0.08715691674750341**. Eight acted lineages and zero observed
  harms do not estimate a population safety guarantee. All unfavorable
  comparisons remain visible: lifecycle no-growth post-learning MSE
  **5.387567850846157e-8** beats formed-taper **5.053203015285058e-6**;
  formed immediate-removal MSE is **0.003831594338787464**. The latter two
  satisfy the stated 0.01 mechanics target without establishing special
  developmental efficiency, unseen-family transfer, near-empty growth or a
  half-size reference result.
- **Provenance:** all five executable-file SHA-256 entries in the supplied
  result matched the inspected bytes; `git diff c720bce --
  src/sultai/repair.py` was empty. Initial result SHA-256:
  `af404afefbb28f97675bf0325294f36e15c9222bf53125d040d8f4af1a68fdfb`.
  Its separate runtime artifact reports 2.819542 seconds, 38,960 KiB maximum
  RSS and exit code zero. Source hashes bind executable content; they do not
  independently prove when research choices were made. Any corrected source
  needs a fresh bound result artifact.

Reviewer-executed verification: five existing focused tests passed in
**2.477 seconds**: formation training isolation, real selection/audit order
and executed query accounting, lifecycle audit-target poisoning, real deletion
and serialized counts, and exact task-gradient/no-copy behavior. The supplied
36-test full-suite result was reported by the lead, not rerun redundantly in
this review. No network, downloads, GPU, long run, tracker mutation or external
repository change was performed.

## Lead integration disposition (after independent review)

The Sol formation owner added `full_ridge_representability`, requiring every
held-out lineage's raw ridge MSE to be below the existing positive-control
test tolerance of `1e-12`. A targeted regression confirms a failed ridge
audit makes this gate false while the paired and healthy gates stay true.
No fixtures, training or numerical settings changed. The full 37-test suite
then passed in 6.597 seconds. Two final integrated runs passed all 16 gates
and produced byte-identical JSON; the saved result's SHA-256 is
`adae7d8ab70e6bd13ddd44dd3127468e3b0937f0cca7fa05b0a7d615559e3326`.
These statements record lead verification, not another independent code pass.
The original P2 finding above is resolved in this delivered implementation.
