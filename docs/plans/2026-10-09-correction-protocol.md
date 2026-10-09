# Hybrid correction protocol and HTTYE gate plan

Status: independently reviewed and frozen before corrected execution.
Base: `88b2b4b6dffb6ad315443b38e29638b7aae87ec8`. John explicitly authorized
the correction, normal PR/merge workflow and bounded HTTYE programme. Cloud
work is limited to short CPU fixtures; the parent coordinates any Nyx burn.
Original snapshots, research imports and result artifacts remain immutable.

## Purpose and interpretations

Fix false confidence in the instrument, not optimize the demonstrator until
it wins. Learned formation, admission, physical removal and subsequent recovery
are separate claims. A host recovering eventually cannot establish useful
formation, developmental benefit or the half-retained-parameter objective.
Six later Claude probe findings are supplied by John; their raw scripts/results
are not present. The earlier committed R1–R4 boundary audit is a distinct review.

The source Page was read on 2026-10-09 at sequence 2 and preserved in
`2026-10-09-httye-source-page.md`. Its hardening-in-progress labels are stale;
the seven gates and their scientific demands remain authoritative intent.
The Simic PDR0057 certificate at e334a0d is a presentation exemplar only.
No Simic implementation or experimental result is transferred as Sultai evidence.

## Frozen bounded correction design

Keep the public four-template nonlinear adapter family, 16 channels and 272
stored parameters. Keep conditioning/selection/audit sizes 64/24/64, teacher
ridge 1e-8, meta ridge .01, admission improvement 1e-10. Preserve v1/v2 reports.
Introduce a separately versioned correction report and command, promoted as the
current bounded instrument. Existing legacy tests remain regression evidence.

### Formation, telemetry and admission

Offline training seeds: 1000..1031. New development episodes: 4000..4007;
confirmation: 5000..5007. Independent clean-healthy dev/confirm seeds:
14000..14007 / 15000..15007. Independent noisy-healthy dev/confirm seeds:
24000..24007 / 25000..25007. All are one shared procedural family, not
independently trained hosts or unseen-family transfer.

Construct and validate all cohorts before fitting. Require unique sample IDs
within each episode and globally disjoint IDs/lineages across every cohort,
including clean/noisy healthy. Record each cohort's lineage IDs, counts and
ordered sample/pair digests. Legacy formation also checks its healthy cohort
before fitting; historical numerics remain unchanged.

Retain all existing methods and their information/cost disclosures. Add:

1. Analytic four-template least squares using the empirical 4x4 Gram system
   and supplied conditioning residuals, regularization 1e-8. No learned map
   or intercept. This is conventional per-episode fitting, not forward-only
   formation. Count its fits and conditioning uses separately.
2. A marginal-only learned former using 96 features: per-channel first/second
   moments of h, tanh(h) and residual. Same offline teacher coefficients and
   meta ridge as the paired former. No input/residual cross-products; residual
   permutation must leave these features unchanged within numerical tolerance.
   This is a stronger marginal control, not an optimal marginal estimator.
3. Explicit zero, negated paired, and one norm-matched random paired candidate,
   sharing candidate count and selection access. Randomness uses a fixed
   independent seed derived from SHA-256 of role and fixture identity, never
   observed outcomes. Draw a Gaussian direction in four template coefficient
   coordinates and scale to the formed parameter norm; zero norm produces zero.
   Record seed derivation and role/partition tags in the report.
4. Noisy healthy identity targets with independent zero-mean Gaussian noise
   sigma .1 per output in conditioning and selection. Clean latent-identity
   audit targets establish whether an admitted repair harms the healthy task.
   Noise uses independent streams by partition; no final audit labels enter
   fitting or admission. Preserve false admission/rejection, no-op and harm
   counts and all raw/admitted losses, not only favorable means. With clean-audit
   improvement d=no_op_loss-raw_candidate_loss, helpful means d>1e-12, harmful
   means d< -1e-12, neutral otherwise; identify exact null separately. False
   admission means admitting harmful or neutral (separate category counts);
   false rejection means rejecting helpful. These are finite-fixture labels.

Fit/freeze candidates before selection. Charge conditioning use, teacher and
meta fits, per-episode analytic fitting, selection and audit calls and retained
assets. No noisy-harm count is a required zero. Existing exact clean-healthy
no-op and analytic in-family representability are known-answer controls.

### Lifecycle and trajectory

Use the historical planted coefficients (.25,-.20,.15,.10), 272-value host,
eight warmup, sixteen hold, forty-eight taper and sixty-four recovery updates;
batch size16. This remains one same-feature task. Stream seed roots are dev
8000,8010 and confirmation9000,9010,9020,9030; use root+1/+2/+3/+4 for
training/conditioning/audit/random respectively. These streams are independent
replicates conditional on this one task, not independent task-family evidence.

Arms retain the six historical names and behavior. Add no_growth_lr_matched
(host LR1.6) and frozen zero/negated/norm-matched-random replacements of the
formed adapter, using the identical insertion/hold/taper/deletion schedule.
Do not confuse these frozen replacements with existing trainable cold/random
additions. Static retained (two trainable copies, LR.8, alpha1) and host-only
LR1.6 must agree in function space at every recorded step within1e-12. Do not
claim analogous exact equivalence for changing-alpha static taper.

The primary diagnostic is the arithmetic mean of audit MSE at65 completed-step
boundaries8..72, including immediately after insertion at8 and after deletion
at72. This fixed pre-recovery trajectory metric cannot be improved by the
subsequent64 recovery updates. Also record immediate insertion change, immediate
removal MSE, maximum trajectory harm, and final recovery MSE separately.
Per comparator, maximum harm=max(0,max_t(formed_loss(t)-comparator_loss(t)))
over boundaries8..72. Summaries retain mean/min/max and counts above/below/
within1e-12, plus every per-stream value; no population confidence interval.

Report paired per-stream AUC contrasts against no_growth, no_growth_lr_matched,
early_static_retained, early_static_taper and all three frozen replacements.
Diagnostic specificity requires insertion improvement and strict AUC improvement
>1e-12 against every named comparator in every reported stream. This epsilon
is only a numerical discrimination tolerance, not an owner-approved worthwhile
benefit margin or calibrated population result. Pre-removal specificity cannot
establish post-removal benefit. Zero must exactly reproduce no_growth and cannot
satisfy strict superiority. Negating an exact planted repair must fail noiseless
insertion usefulness; negating a learned candidate is an experimental outcome
because the learned candidate may be wrong. Random candidates
may occasionally help; retain that result. Endpoint<=.01 is descriptive only.

No final test values select candidates, rates, schedules, seeds, contrasts or
stopping. Fit the frozen paired generator on the training cohort once; lifecycle
uses only its callable formation API. Log example/gradient/parameter-step counts
and all retained/search assets; equal update counts are not equal compute.

## Trust and failure contract

Tiers are relative to operations. T1 owns type and operation-specific values;
T2 owns type but documents specific recoverable domain failures; T3 trusts
neither and must validate/coerce explicitly. `.get()` is prohibited in T1/T2;
the production gate may conservatively prohibit it everywhere. Every T3 public
boundary references a specific paired test and its normalized AST fingerprint;
the test exercises coercion/type rejection and operation-specific value
validation. Fingerprint drift fails a gate and does not silently rebless tests.
Trusted-code faults propagate distinctly from recoverable domain/data failures.
Strict mypy covers every production module with no Any/cast/ignore escapes.
Do not create a generic analyzer or import unrelated ELSPETH infrastructure.

## Execution and stop rules

Independent scientific/design review must accept this interpretation before any
corrected cohort is run. Commit the reviewed protocol before corrected results.
Unit tests use separate small known-answer fixtures, never confirmation seeds.
One development assay validates the instrument; it is not a threshold search.
After source freeze and independent regression review, run confirmation once
and an exact replay only for reproducibility. Record source/plan/data/runtime
identities and each command/exit. Corrections to bugs invalidate affected prior
evidence and require an explicit disposition; never silently recycle it.

Each cloud validation is single-process stdlib CPU, nice10, timeout120 seconds,
no GPU/paid service. This is the existing authorized per-check ceiling, not a
new Nyx campaign budget. Stop on timeout, nonfinite metrics, missing evidence,
failed provenance or reproducibility mismatch. Do not extend timeout, discard
failures or alter confirmatory settings to continue. Software validity and
scientific outcomes are separate fields; a valid negative report is publishable.

## HTTYE gates and final certificate

1. Contract hardening: v2 delivered; new tier/provenance paths require regression
   and quality evidence. Never downgrade the historical result to in-progress.
2. Instrument truth: known answers, negative controls, leakage injections,
   rate-equivalence, real deletion, complete accounting, deterministic replay.
3. Useful repair headroom: report corrected within-family comparisons. Beyond-
   template transfer remains unproven; no later gate inherits a synthetic pass.
4. Honest admission: report clean/noisy outcomes and false admissions/rejections;
   small synthetic results do not constitute calibrated safety.
5. Benefit after removal: report predeclared trajectory and immediate/final
   outcomes versus all named controls. A failed contrast is a scientific negative.
6. Calibrated safety/utility: blocked pending target domain, observable telemetry,
   worthwhile margin, tolerated degradation/risk and independent replication plan.
7. Training readiness: issue a versioned evidence certificate with all prior
   verdicts, source/plan/run/data identities, reviews/dispositions and reproducible
   commands. Completing the certificate does not mark failed/blocked gates passed
   or authorize downstream open-ended training. Certificate completion is a
   separate status; gate7 stays not established while required predecessor
   scientific requirements are failed, blocked or unproven.

Nyx handover will name exact source/command, seeds, host/data/action family,
CPU/GPU/RAM estimates, concurrency, evidence paths, stops and restart behavior.
A broader burn's domain, utility/risk margins and resource ceiling are currently
missing decisions. Parent coordinates sync preserving local work and ELSPETH
priority. No broader burn is silently substituted with this synthetic check.

## Implementation ownership and checks

- Lead: this protocol, integration/report CLI, legacy healthy guard, source
  identity, docs/Page change summary, certificate, Git/PR/merge.
- Formation worker: `formation_controls.py`, `test_formation_controls.py`.
- Lifecycle worker: `trajectory.py`, `test_trajectory.py`, narrow rate parameter
  addition to lifecycle `_sgd_step` only if necessary.
- Trust worker: `trust.py`, `test_trust.py`, `scripts/check_policy.py`, boundary
  annotations (coordinate before editing shared production modules).
- Independent Astra: scientific/design review before assays; final independent
  regression and evidence review before completion.

Test-first regressions must fail for the missing behavior, then pass the fix.
Required full suite and `scripts/check_quality.py` run on frozen final bytes.
Mypy1.19.1/Ruff.15.4 are isolated development tools, never runtime dependencies.
Draft PR first; ready/merge only after exact-head checks and review dispositions.
No protection changes or bypass. Remote main is presently unprotected; report
that fact and still enforce the same local reviewed-head discipline.
