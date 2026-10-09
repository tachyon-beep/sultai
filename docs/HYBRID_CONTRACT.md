# Hybrid technical concept demonstrator contract

The current correction extension is frozen in
[the reviewed correction protocol](plans/2026-10-09-correction-protocol.md).
It adds complete healthy provenance, matched effective-rate and analytic/marginal
controls, noisy admission, frozen bad-repair controls and a pre-recovery
trajectory diagnostic. The contract below remains the historical v2 assay;
its numerical records are preserved. Current report validity is separate from
scientific success, and a completed evidence certificate can record NOT READY.

Protocol v2 adds strict model/report boundaries and explicit validated split
identity. It preserves the v1 numerical experiment at `ccdfd52`. Required
measurement, gate, coverage or provenance omissions are invalid reports, not
failed targets or no-ops. See [TYPING_CONTRACTS.md](TYPING_CONTRACTS.md) for the
separate hardening and verification record.

Status: implementation contract, 2026-10-09. John authorized this bounded
implementation after independent comparison. It supersedes narrower prior
Sultai execution limits, not the unchanged Simic programme. See
`plans/2026-10-09-hybrid-tcd.md` for ownership and validation.

## Source designs and role boundary

The **emmy design** is the original Sultai design/code at `c720bce`, whose
runtime began at `6d655e4`. The **Claude design** is the independently authored
report and subsequent Emrakul/Tamiyo revision preserved under
`research/claude-handoff/` in `72f9eeb`. The **hybrid design** is this contract.
Neither source is silently rewritten. Sultai Emrakul forms and admits an edit
or no-op; Tamiyo nominates an opportunity using coarse evidence. This TCD fixes
that nomination and does not train Tamiyo. Simic's different names/authorities
are unchanged. No new Library download is needed.

## Resource and evidence limit

Python standard library only, one CPU process at lowered priority, no network,
GPU, installed dependency, dataset download or paid service. Each demo has a
120-second external timeout. Stop on a timeout or non-finite result and inspect;
do not silently increase the budget. No statistical power or safety guarantee
is inferred from these small procedural fixtures. Training/search cost is
reported separately from retained parameter count; it is not the final-size
objective. The large Claude corpus/pilot and illustrative percentages/counts
are not approved work.

## Assay A: learned formation with held-out synthetic lineages

The primary adapter remains `h + alpha * (W tanh(h)+b)` with 272 **stored**
parameters. A fixed public four-template coefficient family generates planted
repairs in shared host coordinates. The learned generator is restricted to
that family; its effective output dimension is four despite a materialized
272-parameter adapter. Independently seeded procedural host episodes are the
lineages. There are 32 training, 8 development, and 8 held-out evaluation
lineages, with 64 conditioning, 24 selection and 64 final-test examples each.
Healthy controls use the same input process and a zero target repair.

This is **seen-family synthetic lineage holdout**, not independent trained
host transfer or unseen-family generalization. Existing stronger family
separation remains a later research contract. Seed labels must never be used
as evidence or to pretend the shared template family is disjoint.

Offline conventional teacher fitting supplies training targets. A small
regularized deterministic regression maps permitted evidence to repair
coefficients and is frozen before held-out formation. Held-out formation
performs no task-parameter fitting in this method. Direct oracle residual
observations are a declared privilege. Paired moments contain gradient
information; this is not a strict task-gradient-free result.

Evidence conditions:
- Coarse: fixed aggregate activation/residual/loss statistics without pairing.
- Paired: fixed paired cross-moments along the public basis functions.
- Paired plus fixed identity-host probes: the same pairs and an executed,
  charged fixed probe bank. Identity probes need not help and must not be
  presented as extra identifying information.

Controls: no-op, conventional full ridge, zero-start SGD at 1/2/4 steps,
nearest-neighbor retrieval, three-neighbor interpolation, shuffled conditioning,
and mismatched input/residual pairing. Shuffling deranges entire conditioning
sets between lineages within the same phase. Mismatching rotates residuals
and reconstructs targets as `h_i + residual_j`, preserving identity and
residual marginals. All phase candidate banks are frozen before
selection. Hyperparameters are fixed or chosen only on development lineages.
No test-dependent retries/tuning. Report raw and admitted outcomes, per lineage,
and all unfavorable comparisons. A single candidate plus no-op is K=1, not
best-of-K sampling. Any larger search must be explicitly counted.

Admission compares candidate and no-op on selection examples under a fixed
improvement threshold, preferring no-op on ties. Test labels cannot affect
formation, hyperparameters, admission or stopping. Report selection queries,
final audit queries, acted cases and realized harms. This is an empirical
admission mechanism, not a calibrated harm-rate guarantee.

## Assay B: learning, taper and actual removal

Use a trainable host already allocated 272 weights in the same nonlinear
feature class. Add a separate 272-weight temporary adapter after a fixed
initial training period. Form it from the residual task at the current host,
using the frozen paired generator from assay A. Freeze the temporary adapter;
train the host only through task examples, taper influence by a predeclared
schedule, delete the adapter object/weights and continue host training. Never
copy, algebraically fold or directly assign teacher weights into the host.

All arms share initial host, future examples and fixed update count. Include
no growth, same-time cold/random addition, early static additional capacity,
and early static followed by the same taper/removal schedule. Report unmatched
retained budgets explicitly; the last control matches final size. Include a
frozen-host withdrawal diagnostic or abrupt removal if useful and inexpensive.
Report loss before/after insertion, during taper, immediately after deletion,
and after subsequent learning. Stopping/schedule decisions use no test labels.
Insertion in this assay is raw and scheduled; assay A separately tests
admission/no-op. No safety or admission claim follows from assay B.

This is a **planted same-feature handover mechanics** assay. Its host and
adapter weights could be algebraically combined, but this experiment must
actually learn through examples. It is neither growth from near-empty nor
proof of discovered compression or a special developmental advantage. Count
zero-initialized allocated weights from step zero. Deletion is structural:
serialize the final model without the adapter; optimizer state for it must be
absent. Losing 272 temporary weights from a 544-weight peak is not the user's
half-size success against a tuned reference.

## Separate diagnostic controls

An affine residual is a labeled control. Demonstrate zero-bias folding into a
following linear map; do not equate that with added representational capacity.
Nonzero bias requires boundary semantics for convolutional folding.

The signed-shadow ambiguity fixture verifies identical passive evidence and
`A(+epsilon) = B(-epsilon)`. With a nonlinear downstream response, finite
changes need not be opposite; include a counterexample. Do not claim all
probe choices distinguish every host. These diagnostics are separate from
the learned repair corpus and lifecycle evidence.

## Acceptance and negative readings

Required correctness: finite deterministic outputs; original 12 checks remain
passing; source/fixture immutability; real non-affinity; disjoint lineage and
sample IDs; selection/test isolation; fixed candidates; correct cost ledgers;
healthy no-op preference; affine-folding/probe identities; actual removal.

Demonstration targets, fixed before final evaluation: learned paired formation
reduces held-out mean MSE versus no-op on the planted family; conventional
ridge establishes representability; healthy controls choose no-op; lifecycle
post-removal MSE is at most 0.01 with final adapter absent. Report any failed
target as such. Relative wins over retrieval/SGD/static are outcomes, never
requirements to retune the held-out fixture. No result implies diffusion,
beyond-first-order information, a population safety rate, unseen-family
transfer, a learned removal policy or half-parameter goal attainment.

Review correction after the initial audit: the CLI now explicitly gates ridge
representability at raw MSE below `1e-12` on every held-out lineage, using the
already-existing known-positive test tolerance. This closes an omitted success
condition; it does not change the task, training or hyperparameters.

Before any larger experiment: define a real host/task, permitted observable
telemetry, family split, capability margin, cluster-aware calibration, and
an approved budget/duration/stopping rule. Emrakul outcome/status quality
must be demonstrated before using it to train Tamiyo; staggered training does
not depend on an invented readiness percentage or intervention count.
