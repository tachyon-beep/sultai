# Gate 3 fixed-evaluation repair and protocol review

Status: **CONDITIONAL recommendation; design review only.** Both independent
specialists reviewed both the technical repair and experimental/protocol
consequences. John requested these reviews, not implementation or another run.
Reviewed clean source: `5a2066d4efd1667243530e8b52172afd343f520d`.
This record condenses their separate reports and records the lead's synthesis;
it is not an operative protocol amendment or an execution certificate.

Inputs: [proposal](../../plans/2026-10-12-gate3-fixed-evaluation-proposal.md),
[measured diagnostic](../../certificates/2026-10-12-gate3-twin-diagnostic-result.md),
[diagnostic receipts](../../results/gate3-twin-diagnostic-2026-10-12-01a125af/manifest.json),
and [frozen v4 protocol](../../plans/2026-10-10-gate3-headroom-screen.md).

## Independent PyTorch and operations review

Reviewer: `/root/gate3_torch_ops_panel` (Astra). **CONDITIONAL:** recommend the
bounded repair, protocol drafting, CPU validation and exact-source review;
GPU execution and another pilot remain blocked. Read source/call paths and
verified all 14 diagnostic receipt hashes. No edits, imports, tests, experiments
or tracker writes.

- The observed mismatch is upstream of cross-entropy and matches the 16
  examples changing classification batch size. Padding passes the saved no-op
  comparison, but itself changes eight baseline logits and three losses.
  This is a numerical implementation change, not retrospective pilot acceptance.
- Implement in shared `fit.losses`, padding **before** adapter application.
  `Adapter.apply` has batched operations too. Do not put padding in
  `Host.classify`: that function also serves training and differentiable fits.
- Selection is affected: `fit_sgd` calls `losses` at `fit.py:196`. Close
  candidates can switch winners, with downstream effects larger than the
  measured no-op delta. Preserve selection-only ranking and existing ties;
  recompute rather than inherit old winners or choose an evaluator by outcomes.
- A versioned complete successor or versioned base-plus-extension composition
  needs precise precedence and one active identity. An informal addendum with
  only the old protocol SHA in receipts is insufficient.
- The fit-code-change rule at protocol line 395 explicitly requires a new
  preregistration version/hash, disposition and fresh confirmatory seed range.
  No confirmation has run, but this clause still needs explicit treatment.
  Following the next-unused-range rule is straightforward; retaining an earlier
  allocation requires an operator disposition.
- Existing checkpoints may be inputs to separately authorized engineering
  checks. A later pilot needs new-source admission or an independently reviewed
  inheritance plan. The simplest defensible pilot is a complete linked run;
  budget-saving inheritance is a material decision.

## Independent scientific and experimental-method review

Reviewer: `/root/gate3_science_panel` (Astra). **CONDITIONAL:** recommend shared
128-example evaluation and a versioned protocol extension. A repair/CPU/review
package can be approved next; current evidence does not clear GPU execution or
another pilot. Inspected proposal, protocol, source and saved evidence. No edits,
imports, tests, experiments or tracker writes.

- Enumerate every frozen-loss consumer, including oracle paired-SE inputs and
  cheap/harm evaluation, before implementation. Candidate selection makes this
  a change to experimental computation, not merely presentation.
- Fix deterministic padding source, slicing, real counts and finite handling
  before new outcomes. Mathematical datasets/endpoints can remain the same
  while floating-point evaluation changes. Paired bootstrap uncertainty does
  not excuse failure of the exact numerical replay assertion.
- Bind base protocol plus extension, exact source and reviewed evidence through
  admission, workers, checkpoint validation and results. A prior-protocol-only
  receipt must not certify the repaired evaluator.
- Old candidate rankings, plateau evidence, scientific outputs and timings are
  not new-source evidence. Unchanged host checkpoints are potentially reusable
  inputs only through a reviewed provenance bridge; otherwise retain the
  current fresh-run rule. Do not generally loosen the aggregate source guard.
- Later GPU acceptance must cover intact/bottleneck hosts, all sites, adapter
  families, both cards and intended concurrency before scientific decisions.
  Width-2 GPU coverage is required before accepting a width-2 outcome if those
  inputs were unavailable to the earlier engineering matrix.

## Common recommendation and alternatives

| Choice | Both reviews |
|---|---|
| Shared fixed-128 evaluation | Recommended, conditional on the specifications and validation below. Retains the full global-permutation assertion. |
| Batch-size-preserving twin | Legitimate weaker protocol, not an equivalent repair. Leaves endpoint batch-size sensitivity in place. |
| Numerical tolerance | Not justified by this single observed maximum. Needs independently justified, predeclared bounds and failure rules. |
| Stop/archive | Scientifically valid; retains the failed instrument finding and consumes no more resources. |

Neither review establishes a particular CUDA kernel cause or general invariance.
The diagnostic covers one intact-3000 checkpoint, no adapter, card 0 and one
permutation. It provides no scientific headroom finding.

## Proposed implementation contract

1. Require nonempty aligned read/base/label arrays under the owned-feature
   contract. Traverse real examples in caller order in chunks of 128.
2. For a partial chunk, duplicate its **last complete real example** across
   read/base/labels to exactly 128. Preserve dtype/device/layout; draw no RNG.
3. Apply the adapter after padding, classify, and compute unreduced CE. Retain
   only the first real-count losses, returning exactly N losses in caller order.
4. Preserve inverse-permutation alignment, double-precision mean over N and
   bootstrap over N aligned real differences. Padding adds no statistical
   observations, conditioning pairs, candidates or optimizer steps.
5. Preserve frozen BN/state/gradient checks and existing finite/failure policy.
   A real failing example cannot be dropped or replaced. Failed fits remain
   unselectable; raw unselected development failures remain reported; no-op and
   selected evaluation failure stops. Missing evidence remains an error.

Change the shared frozen evaluator at `src/sultai_screen/fit.py:44`, covering
`mean_loss`, baseline/zero/full-global-twin, `fit_sgd` selection, oracle paired SE,
post-hoc arms, P5/P6 and raw/selected cheap-candidate losses through their existing
consumers. Preserve candidate grids, ordering/ties and selection-only decisions.

Exclude `gradient_step`, host training, optimizer updates, training loss streams,
ridge/former fitting and cached feature extraction. Preserve the historical
stdlib instrument. Both roots stay covered by gates/regressions; both do not
need a padding implementation.

## CPU acceptance package, before exact-source review

| Area | Required evidence and controls |
|---|---|
| Shape/counts | N = 1, 127, 128, 129, 255, 256, 257; cheap synthetic fixtures at 2000/3000/5000. Both adapter and classifier see 128, result has N, exact multiples add no batch. |
| Alignment/weights | Tagged distinct read/base/labels, arbitrary permutation/inverse, independent padded reference, mean/bootstrap use N. Negative controls misalign labels, retain padding or divide by padded count. |
| Mechanism/families | Shape-sensitive mock reproduces old failure and repair pass; real numerical fixtures; no adapter, zero, nonzero affine/tanh/wide at S1/S2a/S2b. Detect padding after adapter, omitted padding and bypass/base substitution. |
| Frozen ownership | BN eval, parameter/buffer hashes unchanged, no host gradients. Mutation, BN-mode and gradient negative controls fail. |
| Failures | Real final-row NaN/Inf remains visible. Preserve raw versus selected failure distinctions, unavailable schedules/all-bank evidence and fixed-rate P5 unavailable-fit behavior. |
| Selection | Controlled close candidates demonstrate new selection scores determine winner; ties/order unchanged; no development-based selection or old-winner inheritance. |
| Training separation | Existing gradient known answer and affected optimizer/small-fit checks pass; no padding enters differentiable training. |
| Quality | Strict typing for both roots, policy and applicable paired AST fingerprints, positive/negative controls, affected runtime tests and unchanged instrument regressions. |

These are acceptance requirements, **not tests run by this design review**.

## Versioning and later execution

Preserve frozen v4 bytes and failed-attempt records. Draft a versioned extension
with precise precedence and a single active identity binding base and extension
hashes. Bind it throughout admission, workers, checkpoints/results and reviewed
certificates, including rejection of absent/mismatched extension identities.
Retain the exact full-global twin, all scientific margins, real sample counts,
grids, schedules, arm definitions, V0/V1 logic and SD-before-median ordering.
Resolve the protocol's fresh-confirmatory-range clause explicitly.

A later GPU proposal must predeclare fixtures, acceptance comparisons, resource
admission, deadlines, cleanup and custody before execution. Minimum coverage:
selection/development tail sizes; intact/bottleneck hosts and sites; no-op, zero
and predetermined nonzero affine/tanh/wide adapters; repeat/global permutation;
both cards and intended three-process-per-card co-tenancy; actual fit replay;
and new required host/B3 timing admission before materiality/width/cheap outcomes.
Record logits diagnostically; require exact losses where exactness is specified.
Failure stops without tolerance fallback, fixture tuning or automatic retry.

Remeasure affected costs, including physical padding, rather than relabel old
timings. Real statistical counts stay distinct from padded physical work.
Checkpoint reuse requires either a fresh linked pilot or an explicitly reviewed
bridge identifying inherited training source, dependencies and artifact hashes.
The current aggregate-source guard rejects different-source host reuse; retain
that protection until an explicit successor contract is reviewed.

All later GPU work retains three custody receipts and the cumulative
**1473.267264504917 process-seconds** within the unchanged **10800-second** cap;
**9326.732735495083 seconds** remain before new charges. Remaining time grants
no attempt. A new authorized stage needs exclusive linked custody and complete
terminal evidence. Confirmation still needs its own reviewed freeze/authorization.

## Lead synthesis and decision boundary

The reviewers agree on direction; their additional requirements are compatible.
The next concrete package is shared-evaluator repair plus precise versioned
protocol drafting, bounded CPU controls/gates and independent exact-source
review. That package awaits John's instruction; this review does not implement
it. Tail duplication, helper layout and test organization are routine engineering
choices within that contract.

The later material decisions are GPU acceptance/pilot scope and allowance,
fresh execution versus explicit checkpoint inheritance, and any exception to
protocol seed/version rules. Present those only after the source/protocol package
is concrete and costed. No production code, frozen protocol, proposal, prior
receipt, run or schedule was changed by this review milestone.
