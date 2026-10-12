# Proposed fixed-size evaluation repair — feedback incorporated

Status: **revised recommendation; not implemented or execution-authorized.**
John requested incorporation of both independent reviews and a walkthrough of
next steps. This revision fulfills the planning request. The original proposal
is preserved at commit `5751f52ae63d64d23f770831ff0f13a49099ecb0`.
Owner: `codex-sultai-gate3`, claim `sultai-ef0da96d8a`.

Both reviewers assessed both engineering and experimental method. Their
[independent findings and detailed control matrix](../reviews/gate3-headroom/fixed-evaluation-design-review-2026-10-12.md)
are the design acceptance requirements for this package. Incorporation into
this plan is not evidence that implementation has passed them.

The [diagnostic result](../certificates/2026-10-12-gate3-twin-diagnostic-result.md)
supports the observed batch-size mechanism, for one intact checkpoint/no-op on
card 0. It neither identifies a particular kernel nor proves adapter, host,
cross-card or concurrency behavior. Padding changes baseline floating-point
outputs itself. Treat the repair as a versioned evaluation change.

## 1. Implement one consistent frozen evaluator

After instruction to implement, change `src/sultai_screen/fit.py:losses`:

1. Require nonempty aligned read/base/labels under the owned-feature contract.
2. Traverse caller order in chunks of at most 128 real examples.
3. Duplicate the partial chunk's **last complete real example** across
   read/base/labels to exactly 128, preserving dtype/device/contiguous layout.
   No random draw or added data access.
4. Apply the adapter **after padding**, then classify and compute unreduced CE.
   Slice back to the real chunk length before returning losses.
5. Return exactly N losses in caller order. Preserve inverse-permutation
   alignment, double-precision mean over N and bootstrap over N real paired
   differences. Padding never enters statistical counts or optimizer steps.

The shared path must cover baseline, zero adapter, the full global twin,
`mean_loss`, `fit_sgd` selection, oracle paired SE, post-hoc arms, P5/P6 and
raw/selected cheap-candidate evaluation. Do not implement padding solely in the
twin assertion. Keep the full global-permutation **bitwise** assertion and stop
semantics; no batch-size-preserving substitute or tolerance fallback.

Keep host training, `gradient_step`, optimizer updates, training loss streams,
ridge/former fitting and cached feature extraction unchanged. Do not implement
padding in `Host.classify`, which serves gradients too. Historical stdlib
instrument behavior stays unchanged; both production roots remain covered by
gates and regressions.

Preserve failure distinctions: real no-op/selected nonfinite evaluation stops;
failed fits remain unselectable/unavailable; unselected raw development failures
remain recorded. Do not drop a failing real example, replace nonfinite values,
use development finiteness to select candidates or treat missing evidence as
success. Preserve unavailable schedules/all-bank evidence and the fixed-rate P5
unavailable-fit distinction.

Selection-only ranking, enumeration/ties, grids and the P6-only eta rule remain.
New scores can legitimately change close winners, plateau flags and downstream
results. Recompute all affected evidence. Never choose between evaluators by
favorable outcomes or retain old winners by fiat. The no-op delta does not bound
the scientific effect of a changed winner.

Deliverable: bounded evaluator patch and tests, with an explicit changed-call-
path inventory and evidence that excluded training paths remain unchanged.

## 2. Draft and enforce the amended protocol identity

Preserve the [frozen v4 protocol](2026-10-10-gate3-headroom-screen.md) bytes and
SHA-256 `7668d20ad55c3a62384975a7cf1f3989dbe69bf701ce02bc9c68ac2601d0d42a`.
Propose a versioned composition of that base and a separately hashed extension,
with precise precedence and a single active identity. The extension must state
the algorithm, all changed loss paths, selection consequences and real counts.
It becomes operative only with explicit disposition and exact-source review.

Implement mandatory identity validation in source provenance, admission,
supervisor/worker receipts, checkpoint/result validation and certificates.
Missing, old-only or mismatched extension identities must fail closed; a generic
evidence-file list is insufficient. Preserve clean-source coverage and hash/
ancestry checks. Do not broadly weaken aggregate-source checks for checkpoint
reuse. Checkpoint metadata may need versioning; old weight bytes stay immutable.

Keep X, rho, Hm, U, rates, schedules, data/splits, sizes, widths, arm definitions,
V0/V1 and SD-before-median ordering unchanged. Numerical replay exactness remains
distinct from paired-bootstrap sampling uncertainty. No new acceptance margin
is inferred from the diagnostic.

The base protocol's fit-code-change rule requires a new version/hash,
disposition and **fresh confirmatory seed range**. Draft compliance with that
rule and verify the next unused allocation before freeze; do not silently
retain an earlier allocation because no confirmation ran. Any exception needs
John's recorded disposition. No confirmatory seeds are executed by this package.

Deliverable: proposed extension plus enforced provenance contract, explicitly
marked pending activation; original sources/results and all failures retained.

## 3. Validate on CPU and review exact committed bytes

Use short, bounded, single-process CPU work with existing dependencies. The
[review control matrix](../reviews/gate3-headroom/fixed-evaluation-design-review-2026-10-12.md#cpu-acceptance-package-before-exact-source-review)
is mandatory, including:

- N = 1, 127, 128, 129, 255, 256, 257 and cheap synthetic 2000/3000/5000;
  exact multiples add no batch, both adapter/classifier observe 128, output is N.
- Tagged alignment, inverse permutation, real-count means/bootstrap, independent
  padded reference and unchanged full-batch behavior. Negative controls detect
  mislabeled examples, retained padding and incorrect denominators.
- Batch-shape-sensitive old-fail/new-pass mock plus real numerical fixtures;
  no/zero/nonzero affine, tanh and wide adapters at S1/S2a/S2b with distinct
  read/base. Detect padding after the adapter and bypass/base substitution.
- Frozen BN, parameters/buffers and absent host gradients; negative controls
  detect mode changes, state mutation and unwanted gradients.
- Real final-row NaN/Inf, raw/selected failure rules and unavailable-fit behavior.
- Controlled close candidates, selection-only ranking and tie behavior; detect
  development-based ranking or inherited old winners.
- Existing gradient known answer, optimizer/small-fit behavior, strict typing
  and policy/paired AST fingerprints where applicable, both-root coverage and
  quality-gate positive/negative controls, affected runtime tests and unchanged
  instrument regressions.
- Missing/wrong/base-only protocol identity and stale source/certificate
  rejection at relevant receipt boundaries, not only serialization round trips.

Use repository gates `scripts/check_quality.py` (which invokes policy checks)
and `scripts/check_policy.py` as required; record exact interpreter/tool paths,
commands, exits, duration and source identity. Retain complete failing evidence.
After checks, commit coherent source/protocol milestones and obtain independent
scientific and PyTorch/operations review of those exact bytes. Address material
findings, rerunning affected checks only as needed. Record a review certificate
that does not authorize GPU launch.

Deliverable: reviewable source/protocol package, CPU evidence and exact-source
acceptance. CPU checks alone do not establish GPU invariance or headroom.

## 4. Present a separately costed GPU acceptance and pilot proposal

Before any GPU instruction, present a fixed validation matrix and exclusive
custody design. Predeclare nonzero adapter fixtures and comparisons; cover
selection/development sizes, intact/bottleneck hosts, S1/S2a/S2b, no/zero/nonzero
affine/tanh/wide adapters, identical repeat/global permutation, both cards and
intended three-process-per-card co-tenancy. Use exact losses for exact assertions;
logits are diagnostic. Cover an actual width-2 host before accepting any later
width-2 outcome if that host was absent from engineering inputs.

Preserved hash-bound checkpoints can be proposed as engineering inputs. For a
pilot, recommend a fresh linked run unless a budget-saving provenance bridge is
explicitly approved and independently reviewed. Such a bridge must identify
original training source, unchanged dependencies, checkpoint/loss-stream hashes
and inherited versus newly measured evidence. Neither route relabels old
rankings, timings or failed attempts as new-source acceptance.

Remeasure affected costs including padding and required host/B3 fit replay and
timing admission before materiality, width or cheap outcomes open. Distinguish
real statistical counts from physical padded rows; do not assume wall time
scales linearly with row overhead. Fix ordering, ceilings and what happens if
the required coverage does not fit the remaining allowance before launch.

Retain all three prior custody receipts and cumulative **1473.267264504917**
process-seconds inside the unchanged **10800-second** aggregate ceiling:
**9326.732735495083 seconds** remain before further charges. Budget is not
authorization. Define separately bounded acceptance/pilot allocations, fresh
resource/owner admission, environment identity, deadline/cleanup reserve,
exclusive linked custody and complete raw/supervisor/GNU/outer exit evidence.
Do not reset receipts or create an attempt by changing the output path.

Every failed acceptance check stops for disposition: no tuned fixture, alternate
evaluator, tolerance fallback, automatic retry or budget increase. Confirmation
still requires its own freeze and instruction after a valid pilot. Frozen
section 9/D-P governs deviations. No schedule or unrelated workload changes.

## Feedback coverage and remaining decisions

| Feedback from the two reviews | Incorporation |
|---|---|
| Narrow diagnostic inference; padding changes baseline | Opening evidence qualification; versioned change in step 2 |
| Exact last-example algorithm, padding before adapter | Step 1 |
| Every loss consumer, including selection/oracle SE/cheap harm | Step 1 inventory |
| Training/gradient/cache exclusion; both-root meaning | Step 1; step 3 regression evidence |
| Finite, unavailable and missing-evidence distinctions | Steps 1 and 3 |
| Candidate ranking, ties, plateau and P6-only selection | Steps 1 and 3; fresh evidence in step 4 |
| N counts/alignment/means/bootstrap and physical overhead | Steps 1, 3 and 4 |
| Boundary/family/mock/reference/mutation controls | Step 3 and linked detailed matrix |
| Strict gates, fingerprints, exact-source independent review | Step 3 |
| Version/precedence/active identity; old-only receipt rejection | Step 2 and step 3 negative controls |
| Fresh confirmatory seeds; explicit exception disposition | Step 2; no exception chosen |
| Alternatives and no tolerance fitted to observed failure | Step 1; alternatives retained in independent review |
| Both cards/concurrency, sites/adapters, width-2 coverage | Step 4 |
| Host/B3 replay, new timing and outcome ordering | Step 4 |
| Checkpoint inheritance versus fresh execution | Step 4 proposal; operator choice pending |
| Immutable failures/custody, charged aggregate budget | Steps 2 and 4 |
| Predeclared fixtures, stop rules, no automatic rerun | Step 4 |
| Separate repair, GPU/pilot and confirmation decisions | Status and deliverables at each step |

All substantive review feedback is incorporated into requirements or an explicit
pending decision. No recommendation is counted as an implementation pass.
The next package to instruct is steps 1–3: repair, amendment drafting, bounded
CPU checks and independent review. Routine helper/test choices fall within that
contract. Step 4's allocation, checkpoint inheritance and any exception to
seed/version rules remain later material decisions. This revised plan alone
clears no implementation, GPU execution, pilot, confirmation or merge.
