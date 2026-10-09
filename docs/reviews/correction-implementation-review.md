# Independent correction implementation reviews

Reviewed source:0463144e3be7a5464cb7905c58d1f628bff70c08, tree-identical to
adc62aff32978085c7f0d29ebc576c271c5912b2. The metadata-only rewrite replaced a
private author email with the existing GitHub no-reply address after GH007;
source and protocol bytes were unchanged. GitHub protection was not disabled.

## Scientific fidelity (independent Astra)

Verdict: accepted subject to one pre-execution reporting correction. Formation
and trajectory calculations, noise, marginal/analytic controls, candidate
ordering, accounting and claim boundaries followed the frozen protocol.

Finding S1: root gate2 said pass from single-run checks alone. The protocol also
requires separate quality/fault-injection and deterministic replay evidence.
Use bounded_checks_passed for a valid single run and reserve complete gate2
adjudication for the evidence certificate. Invalid runs must still say fail.

Review also requires final development/confirmation provenance comparison,
with shared offline training explicitly intentional, and descriptive stream
ranges without population confidence claims. No corrected cohorts or tests
were executed by this reviewer.

## Regression/correctness (independent Astra)

Verdict: block on one P2 contract defect; no other concrete blocker found in
controls, trajectory semantics, cost/provenance, fingerprint enforcement or
CLI publication.

Finding R1: T2 solvers reused T3 finite/array parsing. Nonfinite owned ridge and
ragged owned rows raised InputDataError instead of ContractViolation; arithmetic
overflow from valid finite examples also escaped as InputDataError. This
misclassified owned-code/domain failures as malformed boundary data. Preserve
T3 behavior while checking owned solver preconditions and deliberately naming
recoverable numerical failures. Never broad-wrap unrelated exceptions.

Reviewer reproduced all three counterexamples and ran13 CLI/policy tests in
.566 seconds. No protocol cohorts, source edits or Git changes occurred.

## Disposition

S1: test_single_run_cannot_certify_external_gate_two_evidence failed on the old
pass label, then passed with bounded_checks_passed and an explicit replay
requirement. All eight CLI tests passed afterward. No mathematical change.

R1: separate tests reproduced six precondition/arithmetic misclassifications in
fit/_regress. Narrow entry-validation conversion now raises ContractViolation;
solver arithmetic checks raise FitUnavailable. Sixteen trust tests passed.
The same defect class was checked in analytic_template_fit: two finite-input
overflow probes failed first (ValueError from mixed infinities and OverflowError
from fsum), then passed with deliberate FitUnavailable classification. An
unrelated injected RuntimeError still propagates. The original summation and
ordinary numerical calculations remain unchanged. Existing paired-test ASTs
were not changed. Final full exact-source verification is recorded separately.

No corrected development or confirmation result was used to choose or modify
settings. These are lead/implementer closure checks of the review findings,
not a claim that the independent reviewers reran the complete final suite.
