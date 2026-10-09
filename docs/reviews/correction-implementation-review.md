# Independent correction implementation reviews

Reviewed source: 0463144e3be7a5464cb7905c58d1f628bff70c08, tree-identical to
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

Reviewer reproduced all three counterexamples and ran 13 CLI/policy tests in
0.566 seconds. No protocol cohorts, source edits or Git changes occurred.

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

## Independent R1 closure

The regression reviewer accepted the targeted fix at
64439d80bdd7897aa94f8aa819ab23c17fcaffba. Six targeted tests passed in
0.011 seconds, including the original counterexamples, unrelated-fault
propagation and analytic known-answer behavior. No remaining concrete blocker
was found within R1 scope. No cohorts were executed or files changed. This
closes R1 without implying a second broad review or an independent full-suite run.

## Independent scientific evidence review

Accepted for merge at experiment source
64439d80bdd7897aa94f8aa819ab23c17fcaffba, report SHA-256
0fab1abb37b5d6a15c4082f154978e6c8b86f62503e6ed71cc6232f5dc9de97e.
The reviewer verified commit/tree, protocol and all twelve production hashes;
independently recomputed admission classifications, formation summaries,
aggregate costs, 65-boundary means, comparator differences, maximum harms,
specificity and fixed query/update counts from saved JSON. All agreed.
No corrected experiments were run by the reviewer.

Scientific disposition: accept the instrumentation correction; STOP programme
escalation and NOT READY for downstream training. Paired formation helps within
the planted family, but analytic fitting is substantially more accurate. Noisy
healthy harmful admissions persist. The formed trajectory loses to LR1.6
no-growth and static retained in both streams; deletion and final recovery
remain worse than no-growth. Software-valid true, specificity false and
training-ready false are appropriate.

Gate 1 relies on separate engineering verification, outside this numerical
review. Gate 2 development checks pass but replay is pending; gate 3 transfer
and learned advantage are unestablished; gate 4 retains noisy false admissions;
gate 5 is unfavorable/unestablished; gate 6 blocked; gate 7 not established.
Nyx confirmation and replay may complete the predeclared evidence closure with
unchanged settings. They must not become a search for a favorable result.
No scientific blocker to merging the correction remains.
