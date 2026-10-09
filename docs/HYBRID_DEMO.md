# Hybrid TCD: bounded CPU demonstration

This page records protocol v1 at immutable commit
`ccdfd5225ee5575f365aa6a4a0698bed0f652958`. The current runner's v2 successor
adds strict types and required-state validation while retaining the numerical
experiment. See [TYPING_CONTRACTS.md](TYPING_CONTRACTS.md) for successor
verification and source-bound results. The v1 artifacts below remain unchanged.

The implemented hybrid demonstrates learned repair formation, empirical
admission/no-op and separate task-learning/taper/deletion mechanics. The
no-growth lifecycle control finishes with lower error than the formed-adapter
path. This is useful instrumentation, not evidence of a developmental advantage
or the final half-parameter objective.

## Reproduce

Python 3.10 or newer and the standard library suffice. No install, GPU,
network, external data, model checkpoint or paid service is needed.

```sh
cd /home/john/sultai
sh scripts/hybrid.sh --summary
sh scripts/hybrid.sh > /tmp/sultai-hybrid.json
nice -n 10 timeout 120 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The script lowers priority, limits numerical-library threads to one and imposes
an external 120-second timeout. A failed target makes the CLI exit nonzero;
a non-finite result raises an error. Do not silently extend the budget or tune
against held-out results. The legacy `scripts/smoke.sh` remains unchanged.

The full saved [JSON result](results/hybrid-cpu-2026-10-09.json) contains source
SHA-256 identities, fixed configuration, every lineage/control outcome and cost
ledger. [Runtime evidence](results/hybrid-runtime-2026-10-09.json) is separate
because elapsed time is not deterministic. No learned arrays, credentials or
private raw runtime logs are saved. The report's source identity covers the
five imported executable modules; Git records the launcher, tests and docs.

## Formation result

There are 32 offline training, 8 development and 8 held-out procedural
lineages in one shared public template family; each has 64 conditioning,
24 selection and 64 test examples. Eight paired healthy counterparts use the
test input process with a zero repair. These are manufactured episodes, not
independently trained host models. The adapter stores 272 values but the
generator's effective output has only four degrees of freedom.

Numerical settings and the planned comparison methods were fixed before the
first evaluation. No
hyperparameter selection uses development or test results. Correctness review
subsequently added the omitted conventional-ridge acceptance gate, reusing the
existing known-positive unit-test tolerance of 1e-12; it changed no model,
fixture, optimizer or measured result. The shuffled and mispaired controls were
corrected before the integrated run: cross-lineage conditioning derangement,
and residual mismatching that preserves the identity path and marginals.

| Method | Raw mean MSE | Admitted mean MSE | Acted / 8 |
|---|---:|---:|---:|
| `no_op` | 0.0871569 | 0.0871569 | 0 |
| `coarse` | 0.0822981 | 0.0762721 | 3 |
| `paired` | 0.000333165 | 0.000333165 | 8 |
| `paired_probes` | 0.000333165 | 0.000333165 | 8 |
| `full_ridge` | 1.30255e-20 | 1.30255e-20 | 8 |
| `affine_ridge` | 0.00193728 | 0.00193728 | 8 |
| `sgd_1` | 0.0349361 | 0.0349361 | 8 |
| `sgd_2` | 0.0187724 | 0.0187724 | 8 |
| `sgd_4` | 0.00830852 | 0.00830852 | 8 |
| `retrieval` | 0.0140799 | 0.0140799 | 8 |
| `interpolation_3` | 0.0135426 | 0.0135426 | 8 |
| `shuffled_conditioning` | 0.195178 | 0.0846627 | 1 |
| `mismatched_pairing` | 0.040112 | 0.040112 | 8 |

All eight paired repairs were admitted and none had higher held-out MSE than
no-op. Every healthy method selected no-op. These counts provide no calibrated
population harm-rate bound. The raw learned healthy repairs are nonzero, so
admission is doing useful work; a healthy no-op outcome does not prove a safe
formation mechanism. Each method offers one repair plus no-op, rather than a
best-of-K search.

Paired moments expose task-gradient information and direct oracle residuals
are privileged. The executed identity-host probes add no features and produce
exactly the same result as paired evidence. The conventional nonlinear ridge
control is much more accurate than the learned generator. The affine control
is informative but does not replace the nonlinear primary assay. Shuffling
conditioning degrades raw results; mispairing still preserves some useful
aggregate information and is not expected to erase every learnable signal.

## Lifecycle result

A 272-value host starts allocated at zero, trains for eight minibatches, then
receives a 272-value temporary repair formed from its current residual using
the frozen paired generator. Its initial error falls from 0.0302593 to
0.000983704 at insertion. The repair stays frozen; only task gradients update
the host. Influence holds for 16 updates and tapers for 48; deletion occurs
at update 72 and learning continues for 64 more updates. Each batch has
16 examples. The supplied residual is never projected back into the generator
family; host learning can take it outside that family.

Insertion here is raw and scheduled. Formation assay A separately evaluates
admission; lifecycle results do not establish safe admission. All six arms use
the same task-example stream and 136 host updates. Cold/random additions and
early static modules train through task gradients; the formed module is frozen.
Those differences are explicit controls, not a universal compute match.

| Lifecycle arm | Final MSE | Final parameters |
|---|---:|---:|
| `cold_zero_taper` | 1.64956e-06 | 272 |
| `early_static_retained` | 9.92948e-14 | 544 |
| `early_static_taper` | 3.74876e-06 | 272 |
| `formed_taper` | 5.0532e-06 | 272 |
| `no_growth` | 5.38757e-08 | 272 |
| `random_taper` | 1.80613e-06 | 272 |

For the formed arm, MSE is 0.00383159 at deletion and 0.00000505320 after
subsequent learning, below the predeclared 0.01 target. Zero influence just
before deletion already gives the same output as deletion; the proof of
removal is the missing object and serialized adapter field, not loss alone.
The final state has only `host` and empty `optimizer_state`, with 272 serialized
parameters. Tests check object disposal as well as state and count semantics.

The retained static arm has 544 parameters, so its result has a different final
budget. The early-static taper arm shares the 272-parameter final budget and
also beats the formed path. The target and host share features and could be
algebraically folded; this implementation learns through examples and never
copies teacher weights into the host. It demonstrates mechanics, not discovered
compression. A 544-to-272 drop from temporary peak size is not the user's
half-size claim against a tuned reference.

## Cost and validation

Formation uses 32 offline full-ridge teachers on 2,048 conditioning examples,
three small meta-regressions (28/44/44 learned coefficients), and a 32-teacher
retrieval bank with 8,704 stored values. Four public materialized templates
contain 1,088 values. These formation/search assets are distinct from the
272 retained values of any individual repaired site and must be counted if
kept in a deployed formation system.

Across development, test and healthy formation audits, the ledger charges
32,768 conditioning example uses including offline training, 1,792 executed
identity probes, 8,064 selection queries and 39,936 final audit queries.
Repeated reads, SGD passes and raw/admitted evaluations are counted, even when
they reuse cached synthetic examples. These are conceptual example-output
costs and executed loop counts, not external service calls or FLOP estimates.

The six lifecycle arms use 816 host gradient steps, 336 auxiliary gradient
steps and 13,056 task-example gradient uses, plus one formation call on
64 residual examples and 2,432 audit queries. Lifecycle consumes the already
trained generator; its offline cost is charged in the formation ledger.

All 37 unit tests passed in 6.597 seconds, including the original 12 checks.
The first integrated run took 2.82 seconds, 2.78 user CPU seconds and
38,960 KiB peak resident memory on Nyx. Two final runs after the acceptance
correction took 2.826 and 2.829 seconds and produced byte-identical JSON;
all 16 acceptance gates passed. These are load-dependent measurements, not
performance guarantees. Executable aggregate SHA-256:
`96913b8d85d7d6dfc4feddc4844d0d61db6e51f5462ba483c98654e6a9035dd9`.

The independent [design review](reviews/hybrid-design-review.md) and
[correctness review](reviews/hybrid-correctness-review.md) preserve their
findings. Full CLI results remain descriptive even when acceptance passes.
No result establishes diffusion, unseen-family/real-host transfer, a learned
removal policy, a population safety guarantee or growth from near-empty.

Before a larger stage, select a real task/host, observable telemetry, family
holdout, capability margin, grouped calibration and a concrete compute/stopping
budget. Source import blockers and recovery details are in [RESUME.md](RESUME.md).
