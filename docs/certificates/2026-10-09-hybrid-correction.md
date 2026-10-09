# HTTYE correction evidence certificate — development disposition

Date: 2026-10-09. Decision: **STOP programme advancement / NOT READY for broader
training.** The bounded correction is reviewable and its development evidence
is valid. A complete record of failed, blocked and unproven gates is not a
scientific pass. Nyx confirmation/replay is pending; this is the cloud handover
certificate, not a claim that the remaining execution has occurred.

## Immutable evidence register

| Item | Identity |
|---|---|
| Strict base | `88b2b4b6dffb6ad315443b38e29638b7aae87ec8` |
| Preregistration commit | `944832e7b206ea375b2bba7dc7d8a013c2d32f9e` |
| Protocol SHA-256 | `abbf3f52d1ef4eb148536b3cc0fc5bb13a24962a5353f55d739c541486bda486` |
| Experiment commit | `64439d80bdd7897aa94f8aa819ab23c17fcaffba` |
| Experiment tree | `381dcf3138fda77f41191497f833c7ac27ff6ae9` |
| Production source aggregate | `5b7139166c3fe1d0cb30527aa64004a0a7de5dcd49996bd99747d09c2d348d3b` |
| Development JSON SHA-256 | `0fab1abb37b5d6a15c4082f154978e6c8b86f62503e6ed71cc6232f5dc9de97e` |
| Protocol | `sultai-hybrid-correction-v1`, phase `development` |
| Integration | [PR #1](https://github.com/tachyon-beep/sultai/pull/1); final merge SHA is delivered separately |

[Manifest](../results/correction-2026-10-09/manifest.json) binds artifact and
checked-file hashes. [Development JSON](../results/correction-2026-10-09/development.json)
contains every method, lineage, trajectory checkpoint, cost and provenance
record. [Run metadata](../results/correction-2026-10-09/development-run.json)
records the command, UTC start, zero exit and empty stderr. One run only:
16.13 seconds wall, 16.12 CPU, 64,240 KiB peak RSS, CPython 3.12.14/x86_64.
No confirmation cohort or exact confirmation replay has run in cloud.

## Verification and review

108 unit tests passed in 40.338 seconds; all 17 quality checks passed. Strict
mypy 1.19.1 covers all 12 production modules; Ruff 0.15.4 covers source/tests.
The gate includes positive and negative fixtures, actual-source mutation
controls and T3 test-fingerprint validation. Tests cover healthy/training
collisions before fitting, missing provenance, marginal permutation invariance,
bad repairs, matched-rate equivalence and owned versus recoverable failures.
This is bounded executable coverage, not a proof of every possible T1 invariant.

The saved quality/test results ran on the frozen bytes immediately before
committing the experiment SHA; commit did not alter those bytes. A later
exact-PR-head integration check is reported with the merge handover; it is not
retroactively claimed by these saved logs.

Independent Astra [design review](../reviews/correction-design-review.md)
accepted the frozen protocol before execution. Independent implementation
reviews found S1 (overstated gate-2 pass) and R1 (misclassified solver failures).
Both were reproduced test-first and corrected before any development result.
The regression reviewer independently closed R1 at the experiment SHA with
six targeted tests in 0.011 seconds. See
[review/dispositions](../reviews/correction-implementation-review.md).
Independent scientific evidence review recomputed report aggregates and
contrasts without running cohorts; its disposition is preserved alongside them.

## Measured development results

Formation uses eight evaluation, eight clean-healthy and eight noisy-healthy
procedural lineages, with 32 separate offline training lineages. This remains
one four-template family and privileged residual telemetry. Each episode uses
64 conditioning, 24 selection and 64 audit examples. Every cohort, including
healthy partitions, is checked before fitting; candidates freeze before
selection. All supplied historical findings are distinct from this new assay.

| Evaluation method | Mean admitted audit MSE | Admissions / 8 |
|---|---:|---:|
| No-op | 0.113097774 | 0 |
| Paired learned former | 0.000616514 | 8 |
| Marginal96 former | 0.092968482 | 3 |
| Analytic4x4 | 2.573727e-23 | 8 |
| Full ridge | 1.627951e-20 | 8 |
| Frozen zero | 0.113097774 | 0 |
| Frozen negated | 0.113097774 | 0 |
| Frozen norm-random | 0.069950414 | 5 |

The paired former beats this specified marginal comparator, which is neither
optimal nor representative of all unpaired information. Analytic fitting
nearly solves the known family. That result demonstrates representability,
not learned formation advantage. All clean-healthy methods abstain. In noisy
healthy cases, harmful admissions occur for paired/paired-probes 1/8 each,
analytic4x4 1/8, shuffled 2/8, mismatched 4/8, negated 1/8 and norm-random 3/8.
Full ridge has 0/8 here: this does **not** reproduce or refute the historical
user-reported 2/8 probe with unavailable scripts and different fixtures.
All raw losses, abstentions, helpful/harmful/neutral classifications and false
admissions/rejections are retained. No population safety bound follows.

The lifecycle uses two independent sample streams of the same planted task,
ten arms, and 65 audit boundaries from insertion through deletion, excluding
all later recovery. Mean contrast is comparator MSE minus formed MSE:

| Comparator | Mean contrast | Per-stream min … max | Positive / negative |
|---|---:|---:|---:|
| No-growth LR .8 | +0.002893859 | +0.002802916 … +0.002984801 | 2 / 0 |
| No-growth LR 1.6 | −0.000572017 | −0.000598319 … −0.000545715 | 0 / 2 |
| Static retained LR .8 | −0.000572017 | −0.000598319 … −0.000545715 | 0 / 2 |
| Static taper | +0.000437685 | +0.000428329 … +0.000447041 | 2 / 0 |
| Frozen zero | +0.002893859 | +0.002802916 … +0.002984801 | 2 / 0 |
| Frozen negated | +0.013811757 | +0.013136079 … +0.014487435 | 2 / 0 |
| Frozen random | +0.008185159 | +0.007019650 … +0.009350668 | 2 / 0 |

All eight lifecycle known-answer checks pass. Static retained 544 parameters
at LR .8 matches host-only 272 parameters at LR 1.6 across all 137 boundaries,
maximum prediction difference 8.882e-16. This is a same-feature gradient identity,
not a general statement about capacity. Exact zero matches no-growth.

Formed insertion helps, but its deletion MSE is .00306445 / .00347489 versus
no-growth .000037588 / .000039439. Its final recovery MSE is 4.154e-6 / 4.497e-6,
also worse than no-growth 5.176e-8 / 4.773e-8. Every arm meets endpoint .01,
including zero, negated and random. Endpoint recovery is therefore descriptive
only. Preregistered specificity fails in both streams because matched-rate
and retained-static controls win. No post-removal benefit is established.

## Seven-gate disposition

| Gate | Verdict | Reason / evidence still needed |
|---|---|---|
| 1. Contract hardening | Bounded implementation checks pass | Strict types, boundary fingerprints, isolation and failure-class regressions; no universal semantic proof |
| 2. Instrument truth | Partial: single-run checks pass | Quality/fault injection complete; Nyx confirmation, replay and cross-phase provenance adjudication pending |
| 3. Useful repair headroom | Not established beyond templates | Analytic baseline solves this family; no unseen-host/family transfer |
| 4. Honest admission | Measured; unsafe admissions observed | Retains noisy healthy harm; no calibrated population safety |
| 5. Benefit after removal | Not established; specificity fails | Matched-rate controls win, removal/final losses worse than no-growth |
| 6. Calibrated safety and utility | Blocked | Domain, permitted information, worthwhile/risk margins, independent replication and budget need an owner decision |
| 7. Training readiness | Not established | Scientific prerequisites remain negative, blocked or unproven |

## Stop, remaining work and assumptions

Merge the instrumentation correction if exact-head engineering checks pass.
The predeclared Nyx confirmation/replay may close the evidence record with
unchanged settings; it cannot rescue development or authorize scaling. Parent
owns launch, resource checks, Page changes and STR updates. Stop on invalid
provenance, missing evidence, nonfinite output, timeout or replay mismatch;
retain faults and partial records, with no automatic restart or budget reset.

The [Nyx handover](../NYX_CORRECTION_HANDOVER.md) pins executable source and
commands. The [separate transfer proposal](../plans/2026-10-09-nyx-transfer-proposal.md)
recommends a minimal frozen-host headroom screen and alternatives with explicit
unmeasured resource estimates and proposed margins. No new generator or real
training campaign was implemented or launched.

Open evidence gaps: three exact Library imports, raw later Claude probe
scripts/results, Nyx confirmation/replay, and broader transfer/calibration
choices. Parent AGENTS/Filigree files and Nyx backup bundle were unavailable
in cloud. Prior research and saved results are preserved unchanged. Diffusion,
near-empty developmental origin and half-parameter capability remain hypotheses.
