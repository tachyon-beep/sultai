# Claude comparison reconciled against current Simic

2026-10-09; documentation-only reconciliation. The standalone Sultai CPU TCD
remains unchanged. No Simic code, tracker, run directory, checkpoint or process
was modified; no training, evaluation or test process was launched in Simic.

## Inputs and revision boundary

Claude's new comparison is Library item
`libfile_e75b9396a0508191b5ff86fcd350b047`,
`Pasted text(20261009-054538).txt` (reported 6,450 bytes). All 59 API text
lines were read; `has_more=false`. It is commentary, not primary evidence.
No exact-byte transfer was attempted or claimed. Its cloud paths are citations
inside that commentary, not Nyx paths and not inspected sources here.

Claude cites Simic `16bcf70a2c7fa7997de8634b21b12dbfe3d55d30`.
The inspected clean checkout is
**`454c7314cc2728ceee17513928541d60616bd081`**, five descendant commits later.
The relevant change since Claude's revision is prefix-stable common future
generation for comparisons across training horizons, plus the timing-study
module/plans and running-fleet documentation. The screen/data modules did not
change in that interval. See `simic-source-audit.json` for source hashes.

Method: read exact source bodies, tracked package inventory, existing test
definitions and committed result reports. No raw private runtime logs or
sealed/outer evaluation data were opened. This establishes implemented code
paths and documented results; it is not a fresh runtime reproduction or a
complete audit. No Loomweave MCP tools were exposed, so exact known files
were read directly without creating or refreshing an index.

## Claim-by-claim findings

All Simic paths below refer to the inspected revision above; line anchors
refer to that checkout. Use `git show <revision>:<path>` if it later moves.

| Claim | Evidence | Finding and limit |
| --- | --- | --- |
| Bounded CIFAR harness and four host pathologies | [Host](/home/john/simic/experiments/kernel_demo.py:558), [RunSpec](/home/john/simic/experiments/bounded_data.py:21), [load_fit_dev](/home/john/simic/experiments/bounded_data.py:195) | Implemented. `mild` changes widths, `under_normalized` omits BN and changes initialization gain, `channel_starved` narrows stage-2 interior width, `no_spatial_mix` uses 1×1 stage-2 convolutions. These are designed host alterations on CIFAR, distinct from the separate synthetic smoke dataset |
| No-growth, static and scheduled arms | [ARMS](/home/john/simic/experiments/bounded_comparison.py:49), [train](/home/john/simic/experiments/bounded_comparison.py:810), [attach_seed](/home/john/simic/experiments/bounded_comparison.py:542) | Implemented. Same human-authored seed family; static attaches before training with alpha=beta=1, scheduled attaches at the declared epoch. No learned generator or allocator |
| Paired seed screens | [common future](/home/john/simic/experiments/bounded_comparison.py:600), [pairing checks](/home/john/simic/experiments/bounded_comparison.py:1162), [screen launch](/home/john/simic/experiments/bounded_screen.py:218) | Implemented. Same derived host/body initialization and common future within each seed; pairing verifies initial host identity and no-growth/scheduled identity before germination. Static and scheduled calibration reflect different birth states. Seeds, not arms, are the repeated-measure units |
| Preregistration and immutable snapshots | [load_plan](/home/john/simic/experiments/bounded_screen.py:71), [make_snapshot](/home/john/simic/experiments/bounded_screen.py:204), [analyze](/home/john/simic/experiments/bounded_screen.py:577) | Implemented. Validates declared configuration/contrasts, records plan and analysis-module hashes, snapshots tracked source, rejects dirty launches/exposed test data, requires finished units, refuses changed plans and republication |
| Source-drift-refusing `verify_run` | [source_identity](/home/john/simic/experiments/bounded_comparison.py:409), [verify_run](/home/john/simic/experiments/bounded_comparison.py:1043) | Implemented. Checks completion/artifact hashes, source dictionary against six enumerated files, runtime identity and paired records. It is a scoped consumer check, not proof that arbitrary external generator code or a whole repository has been covered |
| Aurelia commissions; Nissa routes directly to Momir; Tamiyo does not steer | [constitution routing](/home/john/simic/docs/design/02-constitution.md:128), [INV-07/09](/home/john/simic/docs/design/02-constitution.md:201), [INV-35](/home/john/simic/docs/design/02-constitution.md:229) | Correct **HLD authority contract**, not an implemented runtime of those domains. Tracked `src/simic/` contains only `__init__.py` (`hello`) and `py.typed`; no `GrowthIntent`/`TelemetryEnvelope` occurrences were found in tracked Python experiments/src/tests. The bounded runner explicitly disclaims a learned controller/full programme |
| Resolution ±0.018 nats | [screen-v1 report](/home/john/simic/docs/results/2026-10-08-bounded-screen-v1.md:1) | Reported precision for one 48-seed study's late-epoch scheduled-minus-no-growth contrast. Not a universal harness resolution, power guarantee, outer-test result or Sultai result |
| Graft captures 31–60% of static gain | [graft-capture-v2 report](/home/john/simic/docs/results/2026-10-09-graft-capture-v2.md:1) | Reported on `under_normalized`, 10 epochs, 96 seeds per cell: norm 0.60 [0.51,0.72], conv_heavy 0.31 [0.19,0.42]. Descriptive paired-bootstrap capture fractions; static wins the declared contrasts/cost reading there. No rerun or generic generated-repair conclusion |

Existing tests back the intended checks: `test_bounded_hardening.py:69`
rejects unpaired hosts and its next test rejects pre-graft trajectory drift;
`test_bounded_screen.py:211` checks plan drift/republication, and
`test_bounded_execution.py:53` checks snapshot bytes against committed source.
Their definitions were inspected, not executed during this follow-up.

## Reuse assessment: yes to the substrate, through a separate boundary

Prefer a future narrow interface to a pinned version of Simic's existing
measurement/execution machinery over independently rebuilding CIFAR ingress,
common-future pairing, preregistration, evidence sealing and cost accounting.
This is a recommendation for the next design gate, not approval to merge,
migrate, rename, import the live checkout, or extend either implementation now.
Keep Sultai's stdlib 16-channel TCD as its small executable specification.

This is not a ready-made provider API. The current runner hard-codes three
arms, human-authored `seed_type`, a **64-channel 8×8** site, and one germination;
Sultai has a 16-channel pointwise fixture. HLD roles are not callable services.
An integration therefore needs a separately reviewed adapter boundary and
shape/task decision, not a claim that the existing 272 weights already plug in.

A future boundary should carry: pinned execution/spec and host snapshot
identities; role-separated instruction and data-only seed evidence; candidate
artifact/shape/semantics and provider identity for audit; declared budgets;
and paired outcomes, failures, parameter counts and all costs. Generator,
encoder/probe policy, candidate and interface code must also be hashed into
the execution identity; the existing six-file hash list does not cover them.
Run only from a separately pinned source snapshot with new result roots after
budget/ownership agreement. Preserve source-drift refusal and snapshot-based
analysis. Do not monkeypatch a live fleet or weaken verification to admit a
new provider.

Semantic mapping is useful without renaming Sultai: its coarse commissioning
role corresponds to HLD Aurelia, direct seed evidence to Nissa, and candidate
formation/investigation to part of Momir. Simic Tamiyo remains a witness.
The Sultai request's historical Tamiyo label is not evidence that the current
Simic name has changed. Seed telemetry is a functional model input and needs
pairing/shuffle/association tests; observability-inertness alone cannot validate it.

## Early static capacity: two distinct comparisons

Add matched static capacity at the first real-host experimental stage. It asks
whether simply supplying the known useful capacity from the beginning matches
or exceeds the result of generating/inserting it later. Simic's reported
results make this a concrete risk, not a reason to assume Sultai has failed.

1. **Local formation comparison:** from the same frozen intervention snapshot,
   compare generated repairs with conventional training of the same adapter,
   no repair and other declared providers using the same permitted evidence
   and selection budget. This isolates formation; the current ridge TCD only
   supplies a synthetic version of this conventional reference.
2. **End-to-end static-capacity comparison:** train the matched final module
   at the same site from step zero, paired by initial host lineage, task/data,
   future streams and declared final endpoint. Match final retained parameter
   budget and module family where possible. Its intermediate host snapshot is
   necessarily different; it must not be described as the same-snapshot arm.
   Birth calibration, learning head start and executed work differ. Report
   search/training/selection costs separately rather than calling equal epochs
   equal compute. Predeclare immediate and future-horizon endpoints.

A site/module chosen with knowledge of the planted defect is a **privileged
oracle-site capacity control**. It diagnoses the value of a known allocation;
it does not solve practical allocation. A conventional fitted adapter with
oracle residuals is privileged in another way. Neither has oracle optimal
weights by definition, and neither establishes a universal upper bound.
Give all oracle-site repair providers the same site privilege; keep comparisons
within explicit evidence tiers. A practical allocation system must choose
site, size and timing from allowed observations, charge its scouting/selection
cost, and be tested on held-out independent hosts. Include a precommitted
static/heuristic allocation reference when that later system is evaluated.

## Developmental removal versus experimental ablation

John's intent is **addition AND deliberate removal jointly driving growth**.
Removal is an action of the developing model: it may withdraw temporary
structure, delete/compact retained components, free a budget or force useful
reorganization, after which learning continues. The hypothesis concerns how
these additions and removals change subsequent learning and the final capable
model. Removal need not create immediate gain. It is not just post-training
compression, and setting weights/alpha to zero without eventual physical
deletion does not meet the retained-parameter objective.

Experimental ablation instead disables or changes an element to test a causal
claim: shuffle evidence, freeze the host, omit removal, or remove a scaffold
at a controlled time. The same deletion operation can be a developmental
action in one arm and a diagnostic intervention in another; its experimental
role and subsequent learning distinguish them. Frozen-host and final-architecture
retraining controls assess explanations, but are not the growth mechanism.

Later tests must compare joint development with growth-only, matched late
pruning/removal schedules and appropriate no-intervention controls, measuring
learning after each removal and after final withdrawal. Equal capability with
roughly half retained parameters remains the target. A final architecture
that learns equally well from scratch still meets that efficiency target,
while providing weaker evidence for a distinctive developmental trajectory.
