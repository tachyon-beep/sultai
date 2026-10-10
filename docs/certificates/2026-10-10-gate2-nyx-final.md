# Nyx Gate 2 final evidence certificate

Decision: **Gate 2 — bounded instrument truth and deterministic replay: COMPLETE WITH RETAINED EXECUTION QUALIFICATION. Scientific STOP / NOT READY remains unchanged.** [Independent Astra final adjudication](../results/gate2-confirmation-2026-10-10-01a125af/independent-astra-final-review.md) accepts this bounded closure with no remaining blocker in scope. This certificate succeeds the [stopped confirmation checkpoint](2026-10-10-gate2-confirmation-checkpoint.md), preserving its qualification and the [development certificate](2026-10-09-hybrid-correction.md).

## Source and authorization

Executor: task `01a125af-1631-79f1-b7bb-c3cdc20ec13f` on verified `nyx.foundryside.dev`. Primary repository `/home/john/sultai`; exact detached execution worktree `/tmp/sultai-confirmation-64439d8`. Frozen experiment `64439d80bdd7897aa94f8aa819ab23c17fcaffba`, tree `381dcf3138fda77f41191497f833c7ac27ff6ae9`, production aggregate `5b7139166c3fe1d0cb30527aa64004a0a7de5dcd49996bd99747d09c2d348d3b`. Protocol SHA-256 `abbf3f52d1ef4eb148536b3cc0fc5bb13a24962a5353f55d739c541486bda486`.

Live GitHub connector reads verified main identical to merged PR #1 commit `8034cf8e59641c4ae46c12c2d21d7fa6bf171e20`. Executable inputs and protocol equal that merged main. GNU `/usr/bin/time` is available in the actual execution namespace and reports version UNKNOWN. John explicitly authorized historical-claim takeover and CPU clearance for both runs. Prior owner/investigator released their roles with 0/0 attempts. Tracker ownership transferred atomically; recorder blocker closed. User configuration dirt was preserved. No namespace bypass or alternate executor was used for either experiment. Supported Git-metadata escalation created the prescribed worktree after its initial read-only-mount failure; that preflight failure consumed no experiment attempt.

## Exact bounded invocations and exits

Both invocations used the unchanged handover command and `scripts/correction.sh`: GNU time `-v`, nice 10, timeout 120, one stdlib CPU process, one thread, no GPU. Each ran once, sequentially. CPython 3.12.3/x86_64 differs from development CPython 3.12.14/x86_64; historical measurements remain intact.

| Invocation | UTC start / finish, 2026-10-10 | Wall / user / system seconds | Peak RSS KiB | Raw / GNU time / outer tool exit |
|---|---|---|---:|---|
| Confirmation | 12:10:30 / 12:10:49 | 18.82 / 18.77 / 0.04 | 79,132 | 0 / 0 / **1** |
| Exact replay | 12:16:25 / 12:16:44 | 18.53 / 18.46 / 0.06 | 78,288 | 0 / 0 / 0 |

Both stdout/stderr pairs are empty. Complete report size is 2,062,797 bytes per invocation. Evidence remains below the 10 MiB ceiling. There were **one confirmation and one replay**, no retry, confirmation rerun, budget reset or further campaign.

The confirmation's outer execution-tool exit 1 remains unexplained. Saved literal invocation exit 0, GNU time exit 0, complete JSON and empty stderr support successful instrument completion; they do not erase the [conflicting observation](../results/gate2-confirmation-2026-10-10-01a125af/execution-envelope-discrepancy.json). Execution stopped before replay. Independent Astra accepted the internally consistent report with this qualification and the stop. John then explicitly approved retaining this specific qualified confirmation and running the one remaining exact replay; [linked disposition](../results/gate2-confirmation-2026-10-10-01a125af/replay-parent-disposition.md) identifies the attempt/hash/checkpoint. This is a disclosed, owner-dispositioned execution-envelope anomaly, not a repaired or discarded failed result. [Replay's outer exit 0](../results/gate2-confirmation-2026-10-10-01a125af/replay-run.json) does not retrospectively explain confirmation's outer exit 1.

## Reproducibility, provenance and engineering evidence

`cmp` returned 0: confirmation and replay are **byte-identical**, SHA-256 `569a362b6dc987850801eab437b846a8129cf9a6e6d33ead096269b8610b0339`. The exact saved-data Python check extracted from the unchanged handover returned 0; raw stdout/stderr and script bytes are preserved. Frozen worktree remained clean at the expected commit after both invocations.

Development JSON hash `0fab1abb37b5d6a15c4082f154978e6c8b86f62503e6ed71cc6232f5dc9de97e` matches. Source identity and protocol match across phases. Offline training lineages 1000–1031 are intentionally shared. Confirmation held-out evaluation/clean/noisy roots 5000/15000/25000 each contain eight disjoint lineages and 1,216 samples; they do not overlap development held-out lineages. Trajectory roots 9000/9010/9020/9030 are disjoint from development and use prescribed +1/+2/+3/+4 seeds. Digests alone are not set-disjointness proofs: inspected source constructs IDs using lineage/root, partition and index, and runtime global isolation guards enforce within-phase uniqueness. Saved formation/trajectory provenance and independent review supply the cross-phase argument.

Previously saved **108 tests and 17 quality checks** remain matched to the frozen bytes, including strict typing of all twelve production modules, policy/T3 fingerprints, positive/negative and mutation controls. All four artifact and 25 checked-file hashes were reverified. No unchanged tests, policy controls or cohorts were repeated merely for this documentation checkpoint. All eight lifecycle known answers pass; zero matches no-growth, retained-static/LR-matched equivalence holds, physical deletion/count/state and fixed cost/window records are retained. No production code changed.

The independent confirmation review recomputed all 54 formation summaries, 432 admission/classification records, 40 trajectory windows, 28 contrasts and seven contrast summaries, verified finite metrics and additive trajectory costs, and accepted the report's internal consistency. The final replay reviewer verified all 26 raw-file hashes and byte equality with originals, preservation of the original eleven files against checkpoint `289b499`, exact handover-checker extraction and identical saved-data output, frozen source/engineering hashes and the bounded sequential resources. It accepted Gate 2 closure with the retained qualification, without executing cohorts or tests.

## Scientific disposition

Paired evaluation mean admitted MSE is approximately `0.000273462`, versus analytic `1.93e-23`. Paired and paired-probe noisy-healthy harmful admissions are each **2/8**, analytic **2/8**, and norm-random **5/8**. Formed trajectories lose to matched-rate and retained-static controls in **4/4** streams, mean contrast `-0.000528930`. All forty endpoints meet `.01`, including bad-repair controls; endpoint recovery does not establish specificity. Confirmation is software-valid, diagnostic-specificity false, training-ready false. These are bounded synthetic measurements, not transfer or calibrated population safety claims.

| Gate | Final disposition |
|---|---|
| 1. Contract hardening | Previously verified bounded engineering checks retained; no universal semantic proof |
| 2. Instrument truth | COMPLETE WITH RETAINED EXECUTION QUALIFICATION: independent confirmation/replay/provenance closure accepted; parent-dispositioned outer-exit discrepancy retained |
| 3. Useful repair headroom | Not established beyond the template family; analytic control wins |
| 4. Honest admission | Outcomes retained; harmful noisy-healthy admissions observed; no calibrated safety claim |
| 5. Benefit after removal | Not established; specificity fails and matched-rate/retained-static controls win |
| 6. Calibrated safety and utility | Blocked on domain, information, utility/risk margins and replication decisions |
| 7. Training readiness | Not established; scientific predecessors remain negative, blocked or unproven |

## Preservation and stopping point

[Full local evidence packet](../results/gate2-confirmation-2026-10-10-01a125af/) retains raw exits, resources, reports, provenance, hashes, source/runtime identities, reviews and the linked parent disposition. Every copied raw file was verified byte-identical against `/tmp/sultai-confirmation-evidence.ZI3J18`; originals and the frozen worktree remain available. Earlier preflight and stopped-checkpoint bytes remain preserved. Git delivery state is reported separately from run completion; local preservation alone is not off-host backup.

Stop at this bounded reviewed outcome. No further attempt, next gate, GPU/paid work, package install, broader training, Simic/Esper/ELSPETH change, Page edit or schedule change is authorized or claimed. Parent retains Page/schedule ownership. Three exact research originals and missing historical probe bytes remain separate blockers; this replay does not reconstruct them.
