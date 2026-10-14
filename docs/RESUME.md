# Resume Sultai after the bounded correction

## Programme redirected to efficient capacity — 2026-10-13

John restated the objective after the Gate 3 attempt-4 hold: build new, more
efficient capacity, measured as accuracy per retained parameter; repair is
not the goal; Sultai is the add-and-remove alternative to Esper-lite's
plug-in modules. Delivered on `codex/gate3-screen`, merged to `main` 2026-10-13 (`ae35dfe`)
and frozen the same day; v4 (three amendments John took on 2026-10-13 after
the implementation plan's two reviews, §14) is frozen by
[freeze certificate 2](certificates/2026-10-13-m0-freeze-2.md), identity
`m0-v4`; the [implementation plan](plans/2026-10-13-m0-implementation.md)
was accepted at gate A (PR #6, `b79f145`) and implemented on
`nyx/m0-implementation` on 2026-10-13/14: nineteen tasks, each with a
scoped review, a whole-branch final review with two fix rounds, seventeen
per-item exact-source reviews with dispositions and one post-review
commit rechecked clean — certified implementation commit `978850a`
(410 tests; evidence under `results/m0-implementation-2026-10-14/`,
reviews under `reviews/m0-capacity/implementation/`, admission
certificate `certificates/2026-10-14-m0-pilot-admission.json`), merged at
gate B on 2026-10-14 (`03a4a7a`, PR #7). The same day John found the
preregistration's promised survival-curve bounds missing from the
confirmatory plan and took [amendment 3](plans/2026-10-14-m0-amendment-3.md)
(option A: the h = 16/4/2 arms join the §10 set; bounds and the sign test
beside every reported quantity), frozen by
[freeze certificate 3](certificates/2026-10-14-m0-freeze-3.md) as identity
`m0-v4.1`; implemented on `nyx/m0-amendment-3`, certified implementation
commit `c54fb7c` (417 tests; evidence under
`results/m0-amendment-3-2026-10-14/`, reviews under
`reviews/m0-capacity/amendment-3/`, admission certificate
`certificates/2026-10-14-m0-pilot-admission-2.json`), merged at the second
gate B (`086f123`, PR #8). John then took gate-B decision 2, option (ii):
[amendment 4](plans/2026-10-14-m0-amendment-4.md) gives the static host the
§4 no-op rule before any reading, frozen by
[freeze certificate 4](certificates/2026-10-14-m0-freeze-4.md) as identity
`m0-v4.2`; implemented on `nyx/m0-amendment-4`, certified implementation
commit `c49a4de` (420 tests; evidence under
`results/m0-amendment-4-2026-10-14/`, review under
`reviews/m0-capacity/amendment-4/`, admission certificate
`certificates/2026-10-14-m0-pilot-admission-3.json`), merged at the third
gate B (`0d4179a`). **The M0 pilot ran on 2026-10-14 under John's go from
`0d4179a` and finished pilot-clean** (exit 0, 1 h 22 min wall, 9,572 of the
14,400 one-per-card process-seconds charged, both cards verified idle after):
V0 passed on both cards bitwise, every instrument check true, the static
host admitted, all seven extended arms admitted and completed, confirmatory
count 32 (n* = 1 under the D1 half-width rule, so the preregistered floor
binds). Sanitized evidence under `results/m0-pilot-2026-10-14-0d4179a/` and
three independent reviews of the receipts under `reviews/m0-capacity/pilot/`
(PR #10). John then said "fix all the majors and minors":
[amendment 5](plans/2026-10-14-m0-amendment-5.md) carries the survival
refit's selected step count with its rate and the receipt additions the
reviews asked for, frozen by
[freeze certificate 5](certificates/2026-10-14-m0-freeze-5.md) as identity
`m0-v4.3`; implemented on `nyx/m0-amendment-5`, certified implementation
commit `13989d9` (430 tests; evidence under
`results/m0-amendment-5-2026-10-14/`, reviews under
`reviews/m0-capacity/amendment-5/`, admission certificate
`certificates/2026-10-14-m0-pilot-admission-4.json` with the
[source-change disposition](certificates/2026-10-14-m0-source-change-disposition.md)
that admits it over the pilot's source). **The fourth gate B, gate C (John's
read of the pilot) and the confirmatory go, with its ceiling, are John's
next decisions; the confirmatory has not run.** The controller's 40 rulings
are in the SDD ledger on Nyx
(`.superpowers/sdd/2026-10-13-m0-implementation/progress.md`, ignored):

- [STEERING.md](STEERING.md): the objective verbatim, the drift diagnosis,
  measurement rules for every later preregistration, the progress ladder
  M0–M3, and (§6) a pointer to the separate gate-ladder amendment that
  restates correction-protocol gates 3–7 (text accepted by John 2026-10-13).
- [Capacity frontier screen, draft v3](plans/2026-10-13-capacity-frontier-screen.md)
  (rung M0): on data the host already trained on, a nonlinear 272-parameter
  module against a recalibrated no-op, a foldable affine module of equal
  count, structured re-tunes of existing parameters and a decayed finish
  (V1, loss as the sole primary unit, accuracy a labelled secondary);
  overgrow-then-physically-cut versus the
  plug-in module built small at matched budget (V2); same-recipe width
  frontier as the scale; static-from-step-zero as a descriptive; P1 known
  answer kept as instrument validation; core and extended arm tiers with an
  overrun rule; decisions J0–J9 taken 2026-10-13 and recorded in §12 (J6 a
  4 GPU-hour ceiling; J9 a separate [gate-ladder amendment](plans/2026-10-13-gate-ladder-amendment.md)
  whose text John accepted; J10 merged 2026-10-13, `ae35dfe`; v3 frozen, see
  [the freeze certificate](certificates/2026-10-13-m0-freeze.md)). v1 (`5aeee56`) was reviewed by
  three independent readers; reports and dispositions are under
  `docs/reviews/m0-capacity/`. On 2026-10-13 twelve further agents each
  took one of John's decisions J0–J9 and wrote a recommendation with the
  case for and against; the reports, the arithmetic validation and the
  reconciled table with a v3 defect worklist are under
  `docs/reviews/m0-capacity/j-recommendations/` (`SYNTHESIS.md` first).
  Only J6 rejects its recorded default outright (4 GPU-hours); J0 is a
  genuine split (drop or keep the accuracy guard); John later clarified
  that accuracy was Esper-lite's currency, not a required unit.
- [Attempt-4 final disposition](certificates/2026-10-13-gate3-linked-pilot-3-final-disposition.md):
  the repair screen is closed on its finding; v4.2 confirmatory ranges
  retired unused; 4,593.33 s of the Gate 3 cap unspent and not carried by
  default; custody receipts intact. Decision D26 in
  [DECISIONS.md](DECISIONS.md).

Next: John's decisions J1–J10, then an implementation plan with exact-source
reviews (new reviewed fit path for parameter-subset re-tuning, static-module
host, resume-training path, physical truncation), then the pilot go. No GPU
execution, merge or tracker closure is authorised by this package.

## Gate 3 v4.1 fixed-evaluation package — 2026-10-13

John authorised this package on 2026-10-12 ("yes, please authorise and
execute") and assigned it to the Nyx Claude session `sultai-claude-nyx` while
`codex-sultai-gate3` is paused; see the
[authorisation record](certificates/2026-10-13-gate3-fixed-evaluation-authorization.md).
The tracker claim could not be reassigned by this session's tooling and still
names the Codex owner. Delivered on `codex/gate3-screen`:

- Shared frozen evaluator `sultai_screen.fit.losses` now pads every partial
  chunk to exactly 128 rows by repeating the last real example, applies the
  module after padding and returns exactly N losses in caller order. The
  global-permutation twin stays bitwise. Training paths are unchanged.
- [Protocol amendment v4.1](plans/2026-10-13-gate3-headroom-screen-v4.1-amendment.md):
  v4 bytes preserved; one composed identity `gate3-v4.1`; receipts and
  certificates must carry all four identity fields, old-only identities are
  rejected; fresh confirmatory seed ranges (intact 4100–4163, bottleneck
  5200–5399); GPU-time accounting stated; linked pilot 2 custody binding.
- Launcher binding for linked pilot 2 (`--linked-pilot-2`): carries the
  cumulative **1473.267264504917 s** against the unchanged 10,800 s cap and
  binds all three custody receipts and the terminal charge records by hash.
  It refuses to run without John's disposition document
  `docs/certificates/2026-10-13-gate3-linked-pilot-2-disposition.md`, which
  does not exist yet.
- Independent exact-source reviews of the first package commit `56a4512`
  (scientific: accept with conditions; PyTorch/operations: accept) are
  preserved with dispositions in
  `docs/reviews/gate3-headroom/fixed-evaluation-implementation-review-2026-10-13.md`.
  The follow-up commit adopts every condition: the unpadded-instrument
  control at N = 5000, the seed-range disposition wording and v4.1 restart
  blocks in the amendment (rotating the composed identity before any
  certificate), the on-disk disposition check, an explicit hold when a
  custody receipt already exists, the retired diagnostic entry point, and
  tool provenance in the evidence directory.
- Evidence: `docs/results/gate3-fixed-evaluation-2026-10-13/` (quality.json,
  runtime.json, runtime.txt, tools.txt) bound to the final package bytes.
- Costed proposal for the next GPU step, awaiting John's separate go:
  `docs/plans/2026-10-13-gate3-linked-pilot-2-proposal.md`.

**Linked pilot 2 (attempt 3) ran on 2026-10-13 under John's go and stopped
with an INSTRUMENT HOLD** (software valid, execution valid, no scientific
result): "P3 exhausted its materiality ladder". All twelve hosts, every
replay and the repaired global-permutation twin passed on GPU; P1 and P2 are
material on every anchor; P3 at κ = 1.0 clears 0.05 nats on five anchors and
reaches 0.040 on anchor 3002. Attempt charge 1461.08 s; cumulative
**2934.34 s** of 10,800; **7865.66 s** remain. Evidence
`docs/results/gate3-pilot-2026-10-13-linked-2/`; hold and disposition options
in `docs/certificates/2026-10-13-gate3-linked-pilot-2-hold.md` (recommended:
amendment v4.2 appending κ ∈ {2.0, 4.0} to the P3 ladder, one fresh linked
attempt 4). No further attempt, confirmation or merge is authorised.

**John chose option 1 on 2026-10-13 ("Go with 1").** Amendment v4.2
(`docs/plans/2026-10-13-gate3-headroom-screen-v4.2-amendment.md`, identity
`gate3-v4.2` chained over the unchanged v4 and v4.1 bytes) appends
κ ∈ {2.0, 4.0} to the P3 ladder with the rule unchanged, allocates fresh
confirmatory ranges (intact 4200–4263, bottleneck 5400–5599) and binds
linked pilot 3 (attempt 4) to all four custody receipts and the attempt-3
terminal records with opening charge 2934.3435500559863 s. Disposition
record `docs/certificates/2026-10-13-gate3-linked-pilot-3-disposition.md`.
Evidence `docs/results/gate3-v42-p3-ladder-2026-10-13/`. Launch of attempt 4
requires the independent exact-source reviews, the bound certificate and
fresh host checks; status of that attempt is recorded below this line when
it exists.

**Linked pilot 3 (attempt 4) ran on 2026-10-13 under v4.2 and stopped with an
INSTRUMENT HOLD at the bottleneck known answer** (software valid, execution
valid). Everything before it passed: twelve hosts, replays, timing admission,
materiality with the retired P3 rungs replayed exactly and κ = 2.0 chosen,
and the post-hoc known answer (B3 reaches the affine ceiling on P1 on every
anchor). At widths 4 and 2 the bypass module (P6) gains only 0.0005 to 0.026
nats more than the matched post-bottleneck module (P5) on every anchor,
against the 0.05 bar; width 8 is reserved for degenerate training, which did
not occur. Per v4 §5 the primary headroom family is undefined and the screen
stops before freeze with the finding published. Attempt charge 3272.33 s;
cumulative **6206.67 s** of 10,800; **4593.33 s** remain. Evidence
`docs/results/gate3-pilot-2026-10-13-linked-3/`; hold and options in
`docs/certificates/2026-10-13-gate3-linked-pilot-3-hold.md`. No further
attempt, confirmation or merge is authorised.

**No GPU execution, pilot, confirmation or merge is cleared by this package.**
The earlier holds, custody receipts and charges below are unchanged.

## Active Gate 3 implementation — 2026-10-11

John subsequently authorized the separate bounded headroom-screen implementation,
validation, independent specialist review, PR and gated development pilot.
Owner `codex-sultai-gate3` holds `sultai-ef0da96d8a`; branch
`codex/gate3-screen`. See [implementation and launch gates](plans/2026-10-11-gate3-implementation.md)
and the unchanged [v4 source protocol](plans/2026-10-10-gate3-headroom-screen.md).
Implementation source `4936d28c90e9ab3a4e7358c772ae0b7acbaffa54` is accepted
by independent scientific and PyTorch/operations specialists. All 20 quality
checks and 137 CPU tests pass. [Draft PR #3](https://github.com/tachyon-beep/sultai/pull/3)
and the [bound admission certificate](certificates/2026-10-11-gate3-pilot-admission.json)
record readiness for fresh resource/custody checks. No pilot or confirmatory
execution was claimed by that prelaunch checkpoint. The admitted pilot then
stopped on a distribution/runtime version-field mismatch before data loading
or training. **HOLD: execution-invalid; no retry or confirmation.**
The ledger charged 2.370860553928651 process-seconds; both GPUs were verified
idle afterward. See the [execution hold and concrete disposition](certificates/2026-10-11-gate3-pilot-hold.md)
and [byte-preserved evidence](results/gate3-pilot-2026-10-11-01a125af/manifest.json).
The original attempt custody remains intact. John subsequently approved the
[bounded linked retry disposition](certificates/2026-10-11-gate3-linked-retry-disposition.md):
separate distribution/runtime/CUDA provenance checks before custody, regression
controls, exact-patch independent review and an updated bound certificate,
then one linked development pilot retry. The original failure receipts and
2.370860553928651-second charge remain within the same 10,800-second total cap.
Repair commit `bbf9811854f23e40b2700dab9e457852c09ba7da` passed 20 quality
checks and 140 CPU tests and was accepted by both independent specialists.
The linked retry launched from bound certificate commit `9b97fb1` and then
**STOPPED: execution-invalid, resampled no-op evaluation twin not bitwise**.
All twelve development training hosts and host/fit cross-card/co-tenancy replay
checks passed before this first materiality failure. No repair-effect result
was reached. Cumulative charge is **1469.3526015179232 process-seconds**, including
the original failure, under the same 10800-second total cap. Both GPUs were
verified idle afterward; original and linked custody remain intact. See the
[new hold and diagnostic disposition](certificates/2026-10-11-gate3-linked-retry-hold.md)
and [byte-preserved retry evidence](results/gate3-pilot-2026-10-11-01a125af-retry1/manifest.json).
John subsequently approved the separate [one-worker, 120-second diagnostic](certificates/2026-10-12-gate3-twin-diagnostic-authorization.md) on 2026-10-12.
Diagnostic source `9e19e5e9fee3be709b54f810c6b3e4954271416b` passed 20 quality
checks and 146 CPU tests and independent acceptance. The one diagnostic completed
with raw/supervisor/GNU/outer exits 0: **3.9146629869937897 charged process-seconds**,
5.06 seconds wall. Cumulative charge is now **1473.267264504917 seconds** under
the same 10800-second cap. Exactly the 16 examples changing batch size have
changed logits; six losses differ. Identical-repeat, batch-size-preserving and
padded-baseline/twin comparisons are bitwise exact. Frozen state and permutation
round-trip match. See the [measured diagnostic result](certificates/2026-10-12-gate3-twin-diagnostic-result.md)
and [byte-preserved evidence](results/gate3-twin-diagnostic-2026-10-12-01a125af/manifest.json).
The diagnostic localizes this observed discrepancy to changed evaluation batch
size; it does not repair or clear the pilot. All three custody receipts remain.
The [fixed-size evaluation proposal](plans/2026-10-12-gate3-fixed-evaluation-proposal.md)
needs a separate source/protocol disposition; no repair, new pilot, diagnostic
repeat or confirmation is cleared. Both GPUs were verified idle after execution. The historical
Gate 2 verdict is unchanged. Never reset either durable pilot attempt receipt.
John then requested both independent specialists to review both the repair and
protocol questions. Their [combined design-review record](reviews/gate3-headroom/fixed-evaluation-design-review-2026-10-12.md)
conditionally recommends shared fixed-128 frozen evaluation, explicit selection
coverage, versioned protocol identity and meaningful CPU controls. The record
specifies the proposed algorithm and later GPU acceptance matrix. Repair/CPU
implementation remains unapproved; GPU execution, pilot and confirmation remain
uncleared. Checkpoint inheritance and any seed/version exception require an
explicit later disposition. No new experiment or production change occurred.
John then requested incorporation of all review feedback and a next-step
walkthrough. The [revised repair plan](plans/2026-10-12-gate3-fixed-evaluation-proposal.md)
now maps every substantive recommendation to implementation, protocol identity,
CPU acceptance or a later GPU decision. The initial proposal remains preserved
at `5751f52`. This is a planning update; implementation and execution status
remain unchanged.

## Nyx Gate 2 checkpoint — 2026-10-10

**Gate 2 COMPLETE WITH RETAINED EXECUTION QUALIFICATION; scientific STOP /
NOT READY unchanged.** John authorized claim takeover and CPU clearance.
Supported live GitHub verification passed. One frozen confirmation recorded
raw/GNU exits 0 but outer tool exit 1, so execution stopped. Astra accepted
the qualified evidence; John explicitly dispositioned this attempt and
authorized the one remaining exact replay. Replay raw/GNU/outer exits are 0.
Reports are byte-identical, SHA-256
`569a362b6dc987850801eab437b846a8129cf9a6e6d33ead096269b8610b0339`;
exact handover provenance checks pass. Runs took 18.82 / 18.53 seconds and
79,132 / 78,288 KiB peak RSS. Attempts **1/1**, exhausted; no rerun/reset.
Independent Astra final review accepted bounded closure while retaining the
unexplained confirmation outer exit. See the
[final certificate](certificates/2026-10-10-gate2-nyx-final.md) and
[morning handover](results/gate2-confirmation-2026-10-10-01a125af/MORNING-HANDOVER.md).
No later gate, broader training, Page or schedule action is cleared.
Historical development evidence below remains unchanged.

Parent task: `01a10aef-253f-7248-9619-7bb7271430ed`. John assigned exclusive
Sultai correction ownership to this cloud session. The original orientation
was read-only; implementation, draft PR and normal merge were subsequently
authorized. Simic remains with its existing owner and was not modified.

## Delivered evidence and verdict

Repository: `https://github.com/tachyon-beep/sultai`. Original cloud checkout
`/workspace/sultai` remains on `work` at strict successor
`88b2b4b6dffb6ad315443b38e29638b7aae87ec8`. Work is isolated in
`/workspace/sultai-corrections`, branch `codex/hybrid-corrections`, PR
<https://github.com/tachyon-beep/sultai/pull/1>. Inspect current Git/PR state;
this committed document cannot assert its own eventual merge SHA.

Frozen experiment source: `64439d80bdd7897aa94f8aa819ab23c17fcaffba`.
Production source aggregate:
`5b7139166c3fe1d0cb30527aa64004a0a7de5dcd49996bd99747d09c2d348d3b`.
Protocol committed before execution at
`944832e7b206ea375b2bba7dc7d8a013c2d32f9e`; its SHA-256 is
`abbf3f52d1ef4eb148536b3cc0fc5bb13a24962a5353f55d739c541486bda486`.

108 unit tests and all 17 quality checks passed on the frozen implementation
bytes. One development run took 16.13 seconds / 62.7 MiB peak RSS and returned
`software_valid=true`, `diagnostic_specificity=false`, `training_ready=false`.
Formation beats some weak controls but loses to analytic fitting. The lifecycle
loses to a matched-rate host in both streams. Harmful noisy-healthy admissions
remain. Every endpoint arm meets the old .01 threshold, including bad repairs.
These are useful negative findings, not permission to tune confirmation.

Read the [certificate](certificates/2026-10-09-hybrid-correction.md), immutable
[result manifest](results/correction-2026-10-09/manifest.json), and
[independent reviews](reviews/correction-implementation-review.md). Gate 2 is
still incomplete until Nyx confirmation/replay; gates 3/5/6/7 are unproven,
negative or blocked. No broader training readiness is established.

## Current architecture

The standard-library CPU instrument has twelve strictly typed production
modules. `formation_controls` adds all-cohort isolation, noisy healthy cases,
empirical analytic4x4 fitting, marginal96 and zero/negated/norm-random controls.
`trajectory` compares ten arms with a fixed 65-boundary pre-recovery metric and
matched effective learning rate. `correction` binds plan/source/runtime and
publishes complete, non-overwriting evidence; it requires committed inputs.
`trust` separates boundary `InputDataError`, owned `ContractViolation` and
documented `FitUnavailable`. AST fingerprint checks bind T3 boundaries to
specific validation tests. See [typing contracts](TYPING_CONTRACTS.md).

The materialized adapter has 272 values but only four output degrees of freedom
in this shared synthetic family. Physical taper/deletion and recovery work;
neither developmental advantage, unseen-host transfer, diffusion nor the
half-retained-parameter objective is proven. Simic's module growth/Tamiyo half
is distinct and supplies no Sultai experimental evidence.

## Next action and ownership

Parent owns Page edits, STR scheduling, Nyx resource inspection and safe sync.
After verifying the merged PR, follow [Nyx handover](NYX_CORRECTION_HANDOVER.md)
for the fixed confirmation and exact replay, preserving local work and ELSPETH
priority. Do not rerun development, tune controls or extend the 120-second
per-check ceiling. Publish valid negative results and stop the current claim.

A broader transfer screen is a separate owner decision. The
[concrete recommendation and alternatives](plans/2026-10-09-nyx-transfer-proposal.md)
specify a proposed tiny-MLP domain, information access, margins and resource
ceilings; these estimates and budgets are not approved or measured burns.
No paid service, GPU run, long training or Simic/ELSPETH change occurred.

## Preserved history and genuinely missing artifacts

Immutable original TCD: `ccdfd5225ee5575f365aa6a4a0698bed0f652958`;
independent semantic audit: `f8011cecaa9e0e27601e6252ba3145819bb7e4f1`;
strict successor: `88b2b4b6dffb6ad315443b38e29638b7aae87ec8`.
Historical source aggregate:
`2431e053ddd5ddf10d0565923a382023ae3eeb881fd233da777eac28ee75ab1f`;
v2 result hash:
`b4170d0b7da7909aea6744dfd7ea4ec63744c7b28b4eece0e7c6ec48ef5928e7`.
Original research and saved results remain byte-preserved. Historical 55-test,
16-quality/16-mechanics evidence retains its original scope.

Seven delivered Claude handoff files and 37 pinned source-manifest files were
verified during orientation. The three original Library attachment bytes and
the later six Claude probe scripts/results are absent. Their reported outcomes
remain user-supplied findings, not locally reproduced historical measurements.
The new fixed cohorts are new evidence, not exact replays of those probes.

Nyx `/home/john/AGENTS.md`, its Filigree installation and tracker store are
unavailable in cloud. No tracker mutation is claimed. Historical tracker/import
references and same-host backup coverage remain in the prior RESUME at the
strict commit; cloud did not verify those local backup files. GitHub branch
pushes are remote source preservation, not proof of a complete host backup.
