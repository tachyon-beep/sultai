# Resume Sultai after strict-contract hardening

Repository: `/home/john/sultai`, Nyx. Parent task:
`01a10aef-253f-7248-9619-7bb7271430ed`.

## Delivered state

The hybrid TCD is followed by separately authorized strict typing and required-
state hardening. Read [TYPING_CONTRACTS.md](TYPING_CONTRACTS.md) for exact changes,
ELSPETH conventions inspected/adopted, verification, and remaining limits.
[HYBRID_DEMO.md](HYBRID_DEMO.md) preserves the original v1 experimental record.

Known milestones: `ccdfd5225ee5575f365aa6a4a0698bed0f652958` is the immutable
original delivery; `f8011ce` preserves its independent semantic audit and neutral
Claude handoff. The following successor commit contains the fixes/evidence.
Read `git log` and the recovery manifest for the exact final full commit/tree.
Work was isolated on `strict-contracts-ccdfd52` at
`/home/john/Documents/Codex/2026-10-09/task-6/sultai-hybrid`; main is integrated
by fast-forward only after its clean original HEAD is rechecked.

Tracker: `sultai-c994ac71af`, actor/assignee `sultai-contract-lead`. The earlier
hybrid task `sultai-93937931a3` was completed separately. All workers finish
before delivery. Inspect current status before touching this worktree.

## Verification and scientific scope

Final frozen source aggregate:
`2431e053ddd5ddf10d0565923a382023ae3eeb881fd233da777eac28ee75ab1f`.
Final v2 result SHA-256:
`b4170d0b7da7909aea6744dfd7ea4ec63744c7b28b4eece0e7c6ec48ef5928e7`.

55 unit tests passed in 13.162 seconds. The local quality entry point passed
16 checks: strict mypy over eight production modules, Ruff lint/format over
14 source/test files, positive/negative fixtures and real-source-copy mutation
controls. Two complete demos passed all 16 gates, took 3.234/3.241 seconds and
produced byte-identical JSON. Every v1 assay/diagnostic value is unchanged after
excluding only protocol, source/runtime identity and new split provenance.
Artifacts are `docs/results/strict-*` and `hybrid-strict-*`.

Four scoped audit gaps were fixed: malformed affine prediction shapes,
incomplete generator state, missing required evidence/gates, and invalid cost
ledgers. A targeted review caught additional empty nested configuration/source
maps; exact keys/types/counts/digests and regression tests now close that case.
These were boundary counterexamples, not corruption of the original fixed run.
Optional absence is explicit; required missing state cannot become zero/no-op.

The research limits remain: one shared four-template synthetic family, four
output degrees of freedom materialized into 272 values, privileged oracle
residuals, no real-host/unseen-family transfer or calibrated safety guarantee.
The separate same-feature lifecycle physically disposes of a temporary adapter
and finishes at 272 values from a 544-value peak. No-growth and early-static
taper outperform the formed path. No developmental advantage, near-empty
origin, learned removal policy, diffusion or half-size reference result follows.

## Neutral Claude review package

The independent prompt targets ccdfd52, not this successor, and contains none
of the new semantic findings. Attach both:

- `/home/john/Documents/Codex/2026-10-09/task-6/sultai-review-ccdfd52/CLAUDE_REVIEW_HANDOFF.md`
- `/home/john/Documents/Codex/2026-10-09/task-6/sultai-review-ccdfd52/sultai-ccdfd52-review.zip`

Archive SHA-256:
`572fad1389c6607653f4d4bce6de471d6e3c490121235158061f0ee980a396b3`.
Its 37 source files are byte-exact from the pinned commit. Claude web may not
see Nyx; local path mentions are not file transfer. No Claude contact occurred.
Tracked copies of prompt, source manifest, audit, reproducers and lint study
are under `docs/reviews/ccdfd52/`. The source ZIP can be regenerated from the
pinned Git objects and manifest. Historical reproducers intentionally fail on
ccdfd52; they are separate from the current passing test suite.

## Re-entry

1. Inspect `git status --short`, `git log -5 --oneline`, `git worktree list`
   and `git remote -v`; preserve newer work. No remote was created.
2. Read AGENTS.md, TYPING_CONTRACTS.md and HYBRID_CONTRACT.md.
3. Inspect tracker with `filigree show sultai-c994ac71af` in main. Avoid
   `session-context`: this version rewrites instructions/starts dashboards.
4. Run `sh scripts/hybrid.sh --summary` for a bounded CPU check after reboot.
   No runtime dependency install, network or GPU is required.
5. For changed code, run the affected unit checks and the quality entry point:
   `nice -n 10 timeout 120 python3 scripts/check_quality.py`.
   It accepts explicit `--mypy`/`--ruff` paths; never installs automatically.
   Nyx used system mypy 1.19.1 and read-only ELSPETH Ruff binary 0.15.4.

Only short CPU development is authorized. ELSPETH streaming remains priority.
No long training, GPU pilot, paid service, publication, Simic/Esper/ELSPETH
changes or Lavinia project creation occurred here. A separate owner may assess
Lavinia; the [feasibility addendum](reviews/ccdfd52/lavinia-feasibility.md) finds
a narrow kernel plausible with moderate confidence and proposes a separate
one-rule, two-contract experiment. No extraction or generic analyzer rewrite
was performed here.

## Sources and remaining decisions

The emmy snapshot and delivered Claude source documents remain byte-preserved.
All seven delivered handoff files match their provenance hashes. The three
original pasted Library attachments are still absent after supported bounded
transfer failures; do not repeat failed downloads or reconstruct fake originals.
Import issue `sultai-9c01e0943e` remains open and real-host preregistration issue
`sultai-8c727be6b1` remains dependent. See `research/sources.json`.

The evidence-gate sequence supplied for the separately owned Flight Report
graphic is:

1. Trust the instrument: strict contracts, known-answer/healthy/no-op controls,
   malformed-state refusal, held-out isolation and genuinely failed targets.
2. Establish repair headroom: independent trained hosts and bottleneck families,
   usable telemetry, fair optimizer/retrieval/generator baselines and separate costs.
3. Test useful development: live handover, removal and subsequent learning against
   no-growth, static and final-architecture-from-scratch controls.
4. Build a defensible corpus: versioned lineage/site/intervention provenance,
   conditioning/selection/test separation, retained failures and grouped calibration.
5. Make a bounded programme decision: task, capability margin, data sufficiency,
   budget, stopping rules and the next hypothesis.

No competing graphic or additional experiment was created. Small bounded training
experiments may help establish readiness; the full half-parameter ambition is
a later goal, not a prerequisite to every experiment. Tamiyo learning needs
informative audited Emrakul outcomes, not invented readiness percentages.

## Recovery truth

The same-host pack is
`/home/john/Documents/Codex/2026-10-09/task-6/backups/`:
Git bundle, tracker JSONL and `manifest.json`. The final manifest records exact
commit/tree and SHA-256 hashes plus restoration into a fresh bare repository,
`git fsck --full` and exact head/tree checks. Read that coverage rather than
assuming the date is sufficient. Credentials, private raw logs, model weights
and live tracker databases are excluded; tracker data is exported separately.
No remote or verified off-host backup exists. Local commits/bundle protect
against accidental loss of the checkout, not host/disk failure.
