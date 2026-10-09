# Resume after reboot

Checkpoint date: 2026-10-09. Host: nyx.foundryside.dev.
Repository: `/home/john/sultai`. Parent task:
`01a10aef-253f-7248-9619-7bb7271430ed`.

## State

Sultai did not exist before this task. It now has a local Git repository,
own Filigree store, research inventory/comparison, reviewed design, test plan,
and runnable CPU instrument. No training campaign, GPU run, paid service or
remote publication has occurred. Only short synthetic ridge fits and unit
checks ran. Simic, Esper and ELSPETH were not modified.

John clarified TCD as **technical concept demonstrator** during the task.
The bounded first implementation is a short CPU synthetic instrument, not
a completed learned-generator, handover, or physical-removal system.

Follow-up reconciliation (documentation only): read
`docs/research/SIMIC_RECONCILIATION.md`. Simic was verified at
`454c7314cc2728ceee17513928541d60616bd081`, five commits beyond Claude's cited
`16bcf70`. Its bounded harness/screens/drift checks are implemented; its HLD
Aurelia/Nissa/Momir/Tamiyo separation is not an implemented domain runtime.
Prefer a later separate interface to pinned Simic measurement infrastructure
instead of rebuilding it, preserving this standalone TCD. This is not approval
to merge, migrate, start a run or extend either project.

The next real-host design must include matched static capacity early, distinguish
oracle-site/target controls from practical allocation, and treat addition AND
deliberate removal during continued learning as the growth mechanism.
Experimental ablations test that mechanism; they do not define it.

Canonical consolidation (documentation only): read `docs/CONCEPT.md`,
`docs/DECISIONS.md`, and `docs/research/SOURCE_MAP.md`. These integrate the
original reports, local telemetry follow-up, Astra reviews, existing TCD,
current Simic audit and historical Claude handoff reports. Material conflicts
are explicit, including strict formation, null outcomes, representation,
static capacity, physical removal and causal attribution. No implementation
was expanded by these follow-ups.

Claude's separate TCD is an independent adversarial design and is still pending.
Do not send it our design or steer it before completion. The reserved
`docs/research/claude-handoff/` directory is empty. A separate four-report
source bundle and manifest were readable through Library, but supported Nyx
transfers both failed: `library file transfer failed: download failed with
HTTP status 403`. Expected checksums are recorded as reported only; no local
bytes or matches were verified. See `docs/research/CLAUDE_HANDOFF.md` before
any follow-up transfer or design comparison.

Original Library import is blocked: preparation succeeded, but byte transfer
and the one supported retry failed (retry: HTTP 403 for each attachment).
Do not restart an unbounded retry loop. Obtain a working supported Library
transfer/re-attachment before claiming exact research import.

## Re-entry

1. `cd /home/john/sultai`; read AGENTS.md, README.md and this file.
2. Run `git status --short`, `git log -5 --oneline`, and `git remote -v`.
3. Run `filigree session-context`; use atomic `start-work` for any task.
4. Confirm ELSPETH/reboot constraints before any further experiments.
5. Read docs/CONCEPT.md, DECISIONS.md, SCOPE.md, DESIGN.md, TEST_PLAN.md and
   research/SOURCE_MAP.md; distinguish pending sources from settled decisions.
6. Run `sh scripts/smoke.sh` for the 12 short unit checks and JSON smoke.
7. Resolve attachment import and open decisions without modifying other repos.

## Exact deliverables and milestones

- `8e8a9bf`: first durable documentation checkpoint before implementation.
- `74d2cdd`: research synthesis, TCD clarification, revised design and Astra review.
- `6d655e4`: reviewed CPU demonstrator; 12 tests passed on installed Nyx path.
- `8220ccc`: initial reboot handoff. Use `git log` for the follow-up
  reconciliation/consolidation documentation commits.
- `858f6d5`: current Simic source reconciliation and developmental removal.
- The subsequent canonical consolidation adds CONCEPT.md, DECISIONS.md,
  SOURCE_MAP.md and CLAUDE_HANDOFF.md. `git log -3 --oneline` and the recovery
  manifest identify its exact commit; do not assume an older bundle covers it.
- `src/sultai/repair.py`: immutable nonlinear 272-parameter adapter, conventional
  ridge fit, synthetic fixture, disjoint split guard, selection and scalar probe.
- `src/sultai/smoke.py`, `scripts/smoke.sh`, `tests/test_repair.py`: executable
  demonstration and boundary checks, with no non-stdlib runtime dependencies.
- `docs/SMOKE.md`: exact synthetic assumptions, costs, results and limitations.
- `docs/research/COMPARISON.md`, `sources.json`: evidence/identity ledger.
- `docs/reviews/`: independent Astra design and code review plus disposition.

## Remaining scope

There is no trained deterministic generator, diffusion, retrieval/interpolation
benchmark, full coarse/paired/probed comparison, real-host integration, common
future learning, handback, physical removal or half-parameter result. The ridge
fit is conventional new-host optimization, not forward-only learned formation.
The scalar probe is a separate explanatory fixture, not a learned probe policy.
See DESIGN.md for staged requirements before expanding the demonstrator.

## Backup truth

The repo and commits are local to Nyx. No remote is configured or newly
authorized. A local Git commit protects against worktree loss, not disk/host
loss. No off-host backup has been verified.

The same-host recovery pack is at
`/home/john/Documents/Codex/2026-10-09/task-6/backups/`:

- `sultai-2026-10-09.bundle`: committed source/doc history, no ignored files.
- `filigree-2026-10-09.jsonl`: exported Sultai issues, comments and events.
- `manifest.json`: exact bundled commit/tree, sizes and SHA-256 checksums,
  bundle verification and fresh-repository object/commit restoration checks.

Inspect the manifest and recompute hashes before recovery. To restore to a new
path, `git clone <bundle-path> <new-directory>`; initialize a local Filigree
store there and use `filigree import --help` for the exported issue data.
The bundle excludes credentials, private runtime logs, ignored tracker DBs
and research originals (which were never imported). This is **same-host
redundancy only**, not protection from Nyx disk or host loss.

## Validation

All 12 unittest cases pass. Planted held-out mean squared error:
`0.06896343793009375` for no repair and `1.2123623237933552e-20` for the ridge
reference; healthy control remains zero. K=3 costs 72 candidate-example
selection queries and 96 final three-arm test queries per episode. These are
software instrument results from synthetic oracle targets, not evidence of
learned repair, real defect repairability, handover or the final efficiency goal.
Independent Astra correctness review found no material defects.
Those test results belong to the unchanged TCD. The follow-up inspected Simic
source/test definitions and committed reports only; it did not rerun Simic
tests or research results. No source/runtime feature changed in either project.
Track documentation in `sultai-5d22f48e2a`, implementation in
`sultai-b8bae4fcc2`, and the original-import blocker in `sultai-9c01e0943e`.
TCD definition issue `sultai-d06d9abe1d` is resolved by John's clarification.
Next research gate: `sultai-8c727be6b1`; no campaign is authorized by that issue.
Reconciliation documentation task: `sultai-5630caa657`.
Canonical consolidation task: `sultai-0dabe18434`.
