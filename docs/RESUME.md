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

Original Library import is blocked: preparation succeeded, but byte transfer
and the one supported retry failed (retry: HTTP 403 for each attachment).
Do not restart an unbounded retry loop. Obtain a working supported Library
transfer/re-attachment before claiming exact research import.

## Re-entry

1. `cd /home/john/sultai`; read AGENTS.md, README.md and this file.
2. Run `git status --short`, `git log -5 --oneline`, and `git remote -v`.
3. Run `filigree session-context`; use atomic `start-work` for any task.
4. Confirm ELSPETH/reboot constraints before any further experiments.
5. Read docs/SCOPE.md, DESIGN.md, TEST_PLAN.md and research/README.md.
6. Run `sh scripts/smoke.sh` for the 12 short unit checks and JSON smoke.
7. Resolve attachment import and open decisions without modifying other repos.

## Exact deliverables and milestones

- `8e8a9bf`: first durable documentation checkpoint before implementation.
- `74d2cdd`: research synthesis, TCD clarification, revised design and Astra review.
- Use `git log` for the following implementation and final handoff commits.
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
loss. No off-host backup has been verified. A separately verified local bundle
may be added before handoff and must be reported as same-host only.

## Validation

All 12 unittest cases pass. Planted held-out mean squared error:
`0.06896343793009375` for no repair and `1.2123623237933552e-20` for the ridge
reference; healthy control remains zero. K=3 costs 72 candidate-example
selection queries and 96 final three-arm test queries per episode. These are
software instrument results from synthetic oracle targets, not evidence of
learned repair, real defect repairability, handover or the final efficiency goal.
Independent Astra correctness review found no material defects.
Track documentation in `sultai-5d22f48e2a`, implementation in
`sultai-b8bae4fcc2`, and the original-import blocker in `sultai-9c01e0943e`.
TCD definition issue `sultai-d06d9abe1d` is resolved by John's clarification.
