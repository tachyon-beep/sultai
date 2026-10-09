# Resume Sultai hybrid work

Project: `/home/john/sultai`, Nyx. Parent task:
`01a10aef-253f-7248-9619-7bb7271430ed`.

## Delivered milestone

John authorized the hybrid technical concept demonstrator on 2026-10-09 at
09:37 UTC. The bounded implementation is complete. Start with
[HYBRID_DEMO.md](HYBRID_DEMO.md) for commands, measured comparisons and limits;
[HYBRID_CONTRACT.md](HYBRID_CONTRACT.md) records the reviewed scope.

Local milestones:
- `d22bd75`: reviewed contract, preserved emmy snapshot and source provenance.
- `3e01a13`: reconciled concept/scope and stale handoff documentation.
- `6b03f89`: learned formation, controls, lifecycle/removal, CLI and unit tests.
- The following evidence/documentation commit records the final results;
  use Git history and the recovery manifest for its full identity.

Implementation was isolated on branch `hybrid-tcd` at
`/home/john/Documents/Codex/2026-10-09/task-6/sultai-hybrid` and integrated by
fast-forward after confirming main had no external changes. Preserve that
worktree if still attached; inspect status before using or removing it.
Filigree task `sultai-93937931a3` records delivery and final status. Lead actor:
`sultai-hybrid-lead`. All implementation and review workers finished.

Only short single-process CPU development is authorized, with a 120-second
external demo timeout. No GPU pilot, large corpus, paid service, publication,
or Simic/Aurora/ELSPETH changes. ELSPETH streaming remains priority.

## Verified evidence and honest limits

All 37 unit tests passed in 6.597 seconds, including the unchanged 12 legacy
checks. Two final integrated runs took 2.826 and 2.829 seconds, passed all
16 gates and produced byte-identical JSON. Code review found one omitted
ridge acceptance gate; it was fixed using the existing 1e-12 unit-test
tolerance, with a failed-control regression test. No training or fixture
retuning followed held-out evaluation. Independent final evidence review
approved the reported metrics, costs, hashes and bounded claims.

Artifacts:
- `docs/results/hybrid-cpu-2026-10-09.json`
- `docs/results/hybrid-runtime-2026-10-09.json`
- `docs/reviews/hybrid-design-review.md`
- `docs/reviews/hybrid-correctness-review.md`

Result SHA-256:
`adae7d8ab70e6bd13ddd44dd3127468e3b0937f0cca7fa05b0a7d615559e3326`.
Executable aggregate SHA-256:
`96913b8d85d7d6dfc4feddc4844d0d61db6e51f5462ba483c98654e6a9035dd9`.

Paired formation reduces held-out mean MSE from 0.0871569 to 0.000333165
on a shared public four-template family. Its stored repair has 272 values,
but effective output dimension is four. Conventional ridge is more accurate.
This is seen-family procedural holdout using privileged oracle residuals;
it does not establish transfer across trained hosts or unseen families.
Identity probes add no evidence. Eight healthy counterparts all choose no-op,
which is an empirical result, not a population safety bound.

The separate same-feature lifecycle runs task gradients, taper and actual
adapter deletion, finishing with 272 host values from a 544-value peak.
Formed-path final MSE is 5.05320e-6, while no-growth is better at 5.38757e-8.
Early-static taper is also better. Insertion is raw and scheduled; admission
is measured separately in formation. No near-empty growth, developmental
advantage, learned removal policy or half-size reference target is established.
No diffusion model is implemented. The original legacy instrument and all
imported Claude source files remain unchanged.

## Sources and remaining import blocker

The emmy and Claude source designs are preserved under
[designs/](designs/README.md) and `research/claude-handoff/`. Claude's complete
independent TCD and later Emrakul/Tamiyo revision arrived in `72f9eeb`.
All six source hashes match their index. The TCD's added revision-pointer
paragraph is separately identified; removing only that paragraph in memory
matches the supplied 59,999-byte original hash. It is not a saved original
attachment. Comparison completed before hybrid implementation.

The three original pasted Library attachments remain absent. Related Claude
reports do not fulfill their byte-exact import requirement. Supported Library
transfers failed, including bounded retries; do not repeat them. Read
`research/sources.json` for actual local paths and verification status.
Import issue `sultai-9c01e0943e` remains open; future real-host protocol issue
`sultai-8c727be6b1` remains dependent on import. This does not block the
completed bounded synthetic implementation.

## Re-entry and next decision

1. Inspect `git status --short`, `git log -6 --oneline`, `git worktree list`
   and `git remote -v` in Sultai. Preserve any newer work.
2. Read AGENTS.md, HYBRID_CONTRACT.md and HYBRID_DEMO.md.
3. Inspect tracker with `filigree show sultai-93937931a3` from main. Avoid
   `session-context`: this version can rewrite instructions/start dashboards.
   Its database may require filesystem permission even for read commands.
4. For a check after reboot, run `sh scripts/hybrid.sh --summary`; no install
   or network access is required. Run the unit suite only for new changes or
   unresolved correctness concerns.
5. Before a larger stage, select a real host/task, observable evidence, family
   holdout, capability margin, grouped calibration and an approved
   compute/duration/stopping budget. No larger run is authorized by this TCD.

Emrakul forms/admit/no-op and later reductive edits; Tamiyo nominates coarse
opportunities. Audited Emrakul outcomes must become informative before training
Tamiyo on them. No illustrative readiness percentage/intervention count is a
specification. Small fixture failures do not prove unrepairability.

## Recovery truth

Local Git commits protect committed work, not host or disk loss. No remote
is configured and no off-host backup is verified. The same-host pack is at
`/home/john/Documents/Codex/2026-10-09/task-6/backups/`:
`sultai-2026-10-09.bundle`, `filigree-2026-10-09.jsonl`, `manifest.json`.

Read the manifest's exact commit/tree, SHA-256 hashes and restoration-check
result to determine coverage; do not infer coverage from the date. Final
handoff refreshes the pack and checks it in a fresh bare repository with
`git fsck --full` and exact head/tree comparison. The tracker JSONL is an
export, not a live database backup. Credentials, private raw logs, model
caches/checkpoints and live tracker databases are excluded from Git and bundle.
