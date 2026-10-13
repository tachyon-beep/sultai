<!-- filigree:instructions:v3.4.0:e2dfc82c -->
<!-- filigree:last-writer:filigree install -->
## Filigree Issue Tracker

`filigree` tracks this project's work: `filigree session-context` at session
start, then `filigree start-next-work --assignee <name>`. Name yourself on
writes: `filigree --actor <name>` (CLI) or `actor=<name>` (MCP);
`session-context` also reads `FILIGREE_ACTOR`.

Reference: the **filigree-workflow** skill, `filigree --help`, and the
`mcp__filigree__*` tool schemas. Prefer MCP tools; fall back to the CLI.

Two rules `--help` will not tell you:

1. Claim atomically: `work_start` / `work_start_next` (MCP) or `start-work` /
   `start-next-work` (CLI). Never chain a claim with a separate status update;
   that two-step form races other agents.
2. On `SCHEMA_MISMATCH` the installed filigree is older than the project
   database. Surface it to the user; do not retry.
<!-- /filigree:instructions -->

## Programme steering

Read `docs/STEERING.md` before proposing or running any experiment. The
objective is accuracy per retained parameter (John, 2026-10-13); repair
families are instrument validation only. The active draft study is
`docs/plans/2026-10-13-capacity-frontier-screen.md`; `docs/RESUME.md` has the
current state. Nothing runs on GPU without John's explicit go. Approvals are
bundled into gates (John, 2026-10-13): gate A, John accepts an implementation
plan (it then merges and implementation proceeds without further asks); gate
B, the code PR is reviewed and ready (it then merges and the pilot go is
requested); gate C, the pilot results are read (confirmatory go). Outside
those gates nothing merges to `main` or closes in the tracker without his
word. Merge with a merge commit only; never squash, rebase or delete a branch.
