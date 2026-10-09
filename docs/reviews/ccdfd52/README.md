# Pinned ccdfd52 reviews

These reports concern the immutable delivered commit
`ccdfd5225ee5575f365aa6a4a0698bed0f652958`, not the subsequent typing hardening.

- [Neutral Claude handoff](CLAUDE_REVIEW_HANDOFF.md): contains no new semantic
  findings; give this and the separately prepared source archive to Claude.
- [Source manifest](source-manifest.json): 37 files from that exact commit.
- [Independent semantic review](semantic-review.md): four confirmed boundary
  gaps, with fixed-run versus malformed-input reachability distinguished.
- [Reproducers](reproducers.py) and [measured evidence](semantic-review-results.json).
- [Lint portability assessment](lint-portability.md): separate read-only study.

The original review package is at
`/home/john/Documents/Codex/2026-10-09/task-6/sultai-review-ccdfd52/`.
It contains the `snapshot/` directory, `test_semantic_review.py`, neutral prompt,
source ZIP and manifest. The source archive excludes all new semantic findings.
The independent reviewer initially verified 36 files; packaging subsequently
added the original `pyproject.toml`, taking the attachment manifest to 37.
All 37 were reverified without changing their bytes. The report preserves the
reviewer's original count and observations.

`reproducers.py` expects `snapshot/` and `source-manifest.json` alongside it.
Run the original external package, or recreate that layout in a temporary
directory from the pinned Git commit and manifest. Its default run deliberately
fails on ccdfd52: eight test methods demonstrate 13 assertion/subtest failures.
These are historical negative controls, not the successor's test suite.
`--observe` additionally reruns the bounded fixed assay and original suite.
Do not point historical reproducers at a changing checkout and call that a
reproduction of the reviewed commit.

The successor fixes and exact verification are recorded separately in
[TYPING_CONTRACTS.md](../../TYPING_CONTRACTS.md). No original review finding
has been rewritten to describe a fix that was absent at review time.
