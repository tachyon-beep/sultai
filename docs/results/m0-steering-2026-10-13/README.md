# Merge-readiness receipt for the steering package (2026-10-13)

Branch `codex/gate3-screen` at `948f522` (29 commits ahead of `main` at
`8a779c6`, 0 behind, no conflicts; `.weft/` custody receipts are gitignored
and stay on the execution host). Checks run on Nyx:

| Check | Result |
|---|---|
| `scripts/check_policy.py` | Sultai trust policy passed |
| `scripts/check_quality.py` (`quality-948f522.json`) | passed, 20 checks: mypy 1.19.1 strict on 26 source files, ruff 0.15.4 lint and format, positive and negative controls, per-root mutation controls. The ruff binary was taken from the earlier Nyx CI environment under `/tmp/nyx-ci-981ef46-py312/`, since the project has no local install; mypy from the system Python |
| `mypy src scripts` | no issues in 26 source files |
| `unittest discover -s tests` (CPU) | 168 tests OK |

No source file under `src/` or `tests/` changed after `1dfe62f` (Gate 3
v4.2); every commit since is documentation and preserved evidence. No
weights, logs or binaries are tracked; the largest added files are the
attempt-4 oracle receipts (about 0.5 MB each).
