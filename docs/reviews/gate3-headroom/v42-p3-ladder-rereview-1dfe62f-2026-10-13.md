# Targeted independent re-review of the follow-up commit 1dfe62f

Reviewed commit: `1dfe62f146b1ca05d5d155d94a30bbb06df99a20` on `codex/gate3-screen`.
Scope: the diff from the panel-reviewed commit `c25c2ca` in `src`, `tests` and
the v4.2 amendment, checked against the disposition table in
[the implementation review record](v42-p3-ladder-implementation-review-2026-10-13.md).
Fresh read-only subagent, 2026-10-13; nothing run, edited or committed.
Report preserved verbatim below.

Disposition: suggestion 1 is followed by generating the certificate from the
committed bytes at launch time; suggestion 2 is adopted by recording the
NVIDIA driver version in the attempt-4 admission certificate (documentation
only; the environment check in code is unchanged); suggestion 3 is cosmetic
and left for housekeeping.

---

**HEAD confirmed**: `1dfe62f146b1ca05d5d155d94a30bbb06df99a20` on `codex/gate3-screen`, tree clean. Read-only; no tests run, no git writes, no GPU.

**Verdict: ACCEPT** (no blockers; three suggestions, one operational).

**Verified**
1. Pinned gains: `contracts.P3_RETIRED_GAINS` (contracts.py:33-40) equal `payload.rungs.P3["0.25"/"0.5"/"1.0"]` in all six `docs/results/gate3-pilot-2026-10-13-linked-2/materiality-<seed>/result.json`, compared by `float.hex()`; `runs/` copies byte-identical to `docs/results/`; payload `seed` is an int matching the directory.
2. `magnitude_decision` (pilot.py:288-319): six distinct seeds (289-290) → per-anchor exact tuple comparison against the pin, `PilotHold` on any difference (293-297) → ladder walk over `P3_ELIGIBLE` only (302) → existing `InstrumentHold` "exhausted" (313-314). P1/P2 ladders, 0.05 bar and the mean rule unchanged (300-301, 306-312). Matches amendment §3 clause by clause.
3. Exact equality is sound: receipts written by `encode` (contracts.py:98, `json.dumps` → `repr`), loaded by `json.loads` → same double; literals in contracts.py parse identically. Bitwise replay is expected: `host.py`, `data.py`, `impair.py`, `fit.py` byte-identical since attempt 3's commit `7d358bd` (only contracts/pilot/screen changed); `index % 2` card mapping unchanged (pilot.py:726); `losses`/`cached` use fixed chunks (fit.py:64, host.py:227); `make_impairment` draws a fresh generator per rung, so κ=2,4 carry no state into the retired rungs; attempt 3's `replay-card0`/`card1-a`/`card1-b` share `final_state_sha256` with `intact-3000`.
4. `AMENDMENT_SHA` = `sha256sum` of the committed amendment (`43149802…6ede9f`); composed `PROTOCOL_SHA` = `ae7ca18bbd89a978c4f27ae5b49fc2ecaead33c714f462c425d7f1d0460bb651`.
5. Dispositions: Science 1/Ops 1 → contracts.py:28-40, pilot.py:289-302, tests/test_v42_ladder.py:81-98; Science 6 → tests/test_protocol_identity.py:121-130 (full v4.1 identity rejected at pilot.py:393); Science 7 → contracts.py:73. Others "no change", confirmed.
6. Tests honest: retired fixture seeds 3000-3005 carry the pinned gains (test_screen.py:376-391, 745-757; test_v42_ladder.py:35-42); exact-clear, 1e-9 drift, unknown and duplicate seeds covered; `test_retired_gains…` asserts 3002's κ=1 gain < 0.05, so the retired set genuinely failed.
7. Nothing else: 10 files; evidence diff is hash rotation, tmp paths, 166→168 tests. Recomputed aggregate `source_sha256` at HEAD = `36c3fb277d7d56c27b2f193e0a77e762558885ddad8de62def2351e5c57eb617`, equal to runtime.json; per-file hashes match. Nine `PRIOR_RECEIPTS_3` hashes match on disk; `.weft/gate3-pilot-linked-pilot-3.json` absent.

**Findings**
1. SUGGESTION (90, operational) — The preserved review's "Launcher must know" quotes c25c2ca values (`implementation_source_sha256 = c5c6c41e…`, identity `9126b6fb…`). At 1dfe62f the certificate must carry `36c3fb27…` and `protocol_sha256 ae7ca18b…`, `amendment_sha256 43149802…`; the old values fail at pilot.py:393/401. Evidence entries: disposition `a3b3d203…` (unchanged) plus the review doc as committed (`c2e78912…` now; it changes if this re-review is appended). Preserving this review alters docs only, so `source_sha256` holds and `merge-base --is-ancestor` passes; tree must be clean.
2. SUGGESTION (70) — The one unpinned replay input is the NVIDIA driver (currently 580.178.04); `environment_identity` (screen.py:88-108) pins torch/CUDA/numpy only and attempt 3's custody records carry no driver field. A driver-induced drift would correctly hold, but after ~1500 s spent. Record the driver in the pilot-3 certificate or custody.
3. SUGGESTION (60) — A seed outside `P3_RETIRED_GAINS` raises `KeyError`, not `PilotHold` (pilot.py:296; test codifies it). Unreachable from `pilot()` since `INTACT_SEEDS` is fixed; the `test_magnitude_ladder_exhaustion_never_uses_failed_rung` name still describes P1 exhaustion. Cosmetic.
