# Targeted independent re-review of the follow-up commit ea33585

Reviewed commit: `ea3358541728e697d72310d05e19cd5a972acd88` on `codex/gate3-screen`.
Scope: the diff from the panel-reviewed commit `56a4512` in `src`, `tests` and
the amendment text, checked against the disposition table in
[the implementation review record](fixed-evaluation-implementation-review-2026-10-13.md).
Fresh read-only subagent, 2026-10-13; nothing run, edited or committed.
Report preserved verbatim below.

Disposition of its suggestions: none adopted before launch. Suggestion 1
(pre-admission holds exit with a traceback and no written report) matches
every other pre-admission hold and the custody receipt is unaffected;
suggestion 2 (test regex could be narrower) and suggestion 3 (a concurrent
loser of the race surfaces the file-exists error rather than the hold) are
recorded for a later housekeeping commit so that the certificate binds the
reviewed bytes unchanged.

---

HEAD = ea3358541728e697d72310d05e19cd5a972acd88 (confirmed). Read-only; nothing run, edited or committed.

**Verdict: ACCEPT**

Checks:
1. Dispositions implemented as described: Science 1/5 -> tests/test_fixed_evaluation.py:111-135; Science 2/3 -> amendment lines 118-137 and docs/certificates/2026-10-13-gate3-fixed-evaluation-authorization.md:23-26; Science 4 -> src/sultai_screen/pilot.py:488 (`is_file()`), absence test tests/test_protocol_identity.py:180-184; Science 7/Ops 4 -> tools.txt; Ops 1 -> proposal lines 49-50; Ops 2 -> amendment line 88, inside §3 (59-105); Ops 3 -> pilot.py:410-411 plus tests/test_protocol_identity.py:214-236. Ops 5 / Science 6 not adopted, as stated.
2. AMENDMENT_SHA pin (contracts.py:20) = dd30e1d1...32b4d = sha256sum of both the worktree file and `git show ea33585:` bytes. Old hash 7b9fde69 survives only in the verbatim review text. quality.json (24 files) and runtime.json (45 files, aggregate recomputes as compact sorted JSON) all match HEAD bytes; recorded_utc precedes the commit by 15 s.
3. Hold in claim_attempt is correct: `publish` (contracts.py:82-92) is unchanged and still `open("x")`; the `exists()` check only adds an earlier hold. In `pilot()` (pilot.py:567-583) it fires before `root.mkdir`, `admission.json` and `Supervisor`; `mkdir(exist_ok=True)` on the receipt's parent is a no-op when the receipt exists. Same ordering in twin_diagnostic.supervise (368-369). A concurrent loser of the TOCTOU window still gets FileExistsError from `open("x")`, so exclusivity cannot be bypassed.
4. The four FileExistsError->PilotHold changes (test_protocol_identity.py:193, test_screen.py:530 and 566, test_twin_diagnostic.py:156) are exactly the four `claim_attempt` re-claim sites; each still asserts receipt bytes unchanged / child not re-run. `publish`'s own `open("x")` refusal remains directly tested at tests/test_screen.py:69. No weakening.
5. New control (test_fixed_evaluation.py:111-135): ShapeSensitiveHost at N=5000 under `twin_diagnostic.evaluate(padded=False)`; asserts not bitwise, exactly 8 rows with batch size != 128, and `loss != baseline` equals `batch_sizes != baseline_sizes` elementwise (changes exactly where shape changes). Then padded instrument bitwise under the same permutation, and the shared `losses` bitwise under inverse permutation for both `None` and a 4-tensor wide adapter. Matches the disposition.
6. Nothing else slipped in: src/tests diff is exactly the hunks above; doc diffs are RESUME, authorisation record, amendment, new proposal, review record, evidence regeneration (161 tests, hash and temp-path rotation only).

Findings (no blockers):
1. SUGGESTION, confidence 60, pilot.py:410 / 782: a second `--linked-pilot-2` invocation now raises PilotHold, but it is outside the `try` at 782, so it still exits with a traceback and no written report (same as every other pre-admission hold). Acceptable; the disposition wording "explicit hold before any side effect" is accurate.
2. SUGGESTION, confidence 40, tests/test_protocol_identity.py:181: regex `"disposition"` also matches the earlier charge hold at pilot.py:486; `"disposition document"` would discriminate. Fixture is well-formed, so it passes for the right reason today.
3. SUGGESTION, confidence 30, pilot.py:410: a TOCTOU loser surfaces as FileExistsError, not PilotHold. Harmless; exclusivity preserved.
