# Static launch binding: review dispositions

Controller (Claude, Nyx), 2026-10-15. Review: `review-static-binding.md` (determinism reviewer, accept with conditions, 0 blocking / 0 major / 3 minor / 4 notes, scope `git diff d0177b4..27b1394 -- src tests`; re-review of 03073b4 appended). The final certified commit is 76d1fc0.

| # | Finding | Disposition | Where |
|---|---|---|---|
| 1 | Minor: no test launches with the field present and matching the flag | Applied: the S-static success test launches under a certificate carrying `confirmatory_static_included: true` | 03073b4, `tests/test_m0_pilot.py` `test_static_runs_s_static_under_the_14_hour_ceiling` |
| 2 | Minor: `!=` accepts JSON 1/0 as booleans; string or null refused with a misleading message | Applied: `isinstance(selected, bool)` first (non-boolean refused as "must be a JSON boolean"), then `is not`; message prints JSON true/false | 03073b4, `src/sultai_screen/m0/confirmatory.py` `admit_launch` |
| 3 | Minor: a certificate omitting the field binds nothing, so certificate 5 must carry it | Applied outside the code: certificate 5 carries `confirmatory_static_included: true`, `confirmatory_ceiling_hours: 14` and the other gate C fields copied from certificate 4, which it hash-lists | `docs/certificates/2026-10-15-m0-pilot-admission-5.json` |
| 4 | Note: check ordering sound; moving the new check first optional | Not taken: every message on the earlier paths is true and nothing proceeds | — |
| 5 | Note: new mypy assignment error in the test fixture | Applied: `named: dict[str, Json]` | 03073b4 |
| 6 | Note: `!r` prints Python `True` | Applied with finding 2 | 03073b4 |
| 7 | Note: HEAD moved to 95be585 (docs only) during review | Confirmed in re-review: src and tests at 76d1fc0 identical to 03073b4 | — |

Residual (re-review): no test launches with the field false and without `--static`, and no certificate without the field reaches the binding check with `--static` because earlier checks refuse first; both mirror tested paths. Accepted as a note; a malformed-value test was added instead.
