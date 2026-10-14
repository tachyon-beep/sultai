Reviewer: determinism reviewer (Claude), 2026-10-15. Scope: `git diff d0177b4..27b1394 -- src tests` in /home/john/sultai-gc; read-only; no GPU; no CIFAR batch opened. The final certified commit is 76d1fc0.

**Verdict: accept with conditions.** Blocking 0, major 0, minor 3, notes 4.

Condition: the certificate the confirmatory actually launches under must carry `"confirmatory_static_included": true`. That is admission certificate 5, which is named by the docs-only commit 95be585 but does not exist yet. Finding 3 explains why. Findings 1 and 2 are recommended before certification but are not conditions, because both fail closed before custody.

What the change promises: a launch whose `--static` flag disagrees with the certificate's recorded gate C selection is refused before `claim_attempt` and before the output root exists. The single custody receipt and the confirmatory seed block (7500 to 7531) are therefore never consumed under the wrong plan (43,200 s against 50,400 s, with or without the S-static arm). The diff delivers that for every JSON boolean value. Observed: the reviewed change holds this property, apart from the gaps below.

## Findings

### 1. Minor: no test covers a matching field. An over-strict mutant passes all 22 ConfirmatoryTests.

- Evidence. The new test `tests/test_m0_pilot.py:1574-1580` covers only the two mismatch directions. Every other ConfirmatoryTests certificate omits the field: the `certificate()` helper at `tests/test_m0_pilot.py:1396-1418` adds it only when `static_included` is not None, and nothing else passes `static_included`. I replaced `src/sultai_screen/m0/confirmatory.py:277` in a scratch `git archive` copy with `if "confirmatory_static_included" in approved:`, which refuses every certificate that carries the field. All 22 tests still passed (69.6 s, OK).
- Trigger and consequence. Launching the real certificate correctly (field true, with `--static`) takes exactly this path. A later regression that refused the matching case would be invisible to the suite until launch day. The refusal is pre-custody, so the cost is a blocked launch and a re-certification cycle, not a burnt block.
- Fix. Pass `static_included=True` in the static success test at `tests/test_m0_pilot.py:1790-1792` (`self.certificate(14, static_included=True)`, `static=True`). Add one non-static success with `static_included=False` (`static=False`), or keep one success without the field so the absent branch stays covered (see finding 3).

### 2. Minor: the comparison treats JSON numbers 1 and 0 as booleans. Malformed values give a circular message.

- Evidence. `src/sultai_screen/m0/confirmatory.py:277` compares `approved["confirmatory_static_included"] != static`. In Python, `1 != True` and `1.0 != True` are both False (I checked this against `json.loads` output). So a certificate recording `1` or `1.0` binds as true, and `0` or `0.0` binds as false. That is bool-as-number acceptance, which `docs/TYPING_CONTRACTS.md` lists among the ELSPETH conventions Sultai adopted (barrier_scalars: "bool-as-number rejection"). Other boolean certificate fields use `is not True` (`src/sultai_screen/m0/pilot.py:200-203`).
- For a string such as `"true"`, or for `null`, the check fails closed in both modes. That part is sound. But `selected` at `:278` is `value is True`, which is False for these values, so both refusals say "launch without --static". An operator who is already launching without `--static` gets told to do what they are already doing.
- Consequence. There is no custody or seed exposure, because every case is either a correct bind or a refusal before custody. The weakness is type-policy conformance and the operator message.
- Fix. Check the type before comparing:

  ```python
  if not isinstance(value, bool): raise PilotHold("confirmatory_static_included must be a JSON boolean")
  ```

  Then compare with `value is not static`.

### 3. Minor: "a certificate without the field binds nothing" leaves the P1 hazard open for any future certificate that omits the field.

- Evidence. `src/sultai_screen/m0/confirmatory.py:277` binds only when the key is present. Certificates 1 to 3 carry older protocol identities (`m0-v4`, `m0-v4.1`, `m0-v4.2`), so `check_certificate` (`src/sultai_screen/m0/pilot.py:195-199`) refuses them before `admit_launch`. The "older certificates" this branch is meant to keep compatible cannot reach it. Certificate 4 carries the field, true (`docs/certificates/2026-10-14-m0-pilot-admission-4.json`). Commit 95be585 landed on the branch during this review (docs only). It names certificate 5 (`docs/certificates/2026-10-15-m0-pilot-admission-5.json`, not yet present) as superseding certificate 4 and "carrying its gate C fields".
- Trigger and consequence. If certificate 5 is written without `confirmatory_static_included`, launching without `--static` is admitted under the 43,200 s plan and claims custody. That is exactly the P1 scenario.
- Is the absent branch tested? Yes. The 21 other ConfirmatoryTests cases use certificates without the field. Those include the static success at `:1792` and the non-static successes, so absence is exercised with both `static=True` and `static=False`. That is enough to show the branch binds nothing. It does nothing to prevent the omission above.
- Fix. Either require the field once `confirmatory_ceiling_hours` is present (the gate C fields travel together), or treat it as a certification checklist item. The minimum is the condition stated above.

### 4. Note: check ordering is sound, but some messages name a secondary reason.

`src/sultai_screen/m0/confirmatory.py:271-274` (pilot `static_admitted`, then `confirmatory_ceiling_hours == 14`) runs only when `static` is true. The new check at `:277-282` runs on every path that reaches it, and the earlier checks can only raise. So no ordering lets a mismatch through. Two cases:

- A certificate recording false, launched with `--static` against an unadmitted pilot or a 12 h ceiling, is refused with the S-static or 14-hour message rather than the gate C selection. Every one of these messages is true, and the result is a refusal.
- Real certificate 4 launched without `--static`: `static` is False, so lines 271-274 are skipped. Line 277 then sees `True != False` and raises "the certificate records confirmatory_static_included=True; launch with --static". The message is clear.

With `--static`, the real pilot result (`docs/results/m0-pilot-2026-10-14-0d4179a/pilot-result.json`: `static_admitted` true), ceiling 14 and field true pass all three checks.

Optional: move the new check above `:271` so the gate C record is named first.

### 5. Note: the test helper introduces a new mypy assignment error.

`tests/test_m0_pilot.py:1416` reports `Incompatible types in assignment (expression has type "bool", target has type "str")`, because `named` is inferred as `dict[str, str]`. The strict gate (`scripts/check_quality.py`) runs mypy over source roots only, and the test file already has other errors, so nothing gates on it. Fix: annotate `named: dict[str, Json]`.

### 6. Note: the message shows a Python value, not a JSON value.

`{...!r}` at `src/sultai_screen/m0/confirmatory.py:280` prints `True` where the certificate says `true`. This is cosmetic.

### 7. Note: HEAD moved during the review.

The worktree moved from 27b1394 to 95be585 while I was reviewing. `git diff --stat 27b1394..95be585` touches only `docs/certificates/2026-10-14-m0-source-change-disposition.md` (+19). That is outside the src/tests scope, and src and tests are byte-identical to the reviewed commit. Whether the new disposition's hash `af2db4e7…` matches the certified source was not checked.

## Checks

| Check | Evidence | Result |
|---|---|---|
| 1. Refusal before custody and before the output root | `confirmatory()` `src/sultai_screen/m0/confirmatory.py:611-622`: `check_output_root` (validates the path only, `src/sultai_screen/pilot.py:596-603`), root-exists refusal, `read_record`, `check_certificate`, `admit_launch` (new check at `:277-282`), then `environment_identity`, `check_cards`, `claim_attempt` (`:620`), `root.mkdir` (`:621`). `admit_launch` has a single caller (grep). The restart branch comes after the check (`:285`). `assert_refused` (`tests/test_m0_pilot.py:1439-1445`) asserts that neither custody receipt nor `base/run` exists. | Holds on every path. |
| 2. Ordering against the S-static checks | Finding 4. Real certificate 4 without `--static` is refused with "launch with --static". | No slip. One optional message reordering. |
| 3. Test strength | Mutant with the check disabled: the new test fails at `:1578` ("PilotHold not raised"). Mutant checking one direction only (true and not static): fails at `:1580`. Over-strict mutant: all 22 pass (finding 1). `write_pilot(static_admitted=True)` at `:1579` is required to get past `:271`; without it the refusal would carry the S-static message and the "launch without --static" regex would fail, so it is needed rather than weakening. Absence is covered by the 21 other cases (finding 3). | Kills removal and one-direction mutants. Misses the over-strict mutant. |
| 4. Trust policy | No `Any`, `cast`, `.get`, `type: ignore` or `noqa` in the diff. `scripts/check_policy.py --root /home/john/sultai-gc` reports "Sultai trust policy passed". `mypy --config-file pyproject.toml src/sultai_screen/m0/confirmatory.py` reports "no issues". `!=` is sound for JSON true and false. It accepts 0/1 and 0.0/1.0, and fails closed with a circular message for strings and null (finding 2). | Passes the gate. Minor type-policy gap. |
| 5. Gate 3 boundary | `git diff --stat d0177b4..27b1394 --` over pilot.py, screen.py, fit.py, host.py, twin_diagnostic.py, former_bridge.py and alignment.py produced no output (exit 0). | Empty. |
| 6. Rest of the diff (src and tests) | src: only `confirmatory.py:275-282`. tests: the `certificate()` keyword `static_included` (`:1396-1418`) and the new test (`:1574-1580`). Findings 5 and 6. | Nothing further. |

Runs: `PYTHONPATH=src:tests /usr/bin/python3 -B -m unittest test_m0_pilot.ConfirmatoryTests` at the reviewed tree ran 22 tests in 65.0 s, OK. The mutation runs used a scratch `git archive 27b1394` copy under the session scratchpad, never the worktree. Ruff is not installed on this host, so lint and format were not checked. `git status --short` in /home/john/sultai-gc was clean after every run.

## Re-review of 03073b4

Scope: `git diff 27b1394..03073b4 -- src tests` in /home/john/sultai-gc. This was read-only, with no GPU and no CIFAR batch opened. The worktree HEAD was 76d1fc0 at the time. `git diff --quiet 03073b4 -- src tests` against HEAD is clean, so the source and tests under test are those of 03073b4. The rest of 27b1394..76d1fc0 touches docs only (the disposition doc and `docs/results/m0-static-binding-2026-10-15/`), which is out of scope.

| Condition | Evidence | Met |
|---|---|---|
| (1) The field is boolean-typed, compared with `is not`, and the message prints JSON | `src/sultai_screen/m0/confirmatory.py:277-285`: `selected: object`, then `if not isinstance(selected, bool)` raises "must be a JSON boolean", then `selected is not static` raises with `'true' if selected else 'false'`. `isinstance(1, bool)` is False, so JSON 0, 1, 0.0 and 1.0 are now refused (closes finding 2). The message is no longer circular for string or null values, which get the type message. | yes |
| (2) The S-static success launches with `static_included=True` | `tests/test_m0_pilot.py:1793-1795`. The matching path now runs end to end to code 0. The over-strict mutant from finding 1 would now raise at this launch and fail the test. I reasoned this from the code and did not re-run the mutant. | yes |
| (3) A malformed value is refused with "JSON boolean" | `tests/test_m0_pilot.py:1581`, `certificate(14, static_included="yes")`, `static=True`, through `assert_refused`. That helper still asserts that neither custody receipt nor the output root exists. | yes |
| (4) `named` is annotated `dict[str, Json]` | `tests/test_m0_pilot.py:1409`. The helper parameter widens to `bool \| str \| None` at `:1402`. mypy (strict config, `MYPYPATH=src:tests`) reports no errors on lines 1390-1429, 1570-1589 or 1790-1799 (closes note 5). | yes |
| Refusal still precedes custody | `admit_launch` is called at `confirmatory.py:620`, `claim_attempt` at `:626` and `root.mkdir` at `:627`. Both new raises sit inside `admit_launch`, before the restart branch. | yes |
| Nothing else changed in src and tests | The diff touches only the hunks above. No `Any`, `cast`, `.get`, `type: ignore` or `noqa` was added. The Gate 3 diff-stat (pilot.py, screen.py, fit.py, host.py, twin_diagnostic.py, former_bridge.py, alignment.py) over 27b1394..03073b4 is empty. | yes |

Checks:
- `PYTHONPATH=src:tests /usr/bin/python3 -B -m unittest test_m0_pilot.ConfirmatoryTests`: 22 tests in 60.0 s, OK.
- `scripts/check_policy.py --root /home/john/sultai-gc`: passed.
- mypy on `confirmatory.py`: no issues.
- `git status --short`: clean afterwards.

Residuals:
- Finding 3 stays open as the external condition. Certificate 5 must carry `"confirmatory_static_included": true`. A certificate that omits the field still binds nothing, by design.
- Note: no test launches with `static_included=False` and without `--static`. No test sends a certificate without the field through line 277 with `--static`, because the only absent-field `--static` cases are refused earlier, at `:271` or `:273`. Both gaps are symmetric with paths that are tested, so the risk is low.
- Note 4 (message ordering) is unchanged and optional.
