# Independent code review of the gate widening, commit fab22d3

Reviewer: independent code-review subagent, 2026-10-11, read-only, no shell;
reviewed the resulting tree at 1cffdcc rather than the diff. The author
separately ran the widened gate (`docs/results/gate3-gate-widening-2026-10-11/quality.json`,
20/20) and the 109 unit tests. Report preserved verbatim below. Disposition:
no findings; the operational note on the outer timeout is adopted in the
follow-up commit (documented outer `timeout` raised from 600 to 1200 s). The
note on the runtime source-identity manifest in `report_types.py` is out of
scope here: the screen package records its own source identity per §3 rule 6
of the preregistration.

---

No findings at confidence >= 80 in fab22d3. I had no shell tool, so I could not run `git show`, the gates or the tests. I reviewed the post-commit working tree (HEAD is 1cffdcc, clean), so I read the resulting code and not the diff itself. All conclusions are from reading; nothing was executed.

1. **Coverage of the new root.** Every scan site now goes through `SOURCE_ROOTS` or covers `src` as a whole. I found no remaining hard-coded `src/sultai` in the gates.
   - Policy scanning uses `production_sources()`, which is recursive and raises `PolicyInputError` for an empty declared root.
   - The mypy target is `*SOURCE_ROOTS`.
   - The before and after source hashes both use `production_sources()`.
   - Ruff lint and format run on `src tests`.
   - The mutation copy takes all of `src`, and a set-equality guard (`check_quality.py:104`) forces one mutation target per declared root.
   - The `src/sultai` strings left in `report_types.py:34-45` and `hybrid.py:32` are a runtime source-identity manifest. They are not gate scans, so they are outside this commit's scope. They will not list `sultai_screen`, which matters only if `sultai_screen` is meant to appear in report source identity.

2. **Negative controls pass for the right reason.**
   - The mutation copy includes `pyproject.toml`, `tests/` and all of `src/`, and cwd is the real root, so mypy and ruff use the real config.
   - The mypy check requires both `[return-value]` and the mutated filename in the output.
   - The ruff check requires both `F821` and the per-package marker.
   - Each mismatch raises `RuntimeError` loudly instead of passing.
   - The policy mutation `from typing import Any as _quality_policy_mutation` is caught by the `ImportFrom` alias-name branch (`check_policy.py:107-110`), so the "prohibited Any" message is the one the planted defect produces.
   - The T3 lookups use the copied `tests/`, and the unmutated copy already passes policy, so they cannot be the failure cause.
   - One weakness, low confidence and not worth blocking: `check_quality.py:129` tests the two substrings independently, not on the same output line. The risk of a false pass is negligible given the clean baseline.

3. **`from check_policy import ...` at `check_quality.py:15`.** `python3 scripts/check_quality.py` puts `scripts/` at `sys.path[0]`, so the import resolves, and `check_policy` has no import-time side effects. It would fail under `python -m scripts.check_quality` or `python -I`, but neither is the documented invocation. Nothing in the repo imports `check_quality`; `tests/test_trust.py` loads `check_policy` by spec from a file path.

4. **`tests/test_trust.py`.**
   - The `setUp` change creates both declared roots, and `src/sultai` gets `example.py` afterwards. Existing tests therefore still exercise the same paths.
   - `test_empty_production_surface_is_refused` still works because `src/sultai` is scanned first and is empty after the unlink.
   - `test_missing_declared_root_is_refused_by_name` is correct. It asserts a clean baseline, removes `src/sultai_screen`, and checks for "empty" plus the root name. The message "empty production surface: declared source root has no modules: src/sultai_screen" satisfies both. `shutil` is imported at line 7.
   - The disposable real-source test copies all of `src`, so it now covers the new root.

5. **Silent weakening: none found.** Scanning is strictly wider (recursive, plus a second root). The empty-root refusal is stricter than before.

One minor operational note, confidence about 40, so not a finding. `docs/TYPING_CONTRACTS.md:102` documents `timeout 600 python3 scripts/check_quality.py`. The script now runs about 9 mypy invocations, and the new comment at `check_quality.py:17-20` expects 20-40 s per cold run once a real screen package exists. That is roughly 180-360 s of mypy alone, so the outer 600 s `timeout` could kill the run before the 300 s per-check ceiling does. If that happens, the gate fails by clock and not on its own merits. The fix would be to raise the documented outer timeout, for example to 1200.
