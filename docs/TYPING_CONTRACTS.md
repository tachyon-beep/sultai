# Strict typing and required-state contracts

This is successor hardening of the delivered hybrid at
`ccdfd5225ee5575f365aa6a4a0698bed0f652958`. The independent Claude handoff
continues to target that immutable commit. No scientific fixture, optimizer,
schedule, or comparative target is being tuned in this work.

## Why this change

A fresh systems-thinking audit reproduced four contract failures at callable
and producer-report boundaries: missing affine predictions scored as zero loss;
missing generator rows became a valid no-op; required acceptance/healthy/summary
evidence could disappear without failure; incomplete or malformed cost ledgers
could become plausible totals. The ordinary fixed producers did not trigger
these failures, and the original result reproduced exactly. The findings and
reproducers are preserved under [reviews/ccdfd52/](reviews/ccdfd52/semantic-review.md).

John's statement that `.get()` caused historical Esper-lite telemetry failures
is user-reported context, not a newly reproduced Esper finding. The rule here
is semantic: missing required measurement is an error; a known zero and an
explicitly optional absence retain different meanings. Optional defaults are
not banned wholesale.

## ELSPETH conventions inspected and adopted

Read-only inspection used ELSPETH commit
`9e481bc85a32ee15f808c5e2b16cab7634e57b8e`. No ELSPETH files, environments,
experiments or running work were changed. Files inspected directly:

- `AGENTS.md`: measured claims and negative controls for gates.
- `pyproject.toml`: strict mypy, Ruff, explicit checked roots and exceptions.
- `docs/architecture/adr/032-validate-by-trust-domain.md`: distinguish owned
  internal types from parsed external values; runtime structural Protocol
  checks are not nominal security/dispatch controls.
- `src/elspeth/web/coordination/contracts.py`: owned frozen dataclasses, closed
  states, exact integer/string validation rather than truthiness/coercion.
- `src/elspeth/web/execution/schemas.py`: strict responses and unexpected-field
  refusal; no string-to-number coercion to conceal corrupt internal state.
- `src/elspeth/contracts/barrier_scalars.py`: required serialized fields,
  explicit optional `None`, bool-as-number rejection and finite value checks.
- `.agents/skills/tier-model-deep-dive/SKILL.md` and
  `.agents/skills/logging-telemetry-policy/SKILL.md`: required internal-state
  failures stay visible; no silent loss of probative telemetry.

Sultai applies those principles with its own standard-library types and
validators, without importing ELSPETH's Pydantic models or architecture.
The separate [lint portability assessment](reviews/ccdfd52/lint-portability.md)
records additional package, CI, rule and license inspection.

## Static gates and their boundary

Development tools are pinned in the optional `dev` extra: mypy 1.19.1 and
Ruff 0.15.4. They are not runtime dependencies. Existing installed executables
can be supplied explicitly; the gate runner never downloads or installs them.
The ordinary demo remains standard-library only.

Mypy checks all production modules under `src/sultai`, targeting Python 3.10,
with strict mode, explicit-Any refusal, unreachable-code warnings and unused
configuration warnings. Ruff checks/formats `src` and `tests` with Python 3.10
as its target. The selected rules are E/F/W/I/UP/B/C4/RUF. Line wrapping is
owned by the formatter. RUF019 and RUF051 style rewrites are disabled because
explicit presence/deletion checks may carry missing-state semantics. There
are no project-specific source suppressions to conceal type defects.

The local command is:

```sh
nice -n 10 timeout 120 python3 scripts/check_quality.py
```

If tools are outside PATH, supply `--mypy /path/to/mypy --ruff /path/to/ruff`.
Optional `--output /tmp/sultai-quality.json` records every command and exit code.
The command runs strict source checking, lint, formatting and small positive/
negative controls under the exact same configuration. Negative controls cover
missing TypedDict measurements, wrong types, untyped functions, explicit Any,
undefined names/unused imports and formatting drift. A green control is not
substitute evidence for runtime contract tests.

Tests intentionally construct malformed values and use mocks to exercise
refusal paths. They are linted and executed, but are outside the strict mypy
production surface, matching ELSPETH's scoped approach. Object-typed unknown
values are permitted only at parsing boundaries; they must be narrowed before
becoming owned records. Python 3.10 syntax/type targeting is not evidence of
execution on every supported interpreter; the measured Nyx runtime is 3.12.3.

## Reuse and licensing

No `elspeth-lints` code, signing/adjudication machinery, streaming proof checker,
plugin rules or new generic analyzer package is imported or extracted. The
existing generic Ruff/mypy tools and a small Sultai-local configuration are the
appropriate reusable subset. The local runner orchestrates existing tools;
it does not implement a static analyzer.

ELSPETH configuration conventions are adapted with attribution to the inspected
commit above. Its MIT notice is preserved in
[provenance/ELSPETH_LICENSE](provenance/ELSPETH_LICENSE), copyright 2026 John
Morrissey. No license change for Sultai is implied. Ruff/mypy are invoked as
installed development tools rather than redistributed. The assessment records
package/dependency licenses and limits of that inventory.

## Delivered contracts

Model weights and predictions are complete finite arrays with exact dimensions;
zero arrays are valid only when all required values exist. Generator and teacher
state is validated and detached at construction. Prediction metrics require all
16 channels; missing channels cannot reduce the denominator or create zero loss.
Counts are exact nonnegative integers, excluding Boolean/fractional values.
Losses are nonnegative finite numbers; signed deltas remain permitted.

Production reports use concrete total TypedDict schemas with explicit parsers
at publication boundaries. Gate sets, phases, methods, lineages, summaries,
closed configuration maps and source identities have required coverage.
Optional withdrawal/provenance fields are conditionally required for the formed
arm. `formation_acceptance` requires a frozen result from actual split validation,
with minimal lineage/count/digest provenance. Fingerprints record identity;
they do not authenticate malicious producers or prove every possible invariant.

Lifecycle state is explicitly `_Trainable | Adapter | None`. Clearing the
reference releases the object; serialized final state omits adapter weights.
Weak-reference tests cover both trainable and frozen modules. No zero mask
substitutes for deletion. Required missing evidence raises rather than producing
no-op, zero cost or success. A valid failed target remains a Boolean failure and
nonzero CLI exit, distinct from an invalid report.

The one targeted independent closure review accepted R1/R2/R4 and the original
R3 cases, then found incomplete nested count/source maps. That final R3 case
was fixed with closed TypedDicts, exact nested keys/counts, SHA-256 shape and
aggregate checks, and mutation regressions. The final lead verification below
covers the corrected frozen bytes; no second broad independent review is claimed.

## Verification record

- Full runtime suite: **55 tests passed in 13.162 seconds**, exit 0.
- Exact local quality entry point: **16 checks passed**, including strict mypy
  over all **8 production modules**, Ruff lint/format over **14 source/test
  files**, good/bad fixture controls and mutations of a disposable copy of a
  real production module. It verifies source hashes remain unchanged.
- Two final complete demos: **3.234 and 3.241 seconds**, exit 0, all **16
  scientific/software acceptance gates true**, byte-identical JSON.
- After excluding only protocol, source identity, runtime environment and the
  new `formation.isolation` section, the entire report equals the immutable v1
  JSON. Every earlier numerical assay/diagnostic value is unchanged.
- The quality runner itself additionally passed strict mypy and Ruff.

Evidence:
[unit tests](results/strict-unit-tests-2026-10-09.json),
[quality controls](results/strict-quality-2026-10-09.json),
[v2 result](results/hybrid-strict-cpu-2026-10-09.json),
[replay/comparison](results/hybrid-strict-runtime-2026-10-09.json), and
[targeted review](reviews/ccdfd52/semantic-closure-review.md).

Final executable aggregate SHA-256:
`2431e053ddd5ddf10d0565923a382023ae3eeb881fd233da777eac28ee75ab1f`.
Final result SHA-256:
`b4170d0b7da7909aea6744dfd7ea4ec63744c7b28b4eece0e7c6ec48ef5928e7`.

Remaining limits: strict checking covers production code, not every deliberately
malformed test input. Tests execute on CPython 3.12.3; Python 3.10 compatibility
is syntax/type-targeted but not runtime-tested here. These gates do not prove
all cross-field semantics, adversarial authenticity or research generality.
The original exact Library imports remain blocked. No real-host experiment,
substantive Emrakul programme or generic analyzer extraction was run.
