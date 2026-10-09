# ELSPETH lint portability assessment for Sultai

Assessment date: 2026-10-09. Read-only source snapshot: ELSPETH commit `9e481bc85a32ee15f808c5e2b16cab7634e57b8e`; Sultai commit `ccdfd5225ee5575f365aa6a4a0698bed0f652958`. ELSPETH package bootstrap origin: `86f513887e28c96518030a4e05954e2a9c429831` (`feat: bootstrap elspeth-lints foundation`, 2026-05-19). The most recent package-metadata commit at this snapshot is `bb4415c942929e369a0c4b2d85a4b664182f6a83` (2026-09-21). Any copied material should cite the inspected snapshot, rather than claiming it is unchanged since bootstrap.

**Recommendation:** reuse the ordinary Ruff and mypy tools with a small Sultai-local configuration. Adopt ELSPETH's principle of owned internal types and validated external boundaries through Sultai's own dataclasses and parsing code. Do not install, vendor, or extract the whole `elspeth-lints` package for this six-module stdlib demonstrator. A generic analyzer project would be a separate proposal requiring John’s decision.

This report and tiny controls are the only writes. No repository edits, package installations, shared-venv modifications, ELSPETH suite, streaming proof checker, commits, tracker commands, keys, environment-file reads, or network requests occurred. The unrelated ELSPETH untracked work was not opened.

## What is actually reusable

| Item | Evidence at inspected snapshot | Portability and recommendation |
|---|---|---|
| Ruff lint + formatter | [ELSPETH pyproject](/home/john/elspeth/pyproject.toml:302); [CI](/home/john/elspeth/.github/workflows/ci.yaml:113); [hooks](/home/john/elspeth/.pre-commit-config.yaml:45) | Generic tools. Reuse a subset of configuration, change target to Python 3.10 and first-party package to `sultai`. Run check-only gates over explicit source/test paths. |
| Strict mypy | [ELSPETH pyproject](/home/john/elspeth/pyproject.toml:361); [CI](/home/john/elspeth/.github/workflows/ci.yaml:121) | Generic checker. Retain strictness and useful diagnostics; omit Pydantic plugin, third-party import overrides, ELSPETH fixture exclusions, and Python 3.13 target. |
| Simple pre-commit hygiene | [ELSPETH hooks](/home/john/elspeth/.pre-commit-config.yaml:42) | Merge-conflict/debug-statement checks are portable. Optional later convenience; not necessary to deliver the first gates. |
| Secret-pattern shell scanner | [scanner](/home/john/elspeth/scripts/git-hooks/pre-commit-secret-scan.sh:1) | Mostly generic Git-index scanning, Bash/grep dependency. Separate from lint semantics. Optional local adaptation, preserving license. It prints matching lines, so controls must use synthetic credentials and logs must never be treated as safe secret storage. Not needed for the current minimum. |
| AST walker / registry / finding models / emitters | [registry](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/registry.py:29); [walker](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/ast_walker.py:11); [protocols](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/protocols.py:8) | Reusable engineering concepts, but extracting these creates a new owned framework. Registry loads the full built-in rule namespace; protocol models use `StrEnum` and walker exclusions encode ELSPETH agent conventions. No value over existing Ruff/mypy for this task. |
| Dependency audit / mutation tooling | ELSPETH `pyproject.toml:297` (`pip-audit`), `:117` (`mutmut`), dedicated CI jobs | Generic tools, additional workflow costs. They do not address the missing type/wire contracts. Keep out of the minimum subset. |

Sultai's [pyproject](/home/john/sultai/pyproject.toml:9) requires Python >=3.10 and has no runtime dependencies or checker configuration. Its reporting surfaces use bare `dict` annotations, e.g. [hybrid.py](/home/john/sultai/src/sultai/hybrid.py:18) and [lifecycle.py](/home/john/sultai/src/sultai/lifecycle.py:51). Strict typing therefore requires real code changes, not just adding a green-looking config. Mapping-shaped report boundaries should become owned result types or properly specified wire types, with runtime validation at ingestion. The lead is separately reviewing and implementing those contracts.

## Why the custom package is not the minimal option

[Package metadata](/home/john/elspeth/elspeth-lints/pyproject.toml:4) explicitly calls it a workspace-only ELSPETH-specific analyzer. It requires Python >=3.12, `pyyaml>=6.0,<7` and `python-dotenv>=1.2.2,<2`; optional judge extras add OpenAI/Claude SDKs and MCP. No ELSPETH runtime package dependency is declared, but that does not make its semantic policies generic. [ADR-023](/home/john/elspeth/docs/architecture/adr/023-custom-python-ci-analyzer.md:1) explicitly chose custom Python to express ELSPETH abstractions and computed manifests.

The [CLI](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/cli.py:276) loads built-ins for any nonempty rule selection; [rules/__init__.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/__init__.py:5) imports all rule families. Selecting one rule does not isolate its import graph. The CLI also imports judge and allowlist machinery at module import, and the allowlist implementation contains HMAC-verification and signed adjudication behavior. These costs are unrelated to a small local Sultai lint gate.

| Rule family | Actual contract / coupling | Sultai fit |
|---|---|---|
| Trust-tier / trust-boundary | Audit DB versus pipeline versus external-data trust, `contracts → core → engine → plugins` hierarchy, specific boundary decorators, source-param analysis, allowlists/adjudication. See [tier rule](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py:3). | Carry over the design principle, not ELSPETH's labels, hierarchy, signed exemptions, or blanket ban on dictionary defaults. Sultai needs its own owned types and explicit boundary validation. |
| Plugin / Composer / audit evidence | Plugin hash/component/options schema, Composer exception channels, nominal audit bases/decorators, graph attribution and read/write symmetry. | No matching Sultai substrate. Importing rules would largely certify absent ELSPETH constructs or reject valid Sultai patterns. |
| Manifests / CI-enforcer policy | ELSPETH declaration/symbol/test inventories, migration manifest, `scripts/cicd` layout. | Repo-specific process and paths. No current benefit. |
| Masquerade attribute probes | Exact-site baseline plus fixed roots `src/elspeth`, `tests`, `scripts`, `elspeth-lints/src`; baseline drift/occurrence/shape accounting. See [metadata](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/masquerade/metadata.py:13). | Not drop-in. Sultai lifecycle deliberately uses attribute presence for actual temporary-module allocation/deletion. Replacing that requires a deliberate model change, not accepting an inherited rule. |
| Frozen annotations | Rejects mutable container names in fields of syntactically frozen dataclasses; depends on shared allowlist/governance/walking models. See [rule](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/immutability/frozen_annotations/rule.py:95). | Closest broadly reusable AST rule, but not a proof of deep immutability: `Mapping`/`Sequence` are accepted even if backed by mutable objects; aliases/decorator naming also limit detection. Owned tuples, frozen dataclasses, detachment tests and serialization controls matter more here. Defer extraction. |
| Validation theatre | Function-name regex (`validate/check/verify`), nearby deferred words, success literal/factory returns; [implementation](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/contract_invariants/validation_theatre/rule.py:15). | Generic-looking heuristic but will not detect arbitrary fail-open acceptance or malformed result handling. It depends on ELSPETH models/walker and may false-positive on nearby text. Do not use it as evidence that Sultai acceptance gates fail closed. |
| SQLite/session factory / adapter budget | Database portability and repository-specific factory/adapter contracts. | Sultai has no database/runtime plugin architecture. |

Effort estimate (judgment, not measured): adapting checker configuration and a small gate runner is roughly 30–60 minutes; correcting strict typing across six modules and their report boundaries is likely a few focused hours, depending on the wire contract. Vendoring and adapting even selected custom rules adds ongoing analyzer maintenance, fixture parity, path/exclusion control, license/provenance bookkeeping, and false-positive policy. A generic-package extraction is substantially larger and is not authorized by the present assessment.

## Suggested concrete Sultai configuration

The exact external control configuration is [lint-controls/pyproject.toml](/home/john/Documents/Codex/2026-10-09/task-6/sultai-review-ccdfd52/lint-controls/pyproject.toml).

```toml
[tool.ruff]
target-version = "py310"
line-length = 140
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "C4", "RUF"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["sultai"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_unreachable = true
warn_unused_configs = true
show_error_codes = true
disallow_any_explicit = true
```

`disallow_any_explicit` is an intentional extension beyond ELSPETH's config: strict mypy by itself permits explicitly written `Any`. It suits stdlib Sultai once reports are properly typed. It still does not prevent every implicit `Any` crossing a JSON boundary; parse into `object` and validate into owned types. Do not use `dict[str, Any]`, casts, ignores, or catch-and-default branches merely to make the gate pass.

ELSPETH's ignored `RUF019`, `RUF051`, and `SIM401` encode its trust-tier no-`.get()` policy. Those exclusions are not generic style defaults. The suggested subset omits `SIM`; choose it separately if desired. Do not copy `T20` wholesale because Sultai's CLI legitimately prints JSON, nor copy the Typer immutable-default override, Pydantic plugin, or unrelated per-file ignores. If Sultai deliberately adopts a no-defaults policy for owned required report fields, document that semantic choice independently.

Run explicit `ruff check src/sultai tests`, `ruff format --check src/sultai tests`, and `mypy src/sultai` through Sultai's own runner/environment. Tests can remain outside the strict mypy surface initially, as ELSPETH's do. Use an explicit config path, check tool versions, record exit codes without pipeline masking, and ensure the package path exists before running. A CI job is only useful once a remote/CI service exists; a checked-in local command is sufficient now. A pre-commit hook may call the same local command later. Do not install ELSPETH dev extras into either shared environment.

## Controls measured in this assessment

All checks used the same external config and only external tiny fixtures. Existing system mypy was `1.19.1`; existing ELSPETH Ruff binary was `0.15.4`. ELSPETH's mypy was inspected as `1.20.2`, but not used for these controls. Mypy ran with `--no-incremental --cache-dir /dev/null`; Ruff with `--no-cache`; formatter with `--check`. Tools were executed read-only; Sultai package code was not executed. The full control batch took 2.733 seconds according to the command tool, with each subprocess bounded to 30 seconds.

| Instrument / fixture | Exit | Observed result |
|---|---:|---|
| mypy / `good.py` | 0 | No issues in one source file |
| mypy / `bad_type.py` | 1 | `[return-value]` incompatible `str` → `int` |
| mypy / `bad_untyped.py` | 1 | `[no-untyped-def]` |
| mypy / `bad_dict.py` | 1 | `[type-arg]` missing `dict` parameters |
| mypy / `bad_any.py` | 1 | `[explicit-any]` |
| mypy strict-only config / same `bad_any.py` | 0 | Explicit Any accepted, confirming strict mode alone permits it |
| Ruff lint / `good.py` | 0 | All checks passed |
| Ruff lint / `bad_lint.py` | 1 | F401 unused import + F821 undefined name |
| Ruff formatter / `good.py` | 0 | Already formatted |
| Ruff formatter / `bad_format.py` | 1 | Would reformat |

These positive and negative controls establish that the proposed configuration activates the claimed checks. They do **not** establish that current Sultai is lint-clean or that its scientific acceptance contract is sound. After integrating configuration, rerun controls through the exact final runner, and mutate a disposable copy of a real in-scope Sultai file to ensure its path filters are covered. Separately test malformed/missing required wire fields, bool-as-number, nonfinite values, unknown fields where forbidden, failed acceptance results and detached/frozen evidence; each must fail the relevant runtime contract while a valid result passes. Lint cannot substitute for those controls.

ELSPETH itself guards against empty rule selections: [CLI lines 1352 onward](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/cli.py:1352) return exit 2 without explicit `--rules`; `--fail-on-inert` checks incremental rule path coverage. If a custom fragment is ever adapted, prove selected-rule and path coverage as well as a violation/clean pair. A green analyzer that scanned the wrong roots provides no evidence.

## Licensing and provenance

Verified from local license files and distribution metadata, not inferred from package names:

- ELSPETH source/config/scripts: [root LICENSE](/home/john/elspeth/LICENSE:1), **MIT**, copyright (c) 2026 John Morrissey. `elspeth-lints` has no separate subtree LICENSE or license field in its package metadata; root MIT covers the inspected repository software. Retain the copyright and permission notice for copied/substantially adapted material and cite source file(s), inspected commit, and local modifications. Preserve a full MIT notice if vendoring code; avoid relying only on a prose “MIT” label.
- Ruff `0.15.4`: installed `ruff-0.15.4.dist-info/licenses/LICENSE`, **MIT**, copyright (c) 2022 Charles Marsh.
- Mypy `1.20.2`: installed metadata `License-Expression: MIT`; package license attributes Jukka Lehtosalo/contributors and Dropbox. Its bundled **typeshed is Apache-2.0**, explicitly stated in `licenses/mypy/typeshed/LICENSE`. Using the checker as a tool does not require copying its code into Sultai.
- PyYAML `6.0.3`: installed metadata/license **MIT** (Ingy döt Net / Kirill Simonov).
- python-dotenv `1.2.2`: installed metadata **BSD-3-Clause**, confirmed three-clause redistribution conditions in its LICENSE.

Only Ruff/mypy are needed as development tools for the recommended subset. PyYAML/dotenv and optional judge/MCP extras remain unnecessary. This is not a complete license inventory of all optional extras or checker transitive dependencies; none are being redistributed or installed by this assessment. A distributable tool bundle would require its own full dependency-license inventory.

## Confidence Assessment

Overall confidence: **High** for scope/configuration/dependency facts; **Moderate** for effort estimates and future maintenance cost.

| Finding | Confidence | Basis |
|---|---|---|
| Ruff/mypy configuration is the smallest reuse path | High | Actual project configs, six-module stdlib Sultai surface, passing/failing tiny controls |
| Whole package is ELSPETH-specific and Python >=3.12 | High | Package metadata, registry imports, rule source paths/semantics, ADR-023 |
| Strict mode alone does not ban explicit Any | High | Explicit-any control fails only with the extra configured check; ELSPETH config lacks it |
| License/provenance facts | High | Root and installed distribution licenses/metadata, Git history |
| Few hours for full Sultai typing adoption | Moderate | Concrete bare-dict/untyped surfaces; no complete strict baseline or implementation performed here |

## Risk Assessment

Implementation risk: **Low** for local configuration; **Medium** for changing wire/report representation. Reversibility: **Easy**.

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Green lint mistaken for runtime/scientific correctness | High | Medium | Independent acceptance and malformed-wire negative controls |
| Copied Python 3.13 target upgrades away Python 3.10 compatibility | Medium | High if copied unchanged | Target Python 3.10; test supported runtime separately |
| Gate scans zero/wrong files | High | Medium | Explicit paths, valid-source inventory, mutation in actual scanned path |
| Whole analyzer adds irrelevant policy and signing/package maintenance | Medium | High if adopted wholesale | Reuse only ordinary checker configuration |
| Typing “fix” weakens validation through Any/casts/defaults | High | Medium | Owned types, explicit parsers, runtime negative controls, no suppression shortcuts |

## Information Gaps

No gap blocks the minimum configuration recommendation. Remaining implementation facts are the exact final Sultai result/wire contracts and the amount of strict typing work; the lead owns their review and changes. Current Sultai-wide mypy/Ruff status was intentionally not measured here. No redistribution-license inventory of optional SDKs or full transitive dependencies was needed for this bounded assessment.

## Caveats and Required Follow-ups

Before relying on the adopted gate, run the positive/negative controls through its final entry point and verify actual source-path coverage. Finish owned report types and boundary validation before claiming strict typing protects the results. Keep runtime/scientific controls independent of static gates. Maintain Python 3.10 compatibility rather than copying ELSPETH targets. Record MIT provenance for copied material. No generic extraction/package project, ELSPETH changes, installs, or broader runs are authorized by this report.
