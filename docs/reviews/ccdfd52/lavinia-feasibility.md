# Lavinia feasibility addendum

2026-10-09. Bounded read-only assessment of ELSPETH commit `9e481bc85a32ee15f808c5e2b16cab7634e57b8e`, which remained HEAD at the follow-up inspection. Companion assessment: [Sultai minimum reuse](/home/john/Documents/Codex/2026-10-09/task-6/sultai-review-ccdfd52/lint-portability.md). No implementation, repository modification, installation, model call, paid service, training, or suite run was performed.

**Verdict: plausible yes, with moderate confidence.** ELSPETH has a real separable Python analysis core and several algorithms whose policy inputs could live in project adapters. A small separate Lavinia extraction experiment is credible. This is not evidence that the complete package can be made generic cheaply, nor evidence of general semantic understanding. Its current machinery is deterministic AST analysis guided by explicit project contracts. “Agentic semantic awareness” is a plausible agent-facing use of findings, contract metadata and source context, not a demonstrated autonomous reasoning capability of these core modules.

The minimum for Sultai remains Ruff, strict mypy and owned validated contracts. Lavinia should have a separate owner and scope; completing Sultai should not depend on extraction.

## Observed candidates and boundaries

| Observed component | Reusable substance | Coupling to separate or parameterize |
|---|---|---|
| [ast_walker.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/ast_walker.py:167) | Stdlib-only deterministic file traversal; parsed/source/AST records; syntax/read error variants; own-lexical-scope traversal avoids nested-scope leakage. | ELSPETH-specific excluded agent directories/prefixes; explicit-file path resolution behavior; caller decides whether read errors are emitted or skipped. Make exclusions/input ownership explicit, and surface incomplete scans. |
| [protocols.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/protocols.py:43) | Frozen findings; rule identity, severity, scope, metadata; small `analyze(tree, path, context)` protocol. | Category enum encodes ELSPETH families. RuleContext names allowlists and governance state, with a type-only import of ELSPETH Allowlist. `StrEnum` requires Python >=3.11; original package requires >=3.12. Finding canonical keys/fingerprints have suppression compatibility semantics. Keep initial interpreter boundary explicit rather than changing it accidentally. |
| [findings.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/findings.py:1) | Compatibility import only. | It is not an additional analyzer; actual model lives in protocols.py. No need to preserve this shim unless a consumer requires it. |
| [registry.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/registry.py:29) | Deterministic registration, lookup, duplicate rejection and a function-rule adapter. Stdlib plus protocols. | `load_builtin_rules` imports ELSPETH's entire rules namespace. Explicit rule registration can replace that without rewriting the registry algorithm. Default fallback category is ELSPETH MANIFEST. |
| [text/JSON/GitHub reporters](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/emitters/json.py:11) | Each uses stdlib plus Finding. Deterministic machine JSON and readable diagnostics; GitHub command escaping. | Package import names and chosen output schema. JSON currently exports only part of the extended Finding fields; preserve semantics deliberately. |
| [SARIF reporter](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/emitters/sarif.py:11) | Existing structured reporting algorithm, stdlib plus protocols. | ELSPETH name/URI/fingerprint namespace, fixed “precision: high”, fallback category. Tool identity and precision must describe actual Lavinia rules, especially heuristics. Optional after JSON, not a reason to enlarge the first experiment. |
| [fixture_harness.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/fixture_harness.py:31) | Paired clean/violation examples, expected finding comparison, fixture inventory checking. | Repo fixture layout and `config/cicd` allowlist-directory discovery; imports models/walker/registry/reporters. Reuse comparison ideas and selected fixtures before adopting its whole filesystem convention. |
| [validation_theatre rule](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/contract_invariants/validation_theatre/rule.py:15) | Small lexical/context heuristic using names, success returns and deferred text; depends on own-scope walker and finding models rather than runtime ELSPETH. | Naming/factory/deferred-vocabulary policy, metadata categories, import paths. Useful small port/parity specimen, but limited detection and possible nearby-text false positives. |
| [Composer catch-order rule](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/composer/catch_order/rule.py:18) | Alias resolution and ordered handler comparison against declared inheritance relationships. Only stdlib plus models/metadata. | Hardcoded Composer exception hierarchy and path ownership. A project adapter could provide known exception relationships without pretending this is a full Python type resolver. |
| Frozen-annotation checks | AST predicate already identified in companion report; recognizes syntactic frozen dataclass and mutable container names. | Full rule pulls in allowlist/governance machinery. It does not prove runtime deep immutability; adapt just the predicate if there is a real consumer requirement. |

These are actual narrow dependencies, not a claim that every file under `core/` is generic. Existing registry/models/reporters can plausibly be moved into a new namespace with modest changes. Much of the difficulty lies at policy/runner boundaries rather than AST traversal itself.

## What should stay with ELSPETH

Do not take `core/cli.py` wholesale: it imports judge, allowlist and atomic mutation machinery, combines many operational commands, and loads every built-in rule for selected checks. Do not take signed allowlist adjudication, key loading, agent SDK/OpenRouter/MCP extras, release attestations, re-audit vocabularies or migration manifests merely because they reside in the package. They implement specific operational policies and are unnecessary to establish a reusable static kernel.

Trust-tier, audit evidence, plugin contracts, manifest inventories, Composer error-channel and database-factory rules have meaningful project semantics. Their contract facts—owned roots/types, trust boundaries, exception hierarchy, immutable fields, required emissions—belong in explicitly supplied adapters. That does not mean every rule can become a configurable template. Some rules should remain ordinary ELSPETH adapter code using the shared kernel. Start with one demonstrated seam and expand only when a second real consumer needs it.

No upstream ELSPETH interface changes are necessary for a first experiment. Its protocol docs declare compatibility constraints, so an extraction should not quietly rewrite or remove the existing implementation while its release work is active.

## Smallest useful separate experiment

A separate Lavinia owner could propose one isolated, stdlib-only Python >=3.12 prototype, preserving upstream licenses and source provenance. Suggested scope:

1. Copy/adapt only walker, minimal finding/rule contracts, explicit-registration registry and JSON reporter into the new namespace. Keep ELSPETH-specific categories/context/exclusions behind a small adapter or remove unused fields with a recorded compatibility decision. Do not load any ELSPETH built-ins automatically.
2. Port one existing narrow detector and its clean/violation fixtures without changing intended outcomes. Validation theatre is the smallest first parity specimen. If proving project-adapter value is required immediately, instead use catch-order with its hierarchy supplied as contract data; test one tiny synthetic project's unrelated exception names as the second adapter.
3. Add the smallest command that takes explicit roots/files and rule selection and emits findings plus scan coverage/errors. Avoid judge integrations, package publication, daemon/service, dashboards or a new language/dataflow analyzer. Preserve “no selected rule / no in-scope file” rejection and propagate analyzer failures.
4. Run fixture parity against the pinned upstream rule in an isolated external environment, then run known-positive/negative controls through the new command. Prove selection, path scope, syntax/read failures, nested-scope behavior and deterministic reporting. Importing the kernel must not import PyYAML, dotenv, any ELSPETH runtime rule, or judge SDKs.

Success criterion: the same extracted algorithm runs against two differently named project contracts with stable expected findings, the core import graph stays stdlib-only, and mutation controls show an incorrect contract/path/rule selection cannot silently produce a trusted green result. If only a renamed wrapper works and every second project needs core edits, the adapter hypothesis has failed; stop and report that rather than growing a platform.

Rough estimate, explicitly unmeasured: a focused one-rule extraction/parity experiment is approximately 1–3 engineering days. Stabilizing a useful small package and two genuine consumer adapters could take several more days. A broad trust-tier system, generic type/dataflow reasoning, or agent-mediated adjudication would be a separate larger effort whose cost cannot be estimated from this bounded inspection. Reusing an existing generic lint/checker rule remains preferable whenever it already covers the requirement.

## License and provenance

The source is MIT, copyright (c) 2026 John Morrissey, under [ELSPETH LICENSE](/home/john/elspeth/LICENSE:1). `elspeth-lints` has no separate subtree license in the inspected metadata/source inventory. Its bootstrap commit is `86f513887e28c96518030a4e05954e2a9c429831`; extraction provenance must cite the actual inspected `9e481bc85a32ee15f808c5e2b16cab7634e57b8e` snapshot and each copied file, retaining the full MIT notice and recording modifications. There is no observed licensing obstacle to such an isolated prototype.

The candidate kernel modules above use stdlib imports plus their own sibling contracts. The whole original package declares PyYAML (MIT) and python-dotenv (BSD-3-Clause), plus optional judge/MCP dependencies. Leaving those paths out is a plausible way to avoid bringing those dependencies into the first prototype. A published or redistributed package needs a license inventory of its actual build/runtime dependencies; no publication is proposed here.

## Confidence Assessment

Overall confidence: **Moderate**.

| Finding | Confidence | Basis |
|---|---|---|
| A narrow reusable kernel exists | High | Walker/registry/protocol/reporter source imports and implementation inspected directly |
| Explicit policy adapters are realistic for selected rules | High for catch-order seam; Moderate overall | Hardcoded inheritance dictionary cleanly separated from comparison algorithm; other families vary substantially |
| An agent-facing semantic-contract tool is plausible | Moderate | Findings/metadata/source/AST already supply structured context; no agent-facing prototype tested |
| Generic semantic understanding is already present | Insufficient Data | No such capability demonstrated by this core inspection |
| 1–3 days for bounded extraction experiment | Moderate | Judgment from narrow modules and one-rule scope; no extraction performed |

## Risk Assessment

Implementation risk: **Medium**, reversibility **Easy** for an isolated prototype.

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Rule/policy/framework costs grow into a generic rewrite | Medium | Medium | One rule, two small contracts, explicit stop criterion |
| Framework reports green after unreadable/out-of-scope inputs | High | Medium | Coverage/error reporting and mutation controls |
| Heuristic findings overclaimed as semantic proof | High | Medium | Accurate precision, named limitation, runtime checks remain independent |
| ELSPETH release work disrupted | High | Low with separation | Separate owner/repo; pinned copies; no upstream edits during experiment |

## Information Gaps

The intended first agent-facing workflow and second real consumer are not specified. Those decisions determine whether findings plus project contracts are enough or whether a richer context mechanism is needed. Runtime import isolation, parity and package compatibility remain unproved until an authorized experiment. No gap prevents a **plausible yes** assessment; these gaps prevent promising a finished general package or autonomous semantic reasoning.

## Caveats and Required Follow-ups

Keep Lavinia ownership separate from Sultai, approve the first consumer/experiment scope, and verify parity/isolation/negative controls before relying on the kernel. Retain MIT provenance. Leave paid judges, services and broader architecture outside the initial scope. This addendum authorizes no implementation; the parent decides and creates the separate owner under John's conditional authorization.
