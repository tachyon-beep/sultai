# Source map and insight traceability

The canonical reading order is [CONCEPT.md](../CONCEPT.md),
[DECISIONS.md](../DECISIONS.md), [DESIGN.md](../DESIGN.md), then
[TEST_PLAN.md](../TEST_PLAN.md). This map preserves provenance without treating
proposal text as validated results or duplicating unavailable original files.

| Key | Source and status | Where its insights are consolidated |
| --- | --- | --- |
| U | John's delegated brief and direct clarifications, 2026-10-09: TCD means technical concept demonstrator; near-empty start; joint addition/removal; half final retained parameters; training/search cost secondary; seed's independent evidence; delayed Esper-lite report; ELSPETH priority; canonical repo and independent Claude review | CONCEPT objective/mechanism, SCOPE, decisions D01–03/D10/D14/D17, RESUME. User report is attributed, not independently verified |
| R1 | OpenAI initial, `Pasted text.txt`, `libfile_8a7b298915588191b512a92d77443579`; 996 API text lines; exact bytes unavailable | COMPARISON; CONCEPT formation/lifecycle/control/accounting sections; D04/D06/D07/D09/D15. Fixed weight generation, typed supermodule and graph field are candidate stages, not implemented features |
| R2 | Telemetry follow-up, `Pasted text(1).txt`, `libfile_d643f55842e88191b58ec69df6ee006a`; 314 API text lines; exact bytes unavailable | CONCEPT local investigation, decision sufficiency, ambiguity probes, functional diversity, evidence comparison and failure vocabulary; DESIGN information contract; D03/D06/D08 |
| R3 | Claude report, `Pasted text(2).txt`, `libfile_7de516b203408191905c86eb469c63da`; 468 API text lines; exact bytes unavailable | COMPARISON mechanism/scaffolding notes; CONCEPT controls, null and cost categories; explicit corrections/disagreements D01–09/D15. Proposed corpus sizes, budgets and broad architectures are not authorization |
| R4 | Claude comparison, `Pasted text(20261009-054538).txt`, `libfile_e75b9396a0508191b5ff86fcd350b047`; all 59 API text lines read, no materialization attempted | SIMIC_RECONCILIATION, DESIGN reuse/static sections, D06/D11–14. It cited an older Simic revision; current-source verification has precedence for implementation claims |
| H1 | Claude handoff `notes/esper-lite-review.md`, within the 719-line Library-readable bundle `libfile_c1a641abe658819187e616c9dffbf020`; cites historical Esper `2d296b6` | CONCEPT instrumentation and causal-attribution lessons; D19. Historical counts, bug diagnoses and numeric performance are source assertions, not locally reproduced. Current archive attribution docstring checked separately |
| H2 | Same bundle, `notes/simic-seed-comparison.md`, cites Simic `16bcf70` | Existing source audit checks actual substrate and HLD boundaries; D18 preserves distinction between formation abstention and developmental removal. Continuous structure/discrete seed replacement remains proposal |
| H3 | Same bundle, `notes/openai-report-vs-repos.md` | Reinforces rich local investigation, probe controls, static capacity, functional telemetry and naming issues. Its comparison to R4 is conceptual; no byte-identity claim |
| H4 | Same bundle, `research/simic-diffusion-grown-modules.md` | Related Claude research report; preserve the corrections already recorded for R3. Its own verification labels and citation counts describe the report author's checks, not additional checks performed on Nyx |
| A0 | Astra comparison supplied in the delegation; no separate original file provided | Nonlinear 272-parameter minimal probe, paired/probed conditions, retrieval/deterministic/diffusion controls, independent groups, single vs best-of-K, GELU/PCA/teacher-null corrections, and efficiency vs trajectory distinction |
| A1 | [Local independent Astra design review](../reviews/astra-design-review.md), one pass | Adequate ridge reference; information budget and candidate-bank invariance; delayed-learning contract; dual grouping; all physically retained parameters. Disposition in ../reviews/DISPOSITION.md |
| A2 | [Local independent Astra code review](../reviews/astra-code-review.md), one pass | Verified bounded implementation/accounting and independent ridge-equation residual checks; not scientific validation of a generator or growth/removal |
| E | Read-only Esper-lite archive at `5ad6f9a657ceffba04946f68ef2c9b9a21f040b4`; README/ROADMAP/lifecycle documentation and `src/esper/simic/attribution/counterfactual.py:1` | COMPARISON lifecycle lessons and CONCEPT delayed-value/attribution cautions. No broad machinery copied; John's run-specific 3–4 step units remain unresolved |
| V | Read-only Simic audit at `454c7314cc2728ceee17513928541d60616bd081`; [audit manifest](simic-source-audit.json) | SIMIC_RECONCILIATION claim table and reuse boundaries; distinguishes implemented experiment code, HLD-only roles, and reported study-specific results |
| T | Sultai implementation `6d655e4`, unchanged in the consolidation; [SMOKE.md](../SMOKE.md), [synthetic result](../results/synthetic-smoke-2026-10-09.json) | Small executable instrument only: 12 passing checks, oracle ridge fixture and separate scalar probe. No generative or parameter-efficiency result |
| L | Five primary abstract/proceedings checks listed in [COMPARISON.md](COMPARISON.md) | Nearby mechanisms only. Full combined Sultai efficacy, novelty, recent report leads and scale claims remain unverified |
| C-pending | Independent adversarial Claude TCD design in the cloud Sultai project, not yet received here | Reserved for later separate-source comparison; no conclusions or implementation instructions attributed to it now. See CLAUDE_HANDOFF.md |

Exact file IDs, reported sizes and import/read status are in
[sources.json](sources.json). `originals/` holds no source bytes at present.
The bundle and its `verified-manifest.json` also failed local transfer. Their
expected per-report sizes and SHA-256 values are recorded as **reported only**,
not Nyx verification. The reserved `claude-handoff/` directory remains empty.
The bundle reports are not the pending independent adversarial TCD design.
Historical reviews can mention then-unresolved TCD meaning; their preserved
text is not silently rewritten. Current decisions and disposition explain
the subsequent user clarification.

## Cross-cutting observations retained

- Rich context can enable a simpler deterministic model; more elaborate
  generation cannot replace missing evidence. Sampling needs discrimination
  or selection to address genuine ambiguity (R2).
- Weight-space symmetries and host-coordinate drift require functional checks;
  diverse weight vectors need not be diverse repairs (R1–R3, A0).
- A single temporary field may replace several seeds' role only if it preserves
  the needed sequential diversity; this remains a hypothesis (R1, R3).
- Learned builders can be shared and absent at inference; anything still
  required to execute the deployed system must be counted (R1–R3, A1).
- Transfer, large-scale hierarchical control and open graph morphogenesis are
  possible later directions, not supported Sultai results or current scope
  (R1–R3). No corpus size or run plan from a report is adopted automatically.
- Failures, no-op wins, transient harm and abstentions are part of evidence,
  not discardable inconvenient outcomes (R1–R3, A0/A1).
- Same-host removal cost and causal contribution over learning are distinct.
  The additional direct checks were Simic `docs/design/01-claim.md:132` and
  Esper `src/esper/simic/attribution/counterfactual.py:1`; neither check reran
  an experiment (H1/H2, E/V). Paths, revisions and content hashes are in the
  [consolidation audit](consolidation-source-audit.json).
