# Hybrid TCD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Demonstrate bounded learned formation, empirical admission and actual
learning/taper/removal on honest synthetic CPU fixtures.

**Architecture:** Preserve the emmy runtime. Add separate formation and
lifecycle modules, joined by a small reporting CLI. Assay boundaries and
negative readings are specified in `../HYBRID_CONTRACT.md`.

**Tech Stack:** Python 3.10+ standard library, unittest, local Git and Filigree.

**Prerequisites:** John has authorized this implementation and independent
subagents. Work only in this isolated Sultai worktree; no Simic/ELSPETH changes.
The named superpowers skills are not installed in this environment; use the
available subagent tools and the explicit validation/ownership steps below.
The parent performs all integration/commits, avoiding concurrent Git writers.

---

## Task 1 — contract and provenance (lead)

Files: `docs/HYBRID_CONTRACT.md`, this plan, `docs/reviews/hybrid-design-review.md`,
then `docs/DESIGN.md`, `CONCEPT.md`, `DECISIONS.md`, `RESUME.md`, research ledger
and README links. Preserve imported source files byte-for-byte. Correct stale
handoff status and distinguish original missing attachments from delivered
Claude sources. Independent Astra reviews the contract before implementation.
Commit this milestone first.

## Task 2 — formation (Sol formation worker)

Own `src/sultai/formation.py` and `tests/test_formation.py` only. Do not edit
legacy `repair.py`. Reuse its immutable Adapter/Example/Episode and full ridge.
The shared public interface is:

```python
from typing import Sequence
from sultai.repair import Adapter, Example

def form_adapter(generator, conditioning: Sequence[Example]) -> Adapter:
    return generator.form(conditioning)
```

Implement `run_formation()` returning `(report_dict, paired_generator)`;
`paired_generator.form(conditioning)` must be the frozen deterministic
new-host formation API. Lifecycle uses only that callable and does not inspect
its learned parameters. The worker owns exact template definitions and their
counts, recorded in the report. Train/dev/test seeds and hyperparameters are
fixed before held-out output is inspected. Meaningful tests first: isolation,
no-op, public interface, reproducibility and truthful parameter/query accounting.
Run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p test_formation.py -v`.
Done when tests pass, the report retains negative comparisons and no broader
claim is embedded. Do not commit independently.

## Task 3 — lifecycle and mathematical controls (Sol lifecycle worker)

Own `src/sultai/lifecycle.py` and `tests/test_lifecycle.py` only. Use a callable
former supplied by the lead; return JSON-serializable data from
`run_lifecycle(former, formation_label=...)`. Its conditioning examples describe
the residual needed at the current trainable host, expressed as identity-site
oracle targets. Never access final evaluation targets during training or
formation. Define fixed schedules/counts in source before final tests.

Prove no weight copying/folding, exact counts before/after deletion, no
remaining adapter in final serialization, shared futures and no-growth/static
controls. Add mathematical affine-folding and nonlinear signed-probe controls.
Run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p test_lifecycle.py -v`.
Done when no-op/removal invariants pass and the report distinguishes mechanics
from developmental/efficiency claims. Do not commit independently.

## Task 4 — integration and independent review (lead + Astra)

Own `src/sultai/hybrid.py`, `scripts/hybrid.sh`, `docs/HYBRID_DEMO.md` and result
artifact. The CLI runs both assays once and reports software acceptance apart
from comparative outcomes, source hashes, fixed config, deterministic metrics,
parameter and query costs. Time/resource measurement is separate from the
stable deterministic payload. Do not install dependencies.

Run single process, `nice -n 10 timeout 120 sh scripts/hybrid.sh`.
Run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v`.
Expected: all software invariants pass; all measured comparisons remain visible,
including losses. Independent Astra correctness/statistics review reads code,
tests and results. Address concrete blockers once, then obtain a final evidence
review of claims and artifacts. Further review loops only for new defects.

## Task 5 — durable delivery (lead)

Commit coherent code/docs and measured evidence. Fast-forward the untouched
main checkout only after rechecking ownership/clean state; preserve any new
external work instead of overwriting. Update Filigree with actual outcomes.
Refresh same-host Git bundle and tracker export; verify in a fresh bare repo
with `git fsck --full`, exact commit/tree match and SHA-256 manifest. No remote
publication. Final report includes commands, commits, results and limitations.

## Completion record

Tasks 1–4 are complete. Contract checkpoints are `d22bd75` and `3e01a13`;
the implementation is `6b03f89`. All 37 tests passed, and two final full runs
produced byte-identical JSON with all 16 gates passing. The independent code
review's missing ridge gate was corrected without changing fixtures/training;
the final narrow evidence audit approved the saved metrics and limits. See
`../HYBRID_DEMO.md`, `../results/` and `../RESUME.md` for delivery details.
The recovery manifest, rather than an assumed date, records final bundle
coverage and restoration verification for task 5.
