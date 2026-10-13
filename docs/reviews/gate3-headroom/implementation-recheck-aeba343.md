# Independent targeted rechecks of aeba343

Snapshot: `aeba3435da533b03c9b2e17bb6d49bea9775b263`, aggregate
`13030816ed5b2ff69a20fb1ad58196969985e1bdea498d8e5544032503e43653`.
Both independent reviewers retained BLOCK on one direct regression below.
No reviewer edited files, ran imports/tests, or launched CPU/GPU work.

## Scientific reviewer response

**BLOCK: both original blockers are fixed, but the new global-rate check introduces one classification defect.**

1. **Fixed:** B3 retains a finite overall winner when either schedule fails; plateau evidence is explicitly unavailable. All-bank failures retain the candidate reports and costs. `fit.py:270`, `screen.py:394`.
2. **Fixed:** Unselected cheap development nonfiniteness is retained; B2′ selection still uses selection loss alone. `screen.py:307`, `pilot.py:575`.
3. **Remaining:** η_B is chosen from P6, then checked on P5. If that rate failed **during fitting** on P5, `development_loss=None` means *not evaluated*. `pilot.py:563` nevertheless reports an execution-invalid “nonfinite development evaluation.” Distinguish fit-unavailable from evaluated/nonfinite; retain the unavailable fixed-rate arm or apply the existing valid instrument-hold disposition. Preserve η_B without fallback. Add the corresponding P6-success/P5-fit-failure control.

Reviewed `aeba3435da533b03c9b2e17bb6d49bea9775b263`, aggregate `13030816ed5b2ff69a20fb1ad58196969985e1bdea498d8e5544032503e43653`. Saved evidence reports 20 passing quality checks and 136 CPU tests in 25.750 seconds. No edits, execution, imports or GPU work performed.

## PyTorch and operations reviewer response

**BLOCK `aeba3435da533b03c9b2e17bb6d49bea9775b263`: one numerical-policy edge remains.** The operational blockers are fixed.

| Previous finding | Targeted recheck |
|---|---|
| Output receipts invalidate source provenance | **Fixed.** `pilot.py:407–425` restricts staging to ignored `runs/` or `/tmp` before certificate checks or custody. Strict source checking remains intact; later sanitized publication is documented. |
| One failed B3 schedule aborts the available bank | **Fixed.** `fit.py:270–276` selects the complete bank first and explicitly marks missing schedule comparison uncertified. |
| Entire failed bank loses evidence | **Fixed.** `fit.py:222–225` carries every report; `screen.py:394–396` serializes them into the worker failure receipt. |
| Unselected cheap development nonfinite is fatal | **Fixed for evaluated raw candidates; remaining edge below.** |
| B3 replay lacks co-tenancy | **Fixed.** `pilot.py:472–481` launches three fit workers per card concurrently and checks all nine serial/concurrent receipts. Actual GPU equality remains a pilot measurement. |

**P1 remaining — A recoverable P5 fit failure is mislabeled as a fatal development evaluation.** At `pilot.py:551`, the global η selection requires finite selection results only on P6. The corresponding P5 fit can therefore have `selection_loss=None`, with `development_evaluated=false` and `development_loss=None` from `screen.py:308`. Nevertheless, `pilot.py:563` passes it unconditionally to `selected_development`, which raises an execution-invalid “nonfinite development evaluation” although no development evaluation occurred.

Distinguish unavailable fits from evaluated selected-candidate failures. Retain the unavailable fixed-rate P5 result without changing η or using development-based fallback; preserve the fatal rule for an actually evaluated selected candidate.

Reviewed only the bounded diff, dispositions, and saved validation evidence. Saved reports bind aggregate `13030816ed5b2ff69a20fb1ad58196969985e1bdea498d8e5544032503e43653` and report 20 quality checks plus 136 CPU tests passing. No tests, imports, jobs, edits, or GPU work performed.

## Owner disposition

Adopted the retain-unavailable option. `cheap_rate_decision` preserves the
P6-only η choice and publishes per-anchor/family fixed-rate status. A failed
fit must explicitly carry `development_evaluated=false` and no loss; its arm
is `fit-unavailable`. Finite selected fits still call the fatal development
check. No η fallback, threshold, budget or arm definition changes.

The exact requested counterexample is tested over six anchors: P6 chooses
η=3, P5 η=3 fails fitting, another P5 rate succeeds. The rate remains 3 and
all six P5 fixed-rate arms remain unavailable. A selected P6 development
nonfinite outcome still stops. Final narrow closure review is required.
