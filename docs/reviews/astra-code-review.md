# Independent bounded TCD code review

Reviewed 2026-10-09: staged `src/sultai/repair.py`, `src/sultai/smoke.py`, `tests/test_repair.py`, and `docs/SMOKE.md`. One independent correctness pass, with no source edits. TCD is now resolved as **technical concept demonstrator**.

**Disposition: no material mathematical, leakage, or accounting defects found in the implemented bounded CPU instrument.** No corrective source change is required by this review.

- The adapter has 272 learned values and correctly computes the residual with fixed nonlinear features. Ridge fitting solves the intended site-residual problem, including the intercept's ridge penalty, with correct matrix/output orientation.
- Candidate formation reads only conditioning pairs; selection reads only selection pairs; all declared test arms execute after candidate and winner freezing. No test feedback path was found. IDs are checked for within-episode duplication and overlap; the outer guard rejects either shared group identifier. The documents correctly avoid treating synthetic group names as evidence of independent host families.
- K=3 accounting matches execution: 192 conditioning-example uses, 72 selection forward queries, and 96 test forward queries per episode. Repeated evaluation of the first candidate when it wins is genuinely executed and charged. The 816-value candidate bank, 272-value single repair, and absence of retained optimizer state are correctly described; this is not reported as physical model removal.
- The scalar shadow-probe diagnostic is separate from the adapter benchmark. Documentation clearly limits the measured method to paired ridge optimization, with oracle targets and no generator, delayed handover, or parameter-efficiency result.

Validation performed from the staging root with `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1`:

1. `python3 -m unittest discover -s tests -v`: all 12 tests passed.
2. `python3 -m sultai.smoke`: successful deterministic-format JSON; planted test MSE 0.06896343793009375 without repair and 1.2123623237933552e-20 for the first candidate and selected candidate; all healthy-control test losses zero. Counts match the documentation.
3. Independent direct check of `(XᵀX + ridge I)B − XᵀY` through the fitted residual gradients at all three declared ridges: maximum absolute stationarity residuals 9.13e-15, 1.02e-14, and 7.11e-15, all below 1e-12.

Limits already appropriately disclosed: normal equations and absolute-tolerance rank estimation are sufficient for this small bounded fixture, not general numerical guarantees; only within-episode synthetic fitting is exercised by the smoke. Missing original attachments remain a separate research-import blocker. They were not inspected or required to verify this code path.
