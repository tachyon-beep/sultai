# Dispositions of the two reviews of the M0 implementation plan (2026-10-13)

Plan: `docs/plans/2026-10-13-m0-implementation.md`. Spec: the frozen preregistration v3 (not edited by any disposition here).

## Reality review at 91412df (`implementation-plan-reality-review-91412df.md`)

Ten defects, all adopted at `e75b569`: Supervisor `module=` and `Job.extra`; `oracle()` call bytes unchanged; momentum-buffer resume contract; `train_epochs` over `nn.Module` with a hashed `Host`; import-clean `m0/provenance.py`; width-tuple check at load; `sealed` flag as the frozen predicate; `pilot.py:514`; `from_json`/`from_dict` as boundaries; the nine untouched Gate 3 tests enumerated; `with_static` assertion corrected.

## Astra scientific review by Codex at 308181b, reasoning effort high (`implementation-plan-astra-codex-review-308181b.md`)

| Item | Disposition | Where in the plan |
|---|---|---|
| B1 interval contract | Adopted: normal per-example bound on the v4 paired SE; order-statistic percentile anchor bootstrap (500th/9,500th of 10,000); `statistics.median`; sample SD; achieved half-width; exact binomial sign-test bound by order statistics with zeros counted against and `None` below n = 6; aligned per-example loss and correctness vectors in `AnchorResult`; numerical reference tests | Task 13; open item (g) |
| B2 frontier quantity | Adopted: fitted response and sign stated (loss decrease per doubling); within-width seed resampling; joint bootstrap of gain and slope for the fraction interval; `None` with reason on a non-positive slope; descriptive all-10-epoch-width fit, cooled point excluded | Task 13 |
| B3 slicing fold | Adopted: `activation_means` from the fit `read` on every folded-cut path incl. random and ablation cuts; `Cut` records survivors; stable tie-break; nonzero-removed-unit fixture; `fold_residual` takes the fitted adapter and recomputes both terms | Tasks 7, 8 |
| B4 S-static contract | Adopted: `static_hash` over host and module; `save_static`/`load_static`; no-refit term on the complete model; refit as a warm start from the learned module with a fresh-init refit beside as descriptive (open item (f)); both in the worker roster with M-sq \| T-cool | Tasks 11, 16 |
| B5 N0 state and caches | Adopted: each no-op candidate evaluated on features recomputed from itself; `save_noop`/`load_noop` bound by hash and consumed by every later arm; the finished host applies its own N0 rule before its module fit; R-all post-fit recalibration ordered after the complement check and recorded separately; recalibration chunk order and cumulative statistics stated | Tasks 9, 10, 16 |
| B6 missing reports | Adopted: harm inventory (inherited v4 §7, `HARM`), re-tune fraction, unfolded slices, leave-one-out ceiling cut, random D2 and D2s, survival curves with explicit undefined-ratio handling; h = 16/4/2 refits as one 800-step run at the selected M-wide rate | Tasks 13, 16; self-review §6 list |
| B7 schedule fill and measured costs | Adopted: `04-schedule.json` raise-only extension with M-small-B doubled in step; `Workloads` of measured units replaces literals; projection = charged plus remaining; core-over-ceiling stop in stage 4 before any V1 field | Tasks 15, 17 |
| B8 α fallback | Adopted: `ALPHA = 0.1` pinned; `materiality_report` descriptive only; a P1 failure at 0.1 is an instrument hold, not a fallback | Task 12 |
| B9 M0 binding of launcher and stops | Adopted: `m0_stop_report` with the M0 identity; Task 18 passes `module=`; `CONFIRMATORY_ARMS` whitelist enforced through `--arms`; `static_admitted` published in `05-spreads.json`; 14 h owner authorisation as a certificate field; `--restart` bound to a disposition hash and the prior custody receipt; `optimizer.pt` and `static-module.pt` hash-bound in receipts | Tasks 16, 17, 18 |
| B10 GPU equality claims | Adopted: a five-item GPU acceptance matrix in pilot stage 1 on both cards, each booked; `losses` keeps chunk-shaped cross-entropy and is not rewritten; the fold's interior equality stays a tolerance claim, its endpoint residual re-measured on GPU | Tasks 5, 17 |
| N1–N8 | Noted; N1's endpoint wording (last executed rate 0.05/6,250, zero at the boundary) adopted in Task 4's cooldown definition | Task 4 |
| A1 co-tenancy replay | Reported to John as amendment-2 candidate 1; the plan does not choose | Plan tail |
| A2 programme claim vs V1 gate | Reported as candidate 2; Task 14 implements the frozen verdict and states its restricted meaning | Plan tail |
| A3 example-pass arithmetic (4,800,000 vs 4,915,200) | Reported as candidate 3; receipts record actual passes and measured wall | Plan tail |
