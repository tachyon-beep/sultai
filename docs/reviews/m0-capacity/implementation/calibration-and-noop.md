# Exact-source review: calibration-and-noop (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Method: static read of calibrate.py and CalibrateTests against prereg §2 (l.100-110), §4 "Published no-op" (l.216-223), §5 (l.281-282). Tests were not executed: no torch environment is available to this read-only session (collection fails with ModuleNotFoundError). Test-sensitivity statements are by inspection.

Note: the brief cites prereg lines 271-325 for R-bn / R-temp / R-cal; those lines hold the §5 arm table, and the R-bn/N0 text is at l.216-223 and l.281-282. The prereg does not mention LBFGS, chunk 128, momentum None, or "fixed order"; the brief attributes them to the plan (Task 9). They are implementation detail within the prereg's wording, not contradictions.

## Claims and evidence
| claim | source file:line | pinning test | verdict |
|---|---|---|---|
| N0 = host after BN recalibration on the fit examples, no gradient, one pass (l.216-217) | calibrate.py:25-47 (torch.no_grad l.42, single loop over `indices`, fit examples supplied by caller) | test_recalibrate_is_cumulative_over_fixed_chunks_with_the_partial_chunk (num_batches_tracked==2, running mean equals mean of two chunk means; fails if momentum, chunking, or pass count change); test_recalibrate_changes_only_batchnorm_statistics_and_freezes (only BN buffers change, params untouched) | met |
| Choice if R-bn selection loss is lower than raw, else raw (l.217-218) | :109-119 strict `<`, ties raw | test_no_op_picks_recalibration_when_it_lowers_selection_loss; test_no_op_keeps_raw_when_recalibration_does_not_help; test_no_op_tie_keeps_raw (identical twin; `<=` would fail it) | met |
| Choice made on selection, never dev (l.218-219) | :115-116 uses `.selection` only | guarded_no_op wrapper returns SelectionOnly; `seen` length 2 asserted | met (SelectionOnly guard makes any other split access fail, by inspection of helper use) |
| R-bn changes the front, so selection features are recomputed per candidate (implicit in comparing the two hosts) | :115-116 `cached(raw)`, `cached(recalibrated)` | test_no_op_recomputes_selection_features_from_each_candidate (reads differ, equal to each host's own cache) | met |
| Recalibrated host is frozen/sealed; sealed input refused | :31-32, :46 | test_recalibrate_refuses_a_sealed_host_before_any_forward; ..._and_freezes | met (extra guard, harmless) |
| R-temp: one parameter, temperature, fit examples (l.222-223, 282) | :92-94, :60-89 bias=False | test_temperature_scales_with_the_logits (2x logits gives 2x t, t near 1 on calibrated sample) | met; see Minor 1 |
| R-cal: temperature plus 10-vector class bias, 11 numbers, fit examples (l.282) | :97-99 | test_class_bias_has_eleven_numbers_and_fits_no_worse_than_temperature | met |
| Published N0 binding (every later arm loads N0) | :122-139 | test_save_and_load_noop_bind_the_chosen_host | met (not asked verbatim by prereg l.216-223; follows §4 "every arm is fitted on N0") |
| Calibration applied as logits/t + b | :102-106 | test_apply_calibration_is_scaled_logits_plus_bias | met |
| Contract violations on bad logits/labels | :50-57 | test_calibration_contract_violations (shape, length, empty, NaN, label 10; not negative labels or non-long dtype) | met, partly pinned |

## Frozen values checked
| value | prereg | source | match |
|---|---|---|---|
| Recalibration passes | one, no gradient | one loop, no_grad | yes |
| Parameter count R-temp / R-cal | 1 / 11 | 1 / 1+10 | yes |
| Fit set | fit examples | caller-supplied indices/logits (not enforced in this module) | yes by contract |
| N0 tie rule | "else the raw host" (strict lower for R-bn) | `<` | yes |
| Chunk 128, momentum None, LBFGS 50 iterations/history 20/tolerances, strong_wolfe | not in prereg (plan-mandated per brief) | l.20-21, 39, 66-74 | n/a, no conflict |

No frozen margin, grid, seed block, hash or tolerance is touched by this item.

## Findings
- Minor 1 (calibrate.py:66-74; tests): LBFGS settings and convergence are not pinned. No test checks stationarity (gradient near zero) or that 50 iterations suffice; a production change to max_iter=1 or a different start could pass the scaling and "near 1" tests. Fix: add a test asserting the fitted loss gradient norm is below a small bound on a miscalibrated sample, and that the joint loss is no greater than the loss at (t=1, b=0).
- Minor 2 (tests/test_m0_modules.py CalibrateTests): the calibration contract test omits negative labels and non-int64 labels, so those two branches at calibrate.py:55 are unpinned individually. Fix: add the two cases.
- Minor 3 (calibrate.py:109-119): `no_op` does not itself check that `raw` and `recalibrated` are distinct objects or that both are sealed; recalibrate_bn mutates in place, so a caller passing the same host would get a tie and "raw". Not a prereg requirement. Fix: optional guard.
- Minor 4 (calibrate.py:43-45): a trailing chunk of size 1 would make a train-mode BatchNorm raise inside torch; behaviour with the real fit partition is unaffected unless its size is 1 mod 128. Not a prereg matter; note only.

No BLOCKING or Important findings. No extra arms, extra randomness, RNG-order changes, or receipt-byte changes on Gate 3 paths were found in this file (noop.pt is a new artefact, state_dict only). No frozen value needs to change.

## Dispositions (controller, 2026-10-14)

- Minor 1–4: deferred, recorded. LBFGS convergence and the label branches are noted; the plan-mandated BN recalibration chunk weighting (Task 9) goes to John at gate B.

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
