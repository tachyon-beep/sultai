# Exact-source review: modules-slicing-known-answer (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Reviewer process note: an attempted `uv run --frozen` (to execute the tests) created an empty, gitignored `/home/john/sultai/.venv/` (skeleton only, no packages, 2026-10-14 06:05) before failing on the missing lockfile. Removal was refused by the permission classifier, so it is still there; the controller or John should delete it. No tracked file, index or branch was touched. The tests were **not executed**; every "pinning" judgement below is static, by reading the assertions against the source.

Scope: `src/sultai_screen/m0/modules.py` (new file, 169 lines) and `tests/test_m0_modules.py::ModuleTests` at e1df6e1, against the prereg §2 lines 106-109, §3 lines 175-200, §5 lines 291-292 and 318-319, §7 lines 375-379. Callers (`m0/worker.py`, `m0/pilot.py`, `fit.py`) were read only where the claim depends on them, and are marked as such.

## Claims and evidence

| Claim (prereg) | Source file:line | Pinning test (and the production change it catches) | Verdict |
|---|---|---|---|
| §3: "rank hidden units by contribution ‖V[:, i]‖₂ · RMS over the fit examples of tanh(U z + c)_i" | modules.py:50-60 (`_moment`, float64 mean over examples and positions, CHUNK 128), 68-72 (`contribution`) | `test_fold_adds_removed_unit_means_and_unfolded_omits_them` compares `contribution` to an independent float64 `‖V‖·RMS` within 1e-6; catches dropping the norm, the RMS, the square root, or using row norms of V | Satisfied. RMS over examples and positions equals the RMS over examples of the per-example RMS, so including positions is the only coherent reading |
| §3: "remove the lowest" (keep the h largest) | modules.py:120-124 (`rank_cut`, `argsort(-contribution, stable=True)[:h]`) | `test_known_answer_...` (survivors == NONZERO), `test_ties_resolve_to_lower_index` (survivors (1, 3)); catches a reversed sort | Satisfied |
| §3: "slice U, c, V to the survivors" | modules.py:75-85 (`_slice`) | `test_fold_adds_...` asserts `weights[0..2]` equal the indexed originals for folded and unfolded; `test_survivors_sorted_for_every_cut` for `weights[0]` of all three cuts | Satisfied |
| §3 [Ruling R5]: "fold the removed units' means exactly into the bias, d ← d + Σ_removed V[:, i] · mean(a_i)" | modules.py:88-97 (`_folded`, float64 fold, cast back to d's dtype), 100-103 (`cut`), 63-65 (`activation_means`) | `test_fold_adds_...`: folded d equals d + Σ_removed V[:, i]·mean_i within 1e-6, the term being > 1e-3; catches omitting the fold, folding survivors, wrong sign, or using RMS instead of the mean. Only `rank_cut`'s fold is asserted (see finding I1) | Satisfied |
| §3: "print the parameter count" | not in modules.py; `Adapter.count()` (fit.py:47-48) is the count, recording is the caller's (`worker.Collector.cut`, worker.py:684-695) | `test_counts_are_33h_plus_16` (folded and unfolded counts) | Count satisfied here; printing is the caller's obligation, out of item scope |
| §3: "Report the unfolded slice beside it" | modules.py:28-34 (`Cut.unfolded`), 106-110 (`cut_unfolded`), 113-117 (`_cut`) | `test_fold_adds_...` asserts unfolded d is bitwise the original d and that `cut_unfolded` equals `Cut.unfolded`; catches folding into the unfolded slice | Satisfied |
| §3/§5: "a random cut to the same h with the same fold" | modules.py:127-130 (`random_cut`, `randperm(total, generator(seed, "random-cut"))[:h]`, then the shared `_cut`) | `test_known_answer_...` pins seed 0, label "random-cut", draw order and survivors; `test_survivors_sorted_...` pins width and sorting. The fold on the random cut is **not** asserted (I1) | Satisfied in source; fold pinned only through the shared `_cut` |
| §3: "a leave-one-unit-out loss ablation on the fit examples as the ceiling for the cheap criterion" [R5: removal by the mean fold] | modules.py:141-151 (`leave_one_out`), 133-138 (`ablation_cut`, keep the h largest removal costs) | `test_leave_one_out_shape_and_zero_for_silent_unit`: ablation[0] equals loss(cut([1,2,3]) with fold) − full loss to 12 places (catches zeroing instead of folding, a sign flip, or `sum` for `mean`); silent unit ≈ 0. `test_survivors_sorted_...` pins `ablation_cut` = `argsort(-ablation)[:8]` (catches keeping the smallest) | Satisfied. "On the fit examples" is the caller's: worker.py:858 passes `fit` |
| §5 / §6: survival cuts h ∈ {16, 8, 4, 2}, counts 33h + 16 = 544, 280, 148, 82 | no literal in modules.py; `rank_cut(module, read, h)` for any h; worker.py:133 `SURVIVAL_CUTS = (16, 4, 2)` + `SMALL = 8`, stats.py:34 `SURVIVAL_WIDTHS` | `test_counts_are_33h_plus_16` asserts [544, 280, 148, 82] for h in (16, 8, 4, 2); catches a slice that leaves extra tensor entries | Satisfied; the production h literals live outside this item (M2) |
| §7 V0: "a 64-unit module with 56 columns of V exactly zero must, after the criterion cut to 8 with fold, reproduce its output within 1e-6, a random cut must not, and a negative control must fail" | modules.py `rank_cut`, `random_cut`, `cut`; the runtime V0 check is pilot.py:136-140, 237-264 (same fixture, CPU) | `test_known_answer_rank_cut_preserves_output_random_and_wrong_cuts_do_not`: asserts exactly 56 zero columns, rank survivors = NONZERO, folded output within 1e-6, random survivors ≠ NONZERO and output differs > 1e-4, wrong cut (8 zero-V units kept, nonzero ones folded) differs > 1e-4. Catches a reversed ranking, a slice of the wrong axis of V, or a broken fold of zero columns. Runtime path pinned by `tests/test_m0_pilot.py::test_slicing_known_answer_passes_on_the_cpu_fixture` (outside this item) | Satisfied. The "must not" thresholds (> 1e-4) are stricter than the negation of 1e-6, so they cannot pass a reproducing cut |
| §5 extended 4: "Lottery cut, the surviving 8 units' initial weights retrained with the module grid" | modules.py:162-169 (`lottery_start`: `initial_adapter(seed, hidden=64, nonlinear=True, expanded=False)` sliced, unfolded); caller worker.py:884-889 passes the anchor seed, `rank8.survivors` and the module grid | `test_lottery_start_slices_regenerated_initial_tensors` asserts bitwise equality with the indexed `initial_adapter(21, hidden=64, …)` and rejects empty, unsorted and out-of-range keeps | Satisfied in source; that this equals M-wide's actual step-0 tensors holds by construction (fit.py:275-276 uses the same call) but is not asserted (M1) |
| §3 M-exp: "W = W₁W₂, W₁ 16×64, W₂ 64×16, contracted exactly to M-sq after fitting" | modules.py:154-159 (`contract` → `Adapter((w1 @ w2, b), nonlinear=True)`); shapes from fit.py:70-71 | `test_contract_reproduces_expanded_output`: two tensors, nonlinear, bias bitwise, output within 1e-6, three-tensor requirement; catches `w2 @ w1`, an affine flag, or a dropped bias | Satisfied |
| §3 frontier table: tail widths per s | modules.py:17-23 (`FRONTIER_WIDTHS`) | `test_frontier_widths_table` (exact dict); parameter counts per s pinned in `tests/test_m0_host.py` (outside this item) | Satisfied |

## Frozen values checked

| Value | Prereg | Source | Match |
|---|---|---|---|
| s = 0.25 tail widths | 16, 16, 32, 32 | modules.py:18 `(16, 16, 32, 32)` | yes |
| s = 0.5 | 32, 32, 64, 64 | modules.py:19 | yes |
| s = 0.75 | 48, 48, 96, 96 | modules.py:20 | yes |
| s = 1.0 | 64, 64, 128, 128 | modules.py:21 | yes |
| s = 1.5 | 96, 96, 192, 192 | modules.py:22 | yes |
| M-wide hidden width / lottery width | h = 64 | modules.py:24 `LOTTERY_WIDTH = 64` | yes |
| Sliced counts 33h + 16 | 544, 280, 148, 82 for h = 16, 8, 4, 2 | arithmetic of `Adapter.count()` on the sliced four tensors; pinned by test | yes (no literal in source; production h literals in worker.py:133-135, stats.py:34) |
| Known answer module | 64 units, 56 zero V columns, cut to 8 | test fixture `NONZERO` (8 units), 64-unit `sparse_module`; runtime fixture pilot.py:136 `SLICING_UNITS` (identical tuple) | yes |
| Known answer tolerance | 1e-6 | test `assertLess(..., 1e-6)`; pilot.py:258 `rank_d <= 1e-6` | yes |
| Fold formula | d ← d + Σ_removed V[:, i] · mean(a_i) | modules.py:96-97 | yes |
| Contribution criterion | ‖V[:, i]‖₂ · RMS of tanh(U z + c)_i | modules.py:70-72 | yes |
| M-exp shapes | W₁ 16×64, W₂ 64×16 | fit.py:71 (shapes), modules.py:158-159 | yes |

No frozen value would have to change for the code and the document to agree.

## Findings

**I1 — Important.** `tests/test_m0_modules.py:136-147` (`test_survivors_sorted_for_every_cut`) and `:90-109`. *What:* the mean fold is asserted only for `rank_cut` (`test_fold_adds_...`). For `random_cut` and `ablation_cut` the tests check survivors and `weights[0]` only, and the known answer's random assertion (`> 1e-4`) passes with or without the fold. *Why:* §3 and §5 require "a random cut to the same h **with the same fold**", and D2's random-rank versions and the leave-one-out ceiling are read from these cuts. Today the source is correct because all three route through the shared `_cut` (modules.py:113-117), so a change to `_cut` is caught; but a production change that built either control with `cut_unfolded` for both fields (or skipped `activation_means`) would pass every test. *Fix:* for each of the three cuts in `test_survivors_sorted_for_every_cut`, assert `folded.weights[3]` equals `adapter.weights[3] + Σ_removed V[:, i]·activation_means[i]` within 1e-6 and that `unfolded.weights[3]` is bitwise the original d.

**M1 — Minor.** `src/sultai_screen/m0/modules.py:168`; `tests/test_m0_modules.py:192-208`. *What:* `lottery_start` regenerates "the initial weights" by calling `initial_adapter(seed, hidden=64, nonlinear=True, expanded=False)`, and the test compares against the same call. Equality with M-wide's actual step-0 adapter holds only because `fit_sgd` (fit.py:275-276) makes the identical call with the anchor seed. *Why:* a later change to `fit_sgd`'s init path (label, scale, device order) would desync the lottery start from the real initial tensors without any test failing. *Fix:* add a test that `fit_sgd(..., hidden=64, steps=1, checkpoints=(1,))`'s pre-step adapter, or `initial_adapter` as reached through `grid(hidden=64)`, equals `lottery_start(seed, tuple(range(64)), cpu)` bitwise; or expose the init draw used by `fit_sgd` and call it from `lottery_start`.

**M2 — Minor.** `src/sultai_screen/m0/modules.py` (absent). *What:* the item names "the survival cuts 16/8/4/2", but this source holds no such literal; production widths are `SURVIVAL_CUTS = (16, 4, 2)` plus `SMALL = 8` in worker.py:133-135 and `SURVIVAL_WIDTHS = (16, 8, 4, 2)` in stats.py:34, both outside this diff. The ModuleTests tuple `(16, 8, 4, 2)` is a test-local literal, so it pins the count arithmetic but not the production widths. *Why:* the frozen h set is pinned by whichever item reviews worker.py and stats.py, not by this one. *Fix:* none here; the controller should confirm that the worker or stats item pins `SURVIVAL_CUTS`, `SMALL` and `SURVIVAL_WIDTHS` against {16, 8, 4, 2}.

Non-findings, recorded for the controller:
- Behaviour not in the prereg: ties go to the lower unit index (stable argsort) for rank and ablation cuts; this is deterministic, consumes no randomness and is pinned by `test_ties_resolve_to_lower_index`. The random cut draws from a new isolated stream `generator(seed, "random-cut")`; it does not touch the global RNG or the `oracle-init`/`oracle-order` streams, so no existing RNG order changes. No Gate 3 path or receipt byte is touched by modules.py.
- Ruling R5 (leave-one-out removes by the mean fold, not by zeroing) is an interpretation of text that does not say which, not an amendment; it is pinned to 12 places.
- `_moment` takes tanh in the adapter dtype (float32) and accumulates in float64; an empty `read` would divide by zero. Neither matters at 2,000 fit examples and neither is specified by the prereg.

## Dispositions (controller, 2026-10-14)

- I1 (Important): FIXED (item G1). M1: FIXED (item G2). M2: FIXED (item G3: survival width literals pinned in the worker and stats tests).

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
