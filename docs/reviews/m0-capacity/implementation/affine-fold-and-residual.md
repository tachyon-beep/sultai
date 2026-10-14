# Exact-source review: affine-fold-and-residual (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Scope: `src/sultai_screen/m0/fold.py` (59 lines, new in b79f145..e1df6e1; working tree identical to e1df6e1) and `tests/test_m0_modules.py::FoldTests` (lines 255-390). Preregistration: `docs/plans/2026-10-13-capacity-frontier-screen.md` §1 l.45-47, §2 l.107-108, §3 table l.183, §5 l.284, §6 l.333-335, §7 V0 l.380-383, §9 l.530 and l.551; Ruling R6 (`progress.md` l.50).

Evidence gathered (CPU only, read-only; scratch copies for mutations):
- `PYTHONPATH=src python -m pytest tests/test_m0_modules.py -k FoldTests`: 9 passed, 7 subtests passed.
- Measured on the test host (make_host(3000, 0)): interior fold error in float64 on the stored float32 params 6.96e-8 (b != 0) and 7.45e-8 (b = 0, whole map); same comparison through the float32 conv 1.67e-6 and 1.55e-6; border over-add with b != 0 reaches 0.21. r_aff: 0.0 exactly for the zero adapter, 4.93e-4 for b ~ N(0, 1) with W = 0, -1.28e-5 for affine(7).
- Nine single-point mutations of fold.py, each run against FoldTests (results in the table and findings below).

## Claims and evidence

| Claim (preregistration) | Source file:line | Pinning test | Verdict |
|---|---|---|---|
| §2 l.107-108: "the exact fold of a fitted M-aff into `tail[0]`" | fold.py:24-49 (`fold_affine`): new host whose `tail.0.weight`/`tail.0.bias` absorb the adapter; every other tensor is loaded unchanged (l.32, 41-42, 47) | `test_only_tail0_weight_and_bias_change_and_original_untouched` (fails if any other tensor changes, if dtype/shape changes, or if the input host is mutated) | Met |
| §3 l.183 / §5 l.284: M-aff is u = z + W z + b, "the same two-tensor adapter [as M-sq] differing in one flag" | fold.py:15-21 (`_affine`) requires exactly two tensors, `nonlinear=False`, W [16,16], b [16]; fold.py:29-30 requires width 0, where `read` is `base` (host.py:68 Identity bottleneck; fit.py:31-32 `base + W read + b`), so u = z + W z + b holds | `FoldTests.test_contract_violations` (nonlinear flag, wrong W/b shapes, three tensors, four-tensor module, widths 2 and 4 all refused) | Met |
| §3 l.183: "The fold K' = K(I + W) is exact everywhere (1e-6 on CPU)" [R6: 1e-6 in float64] | fold.py:36-38: K'[o,i] = sum_j K[o,j](I + W)[j,i] via `einsum("ojhw,ji->oihw")`, in float64, cast back to the stored dtype at l.41 | `test_zero_bias_fold_is_exact_everywhere` (float64 whole map <= 1e-6; float32 <= 1e-5 per R6). Mutation `"ojhw,ij->oihw"` (K(I + W^T)) fails 2 tests | Met under R6 (measured 7.45e-8 float64; 1.55e-6 float32, see note under the frozen-values table) |
| §3 l.183: "the bias fold is exact in the interior" | fold.py:39: c'[o] = c[o] + sum_{j,h,w} K[o,j,h,w] b[j] (the full-receptive-field sum, correct off the zero-padded ring) | `test_interior_exact_and_border_over_adds_when_bias_nonzero` (interior [1:15, 1:15] <= 1e-6 in float64). Mutations: bias fold dropped -> 2 fail; centre-tap-only bias -> 1 fails | Met (measured 6.96e-8) |
| §3 l.183 / §1 l.46-47: "over-adds on the one-pixel zero-padded border ring (60 of 256 positions)"; the border residual is measured, not asserted away | fold.py:25 docstring; no production code asserts or corrects the border (correctly: it is left in the folded host and shows up in r_aff) | Same test: asserts the ring has 60 positions and that the border differs by > 1e-5 when b != 0 | Met |
| §6 l.333-335: r_aff = L(host with `tail[0]` replaced by the folded weights) - L(host + fitted M-aff) on dev, mean cross-entropy, per anchor | fold.py:52-59: first term `mean_loss(folded, features_from_images(folded, data, development_indices), None)` (whole folded host from images), second `mean_loss(host, development, adapter)` (cached site + adapter); order and sign as specified; returns a Python float | `test_residual_finite_and_nonzero_with_bias` (value equals the explicit expression; sign flip -> fails; adapter dropped from the second term -> fails); `test_residual_zero_bitwise_for_zero_adapter` (0.0 exactly). "From images" is not pinned: see finding M1 | Met in source; partially pinned |
| r_aff is computed on the folded host of this (host, adapter) pair | fold.py:56-57: refuses a `folded` whose state hash differs from `fold_affine(host, adapter)` | `test_residual_rejects_mismatched_folded_host` (unfolded host and a fold of another adapter both refused). Mutation removing the check -> fails | Met |
| "on dev" | Caller-supplied: `development` features and `development_indices`; no check that they are the same examples. Call sites pass `cache.development` with `data.development` (worker.py:788, 1268-1270; host.py:349 builds `development` from `data.development`) | None (see finding M2) | Met at the call sites; not enforced in fold.py |
| §7 V0 l.380-383: |r_aff| <= X/4 = 0.005 on every development anchor | Not in this item: `FOLD_TOLERANCE` (identity.py:31) is applied in pilot.py:484 and confirmatory.py:311; fold.py only produces r_aff | (outside item) | N/A for this item; literal checked below |

## Frozen values checked

| Value | Prereg | Source | Match |
|---|---|---|---|
| Fold residual bound X/4 | 0.005 nats (§7 l.381); X = 0.02 (§7 l.388) | identity.py:31 `FOLD_TOLERANCE = 0.005`; identity.py:26 `X = 0.02` (not referenced by fold.py) | Yes |
| Fold exactness | "1e-6 on CPU" (§3 l.183); R6: evaluated in float64 on the stored float32 params, with a recorded float32 bound of 1e-5 | Test-only: test_m0_modules.py:291, 293, 301, 303 (1e-6 float64; 1e-5 float32). fold.py computes in float64 (l.36-39) | Yes under R6 |
| Border ring size | 60 of 256 positions (§3 l.183) | test_m0_modules.py:294-296 asserts 60 on the 16x16 site | Yes |
| M-aff count / shape | 272 = W [16,16] + b [16] (§3 l.183, §5 l.284) | fold.py:19 | Yes |
| Fold target | `tail[0]` (§2 l.108, §5 l.284) | fold.py:33-35 (`tail.0.weight`/`tail.0.bias`, 16 input channels, 3x3) | Yes |

Note on R6: the float32 forward of the folded convolution is 1.55-1.67e-6 from the unfolded one on CPU, above a literal float32 reading of "1e-6 on CPU"; the float64 fold error is 7e-8. R6 reads the §3 figure as a statement about the fold algebra. This does not touch a frozen value: the "fold tolerance" that §9 l.551 freezes is the V0 bound 0.005, which is unchanged, and no production gate uses 1e-6. Not BLOCKING. The task-8 report's suggestion to ledger the wording alongside R6 still stands.

Behaviour not asked for (item 4 of the brief):
- fold.py:44-45 constructs the result `Host` under `torch.random.fork_rng(devices=[])`, so the fold draws nothing from the caller's global RNG stream. This prevents an RNG-order change rather than introducing one; pinned by `test_fold_does_not_consume_global_rng` (removing the fork fails it).
- fold.py:56 re-runs the fold to check the state hash: a duplicate deterministic computation with no randomness and no receipt bytes.
- fold.py:29-30 refuses bottleneck widths 2, 4, 8. Every M0 host is built with width 0 (worker.py:279, 339, 360, 408, 1160-1206), so this narrows nothing the preregistration uses.
- No extra arms, no randomness, no Gate 3 path or receipt touched (fold.py imports only host/fit/data/m0.data helpers; r_aff reaches receipts through worker.py, outside this item).

## Findings

**M1 (Minor): "from images" is not pinned by any test.** fold.py:58-59; test_m0_modules.py:333-356.
- What: changing the first term to `mean_loss(folded, development, None)`, which evaluates the folded tail on the cached site instead of running the folded host from images, passes all 9 FoldTests. The test builds `development` with `features_from_images(self.host, ...)`, and the front is unchanged by the fold, so both routes give bitwise-equal features on CPU.
- Why: the brief's reading of §6 is "L(folded, from images)". On CPU the two routes agree, so the risk is small. The production cache comes from `cached()` on another code path, and on a card the two routes need not be bitwise equal; a refactor to the cached route would go unnoticed.
- Fix: in a test, pass a `development` whose `read`/`base` are altered (for example `+ 1.0`), and assert the result equals `mean_loss(folded, features_from_images(folded, data, idx), None) - mean_loss(host, altered, adapter)`. This discriminates the two routes; a `wraps=` patch of `features_from_images` would be weaker evidence.

**M2 (Minor): fold_residual does not bind `development` to `development_indices`.** fold.py:52-59.
- What: the two terms are evaluated on caller-supplied features and on caller-supplied indices separately. Nothing checks that they cover the same examples (same count, same labels, same order).
- Why: r_aff is defined as a difference over the same dev set. Both current call sites (worker.py:788, 1268-1270) pass `cache.development` with `data.development`, which `cached()` builds from the same indices (host.py:349). The code is correct today, but the guarantee lives in the callers.
- Fix: after l.58, `if len(from_images.labels) != len(development.labels) or not torch.equal(from_images.labels, development.labels.to(from_images.labels.device)): raise ContractViolation(...)`, plus a test that passes a mismatched index set.

**M3 (Minor): the docstring's "in float64" is not pinned.** fold.py:27, 36-39.
- What: computing the fold in float32 (dropping the `.double()` casts) passes all 9 FoldTests, because the float32-computed fold still sits well under 1e-6 in the float64 interior comparison.
- Why: this is not a preregistration claim. It is an implementation precaution the tests do not hold, recorded so that the docstring is not read as tested.
- Fix: optional. Assert that the folded `tail.0.weight` equals `einsum(K.double(), I + W.double()).float()` bitwise, or leave it untested and mark the docstring as non-normative.

No BLOCKING findings: nothing requires a frozen value to change for the code and the document to agree.

Counts: BLOCKING 0, Important 0, Minor 3.

## Dispositions (controller, 2026-10-14)

- M1–M3: deferred, recorded. Both call sites pass `cached()` features built from the same indices; `fold-check` (R13/R16) recomputes the residual from images in a separate process on card, which pins the route operationally.

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
