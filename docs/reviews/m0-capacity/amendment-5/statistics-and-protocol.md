# M0 amendment 5: independent exact-source review, statistics and protocol (2026-10-14)

Reviewer: experiment-statistics reviewer (counterfactual-statistics pack, SME protocol). Scope: branch `nyx/m0-amendment-5` in `/home/john/sultai-a5`, diff `main (0d4179a)..4a2a005` (extended from 712b09a at the controller's request; the final reviewed commit is `13989d9`). Read-only; no GPU; no CIFAR batch file opened. Tests were run on CPU from the worktree source (`PYTHONPATH=/home/john/sultai-a5/src`, no cache writes). The three pilot reviews were read from `nyx/m0-pilot-evidence:docs/reviews/m0-capacity/pilot/`; the pilot's `04-admission.json`, `06-confirmatory-count.json`, `00-opening.json` and `pilot-result.json` from the same branch.

## Summary

Accept. 0 blocking, 0 major, 2 minor, 8 observations.
Every pinned hash, the composed identity `m0-v4.3`, the admitted source hash and every fixture literal recompute exactly; the six earlier documents are byte-identical to `main`; no margin, rule, arm set, seed block or ceiling moves; the Gate 3 files are untouched.
Most important result: A5.1's booking closes two independent ways from the pilot's own receipts. The launcher's uplift is 3 × 0.004380737 × (3,200 − 800) = 31.54 s per anchor (1,009 s at n = 32), and re-projecting the pilot's kernels under the new `refit-bound` unit moves the kernel per-anchor term by exactly the same 31.54 s (688.88 → 720.42). With the uplift the measured projection is 25,271 s plain and 28,256 s static against 43,200 / 50,400 s.
The two minors are a narrative caveat on the projected half-widths and a test-coverage gap on the corner classifier; neither is a condition.

## Findings

**F1. Minor. The projected half-width is a point projection from a six-atom empirical distribution and carries no interval on itself.**
Evidence: `src/sultai_screen/m0/stats.py:400-411` (`anchor_bound(size=)` draws `(samples, n)` indices from the six values under `generator(seed, "anchor-bootstrap")`), `:419-425` (`projected_half_width`), `pilot.py:1135-1156` (`half_widths(quantities, n)`, labelled `projected_half_width_development_only: True`, `projected_half_width_anchors: n`). Reasoning: at n = 32 the replicate median is almost always one of the six atoms or a midpoint, so the 5/95 order statistics are discretised; and the six values' own SD uncertainty (statistics review F7: 0.62× to 2.45×) is not propagated. This is exactly the method the statistics review F5 used to get "about 0.0021, 0.0008, 0.0071, and D1 0.0014", it is labelled a projection, and nothing reads it: `confirmatory.py:505-506` and `:561` call `anchor_bound`/`achieved_half_width` without `size`, on confirmatory values. `half_widths` keeps D2, D3, D5 achieved widths and reasons value-identical (same call, same seed, same key order; only three keys are added to the record). BOOTSTRAP_SEED reuse across the `(samples, 6)` and `(samples, n)` draws shares a stream prefix; that correlates two descriptive numbers and affects no inference. Test `test_m0_stats.py:447-455` pins `projected == achieved` at n = 6 to 15 places and monotone narrowing. Confidence: high. Action: none in code; the gate-C narrative should call it a point projection with no interval of its own.

**F2. Minor (test only). The three-way corner classification is not exercised by any test.**
Evidence: `worker.py:218-233` (`_position`, `corner`): `sorted(set(rates))` so `MODULE_RATES = (1.0, 0.3, 0.1)` classifies 0.1 lowest, 1.0 highest, 0.3 interior; steps 200 first, 3,200 last, 800 interior; M-small-B takes its own candidates' axis (`worker.py:706-708` passes `c.steps for c in fit.candidates`, which are 400/1,600/6,400), so 400 first, 6,400 last; one-value axis gives "single"; equality is exact because `fit.best` is one of the candidates. `edge_of_grid` is untouched (`worker.py:705`, `:1082`, `fit.py:440-441`). `tests/test_m0_worker.py:524-531` only reaches "single" (rates patched to one value) and first/last; `tests/test_m0_pilot.py:157` uses the literal `interior@last`. Classification verified by reading, not by a test. Confidence: high that the code is right; the gap is coverage. Action: a short unit test of `worker.corner` with `MODULE_RATES`, `RETUNE_RATES`, `CHECKPOINTS` and `CHECKPOINTS_B`; recommended before gate B, not a condition (no verdict or preregistered flag reads the field). Note that taking it changes `tests/`, hence `source_identity()`, hence the disposition's admitted hash (F10).

**F3. Observation. The uplift is applied to the static per-anchor term, and should be.**
Evidence: `confirmatory.py:303-306`: `per` selects `static_per_anchor_gpu_process_seconds` when static, and `+ uplift` is added to whichever is read; `CONFIRMATORY_ARMS | STATIC_ARMS` contains the three survival refits (`stats.py:78-98`, `costs.py:110-112`), so the static measured term carried the same three 800-step refits. The published `survival_refit_uplift_seconds` is the one scalar (`confirmatory.py:312`). Note the term being raised is the R30-scaled whole-child charge (pilot `scale` 0.865) while the uplift is an unscaled kernel delta: conservative by about 4 s per anchor. Confidence: high. Action: none.

**F4. Observation. The v4.2 admission path binds by the (version, sha) pair only.**
Evidence: `confirmatory.py:101` `ADMITTED_PILOT_IDENTITIES = {"m0-v4.2": "a2c1e55d…"}` equals certificate 3's `protocol_sha256` and my recomputation of `sha256("m0-v4.2\n" + six hashes)`; `pilot_protocol` (`:212-230`): same identity → `same_protocol: True`; current version with any differing key → `PilotHold` naming the key (covered by `tests/test_m0_pilot.py:1445-1446`, `protocol_sha256 = "0"*64`); v4.2 with the literal sha → admitted with `admitted_by` published; v4.2 with another sha or v4.1 → hold (`:1486-1493`). No other past identity is admitted. The pilot result's six document hashes are not compared to the current six on the admitted path; the composed sha implies them for a genuine result. Confidence: high. Action: none.

**F5. Observation. The V0 no-op replay is charged at run time but not in the admission projection.**
Evidence: `confirmatory.py:348-368` runs `noop-3000-gpu{0,1}` on the `replay-3000-gpu0` host and holds on any of `raw_sha256`, `recalibrated_sha256`, `noop_sha256` differing; `fixed` (`:302`) is the pilot's `fixed_v0_gpu_process_seconds` (552.58 s), measured before this child existed. Two children at the pilot's N0 line (4.71 s each): about 10 s against 43,200. Confidence: high. Action: none; a sentence in the narrative.

**F6. Observation. `environment_vs_pilot` will show the four new fields as `pilot: null, equal: false` for the pilot of record.**
Evidence: the pilot's `00-opening.json` environment carries only `cuda, numpy, numpy_wheel, torch, torch_wheel`; `confirmatory.py:333-342` reports every key of either record; `provenance.py:55-80` adds `python, driver, cudnn, gpu_names`, recorded, never compared (`tests/test_m0_provenance.py:44-78`). The opening record is hash-listed like every pilot receipt (`confirmatory.py:267-270`, test `:1432-1440`). Confidence: high. Action: the narrative should say the four are new fields, not drift.

**F7. Observation. The static host's no-op receipt publishes both losses but not the fit-index hash.**
Evidence: `worker.py:492-512` (`recalibrate_static(unsealed, data.images, seen_fit(data))`, `raw_selection_loss`, `rbn_selection_loss`, no `fit_indices_sha256`); N0 and T-cool do (`worker.py:430-444`, `:559-585`, `tensor_sha` at `:210-211` hashes the same tensor object the recalibration consumed). The amendment's item 2 parenthetical names N0 and T-cool, and determinism F1's action names the noop receipt, so the text and code agree. Confidence: high. Action: optional one-line addition for uniformity.

**F8. Observation. The N0 losses now come from the deciding object and are the same values as before.**
Evidence: `calibrate.py:111-130` `NoOpChoice` carries `raw_loss`/`recalibrated_loss` computed on `cached(host, data).selection`; `host.py:335-349` and `data.py:30-40` chunk by 128 over the same normalised front/bottleneck, so the previously published `features_from_images(..., data.selection)` losses are the same quantities. `tests/test_m0_modules.py:502-503` pins label ↔ strict `<`. Confidence: high.

**F9. Observation. `environment_identity()` now shells out to `nvidia-smi` in every child.**
Evidence: `provenance.py:41-53` (20 s timeout, absent on failure), `:69-71` `cudnn.version()` (library load, no CUDA context). About 0.1 s per child inside its charged interval; the carried pilot kernels do not include it. Confidence: medium on the per-call cost (not measured). Action: none.

**F10. Observation. The admitted source hash is the tree at 4a2a005.**
Evidence: `source_identity()` at 4a2a005 returns `93501f9ab102c7cde7ec05451001cfa9e19c3cf07ebd88730fc8b6a38bd3d383`, equal to the disposition. If `13989d9` changes any file under `src`, `scripts`, `tests` or `pyproject.toml`, the disposition must be re-pointed again. Not a finding; a limit.

### Protocol and text-versus-code (clean)

- A5.1 worker: `worker.py:967` `steps=wide.best.steps` is the only change inside `_wide_arms` (diff hunk `@@ -935,7 +964,7`); the single-run record publishes `rate`, `steps`, `step0_selection_loss` (`:718-731`); `SURVIVAL_REFIT_STEPS` removed and its absence pinned (`tests/test_m0_worker.py:329`; the refit's steps pinned to M-wide's at `:518-519`); harm label becomes `<rate>@<selected steps>`, parsed as before by `candidate_rate`.
- A5.1 booking: `receipts.py:49-55` `GRID_STEPS["refit-bound"] = CHECKPOINTS[-1]` (3,200); `costs.py:110-112` `("grid:refit-bound", "evaluation")`; `confirmatory.py:187-209` `workloads_of` pops `refit-single`, sets `refit-bound = module / len(MODULE_RATES)` (same `per_step × 3,200 × ratio` the pilot would have derived) and `uplift = 3 × max(0, bound − single)`. Matches the amendment's "three refits times (last checkpoint − 800) steps at the measured per-step module kernel".
- No frozen value moves: `identity.py` diff is confined to `AMENDMENT_5_*`, `PROTOCOL_VERSION`, the composition line and the two dict entries; `X, X2, FOLD_TOLERANCE, CERTIFIED_FRACTION, S_STATIC_SD_LIMIT, M0_CONFIRMATORY_SECONDS (43,200), M0_CONFIRMATORY_STATIC_SECONDS (50,400), MODULE_RATES, RETUNE_RATES, CHECKPOINTS, CHECKPOINTS_B` unchanged; `stats.CONFIRMATORY_ARMS`, `BLOCKS` untouched.
- Gate 3 paths: `git diff --stat main..4a2a005 -- src/sultai_screen/{pilot,screen,fit,host}.py src/sultai_screen/m0/static.py` is empty; nothing outside `src/sultai_screen/m0/` changed under `src`.
- Items 3, 4, 5: `replay_noop` as above; `certificate_sha256` and `custody` on both terminal records including the stop paths (`pilot.py:1311-1323`, `confirmatory.py:684-699`; `certificate_sha` is bound before each `try`); `Run.run` publishes `<job>.exit.json` from the ledger in a `finally` (`pilot.py:360-375`); `provenance()` records the eight named variables, absent as `null` (`provenance.py:87-89`).
- "Reported, not changed" against the reviews' Action lines: statistics F1 "none before the confirmatory; report the bound beside the edge-of-grid counts. X must not move"; F3 "none required"; F4 "none; note it in the read"; F9 "wording only"; training dynamics F2 "report; no amendment"; F7 "report; a grid shift … is not warranted"; determinism F6 "none required; note the limit". Forward map: statistics F2 and TD F1 → A5.1; determinism F1 → items 2 and 3; F2 → item 1; F3 → item 4; F4 → item 2; F5 → item 5; statistics F5 → item 6; TD F3 with statistics F1 → item 7. No reviewer action is orphaned; TD F5 appears under the read items.

## Recomputations

| Quantity | Expected (certificate / code / test) | Found | Match |
|---|---|---|---|
| sha256 preregistration v4 | `c66599459adc898f96322ab11dfa3c6737cd3559e06c9432d79e377e00c03fd5` | same | yes |
| sha256 STEERING | `08a9e1aaf79d3048a5d6f6f4d6138a9c7049f32b61c0250e1505773a27fa3ffd` | same | yes |
| sha256 gate-ladder amendment 1 | `af12d474a97aad83bc63405034fec3716170f7652f34364588750732c97b6a89` | same | yes |
| sha256 gate-ladder amendment 2 | `b4740b147a4a1ced52a6f0721215679c1567078260d5c11a92c35fa2065c2d77` | same | yes |
| sha256 M0 amendment 3 | `6dcfd9f2a781dc56d39c6e4fe548fc4728c2c743a8e5a978d0e4b9741459e862` | same | yes |
| sha256 M0 amendment 4 | `d8dd571bb39718946af06a442ec74302b704c1bede090c83eeaed9d59dcdcc34` | same | yes |
| sha256 M0 amendment 5 | `0f6c2fa3bf8240e6f52a038a25c0d5d543e8f0592eeb2681a1072c2df1a5b6ed` | same | yes |
| Composed `m0-v4.3` (freeze 5, `identity.py:32-34`, `test_m0_identity.py:100`) | `bde550df1ac6ca179dd1fe18ed3a2d040eef1225522f913fd46b983fb24cc395` | same (`printf` recipe from the certificate) | yes |
| Composed `m0-v4.2` (certificate 3, `ADMITTED_PILOT_IDENTITIES`, `confirmatory.py:101`) | `a2c1e55deffb144c9ab596ed7817bc6546f88ac045a02fc0447ae25896db6007` | same | yes |
| Six earlier documents vs `main` (`git diff main --`) | 0 bytes | 0 bytes | yes |
| `tests/test_m0_identity.py` pins seven files + literal `PROTOCOL_SHA` | seven | seven (`DOCUMENTS` 52-60; per-file asserts 84-89; literal 100) | yes |
| Source hash at 4a2a005 (disposition) | `93501f9ab102c7cde7ec05451001cfa9e19c3cf07ebd88730fc8b6a38bd3d383` | same (`source_identity()`) | yes |
| `GRID_STEPS["refit-bound"]` | 3,200 | 3,200 | yes |
| `per_anchor(w, CONFIRMATORY_ARMS)` | 1,071.5 | 1,071.5 | yes |
| `per_anchor(w, CONFIRMATORY_ARMS ∪ STATIC)` | 1,181.5 | 1,181.5 | yes |
| `stage_seconds("core", w)` | 6,729 | 6,729 | yes |
| `project_confirmatory(32)` | 35,788 | 35,788 | yes |
| `with_static(32)` | 39,308 | 39,308 | yes |
| `project_confirmatory(42)` | 1,500 + 42 × 1,071.5 = 46,503 | 46,503 | yes |
| `terms["raw"]`, `scale` | 1,191.5; 871.5/991.5 | 200 + 991.5; (1,071.5 − 200)/(871.5 + 120) | yes (by definition) |
| `measured_confirmatory(37, 1,071.5, 455)` | 40,100.5 | 40,100.5 | yes |
| `measured_confirmatory(37, 1,191.5, 455)` | 44,540.5 | 44,540.5 | yes |
| `workloads_of` on a `refit-single = 10` record | bound 20, uplift 30 | 20, 30 | yes |
| Kernel projection in `carried_projection` | 600 + 32 × 1,071.5 = 34,888 | 34,888 | yes |
| Uplift test: refused at 1,313 + 30 | 600 + 32 × 1,343 = 43,576 > 43,200 | refused | yes |
| Real pilot `refit-bound` | 42.0551 / 3 = 14.018 s | 14.018 | yes |
| Real uplift per anchor; × 32 | amendment "about 32 s; about 1,000 s" | 31.541 s; 1,009.3 s | yes |
| Real measured projection, plain / static | ≤ 43,200 / ≤ 50,400 | 25,270.6 / 28,255.8 | within |
| Real kernel projection, plain / static | — | 23,606.0 / 26,454.5 (per-anchor 720.42 = 688.88 + 31.54) | consistent |
| Tests (13 files, CPU, worktree source) | pass | 273 passed, 64 subtests passed; the 26 Confirmatory/ReadOnce tests re-run at 4a2a005 after the branch moved, 26 passed | yes |

## What I did not check

- Nothing was run on GPU; no child, no replay, no recalibration. `replay_noop`'s cross-card agreement and `nvidia-smi` cost are exercised only by the fake cluster and by reading.
- The achieved-width byte identity was verified by code path and seed, not by re-running the pilot's `06-confirmatory-count.json`; the pilot's published D2/D3/D5 widths were reproduced by the statistics review, not again here.
- The projected half-width's numerical values at n = 32 for the pilot's D-values were not recomputed (they will appear only on the next pilot or in a narrative); the statistics review's F5 figures are the reference.
- `Supervisor.ledger` semantics behind `<job>.exit.json`, and the shared Gate 3 process record, were taken from the passing tests and the empty Gate 3 diff.
- Files outside `src`, `tests`, `docs/plans/2026-10-14-m0-amendment-5.md` and the two certificates were not reviewed; `docs/reviews/m0-capacity/amendment-5/` (the other exact-source review) was not read, to stay independent.
- The gate-B admission certificate 4 does not yet exist; its binding of `m0-v4.3` to `13989d9` and the source hash is outside this review.
