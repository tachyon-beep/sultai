# Exact-source review: seen-fit-partition (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Reviewed read-only: src/sultai_screen/m0/data.py (44 lines), tests/test_m0_fit.py::FitTests, the diff, sultai_screen/data.py (partitions, generator), host.py (cached.part). Ran the three directly relevant tests (seen_fit x2, features_from_images): 3 passed (PYTHONPATH=src).

## Claims and evidence
| claim | source file:line | pinning test | verdict |
|---|---|---|---|
| 2,000 seen examples drawn by fixed seed from the 40,000 training partition (§4) | data.py:22-27 (randperm(40000) of data.train, first 2000) | test_seen_fit_properties_and_sha (len 2000, unique, subset of train, int64) | satisfied; fails if count, pool or draw changes |
| Disjoint from selection (3,000), development (5,000) and conditioning (2,000) | follows from data.train being fit[:40000] (sultai_screen/data.py partitions); seen_fit only indexes data.train | same test, three empty-intersection asserts on real partitions | satisfied; fails if the draw leaves data.train |
| Label `m0-seen-fit-v1`; `SEEN_FIT_SHA = a4130e3c…` | data.py:14-19, 26 | same test: SHA-256 of "<i8" bytes of the real result equals the literal | satisfied; any change of seed, label, order or slice breaks the hash |
| 40,000 / 2,000 / 3,000 / 5,000 sub-partition carried over unchanged (§2) | data.py does not touch partitions(); TRAIN_COUNT=40_000 guard at :23-24 | test_seen_fit_rejects_wrong_train_size (39,999) | satisfied; guard is the only addition |
| Conditioning 2,000 only for P1 and M-sq-fresh | not in this item (seen_fit never reads data.conditioning) | n/a | out of scope; seen_fit is consistent |
| Test batch never read | data.py reads only data.images/labels at caller-given indices; no file access, no test-batch symbol | none in this item (seal pinned elsewhere) | no violation; not pinned here |
| Front/bottleneck features for arbitrary ids "mirroring cached.part" (plan Task 5) | data.py:30-44 vs host.py:339-347: same 128 chunking, normalized, front, bottleneck, no_grad, detach, labels indexing, plus frozen-hash check before and after | test_features_from_images_matches_cached_part | source matches; test weak (see F1) |

## Frozen values checked
| value | prereg | source | match |
|---|---|---|---|
| seed | 20261013 (plan Task 5) | data.py:14 | yes |
| label | m0-seen-fit-v1 (§4) | data.py:15 | yes |
| count | 2,000 | data.py:16 | yes |
| train size | 40,000 | data.py:17 | yes |
| SEEN_FIT_SHA | a4130e3c… | data.py:19 a4130e3c0eb483fbf0af0d975ddc341bed1781819a249f974baca07d73d21c9c | yes, prefix matches and test recomputes full hash from production partitions |
| chunk size | 128 (cached.part) | data.py:38 | yes |

No frozen value needs changing. No amendment territory.

## Findings
**F1 Minor, tests/test_m0_fit.py test_features_from_images_matches_cached_part.** Despite its name it never compares against `cached()`/`part` or an independent unchunked forward; it asserts only shape (132 rows), labels, and that duplicate ids give equal rows. Changing normalization, dropping the bottleneck, or using different chunking would not fail it (row 0 == row 4 holds regardless). Fix: compare read/base to an independent `host.front(normalized(images[ids]))`/`host.bottleneck` on the fixture, or to `cached(host, data_with_ids_as_selection).selection`.
**F2 Minor, data.py:30-44.** No assertion that ids are within the train/selection/development pools (a test-batch id cannot be excluded by this function, since data.images holds only the 50,000 train-batch images, so the seal is held by dataset construction, not here). Not a deviation; noting that the seal claim is not pinned by this item. Fix: none needed, or add a bounds check if the controller wants defence in depth.
**F3 Minor, tests.** test_seen_fit_rejects_wrong_train_size covers only a short train; the guard is `!=`, so an over-long one is also rejected but untested. Negligible.

Counts: BLOCKING 0, Important 0, Minor 3.

## Dispositions (controller, 2026-10-14)

- F1–F3 (Minor): deferred, recorded. The data seal is held by dataset construction (only the 50,000 train-batch images are ever loaded); the `features_from_images` comparison and the over-long-train guard are noted for a later test pass.

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
