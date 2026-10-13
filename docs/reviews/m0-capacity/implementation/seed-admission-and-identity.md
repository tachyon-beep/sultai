# Exact-source review: seed-admission-and-identity (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Run: `PYTHONPATH=src python -B -m pytest -p no:cacheprovider tests/test_m0_identity.py` -> 8 passed. The four document SHA-256 values computed from the working tree equal the literals in identity.py. No frozen value needs changing (no BLOCKING item).

## Claims and evidence
| claim | source file:line | pinning test | verdict |
|---|---|---|---|
| §2: seed admission beyond 3000-3005 (main() refuses other seeds today) | seeds.py:23-33 BLOCKS, 39-44 admit(); used at worker.py:1384 | SeedTests.test_blocks_match_appendix_a (exact dict equality; changing any range fails), test_admit | met |
| §11: blocks per lineage (0.5: 6000-6005; 1.5: 6012-6017; 0.75: 6006-6011; cooled 6018-6023; 0.25: 6024-6029; confirmatory 7500-7599; restart 7700-7799; anchors own seeds 3000-3005) | seeds.py:23-33 | test_blocks_match_appendix_a | met |
| §11: reserved M2 block 6100-6199, never used by M0 | seeds.py:34, 40-41 | test_blocks_match_appendix_a pins the range; test_blocks_are_disjoint... pins non-overlap; test_admit(6100, "anchor") | met; see Finding 2 (branch not independently pinned) |
| Appendix A: collides with nothing (Gate 3 4000-4363, 5000-5799) | seeds.py | test_blocks_are_disjoint_from_each_other_and_from_reserved_and_gate3 | met; dev block 3000-3099, 1000-1031, 7001-7004 not in the disjointness list (Finding 3) |
| §2/§11: width scale per lineage (s = 0.25/0.5/0.75/1.5; cooled and anchor/confirmatory/restart reference 1.0) | seeds.py:36-45, 47-49 | SeedTests.test_width_scale | met |
| §11: PROTOCOL_VERSION "m0-v4"; PROTOCOL_SHA = SHA256("m0-v4\n"+prereg+"\n"+steering+"\n"+amend1+"\n"+amend2+"\n") | identity.py:23-30 | IdentityTests.test_four_documents_and_composed_identity_are_pinned (recomputes composition, pins the literal 3a76c6db..., and the file hashes) | met |
| §11: protocol_identity() carries protocol_version, protocol_sha256, preregistration_sha256, steering_sha256, amendment_sha256, amendment_2_sha256 | identity.py:50-59 | same test (exact key-set equality) | met |
| §11: source_identity() pins the four documents, not the Gate 3 chain | identity.py:62-70, 84 | IdentityTests.test_source_identity_names_the_changed_document (mutates each document in a git fixture; asserts the named PilotHold; dirty tree hold) | met |
| §11: margins X, X2, M_S, RHO, HARM, FOLD_TOLERANCE | identity.py:32-37 | IdentityTests.test_margins_and_ceilings_are_the_frozen_values (literals) | met |
| §3-§7: ceilings 14,400 / 43,200; PILOT_SECONDS untouched | identity.py:40-41 | same test; asserts PILOT_SECONDS == 3*3600 | met |
| §3-§7: CERTIFIED_FRACTION 0.75, S_STATIC_SD_LIMIT 0.0275 | identity.py:38-39 | no test in this item pins the literal (Finding 1) | source matches; unpinned |
| §4: grids (1,.3,.1) and (1,.3,.1,.03,.01); checkpoints (200,800,3200),(400,1600,6400); R-all every 40 to 800 | identity.py:42-47 | no literal-pinning test in this item (Finding 1); test_m0_worker.py:167-170 monkeypatches them, test_m0_pilot.py compares to the imported constant (tautological) | source matches; unpinned |
| §2: width-tuple training check; optimiser state and resume path; epoch count | not in these files (worker.py:1381-1387 touches scale and epochs) | n/a | out of scope for this item's paths |
| provenance.py: envelope and environment identity (no spec text) | provenance.py:15-56 | EnvelopeTests (key set, import isolation) | no preregistration claim; see Finding 4 |

## Frozen values checked
| value | prereg | source | match |
|---|---|---|---|
| X | 0.02 (§6, J1) | identity.py:32 | yes |
| X2 | 0.01 | identity.py:33 | yes (task brief; not re-derived from prereg text) |
| M_S | 0.02 (J3) | identity.py:34 | yes |
| RHO | 0.8 (line 82) | identity.py:35 | yes |
| HARM | 0.01 (line 80) | identity.py:36 | yes |
| FOLD_TOLERANCE | X/4 = 0.005 (line 381) | identity.py:37 | yes |
| CERTIFIED_FRACTION | 75 percent (line 394) | identity.py:38 | yes |
| S_STATIC_SD_LIMIT | 0.0275 (line 414) | identity.py:39 | yes |
| pilot ceiling | 4 GPU-h = 14,400 s (line 498) | identity.py:40 | yes |
| confirmatory ceiling | 12 GPU-h = 43,200 s (line 564) | identity.py:41 | yes |
| module grid | {1, 0.3, 0.1} (line 227) | identity.py:42 | yes |
| re-tune grid | {1, 0.3, 0.1, 0.03, 0.01} (line 228) | identity.py:43 | yes |
| checkpoints | {200, 800, 3200} / {400, 1600, 6400} (lines 232, 248) | identity.py:44-45 | yes |
| R-all checkpoints | every 40 to 800 (line 315) | identity.py:47 | yes |
| seed blocks | §11 table, Appendix A | seeds.py:23-34 | yes (all eight blocks and RESERVED_M2) |
| PROTOCOL_VERSION | m0-v4 | identity.py:27 | yes |
| PROTOCOL_SHA | 3a76c6dbe31ca3f80ec301ed0523b1f382148f8114ee02a164ae4ae9557f6e5d | computed at import; test pins literal | yes |
| prereg / steering / amend1 / amend2 SHA | files on disk at e1df6e1 | identity.py:23-26 | yes (sha256sum equal) |

## Findings
1. Important. identity.py:38-39, 42-47. CERTIFIED_FRACTION, S_STATIC_SD_LIMIT, the two rate grids, both checkpoint axes and RALL_CHECKPOINTS have no test that compares them to a literal (the item's tests cover only margins, ceilings, hashes). Elsewhere tests either import the same constant (test_m0_pilot.py:432, 995) or monkeypatch it (test_m0_worker.py:167-170); test_m0_stats.py:337 uses 0.0275 as an input literal, not the constant. A production edit such as CERTIFIED_FRACTION = 0.7 or dropping 0.01 from RETUNE_RATES would pass. Fix: extend test_margins_and_ceilings_are_the_frozen_values with literal asserts for these.
2. Minor. seeds.py:40-41, test_admit. The RESERVED_M2 branch is not independently pinned: admit(6100, "anchor") also fails the block check, so deleting the branch keeps all tests green (the hold message is the only difference; no assertion on text). Fix: assert the "reserved for M2" message, or add RESERVED_M2 as an explicit membership check with a lineage that could otherwise reach it (none exists, so the branch is dead code today; harmless defence in depth).
3. Minor. test_blocks_are_disjoint_from_each_other_and_from_reserved_and_gate3. The comparison list omits the other Appendix A owners (1000-1031, 2000-2007, 3006-3099 dev, 7001-7004, 8000s, etc.). Current blocks do not collide with them; the test would not catch a future drift. Fix: add those ranges to `others`, treating 3000-3005 as the permitted overlap with 3000-3099.
4. Minor. EnvelopeTests asserts key-set equality with a hard-coded list, not equality with the Gate 3 envelope function, so "reproduces the Gate 3 receipt keys" is only as good as that list. wall_seconds/peak values and key order are not compared. No spec text covers provenance.py; behaviour is a copy of the Gate 3 shape, nothing extra found. Also source_identity's untracked/empty production-coverage hold (identity.py:78-81) has no test.
5. Minor (scope note, not a defect). The §2 items "training check that reads the width tuple", optimiser-state saving, resume path and 20-epoch decay are not in this item's three files; seeds/identity only supply the lineage scale and block. Reviewed elsewhere.

No extra arms, extra randomness, RNG-order change or Gate 3 receipt byte change found in these three files. Nothing requires amending a frozen value.

## Dispositions (controller, 2026-10-14)

- Important 1: FIXED in the post-review commit (item C1: literal asserts for CERTIFIED_FRACTION, S_STATIC_SD_LIMIT, MODULE_RATES, RETUNE_RATES, CHECKPOINTS, CHECKPOINTS_B, RALL_CHECKPOINTS).
- Minor 2–5: deferred, recorded (RESERVED_M2 branch is defence in depth; Appendix A owner ranges; envelope key list; scope note).

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
