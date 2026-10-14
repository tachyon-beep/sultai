# Scoped re-review: amendment 3 round 1 (commit d2e30b3)

**Verdict:** CLEAN

## Items checked

| Item | Evidence | Result |
|---|---|---|
| I1 | A3.3 removed. New section "What this amendment does not decide": "Gate-B decision 2 (whether S-static should carry a no-op rule of its own, on which the preregistration is silent) is not taken here; it remains John's, and the implemented reading (no no-op rule on the static host; D5 against the anchor's own no-op route) stands only until he decides. Any change to it is amendment 4." Owner table row now says "nothing else in this amendment is a decision". | fixed |
| M1 | Opening now reads "in one frozen value, the §10 confirmatory arm set (A3.1), and fixes the reading of §6's 'across anchors' sentence, which changes no frozen value (A3.2)"; Frozen values adds "A3.2 fixes a reading and changes no value". | fixed |
| M2 | tests/test_m0_identity.py line 1 says "five-document source identity". | fixed |
| Earlier four documents | `git diff --stat 03a4a7a..d2e30b3` over the four paths is empty. | unchanged |
| F1, F3 | ReadOnceTests (3 tests) pass at HEAD; probes below. | fixed |
| F2 | Amendment A3.1: "nominal: 3.2 s per 800 steps plus one development evaluation, so about 3.3 s per refit, three refits or about 10 s per anchor, about 320 s at n = 32 against the 43,200 s ceiling; the pilot's measured kernel replaces these nominals". | distinguishes per refit from per anchor. Note: the fixture books 10 s per refit (960 s at n = 32); the amendment's nominal (3.3 s/refit) is a different figure from the review's fixture, and the sentence now states it as nominal and defers to the measured kernel. Not a defect in this round. |

## Hashes recomputed

| Quantity | Recomputed | identity.py | test literal | freeze cert 3 | Match |
|---|---|---|---|---|---|
| Preregistration | c66599459adc898f96322ab11dfa3c6737cd3559e06c9432d79e377e00c03fd5 | same | (via import) | same | yes |
| STEERING | 08a9e1aaf79d3048a5d6f6f4d6138a9c7049f32b61c0250e1505773a27fa3ffd | same | - | same | yes |
| Amendment 1 | af12d474a97aad83bc63405034fec3716170f7652f34364588750732c97b6a89 | same | - | same | yes |
| Amendment 2 | b4740b147a4a1ced52a6f0721215679c1567078260d5c11a92c35fa2065c2d77 | same | - | same | yes |
| Amendment 3 | 6dcfd9f2a781dc56d39c6e4fe548fc4728c2c743a8e5a978d0e4b9741459e862 | same | same (AMENDMENT_3_SHA literal) | same | yes |
| Composition m0-v4.1 | dfd13cb45100d5da6bca0ac3348ce9d617d614132e9ceba07c74d57c45f33ca7 | computed in code from the five | same (PROTOCOL_SHA literal) | same | yes |

The committed blob of amendment 3 at d2e30b3 hashes to the same value as the working-tree file.

## Probes

All run against ReadOnceTests only; the file was restored byte-exactly after each (cmp identical to the saved copy, `git diff --stat` empty).

| Probe | Test that fails | Result |
|---|---|---|
| M4 (beside raises InstrumentHold on undefined) | test_an_undefined_no_verdict_quantity_is_published_naming_the_anchor_and_the_verdict_is_still_read (error) | caught |
| M5 (column filters Nones, no hold) | test_an_undefined_verdict_quantity_holds_the_reading, subtests D1, D1prime, D2, D3, delta_acc | caught |
| M12 (survival reason constant) | test_an_undefined_no_verdict_quantity_is_published_naming_the_anchor_and_the_verdict_is_still_read | caught |
| M13 (sign_test_lower = bootstrap lower) | test_sign_test_bounds_are_the_order_statistics_of_each_column | caught |
| M2 (survival sign_test_lower = None) | test_sign_test_bounds_are_the_order_statistics_of_each_column | caught |

## Remaining findings

None.

Note: `git status --porcelain` prints only `?? docs/results/m0-amendment-3-2026-10-14/`, an untracked directory that was already present in the session's initial git status; no tracked file or index change, `git diff --stat` empty.
