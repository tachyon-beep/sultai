# Exact-source review: identity-under-amendment-3 (commit 3021ff3)

**Verdict:** ACCEPTED WITH FINDINGS

Reviewer: independent exact-source reviewer (read-only). Working tree, index and branch untouched (`git status --short`
before and after: only the pre-existing untracked `docs/results/m0-amendment-3-2026-10-14/`). Hashes were taken from
the committed blobs (`git show 3021ff3:<path> | sha256sum`) and cross-checked against the working tree and against
03a4a7a. Test runs were made on a `git archive 3021ff3` export in the session scratchpad with `/usr/bin/python3 -m
unittest` (the repo `.venv` has neither numpy nor pytest); no CIFAR file was opened.

## Claims and evidence

| Claim | Source file:line | Pinning test | Verdict |
|---|---|---|---|
| `PROTOCOL_VERSION = "m0-v4.1"` | `src/sultai_screen/m0/identity.py:25` | `tests/test_m0_identity.py:87` | Holds |
| `PROTOCOL_SHA = sha256("m0-v4.1\n"+P+"\n"+S+"\n"+A1+"\n"+A2+"\n"+A3+"\n")`, every element newline-terminated, order prereg, STEERING, A1, A2, A3 | `identity.py:26-28` | `test_m0_identity.py:84-90` (recomposed and literal `9f2bf07c…`) | Holds; recomputed independently with `printf` |
| Five document hashes pinned as literals | `identity.py:20-24` | `test_m0_identity.py:77-81` (re-hash of each file), `:88` (A3 literal) | Holds |
| `protocol_identity()` carries `amendment_3_sha256` | `identity.py:59` | `test_m0_identity.py:91-102` (exact key set) | Holds |
| `source_identity()` verifies all five documents and names the changed one in its PilotHold | `identity.py:64-72` (`"amendment 3"` at :69; message `f"committed {name} identity mismatch"` at :72) | `test_m0_identity.py:119-134` | Holds |
| Dropping amendment 3 from the loop fails `test_source_identity_names_the_changed_document` | `identity.py:69` | `test_m0_identity.py:128-134` | Holds, mutation-tested (see note A) |
| Old admission certificate (978850a, `m0-v4`) is rejected by code, not only by text | `src/sultai_screen/m0/pilot.py:197-199` iterates `protocol_identity().items()`; `docs/certificates/2026-10-14-m0-pilot-admission.json:2-3` has `m0-v4`/`3a76c6db…` and no `amendment_3_sha256` | `tests/test_m0_pilot.py:1643, 1686-1692` (check_certificate tests) | Holds |
| Receipts from the v4 identity are refused on reuse | `src/sultai_screen/m0/worker.py:244-246` (`protocol_sha256` compared) | worker tests | Holds |
| Worker/pilot receipts carry `m0-v4.1` | via `protocol_identity()` | `tests/test_m0_worker.py:353, 844`; `tests/test_m0_pilot.py:416, 428, 856, 1535` | Holds; no stale `m0-v4` pin remains in src/scripts/tests |
| Four previously pinned documents byte-unchanged | — | `git diff 03a4a7a..3021ff3 --stat` lists none of them; blob hashes at 03a4a7a equal those at 3021ff3 | Holds |
| v4 identity stays on record | `docs/certificates/2026-10-13-m0-freeze-2.md:26` (`3a76c6db…`), `docs/certificates/2026-10-14-m0-pilot-admission.json:2-3`, both untouched by 3021ff3; freeze-3 lines 3-8 cite them | — | Holds |
| Freeze certificate 3 voids the 978850a admission certificate for execution | `docs/certificates/2026-10-14-m0-freeze-3.md:3-8` ("which bound that identity and is void for execution from this freeze on"); `:61-63` (new gate-B admission certificate required) | `pilot.py:197-199` enforces it | Holds |
| Amendment 3 line citations | A3 cites §5 l.292, §6 l.366-368, §7 l.429, §10 l.558-561 of the preregistration | checked against the file | All four citations are accurate; refit booking arithmetic (3 × 3.2 s + evals ≈ 10 s/anchor, ≈ 320 s at n = 32) matches §8 l.455 |

**Note A (why the mutant fails).** With `(AMENDMENT_3_PATH, AMENDMENT_3_SHA, "amendment 3")` deleted from
`identity.py:69`, the baseline export passes 8/8 and the mutant fails exactly
`test_source_identity_names_the_changed_document`:
`AssertionError: "amendment 3" does not match "run requires committed, unchanged source"`. The failure does not come
from a hash check: the test's own `DOCUMENTS` tuple (`test_m0_identity.py:48-54`) still lists amendment 3, so the
fixture commits it; appending a byte to it passes the four remaining hash checks, then `git status --porcelain` is
dirty (`identity.py:74-76`) and the generic dirty-tree hold is raised, which does not match the regex `"amendment 3"`.
The discrimination therefore rides on the per-document hold message, which is what the test means to pin; it is a
real failure. (Had the test's `DOCUMENTS` tuple also dropped amendment 3, the mutant would pass; the
`test_five_documents_and_composed_identity_are_pinned` key/hash assertions would still not catch a loop omission.
This is acceptable because the two lists are independently edited and both are under review.)

## Hashes recomputed

| Document | Pinned | Recomputed (blob at 3021ff3; worktree identical) | Match |
|---|---|---|---|
| Preregistration `docs/plans/2026-10-13-capacity-frontier-screen.md` | `c66599459adc898f96322ab11dfa3c6737cd3559e06c9432d79e377e00c03fd5` | `c66599459adc898f96322ab11dfa3c6737cd3559e06c9432d79e377e00c03fd5` (same at 03a4a7a) | Yes |
| STEERING `docs/STEERING.md` | `08a9e1aaf79d3048a5d6f6f4d6138a9c7049f32b61c0250e1505773a27fa3ffd` | `08a9e1aaf79d3048a5d6f6f4d6138a9c7049f32b61c0250e1505773a27fa3ffd` (same at 03a4a7a) | Yes |
| Amendment 1 `docs/plans/2026-10-13-gate-ladder-amendment.md` | `af12d474a97aad83bc63405034fec3716170f7652f34364588750732c97b6a89` | `af12d474a97aad83bc63405034fec3716170f7652f34364588750732c97b6a89` (same at 03a4a7a) | Yes |
| Amendment 2 `docs/plans/2026-10-13-gate-ladder-amendment-2.md` | `b4740b147a4a1ced52a6f0721215679c1567078260d5c11a92c35fa2065c2d77` | `b4740b147a4a1ced52a6f0721215679c1567078260d5c11a92c35fa2065c2d77` (same at 03a4a7a) | Yes |
| Amendment 3 `docs/plans/2026-10-14-m0-amendment-3.md` | `a8ff6ff225c5d6111dce9ed1682e5dc7fb2f0d3301292ab25869f7dc35b21a8f` | `a8ff6ff225c5d6111dce9ed1682e5dc7fb2f0d3301292ab25869f7dc35b21a8f` (absent at 03a4a7a) | Yes |
| Composed v4.1 `printf 'm0-v4.1\n%s\n%s\n%s\n%s\n%s\n'` | `9f2bf07c592300aa423fea642ba89e0a327d5f3c6711ccc42e6d469d61f0d28c` | `9f2bf07c592300aa423fea642ba89e0a327d5f3c6711ccc42e6d469d61f0d28c` | Yes |
| Composed v4 (on record) `printf 'm0-v4\n%s\n%s\n%s\n%s\n'` | `3a76c6dbe31ca3f80ec301ed0523b1f382148f8114ee02a164ae4ae9557f6e5d` | `3a76c6dbe31ca3f80ec301ed0523b1f382148f8114ee02a164ae4ae9557f6e5d` | Yes |

Literal sweep (every 64-hex string at 3021ff3): `identity.py:20-24` (five document hashes), `test_m0_identity.py:88,90`
(A3, composed v4.1), freeze certificate 3 `:25-29,34` (five documents, composed v4.1), freeze certificate 2 `:19-22,26`
(four documents, composed v4), amendment 3 `:7` (preregistration only). Every literal equals the recomputation.
Amendment 3 correctly does not contain its own hash; `a8ff6ff2…` appears only in `identity.py`, the identity test and
freeze certificate 3.

## Findings

### Important

**I1. A3.3 is presented as an owner decision without the owner's word.** `docs/plans/2026-10-14-m0-amendment-3.md:3-4`
says the amendment is "taken by John on 2026-10-14, in his words below", and `:63` heads A3.3 "(gate-B decision 2,
recorded)"; but the owner-decision table (`:72-76`) quotes John only for "A3.1 and A3.2 taken (option A …)". The
ledger confirms decision 2 was never taken: the draft headed it "include only if John takes it"
(`.superpowers/sdd/2026-10-13-m0-implementation/amendment-3-draft.md:23`), the memo offered it "if wanted"
(`bounds-gap-memo.md:66`), `gate-b-points.md:5` still asks "Accept, or amend (would be amendment 3)?", and
`progress.md` R35 records that "The gate-B decision 2 (S-static N0) is recorded in A3.3 as the implemented reading;
decisions 1, 3, 4, 5, 6 remain John's" (a controller choice, not his). Why it matters: a frozen, hashed preregistration
amendment now asserts a resolution of an open owner question under John's name; the status line covers A3.3 by its
wording. It changes no value and S-static is descriptive outside the 14-hour flag, so it is not blocking. Fix: get
John's one-line word on gate-B decision 2 at the gate-B request and record the quote in freeze certificate 3 /
the admission-2 certificate as the decision of record for A3.3; or, if he does not take it, re-cut amendment 3 without
A3.3 (new hash, new composed identity, identity.py and test literals updated) before the admission certificate is
issued.

### Minor

**M1. Internal tension in what amendment 3 says it changes.** `amendment-3.md:8-9` says the preregistration is amended
"in two places: the §10 confirmatory arm set and the reading of §6's 'across anchors' sentence"; `:83-84` says
"Changed: the §10 confirmatory arm set only (A3.1)". A3.2 widens what is published at confirmatory n (bounds and
sign test beside every §6 quantity) without touching any frozen value or verdict input, and the identity paragraph of
§11 (`capacity-frontier-screen.md:587-598`) is superseded by `:84-88`; neither of those is named in the "Changed" line.
Reconcilable (the "Frozen values" list is about values, and A3.2 is a reading, not a value), but a reader of the frozen
text gets two different scopes. Fix: state the reading once in freeze certificate 3 or the admission-2 certificate
("Changed values: §10 arm set only; changed reading: §6 across-anchors reporting scope; §11 identity superseded"); no
re-cut needed unless I1 forces one, in which case fix the sentence in the re-cut.

**M2. Stale docstring.** `tests/test_m0_identity.py:1`: "the four-document source identity" should read five. Fix:
one-word edit in the certificate commit (test file is not a pinned document).

**M3. Documentation lag: `docs/RESUME.md:9-12`** presents freeze certificate 2 and "identity `m0-v4`" as the current
freeze and identity, and (`:20-21`) gate B as John's next decision for 978850a. Fix: update RESUME to v4.1
(`9f2bf07c…`, freeze certificate 3, amendment 3, 978850a admission void, admission-2 pending).

**M4. Documentation lag: `docs/DECISIONS.md`** has D28 (v4 freeze, identity `m0-v4` = `3a76c6db…`, line 54, correct
as history) and D29 (978850a certified, line 55) but no row for amendment 3 / identity `m0-v4.1`. Fix: add a D30 row
(John's quote, A3.1-A3.3 with I1's disposition, identity `9f2bf07c…`, 978850a admission void for execution).

Not findings (recorded for completeness): preregistration §11 lines 587-588 still say `m0-v4` (frozen bytes, by design
superseded by amendment 3); `docs/plans/2026-10-13-m0-implementation.md:75,86,88` pin `m0-v4`/`3a76c6db…` (the accepted
gate-A plan, a historical record); `docs/certificates/2026-10-14-m0-pilot-admission.json:2-3` keeps `m0-v4` (legitimately
on record, voided by freeze certificate 3 and rejected by `pilot.py:197-199`). STEERING mentions only that its hash is
part of the M0 identity, not which version; nothing to change there (and it is pinned). `contracts.py`/Gate 3 tests
pin `gate3-v4.2`, untouched and out of scope. `tests/test_m0_pilot.py:417` asserts `amendment_2_sha256` but not
`amendment_3_sha256` on the pilot result; harmless (the result spreads `protocol_identity()`, whose key set is pinned).

## Dispositions (controller, commit d2e30b3)

| Finding | Disposition |
|---|---|
| I1 (Important) | Adopted. John's words took option A only. Amendment 3 re-cut without the S-static clause; it now states that gate-B decision 2 is not taken and remains his. New amendment hash 6dcfd9f2…, composed identity dfd13cb4…, freeze certificate 3 repinned. |
| M1 | Adopted: "one frozen value changed (A3.1), one reading fixed (A3.2)". |
| M2 | Adopted: the identity test docstring says five documents. |
| M3 | Adopted in the certificate commit: RESUME.md names m0-v4.1 as current. |
| M4 | Adopted in the certificate commit: DECISIONS.md row D30. |

Hashes at d2e30b3 are re-verified by the scoped re-review (re-review-d2e30b3.md).
