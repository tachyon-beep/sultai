# Independent exact-source reviews of the v4.1 fixed-evaluation package

Reviewed commit: `56a4512ca660255ffc7916aab9343d48763eca9b` on `codex/gate3-screen`,
the first commit of the package. Two fresh read-only subagents, 2026-10-13,
dispatched by the package author (`sultai-claude-nyx`); neither edited files,
ran the suite or touched a GPU. Reports preserved verbatim below; the
disposition table at the end records what the follow-up commit changed. The
follow-up commit rotates the amendment hash (and so the composed identity)
because it edits the amendment text; no certificate had been issued against
the reviewed hash.

## Disposition

| Finding | Disposition |
|---|---|
| Science 1 (80): old-failure half of the control matrix not demonstrated | Adopted. New test runs the preserved unpadded instrument on the shape-sensitive classifier at N = 5000: losses change exactly where batch shape changes; padded instrument, shared evaluator and the wide module are bitwise under the same permutation. |
| Science 2 (85): amendment and authorisation record disagree on the seed-range disposition | Adopted. Amendment §5 now states the range is the base §4 restart block, that John's authorisation is the disposition for the fit-code change, and that John's recorded confirmation of the allocation is a precondition of the confirmatory freeze; authorisation record aligned. |
| Science 3 (70): no v4.1 restart block named | Adopted. §5 names intact 4200–4263, bottleneck 5400 onward ≤ 5599. |
| Science 4 (60): disposition existence enforced only by call order | Adopted. `linked_pilot_two` checks the document exists on disk; test covers absence. |
| Science 5 (50): matrix gaps (wide module, real sizes, close candidates) | Partly adopted: wide module and N = 5000 covered by the new control. Close-candidate selection control not added: `choose` and the ranking are unchanged and already tested. |
| Science 6 (40): reference shares the expand+cat idiom | Noted; the unpadded-instrument control is the stronger cross-check. |
| Science 7 (30): tool paths in scratchpad | Adopted as `tools.txt` in the evidence directory (also ops 4). |
| Ops 1 (75): cap exhaustion spends attempt 3 | Adopted in the linked pilot 2 proposal for John's disposition. |
| Ops 2 (70): twin diagnostic cannot re-execute under v4.1 | Adopted. Stated in amendment §3. |
| Ops 3 (60): second invocation ends in a traceback, not a hold | Adopted. Existing custody receipt now raises an explicit hold before any side effect; test added. |
| Ops 4 (50): ephemeral tool paths | Adopted, `tools.txt`. |
| Ops 5 (30): `read` padded when no module | Not adopted; cosmetic, keeps one code path. |

---

## Scientific and experimental-method review (verbatim)

HEAD verified: `git rev-parse HEAD` = 56a4512ca660255ffc7916aab9343d48763eca9b, branch codex/gate3-screen, tree clean. Prereg SHA 7668d20a… and amendment SHA 7b9fde69… recomputed from the committed bytes and match contracts.py pins. Every source hash in quality.json and runtime.json matches `git show 56a4512:<file>`. Read-only review; no tests run, no GPU.

### Verdict: ACCEPT WITH CONDITIONS (conditions = findings 1 and 2)

#### Q1 Evaluator — matches amendment §3
`src/sultai_screen/fit.py:47-75`: chunks of `EVAL_CHUNK=128` in caller order (64-65); `_padded` duplicates the last real row across read/base/labels (66-67; 1-D labels work because `shape[1:]` is empty); adapter applied after padding (68); classify, unreduced CE, slice to real count (69-70); `torch.cat` gives exactly N in order (71); no RNG; `no_grad`; frozen hash before/after (60-61, 72). Padding never reaches a statistic: `mean_loss` (79), `identity_check` (83-88), `fit_sgd` selection (214-215), oracle paired difference and `paired_se` (296-297), `cheap_worker` raw loss (`screen.py:330`). All screen consumers go through `losses`/`mean_loss`/`identity_check`: `screen.py:203, 225-236, 251-282, 304-306, 330`. Untouched: `gradient_step` (fit.py:140-156, still unpadded 128 accumulation), `host.py:162` training CE, `ridge`, `former_fits`, cache. Only other evaluator in the package is `twin_diagnostic.evaluate` (twin_diagnostic.py:83-115), the preserved historical instrument that the design review required to keep; it is not a screen consumer.

#### Q2 Thresholds, §7 clause, seeds
Amendment §7 and §4 leave X, ρ, Hm, U, η sets, 2000/3000/5000, arms, V0/V1, ordering, stop rules unchanged; §3 touches only evaluation. v4 §7 (prereg lines 396-398) requires new version+SHA, explicit disposition, fresh range: composed identity `gate3-v4.1` (contracts.py:19-24) is the new SHA; ranges 4100–4163 / 5200–5399 are exactly v4 §4's own designated restart blocks (prereg lines 169-170), so disjoint from 3000–3099, 4000–4063, 5000–5199 and former seeds 1000–1031 by construction. Disposition: see finding 2.

#### Q3 Identity
Writers all emit the four fields: `screen.py:77` (source_identity → provenance → worker results 396/412 and training receipts), `pilot.py:376` (stop report), `410` (custody, via check_certificate identity), `590` (admission.json), `776` (pilot result). Readers reject old-only: `pilot.py:380-383` (certificate: missing or mismatched any of the four), `screen.py:119-124` (training receipt: key required and composed hash compared; dropping the bare prereg comparison is not a loosening since `protocol_sha256` composes it). No remaining boundary compares `preregistration_sha256` alone (grep over src/, scripts/). Prior-attempt records read by `linked_retry`/`linked_pilot_two`/`twin_diagnostic.verify_prior` are bound by hash, not by identity, as intended.

#### Q5 Linked pilot 2
`pilot.py:461-499`: `CUMULATIVE_CHARGE = 1473.267264504917` equals 2.370860553928651+1466.9817409639945+3.9146629869937897 in both association orders; `RETRY_CHARGE` equals attempt1+attempt2. All six pinned hashes match the live `.weft/` and `runs/` files; ledger-015 is the terminal retry ledger. Opening charge wired to `Supervisor` (597) under unchanged `PILOT_SECONDS=10800`. Further attempts: `publish` uses `open("x")` (contracts.py:84) so all three existing receipts and the future pilot-2 receipt block re-claims; `retry and second` refused (565). Disposition file `docs/certificates/2026-10-13-gate3-linked-pilot-2-disposition.md` does not exist.

#### Q6 Scope
The pilot-2 binding exceeds the literal "repair, amendment, CPU controls, review" wording but is inert (refuses without the disposition file; no GPU path) and matches the design review's "exclusive linked custody and complete terminal evidence" requirement. No guard loosened; `check_certificate({})` changing from KeyError to PilotHold is stricter.

### Findings

1. **SUGGESTION (condition before GPU), confidence 80** — `tests/test_fixed_evaluation.py:102-108`. The "old failure" half of the reviewers' matrix (design review line 120) is not demonstrated: the control compares against a whole 300-row batch, never the old 128-chunk evaluator. Resolve: add a test running `twin_diagnostic.evaluate(ShapeSensitiveHost, data, order, padded=False)` on identity vs global order and asserting the twin is not bitwise, then `padded=True`/`losses` passes.

2. **SUGGESTION (condition before confirmatory freeze), confidence 85** — amendment line 115-116 says "The disposition is John's authorisation of 2026-10-12", but the authorisation record lines 23-25 states the confirmatory seed-range allocation is a separate decision for John's confirmation at freeze. v4 §7 requires disposition and range together. Resolve: John records the range disposition (or the amendment is corrected, which rotates AMENDMENT_SHA/PROTOCOL_SHA and expires every certificate — do it before any certificate is issued).

3. **SUGGESTION, 70** — amendment §5 consumes v4's restart blocks but names no v4.1 restart block (4200–, 5400–). State it in the same edit as finding 2.

4. **SUGGESTION, 60** — `pilot.py:486` only checks the disposition path is named; existence/hash is enforced by `check_certificate` at 571 running before 574. Order-dependent, same as `linked_retry`; a one-line `is_file()` check in `linked_pilot_two` would make it self-sufficient.

5. **SUGGESTION, 50** — matrix gaps: no 4-tensor wide adapter through the padded path; no 2000/3000/5000 fixtures (tail 8 is the real failing case); no "close candidates" selection control. `Adapter.apply` is batch-agnostic and `choose` is unchanged, so low risk.

6. **SUGGESTION, 40** — `reference()` (test lines 40-54) is independent of `_padded`/`losses` but shares the expand+cat idiom and primitives; `tests/test_twin_diagnostic.py:52-55` supplies the stronger cross-check against the unpadded instrument on CPU. Adequate; do not overclaim.

7. **SUGGESTION, 30** — quality.json tool paths point into a session scratchpad venv; versions are recorded (mypy 1.19.1, ruff 0.15.4) so reproducible, but a reviewer cannot re-run the exact commands.

---

## PyTorch and operations review (verbatim)

**HEAD confirmed:** `git rev-parse HEAD` = `56a4512ca660255ffc7916aab9343d48763eca9b`, branch `codex/gate3-screen`, worktree identical to the commit for src/tests/scripts/pyproject. Read-only review; nothing executed on CUDA.

### Verdict: ACCEPT (no blockers; suggestions below)

#### Q1 Numerical correctness of the padded evaluator (fit.py:47-75)
Structural argument: every op in `Host.classify` (host.py:63-75) reduces within a row only — conv over c_in×k×k, BN-eval/ReLU/maxpool/`mean(dim=(2,3))` per-sample, Linear over K=128, log-softmax over 10 classes. `Adapter.apply`'s `einsum("ij,bjhw->bihw")` (fit.py:29,32) contracts over j only, so applying the module after padding is also per-row. With `use_deterministic_algorithms(True)`, `cudnn.deterministic`, `benchmark=False`, TF32 off (host.py:21-34), kernel choice and accumulation order are functions of shape/layout/alignment, and every chunk is now (128,16,16,16) contiguous NCHW. Chunk slices start at offsets k×2 MB and padded chunks are fresh allocations, so alignment matches between identity and twin.

Empirical CUDA confirmation is published, not just in the prompt: `docs/results/gate3-twin-diagnostic-2026-10-12-01a125af/worker-result.json` — `padded_pair` (padded identity vs padded global permutation) loss and logits `bitwise_equal: true`; `preserve_batch_size` bitwise; `original_global_permutation` 16 mismatches, max 1.9e-6. (`padded_identity` vs the unpadded baseline differs on the 8 tail rows by 1.2e-6, as expected: the tail changed shape.) Caveat: the diagnostic padded only base/labels with no adapter; the production extension to `read` and `adapter.apply` is structural.

CPU check (torch 2.9.1+cu128): `torch.cat((v, v[-1:].expand(...)))` gives a fresh contiguous NCHW float32 tensor, strides (4096,256,16,1), not sharing storage; full chunks return the original view (no copy); int64 labels pad correctly; N<128 yields one 128-row chunk; N=128k pads nothing. No in-place ops, no aliasing; under `no_grad` with frozen host and detached adapters, no autograd concern. Correction to the question: the site is 16×16×16 (front ends in `MaxPool2d(2)`, host.py:61), not 16×32×32.

#### Q2 Alignment
Both the identity and twin calls traverse `range(0, count, 128)` with the same `count` (fit.py:64-65); boundaries are identical by construction. `chunk[: stop - start]` keeps real rows; `torch.cat(result)` yields N losses in caller order; `argsort(order)` is the inverse permutation (fit.py:86-88). Tests cover this incl. a shape-sensitive host (test_fixed_evaluation.py:429-436).

#### Q3 Consumers and timing
All evaluation sites go through `losses`: `mean_loss`, `identity_check`, `fit_sgd` selection (fit.py:214), oracle paired difference (fit.py:296), `cheap_worker` (screen.py:330) and every `mean_loss` in screen.py. `gradient_step` (fit.py:140-156), `train_host`, `cached` (host.py:220-236) are byte-for-byte unchanged in the diff. Extra work: +120 rows at N=5000 (~0.3%); `check_timing` (pilot.py:347-356) only compares fit hashes/finiteness, no threshold reads `development_evaluation_seconds`. Minor: `_padded(data.read[...])` is computed when `adapter is None` (fit.py:66), harmless.

#### Q4 Memory
128-row chunk = 2 MB per fp32 tensor. 39 of 40 development chunks were already 128 rows, so peak is unchanged; the only delta is two fresh 2 MB cats for the tail chunk (plus 1 KB labels) instead of views. No change to the three-process co-tenancy budget.

#### Q5 Identity plumbing
`PROTOCOL_SHA = sha256("gate3-v4.1\n"+PREREG+"\n"+AMENDMENT+"\n")` (contracts.py:23) matches amendment §1; `AMENDMENT_SHA` 7b9fde69… matches `git show` of the file. `check_certificate` uses `key not in approved or approved[key] != value` (pilot.py:381-383) — membership test, no `.get`; `_training` uses `"protocol_sha256" not in original_identity` (screen.py:119). stop_report (pilot.py:376), admission (590), pilot-result (776), source_identity (screen.py:77) all carry the four fields. `twin_diagnostic.py` type-checks (in quality.json's 24 files, `strict_source_types` exit 0) but is now frozen history: `verify_prior` (twin_diagnostic.py:220-224) pins attempt-2 source bytes, so it raises before GPU work under v4.1.

#### Q6 Linked pilot 2
All six `PRIOR_RECEIPTS_2` literals (pilot.py:466-473) match sha256 of the published copies under docs/results/ at the commit and of the local gitignored `.weft/` and `runs/` files. `charge.json` carries opening 1469.3526015179232, total 1473.267264504917; float sums reproduce the literals exactly. `Supervisor(root, data, CUMULATIVE_CHARGE)` passes `0 <= x < 10800` (pilot.py:132). `retry and second` guard fires before any side effect (565-566); CLI flag wired (798-810); receipt path matches amendment §6. Reset path: none in code — `claim_attempt` → `publish` → `open("x")` (contracts.py:84) raises `FileExistsError` on a second invocation before `root.mkdir`; only manual deletion of `.weft/gate3-pilot-linked-pilot-2.json` could reset. No `linked_pilot_three` exists. Certificate must name `DISPOSITION_2`, which must exist, be hashed and committed (dirty tree → hold).

#### Q7 Operational
quality.json: 20 checks, 24 files, hashes for fit.py (fd460da0…), pilot.py (f094a2a5…), contracts.py, screen.py match `git show 56a4512:`. runtime.json `source_sha256` 70ff7805… matches my independent recomputation over the 45 tracked files. Reader must know: `/tmp/sultai-screen-mypy-wrapper` is a two-line shell script exec'ing mypy 1.19.1 from this session's scratchpad venv with `--python-executable /usr/bin/python3`; ruff 0.15.4 is from the same scratchpad venv. Both paths are baked into quality.json and will not exist in a fresh session (intent documented at docs/plans/2026-10-11-gate3-implementation.md:114).

### Findings
1. SUGGESTION (75): budget risk, pilot.py:148. Remaining 9,326 s; attempt 2 spent 1,467 s reaching the first materiality child, and every later stage (6 posthoc with 24-candidate oracles, 6-12 bottleneck oracles, possible 6 bottle2 trainings, 6 cheap) has never been ledgered. Cap exhaustion mid-run yields `execution-invalid` and spends attempt 3 with `further_attempt_authorized: False`. Resolve: state the stage estimate and this consequence in the linked-pilot-2 disposition.
2. SUGGESTION (70): twin_diagnostic.py:220-224 cannot re-execute under v4.1. Resolve: one line in the amendment or RESUME noting the diagnostic entry point is retired.
3. SUGGESTION (60): a second `--linked-pilot-2` invocation ends in an uncaught `FileExistsError` traceback (pilot.py:580 via contracts.py:84), not a `PilotHold`; durable and correct, but no report is written. Resolve: optional explicit `PilotHold` when the receipt exists.
4. SUGGESTION (50): quality evidence depends on ephemeral /tmp and scratchpad tool paths. Resolve: record the wrapper contents and venv tool versions in the evidence directory.
5. SUGGESTION (30): fit.py:66 pads `read` even when unused; cosmetic.
