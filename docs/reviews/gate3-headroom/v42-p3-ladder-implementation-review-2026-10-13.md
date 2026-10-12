# Independent exact-source reviews of the v4.2 P3-ladder package

Reviewed commit: `c25c2ca260381befef2b300e4804b3c56ea91d78` on `codex/gate3-screen`.
Two fresh read-only subagents, 2026-10-13, dispatched by the package author
(`sultai-claude-nyx`); neither edited files, ran the suite or touched a GPU.
Reports preserved verbatim below. Both converged on one gap: amendment §3
promised that the three attempt-3 rungs could never be chosen, but the
magnitude decision still walked the whole ladder. The follow-up commit
adopts the stronger form the PyTorch reviewer proposed, and a targeted
re-review of that diff is preserved separately. The amendment text changes,
so its hash and the composed identity rotate; no certificate had been issued
against the reviewed hash.

## Disposition

| Finding | Disposition |
|---|---|
| Science 1 BLOCKER (70) / Ops 1 (80): §3 clause not enforced | Adopted, strong form. `contracts.P3_RETIRED` = {0.25, 0.5, 1.0} with attempt 3's published gains pinned per anchor as exact floats; `magnitude_decision` requires the fresh hosts to replay them exactly (any difference is a hold) and chooses only among `P3_ELIGIBLE` = {2.0, 4.0}. Amendment §3 rewritten to say exactly this. Tests cover: exact replay required, a retired rung that now clears is a hold, a 1e-9 drift is a hold, unknown or duplicate anchors refused, first eligible rung chosen, exhaustion hold. |
| Science 2, 3, 4, 5 (85–90): scope, numerics at κ = 2 and 4, seed ranges, reset paths | Confirmed; no change. |
| Science 6 (70): no test feeds the v4.1 identity to `check_certificate` | Adopted: a certificate carrying the full v4.1 identity is rejected. |
| Science 7 (60): `protocol_identity` docstring cites v4.1 §1 | Adopted (docstring). |
| Ops 2 (90): P3 numerics at κ = 2.0 and 4.0 safe; κ = 4 inverse margin ~1e-6 against 1e-4 | Noted; no change. |
| Ops 3 (95): cost negligible | Noted. |
| Ops 4, 5, 6: identity plumbing, linked pilot 3 binding, evidence hashes verified | Confirmed; no change. Pre-existing note that missing certificate keys raise `KeyError` before custody is recorded for housekeeping. |
| Ops launcher notes | Followed at launch: clean tree with certificate committed, nine receipts byte-preserved, no pilot-3 custody, fresh handover check, cap consequence understood. |

---

## Scientific and experimental-method review (verbatim)

**HEAD confirmed**: `c25c2ca260381befef2b300e4804b3c56ea91d78` on `codex/gate3-screen`, tree clean. Read-only; no tests run, no GPU.

**Verdict: ACCEPT WITH CONDITIONS** (one condition, cheap to resolve; everything else is clean).

**Verified facts**
- Base/v4.1/v4.2 SHA-256 recomputed and equal the pins in `src/sultai_screen/contracts.py:16,22,24`; the v4.1 hash is quoted verbatim in the v4.2 text (amendment line 93) and verified at `src/sultai_screen/screen.py:65-66`; composed `PROTOCOL_SHA` recomputes to `9126b6fb…2aba`. `test_protocol_identity.py:95` rejects old-only certificates/receipts.
- `impair.py`, `fit.py`, `host.py` untouched (hashes `22242da3…`, `fd460da0…`, `fd8a263e…` equal the evidence directory's `source_files`); `contracts.py`/`pilot.py`/`screen.py` hashes equal `quality.json` and `runtime.json` (166 tests OK), so the evidence was produced from the committed bytes.
- Nine prior receipts: `.weft/*` and both `runs/gate3-pilot-2026-10-13-linked-2/{pilot-result,ledger-020}.json` (the files read at launch, not only the `docs/results` copies) hash to the pins at `pilot.py:233-238`; `pilot-result.json` carries `charged=2934.3435500559863, verdict=instrument-failed, software_valid=True, pilot_clean=False`; `ledger-020.json` carries opening 1473.267…, ceiling 10800, 36 entries summing to 1461.076….

**Findings**

1. **BLOCKER (confidence 70) — amendment §3 states a rule the instrument does not implement.** `docs/plans/…v4.2-amendment.md:120-124` says both "the rule is unchanged" and "the three rungs that failed in attempt 3 … cannot be chosen by a later measurement that happens to pass them." `magnitude_decision` (`pilot.py:290-301`) iterates `P3_LADDER` from 0.25 and will select 1.0 if it clears on all six anchors. No test asserts the second sentence (`tests/test_v42_ladder.py:495-504` uses 1.0=0.04 so it never exercises it; `test_screen.py:374` tests P1 exhaustion despite its name). Practical exposure is low: fresh training of the same seeds 3000–3005 under the determinism contract should reproduce attempt 3's 0.0399 on 3002 bitwise. But this is the v4.1 §2 pattern in miniature (protocol promises more than the implementation enforces), and a selection of κ=1.0 would violate the hashed text and void the attempt. Resolve one way before launch: (a) delete the clause so "rule unchanged" is literally true, or (b) enforce it (P3 eligibility starts at 2.0, with 0.25/0.5/1.0 evaluated for the curve only; a failed rung that now passes is a replay-mismatch hold) plus a test with weak 1.0 ≥ 0.05. Either edits amendment bytes → new `AMENDMENT_SHA` → re-review of that one-line change.

2. SUGGESTION (85) — Scope otherwise exactly "append two rungs." Only `P3_LADDER` (`contracts.py:25`), `MATERIALITY_LADDERS` (`screen.py:54`) and the P3 tuple in `magnitude_decision` change; P1/P2 ladders, 0.05 bar, 1e-4 inverse tolerance, arms, X/ρ/Hm/U untouched. Downstream κ consumers (`screen.py:259` P3 and P3′ in `posthoc_worker`, `pilot.py:718-731`) take one scalar; `costs.py` has no rung dependence. Added cost: two extra A1 evaluations per anchor, negligible against the 3,000–4,000 s projection.

3. SUGGESTION (90) — No degeneracy at κ ∈ {2, 4}. `undo` is `(z+c)−c` through an identity einsum (`impair.py:24-27`); float32 error ~1e-7 against a 1e-4 bar (attempt 3 passed at κ=1 on GPU). A3's ridge has an intercept column (`fit.py:103`) so the bias inverse is exactly representable; Gram conditioning is unaffected by a shift. A2 tanh saturation and A7 input-distribution drift on P3′ worsen with κ but are predeclared (v4 §6 D21, V2 limit) and gate nothing. Note that `materiality_worker` (`screen.py:244-245`) raises `PilotHold` on an inverse miss at *any* rung, so attempt survival now also depends on the κ=4 margin (≈1e-6; low risk).

4. SUGGESTION (90) — Seed ranges consistent: v4 §4 (base line 168-170) → v4.1 §5 named 4200–4263/5400–5599 as the next block → v4.2 §4 takes exactly that and names 4300–/5600– next; disjoint from 3000–3099, 4000–/5000–, 4100–/5200–. No code encodes confirmatory seeds yet (grep clean), so the allocation is text-only until the freeze.

5. SUGGESTION (85) — Reset paths: `.weft/` and `runs/` are gitignored (`.gitignore:10,15`), so receipts remain machine-local as in every prior attempt; deleting `.weft/gate3-pilot-linked-pilot-2.json` blocks attempt 4 ("missing or changed") rather than enabling a reset; `claim_attempt` refuses an existing `PILOT_3_RECEIPT`; dual linkage flags refused (`pilot.py:292`). Pre-existing design, unchanged.

6. SUGGESTION (70) — Tests: the anchor-3002 fixture is a valid decision-rule control, not evidence for the κ=2 prediction (the amendment says so). `exhausted` fixture's `"4.0": 0.0` is non-physical for a monotone family; harmless. No test feeds the actual v4.1 `protocol_sha256` (`97b177…`) to `check_certificate`; rejection is implied by any-field mismatch rather than demonstrated for the predecessor.

7. SUGGESTION (60) — Beyond "Go with 1": the `--linked-pilot-3` entry and linkage are necessary machinery; retiring v4.1 confirmatory ranges follows v4 §7 applied conservatively (ladder is not impairment code, but the choice is harmless). `protocol_identity` docstring (`contracts.py:58`) still cites v4.1 §1; cosmetic.

---

## PyTorch and operations review (verbatim)

**HEAD** `c25c2ca260381befef2b300e4804b3c56ea91d78` on `codex/gate3-screen`, tree clean. Read-only review; no tests run, no GPU.

**Verdict: ACCEPT WITH CONDITIONS** (one protocol/code gap, no blocker).

**Findings**

1. SUGGESTION (conf 80) — Amendment §3 clause not enforced. `docs/plans/...v4.2-amendment.md:49-51` says the three attempt-3 rungs "cannot be chosen by a later measurement that happens to pass them", but `magnitude_decision` (`src/sultai_screen/pilot.py:290-305`) walks all of `P3_LADDER` and takes the first rung clearing 0.05 on every anchor; κ=1.0 would be chosen if anchor 3002 came in at ≥0.05. `tests/test_v42_ladder.py:42-52` only exercises the `"1.0": 0.04` shape. Mitigation: training path is byte-identical to attempt 3, seeds fixed, determinism flags set (`host.py:22-27`), so the first three P3 gains should reproduce bitwise (3002: −0.00746, −0.00364, 0.03987 from `docs/results/gate3-pilot-2026-10-13-linked-2/materiality-3002/result.json`). Condition: launcher verifies that equality after the materiality stage; divergence is an environment finding, not a pass.

2. SUGGESTION (conf 90) — P3 numerics at κ=2.0/4.0 are safe. `impair.py:57-58` computes `offset` in float64 and casts once (`impair.py:65`); `undo` (`impair.py:28`) subtracts the identical float32 tensor, so there is no apply/undo mismatch. κ·RMS is the L2 norm of the 16-vector; per-channel shift is max|dir_i|·κ·RMS ≈ 0.4·4·0.49 ≈ 0.83 (worst case ≈ 2.0). `(z+c)−c` has an absolute, not relative, error bound: ≤ 1 ulp(|z+c|) ≈ 2.4e-7 (CPU check: 1.2–2.4e-7 max, 87–93% of elements exact). `linalg.inv(eye)` returned `eye` bitwise; identity einsum is exact (1·x + Σ0·x) with TF32 off (`host.py:26-27`). Downstream BN/ReLU/CE are mildly Lipschitz, leaving ~3 orders of headroom under the 1e-4 check (`screen.py:244`). Residual: a nonfinite loss at κ=4.0 would raise `PilotHold` (`fit.py:73-74`), a software stop that spends the attempt; log-softmax makes this implausible at these magnitudes.

3. SUGGESTION (conf 95) — Cost negligible. Attempt 3 materiality workers ran 1.58–1.71 s in-process, charged 2.8–3.1 s each (`ledger-020.json` last six entries). Rungs 8→10 ≈ +25% of ~1.6 s ≈ +0.4 s/anchor, ≈ +2.5 s total against 7865.66 s remaining. `project_cost` (`costs.py`) has no materiality term, so timing admission is untouched. Single-job Supervisor deadline ≈ remaining−10 s (`pilot.py:148-152`).

4. Identity plumbing verified. `source_identity` checks base, v4.2 and v4.1 bytes (`screen.py:61-66`); pins match on disk (`0933ad4e…`, `dd30e1d1…`, `7668d20a…`); v4.2 file quotes the v4.1 hash. Composed SHA independently recomputed: `9126b6fb…9a2cd5aba`. v4.1-identity certificates rejected at `pilot.py:382-384`; v4.1 training receipts at `screen.py:126-132`. Fixture patches `AMENDMENT_V41` (`tests/test_protocol_identity.py:41,81,103`).

5. `linked_pilot_three` (`pilot.py:535-568`) verified. All nine pinned hashes match local `.weft/` and `runs/` files; the three new ones match published copies (`attempt-custody.json` acabf167…, `pilot-result.json` dd8659a0…, `ledger-020.json` 0e1c184b…). Attempt-3 terminal fields (verdict `instrument-failed`, charged 2934.3435500559863, opening 1473.267264504917, ceiling 10800) match the published records. Opening charge wired at `pilot.py:667-669`; one-linkage guard `pilot.py:620-621`; CLI `--linked-pilot-3` `pilot.py:875-879,887`; custody `.weft/gate3-pilot-linked-pilot-3.json` (absent now, gitignored). No reset path; `claim_attempt` refuses an existing receipt (`pilot.py:409-413`). Minor, pre-existing: missing certificate keys raise `KeyError` rather than `PilotHold`, still before `claim_attempt`.

6. Evidence hashes match `git show c25c2ca:` for contracts.py (0d97ae03…), pilot.py (6f686749…), screen.py (6ce27840…), impair.py and test_v42_ladder.py; `runtime.json` `source_sha256` c5c6c41e… equals the recomputed aggregate at c25c2ca. `quality.json` lists only the 24 src files (no tests, no aggregate) — consistent with prior packages.

**Launcher must know (`--linked-pilot-3`)**
- Working tree must be clean: commit the certificate and review docs first. HEAD may advance past c25c2ca (`merge-base --is-ancestor`, `pilot.py:393-397`) provided src/scripts/tests/pyproject.toml bytes are unchanged.
- Certificate: four v4.2 identity fields, `implementation_source_sha256 = c5c6c41e…`, `linked_pilot_3_approved: true`, `prior_charge_gpu_process_seconds: 2934.3435500559863`, ≥3 evidence entries with sha256 including the disposition at `a3b3d203…`.
- Nine prior receipts must stay byte-preserved; `.weft/gate3-pilot-linked-pilot-3.json` must not exist; new root under `runs/`.
- `check_handover` (`pilot.py:74-113`) still needs both Simic `launch-finished.json` receipts, both pinned GPU UUIDs idle, no compute apps.
- Projected 3000–4000 s of 7865.66 s remaining; cap stop spends the attempt.
- Post-materiality: confirm κ≤1.0 gains equal attempt 3 bitwise (finding 1).
