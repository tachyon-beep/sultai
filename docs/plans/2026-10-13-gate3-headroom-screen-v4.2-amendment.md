# Gate 3 headroom screen — protocol amendment v4.2 (P3 materiality ladder extended)

Status: **operative amendment, pending exact-source independent review and
John's bound certificate for linked attempt 4.** It supersedes amendment
v4.1 as the active amendment and incorporates v4.1 by reference; the
unchanged v4 base and the v4.1 text remain in force except where this
amendment says otherwise. It authorises no GPU execution by itself.

Author: `sultai-claude-nyx`, 2026-10-13, under John's disposition "Go with 1"
recorded in
[the linked pilot 3 disposition](../certificates/2026-10-13-gate3-linked-pilot-3-disposition.md),
choosing option 1 of
[the attempt-3 hold](../certificates/2026-10-13-gate3-linked-pilot-2-hold.md).

## 1. Identity and precedence

- **Base**: `docs/plans/2026-10-10-gate3-headroom-screen.md`, v4, bytes
  unchanged, SHA-256
  `7668d20ad55c3a62384975a7cf1f3989dbe69bf701ce02bc9c68ac2601d0d42a`.
- **Superseded amendment v4.1**:
  `docs/plans/2026-10-13-gate3-headroom-screen-v4.1-amendment.md`, bytes
  unchanged, SHA-256
  `dd30e1d171c696f3a51dba291475b2cc09ff0626442ccc411cdbfcfd58c32b4d`,
  pinned by the package as `AMENDMENT_V41_SHA`. Every v4.1 clause stands
  unless §3 to §5 below override it.
- **This amendment**: hashed by the package as `AMENDMENT_SHA`;
  `tests/test_v42_ladder.py` fails if the bytes and the pin diverge.
- **Active identity**: `protocol_version = "gate3-v4.2"` and
  `protocol_sha256 = SHA-256("gate3-v4.2\n" + base_sha + "\n" + amendment_sha + "\n")`,
  where `amendment_sha` is this file's hash. Because this file quotes the
  v4.1 hash, the chain v4 → v4.1 → v4.2 is fixed by one identity. Receipts
  carry the same four fields as under v4.1 §1; a v4.1 or v4-only identity
  is rejected.

## 2. Why

Linked pilot 2 (attempt 3) ran clean through training, replay, timing
admission and the repaired twin, and stopped at the materiality stage with
an instrument finding: P3 exhausted its ladder {0.25, 0.5, 1.0}. At κ = 1.0
the exact-inverse gain cleared 0.05 nats on five development anchors
(0.13 to 0.25 nats) and reached 0.040 nats on anchor 3002. Lower rungs were
small and occasionally negative on several anchors: a bias shift of a
batch-normalised, rectified feature is largely absorbed downstream, and the
gain grows faster than linearly in κ. v4 §9 forbids the pilot from adding
rungs; v4 §9 item 4 requires a new draft before freeze. This is that draft.
P5, P6 and P8 were never opened; no scientific quantity was read.

## 3. P3 ladder (overrides the P3 line of the v4 §5 ladder rule)

The P3 ladder is **κ ∈ {0.25, 0.5, 1.0, 2.0, 4.0}**, in that order. The
rule is unchanged: the magnitude is the first rung at which A1 gain is at
least 0.05 nats on every development anchor; the three rungs that failed in
attempt 3 remain on the record as failed and cannot be chosen by a later
measurement that happens to pass them. The materiality worker evaluates all
five rungs on every anchor so the whole curve is published. P1 and P2
ladders, the rule's threshold and every other magnitude are unchanged. The
prediction from attempt 3's curvature that κ = 2.0 clears the bar on anchor
3002 is a prediction, not evidence.

## 4. Version and fresh confirmatory seed ranges (v4 §7 rule, as in v4.1 §5)

The ladder lives in impairment-adjacent code, so the v4 §7 rule applies:
new version and hash (this amendment), explicit disposition (John's
"Go with 1"), fresh confirmatory range. The v4.1 ranges (intact 4100–4163,
bottleneck 5200–5399) were never executed and are retired unused. Under
v4.2:

| Lineage | v4.2 confirmatory seeds |
|---|---|
| Intact anchors | 4200–4263, 64 anchors |
| Bottleneck anchors | 5400 onward, contiguous, ≤ 5599 |

A restart after a failed v4.2 confirmatory attempt takes intact 4300–4363
and bottleneck 5600 onward ≤ 5799 with linkage. Development anchors stay in
3000–3099; the next pilot trains its hosts fresh. John's recorded
confirmation of the allocation is a precondition of the confirmatory
freeze, not of the development pilot.

## 5. Linked pilot 3 (attempt 4) and accounting (extends v4.1 §6)

- Charges carried: attempt 1 2.370860553928651 s; attempt 2
  1466.9817409639945 s; diagnostic 3.9146629869937897 s; attempt 3
  1461.0762855510693 s; cumulative **2934.3435500559863 s**; remaining
  **7865.656449944014 s** of the unchanged 10,800 s cap. Attempt 3 reached
  the materiality stop in 1461 s; a fresh attempt 4 is projected at 3,000
  to 4,000 s. Cap exhaustion is an execution-invalid stop that spends the
  attempt.
- Custody receipt `.weft/gate3-pilot-linked-pilot-3.json`. Admission binds
  by hash all four prior custody receipts, the terminal charge records of
  attempt 2 (pilot result and final ledger), the diagnostic charge, and the
  terminal records of attempt 3 (pilot result with verdict
  `instrument-failed` and final ledger), carries the cumulative charge as
  the opening charge, and requires a review certificate naming John's
  disposition document for attempt 4. No checkpoint from attempts 1 to 3 is
  inherited.

## 6. What this amendment does not change

Everything in v4.1 §3 (the fixed-shape evaluator), §4 (twin and counts) and
§6 accounting not restated here; X, ρ, Hm, U, the η sets, the conditioning
and selection sizes, the arms, V0/V1, the ordering and stop rules, the data
seal, the split artefact, the determinism flags and the confirmatory
ceilings. The Gate 2 verdict is unchanged.

## 7. Acceptance of this package

Gates green on both roots with negative controls; the CPU suite green
including `tests/test_v42_ladder.py`; independent scientific and
PyTorch/operations review of the exact committed bytes preserved under
`docs/reviews/gate3-headroom/`; evidence under
`docs/results/gate3-v42-p3-ladder-2026-10-13/`; then the bound certificate
and fresh host checks before launch.
