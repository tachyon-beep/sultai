# Gate 3 headroom screen — protocol amendment v4.1 (fixed-shape frozen evaluation)

Status: **operative amendment to the frozen v4 preregistration, pending
exact-source independent review and John's go for execution.** This file is
hashed separately and composed with the unchanged v4 bytes into one active
protocol identity (section 1). It authorises no GPU execution by itself.

Author: `sultai-claude-nyx` (the v4 author), 2026-10-13, under John's
authorisation of 2026-10-12 recorded in
[the authorisation certificate](../certificates/2026-10-13-gate3-fixed-evaluation-authorization.md).
Owner of record in the tracker remains `codex-sultai-gate3` (paused); the
tracker claim could not be reassigned by this session's tooling.

## 1. Identity, precedence and what receipts must carry

- **Base**: `docs/plans/2026-10-10-gate3-headroom-screen.md`, v4, bytes
  unchanged, SHA-256
  `7668d20ad55c3a62384975a7cf1f3989dbe69bf701ce02bc9c68ac2601d0d42a`.
- **Amendment**: this file, hashed by the screen package as
  `AMENDMENT_SHA` in `src/sultai_screen/contracts.py`. The file cannot state
  its own hash; the package pins it and the unit test
  `tests/test_protocol_identity.py` fails if the bytes and the pin diverge.
- **Active identity**: `protocol_version = "gate3-v4.1"` and
  `protocol_sha256 = SHA-256("gate3-v4.1\n" + base_sha + "\n" + amendment_sha + "\n")`,
  computed in `contracts.py` from the two pinned hashes.
- **Precedence**: this amendment overrides the base only in the clauses
  enumerated in sections 3 to 6. Everything else in v4 governs unchanged.
  There is exactly one active identity; a v4-only identity is retired.
- **Receipts**: every source-identity record, admission record, worker
  result, training checkpoint receipt, stop report, pilot result and review
  certificate carries all four fields: `protocol_version`, `protocol_sha256`,
  `preregistration_sha256` (base) and `amendment_sha256`. A record that
  carries only the base field, or a mismatching composed hash, is rejected
  with a hold. Training receipts from a different `protocol_sha256` cannot be
  loaded as anchors.

## 2. Why this amendment exists

The linked development pilot of 2026-10-11 (attempt 2) stopped at its first
materiality step with "resampled no-op evaluation twin is not bitwise". The
twelve development hosts, every replay check and timing admission had passed.
The bounded diagnostic of 2026-10-12 localised the discrepancy: exactly the
16 examples whose evaluation batch size changed under the permutation had
changed logits, with a maximum loss delta of about 6e-7 nats; an identical
repeat, a batch-size-preserving permutation and a padded evaluation were all
bitwise exact. See
[the diagnostic result](../certificates/2026-10-12-gate3-twin-diagnostic-result.md)
and [the design review](../reviews/gate3-headroom/fixed-evaluation-design-review-2026-10-12.md).

The v4 twin assertion asked for bitwise invariance under a resampled
evaluation order. The determinism contract in base §3 guarantees
run-to-run reproducibility for identical shapes; it never guaranteed
invariance to batch shape. With 5,000 development examples in chunks of 128
the last chunk had 8 rows, so a permutation moved examples between shapes.
The assertion was stronger than the contract supported. The implementation
stopped where it should have; the protocol over-promised. No scientific
quantity was read, so no scientific threshold is touched by this amendment.

## 3. Amended frozen evaluation (overrides base §3 rule on evaluation and §6 arm evaluation)

Every frozen evaluation of per-example losses in the screen uses one shared
evaluator, `sultai_screen.fit.losses`, with the following algorithm:

1. Inputs are nonempty, aligned `read`, `base` and `labels` of the same
   length N under the owned-feature contract; otherwise a contract violation.
2. Traverse the examples in caller order in chunks of exactly **128 rows**.
3. A partial final chunk is padded to 128 rows by duplicating its **last real
   example** across `read`, `base` and `labels`. No random draw, no extra
   data access, dtype, device and contiguous NCHW layout preserved.
4. The module (if any) is applied **after** padding, then the frozen
   downstream classifies and unreduced cross-entropy is computed.
5. Only the first real-count losses of the chunk are retained. The result
   has exactly N losses in caller order. Padding never enters a mean, a
   bootstrap, a candidate score, a count or an optimiser step.

Consumers covered by the shared evaluator, all of which change numerically:
the no-op and zero-module identity checks and the global-permutation twin;
the development losses of every post-hoc arm A1, A2, A3, A7, A8; the exact
inverse endpoint check (1e-4 nats) for A1; the selection score that chooses
every B3 oracle candidate and every cheap-bank candidate; the oracle schedule
comparator paired difference and its paired bootstrap SE; the P5 and P6
oracle development losses and their paired gain; the raw development
evaluation of every cheap candidate and the harm screen; the timing-stage
no-op development evaluation. Selection remains selection-only; a close
candidate may change winner under the new scores and all evidence is
recomputed, never inherited.

Unchanged paths: host training, the full-batch gradient step (an exact
mean-gradient accumulation over 128-row chunks of the 2,000 conditioning
examples, not an evaluation), optimiser updates, training loss streams, ridge
and former fitting, cached feature extraction. `Host.classify` is unchanged
because it also serves differentiable fits.

Alternatives considered and not adopted: a chunk size dividing every
evaluated set size (1,000 divides 2,000, 3,000 and 5,000) gives the same
shape guarantee with no padding but forces every CPU fixture onto multiples
of the chunk; a batch-size-preserving twin leaves the endpoint's batch-shape
sensitivity in place; a numerical tolerance is not justified by one observed
maximum.

## 4. Twin assertion and real counts (clarifies base §7 V0 clause 5 and §9 item 7)

The twin remains the **full global permutation** of the evaluation set,
asserted **bitwise** on every anchor, with stop semantics unchanged. With
every chunk at the same shape the determinism contract now covers the
assertion. Conditioning (2,000), selection (3,000) and development (5,000)
counts, every mean over N, the paired bootstrap over N differences, the
grids, schedules, rate sets, widths, arm definitions, V0/V1 logic and the
SD-before-median ordering are unchanged.

## 5. Fit-code change rule: new version and fresh confirmatory seed ranges (base §7 and §10)

Base §7 requires that any change to fit code carry a new preregistration
version with a new SHA-256, an explicit disposition and a fresh confirmatory
seed range. This amendment is that version. The disposition is John's
authorisation of 2026-10-12. The confirmatory ranges of v4 (intact
4000–4063, bottleneck 5000 onward ≤ 5199) were never executed and are
**retired unused**. The confirmatory ranges under v4.1 are:

| Lineage | v4.1 confirmatory seeds |
|---|---|
| Intact anchors | 4100–4163, 64 anchors |
| Bottleneck anchors | 5200 onward, contiguous, ≤ 5399 |

Development anchors stay in 3000–3099. The next development pilot trains
its hosts fresh; no checkpoint from attempts 1 or 2 is inherited.

## 6. GPU-time accounting and the linked development pilot (base §8 and §9)

- The pilot ceiling of 3 GPU-hours is operative as **10,800 aggregate
  process-seconds**: each launched child is charged its wall time from
  process creation through cleanup, co-tenants charged separately. This is
  the accounting the implementation has used since attempt 1 and it
  overcounts occupied-card time. The confirmatory ceiling's accounting is
  settled at the confirmatory freeze, not here.
- The ceiling is shared across every linked attempt. Charges carried
  forward: attempt 1 (provenance stop) 2.370860553928651 s; attempt 2 (twin
  stop) 1466.9817409639945 s; diagnostic 3.9146629869937897 s; cumulative
  **1473.267264504917 s**; remaining **9326.732735495083 s**. Budget is not
  authorisation.
- The next development pilot is **linked pilot 2** (attempt 3). Its custody
  receipt is `.weft/gate3-pilot-linked-pilot-2.json`. Admission binds, by
  hash, the three prior custody receipts and the terminal charge records of
  attempt 2 and the diagnostic, carries the cumulative charge as its
  opening charge, and requires a review certificate that names John's
  separate disposition document for that pilot. A different output path
  cannot create a further attempt. Timing admission, replay and every §9
  item are remeasured; nothing from attempts 1 or 2 is evidence for v4.1.

## 7. What this amendment does not change

X = 0.02 nats, ρ = 0.8, Hm = 0.01, U = 25%, η-set and η-set-cheap, the
conditioning and selection sizes, the schedule axis, the arms, the verdict
logic, the stop rules, the ordering rule, the data seal, the split
artefact, the determinism flag set, and the confirmatory ceilings of
16 GPU-hours and 10 hours wall. The Gate 2 verdict is unchanged.

## 8. Acceptance of this package

Before any GPU use: strict typing, policy, lint and format gates green on
both declared roots with negative controls; the CPU suite green including
`tests/test_fixed_evaluation.py` and `tests/test_protocol_identity.py`; independent scientific and PyTorch/operations
review of the exact committed bytes, preserved under
`docs/reviews/gate3-headroom/`; evidence under
`docs/results/gate3-fixed-evaluation-2026-10-13/`. Then John's separate go
and disposition document for linked pilot 2.
