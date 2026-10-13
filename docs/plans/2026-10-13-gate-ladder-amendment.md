# Gate-ladder amendment 1 to the hybrid correction protocol (2026-10-13)

**Status: drafted under John's instruction of 2026-10-13 ("ok go with the
recommendations"); the gate mapping below awaits his acceptance before the
M0 freeze. Until then this file authorises nothing and changes no verdict.**

Amends, prospectively, the frozen
[hybrid correction protocol](2026-10-09-correction-protocol.md) (SHA-256
`abbf3f52…`, pinned in the Gate 2 certificate, in run receipts and in
`src/sultai/correction.py`). **The protocol's bytes are not edited.** This
amendment is a separate, dated document; once accepted, its SHA-256 is bound
into the M0 study's identity beside the preregistration and the steering
statement (M0 draft §11), following the v4.1 precedent of one active
identity composed from separately hashed documents.

## Why an amendment and not a reading

The frozen protocol names the HTTYE Page's seven gates "authoritative
intent", and that Page reserves to the owner the right to "approve or amend
the minimum gate set". On 2026-10-13 John restated the objective
([STEERING.md §1](../STEERING.md)): efficient added capacity, not repair.
Gates 3 to 5 are worded as repair gates. A "reinterpretation" authored by
`sultai-claude-nyx` is weaker than the owner's amendment the source foresaw,
and a later reader of the frozen file would still see repair gates with a
Gate 7 that depends on them. Two independent reviews (a product-decision
critic and Codex; see
[`docs/reviews/m0-capacity/j-recommendations/`](../reviews/m0-capacity/j-recommendations/SYNTHESIS.md))
reached the same form.

## The seven gates under the restated objective

| Gate | Frozen wording (2026-10-09) | Status before this amendment | Prospective reading after acceptance |
|---|---|---|---|
| 1 | Contract hardening | delivered (v2) | **Unchanged.** |
| 2 | Instrument truth | certified on the synthetic lifecycle instrument ([certificate](../certificates/2026-10-09-hybrid-correction.md)) | **Unchanged.** Gate 2 certifies the synthetic instrument only; it does not cover the CIFAR screen. M0's instrument truth is its own V0 and P1 known answer. No pinned byte changes, so the Gate 2 verdict stands as issued. |
| 3 | Useful repair headroom | **not established.** The Gate 3 headroom screen (v4–v4.2) stopped on an instrument finding ([attempt-4 hold](../certificates/2026-10-13-gate3-linked-pilot-3-hold.md), [final disposition](../certificates/2026-10-13-gate3-linked-pilot-3-final-disposition.md)); the hold itself says the finding "is not a Gate 3 verdict", and the gate's transfer clause was never tested | **Retired as not established, not passed, and replaced** by a prospective capacity-headroom requirement read by rungs M0 and M1 ([STEERING §5](../STEERING.md)): a nonlinear module at a known site must beat every named cheaper route to the same loss at matched retained parameters, data and budget (M0 V1), and a builder must then find that capacity more cheaply than SGD (M1). Repair families remain instrument validation only. |
| 4 | Honest admission | not established (small synthetic results) | **Requirement unchanged; deferred.** No rung earlier than M1/M2 claims it; M0 inserts raw and admits nothing. |
| 5 | Benefit after removal | **not established** on the synthetic lifecycle; the Gate 2 certificate records "specificity fails" | **Requirement unchanged; next read by rung M2** under its own preregistered test, with predeclared trajectory, immediate-removal and final outcomes against no growth, training the final architecture directly, and early static capacity at matched information and compute. **M0's V2 (terminal truncation) never reads Gate 5.** |
| 6 | Calibrated safety/utility | blocked | **Partly instantiated, still blocked.** M0 fixes margins (X, X₂, M_s), a harm screen (Hm) and an accuracy secondary for one domain; the target domain, observable telemetry, tolerated degradation and an independent replication plan remain missing. |
| 7 | Training readiness | not established | **Predecessor set restated** as rungs M0–M3 plus gates 4 and 6. The retired repair Gate 3 is not a failed predecessor; it is replaced. Gate 7 stays not established while any restated predecessor is failed, blocked or unproven, and completing a certificate never marks a gate passed. |

## What this amendment does not do

It does not reopen or amend the closed Gate 3 headroom screen (v4, v4.1,
v4.2) or its final disposition. It does not change the Gate 2 verdict or any
pinned hash. It does not authorise any execution; M0's pilot and confirmatory
each require John's separate go. It converts no retired or failed evidence
into a pass.

## Acceptance

John accepts or amends the table above before the M0 freeze. On acceptance
this file's SHA-256 is recorded in the M0 package identity, decision D27 in
[DECISIONS.md](../DECISIONS.md) points here, and [STEERING.md §6](../STEERING.md)
carries only a link.
