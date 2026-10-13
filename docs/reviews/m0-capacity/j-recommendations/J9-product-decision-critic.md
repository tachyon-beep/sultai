# J9: Gate reinterpretation (STEERING §6) — reading, supersession or amendment
Agent: axiom-product-management:product-decision-critic   Date: 2026-10-13

## Options considered
- **A. Adopt the §6 reading as written**, recorded only in D26. Viable but under-specified: maps gates 3 and 5, calls 4/6/7 "unchanged", leaves the frozen file with no forward pointer.
- **D. Same substance as a separate, dated, hashed gate amendment**: protocol bytes untouched, all seven gates mapped, indexed from README and D26. Precedent: the v4.1/v4.2 amendment files.
- Dropped: *supersede the whole protocol* — it is the synthetic instrument's full spec, SHA-pinned by the Gate 2 certificate and receipts; renumbering orphans gate numbers in three certificates. *Keep gates literal, separate track* — contradicts D26 and leaves Gate 7 blocked forever by repair gates nobody will pursue.

## Recommendation
Adopt §6's substance as **D**: a one-page `docs/plans/2026-10-13-gate-ladder-amendment.md`, John's confirmation recorded, SHA pinned in the M0 identity beside STEERING, pointer added at README.md line 25 and in D26; STEERING §6 becomes a link. Rows: Gates 1–2 unchanged (Gate 2 certifies the synthetic instrument only; M0's instrument truth is V0/P1). Gate 3 **retired as not established, replaced by rungs M0–M1** — not "closed", not passed. Gate 4 deferred to M1/M2; no earlier rung claims it. Gate 5 "not established on the synthetic lifecycle; next read by M2 under its own preregistered test, never by M0 V2". Gate 6 partly instantiated by J0–J3, Hm and the accuracy guard; still blocked on replication. Gate 7's predecessors restated as M0–M3 plus gates 4 and 6; the retired repair gate is not a failed predecessor.

## Strongest case for
1. **Authority.** The protocol declares the HTTYE Page's "seven gates and their scientific demands … authoritative intent", and the Page reserves to the owner "approve or amend the minimum gate set". Amending is the owner act the source foresaw; a "reading" authored by `sultai-claude-nyx` is weaker than what John holds.
2. **The frozen file cannot point forward and must not be edited**: its SHA `abbf3f52…` (verified today) is pinned in the Gate 2 certificate and run receipts, and `correction.py` pins the path. A later reader of the protocol sees repair gates and a Gate 7 that depends on them; only an indexed amendment stops that reader being misled.
3. **§6 changes meaning it does not acknowledge.** "Gate 3 is closed on the attempt-4 finding" conflates the screen's closure with the gate's; the hold says the finding is "not a Gate 3 verdict", and the gate's transfer clause was never tested. "Gate 5 stays open" misdescribes a gate already certified "not established; specificity fails". Gate 4 has no rung reading it (M0 inserts raw); Gate 6's margins are being decided now as J1–J3; Gate 7 inherits a permanently unestablished predecessor unless restated.
4. **Gate 5 → M2 is faithful**: the Page demands "retained benefit after removal … against no growth, training the final architecture directly, and early static capacity at matched information and compute", which is M2's measure. Keep it; V2-as-Gate-5 was rightly rejected (A4).
5. **Gate 2 stays intact** because no pinned byte changes; say so, and that it does not cover the CIFAR instrument.

## Strongest case against
One more file in a programme the alignment review already called oversized; the seven-row mapping could live in the D26 row plus a README pointer with equal durability. Option A with three wording fixes (retired not closed; Gate 5 not established, next read M2; Gate 7 predecessors restated) and the README pointer captures most of the value. The hash pin also touches the M0 identity string.

## Dependencies on other decisions
- **J10**: nothing is durable until merged.
- **J-D2**: V2 as truncation precedent is what keeps Gate 5 with M2.
- **J-D02**: M2's start decides on which host Gate 5's "early static capacity at matched compute" is read; name it as the open input.
- **J0–J3**: Gate 6's margin/degradation items in miniature; do not call Gate 6 unchanged.
- **J-retire**: retired families "stay published Gate 3 evidence" — consistent with retiring the gate.

## Evidence
- Gate wording, Gate 7 rule, Page as authoritative intent, frozen status: `docs/plans/2026-10-09-correction-protocol.md:153-172, 18-21, 3-4`; one commit `944832e`, never edited.
- Page gate demands and owner choice: `docs/plans/2026-10-09-httye-source-page.md:23-37, 45`.
- Certified statuses: `docs/certificates/2026-10-10-gate2-nyx-final.md:7` (SHA pin), `:38-46`; `docs/certificates/2026-10-09-hybrid-correction.md:113-121`.
- Pins: `src/sultai/correction.py:23-24`; `docs/results/gate2-confirmation-2026-10-10-01a125af/confirmation-run.json:41`.
- "Not a Gate 3 verdict": `docs/certificates/2026-10-13-gate3-linked-pilot-3-hold.md:66-68`; closure: `…-final-disposition.md:3-4, 45`.
- Open transfer clause, no admission: `docs/plans/2026-10-10-gate3-headroom-screen.md:36-38, 60-63, 73-76`.
- Proposal and ladder: `docs/STEERING.md:119-127, 102-116`; `docs/DECISIONS.md:52`; `docs/plans/2026-10-13-capacity-frontier-screen.md:426, 402-408`.
- Amendment precedent: `docs/plans/2026-10-13-gate3-headroom-screen-v4.1-amendment.md:3-6, 16-24`. Index: `README.md:25`.
- Reviews: `docs/reviews/m0-capacity/alignment-review-v1-2026-10-13.md:26`; `v1-dispositions-2026-10-13.md:17`.

## Versus the recorded §12 default
Differs in form, not substance: the Gate 3→M0 and Gate 5→M2 moves are right. The default records them as a reading with no mapping for gates 4, 6, 7, wording that misstates certified status, and no pointer reachable from the frozen file; the amendment form fixes those for one short file.

## What would change this recommendation
If John will update the parent-owned HTTYE Page himself, the repo record shrinks to a pointer and A suffices. If he wants gate numbering abandoned, supersession is the right form, with a cross-reference note in the three certificates that cite gate numbers.

## Confidence
Medium-high: authority chain and SHA pins verified from source; the 4/6/7 mapping is my reading of wording no certificate has yet dispositioned under the new objective.
