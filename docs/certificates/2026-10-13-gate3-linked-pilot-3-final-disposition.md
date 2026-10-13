# Gate 3 attempt-4 hold: final disposition under John's redirection, 2026-10-13

**The Gate 3 headroom screen (v4, v4.1, v4.2) is closed on the attempt-4
instrument finding. No further attempt, no confirmatory, no amendment.**

The [attempt-4 hold](2026-10-13-gate3-linked-pilot-3-hold.md) offered three
options. John did not choose among them. He restated the objective instead:

> "Ok what we are trying to do here is build new more efficient capacity into
> the model, we aren't trying to repair it specifically, just increase its
> accuracy at the most efficient parameter cost possible."

> "Sultai is an alternate theory of the possible using diffusion to add and
> remove parameters instead of just plugging in a whole module."

> "before we do anything else, I'd like you to make sure sultai is steered
> properly, make sure the intent is durable and we actually recut the
> experiment to steer it at that intent … I want an updated experimental
> guide that measures progress against the actual goal"

Under that instruction the repair screen's primary question (bypass versus
post-bottleneck repair specificity) is a proxy the programme no longer
asks. This record therefore closes the screen with its attempt-4 finding as
the published result and retires what was never executed:

| Item | Disposition |
|---|---|
| Attempt-4 finding | Published: the primary repair family is undefined on this host; the module's gain is dominated by generic re-tuning ([hold](2026-10-13-gate3-linked-pilot-3-hold.md)) |
| v4.2 confirmatory seed ranges (intact 4200–4263, bottleneck 5400–5599, restarts 4300–/5600–) | Retired unused |
| Gate 3 pilot cap | 6,206.673065379029 s charged of 10,800; 4,593.326934620971 s unspent. The remainder is not carried into the recut study by default; see decision J6 in the recut draft |
| Custody receipts `.weft/gate3-pilot*.json` (five) | Intact, never reset |
| Evidence directories under `docs/results/gate3-*` | Preserved, byte-identical to `runs/` |
| Option 2 (bounded diagnostic) and option 3 (v5 bottleneck redesign) | Not taken; superseded by the recut |
| Tracker claim `sultai-ef0da96d8a` | Remains with `sultai-claude-nyx`; closure or re-scoping is recorded there when the tracker accepts the write |

**Correction to the preserved hold record** (found by the M0 reality
review, the hold's bytes are not edited): the hold says oracle plateau was
"certified on one of twelve width-4 fits". The receipts certify anchor
3003 at both P5 and P6 at width 4, so the count is two of twelve width-4
grids, two of twenty-four bottleneck grids, and two of thirty grids
including the six P1 grids. The reading of the hold is unchanged.

Successors: the [steering statement](../STEERING.md), the
[capacity frontier screen draft v1](../plans/2026-10-13-capacity-frontier-screen.md)
and decision D26 in [DECISIONS.md](../DECISIONS.md). The Gate 2 verdict is
unchanged. This record is a documentation change; `source_sha256` of the
Gate 3 package is unaffected.
