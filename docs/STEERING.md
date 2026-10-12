# Sultai steering statement

Status: programme steering record, written 2026-10-13 by `sultai-claude-nyx`
under John's instruction. It governs how every later preregistration states
its primary endpoint. It authorises no execution. It becomes durable for the
repository when the branch carrying it is merged to `main`; until then it is
a committed proposal on `codex/gate3-screen`.

## 1. The objective, in John's words

On 2026-10-13, after the Gate 3 attempt-4 hold, John wrote:

> "Ok what we are trying to do here is build new more efficient capacity into
> the model, we aren't trying to repair it specifically, just increase its
> accuracy at the most efficient parameter cost possible."

and, pointing at the archived Esper-lite project as the baseline theory:

> "Sultai is an alternate theory of the possible using diffusion to add and
> remove parameters instead of just plugging in a whole module."

and then:

> "I'd like you to make sure sultai is steered properly, make sure the intent
> is durable and we actually recut the experiment to steer it at that intent
> … I want an updated experimental guide that measures progress against the
> actual goal."

[CONCEPT.md](CONCEPT.md) already records the long-run form of this objective:
comparable capability at about half the final retained parameters of a
credible reference, developed by addition **and** deliberate physical
removal, with diffusion as one candidate builder rather than the definition
(decisions D01, D02, D09, D14 in [DECISIONS.md](DECISIONS.md)). This
statement does not restate that; it fixes how progress toward it is measured
and why the programme had drifted.

## 2. The baseline theory: Esper-lite

Esper-lite (archived at `/home/john/archive/esper-lite`, read-only) is a
morphogenetic framework: "grow capabilities, don't just train weights". A
seed module from a blueprint catalogue is germinated at a slot, trained in
isolation, blended in, and fossilised if it earns its keep; its reward is
accuracy minus parameter rent. Sultai's departure is the builder, not the
goal: instead of plugging in a whole catalogue module, a generator proposes
an oversized local addition from local evidence, selection and physical
truncation remove what does not pay, and sometimes nothing survives. Both
theories are judged by the same currency: **accuracy gained per parameter
retained, at disclosed compute.**

## 3. The drift, and why it happened

The Gate 3 headroom screen (`plans/2026-10-10-gate3-headroom-screen.md`,
v4 to v4.2) asked a repair-specificity question: does a loss-fitted module
gain more at a bypass site than at the matched post-bottleneck site. That
framing was inherited from the independent Claude demonstrator design, where
the repair families are the *precondition* for the generator route (is there
anything for a generator to amortise?). That design says in its own words
that its first rung "cannot show parameter efficiency, and it is not meant
to". Four attempts and 6,207 GPU process-seconds later, the screen stopped on
its own rule because the primary repair family could not be defined on the
host. The instrument is sound; the question was a proxy. The attempt-4
evidence already contains the first reading on the real objective, by
accident: a 272-parameter module, 0.1 percent of the host, lowered
development loss on every host it was fitted to, whether or not it could see
any "lost" information
([attempt-4 hold](certificates/2026-10-13-gate3-linked-pilot-3-hold.md)).
Nothing in that evidence separates new capacity from re-tuning of an
undertrained host, which is exactly the control the recut study adds.

## 4. Rules for every preregistration from here on

1. **The primary endpoint is stated in retained-parameter and loss units.**
   Every arm reports a point on the plane (physically retained parameters,
   development loss) with its compute booked. Development accuracy is
   reported beside loss; loss is primary because it is the quantity every
   arm optimises and the one the existing instrument measures with
   paired-bootstrap intervals.
2. **"Efficient" is a comparison against named cheaper routes to the same
   loss**, never against zero: re-tuning an equal count of the host's own
   parameters on the same data; more training of the host; the same
   capacity present from the start; and, as the scale reference, the
   from-scratch width frontier of the host family at matched compute.
3. **Removal means slicing.** A module counts as smaller only when its
   tensors are physically smaller and the printed parameter count falls.
   Zero masks, zero gates and zero blend weights are not removal.
4. **Compute is booked, not hidden.** Example passes and optimiser
   parameter-steps per arm, as the Gate 3 ledger already does; corpus,
   generator and probing costs join the ledger when those arms exist.
5. **Repair families are instrument validation only.** The P1 known answer
   (exact inverse, affine ceiling, ρ = 0.8) stays in every study as the
   check that the fitter works. No repair family is a primary endpoint again
   unless John re-states the objective.
6. **Every margin is fixed by John before the pilot opens**, as in v4 §9.

## 5. Progress ladder, with the measure at each rung

| Rung | Question | Primary measure | What beats it |
|---|---|---|---|
| **M0** (next: [capacity frontier screen](plans/2026-10-13-capacity-frontier-screen.md)) | Does capacity added at a site by an ordinary loss fit buy loss that re-tuning an equal count of existing parameters does not, and does an overgrown module physically cut to the small budget keep more of its gain than a module built small? | Per-anchor paired gain differences in nats at matched retained parameters and matched data; frontier-equivalent parameters as the scale | Re-tune control equal to the module; built-small equal to cut-down |
| **M1** | Can a learned or generated builder find that capacity at lower cost, or better, than the SGD fit, on hosts it has not seen? | Recovered fraction of the SGD-fit gain at zero task-gradient steps, against retrieval and a deterministic hypernetwork at matched evidence | Retrieval equal to generation; or SGD in a few steps equal to both |
| **M2** | Starting from a starved host, does grow-and-remove reach the frontier at fewer retained parameters than static width? | Frontier excess in nats at the retained count, at matched compute, against static capacity and the cut-down architecture retrained from scratch | Static capacity at the same count equal or better |
| **M3** | The half-retained-parameters claim | Loss of the grown model against the from-scratch width sweep at matched or larger compute, all cost ledgers disclosed | The sweep at half the parameters |

A rung opens only when the previous rung's verdict says "present" under its
preregistered margins; a negative at any rung is a published finding, not a
reason to move the goalposts.

## 6. Gate reinterpretation (proposed, pending John)

The frozen [correction protocol](plans/2026-10-09-correction-protocol.md)
words gates 3 to 5 as repair gates. Proposed reading under this statement,
not an edit of the frozen text: Gate 3 "useful repair headroom" is closed on
the attempt-4 finding; the M0 screen stands in its place as "useful capacity
headroom"; Gate 5 "benefit after removal" is read as M0's add-and-remove
contrast. Gates 4, 6 and 7 are unchanged. John confirms or amends this
reading before the M0 freeze.

## 7. Where to look

- The recut study: [capacity frontier screen draft v1](plans/2026-10-13-capacity-frontier-screen.md).
- Closure of the repair screen: [attempt-4 final disposition](certificates/2026-10-13-gate3-linked-pilot-3-final-disposition.md).
- Decision record: D26 in [DECISIONS.md](DECISIONS.md).
- Session pickup: [RESUME.md](RESUME.md).
