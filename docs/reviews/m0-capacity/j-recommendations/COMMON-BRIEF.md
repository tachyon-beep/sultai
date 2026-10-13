# Common brief for the M0 decision recommendations (2026-10-13)

You are one of several independent reviewers asked to recommend on a decision
that John (the project owner) must make before the M0 "capacity frontier
screen" preregistration is frozen. Repo: /home/john/sultai, branch
codex/gate3-screen (HEAD fdef5fc). Nothing has run on GPU for M0; nothing may.

## John's instruction, verbatim

> "please send out those subagents and have them each make a recommendation
> but also provide a strong rationale for and against the recommendation. (if
> there's only one recommendation that's viable, they only need to make one,
> don't make up options for the sake of it)"

and, on the unit of the objective (sent while this batch was being prepared):

> "also accuracy was just a value we used in esper, its not 'the value we
> have to use'"

The objective, in John's words (docs/STEERING.md §1): "build new more
efficient capacity into the model, we aren't trying to repair it
specifically, just increase its accuracy at the most efficient parameter cost
possible" and "Sultai is an alternate theory of the possible using diffusion
to add and remove parameters instead of just plugging in a whole module."

## Evidence you may read (all read-only)

- docs/STEERING.md (steering statement; rules §4, ladder §5, gate reading §6)
- docs/plans/2026-10-13-capacity-frontier-screen.md (draft v2; §12 is the
  decision table with the AUTHOR'S recorded defaults. Treat each default as
  one candidate, not the answer. Judge independently.)
- docs/reviews/m0-capacity/*.md (three v1 reviews and the dispositions: these
  are what earlier readers argued; judge independently, do not defer)
- docs/certificates/2026-10-13-gate3-linked-pilot-3-hold.md and
  -final-disposition.md (attempt-4 finding and closure)
- docs/results/gate3-pilot-2026-10-13-linked-3/ (attempt-4 receipts:
  02-magnitudes.json, 03-posthoc-known-answer.json, execution-summary.json,
  01-timing-replay-decision.json, per-anchor */result.json, ledger-*.json)
- src/sultai_screen/ (host.py, fit.py, screen.py, contracts.py, pilot.py,
  costs.py, data.py, impair.py) and tests/
- docs/CONCEPT.md, docs/DECISIONS.md, docs/plans/2026-10-09-correction-protocol.md
- docs/research/claude-handoff/sultai-concept-demonstrator-design.md
- /home/john/archive/esper-lite (README.md, ROADMAP*, docs/) as the baseline
  theory, read-only

Known facts you may rely on (verify if you cite them): reference host
284,634 parameters, trained 10 epochs; module M-sq 272 parameters; attempt-4
P5/P6 single-arm gains 0.037–0.260 nats on impaired hosts fitted on held-out
data; no unimpaired-host fit and no paired difference between two fitted arms
exists in the receipts; measured kernels: host train 44.5 s charged, oracle
~50 s per 24-fit grid, posthoc ~55 s, dev eval 0.038 s; core ledger ≈ 8,800
process-seconds, extended ≈ 3,100 s; PILOT_SECONDS = 3*3600 is existing code;
Gate 3 unspent remainder 4,593.33 s.

## Hard rules

- Read-only on the repository. Do not edit, create or delete any repo file.
- No GPU use, no training, no `git` commands that write (no add/commit/
  checkout/stash/reset). `git log`/`git show`/`git diff` are fine.
- Never read any CIFAR test batch. Never touch `.weft/`.
- Do not spawn sub-agents.
- Write exactly one file: your assigned output path. Nothing else.
- Do not run the test suite on GPU; CPU tests (`PYTHONPATH=src /usr/bin/python3
  -B -m unittest discover -s tests`) are permitted if you need them.

## Report template (cap ~800 words; write it to your assigned path)

```
# <Decision id>: <one-line title>
Agent: <your agent type>   Date: 2026-10-13

## Options considered
Viable options only. One line on anything you dropped and why.

## Recommendation
One option, stated so John can accept it in a sentence.

## Strongest case for
## Strongest case against
(about the recommendation, not about every option)

## Dependencies on other decisions
If this depends on another J being decided a particular way, say which and
how (J0 ↔ J1–J3; J-D2 ↔ J2; J4 ↔ J6; J-host ↔ J4/J6 are known couplings).

## Evidence
path:line citations for every number or claim you rely on.

## Versus the recorded §12 default
Match or differ, and why, in two or three sentences.

## What would change this recommendation
## Confidence
low / medium / high, one line of reason.
```
