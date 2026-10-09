# esper-lite: what it was, why it stalled, what Sultai keeps

Read 2026-10-09 from github.com/tachyon-beep/esper-lite at `main` @ `2d296b6c` (2290 commits, 2025-11-26 to 2026-07-06). Read-only; nothing changed in the repo.

## What it was

A PPO controller trained to change a host network's topology *during* training. Every RL step is one host epoch: a 150-step episode is one full host training run on CIFAR. Up to 3 slots can hold a "seed" module. The policy watches host and seed statistics and decides what to do with each seed.

## The three seed/reap subsystems and whether they matter for Sultai

| Subsystem | What it did in esper-lite | Role in Sultai |
|---|---|---|
| **Kasmina** (body) | Host network plus `SeedSlot`s (`src/esper/kasmina/slot.py`), a blueprint registry of CNN/transformer seed modules, alpha-blending schedules, and gradient isolation so a seed trains "behind" the host without changing its forward pass. The lifecycle runs DORMANT → GERMINATED → TRAINING → BLENDING → HOLDING, then FOSSILIZED or PRUNED → EMBARGOED → RESETTING (`leyline/stages.py`). | Slots, blueprints, blending and the lifecycle state machine go. Two ideas may survive: a host you can switch modules off in, and gradient isolation, if Sultai still has components it wants to measure separately. |
| **Tamiyo** (brain, seeding) | A factored-action LSTM PPO policy (`tamiyo/networks/factored_lstm.py`, 512 features with a 512 LSTM). Its ops are WAIT, GERMINATE, SET_ALPHA_TARGET, PRUNE, FOSSILIZE and ADVANCE, each with slot, blueprint and alpha-speed heads. The observation is 128 dims, of which 31 per slot × 3 slots are slot-shaped. | The action space and observation are built around discrete slot decisions, so neither carries over. Most of the 2026 RL work (value-target variance, EV-stabilization, HRA value heads) treats symptoms of this particular sparse, factored, recurrent PPO setup. If Sultai has no discrete grow/prune decisions, all of that goes with it. |
| **Emrakul** (immune system, reaping) | **Never built.** It exists only as design docs (`docs/plans/planning/emrakul1/`). It was meant to be a decay policy over committed (fossilized) structure: fossilized seeds get rewrapped as "phages" that are probed, sedated and lysed when they stop paying rent. What actually ran was PRUNE from Tamiyo plus a rent penalty. | Obsolete as a separate controller. The question it was meant to answer is still live, though: does this component still earn its cost? That is an ablation question. |

## What "collapsed under telemetry" means in the record

- **Telemetry came first.** The 2nd and 3rd commits (2025-11-26) were "rich telemetry design" and "v2 telemetry system". The telemetry stack (Leyline contracts, Simic emitters, Nissa, and Karn with the Sanctum TUI and Overwatch web UI) is about **33k of 77k Python lines (~43%)**, plus ~14k lines of TS/Vue. Kasmina, Tamiyo and Tolaria *together* are ~12k. It has 35 event types, 27 payload classes, ~600 typed fields and 179 TELE-xxx metric specs.
- **The learner was broken while the dashboards grew.** 1424 commits landed in Dec 2025, most of them Sanctum and Overwatch. `docs/analysis/2026-01-03-training-failure-analysis.md` shows value-function collapse (negative EV from batch 1), and the diagnostic needed to see it was not logged. Work stopped on 2026-01-17 and nothing landed for 5 months.
- **The telemetry was wrong in ways that drove decisions.**
  - A false "entropy collapse": the metric was learnable_fraction × conditional entropy, diluted by ~60% forced steps, and `entropy_loss=0.0` was a hardcoded stub (PDR-0006). The Jan 2026 entropy-floor fixes were likely chasing it. That is my inference from the metric definition, not something the docs state.
  - The June 2026 health audit (`archive/arch-analysis-2026-06-13-1138-telemetry-health/`) found 17 P1 blockers. Missing data was shown as zero, the static and fixed baselines were WAIT-only placeholders ("baseline comparisons are not valid"), and the param_ratio semantics diverged across exports.
  - A telemetry forward pass poisoned the autocast cache, so BF16 training silently stopped training the policy (`81a3a3a7`, introduced and fixed 2026-06-14).
- **Proof tooling took over.** In the June restart the agent "self-expanded scope into a full reward-efficiency proof regime" (`2026-06-15-whats-left-rom.md`). As of the last checkpoint (2026-07-06), the current gate is still "NOT-yet-freezable pending the telemetry wrapper".

## Did it work?

- **Morphogenesis helps the host:** all seeds on vs all disabled gives +7.7pp (n=6, ~38% → ~46%).
- **The learned controller never shows a win.** There is no result where PPO beats the heuristic or a static schedule. Final accuracy is arm-invariant at ~42–44% with every CI spanning 0. The north-star "committed J" is ~0, and the Shapley-credit A/B was null.

The one positive result is itself an ablation result. Nothing shows the controller doing better than a fixed schedule.

## What Sultai keeps

1. **The ablation/counterfactual engine** (`simic/attribution/counterfactual.py`) has full-factorial removal cost up to 4 components and sampled Shapley above that. Its own docstring carries the caveat that matters most if ablation is central: removal cost at epoch T is *confounded*, because the host has adapted to the component's presence. "How much worse if I remove it now" is not "how much did it add". The clean answer needs parallel control runs.
2. **Placebo noise floor.** Measure what a null player scores before claiming any effect (tau ≈ +0.28pp, and it turned out to be magnitude-dependent).
3. **Accuracy minus rent** as the objective, and param-normalized contribution (J).
4. **A fixed-schedule baseline from day one.** esper-lite's baselines were placeholders for months.
5. **Telemetry lessons:**
   - Keep it small until the learner works.
   - Missing must never render as zero.
   - Derived metrics need their definition written next to them.
   - Telemetry must never run inside the training forward or autocast path.

## Not settled by the repo

- **Ablation vs diffusion.** The repo has no diffusion code or design (no hits in `src/`), and nothing ties "ablation" to "diffusion". It doesn't say whether ablation replaces the diffusion framing or sits alongside it.
- ~~What "simic" means.~~ Settled 2026-10-09: simic is John's reboot of esper-lite as a whole. Sultai is a variant of simic that swaps the seed architecture (Kasmina/Tamiyo/Emrakul) for an alternative. The simic repo itself has not been read; this review covers esper-lite only.
- **Whether a controller exists at all.** If Sultai is diffusion-style, with structure emerging continuously instead of being decided slot by slot, it's unclear whether any RL controller remains, or whether ablation is the only selection mechanism.
