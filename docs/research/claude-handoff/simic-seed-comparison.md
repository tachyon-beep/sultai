# simic vs the esper-lite seed design: what Sultai would replace

Read 2026-10-09 from github.com/foundryside-dev/simic at `main` @ `16bcf70` (198 commits). Read-only. Companion to [esper-lite-review.md](esper-lite-review.md).

## Where simic is

- `src/simic/` is a 2-line package scaffold. The 14-domain HLD under `docs/design/` is a design, not code.
- What runs is `experiments/`: `kernel_demo.py` (4.3k lines) and the bounded comparison, screen and lifecycle studies (~3k lines). These train one CIFAR host three ways: no growth, static added capacity, and a scheduled graft.
- Results so far:
  - The instrument resolves ±0.018 nats.
  - On `mild` the graft earns nothing.
  - On a deliberately broken host (`under_normalized`), the graft captures 60% of what static capacity gains with the `norm` seed, and 31% with `conv_heavy`. **Static beats the graft in both.**
  - Rung 4 (graft timing and training horizon) is next. The ladder stops there if neither changes the outcome.
- There is no learned controller and no growth-superiority result (README, `docs/product/current-state.md`).

## What simic kept from the seed design

| esper-lite | simic (HLD) | simic (running code) |
|---|---|---|
| **Kasmina**: slots, blueprint seeds, gradient isolation, alpha blend, lifecycle | **Wrenn** embodies growth reversibly. Growth germinates at zero influence, matures behind the host ("nursery mode", the default), blends, then commits. The unit is still a residual insert `h' = h + α·B(h)` at a region (`06-growth-model.md`). | `kernel_demo.Slot` with stages DORMANT → GERMINATED → TRAINING → BLENDING → FOSSILIZING → FOSSILIZED, STE germination, blend and beta ramps, and a 4-seed menu (`norm`, `attn`, `conv_light`, `conv_heavy`). Lifecycle v2 is a curvature clamp to stop the graft diverging at germination. |
| **Blueprints** (a human-authored menu) | **Momir** *generates* the growth from live host state. **Elesh** canonicalises it and **Urabrask** compiles it. | Not built. The code still uses the fixed seed menu. |
| **Tamiyo** (PPO controller deciding germinate/prune/fossilize) | Split up. **Aurelia** commissions (whether and where to grow) inside **Ugin**'s budget. **Isperia** admits or rejects against a mandatory no-op. The name "Tamiyo" now means a **witness only**: it powers the dashboards and flight recorder and must not change training (INV-35). | Fixed schedule, no controller. |
| **Emrakul** (planned decay policy, never built) | Kept and specified: periodic tenancy review → RETAIN / RETEST / SEDATE / DECAY / LYSE, acting only with an Isperia warrant. Sedation comes before lysis. | Not built. |

**Bottom line:** simic changed *who decides* and *how a growth is proven*. It kept the seed itself: a discrete module inserted at a region, then germinated, matured, blended, committed and later lysed. Seed and reap are still the spine. Kasmina became Wrenn, and the Tamiyo controller was replaced by Aurelia plus Isperia.

## What Sultai would replace (if it drops discrete seeds)

> **Correction (2026-10-09, after John's OpenAI report):** the intended Sultai keeps a module at a site. It replaces gradient seed incubation with gradient-free generative formation, and replaces Emrakul's reaping with a presence head plus compile-away gates at formation time. See [openai-report-vs-repos.md](openai-report-vs-repos.md). The list below is the broader case.

All of these exist because growth is a discrete object with a lifecycle:

- **Wrenn's lifecycle**: slots/regions, STE germination, alpha and beta ramps, fossilize/commit, and the whole lifecycle-v2 stability work (`docs/bounded-lifecycle-v2.md`).
- **Momir → Elesh → Urabrask**: generate a growth graph, canonicalise it, compile it. If there is no graph object to admit, there is nothing for this pipeline to carry.
- **Aurelia's "whether and where to grow"** and **Emrakul's lysis of committed structure**. These exist as decisions about discrete units, so they change form. Ablation could take over Emrakul's role as the mechanism that judges what still earns its keep. That's my reading of "simic + ablation"; the repo doesn't say it.
- `kernel_demo`'s seed menu and the `Slot`/`Stage` code.

## What carries over to Sultai unchanged

The parts that don't care what the unit of change is:

- **Tolaria's matched branches**: snapshot the host, run candidate, control and **mandatory no-op** branches on identical future data, deterministically. This is the infrastructure an ablation approach needs most.
- **Jin-Gitaxias / Isperia**: evidence and judgement kept apart, blinded to provenance, with doing nothing as a real competitor (INV-15/16/17/18).
- **Leyline** contracts, **Urborg** append-only history, **Ugin** budgets, and **Nissa** observing from the *ablated* path, so the deficit is read without help from existing growth.
- **Tamiyo as a witness.** This is the esper-lite telemetry lesson, written down as rules: observability cannot change training (INV-35), and missing UI data fails visibly instead of defaulting.
- **The bounded ladder and harness**: pre-registered studies, paired 48-seed screens, `verify_run` source-drift refusal, and fit/dev/outer separation. It also comes with a ready comparator. Whatever replaces the seed has to beat **static added capacity**, which the graft currently does not.

## The one methodological point to settle early

simic's own claim chapter (`01-claim.md:132`) puts the confound squarely:

> Removing that component from the same host measures both the component's intrinsic value and the host's acquired dependence. Reliable admission and retention therefore require matched no-intervention branches, not same-host ablation alone.

esper-lite's `counterfactual.py` docstring says the same. If ablation is central to Sultai, the design needs to say which question each ablation answers, intrinsic value or acquired dependence, and whether it runs as same-host removal or against a matched branch.

## Not settled by either repo

- Neither repo has a diffusion design or diffusion code.
- What replaces the seed as the unit of change is open. Options include continuous capacity everywhere, per-unit or per-channel gates, and structure that emerges through training rather than being inserted.
