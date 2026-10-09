# Sultai TCD revision 2: Emrakul owns the site, Tamiyo forecasts opportunity

*9 October 2026. This amends [the design report](reports/Sultai%20concept%20demonstrator%20design.md) after John's direction in the project thread. Where the two disagree, this file wins.*

## What John decided

- The generator is called **Emrakul**. It is fully responsible for the best outcome at the site it lands on. That includes "no-op, because nothing was wrong here", and it is responsible for site safety.
- Enhancement can be **additive or reductive**.
- **Tamiyo** is trained to forecast the opportunity for enhancement at a site, using data from running Emrakul across many intervention scenarios.
- The intent is to put most of the engineering difficulty in Emrakul, which keeps everything else simple.

## Sultai's role table

This replaces the simic role mapping in the report's section "Simic's instrument fits; its runner, host and seed lifecycle do not". Sultai deliberately reuses two simic names with new meanings. In simic, Emrakul only acts after commit, to decay and lyse, and Tamiyo only observes. Sultai documents should state this once and then use its own meanings.

| Role | Sultai | Inputs | Output | Trained how |
|---|---|---|---|---|
| Where to look | **Tamiyo**, opportunity forecaster | Coarse per-site and host-level telemetry only | Forecast utility of sending Emrakul to each site, given a fixed Emrakul version | Supervised regression on Emrakul's realised outcomes |
| What to do there | **Emrakul**, site owner | Site, budget, rich local telemetry and probe responses | One edit (add, replace, reduce) or no-op, plus a status | Offline on the oracle repair corpus; formation itself takes no task gradients |
| Did it actually help | **Independent audit** (the harness's matched no-op branches) | Emrakul's edit and the untouched twin | Measured gain and harm | Not trained; it is measurement |
| Record | Run record and dashboards | Everything | Nothing that affects training | n/a |

Emrakul's status vocabulary comes from John's follow-up report: *nothing wrong here*, *information not available at this site*, *this site cannot influence the outcome*, *the permitted edit cannot express the fix*, and *no repair found within budget*. The last is distinct from "no repair exists". Tamiyo receives the status and the utility, never the edit.

**The audit is not a second decision-maker.** Emrakul owns admission at its site. The audit exists because a component that is the only witness to its own safety cannot reveal a harm it fails to see. In the TCD, every Emrakul decision is paired with a no-op twin that Emrakul does not control, and the safety claim is judged on those pairs. If Sultai later needs a runtime check, the audit's measurements are what it would use.

## Additive and reductive edits

The report's module could only add: a 272-parameter residual at an interface with no host operator of its own. Emrakul needs an action space in which a reduction is possible and a no-op means "keep what is there".

**Emrakul edits a site operator.** A site is still a (read tensor, write tensor) pair, now holding the host's existing operator W_host there (the identity where there is none). Emrakul emits a replacement W′ and stores it at a retained rank r′:

- **No-op:** W′ = W_host; parameter change 0.
- **Additive:** W′ = W_host + ΔW, where the host operator is the identity or ΔW raises the effective rank. This is the report's 272-parameter residual when W_host is the identity.
- **Reductive:** W′ is a lower-rank replacement for W_host, stored as factors. The parameter change is negative.

Utility at a site is U = L(no-op) − L(edit) − λ·ΔP, with λ fixed in the pre-registration. The quality-against-parameters frontier is reported as well, so that the choice of λ cannot carry the result. Because no-op is always legal, Emrakul's utility is never negative if its safety contract holds.

**The TCD gets a second site type.**

| Site type | Host operator | Edits allowed | Parameters at stake |
|---|---|---|---|
| S-add | Identity at the 16-channel interface z | No-op, additive | 0 to 272 |
| S-red | A downstream 64→64 1×1 convolution | No-op, additive residual, low-rank replacement | 4,160 down to 128·r′ + 64 |

**Reductive pathologies with ground truth.**

| ID | Pathology | Ground truth |
|---|---|---|
| P11 | Over-provisioned operator: the host trains with S-red full rank, but the operator's useful rank is low | Measured by the truncation-loss curve on the frozen host |
| P12 | Harmful structure: inject a post-hoc noise-carrying component into S-red's output | Exact: removing the injected component restores intact loss |
| P13 | Load-bearing operator: the full rank is genuinely used | Reduction harms; the correct answer is no-op |

**Reductive baselines Emrakul must beat.** A reduction can be computed directly by SVD truncation of the host weights, so Emrakul earns nothing unless it does better than that. The baselines, all at matched ΔP and with the same selection set, are:
- plain SVD truncation at the best rank;
- activation-weighted SVD (weights the truncation by input activation statistics; label-free, so tier T0);
- truncation with the rank chosen by forward evaluation on the selection set.

## Site safety, made measurable

"Responsible for site safety" becomes a contract with two numbers, fixed in the pre-registration:

1. **Aggregate harm.** Among sites where Emrakul acts, P(loss on held-out examples exceeds the no-op twin by more than the noise floor) ≤ δ, with δ = 0.1 at the TCD's calibration size.
2. **Class-level harm.** No class's development accuracy drops by more than a pre-registered amount relative to the twin. Without this, a mean-loss gain could hide a regression on one class.

Emrakul is calibrated to this contract with Learn-then-Test on development lineages, as the report already specifies for the null. The audit then measures realised harm on the test lineages. In the live experiment it also measures harm after the host keeps training, because an edit that was safe when inserted can stop being safe once the host adapts around it.

## A Tamiyo rung that needs no live growth

Tamiyo's training data is Emrakul's outcome log, so Tamiyo can be trained as soon as Emrakul exists. It no longer waits for the report's rung 3. This makes John's point concrete: once Emrakul is frozen, a label costs one Emrakul deployment (forward passes plus selection), not an oracle fit.

**Host change.** Each host needs several sites. The TCD host exposes eight: the S-add interface, the S-red convolution, and six further 1×1 operators across depth, some impaired and some intact. Emrakul is run on **every** site of every test host, rather than on sites someone chose, so Tamiyo's labels are not biased by where scenarios happened to be run.

**What Tamiyo sees.** Only coarse telemetry: per-site loss attribution proxies, activation spectra and saturation, and host-level loss and training stage. It never sees Emrakul's example-level tuples or probe responses. That keeps John's split honest: Tamiyo forecasts *whether* there is an opportunity, and Emrakul works out *what* it is.

**Labels.** For each (host, site): Emrakul's realised U from the audit, its status, and the Emrakul version. Labels from different Emrakul versions are never mixed. When Emrakul changes, Tamiyo's data is collected again.

**Who a no-op counts against (John, 9 October).** The system is safe when Emrakul returns a no-op, but Tamiyo has still failed: she flagged an opportunity that was not there. Her training signal therefore depends on Emrakul's status, not only on U.

| Emrakul's status | Counts against | Tamiyo's label |
|---|---|---|
| Nothing wrong here | Tamiyo | Severe false alarm |
| Information not available here, or this site cannot influence the outcome | Tamiyo (wrong site; a real problem may exist elsewhere) | False alarm, with the problem possibly real |
| The permitted edit cannot express the fix | Neither; a budget or vocabulary limit | Opportunity real but not reachable; excluded from the false-alarm loss |
| No repair found within budget | Emrakul | Excluded from Tamiyo's loss |
| Edit made, U > 0 | Tamiyo credited | U |

Three safeguards come with this split:
- **Status accuracy.** Tamiyo is punished on Emrakul's word, so the TCD checks Emrakul's statuses against the oracle ground truth. If Emrakul says "nothing wrong here" where an oracle repair exists, Tamiyo would learn to avoid sites Emrakul finds hard. The status confusion matrix is a reported metric, and rung 1c does not run until "nothing wrong here" is precise enough. The threshold is fixed in the pre-registration.
- **Forced choice.** A heavy false-alarm penalty pushes a forecaster to flag nothing. Tamiyo is therefore trained and evaluated as a ranker with a fixed budget of k sites per host per round. She cannot abstain her way to a good score.
- **Misses as well as false alarms.** Emrakul runs on every site, so sites Tamiyo ranked low but where Emrakul would have gained are visible. They are scored as misses.

**Tamiyo gates (rung 1c), on held-out lineages.**

| Gate | Rule | If it fails |
|---|---|---|
| T1 forecast | Rank correlation between forecast and realised U ≥ 0.5, lower bound > 0.3 | Tamiyo's coarse telemetry cannot see opportunity; give it more or stop the split |
| T2 selection | Sending Emrakul to Tamiyo's top-k sites recovers ≥ 0.7 of the gain from running Emrakul on all sites, beating random and a saturation heuristic | Same |
| T3 worth having | T2 achieved at ≤ 1/3 of the cost of running Emrakul everywhere | At this scale it is cheaper to run Emrakul on every site, and Tamiyo is not yet needed |

Gate T3 is the honest test of Tamiyo's role. If Emrakul is cheap enough to run everywhere, Tamiyo adds nothing at toy scale. That would not count against Sultai, because the cost argument only applies at larger scale. Its outcome should still be recorded rather than assumed.

## What this does to the plan

**Order.**
- 1a: Emrakul, additive, frozen host.
- 1a-red: Emrakul, reductive, frozen host.
- 1b: live insertion against static capacity.
- 1c: Tamiyo.

1a and 1a-red share the corpus pipeline. 1c reuses their outcome logs and adds the exhaustive multi-site runs.

**Gates added to the report's readings table.**
- **G6 reductive:** Emrakul's reductive edits beat activation-weighted SVD at matched ΔP on P11 and P12 (lower bound > 0), and return no-op on at least 0.7 of P13 cases.
- **G5** now covers both harm numbers above.
- **STOP, reductive line:** if activation-weighted SVD matches Emrakul, reduction is a solved data-free operation and Emrakul's value is additive only.

**Generator family.** John calls Emrakul the diffusion model. The design keeps the report's ladder: retrieval, then a deterministic hypernetwork, then a mixture, then flow matching, then diffusion. "Emrakul" names the role; whichever rung of the ladder passes is its implementation. If a deterministic hypernetwork does as well as diffusion, Emrakul is a hypernetwork. The report's "drop stochastic generation" pivot still applies.

**Compute.** Adding the S-red corpus, eight sites per host and exhaustive Tamiyo labelling adds about 10–20 GPU-hours to the report's 25–45. The total is roughly 35–65 GPU-hours plus the pilot. This is an estimate until the pilot's timing check runs.

**Build plan additions.**
- `sultai_host.py` gains the multi-site host and pathologies P11–P13.
- `sultai_corpus.py` gains reductive oracle fits and the truncation-loss curves.
- A new `sultai_tamiyo.py` holds the forecaster, the exhaustive labelling run and the T1–T3 analysis.
- The ADR in step 0 records Sultai's role table, including the change of meaning for Emrakul and Tamiyo relative to simic.

## Unchanged

The rest of the report stands:
- the telemetry tiers and the gradient audit;
- the 16-channel host;
- the lineage splits;
- the 96-unit live test against static capacity;
- the cost ledgers;
- the reading of "ablation" as subtraction at formation, with retention measured against matched no-op branches.

Emrakul's reductive edits make that last reading more literal: subtraction is now something Emrakul can do to the host, not only to its own output.

## The host starts in crisis, and Tamiyo's job is sequential (John, 9 October)

John's intent is that the host begins very sparse, so it is in crisis almost immediately, and Tamiyo slowly teases it towards the best possible outcome. This changes three things.

**Tamiyo is a sequence of decisions, not one ranking.** Early on almost every site has an opportunity, so the false-alarm penalty rarely fires. The hard problem is order and pacing under a budget: which site first, how much to grow before compacting, and when to stop. The TCD still starts Tamiyo as the cheapest thing that could work, and adds lookahead only if it loses:
- **Greedy:** each round, send Emrakul to the sites with the highest forecast utility.
- **Baselines Tamiyo must beat:** a fixed schedule (intervene at every site at fixed epochs), round-robin, and random site order at the same budget. Esper-lite's controller never beat a fixed schedule, so this comparison comes first.
- **Reinforcement learning or planning** only if greedy loses to the fixed schedule, or if a measured case shows that order changes the final outcome.

**Emrakul's training data must include crisis states.** The rung 1a corpus uses trained, impaired hosts on a frozen snapshot. A starved host early in training looks different: activations are unsettled, and the right edit can change epoch to epoch. The corpus therefore adds snapshots taken from starved hosts at several early epochs. Emrakul is evaluated separately on those, because "works on a converged host" does not imply "works during crisis".

**Growth then compaction is the expected path.** In crisis most useful edits are additive. Reductive edits become useful once the host has more capacity than it needs at some sites. The TCD logs the add/reduce mix over the run as a descriptive result, and does not impose it.

**What "very sparse" means.** This design takes it as a host with very small width (few channels at every site), because that gives Emrakul whole operators to add. If John means sparse connectivity at normal width (most weights zero), the reductive and additive edits become mask changes rather than operator changes, and the site definition needs revisiting.

**Plan consequence.** The starved-host family (P10) moves from a single live-test host to the main object of study. A new rung 1d runs the full loop on it: start starved, then Tamiyo and Emrakul for R rounds under a parameter budget. Its comparators are the fixed schedule, static capacity at the final size from step zero, and a from-scratch width sweep at the final parameter count. This is the smallest version of the half-parameters experiment, and it runs only after 1a, 1b and 1c pass.

## Coordinated interventions across sites (John, 9 October)

John expects the smart work to show up here. Some fixes need two coordinated edits, one early and one late, as with simic's norm and attention seeds. Recognising "we need dual interventions here and here" is Tamiyo's job.

**Why greedy single-site Tamiyo cannot see this.** The value of a pair can be superadditive: neither edit helps on its own, or one even harms, while the two together help. Define the interaction for sites A and B as I(A,B) = U(A+B) − U(A) − U(B). A forecaster that scores sites one at a time sees U(A) and U(B) near zero and never proposes the pair.

**Design changes.**
- **Tamiyo's actions include site sets.** She forecasts U for single sites and for pairs. Pairs are limited to a declared candidate list (for example, sites that read and write along the same path) so the action space does not explode.
- **Emrakul deploys a pair jointly.** One call covers both sites, and the second edit is conditioned on the first. The safety contract applies to the pair as a unit, because half of a correct pair can be unsafe alone (one half of an attention-and-norm fix can destabilise the host). Emrakul's statuses gain one entry: *needs a partner edit at another site*. That turns a single-site failure into a hint for Tamiyo, rather than a false alarm.
- **Ground truth for interaction is measured exhaustively.** On the eight-site TCD host there are 28 pairs. Emrakul is cheap once trained, so every single site and every pair is measured on test hosts. That shows where superadditive pairs actually exist, and Tamiyo is scored against it.
- **Constructed pair cases.** P14 is a two-site defect where the fix at the early site moves activations out of the range the later layers expect, so it only pays once the late site is also corrected. The exact construction is settled in the pilot, with I(A,B) > 0 verified by oracle fits before the case is used. Literal attention and normalisation edits need Emrakul's operator vocabulary (rung 2). At rung 1 the pairs use the linear operators Emrakul already has.

**Gate (rung 1d).** On hosts with measured positive interaction, Tamiyo with pair actions beats greedy single-site Tamiyo at the same intervention budget, with lower bound > 0. On hosts where interaction is near zero, she proposes pairs no more often than a pre-registered rate. Passing the first half without the second would mean she proposes pairs indiscriminately and wastes budget.

**Natural test.** simic's `under_normalized` host already has a measured deficit that the `norm` seed repairs. Once Emrakul has a typed vocabulary, that host is where the attention-and-norm pairing John describes can be tested against simic's existing results.

**Two edits or one spanning module (open, John, 9 October).** A pair could instead be served by one bespoke module that reads at the early site and writes at the late one. The (read tensor, write tensor) site definition already allows this, because a spanning module is just a site whose read and write points are far apart, like the P6 bypass. Which answer is better is not known, so the design does not choose it. When Tamiyo nominates a pair, Emrakul may return either form, since it owns the outcome. On the pair cases the TCD runs both forms as arms at matched parameters: two coordinated edits, and one spanning module. The result is logged as a descriptive finding for rung 1d. A spanning module that wins consistently would mean Tamiyo's pair action is really "nominate a read point and a write point", and Emrakul decides the shape.

## Emrakul sizes its own deployment (John, 9 October)

John wants Emrakul to recognise a big, rich area and deploy something larger along one dimension or another, either on its own initiative or steered by Tamiyo. That makes **size a generated quantity**. The report's fixed 272-parameter module cannot express this: on a 16-channel interface it tops out at rank 16 and can only shrink.

**Change to the module.** The site's generated object becomes an overcomplete set of up to R components. Each component reads one direction, applies a nonlinearity, writes one direction, and carries a gain; this is the earlier project report's rank-ordered form. R is allowed to exceed the interface width (R = 64 on the 16-channel site, a 4× expansion). Emrakul's size choice is the number of components with non-negligible gain. Zero components is the no-op. The cost is the permutation symmetry the 272-parameter module avoided, so the generator becomes a set model, as the earlier report specified.

**Keep both module families.** The 272-parameter linear module stays as the first, fastest falsification test in rung 1a. The sized module runs on the same corpus as a second family. If the linear module already fails G1–G3, the sized family is not run.

**Dimensions of "big".** At rung 1 only width (component count) varies. Depth (stacked blocks), spatial extent (kernel size) and span (how far apart the read and write points are) join at rung 2, together with the typed operator vocabulary.

**Ground truth for size.** Every corpus case gets oracle fits at several widths (R ∈ {2, 4, 8, 16, 32, 64}). That gives a gain-against-size curve, and the target size is the smallest width within a pre-registered tolerance of the curve's plateau. The report's *capacity-limited* label is the case where that knee sits above what the site currently allows. Emrakul is scored on how close its chosen size is to the knee, and on the gain it achieves per parameter.

**Who decides size.** By default Emrakul does. Tamiyo's lever is a size cap, the budget she grants with the nomination. Her forecast already predicts how large the opportunity is, so a rich area naturally gets a larger cap. Giving Tamiyo direct control of shape is held back, and comes in only if Emrakul's size choices turn out poor.

**Test of "rich area, go big".** Across corpus cases, Emrakul's chosen size should track the oracle knee: rank correlation ≥ 0.5, with a pre-registered lower bound. A generator that always deploys the maximum and lets truncation sort it out would fail this test. So would one that always deploys small and leaves gain unclaimed.

## Scope for the first build (John, 9 October)

John warned against adding things now because they would be hard to engineer later, when doing so breaks the engineering now. Applying that to this revision, including my own earlier "do from day one" list, the first build is cut back to rung 1a as the report specified it:

- **One host, one site (S-add), the 272-parameter linear module, settled impaired hosts, and a binary outcome (repair or no-op).**
- **Recorded, not used:** every corpus case stores its pathology ID and construction parameters. Statuses, size knees and reductive labels can all be derived from that later without rebuilding the corpus. This costs nothing now.

**Deferred**, each to the rung that first needs it:

| Item | Needed at |
|---|---|
| Full status vocabulary | 1c (Tamiyo) |
| Reductive edits, S-red site, P11–P13 | 1a-red |
| Multi-site host | 1c |
| Crisis-state snapshots | 1d |
| Pairs and spanning modules | 1d |
| Sized deployment (overcomplete components, width sweep) | After 1a passes; structure rung for the other dimensions |

The host code should keep site definitions as data, a list of (read, write) pairs, so that adding sites later is a configuration change rather than a rewrite. That is the only accommodation made now.

**Scaffolds are disposable (John, 9 October).** If generating a simple example takes under two hours, hosts, corpora and harness code are rebuilt for each rung rather than designed for reuse. The estimate puts one host plus its oracle fits and telemetry at minutes on the local cards, and the pilot's timing check confirms it. The two accommodations above stay only because they cost nothing. What must not be disposable is reproducibility: each scaffold is generated from committed code and `derive()` seeds, so any result can be regenerated exactly.
