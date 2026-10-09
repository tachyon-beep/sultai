# John's OpenAI report vs the project's research report, esper-lite and simic

Inputs:
- **OAI**: John's upload of 2026-10-09 at `/mnt/project-files/uploads/hearth/29f23cee-10a9-4b2a-ae47-52624c81c33b`. It is OpenAI deep research, "Simic: Local Generative Overgrowth and Diffusion-Grown Neural Modules", lines 1–1009, followed by John's follow-up question and its answer, lines 1011–1313.
- **CR**: the project's research report, `/mnt/project-files/research/simic-diffusion-grown-modules.md`.
- **Repos**: esper-lite @ `2d296b6`, simic @ `16bcf70`. See `esper-lite-review.md` and `simic-seed-comparison.md`.

Both reports answer the same brief with the same skeleton: literature table, three mechanisms, scaffolding section, three rungs.

## Where they agree

Both reports frame the idea as an **amortised local repair prior**, and both say diffusion is on probation. They agree on the plan:
- rung 1 is fixed-shape and oracle-site on CIFAR-10;
- training includes null cases;
- a deterministic hypernetwork is a required control;
- data is split by host lineage or run.

They also agree on the later stages: structure comes at rung 2, learned site selection at rung 3, and "half the parameters" is judged against strong compact models on a Pareto curve. Both rate "half" as speculative.

## Where OAI goes further than CR

1. **The seed has its own evidence channel and an investigation loop** (the follow-up). In the follow-up, Tamiyo only says "investigate here". The seed reads *paired, example-level* tuples (activation, prediction, label, loss) and runs a bounded set of forward-only probes on a frozen snapshot before proposing repairs. CR's seed sees one aggregate "site signature" and is generated in one shot. OAI adds active system identification; CR has nothing like it.
2. **An experiment on the information contract.** OAI holds Tamiyo's instruction fixed and varies what the seed knows: coarse summaries, then paired passive telemetry, then telemetry plus probes. It adds a probe-shuffling control. This directly tests John's division of labour. CR does not vary the seed's evidence.
3. **A status vocabulary for failure.** OAI separates three cases: the information is absent at the site, the site can't influence the outcome, or the module can't express the fix. Each maps to a different message back to Tamiyo, and "no repair found" is kept distinct from "no repair exists".
4. **A non-diffusion alternative.** OAI points to a learned developmental rule (MetaLearnNCA, a 6 Oct 2026 preprint) that updates the seed's program from telemetry without gradients.

## Where CR goes further than OAI

- **A memorisation control.** CR includes nearest-neighbour retrieval of the closest training repair. OAI has no equivalent.
- **A gradient-initialisation control.** CR adds GradMax-style SVD gradient initialisation as a control.
- **Host-basis coordinates.** CR generates in PCA coordinates of the site. OAI avoids the problem instead, by branching all hosts from a common anchor.
- **Null and compactness built into the generated object.** CR uses rank-ordered gains, so an all-zero sample means "insert nothing".
- **Finer held-out splits.** CR holds out width, block position and null type separately.

## Where they conflict, with each other or with the repos

1. **Gradient-equivalent features.**
   - OAI says activation × error statistics should be *excluded* from the strict no-backprop version, because they are a derivative computed another way.
   - CR's default signature includes a ridge probe from site activations to `(onehot − softmax)`. That is that kind of feature. CR does drop it in its "P1-strict" variant.
   - So the two disagree on which version is the headline. I'd make the strict version the headline and report the probe-assisted one separately.
2. **How "nothing" is represented.** OAI uses a separate presence head. CR uses the empty set as a sampled outcome. Rung 1 could run both, since they share a corpus.
3. **The module.** OAI uses a 272-parameter full-rank 1×1 conv, which has no hidden layer and therefore no permutation symmetry. CR uses a rank-32 GELU bottleneck, which has symmetry the canonicalisation has to handle. OAI's choice makes the first result easier to read. CR's choice makes compactness measurable at rung 1.
4. **Neither report has seen simic.** Both inspected esper-lite only, and CR proposes building rung 1 on esper-lite's `cifar_baseline`. simic already has the better substrate:
   - a bounded CIFAR harness with synthetic host pathologies (`under_normalized`, `channel_starved`, `no_spatial_mix`, `mild`);
   - paired multi-seed screens, pre-registration, and source-drift-refusing `verify_run`;
   - a measured resolution of ±0.018 nats.
   
   Rung 1 should be built there.
5. **Static capacity is missing from rung 1 in both reports.** simic's ladder found that its scheduled graft loses to plain static added capacity, capturing 31–60% of static's gain. Neither report's rung-1 controls include that comparator; both defer strong static baselines to rung 3. I'd add it at rung 1, because it is the bar the current approach already fails.
6. **What "Tamiyo" means.**
   - Both reports use Tamiyo as the controller, which is esper-lite's meaning.
   - In simic's HLD, Tamiyo is a witness that must not steer. The commissioning role OAI describes is simic's **Aurelia**.
   - The follow-up's routing is also already simic's rule. Aurelia sends only a brief, and Nissa publishes evidence directly to Momir (INV-07/09).
   - So Sultai is close to *simic with Momir implemented as a learned generator plus an investigation loop, forming repairs without gradients*.
7. **Telemetry risk moves.** In OAI's design, the seed's telemetry is the *input to a learned model*, not a dashboard. Bad or lossy telemetry would not mislead a person, as it did in esper-lite. It would quietly weaken the generator, and the result would read as "the idea fails". The shuffled-condition and shuffled-probe controls are the guard and belong in rung 1. simic's "observability is inert" rule does not cover this case.

## On "simic + ablation"

Neither report uses ablation as a mechanism. In both, ablations are *controls*: shuffled or coarsened conditioning, a frozen host during scaffolding, and retraining the final architecture from scratch. If John means something else by it, neither the reports nor the repos define it.
