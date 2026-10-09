# Make a generated repair beat cheap gradients

> **Revision 2 (9 October 2026):** John has set Sultai's roles. Emrakul is the generator and owns each site, including no-op and site safety. Edits can be additive or reductive. Tamiyo forecasts opportunity, trained on Emrakul's outcomes. See [emrakul-tamiyo-revision.md](../emrakul-tamiyo-revision.md), which overrides this report where they differ, in particular the simic role mapping below.

*Design report for John, 9 October 2026. Inputs: the five research notes in `research_notes/Sultai concept demonstrator design/`, the project's earlier report on diffusion-grown modules, the three repository notes (esper-lite, simic, and the comparison of the two prior reports), John's OpenAI deep-research report with its follow-up, and a read-only look at the simic checkout (`16bcf70`).*

The technical concept demonstrator (TCD) for Sultai should be **two linked experiments on one purpose-built 16-channel CIFAR-10 host, built inside simic's experiment conventions**. The first (rung 1a) asks whether a small learned generator, given only local telemetry from a frozen host and no task gradient, can emit a 272-parameter residual repair, or decide to emit nothing, for host lineages it has never seen, and do so better than nearest-neighbour retrieval, shuffled conditioning, a one-shot gradient initialisation and a few steps of ordinary SGD at matched cost. The second (rung 1b) asks the question simic's own ladder has already answered badly for its scheduled graft: inserted into a live, deliberately starved host and then trained, does the generated module beat no growth and come within 0.025 nats of **static added capacity present from step zero**, the bar the current graft fails (it captures only 31–60% of static's gain)? The demonstrator settles the four open conflicts as follows. **Activation-times-error statistics are gradient information** (they are exactly the gradient of a linear readout from the site, and the module gradient once the downstream Jacobian is known), so telemetry is graded into tiers. The phrase "without task gradients" is reserved for the label-free tiers, and every tier is audited for how much gradient it leaks. The harness is **simic's**, not esper-lite's. **Static capacity enters at rung 1**. The controller is **Aurelia** issuing a fixed "investigate this site" brief, judgement against the mandatory no-op belongs to **Isperia**, and **Tamiyo only observes**. "Ablation" is read as subtraction: removal happens at formation (rank truncation down to an empty module), and value is proven by a pre-registered retention-ablation contrast against matched no-op branches. The whole TCD costs roughly 25–45 GPU-hours on the local pair of RTX 4060 Ti cards plus a pilot. It ends in a pre-registered go, pivot or stop reading that tells John whether the half-parameters programme is worth climbing, and in what form.

**Evidence tags.** Literature claims keep the tags their source notes used: **[P]** or **[fetched]** means the primary source was read directly; **[S]**, **[F]** or **[snippet]** means it was confirmed only at abstract or search-extract level; **[B]** or **[background]** means background knowledge not verified in this round; **[D]**, **[C]** or **[inference]** marks the note author's own derivation. Untagged statements are this report's design reasoning. Repository facts cite file paths in the simic checkout.

## The design at a glance

| Decision | Choice for the TCD | Why |
|---|---|---|
| Meaning of "ablation" | Subtraction at formation (truncate to rank r, including r = 0) plus a retention-ablation contrast against matched no-op branches as the evidence | The only reading that yields a falsifiable claim; knockout deltas on the grown host are confounded by acquired dependence |
| Hidden gradients | Yes, paired activation-times-error statistics are gradient information; graded tiers T0, T0+P, T1, T1+P with a gradient-recoverability audit | Settles the conflict between the two prior reports by measurement, not definition |
| Harness | simic's data split, RNG derivation, common-future pairing, paired statistics, pre-registration and run verification; a new runner, host and telemetry | simic's instrument is proven to ±0.018 nats; its runner and host do not fit a frozen-host, 16-channel study |
| Controller naming | Aurelia commissions (fixed brief), Nissa publishes telemetry directly to the generator, Momir generates, Isperia admits against no-op, Tamiyo witnesses | Matches simic's Namespec 2.0 and INV-07, INV-09, INV-15, INV-16 and INV-35 |
| Host | Small CNN with a 16-channel post-ReLU interface; about 0.28M parameters intact, about 0.07M starved | Exact ground truth for surgical impairments; a starved variant for the live test |
| Site | A (read tensor, write tensor) pair | A bottleneck upstream of the read point is locally unrepairable; a site that straddles it is repairable |
| Module | 16→16 residual 1×1 convolution with bias, 272 parameters, compacted by SVD truncation | No hidden-unit symmetry; rank is a natural compaction axis with known ground truth |
| Generator | Ladder from retrieval to deterministic hypernetwork to K-head mixture to conditional flow matching (≤100k parameters); diffusion only if flow matching fails diagnosably | No evidence that diffusion is needed at 272 dimensions; memorisation risk is high at this corpus size |
| Null | Presence score calibrated by Learn-then-Test on held-out lineages; abstentions analysed intention-to-treat | Simic's mandatory no-op and complete-history rules |
| Static baseline | At rung 1b as a from-scratch arm (also the "final architecture retrained from scratch" control) | It is the bar simic's graft already fails |
| Seeds | Rung 1a: 56 anchor lineages (24 train, 8 development, 24 test). Rung 1b: 96 paired units | Recomputed from the paired SD; detail in Appendix D |
| Compute | About 25–45 GPU-hours plus a 3-hour pilot, on the two local GPUs | Corpus building and oracle fits dominate |

## Ablation means subtraction at formation and contrast as evidence

John's shorthand "simic + ablation ⇒ sultai" admits three readings. In the first, **removal at generation time**, the generator over-provides and then subtracts: near-zero components are cut and "insert nothing" is a legal output. In the second, **host knockout as diagnosis**, parts of the host are knocked out to find what is missing. In the third, **retention ablation**, the grown module is removed at the end, to test whether it earned its place. A fourth meaning already exists in simic's design: Nissa observes the host on its *ablated* path, with existing growth switched off, so that a deficit is read without help from earlier growth.

This report adopts readings one and three together and keeps reading two only as a narrow telemetry tier. **Ablation is Sultai's selection operator: what the generator emits is what survives subtraction, and what is retained is decided by a measured contrast against never having inserted it.**

A knockout measures a total effect: the component's intrinsic contribution, plus the host's acquired dependence on it, plus compensation elsewhere. The operator (zero, mean, resample, optimal constant) changes the answer too. Mean ablation can make a model look *better* **[S]** ([Redwood](https://alignmentforum.org/posts/kcZZAsEjwrbczxN2i/causal-scrubbing-appendix)), and in Chinchilla-7B another layer compensates for an ablated one, restoring about 70% of the logit drop **[S]** ([McGrath et al.](https://arxiv.org/pdf/2307.15771)). Simic's claim chapter draws the same conclusion: same-host removal mixes intrinsic value with acquired dependence, so admission and retention need matched no-intervention branches (`docs/design/01-claim.md:132`).

Retention ablation is therefore measured three ways at the end of each live run. **Intrinsic value** is L(no-op branch) minus L(grown, module intact). **Acquired dependence** is L(grown, module zero-ablated) minus L(no-op branch). The **raw knockout delta** is their sum and is never used as evidence. Zero ablation of the residual branch is the pre-registered "never inserted" operator. Resample ablation is the distribution-preserving check, and optimal-constant ablation **[S]** ([Li & Janson](https://arxiv.org/pdf/2409.09951)) catches a module that merely learned a bias shift.

Removal at formation has two consequences. Every "insert nothing" case is logged and analysed intention-to-treat, because dropping those cases would select on generator confidence, the winners-only bias INV-31 forbids. And compaction must beat a **random cut at the same rank**, because pruning-at-initialisation methods were found mostly to choose per-layer ratios rather than specific weights **[B]** ([Frankle et al. 2021](https://arxiv.org/abs/2009.08576)). Host knockout survives only as the T0+P tier below: forward perturbations of the site on a frozen snapshot copy, observing changes in logits but never in loss, because a loss-observing knockout is a gradient measurement by another route.

This is an interpretation, and John should confirm it before the plan freezes. If he instead meant reading two as the heart of Sultai (the module is formed from what host knockouts reveal), the tier structure still holds: knockout evidence moves from an optional tier to the primary conditioning signal, and the gradient audit becomes the main result rather than a guard.

## Activation-times-error statistics are gradients, so the tiers name the claim

John's report excludes activation-times-error statistics from the strict experiment as a derivative computed another way; the project's earlier report put exactly such a ridge probe into its default signature. Elementary calculus settles it. Let h be the 16-dimensional site activation, the module be h + Wh + b at zero initialisation, r = y − p the label residual, and J the Jacobian of the frozen downstream with respect to the site. Then the exact module gradient is **∇_W L = −E[Jᵀ r hᵀ]**, summed over spatial positions **[D]**. The cross-moment E[h rᵀ] is exactly the gradient of a hypothetical linear readout from the site, and it becomes the module gradient once J is known or roughly constant. A ridge probe is the same cross-moment preconditioned by the inverse activation covariance. Per-class activation means already supply half of it, since E[h yᵀ] is the class-conditional means weighted by priors **[D]**. Every loss-observing probe in John's proposed menu (channel rescale, channel shift, small cross-channel residual) is a gradient coordinate or projection **[D]**.

The literature uses "forward-only" in the same sense. The forward gradient is an unbiased gradient estimate from one directional derivative **[F]** ([Baydin et al.](https://arxiv.org/pdf/2202.08587)), and perturbing activations, Sultai's probe regime, makes it competitive on CIFAR-10 **[F]** ([Ren et al.](https://arxiv.org/pdf/2210.03310)). PEPITA loses eight points when its injected output error is zeroed **[F]** ([Dellaferrera & Kreiman](https://arxiv.org/pdf/2201.11665)). VSML **[F]** ([Kirsch & Schmidhuber](https://arxiv.org/abs/2012.14905v3)) and MetaLearnNCA, the 6 October 2026 preprint John's report highlights **[F]** ([arXiv 2610.08479](https://arxiv.org/abs/2610.08479)), both feed error forward. None is label-free.

**The answer to the open question is therefore yes: paired activation-times-error telemetry counts as gradient information, though not as a backward pass.** The two prior reports were each right in a different sense. The design consequence is that the claim must be named by the information the generator received, and the TCD tests four nested telemetry tiers on identical host cases.

| Tier | Generator receives | Error credit? | Downstream sensitivity J? | Claim it licenses |
|---|---|---|---|---|
| **T0** | Unlabelled site statistics and unlabelled example tuples (pooled h, p) | No | No | "Formed without task gradients, from the host's internal state" |
| **T0+P** | T0 plus change-in-logits responses to a fixed menu of 64 forward probes on a frozen snapshot copy | No | Yes, label-free | "Formed without task gradients, with forward interventions" |
| **T1** | T0 plus labels: tuples (h, p, y, loss), class means, confusion matrix | Yes | No | "Formed without a backward pass or derivative measurement, from labelled telemetry" |
| **T1+P** | T1 plus the probes | Yes | Yes | "Formed from telemetry sufficient to reconstruct the gradient" |

Loss-observing probes (the notes' Tier 2b) are left out of the TCD: they are zeroth-order gradient descent with a learned step, and the gradient comparator below already covers what a gradient would give. The generator is also **not given downstream host weights**, because those would let it infer J and would break the premise that the module forms from *local* evidence. Adding the next layer's weights is a later ablation.

Three guards turn the "hidden gradient" objection into numbers. A **gradient-recoverability audit** fits ridge and small-MLP decoders from each tier's telemetry to the true zero-initialisation module gradient on training lineages and reports held-out cosine **[D, proposed]**; T1 is the interesting case, because a generator trained across aligned hosts could learn a typical J. A **one-shot gradient comparator** initialises from the SVD of the true gradient on the same support examples, the residual analogue of GradMax **[F]** ([Evci et al.](https://arxiv.org/abs/2201.05125)). And **few-step SGD on the same examples at matched cost** answers the most dangerous failure mode John's report names: a 272-parameter adapter may be fixable in one or two ordinary steps. A T1 generator that only matches the gradient comparator is an amortised gradient step; one that beats it is using information beyond first order, which is the genuinely new claim.

One unit test shows what each tier can see. John's follow-up and the host notes both use a sign-ambiguity example; this report's version makes passive evidence *exactly* identical. Host A has site channel k zeroed post hoc; host B also negates the next convolution's input weights for channel k. Since z_k is always zero, A and B compute the same function and all passive telemetry, labelled or not, is bitwise equal, yet the best repair writing into channel k has opposite signs. T0 and T1 cannot separate them; two ±ε probes on channel k (T0+P) separate them exactly (Appendix A). It is the cleanest test of John's claim that probes resolve what passive telemetry cannot.

The declared probe budget is 64 forward passes (±ε shifts and ±ε rescales of each channel), well below the 544 central differences that full gradient recovery would need. The support set is 1,024 class-balanced examples, above the roughly 500 the note estimates analytically for the covariance and cross-moment **[D]**. Appendix B lists every field per tier.

## Simic's instrument fits; its runner, host and seed lifecycle do not

Neither prior report had read simic, and both proposed esper-lite's CIFAR baseline. The conflict is settled in favour of **simic's harness**. It already has a fixed 45,000 fit / 5,000 development split held on the device, named RNG streams, common-future pairing across arms, a bitwise GPU determinism profile, launches from git-archived snapshots, a `verify_run` that refuses source drift, paired intervals with a bootstrap check and a pre-registration schema. Above all, it has **demonstrated resolution**: ±0.018 nats on the scheduled-minus-no-growth contrast at n = 48 (`docs/results/2026-10-08-bounded-screen-v1.md`). Esper-lite's equivalents carried placeholder baselines for months and telemetry that drove wrong decisions (`notes/esper-lite-review.md`). The harness does not fit everywhere, and the gaps are structural.

| simic component | Fits the TCD? | Why, and what to do |
|---|---|---|
| Data path, fit/dev split, `runs/cifar-fit-only` view | Yes | Import as is |
| `derive()` RNG streams, `CommonFuture`, `enable_class1` determinism, `score()` | Yes | Import; branches from a snapshot reuse the same pairing |
| `make_snapshot`, `interval`, launch refusal rules, prereg schema keys | Yes | Import; extend the plan schema with Sultai's keys in a new module |
| `verify_run`, `expected_spec`, cost validators | Pattern only | They assume three arms (`ARMS`, `bounded_comparison.py:49`); write a Sultai verifier on the same model |
| `bounded_comparison.py` runner | No | Joint training from step zero only, no frozen-host mode, checkpoints marked final-inference-only (`:763-783`) |
| `kernel_demo.Host` | No | One 64-channel site hard-coded (`:584`); no 16-channel interface |
| `Slot`, straight-through germination, `tau_init` birth-scale calibration | No | Sultai replaces the seed lifecycle; `tau_init` would rescale a generated module to rms 0.05 of h and destroy its learned scale (`:738-751`) |
| `kernel_demo` `Snapshot` / `run_arm` / `run_fan` | Pattern only | Right idea (bitwise-verified no-op twin) but tied to the parked 40-epoch config; re-implement the pattern |
| `TelemetryRecord`, `attach_stat_hooks` | No | Per-epoch, partly gradient-based, no per-example capture; forward-only hooks are new |
| Cost fields | Partly | Keep the units (example passes, optimiser parameter-steps, wall time; no FLOPs by policy) and add corpus, meta-training and deployment ledgers |

The sanctioned pattern for this is `experiments/lifecycle_validation.py`, which builds a new study by importing primitives from the bounded modules instead of editing them. Editing `bounded_comparison.py`, `bounded_data.py`, `kernel_demo.py`, `pyproject.toml` or `uv.lock` would break `verify_run` for archived runs, and `docs/product/current-state.md` forbids it.

**Naming is settled by simic's constitution, not by either prior report.** Both reports call the controller Tamiyo, which is esper-lite's meaning and Namespec 1.0's. Under Namespec 2.0 (ADR-0008), Tamiyo is a witness whose disconnection cannot change training (INV-35). John's follow-up described a controller that only flags that intervention is needed, while the module reads its own local telemetry. In simic that is already the law. **Aurelia** commissions work with a `GrowthIntent` that may carry only scope and operational constraints; a diagnosis or mechanism hint is schema-invalid (INV-09). **Nissa** publishes the same observation directly to the designer, never through Aurelia (INV-07). The TCD therefore uses this mapping:

| Role in the TCD | simic authority | Implementation at rung 1 |
|---|---|---|
| "Investigate site s within budget" | Aurelia, inside Ugin's budget | A fixed stub; no learned controller |
| Telemetry at tier T0–T1+P, including probe responses | Nissa | Computed on a frozen snapshot copy, never in the training forward (INV-34) |
| Generator, including abstention and rank truncation | Momir | The generator ladder below |
| Channel alignment, sign and factor conventions | Elesh | Deterministic canonicalisation |
| Physical slicing into low-rank factors | Urabrask | Semantics-preserving; checked against the dense reference |
| Matched no-op and candidate branches, evidence | Jin-Gitaxias in Tolaria | The rung 1b branch runner |
| Admission against the mandatory no-op, harm veto first | Isperia | Calibrated threshold (INV-15, INV-16, INV-45) |
| Blending | Wrenn | Short α ramp, no straight-through incubation |
| Dashboards and run record | Tamiyo | Witness only |

Two consequences follow. First, generation-time removal sits before commitment, so it belongs to Momir and Isperia rather than Emrakul, which may not manage unborn structure (INV-29). Sultai absorbs Emrakul's reaping role for unborn structure and defers post-commit retention to later rungs. Second, the probes expose one real gap in the HLD: only Jin-Gitaxias runs experiments in Tolaria, yet John wants the module to investigate. The TCD keeps the authority lines intact by declaring the probe menu an **interventional observation profile executed by Nissa on a snapshot copy**, budgeted by Ugin (INV-23). Momir still only receives evidence. Whether a learned probe policy ever lets Momir choose its own experiments is a rung 2 design question for an ADR. ADR-0018 already places the bounded line outside the HLD authority vocabulary, so the TCD can follow the same precedent while recording this mapping, which lets it migrate into Phase A later.

## A 16-channel host with exact ground truth for most cases

The host is new code whose job is to give known answers, not to score well. A two-convolution stem (32 channels) feeds a **producer** convolution whose 16-channel post-ReLU output z (16×16 spatial) is the primary site; four downstream convolutions (64, 64, 128, 128) and a linear head follow, about 0.28M parameters, ResNet-20 scale (Appendix A). The **starved variant** halves the downstream widths to about 0.07M parameters and is both the live-test host and the starting point for the long-term target. Training uses simic's SGD recipe for 10 epochs on 40,000 examples. Downstream BatchNorm stays in eval mode whenever a module is fitted or evaluated, because recalibrated BN would silently "repair" channel attenuation, the same confound as simic's `under_normalized` pathology **[inference]**.

**A site is a (read tensor, write tensor) pair**, not a point. This is the correction carried over from the host notes. A narrow bottleneck trained into the host upstream of the read point is locally unrepairable: by the data-processing inequality, no function of the site tensor can carry more label information than the tensors it is computed from **[background]**. The same bottleneck becomes repairable if the site reads the wide tensor *before* the bottleneck and writes residually *after* it, which makes the module a bypass. The bottleneck family below exploits this. The two site definitions sit on the *same* trained host, which gives matched pairs with opposite ground-truth labels.

The pathology set mixes surgical impairments, which have exact answers, with trained-through impairments, whose ground truth must be measured.

| ID | Pathology | Regime | Site | Expected label | Ground truth |
|---|---|---|---|---|---|
| P1 | Attenuate a group of g ∈ {2, 4, 8} channels by α ∈ {0.1, 0.3} | Post-hoc | z → z | Repairable, rank g | Exact: T⁻¹ |
| P2 | Ill-conditioned mixing T = U diag(s) Vᵀ, smallest s ∈ {0.1, 0.03} | Post-hoc | z → z | Repairable, full rank | Exact: T⁻¹ |
| P3 | Additive bias shift | Post-hoc | z → z | Repairable, bias only | Exact |
| P4 | Dead-channel ambiguity pair A/B | Post-hoc | z → z | Partially repairable; repairs differ in sign | Passive telemetry bitwise identical; probe answer exact |
| P5 | Trained-through 1×1 bottleneck 16→4→16 after z, read after the bottleneck | Trained-through | z̃ → z̃ | Locally unrepairable | Measured: site-sufficiency ceiling ≈ host |
| P6 | Same hosts as P5, read before the bottleneck | Trained-through | z → z̃ (bypass) | Repairable | Measured |
| P7 | Project out 32 of 128 channels after the last convolution's BN | Post-hoc | z → z | Wrong site | Measured: oracle at z ≈ 0, oracle at the true site > 0 |
| P8 | No impairment | n/a | z → z | Null | Measured: oracle gain sets the "repair" threshold |
| P9 | 40% symmetric label noise; and a 10%-data host | Trained-through | z → z | Non-capacity null | Measured with the oracle fitted under the same data regime |
| P10 | Starved host | Trained-through | z → z | Capacity deficit, to be measured | Measured; required before rung 1b |

Every case gets three measured numbers **[inference]**: the **site-sufficiency ceiling** (a fresh, generous downstream trained on frozen site features), the **oracle adapter** (best of seeds, a learning-rate grid and a 4× schedule) and the **intact reference**. A case is *repairable* when the oracle gain beats the intact-host oracle gain by at least 0.05 nats, *locally unrepairable* when the ceiling is within noise of the host, *wrong-site* when the ceiling exceeds the host but only an oracle at the true site gains, and *capacity-limited* when a wider adapter breaks a plateau the 272-parameter one cannot. Intact hosts go through the same procedure, because a learned stitch can improve a weaker top with no impairment at all **[snippet]** ([Bansal et al.](https://papers.nips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html)), and stitches succeed even between dissimilar layers **[snippet]** ([Hernandez et al.](https://arxiv.org/abs/2303.11277)).

Hosts come in **anchor lineages**: each anchor trains from its own initialisation to a stability point, then spawns branches that differ only in data order and augmentation. After an early instability window, children share one linearly connected basin; Frankle and colleagues place it near 3% of training for ResNet-20 on CIFAR-10 **[snippet]** ([Frankle et al.](https://proceedings.mlr.press/v119/frankle20a.html)), but that will not transfer to a 10-epoch narrow host, so the pilot measures it (interpolation barrier, site CKA and identity-stitch penalty at six spawn times) **[inference]**. Because INV-32 forbids branches of one base trajectory from crossing splits, **all confirmatory splits are by anchor lineage**. Across lineages, site channels are aligned to a reference host by Hungarian matching on activation correlations over a fixed unlabelled batch, which is legal at every tier.

## A 272-parameter module, a modest generator ladder, and a null that must be calibrated

The module is John's report's choice: **u = z + α(Wz + b), 272 parameters**, with no hidden units and so no permutation symmetry; the only symmetry problem left is cross-host channel correspondence, which alignment handles. The earlier project report's rank-32 GELU bottleneck made compactness visible at rung 1 at the price of that symmetry. This design gets compactness without it: generation-time compaction is an **SVD truncation** of the generated W at a calibrated threshold, stored as 32r + 16 parameters of factors when the retained rank r is below 8, with r = 0 and a negligible bias as the empty module. On P1 the true rank of T⁻¹ − I is exactly g, so the TCD can check whether retained rank tracks g, whether truncation beats a random cut at the same rank, and whether r = 0 coincides with the null classes. Truncation is a data-free operation on weights, so it adds no task gradient. Overcomplete modules and data-free scoring such as SynFlow **[S]** ([Tanaka et al.](https://arxiv.org/pdf/2006.05467)) wait for rung 2.

The generator question is secondary to the information question, and the ladder says so. Zeng and colleagues found that prominent weight generators produce replicas or interpolations of their training checkpoints and fail to beat noise-added or averaged weights on accuracy against novelty **[P]** ([Zeng et al.](https://github.com/boyazeng/weight_memorization)). Image diffusion memorises heavily below a few thousand samples **[S]** ([Gu et al.](https://arxiv.org/html/2310.02664v2)). A corpus of about 2,000 repairs sits in that danger zone, so the TCD trains seven methods on identical inputs:

| Rung of ladder | Method | Its job |
|---|---|---|
| L0 | No-op | Utility zero by definition |
| L1 | Nearest-neighbour retrieval in condition space; retrieval plus Gaussian noise swept over σ | Memorisation and noise baselines (Zeng et al.) |
| L2 | Kernel-weighted k-nearest-neighbour mean | Interpolation baseline |
| L3 | Deterministic hypernetwork (ridge, then a small MLP) | The strongest cheap explanation of any success |
| L4 | K-head mixture hypernetwork, winner-takes-all | Multimodality without iterative sampling |
| L5 | Conditional flow matching (rectified flow, MLP velocity field with FiLM conditioning, ≤100k parameters) | The stochastic candidate |
| L6 | Conditional diffusion | Only if L5 fails for an identifiable reason |

Flow matching is the default stochastic family because it is a reweighted diffusion objective with simpler training and few-step sampling, and recent weight-generation work moves that way **[S]** ([arXiv 2609.32833](https://arxiv.org/abs/2609.32833)); no controlled head-to-head exists, so this is an inference. All methods share one conditioning encoder: a per-example MLP with attention pooling, concatenated with precomputed sufficient statistics.

Before a generator is chosen, the corpus tests whether stochastic generation can earn its place. For 40 training hosts, five oracle repairs per case are compared in function space; if the *mean* of same-case oracles repairs as well as each one, the repairs are connected and a deterministic predictor suffices **[B]**, and the multimodality argument for diffusion has no purchase at this rung. All methods are compared at the same candidate budget K ∈ {1, 2, 4, 8, 16} under the same forward selector on a disjoint selection set, K = 1 is always reported, and self-distance accompanies best-of-K because a diffuse sampler can win that metric without being better **[S]** ([min-of-K bias](https://arxiv.org/pdf/2206.08641)).

The null is a selective-prediction problem. A presence head and the fraction of K samples truncating to rank zero are both calibrated with Learn-then-Test on the development lineages to P(harm | act) ≤ 0.1, harm meaning a loss increase beyond the no-op twin floor **[S]** ([Angelopoulos et al.](https://arxiv.org/pdf/2110.01052v4)); with 100–200 calibration cases that is about the achievable floor. Momir's abstention and Isperia's rejection after comparison with the measured no-op are distinct, and both are logged, with the harm veto applied before any utility comparison (INV-45).

## Two experiments, frozen repair and live capture, with static capacity at rung 1

Rung 1a puts every method on the same frozen host cases. Rung 1b inserts the module into a live host. The split matters because a frozen-host test measures immediate repair utility, as John's follow-up notes, while simic's open failure is about what happens once the host keeps training.

**Rung 1a (frozen repair) arms.** Every arm is evaluated on the same held-out case, with the downstream frozen and BN in eval mode.

| Arm | Description | What it rules out |
|---|---|---|
| A0, A0′ | No module; and a no-op twin on resampled evaluation order | Sets the zero and the noise floor |
| O | Oracle adapter (task gradients, generous budget) | Upper bound; also the corpus teacher |
| A2(k) | Zero- and random-initialised adapter after k ∈ {1, 2, 5, 20, 100} SGD steps on the conditioning examples | "Ordinary training already does this cheaply" |
| C4 | Gradient-SVD and line-searched gradient-step initialisation on the same examples | "The generator is an amortised gradient step" |
| G(tier) | Each generator at T0, T0+P, T1 and T1+P | The information contract |
| G-shuf | Same generator, conditions shuffled across cases (and probe responses shuffled separately) | "It emits a generic good adapter" |
| G-perm | Generated W with rows and columns permuted within the matrix, matched norm | "Only the scale and statistics matter" |
| R, R+σ, kNN | Retrieval, retrieval plus noise, interpolation | Memorisation |
| G-null | Cases where Momir abstained or Isperia rejected, analysed intention-to-treat | Selection on confidence |

**Rung 1b (live capture) arms.** These run on the starved host family P10, as matched branches forked per unit at epoch 3 of 10. The fork epoch is fixed for the TCD; simic's rung 4 is currently measuring how much timing matters for its graft, and that result should inform any later change.

| Arm | Description | What it rules out |
|---|---|---|
| B0, B0′ | No-op continuation and its bitwise-verified twin | Extra steps; the branch noise floor |
| BG | Generated module (T1, or the best tier that passed rung 1a), admitted by Isperia, blended over 100 steps, then trained jointly | The treatment |
| BC | Conventional graft: zero-initialised module, same insertion epoch and blend, trained jointly | "Generation adds nothing over inserting capacity at that time" |
| BS | Static capacity: the same 272-parameter module present from step zero, trained jointly (simic's static arm, here a separate from-scratch run on the same unit seed) | The bar simic's graft fails; doubles as "final architecture retrained from scratch" |
| BS+ | Compute-matched static: BS trained for extra steps equal to BG's formation and integration cost | "Gains were bought with compute" |

**Adding static capacity at rung 1 is settled: yes, in rung 1b, as the primary comparator after no-op.** Neither prior report included it before rung 3, but it is the comparator everything in simic has lost to: on `under_normalized`, static minus no growth was −0.119 [−0.151, −0.088] nats, the graft captured 0.60 (`norm`) and 0.31 (`conv_heavy`) of that, and static beat the graft in both (`docs/results/2026-10-09-graft-capture-v2.md`). For a fixed-shape module, static from step zero is also "the final architecture retrained from scratch", which answers the rethinking-pruning objection at rung 1 **[B]** ([Liu et al.](https://arxiv.org/abs/1810.05270)). A widened-layer static arm waits for rung 2, where structure is in play.

At the end of every 1b unit, the retention-ablation contrasts from the first section are measured on BG: intrinsic value against B0, acquired dependence under zero ablation, and the resample and optimal-constant checks. That is where reading three of "ablation" becomes evidence.

Rung 1b has a gate of its own. The starved host must show a measured deficit before the study freezes: in the pilot, static minus no-op of at least 0.05 nats, the floor screen v1 declared. If the starved family does not deliver one, 1b runs on the P6 bypass hosts, whose deficit exists by construction. That substitution is decided before launch, never after.

## Lineage splits, endpoints in nats, and 96 seeds for the live test

Data partitions follow simic's fixed permutation. Of the 45,000 fit examples, **40,000** train hosts and fit oracles, **2,000** form the conditioning pool (telemetry and probes, class-balanced draws of 1,024), and **3,000** are the selection set (candidate choice and Isperia's no-op comparison). The **5,000** development examples are the endpoint, as in every simic screen so far. The CIFAR-10 test batch is never read. Opening it for a confirmatory replication is John's decision.

For the surgical and null families, **56 anchors** spawn 8 branches each: 24 anchors train the generators, 8 serve generator development and null calibration, and 24 are the confirmatory test set. Two branches of each training anchor form a *seen-lineage* diagnostic that is never counted as independent evidence (INV-32). Scenario splits sit inside that structure: P1 group sizes 2 and 8 train and size 4 tests interpolation; the whole P2 mixing family is held out to test transfer to an unseen family; label-noise nulls train and 10%-data nulls test. The bottleneck family uses 32 anchors of 4 branches and each non-capacity null 16. About 1,900 training cases support a learning curve at 125, 250, 500, 1,000 and 1,900 repairs; transfer shows when the generator keeps improving after retrieval flattens **[B]**, and if no gap appears the TCD says so rather than scaling the generator.

The primary rung 1a metric is the **recovered fraction RF = (L_A0 − L_G) / (L_A0 − L_O)** in development cross-entropy, computed per repairable case at zero task-gradient steps. Cases are averaged within each anchor, so that the anchor is the unit of analysis, and RF is reported with t intervals and a cluster bootstrap over the 24 test anchors. Secondary metrics are absolute gain in nats; the learning curve Q(k) after k integration steps for G against A2; the retained rank against ground truth on P1; null precision, recall and realised harm at the calibrated threshold; the audit cosines; and the anti-memorisation dossier (Appendix C). No pilot has measured the anchor-level SD of RF differences. If it is 0.2, 24 test anchors detect an RF difference of about 0.12 at 80% power, which is adequate given that the go thresholds below sit at 0.2. If the pilot shows a larger SD, test anchors are added before the plan freezes.

Rung 1b uses simic's endpoint, **mean development cross-entropy over epochs 7–9**, paired within unit. Its seed count is recomputed from the paired SD. Taking ±0.018 nats as a 95% half-width at n = 48 gives SD ≈ 0.064 nats. The published result in fact labels that interval 97.5% (Bonferroni across two co-primaries) and reports an SD of 0.054, so 0.064 is conservative for contrasts against no-op. Contrasts involving the static arm were noisier in screen v1, with per-unit SDs of 0.087 and 0.095, which the result attributes, plausibly, to static capacity being born at step zero (`docs/results/2026-10-08-bounded-screen-v1.md`). This design therefore uses **0.064 for no-op contrasts and 0.09 for static contrasts**.

| Contrast | SD | Target | n (80% power) |
|---|---:|---|---:|
| BG − B0, superiority | 0.064 | δ = 0.025 | 54 |
| BG − B0, superiority | 0.064 | δ = 0.020 | 83 |
| BG − BS, superiority | 0.09 | δ = 0.030 | 73 |
| BG − BS, non-inferiority | 0.09 | margin 0.025 | 83 |

**Ninety-six units** (two 48-unit fleets, simic's precedent) give a minimum detectable effect of 0.019 nats against no-op and 0.026 nats against static, and 80% power for non-inferiority to static at a 0.023-nat margin. The contrasts run under fixed-sequence gatekeeping (BG < B0, then BG < BC, then BG non-inferior to BS, then BG against BS+), each at α = 0.05 only if the previous one passed. That holds family-wise error without a Bonferroni penalty. Simic's screens used Bonferroni across co-primaries instead, and the plan should note the change and its reason.

## Every example pass is booked: costs and a 25–45 GPU-hour budget

Amortised methods hide cost in corpus construction. The TCD therefore books five ledgers separately, in simic's units: example passes by component, optimiser parameter-steps and wall time, with the hardware stated and no FLOPs, following simic's policy (`bounded_screen.py:41`). The precedent is learned-optimiser work that reports meta-training compute up front: VeLO's roughly 4,000 TPU-months, and Celo's few GPU-hours as a headline number **[S]** ([Celo](https://arxiv.org/pdf/2501.12670)). NAS reporting norms add search cost and random baselines at matched budget **[B]** ([Lindauer & Hutter](https://arxiv.org/abs/1909.02453)).

| Ledger | Contents |
|---|---|
| C_host | Anchor and branch training; identical across arms by construction of the fork |
| C_corpus | Every oracle fit, capacity-certification sweep, ceiling model and telemetry capture used to make training data |
| C_meta | Generator, encoder and presence-head training, including hyperparameter search |
| C_deploy | Per case: telemetry forwards, the 64 probe forwards, generator inference × K, selection forwards, Isperia's no-op comparison |
| C_integrate | Task-gradient steps from insertion to the end of the run, above what B0 spends |

The break-even count N* = (C_corpus + C_meta) / (C_A2-equivalent − C_deploy) is reported even if it is absurd at TCD scale. Here C_A2-equivalent is the cost of the cheapest ordinary training that reaches the generator's gain. A reviewer will compute N* anyway.

The compute estimate (itemised in Appendix G) rests on simic's measured timing, 96 three-arm GPU units in 18–21 minutes on the two local cards (`research_notes/.../simic_harness_reuse.md`), and on airbench-class CIFAR-10 training taking seconds per run on one A100 **[fetched]** ([airbench](https://github.com/KellerJordan/cifar10-airbench)). Corpus building dominates: about 750 hosts and 12,000 oracle fits on cached site activations. The total is **about 25–45 GPU-hours plus a 3-hour pilot**, one to two days on the local pair, well under the earlier report's 60–80 A100-hours, which assumed 30-epoch hosts and no caching. No narrow-host timing exists yet, so the pilot's first job is a 10-run timing check. GPU time outside PDR-0050's rung-4 window is owner-reserved, so the TCD needs John's explicit authorisation.

## Pre-registered readings: go, pivot or stop

The readings are hierarchical. Each rung 1a gate is tested only if the gates before it passed, on the 24 test anchors, with one-sided 95% intervals unless stated. The tier attribution is pre-registered too. **The lowest tier at which G1–G3 pass names the claim.**

| Gate | Rule | If it fails |
|---|---|---|
| G0 instrument | ≥90% of repairable-labelled test cases have oracle gain ≥ 0.05 nats; intact-host oracle gain ≤ 0.01 nats on average; analytic T⁻¹ restores intact loss within tolerance; the P4 passive-equality and probe-antisymmetry unit tests pass | `instrument_failure`: fix the host or corpus; no Sultai reading |
| G1 useful without gradients | Mean RF_G ≥ 0.5, lower bound > 0.3 | Go to the pivot table |
| G2 conditioning matters | RF_G − RF_G-shuf ≥ 0.2, lower bound > 0 | Stop the generator line |
| G3 transfer, not memory | RF_G exceeds retrieval, retrieval plus noise and k-NN mean (lower bound > 0), and held-out-family RF ≥ 0.3 | Stop or pivot to alignment |
| G4 beats cheap training | At matched example passes (including C_deploy), G beats A2(k) with lower bound > 0, or reaches 90% of oracle in ≤ half A2's steps | Pivot to generated initialisation, or stop |
| G5 calibrated null | Harm rate among acted cases ≤ 0.1 at δ = 0.1, abstention recall on null classes ≥ 0.7, and acting on ≥ 0.7 of repairable cases | Pivot: the null decision is not ready, and admission stays with the measured no-op |
| R1–R3 live | BG < B0 (upper bound < 0); BG < BC; BG non-inferior to BS at 0.025 nats | See below |

These combine into a small number of outcomes.

| Outcome | Condition | What John does next |
|---|---|---|
| **GO-strict** | G0–G5 pass at T0 or T0+P; R1–R3 pass | Sultai's strong claim holds at toy scale: proceed to rung 2 with the label-free contract |
| **GO** | G0–G5 pass only at T1 or T1+P; R1–R3 pass; G not worse than C4 | Proceed to rung 2, naming the claim "no backward pass, labelled telemetry". If G beats C4, emphasise the beyond-first-order result |
| **PIVOT: probes are the mechanism** | T1 fails G1–G3 and T1+P passes | Make the investigation loop the core of rung 2 (a learned probe policy in the style of Deep Adaptive Design **[F]** ([Foster et al.](https://proceedings.mlr.press/v139/foster21a.html))) |
| **PIVOT: generated initialisation** | G1 fails but G with k steps beats A2(k) under G4 | Reframe Sultai as amortised initialisation, the role the earlier report assigned to D2NWG-style transfer; formation is then not gradient-free in effect |
| **PIVOT: alignment** | The seen-lineage diagnostic passes but cross-anchor G3 fails | The bottleneck is channel basis, not repair knowledge; rung 2 starts with alignment or permutation-equivariant generators |
| **PIVOT: drop stochastic generation** | L3 or L4 matches L5 at matched K | Keep the deterministic generator; the multimodality analysis says why |
| **STOP the generator line** | G0 passes but G2 or G3 fails at every tier; or A2 at k ≤ 2 matches G at lower total cost | Amortising a 272-parameter repair solves a problem SGD solves cheaply. Do not scale the generator; record the null result |
| **STOP the insertion route to half-parameters** | Rung 1a passes but BG ≤ BC and BG is clearly worse than BS (R3 fails with the interval excluding the margin) | Even a working generator does not change simic's core finding. The half-parameters target would need capacity present from the start (static or ExpandNets-style), not grown in |

The rung 1b reading also reports, descriptively, the capture fraction (BG − B0)/(BS − B0), set against graft-capture v2's 0.60 and 0.31. It reports the retention-ablation decomposition as well.

## Build plan in simic's conventions

The work lands as a new study line in the simic repository. It imports from the bounded modules and edits none of them. Each step ends in reviewed, tested code, and nothing reaches a GPU before John signs the plan.

| Step | Deliverable | Notes |
|---|---|---|
| 0 | Owner decisions: confirm the ablation reading; in-repo study line or fork; GPU authorisation; an ADR scoping Sultai as a bounded study with the role mapping above | Vision and GPU are owner-reserved (`current-state.md`); ADR-0018 is the precedent |
| 1 | Filigree epic and per-step issues; `docs/sultai-tcd.md` design note (the analogue of `docs/bounded-comparison.md`) | Cite tracker IDs so `test_doc_references.py` resolves them |
| 2 | `experiments/sultai_host.py`: host, starved variant, (read, write) sites, impairment operators, frozen-BN evaluation mode | `tests/unit/test_sultai_host.py` with the known-answer tests: T⁻¹ restores; P4 passive equality; probe antisymmetry; A's oracle worsens B |
| 3 | `experiments/sultai_corpus.py`: anchors and branches with `derive()` streams, spawn calibration, activation caching, oracle fits with certification, ceiling models, labels, manifests and checksums | Batch oracle fits per host with `torch.func.vmap`; ledgers per Appendix F |
| 4 | `experiments/sultai_telemetry.py`: tiers T0 to T1+P on snapshot copies, probe menu, gradient-recoverability audit, gradient comparator | Missing values raise rather than render as zero; no telemetry inside the training forward or autocast (the esper-lite lessons) |
| 5 | `experiments/sultai_generators.py`: L1–L5, the shared encoder, presence head, Learn-then-Test calibration, SVD truncation, random-cut control, Elesh-style canonicalisation | CPU-trainable; `tests/unit/test_sultai_generators.py` |
| 6 | `experiments/sultai_screen.py`: launch and analysis for 1a and 1b, a Sultai verifier on the `verify_run` model, anchor-cluster analysis, gatekeeping reading rules | Imports `make_snapshot`, `interval`, `derive`, `CommonFuture` |
| 7 | `docs/prereg/sultai-tcd-pilot.json` (exploratory), owner sketch, CPU dry run, pilot, `docs/results/<date>-sultai-tcd-pilot.md` | The pilot sets the spawn point, test-anchor count, starved deficit and timings |
| 8 | `docs/prereg/sultai-tcd-1a.json` and `sultai-tcd-1b.json`, frozen after an owner sketch (PDR-0053) and a dry run | Every threshold in the readings table is fixed here |
| 9 | Runs; `docs/results/<date>-sultai-tcd-1a.md` and `-1b.md`; John writes the PDR applying the reading | Outer data stays closed |

Steps 2 to 6 can be built and dry-run on CPU while rung 4 occupies the GPUs. If the role mapping is adopted, it also pulls the HLD's Phase A forward on one narrow front. A typed `GrowthIntent` stub, a Nissa telemetry envelope with an interventional profile, and an Isperia no-op comparison are small, real Leyline contracts.

## Later rungs climb toward half the parameters

Rung 1 cannot show parameter efficiency, and it is not meant to. The later rungs add one unknown at a time.

**Rung 2: structure, compaction and scaffolding at a known site.** The module becomes overcomplete: a rank-R GELU bottleneck with gains, or a small typed vocabulary (none, 1×1, depthwise 3×3, squeeze-excitation). Compaction becomes generation-time truncation plus data-free scoring, always against a random cut at the same ratio. The investigation loop gains a learned probe policy if rung 1 pointed that way. A MetaLearnNCA-style learned developmental rule enters as an alternative to sampling. A scaffold mode tests whether temporary structure helps the host's path and survives removal. That test runs against compute-matched static, the final architecture retrained from scratch, a frozen random branch and distillation. ExpandNets is the key precedent: temporary linear over-parameterisation can improve a compact model of fixed final size **[B]** ([Guo et al.](https://arxiv.org/abs/1811.10495)). Here the widened-layer static arm and the retention-ablation operators become primary.

**Rung 3: Aurelia chooses where, and growth repeats.** Starting from the starved host with 8–16 legal sites, Aurelia first becomes a site-value predictor trained on attempted oracle repairs, a contextual bandit before any reinforcement learning. Telemetry for each new site is read on Nissa's ablated path. Growth repeats until the marginal gain per parameter falls below a threshold. Comparators are GradMax- and Firefly-style gradient growth and RigL at matched sparsity; Firefly's CIFAR-10 comparison matches model size rather than reducing it **[S]** ([Wu et al.](https://arxiv.org/pdf/2102.08574)).

**Rung 4: the half-parameters claim.** The reference is a **from-scratch width sweep at matched or larger compute**, plotted as a parameters-versus-loss frontier, not the unstarved host. The claim is that the grown model sits above that frontier at half the parameters of the from-scratch model it matches, with all five cost ledgers disclosed. No grown-network result found in this review shows half the parameters at matched CIFAR-10 accuracy against a strong, equally tuned compact baseline. Pruning studies routinely compare against weak or inconsistent baselines **[S]** ([Blalock et al.](https://arxiv.org/pdf/2003.03033)), and compact CIFAR ResNets of about 0.27–0.85M parameters form a hard frontier **[B]** ([He et al.](https://arxiv.org/abs/1512.03385)). The starved-host design makes the target easy to state and hard to win.

## Conclusion

The TCD's design turns on one point that neither prior report quite stated. Sultai's distinctive bet is not diffusion, and not even generation. It is the claim that **local evidence carries information about a repair that ordinary gradient descent on the same evidence does not already extract cheaply**. A 272-parameter module at a frozen 16-channel site is small enough for few-step SGD to fix many defects quickly. So the demonstrator is built to fail honestly in that direction, and to name what kind of information did the work when it succeeds. The tier ladder and the gradient audit convert a terminology dispute into a measured quantity. The dead-channel pair gives an exact case where only forward interventions can work. Static capacity at rung 1b ties the outcome to the one result simic already holds: inserted capacity has so far lost to capacity that was simply present from the start.

The most useful outcome may be a pivot rather than a go. If probes turn out to be what matters, the research object becomes an investigation loop, and the generator is incidental. If only generated initialisation survives, Sultai becomes a faster way to train inserted modules, which is worth less but is still measurable. And if the live test shows that even good generated repairs lose to static capacity, the half-parameters programme should move away from growth by insertion and toward structure present from the start and later compacted. That would be a cheap and decisive thing to learn in one to two days of GPU time.

---

## Appendix A. Host, sites and pathology construction

**Host (intact).** stem: conv3×3 3→32, BN, ReLU; conv3×3 32→32, BN, ReLU; max-pool 2 (16×16). Producer: conv3×3 32→16, BN, ReLU, output z (16×16×16). Site S1 reads and writes z: u = z + α(Wz + b). Downstream: conv3×3 16→64, BN, ReLU; conv3×3 64→64, BN, ReLU; pool (8×8); conv3×3 64→128, BN, ReLU; conv3×3 128→128, BN, ReLU; pool (4×4); global average pool; linear 128→10. **Starved:** downstream widths 32, 32, 64, 64. **Bottleneck family (P5/P6):** after z insert 1×1 16→4, BN, ReLU, 1×1 4→16, BN, ReLU, output z̃, trained through. Site S2a reads and writes z̃. Site S2b reads z and writes z̃ (bypass). Both are 272-parameter modules on the same host.

**Post-hoc impairments** are applied to z after the producer's ReLU and before the module, with downstream BN frozen. P1: z ← D z with D diagonal, α on a random channel group of size g. P2: z ← T z, T = U diag(s) Vᵀ with Haar-random U and V and log-spaced s down to s_min. P3: z ← z + c with c drawn per channel at a fixed norm. P7: after the final convolution's BN, project out a random 32-dimensional subspace of the 128 channels; verify "wrong site" with an oracle 128→128 1×1 adapter at that point (verification only).

**P4 dead-channel ambiguity pair.** Host A: zero channel k of z post hoc (on top of an intact host). Host B: as A, with the input weights of the first downstream convolution for channel k negated. Because z_k ≡ 0, A and B compute identical functions and all passive telemetry is bitwise equal (unit test 1). The probe z_k ← ±ε gives change-in-logits of opposite sign in A and B (unit test 2). The oracle repair rows writing into channel k satisfy W_B[k,:] = −W_A[k,:] up to fitting noise. Applying A's oracle to B raises B's loss (unit test 3). The repair is partial, because channel k is reconstructed only from the other 15 channels.

**Labelling protocol.** Ceiling: a fresh downstream of twice the width, trained 20 epochs on frozen site features. Oracle: three seeds × learning rates {0.3, 0.1, 0.03, 0.01} × schedules {1×, 4×}, keeping the best on the selection set; certify a plateau when all of them agree within the twin noise floor and the training loss also plateaus. Labels follow the rules in the main text, with thresholds frozen in the pre-registration.

## Appendix B. Telemetry fields by tier

| Field | Size | T0 | T0+P | T1 | T1+P |
|---|---:|:-:|:-:|:-:|:-:|
| Per-channel mean and variance of z | 32 | ✓ | ✓ | ✓ | ✓ |
| Covariance of z (upper triangle) and eigen-spectrum | 136 + 16 | ✓ | ✓ | ✓ | ✓ |
| Dead and saturated fractions per channel | 32 | ✓ | ✓ | ✓ | ✓ |
| Prediction entropy and margin histograms; predicted-class marginal | 20 + 10 | ✓ | ✓ | ✓ | ✓ |
| Unlabelled tuples (pooled z, p) | 1,024 × 26 | ✓ | ✓ | ✓ | ✓ |
| Probe responses: mean change in logits for ±ε shift and ±ε rescale of each channel | 64 × 10 | | ✓ | | ✓ |
| Labelled tuples (pooled z, p, y, loss) | 1,024 × 28 | | | ✓ | ✓ |
| Class-conditional means of z, confusion matrix, per-class loss | 160 + 100 + 10 | | | ✓ | ✓ |
| Site interface descriptor (shape, stage, normalisation) | small | ✓ | ✓ | ✓ | ✓ |
| Downstream host weights | n/a | never | never | never | never |

Everything is computed on a frozen snapshot copy, on the conditioning pool only, in full precision, outside the training path. A missing field is an error, never a zero. The audit decoders are fitted on training lineages and scored on test lineages; their target is the true (∇_W L, ∇_b L) at zero initialisation on the same 1,024 examples.

## Appendix C. Anti-memorisation dossier (reported for every generator)

These requirements follow Zeng and colleagues' tests, adapted to conditional, held-out-lineage generation **[P]** ([Zeng et al.](https://github.com/boyazeng/weight_memorization)). First, nearest-neighbour distance histograms in the aligned canonical representation, with three curves: train to train, generated (test case) to train, and the test case's own fresh oracle to train. Second, function-space similarity of each generated module's site output, on a fixed probe batch, to (a) its nearest corpus repair and (b) the case's own oracle. Generalisation means being closer to (b) than to (a). Third, a quality-against-novelty frontier including retrieval plus noise across σ, k-NN interpolation, the shuffled-condition generator and no-op. Fourth, the least-squares residual of each generated module onto the span of its k nearest corpus repairs. Fifth, best-of-K curves at matched K with self-distance. The shuffled-condition control is scored on host-specific gain, not plausibility, because random-label conditioning is known to induce memorisation **[S]** ([Gu et al.](https://arxiv.org/html/2310.02664v2)).

## Appendix D. Power arithmetic

Normal approximation with a +2 small-sample correction: n = ((z₁₋α/s + z_power) · SD / δ)² + 2. SD 0.064 comes from ±0.018 at n = 48 read as a 95% half-width (0.018 · √48 / 1.96). Read as the 97.5% interval the result document actually reports, it gives about 0.056, and the document's own SD for that contrast is 0.054. Static-arm contrasts in screen v1 had SDs of 0.087 and 0.095.

| SD | δ or margin | Test | n at 80% | n at 90% |
|---:|---:|---|---:|---:|
| 0.054 | 0.025 | superiority, two-sided 0.05 | 39 | 52 |
| 0.064 | 0.020 | superiority | 83 | 110 |
| 0.064 | 0.025 | superiority | 54 | 71 |
| 0.064 | 0.030 | superiority | 38 | 50 |
| 0.090 | 0.025 | superiority | 104 | 139 |
| 0.090 | 0.030 | superiority | 73 | 97 |
| 0.064 | 0.025 | non-inferiority, one-sided 0.05 | 43 | |
| 0.090 | 0.025 | non-inferiority | 83 | |

At n = 96: the minimum detectable effect at 80% power is 0.019 nats (SD 0.064) and 0.026 nats (SD 0.09); the achievable non-inferiority margin is 0.023 nats (SD 0.09). For rung 1a, with an assumed anchor-level SD of 0.2 in RF differences, 24 test anchors detect 0.12 and 32 anchors detect 0.10. The pilot replaces the 0.2 assumption. Equivalence or non-inferiority claims use two one-sided tests **[B]** ([Lakens](https://doi.org/10.1177/1948550617697177)). Reports include probability of improvement and the full distribution of paired differences **[B]** ([Agarwal et al.](https://arxiv.org/abs/2108.13264)).

## Appendix E. Integration protocol for rung 1b

At the fork epoch the base host is snapshotted, and every branch reuses it bitwise. Nissa computes the telemetry on a copy; Momir generates; Elesh aligns; Urabrask slices; Isperia compares the candidate with the no-op on the selection set and admits or rejects. An admitted module enters with α ramped linearly from 0 to 1 over 100 steps. There is no straight-through incubation and no `tau_init` rescaling. Host and module then train jointly under simic's optimiser for the remaining epochs. The module gets the host's learning rate, a choice noted as a free parameter for rung 2, since newly grown parts may need rebalancing. A rejected candidate leaves the branch identical to B0, and the unit stays in the BG arm for intention-to-treat analysis. BC follows the same schedule with W = 0, b = 0. BS is a separate run from step zero with the module present at α = 1, sharing the unit's data stream. After the final epoch, retention ablation evaluates BG with the module zeroed, resampled within the batch, and replaced by its optimal constant.

## Appendix F. Cost ledger fields

Per unit and per ledger (C_host, C_corpus, C_meta, C_deploy, C_integrate), record: forward example passes, backward example passes, optimiser parameter-steps, wall seconds, device, and peak resident parameters. Also record final stored parameters (host plus retained factors), the retained rank, the K used, the probe count, and whether Momir abstained or Isperia rejected. Generator parameters are counted in the deployed total only if the generator must remain at inference; in this design it does not. N* is computed per method from these ledgers.

## Appendix G. Compute estimate

| Item | Volume | Estimate on 2× RTX 4060 Ti |
|---|---|---:|
| Pilot (timing, spawn point, noise floors, RF variance, starved deficit) | about 150 short runs | 3 h |
| Hosts, all families | about 750 hosts, about 10 epochs each on 40,000 examples | 3–6 h |
| Oracle fits on cached site activations, with certification sweeps | about 12,000 fits | 6–14 h |
| Ceiling models | about 64 | 0.5–1 h |
| Telemetry, probes, audit, gradient comparator | about 4,000 cases | under 1 h |
| Generators: 7 methods × 4 tiers × 5 corpus sizes × 3 seeds | about 420 small trainings | 7–15 h, CPU-capable |
| Rung 1a evaluation, including A2 learning curves | about 1,900 test cases × 15 arms | 2–4 h |
| Rung 1b: 96 units × (base, branches, static, compute-matched static) | about 4,000 small-host epochs | 2–4 h |
| **Total** | | **about 25–45 GPU-hours** |

Assumptions: narrow hosts are overhead-bound rather than FLOP-bound, so batching several hosts or oracle fits with `torch.func.vmap` is the main lever. Oracle fits train the module on cached site activations (about 330 MB per host in fp16), so only the downstream runs forward and backward. The earlier report's figure assumed a different hardware class and longer schedules. All figures are estimates to be replaced by the pilot's timings.
